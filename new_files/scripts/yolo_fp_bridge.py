#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_fp_bridge.py  —  Service 触发模式
=======================================
提供 ROS2 service：/foundationpose/detect
  Request : string class_name  （"k2c" 或 "j2"）
  Response: bool success, string message, geometry_msgs/PoseStamped pose

FSM 流程：
  1. FSM 调用 /foundationpose/detect (class_name="k2c")
  2. bridge 调用 /yolo_detect 获取当前帧的 RGB/Depth/Mask
  3. 若类别改变则重启 FoundationPose（加载对应 .obj）
  4. 将图像送入 FP selector，触发位姿估计
  5. 等待 /tracking/output 或 /pose_estimation/output 有结果
  6. 返回 PoseStamped 给 FSM

话题输出（供订阅）：
  /foundationpose/k2c/pose  — geometry_msgs/PoseStamped（k2c 每次估计结果）
  /foundationpose/j2/pose   — geometry_msgs/PoseStamped（j2，接口预留）
  /foundationpose/pose      — 最新一次结果（任意类别）

启动方式：
  source /opt/ros/humble/setup.bash
  source /path/to/isaac_ros_foundationpose/install/setup.bash
  source /path/to/niusuo_perception/install/setup.bash
  ros2 launch isaac_ros_foundationpose yolo_fp_bridge.launch.py
"""

import copy
import os
import sys
import time
import math
import signal
import subprocess
import threading

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray

from yolo_interfaces.srv import YoloDetect, FpDetect


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# 所有支持的类别 → 输出 topic 映射
SUPPORTED_CLASSES = {
    'k2c': '/foundationpose/k2c/pose',
    'j2':  '/foundationpose/j2/pose',
}


class YoloFpBridgeNode(Node):

    def __init__(self):
        super().__init__('yolo_fp_bridge')

        # ── 参数 ──────────────────────────────────────────────
        self.declare_parameter('mesh_dir', '/media/rykj/nvme/jetson')
        self.declare_parameter('class_to_mesh', 'k2c:K2,j2:J2')
        self.declare_parameter('yolo_service', 'yolo_detect')
        self.declare_parameter('camera_info_topic', '/right_camera/color/camera_info')
        self.declare_parameter('refine_engine', '/tmp/refine_trt_engine.plan')
        self.declare_parameter('score_engine', '/tmp/score_trt_engine.plan')
        self.declare_parameter('refine_model', '/tmp/refine_model.onnx')
        self.declare_parameter('score_model', '/tmp/score_model.onnx')
        self.declare_parameter('default_texture', '/tmp/k2c_texture.png')
        self.declare_parameter('fp_wait_timeout', 90.0)
        # 等待 FP 输出位姿的超时（秒）
        self.declare_parameter('pose_timeout', 10.0)

        self.mesh_dir = self.get_parameter('mesh_dir').value
        raw_map = self.get_parameter('class_to_mesh').value
        self.class_to_mesh: dict[str, str] = {}
        for pair in (raw_map or '').split(','):
            pair = pair.strip()
            if ':' in pair:
                k, v = pair.split(':', 1)
                self.class_to_mesh[k.strip()] = v.strip()

        yolo_service        = self.get_parameter('yolo_service').value
        camera_info_topic   = self.get_parameter('camera_info_topic').value
        self.refine_engine  = self.get_parameter('refine_engine').value
        self.score_engine   = self.get_parameter('score_engine').value
        self.refine_model   = self.get_parameter('refine_model').value
        self.score_model    = self.get_parameter('score_model').value
        self.default_texture = self.get_parameter('default_texture').value
        self.fp_wait_timeout = self.get_parameter('fp_wait_timeout').value
        self.pose_timeout   = self.get_parameter('pose_timeout').value

        self.bridge = CvBridge()

        # ── 状态 ──────────────────────────────────────────────
        self.camera_info: CameraInfo | None = None
        self.current_class: str | None = None
        self.fp_process: subprocess.Popen | None = None
        self._fp_lock = threading.Lock()
        self._detect_lock = threading.Lock()   # 同一时间只处理一个 detect 请求

        # 等待 FP 位姿结果用
        self._pose_event = threading.Event()
        self._latest_pose: PoseStamped | None = None
        self._latest_class: str | None = None  # 当前正在估计的类别

        # ── 发布到 selector 输入话题 ───────────────────────────
        self.img_pub   = self.create_publisher(Image,      '/image',        10)
        self.depth_pub = self.create_publisher(Image,      '/depth_image',  10)
        self.seg_pub   = self.create_publisher(Image,      '/segmentation', 10)
        self.cam_pub   = self.create_publisher(CameraInfo, '/camera_info',  10)

        # ── 各类别位姿输出话题 ────────────────────────────────
        self._pose_pubs: dict[str, object] = {}
        for cls, topic in SUPPORTED_CLASSES.items():
            self._pose_pubs[cls] = self.create_publisher(PoseStamped, topic, 10)
        self._pose_pub_latest = self.create_publisher(
            PoseStamped, '/foundationpose/pose', 10)

        # ── 订阅相机内参 ───────────────────────────────────────
        self.create_subscription(CameraInfo, camera_info_topic,
                                 self._cam_info_cb, SENSOR_QOS)

        # ── YOLO 服务客户端 ────────────────────────────────────
        self.yolo_cli = self.create_client(YoloDetect, yolo_service)

        # ── FP selector 触发服务客户端 ─────────────────────────
        self.trigger_cli = self.create_client(
            Trigger, '/foundationpose/trigger_pose_estimation')

        # ── 订阅 FP 位姿输出 ───────────────────────────────────
        self.create_subscription(Detection3DArray, '/tracking/output',
                                 self._tracking_cb, 10)
        self.create_subscription(Detection3DArray, '/pose_estimation/output',
                                 self._pose_est_cb, 10)

        # ── 对外暴露的服务（FSM 调用） ─────────────────────────
        self.detect_srv = self.create_service(
            FpDetect, '/foundationpose/detect', self._handle_detect)

        self.get_logger().info(
            f'YoloFpBridge 已启动 [Service 模式]\n'
            f'  服务:    /foundationpose/detect\n'
            f'  话题:    /foundationpose/k2c/pose  /foundationpose/j2/pose\n'
            f'  mesh_dir={self.mesh_dir}  mapping={self.class_to_mesh}'
        )

    # ------------------------------------------------------------------
    # 相机内参回调
    # ------------------------------------------------------------------
    def _cam_info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(
                f'获取相机内参  frame={msg.header.frame_id}  {msg.width}×{msg.height}'
            )

    # ------------------------------------------------------------------
    # FP 位姿输出回调 — 唤醒等待中的 service handler
    # ------------------------------------------------------------------
    def _on_fp_pose(self, msg: Detection3DArray, source: str):
        if not msg.detections:
            return
        ps = self._build_pose_stamped(msg)
        if ps is None:
            return

        # 打印日志
        p = ps.pose.position
        r = ps.pose.orientation
        self.get_logger().info(
            f'[{source}] xyz=({p.x:.4f},{p.y:.4f},{p.z:.4f})  '
            f'q=({r.x:.4f},{r.y:.4f},{r.z:.4f},{r.w:.4f})'
        )

        # 发布到对应类别 topic 和公共 topic
        cls = self._latest_class
        if cls and cls in self._pose_pubs:
            self._pose_pubs[cls].publish(ps)
        self._pose_pub_latest.publish(ps)

        # 唤醒正在阻塞的 service handler
        self._latest_pose = ps
        self._pose_event.set()

    def _tracking_cb(self, msg: Detection3DArray):
        self._on_fp_pose(msg, 'Tracking')

    def _pose_est_cb(self, msg: Detection3DArray):
        # 仅在 tracking 没有先给出结果时才使用
        if not self._pose_event.is_set():
            self._on_fp_pose(msg, 'PoseEstimation')

    # ------------------------------------------------------------------
    # 核心：/foundationpose/detect 服务处理
    # ------------------------------------------------------------------
    def _handle_detect(self, request: FpDetect.Request,
                       response: FpDetect.Response) -> FpDetect.Response:
        cls = request.class_name.strip().lower()

        if cls not in SUPPORTED_CLASSES:
            response.success = False
            response.message = (
                f'不支持的类别 "{cls}"，可用: {list(SUPPORTED_CLASSES.keys())}'
            )
            self.get_logger().warn(response.message)
            return response

        # 同一时间只处理一个请求（串行化）
        if not self._detect_lock.acquire(blocking=False):
            response.success = False
            response.message = '上一次检测尚未完成，请稍后再试'
            self.get_logger().warn(response.message)
            return response

        try:
            return self._do_detect(cls, response)
        finally:
            self._detect_lock.release()

    def _do_detect(self, cls: str,
                   response: FpDetect.Response) -> FpDetect.Response:
        # ── 1. 检查相机内参 ───────────────────────────────────
        if self.camera_info is None:
            response.success = False
            response.message = '尚未收到相机内参，请检查相机是否运行'
            self.get_logger().error(response.message)
            return response

        # ── 2. 调用 YOLO 服务 ─────────────────────────────────
        if not self.yolo_cli.wait_for_service(timeout_sec=3.0):
            response.success = False
            response.message = 'YOLO 服务不可用（等待 3s 超时）'
            self.get_logger().error(response.message)
            return response

        req = YoloDetect.Request()
        req.class_name = cls
        future = self.yolo_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        yolo_resp = future.result()

        if yolo_resp is None or not yolo_resp.success or not yolo_resp.detections:
            msg = f'YOLO 未检测到 [{cls}]'
            if yolo_resp and yolo_resp.message:
                msg += f'：{yolo_resp.message}'
            response.success = False
            response.message = msg
            self.get_logger().info(msg)
            return response

        # 取置信度最高的检测结果
        best = max(yolo_resp.detections, key=lambda d: d.confidence)
        self.get_logger().info(
            f'YOLO 检测到 [{cls}] conf={best.confidence:.3f}  '
            f'3D=({best.center_point[0]:.3f},'
            f'{best.center_point[1]:.3f},'
            f'{best.center_point[2]:.3f})m'
        )

        # ── 3. 查找对应 .obj 文件 ─────────────────────────────
        obj_path = self._find_obj(cls)
        if obj_path is None:
            mesh_name = self.class_to_mesh.get(cls, cls)
            response.success = False
            response.message = f'未在 {self.mesh_dir} 找到 {mesh_name}.obj'
            self.get_logger().error(response.message)
            return response

        # ── 4. 如果类别切换或 FP 崩溃，重启 FP ──────────────
        fp_crashed = (self.fp_process is not None and
                      self.fp_process.poll() is not None)
        if cls != self.current_class or fp_crashed:
            reason = '类别切换' if cls != self.current_class else 'FP 进程异常'
            self.get_logger().info(f'{reason}，重启 FoundationPose → {obj_path}')
            self._launch_fp(obj_path)
            self._latest_class = cls
            self.current_class = cls
            self.get_logger().info(
                f'等待 FoundationPose 就绪（最多 {self.fp_wait_timeout:.0f}s）...'
            )
            if not self._wait_fp_ready():
                response.success = False
                response.message = 'FoundationPose 启动超时'
                self.get_logger().error(response.message)
                return response
            self.get_logger().info('FoundationPose 已就绪')
        else:
            self._latest_class = cls

        # ── 5. 图像转换 ───────────────────────────────────────
        try:
            rgb_out   = self._to_rgb8(yolo_resp.rgb, self.bridge)
            depth_out = self._to_32fc1(yolo_resp.depth, self.bridge)
            mask_msg  = best.mask
        except Exception as e:
            response.success = False
            response.message = f'图像转换失败: {e}'
            self.get_logger().error(response.message)
            return response

        # 统一时间戳（ExactSync 要求）
        stamp = yolo_resp.rgb.header.stamp
        for m in (rgb_out, depth_out, mask_msg):
            m.header.stamp = stamp
        cam_info = copy.deepcopy(self.camera_info)
        cam_info.header.stamp = stamp

        # ── 6. 清空上次位姿结果，发布图像 ─────────────────────
        self._pose_event.clear()
        self._latest_pose = None

        self.img_pub.publish(rgb_out)
        self.depth_pub.publish(depth_out)
        self.seg_pub.publish(mask_msg)
        self.cam_pub.publish(cam_info)
        self.get_logger().info(
            f'已发布图像: RGB {rgb_out.width}×{rgb_out.height}  '
            f'Depth {depth_out.width}×{depth_out.height}  '
            f'Mask {mask_msg.width}×{mask_msg.height}'
        )

        # ── 7. 触发 FP selector ───────────────────────────────
        if self.trigger_cli.service_is_ready():
            trig_fut = self.trigger_cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, trig_fut, timeout_sec=3.0)
            trig_res = trig_fut.result()
            if trig_res and trig_res.success:
                self.get_logger().info('已触发位姿估计')
            else:
                msg_str = trig_res.message if trig_res else '超时'
                self.get_logger().warn(f'触发返回: {msg_str}（继续等待）')
        else:
            self.get_logger().warn('trigger 服务未就绪，跳过触发，直接等待输出')

        # ── 8. 等待 FP 输出位姿 ───────────────────────────────
        if not self._pose_event.wait(timeout=self.pose_timeout):
            response.success = False
            response.message = (
                f'等待 FoundationPose 位姿超时（{self.pose_timeout:.0f}s）'
            )
            self.get_logger().warn(response.message)
            return response

        # ── 9. 返回结果 ───────────────────────────────────────
        response.success = True
        response.message = f'OK  class={cls}'
        response.pose = self._latest_pose
        p = response.pose.pose.position
        r = response.pose.pose.orientation
        self.get_logger().info(
            f'[detect/{cls}] 成功  '
            f'xyz=({p.x:.4f},{p.y:.4f},{p.z:.4f})  '
            f'q=({r.x:.4f},{r.y:.4f},{r.z:.4f},{r.w:.4f})'
        )
        return response

    # ------------------------------------------------------------------
    # 查找 obj 文件
    # ------------------------------------------------------------------
    def _find_obj(self, class_name: str) -> str | None:
        mesh_name = self.class_to_mesh.get(class_name, class_name)
        for fname in os.listdir(self.mesh_dir):
            if fname.lower() == f'{mesh_name.lower()}.obj':
                return os.path.join(self.mesh_dir, fname)
        return None

    # ------------------------------------------------------------------
    # FoundationPose 子进程管理
    # ------------------------------------------------------------------
    def _launch_fp(self, mesh_path: str):
        with self._fp_lock:
            self._kill_fp_locked()
            base = os.path.splitext(mesh_path)[0]
            texture = (base + '.png') if os.path.exists(base + '.png') \
                      else self.default_texture
            cmd = [
                'ros2', 'launch', 'isaac_ros_foundationpose',
                'isaac_ros_foundationpose_tracking.launch.py',
                f'mesh_file_path:={mesh_path}',
                f'texture_path:={texture}',
                f'refine_engine_file_path:={self.refine_engine}',
                f'score_engine_file_path:={self.score_engine}',
                f'refine_model_file_path:={self.refine_model}',
                f'score_model_file_path:={self.score_model}',
                'launch_rviz:=False',
            ]
            self.get_logger().info(f'启动 FP: {" ".join(cmd)}')
            self.fp_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )

            def _log():
                for line in self.fp_process.stdout:
                    try:
                        self.get_logger().info(
                            '[FP] ' + line.decode('utf-8', errors='replace').rstrip()
                        )
                    except Exception:
                        pass
            threading.Thread(target=_log, daemon=True).start()

    def _kill_fp_locked(self):
        if self.fp_process and self.fp_process.poll() is None:
            self.get_logger().info('停止 FoundationPose...')
            try:
                os.killpg(os.getpgid(self.fp_process.pid), signal.SIGINT)
                self.fp_process.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.fp_process.pid), signal.SIGKILL)
                except Exception:
                    pass
        self.fp_process = None

    def _wait_fp_ready(self) -> bool:
        deadline = time.monotonic() + self.fp_wait_timeout
        while time.monotonic() < deadline:
            if self.trigger_cli.service_is_ready():
                return True
            time.sleep(1.0)
        return False

    # ------------------------------------------------------------------
    # 图像格式转换
    # ------------------------------------------------------------------
    @staticmethod
    def _to_rgb8(msg: Image, bridge: CvBridge) -> Image:
        if msg.encoding.lower() == 'rgb8':
            return msg
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        out.header = msg.header
        return out

    @staticmethod
    def _to_32fc1(msg: Image, bridge: CvBridge) -> Image:
        if msg.encoding.lower() == '32fc1':
            return msg
        depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001
        out = bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding='32FC1')
        out.header = msg.header
        return out

    # ------------------------------------------------------------------
    # 构建 PoseStamped
    # ------------------------------------------------------------------
    def _build_pose_stamped(self, msg: Detection3DArray) -> PoseStamped | None:
        if not msg.detections:
            return None
        best = max(
            msg.detections,
            key=lambda d: (max(r.hypothesis.score for r in d.results)
                           if d.results else 1.0)
        )
        ps = PoseStamped()
        ps.header = msg.header
        if self.camera_info and self.camera_info.header.frame_id:
            ps.header.frame_id = self.camera_info.header.frame_id
        ps.pose = best.bbox.center
        return ps

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def destroy_node(self):
        with self._fp_lock:
            self._kill_fp_locked()
        super().destroy_node()


def main():
    rclpy.init(args=sys.argv)
    node = YoloFpBridgeNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
