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
import json
import os
import sys
import time
import math
import signal
import subprocess
import threading
import datetime

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
        # 计时日志保存目录（空则不保存）
        self.declare_parameter('timing_log_dir', '/media/rykj/nvme/jetson/fp_timing_logs')

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
        self.timing_log_dir = self.get_parameter('timing_log_dir').value
        if self.timing_log_dir:
            os.makedirs(self.timing_log_dir, exist_ok=True)

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
        response.timing_json = ''
        timing = {'class': cls,
                  'timestamp': datetime.datetime.now().isoformat(),
                  'fp_model_loaded_before': False}
        t0 = time.monotonic()

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

        t_yolo_start = time.monotonic()
        req = YoloDetect.Request()
        req.class_name = cls
        _yolo_event = threading.Event()
        _yolo_result = [None]
        def _yolo_cb(f):
            _yolo_result[0] = f.result()
            _yolo_event.set()
        future = self.yolo_cli.call_async(req)
        future.add_done_callback(_yolo_cb)
        _yolo_event.wait(timeout=8.0)
        yolo_resp = _yolo_result[0]
        t_yolo_end = time.monotonic()
        timing['yolo_ms'] = round((t_yolo_end - t_yolo_start) * 1000, 1)

        if yolo_resp is None or not yolo_resp.success or not yolo_resp.detections:
            msg = f'YOLO 未检测到 [{cls}]'
            if yolo_resp and yolo_resp.message:
                msg += f'：{yolo_resp.message}'
            response.success = False
            response.message = msg
            self.get_logger().info(msg)
            return response

        best = max(yolo_resp.detections, key=lambda d: d.confidence)
        timing['yolo_confidence'] = round(float(best.confidence), 4)
        timing['yolo_center_3d'] = [round(float(v), 4) for v in best.center_point]
        self.get_logger().info(
            f'YOLO [{cls}] conf={best.confidence:.3f}  '
            f'3D=({best.center_point[0]:.3f},'
            f'{best.center_point[1]:.3f},'
            f'{best.center_point[2]:.3f})m  '
            f'耗时={timing["yolo_ms"]:.0f}ms'
        )

        # ── 3. 查找对应 .obj 文件 ─────────────────────────────
        obj_path = self._find_obj(cls)
        if obj_path is None:
            mesh_name = self.class_to_mesh.get(cls, cls)
            response.success = False
            response.message = f'未在 {self.mesh_dir} 找到 {mesh_name}.obj'
            self.get_logger().error(response.message)
            return response
        timing['mesh_file'] = obj_path

        # ── 4. 如果类别切换或 FP 崩溃，重启 FP ──────────────
        fp_crashed = (self.fp_process is not None and
                      self.fp_process.poll() is not None)
        fp_needs_launch = (cls != self.current_class or fp_crashed)
        timing['fp_model_loaded_before'] = not fp_needs_launch

        if fp_needs_launch:
            reason = '类别切换' if cls != self.current_class else 'FP 进程异常'
            self.get_logger().info(f'{reason}，重启 FoundationPose → {obj_path}')
            t_fp_launch = time.monotonic()
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
            t_fp_ready = time.monotonic()
            timing['fp_load_ms'] = round((t_fp_ready - t_fp_launch) * 1000, 1)
            self.get_logger().info(
                f'FoundationPose 已就绪  模型加载={timing["fp_load_ms"]:.0f}ms'
            )
        else:
            timing['fp_load_ms'] = 0.0
            self._latest_class = cls

        # ── 5. 图像转换 ───────────────────────────────────────
        t_img_start = time.monotonic()
        try:
            rgb_out   = self._to_rgb8(yolo_resp.rgb, self.bridge)
            depth_out = self._to_32fc1(yolo_resp.depth, self.bridge)
            mask_msg  = best.mask
        except Exception as e:
            response.success = False
            response.message = f'图像转换失败: {e}'
            self.get_logger().error(response.message)
            return response

        stamp = yolo_resp.rgb.header.stamp
        for m in (rgb_out, depth_out, mask_msg):
            m.header.stamp = stamp
        cam_info = copy.deepcopy(self.camera_info)
        cam_info.header.stamp = stamp
        timing['img_convert_ms'] = round((time.monotonic() - t_img_start) * 1000, 1)

        # ── 6. 清空位姿事件，发布图像 ─────────────────────────
        self._pose_event.clear()
        self._latest_pose = None

        t_pub = time.monotonic()
        self.img_pub.publish(rgb_out)
        self.depth_pub.publish(depth_out)
        self.seg_pub.publish(mask_msg)
        self.cam_pub.publish(cam_info)
        timing['img_pub_ms'] = round((time.monotonic() - t_pub) * 1000, 1)
        self.get_logger().info(
            f'已发布图像 RGB {rgb_out.width}×{rgb_out.height}  '
            f'发布耗时={timing["img_pub_ms"]:.0f}ms'
        )

        # ── 7. 触发 FP selector ───────────────────────────────
        t_trigger = time.monotonic()
        if self.trigger_cli.service_is_ready():
            _trig_event = threading.Event()
            _trig_result = [None]
            def _trig_cb(f):
                _trig_result[0] = f.result()
                _trig_event.set()
            trig_fut = self.trigger_cli.call_async(Trigger.Request())
            trig_fut.add_done_callback(_trig_cb)
            _trig_event.wait(timeout=3.0)
            trig_res = _trig_result[0]
            if trig_res and trig_res.success:
                self.get_logger().info('已触发位姿估计')
            else:
                msg_str = trig_res.message if trig_res else '超时'
                self.get_logger().warn(f'触发返回: {msg_str}（继续等待）')
        else:
            self.get_logger().warn('trigger 服务未就绪，直接等待输出')
        timing['trigger_ms'] = round((time.monotonic() - t_trigger) * 1000, 1)

        # ── 8. 等待 FP 输出位姿 ───────────────────────────────
        t_wait = time.monotonic()
        if not self._pose_event.wait(timeout=self.pose_timeout):
            response.success = False
            response.message = (
                f'等待 FoundationPose 位姿超时（{self.pose_timeout:.0f}s）'
            )
            self.get_logger().warn(response.message)
            return response
        timing['fp_infer_ms'] = round((time.monotonic() - t_wait) * 1000, 1)

        # ── 9. 汇总计时 ───────────────────────────────────────
        timing['total_ms'] = round((time.monotonic() - t0) * 1000, 1)

        p = self._latest_pose.pose.position
        r = self._latest_pose.pose.orientation
        timing['pose'] = {
            'x': round(p.x, 6), 'y': round(p.y, 6), 'z': round(p.z, 6),
            'qx': round(r.x, 6), 'qy': round(r.y, 6),
            'qz': round(r.z, 6), 'qw': round(r.w, 6),
        }

        self._print_and_save_timing(timing)

        response.success = True
        response.message = f'OK  class={cls}'
        response.pose = self._latest_pose
        # 供测试脚本汇总平均耗时（与 fp_timing.jsonl 同结构）
        response.timing_json = json.dumps(timing, ensure_ascii=False)
        return response

    # ------------------------------------------------------------------
    # 计时结果打印 + 保存
    # ------------------------------------------------------------------
    def _print_and_save_timing(self, t: dict):
        sep = '─' * 60
        loaded_str = '（已加载，跳过）' if t['fp_model_loaded_before'] else ''
        lines = [
            '',
            sep,
            f'  6D 位姿估计计时报告  [{t["class"]}]  {t["timestamp"]}',
            sep,
            f'  ① YOLO 检测耗时          : {t["yolo_ms"]:>8.1f} ms',
            f'     置信度={t["yolo_confidence"]:.4f}  '
            f'中心3D={t["yolo_center_3d"]}m',
            f'  ② FP 模型加载耗时         : {t["fp_load_ms"]:>8.1f} ms  {loaded_str}',
            f'     mesh: {t["mesh_file"]}',
            f'  ③ 图像转换耗时            : {t["img_convert_ms"]:>8.1f} ms',
            f'  ④ 图像发布耗时            : {t["img_pub_ms"]:>8.1f} ms',
            f'  ⑤ FP 触发耗时             : {t["trigger_ms"]:>8.1f} ms',
            f'  ⑥ FP 推理/位姿计算耗时    : {t["fp_infer_ms"]:>8.1f} ms',
            sep,
            f'  总耗时（不含模型加载）    : '
            f'{t["total_ms"] - t["fp_load_ms"]:>8.1f} ms',
            f'  总耗时（含模型加载）      : {t["total_ms"]:>8.1f} ms',
            sep,
            f'  位姿结果:',
            f'    位置 x={t["pose"]["x"]:+.6f}  y={t["pose"]["y"]:+.6f}'
            f'  z={t["pose"]["z"]:+.6f}  (m)',
            f'    四元数 qx={t["pose"]["qx"]:+.6f}  qy={t["pose"]["qy"]:+.6f}'
            f'  qz={t["pose"]["qz"]:+.6f}  qw={t["pose"]["qw"]:+.6f}',
            sep,
        ]
        report = '\n'.join(lines)
        self.get_logger().info(report)

        if not self.timing_log_dir:
            return
        # 保存 JSONL（每次追加一行）
        jsonl_path = os.path.join(self.timing_log_dir, 'fp_timing.jsonl')
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
        # 保存可读文本（每次追加）
        txt_path = os.path.join(self.timing_log_dir, 'fp_timing.txt')
        with open(txt_path, 'a') as f:
            f.write(report + '\n')
        self.get_logger().info(f'计时日志已保存: {jsonl_path}')

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
