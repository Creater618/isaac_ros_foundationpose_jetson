#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_fp_bridge.py
=================
将 YOLO 检测结果桥接到 isaac_ros_foundationpose 的适配节点。

功能：
  1. 周期轮询 yolo_detect 服务获取指定类别的检测结果
  2. 根据 class_name（或 class_to_mesh 映射）在 mesh_dir 下查找对应 .obj 文件
  3. 若检测到新类别（或 FoundationPose 未启动），自动以对应 .obj 重启 FP 子进程
  4. 将 YOLO 的 RGB（bgr8→rgb8）、Depth（16UC1 mm→32FC1 m）、Mask（mono8）、
     CameraInfo 发布到 FoundationPose selector 的输入话题
  5. 调用 /foundationpose/trigger_pose_estimation 触发位姿估计
  6. 订阅 /pose_estimation/output 和 /tracking/output，打印 6D 位姿结果

用法（在 foundationpose conda 环境中）：
  source /opt/ros/humble/setup.bash
  source /media/rykj/nvme/jetson/isaac_ros_foundationpose/install/setup.bash
  source /media/rykj/nvme/jetson/ga/code/ros2_ws/install/setup.bash
  ros2 launch isaac_ros_foundationpose yolo_fp_bridge.launch.py

可调参数（--ros-args -p key:=value）：
  mesh_dir          /media/rykj/nvme/jetson   .obj 文件搜索目录
  class_names       k2c                        要跟踪的类别列表（逗号分隔）
  class_to_mesh     k2c:K2                     YOLO类名到obj文件名的映射（逗号分隔键值对，如k2c:K2,j2:J2）
  yolo_service      yolo_detect               YOLO 服务名
  camera_info_topic /right_camera/color/camera_info
  poll_interval     1.0                       轮询间隔（秒）
  trigger_service   /foundationpose/trigger_pose_estimation
  refine_engine     /tmp/refine_trt_engine.plan
  score_engine      /tmp/score_trt_engine.plan
  refine_model      /tmp/refine_model.onnx
  score_model       /tmp/score_model.onnx
  default_texture   /tmp/k2c_texture.png
  fp_wait_timeout   60.0                      等待 FP 启动的超时（秒）
"""

import copy
import os
import sys
import time
import signal
import subprocess
import threading

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray

from yolo_interfaces.srv import YoloDetect


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class YoloFpBridgeNode(Node):

    def __init__(self):
        super().__init__('yolo_fp_bridge')

        # ── 参数 ──────────────────────────────────────────────
        self.declare_parameter('mesh_dir', '/media/rykj/nvme/jetson')
        # YOLO 实际类名列表（如 k2c, j2），逗号分隔
        self.declare_parameter('class_names', 'k2c')
        # YOLO类名 → OBJ文件名 映射，格式: "k2c:K2,j2:J2"
        self.declare_parameter('class_to_mesh', 'k2c:K2')
        self.declare_parameter('yolo_service', 'yolo_detect')
        self.declare_parameter('camera_info_topic', '/right_camera/color/camera_info')
        self.declare_parameter('poll_interval', 1.0)
        self.declare_parameter('trigger_service', '/foundationpose/trigger_pose_estimation')
        self.declare_parameter('refine_engine', '/tmp/refine_trt_engine.plan')
        self.declare_parameter('score_engine', '/tmp/score_trt_engine.plan')
        self.declare_parameter('refine_model', '/tmp/refine_model.onnx')
        self.declare_parameter('score_model', '/tmp/score_model.onnx')
        self.declare_parameter('default_texture', '/tmp/k2c_texture.png')
        self.declare_parameter('fp_wait_timeout', 60.0)

        self.mesh_dir = self.get_parameter('mesh_dir').value

        raw_classes = self.get_parameter('class_names').value
        if isinstance(raw_classes, str):
            self.class_names = [c.strip() for c in raw_classes.split(',') if c.strip()]
        else:
            self.class_names = list(raw_classes)

        # 解析 class_to_mesh 映射
        raw_map = self.get_parameter('class_to_mesh').value
        self.class_to_mesh: dict[str, str] = {}
        if raw_map:
            for pair in raw_map.split(','):
                pair = pair.strip()
                if ':' in pair:
                    k, v = pair.split(':', 1)
                    self.class_to_mesh[k.strip()] = v.strip()

        yolo_service = self.get_parameter('yolo_service').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        poll_interval = self.get_parameter('poll_interval').value
        trigger_srv = self.get_parameter('trigger_service').value
        self.refine_engine = self.get_parameter('refine_engine').value
        self.score_engine = self.get_parameter('score_engine').value
        self.refine_model = self.get_parameter('refine_model').value
        self.score_model = self.get_parameter('score_model').value
        self.default_texture = self.get_parameter('default_texture').value
        self.fp_wait_timeout = self.get_parameter('fp_wait_timeout').value

        self.bridge = CvBridge()

        # ── 状态 ──────────────────────────────────────────────
        self.camera_info: CameraInfo | None = None
        self.current_class: str | None = None
        self.fp_process: subprocess.Popen | None = None
        self._fp_lock = threading.Lock()

        # ── 发布到 selector 输入话题 ───────────────────────────
        self.img_pub = self.create_publisher(Image, '/image', 10)
        self.depth_pub = self.create_publisher(Image, '/depth_image', 10)
        self.seg_pub = self.create_publisher(Image, '/segmentation', 10)
        self.cam_pub = self.create_publisher(CameraInfo, '/camera_info', 10)

        # ── 订阅相机内参（缓存一次即可）──────────────────────────
        self.create_subscription(CameraInfo, camera_info_topic,
                                 self._cam_info_cb, SENSOR_QOS)

        # ── YOLO 服务客户端 ────────────────────────────────────
        self.yolo_cli = self.create_client(YoloDetect, yolo_service)

        # ── FP 触发服务客户端 ──────────────────────────────────
        self.trigger_cli = self.create_client(Trigger, trigger_srv)

        # ── 订阅 FP 输出（打印结果）────────────────────────────
        self.create_subscription(Detection3DArray, '/pose_estimation/output',
                                 self._pose_est_cb, 10)
        self.create_subscription(Detection3DArray, '/tracking/output',
                                 self._tracking_cb, 10)

        # ── 轮询定时器 ────────────────────────────────────────
        self.timer = self.create_timer(poll_interval, self._poll)

        self.get_logger().info(
            f'YoloFpBridge 启动  mesh_dir={self.mesh_dir}  '
            f'classes={self.class_names}  mapping={self.class_to_mesh}  '
            f'poll={poll_interval}s'
        )

    # ------------------------------------------------------------------
    # 相机内参回调
    # ------------------------------------------------------------------
    def _cam_info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(
                f'获取相机内参  frame={msg.header.frame_id}  '
                f'{msg.width}×{msg.height}'
            )

    # ------------------------------------------------------------------
    # FP 位姿输出回调（打印结果）
    # ------------------------------------------------------------------
    def _pose_est_cb(self, msg: Detection3DArray):
        for det in msg.detections:
            p = det.bbox.center.position
            r = det.bbox.center.orientation
            self.get_logger().info(
                f'[PoseEstimation] xyz=({p.x:.3f},{p.y:.3f},{p.z:.3f})  '
                f'qxyzw=({r.x:.3f},{r.y:.3f},{r.z:.3f},{r.w:.3f})'
            )

    def _tracking_cb(self, msg: Detection3DArray):
        for det in msg.detections:
            p = det.bbox.center.position
            r = det.bbox.center.orientation
            self.get_logger().info(
                f'[Tracking]       xyz=({p.x:.3f},{p.y:.3f},{p.z:.3f})  '
                f'qxyzw=({r.x:.3f},{r.y:.3f},{r.z:.3f},{r.w:.3f})'
            )

    # ------------------------------------------------------------------
    # 查找 obj 文件（支持 class_to_mesh 映射）
    # ------------------------------------------------------------------
    def _find_obj(self, class_name: str) -> str | None:
        """
        在 mesh_dir 下查找对应的 .obj 文件。
        优先使用 class_to_mesh 映射中指定的文件名，
        若无映射则直接用 class_name 作文件名（大小写不敏感）。
        """
        # 优先使用映射后的文件名
        mesh_name = self.class_to_mesh.get(class_name, class_name)

        for fname in os.listdir(self.mesh_dir):
            if fname.lower() == f'{mesh_name.lower()}.obj':
                full_path = os.path.join(self.mesh_dir, fname)
                self.get_logger().info(
                    f'找到 [{class_name}] 对应网格: {full_path}'
                )
                return full_path
        return None

    # ------------------------------------------------------------------
    # FoundationPose 子进程管理
    # ------------------------------------------------------------------
    def _launch_fp(self, mesh_path: str):
        """以新的 mesh_file_path 重启 FoundationPose。"""
        with self._fp_lock:
            self._kill_fp_locked()

            # 查找同名 texture（可选）
            base = os.path.splitext(mesh_path)[0]
            texture = (base + '.png') if os.path.exists(base + '.png') else self.default_texture

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
            self.get_logger().info(f'启动 FoundationPose: {" ".join(cmd)}')
            self.fp_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )

            # 后台线程打印 FP 日志
            def _log_fp():
                for line in self.fp_process.stdout:
                    try:
                        self.get_logger().info(
                            '[FP] ' + line.decode('utf-8', errors='replace').rstrip()
                        )
                    except Exception:
                        pass

            threading.Thread(target=_log_fp, daemon=True).start()

    def _kill_fp_locked(self):
        """杀掉当前 FP 子进程（已持有锁）。"""
        if self.fp_process and self.fp_process.poll() is None:
            self.get_logger().info('停止当前 FoundationPose...')
            try:
                os.killpg(os.getpgid(self.fp_process.pid), signal.SIGINT)
                self.fp_process.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.fp_process.pid), signal.SIGKILL)
                except Exception:
                    pass
        self.fp_process = None

    def _fp_is_ready(self) -> bool:
        """检查 FP 触发服务是否可用。"""
        return self.trigger_cli.service_is_ready()

    def _wait_fp_ready(self) -> bool:
        """阻塞等待 FP 就绪，超时返回 False。"""
        deadline = time.monotonic() + self.fp_wait_timeout
        while time.monotonic() < deadline:
            if self._fp_is_ready():
                return True
            time.sleep(1.0)
        return False

    # ------------------------------------------------------------------
    # 图像格式转换
    # ------------------------------------------------------------------
    @staticmethod
    def _to_rgb8(msg: Image, bridge: CvBridge) -> Image:
        """将任意编码的 BGR/RGB 图像转为 rgb8。"""
        enc = msg.encoding.lower()
        if enc == 'rgb8':
            return msg
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        out.header = msg.header
        return out

    @staticmethod
    def _to_32fc1(msg: Image, bridge: CvBridge) -> Image:
        """将 16UC1（毫米）或 32FC1 深度图转为 32FC1（米）。"""
        enc = msg.encoding.lower()
        if enc == '32fc1':
            return msg
        depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001  # mm → m
        out = bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding='32FC1')
        out.header = msg.header
        return out

    # ------------------------------------------------------------------
    # 主轮询回调
    # ------------------------------------------------------------------
    def _poll(self):
        if not self.yolo_cli.service_is_ready():
            self.get_logger().warn('等待 yolo_detect 服务...', throttle_duration_sec=5.0)
            return
        if self.camera_info is None:
            self.get_logger().warn('等待 camera_info...', throttle_duration_sec=5.0)
            return

        # 依次查询每个目标类别，取置信度最高的检测
        best_class = None
        best_conf = 0.0
        best_det = None
        best_rgb = None
        best_depth = None

        for cls in self.class_names:
            req = YoloDetect.Request()
            req.class_name = cls
            future = self.yolo_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            resp = future.result()
            if resp is None or not resp.success or not resp.detections:
                continue
            for det in resp.detections:
                if det.confidence > best_conf:
                    best_conf = det.confidence
                    best_class = cls
                    best_det = det
                    best_rgb = resp.rgb
                    best_depth = resp.depth

        if best_det is None:
            self.get_logger().info(
                f'未检测到目标（classes={self.class_names}）', throttle_duration_sec=5.0
            )
            return

        self.get_logger().info(
            f'检测到 [{best_class}] conf={best_conf:.2f}  '
            f'3D=({best_det.center_point[0]:.3f},'
            f'{best_det.center_point[1]:.3f},'
            f'{best_det.center_point[2]:.3f})m'
        )

        # ── 查找对应 obj 文件 ─────────────────────────────────
        obj_path = self._find_obj(best_class)
        if obj_path is None:
            mesh_name = self.class_to_mesh.get(best_class, best_class)
            self.get_logger().error(
                f'未在 {self.mesh_dir} 找到 {mesh_name}.obj（class={best_class}），跳过'
            )
            return

        # ── 若类别变化，重启 FP ───────────────────────────────
        if best_class != self.current_class:
            self.get_logger().info(
                f'类别切换 {self.current_class} → {best_class}，'
                f'加载网格: {obj_path}'
            )
            self._launch_fp(obj_path)
            self.current_class = best_class
            self.get_logger().info(f'等待 FoundationPose 就绪（最多 {self.fp_wait_timeout:.0f}s）...')
            if not self._wait_fp_ready():
                self.get_logger().error('FoundationPose 启动超时，跳过本次')
                return
            self.get_logger().info('FoundationPose 已就绪')

        # ── FP 进程意外退出则重启 ──────────────────────────────
        if self.fp_process and self.fp_process.poll() is not None:
            self.get_logger().warn('FoundationPose 意外退出，重启...')
            self._launch_fp(obj_path)
            if not self._wait_fp_ready():
                self.get_logger().error('FoundationPose 重启超时')
                return

        # ── 若 FP 触发服务忙，跳过 ────────────────────────────
        if not self._fp_is_ready():
            return

        # ── 图像转换并发布（共用同一时间戳） ─────────────────────
        stamp = best_rgb.header.stamp

        try:
            rgb_out = self._to_rgb8(best_rgb, self.bridge)
            depth_out = self._to_32fc1(best_depth, self.bridge)
            mask_msg: Image = best_det.mask
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return

        # 统一时间戳（ExactSync 要求完全一致）
        for msg in (rgb_out, depth_out, mask_msg):
            msg.header.stamp = stamp

        cam_info = copy.deepcopy(self.camera_info)
        cam_info.header.stamp = stamp

        self.img_pub.publish(rgb_out)
        self.depth_pub.publish(depth_out)
        self.seg_pub.publish(mask_msg)
        self.cam_pub.publish(cam_info)

        self.get_logger().info(
            f'已发布: RGB {rgb_out.width}×{rgb_out.height}  '
            f'Depth {depth_out.width}×{depth_out.height}  '
            f'Mask {mask_msg.width}×{mask_msg.height}'
        )

        # ── 触发位姿估计 ──────────────────────────────────────
        trigger_future = self.trigger_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, trigger_future, timeout_sec=3.0)
        result = trigger_future.result()
        if result and result.success:
            self.get_logger().info('已触发位姿估计')
        else:
            msg_str = result.message if result else '超时'
            self.get_logger().warn(f'触发失败: {msg_str}')

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
