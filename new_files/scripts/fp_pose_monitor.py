#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fp_pose_monitor.py
==================
实时监控 FoundationPose 的 6D 位姿输出，并验证数据质量。

用法（任意终端，只需 source ROS 环境）：
  source /opt/ros/humble/setup.bash
  source /media/rykj/nvme/jetson/isaac_ros_foundationpose/install/setup.bash
  python3 fp_pose_monitor.py [--duration 30]

  或直接 ros2 run：
  ros2 run isaac_ros_foundationpose fp_pose_monitor.py

订阅：
  /tracking/output          — FoundationPose 跟踪输出  (Detection3DArray)
  /pose_estimation/output   — FoundationPose 初始估计  (Detection3DArray)
  /foundationpose/pose      — 桥接节点转发的 PoseStamped（供运控使用）

输出：
  终端彩色实时表格 + 统计（接收频率、位姿是否有效、x/y/z 范围）
  按 Ctrl+C 退出，退出时打印汇总
"""

import sys
import time
import math
import argparse
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection3DArray

# ANSI 颜色
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def quat_to_euler_deg(x, y, z, w):
    """四元数 → RPY（度）。"""
    # roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    # yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def quat_norm(x, y, z, w):
    return math.sqrt(x*x + y*y + z*z + w*w)


class PoseMonitor(Node):

    def __init__(self, duration: float):
        super().__init__('fp_pose_monitor')
        self.duration = duration
        self.start_time = time.monotonic()

        self._lock = threading.Lock()
        self._stats = {
            'tracking': {'count': 0, 'valid': 0, 'last_t': None,
                         'hz_buf': [], 'last_pose': None},
            'estimation': {'count': 0, 'valid': 0, 'last_t': None,
                           'hz_buf': [], 'last_pose': None},
            'pose_stamped': {'count': 0, 'valid': 0, 'last_t': None,
                             'hz_buf': [], 'last_pose': None},
        }

        self.create_subscription(
            Detection3DArray, '/tracking/output',
            lambda m: self._det_cb(m, 'tracking'), RELIABLE_QOS)
        self.create_subscription(
            Detection3DArray, '/pose_estimation/output',
            lambda m: self._det_cb(m, 'estimation'), RELIABLE_QOS)
        self.create_subscription(
            PoseStamped, '/foundationpose/pose',
            self._ps_cb, RELIABLE_QOS)

        # 打印定时器：每秒刷新
        self.create_timer(1.0, self._print_status)
        self.get_logger().info('FP 位姿监控已启动，按 Ctrl+C 退出')

    def _record(self, key: str, pose_dict: dict, is_valid: bool):
        now = time.monotonic()
        s = self._stats[key]
        with self._lock:
            s['count'] += 1
            if is_valid:
                s['valid'] += 1
            if s['last_t'] is not None:
                dt = now - s['last_t']
                if 0 < dt < 5.0:
                    s['hz_buf'].append(1.0 / dt)
                    if len(s['hz_buf']) > 20:
                        s['hz_buf'].pop(0)
            s['last_t'] = now
            s['last_pose'] = pose_dict

    def _det_cb(self, msg: Detection3DArray, key: str):
        if not msg.detections:
            self._record(key, None, False)
            return
        det = msg.detections[0]
        p = det.bbox.center.position
        r = det.bbox.center.orientation
        norm = quat_norm(r.x, r.y, r.z, r.w)
        is_valid = (
            abs(p.x) + abs(p.y) + abs(p.z) > 1e-6 and
            abs(norm - 1.0) < 0.05
        )
        pose_dict = dict(x=p.x, y=p.y, z=p.z,
                         qx=r.x, qy=r.y, qz=r.z, qw=r.w, norm=norm,
                         frame=msg.header.frame_id)
        self._record(key, pose_dict, is_valid)

    def _ps_cb(self, msg: PoseStamped):
        p = msg.pose.position
        r = msg.pose.orientation
        norm = quat_norm(r.x, r.y, r.z, r.w)
        is_valid = abs(p.x) + abs(p.y) + abs(p.z) > 1e-6 and abs(norm - 1.0) < 0.05
        pose_dict = dict(x=p.x, y=p.y, z=p.z,
                         qx=r.x, qy=r.y, qz=r.z, qw=r.w, norm=norm,
                         frame=msg.header.frame_id)
        self._record('pose_stamped', pose_dict, is_valid)

    def _hz(self, key: str) -> str:
        buf = self._stats[key]['hz_buf']
        if not buf:
            return f'{RED}-- Hz{RESET}'
        hz = sum(buf) / len(buf)
        color = GREEN if hz > 0.5 else YELLOW
        return f'{color}{hz:.1f} Hz{RESET}'

    def _pose_line(self, key: str) -> str:
        s = self._stats[key]
        pose = s['last_pose']
        if pose is None:
            return f'  {RED}暂无数据{RESET}'
        x, y, z = pose['x'], pose['y'], pose['z']
        roll, pitch, yaw = quat_to_euler_deg(pose['qx'], pose['qy'],
                                              pose['qz'], pose['qw'])
        norm_ok = abs(pose['norm'] - 1.0) < 0.05
        q_str = (f'{GREEN}✓{RESET}' if norm_ok
                 else f'{RED}✗ norm={pose["norm"]:.3f}{RESET}')
        valid_ratio = s['valid'] / max(s['count'], 1) * 100
        return (
            f'  位置: {CYAN}x={x:+.4f}  y={y:+.4f}  z={z:+.4f}{RESET} m\n'
            f'  姿态: roll={roll:+6.1f}°  pitch={pitch:+6.1f}°  '
            f'yaw={yaw:+6.1f}°  q{q_str}\n'
            f'  frame: {pose["frame"] or "(无)"}  '
            f'有效率: {valid_ratio:.0f}%  总计: {s["count"]} 帧'
        )

    def _print_status(self):
        elapsed = time.monotonic() - self.start_time
        remaining = self.duration - elapsed if self.duration > 0 else -1

        lines = []
        lines.append(f'\n{"="*60}')
        lines.append(
            f'{BOLD}FP 位姿监控{RESET}  '
            f'运行 {elapsed:.0f}s'
            + (f'  剩余 {remaining:.0f}s' if remaining > 0 else '')
        )
        lines.append('─'*60)

        for key, label in [
            ('tracking',    '【Tracking 跟踪输出】   → 运控主要来源'),
            ('estimation',  '【PoseEstimation 初估】  → 首帧/重置'),
            ('pose_stamped','【/foundationpose/pose】 → PoseStamped 话题'),
        ]:
            lines.append(f'\n{BOLD}{label}{RESET}  {self._hz(key)}')
            lines.append(self._pose_line(key))

        lines.append('─'*60)
        lines.append('订阅话题: /tracking/output  /pose_estimation/output  '
                     '/foundationpose/pose')
        lines.append('运控订阅: /foundationpose/pose  (geometry_msgs/PoseStamped)')
        print('\n'.join(lines))

        if self.duration > 0 and elapsed >= self.duration:
            self._print_summary()
            rclpy.shutdown()

    def _print_summary(self):
        print(f'\n{"="*60}')
        print(f'{BOLD}测试完成 - 汇总报告{RESET}')
        for key, label in [('tracking', 'Tracking'),
                            ('estimation', 'PoseEstimation'),
                            ('pose_stamped', 'PoseStamped')]:
            s = self._stats[key]
            hz_avg = (sum(s['hz_buf']) / len(s['hz_buf'])) if s['hz_buf'] else 0.0
            valid_pct = s['valid'] / max(s['count'], 1) * 100
            status = GREEN + '正常' + RESET if s['count'] > 0 else RED + '无数据' + RESET
            print(f'  {label:20s}: {status}  {s["count"]} 帧  '
                  f'{hz_avg:.1f} Hz  有效率 {valid_pct:.0f}%')
        print('='*60)


def main():
    parser = argparse.ArgumentParser(description='FoundationPose 位姿监控工具')
    parser.add_argument('--duration', type=float, default=0,
                        help='运行时长（秒），0=持续运行直到 Ctrl+C')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = PoseMonitor(args.duration)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._print_summary()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
