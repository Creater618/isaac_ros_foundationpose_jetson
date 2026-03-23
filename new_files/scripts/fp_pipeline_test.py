#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fp_pipeline_test.py
===================
完整 pipeline 测试脚本：调用 /foundationpose/detect，验证从 YOLO 输入到
6D 位姿输出的全流程，并打印 + 保存详细计时报告。

用法：
  conda activate foundationpose
  source /opt/ros/humble/setup.bash
  source <isaac_ros_fp>/install/setup.bash
  source <niusuo_perception>/install/setup.bash

  # 单次测试
  python3 fp_pipeline_test.py k2c

  # 多次测试取均值
  python3 fp_pipeline_test.py k2c --repeat 5

  # 指定日志输出目录
  python3 fp_pipeline_test.py k2c --repeat 3 --log-dir /tmp/fp_test
"""

import argparse
import json
import math
import os
import sys
import datetime
import time

import rclpy
from rclpy.node import Node
from yolo_interfaces.srv import FpDetect


ANSI = {
    'GREEN':  '\033[92m',
    'YELLOW': '\033[93m',
    'RED':    '\033[91m',
    'CYAN':   '\033[96m',
    'BOLD':   '\033[1m',
    'RESET':  '\033[0m',
}


def colored(text, key):
    return ANSI.get(key, '') + str(text) + ANSI['RESET']


def quat_to_euler_deg(x, y, z, w):
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr, cosr))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny, cosy))
    return roll, pitch, yaw


class PipelineTester(Node):
    def __init__(self):
        super().__init__('fp_pipeline_tester')
        self.cli = self.create_client(FpDetect, '/foundationpose/detect')

    def run_once(self, class_name: str, idx: int) -> dict | None:
        """调用一次 /foundationpose/detect，返回含计时的结果字典。"""
        if not self.cli.wait_for_service(timeout_sec=5.0):
            print(colored('  /foundationpose/detect 服务不可用', 'RED'))
            return None

        req = FpDetect.Request()
        req.class_name = class_name

        t_call = time.monotonic()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=120.0)
        t_return = time.monotonic()
        round_trip_ms = (t_return - t_call) * 1000

        resp = future.result()
        if resp is None:
            print(colored(f'  [{idx}] 服务调用超时（无响应）', 'RED'))
            return None

        result = {
            'index': idx,
            'timestamp': datetime.datetime.now().isoformat(),
            'class': class_name,
            'success': resp.success,
            'message': resp.message,
            'round_trip_ms': round(round_trip_ms, 1),
        }

        if resp.success:
            p = resp.pose.pose.position
            r = resp.pose.pose.orientation
            roll, pitch, yaw = quat_to_euler_deg(r.x, r.y, r.z, r.w)
            result['pose'] = {
                'frame': resp.pose.header.frame_id,
                'x': round(p.x, 6), 'y': round(p.y, 6), 'z': round(p.z, 6),
                'qx': round(r.x, 6), 'qy': round(r.y, 6),
                'qz': round(r.z, 6), 'qw': round(r.w, 6),
                'roll_deg': round(roll, 2),
                'pitch_deg': round(pitch, 2),
                'yaw_deg': round(yaw, 2),
            }
        return result

    def print_result(self, r: dict):
        sep = '═' * 60
        print(f'\n{colored(sep, "CYAN")}')
        status = colored('✓ 成功', 'GREEN') if r['success'] else colored('✗ 失败', 'RED')
        print(f'  [{r["index"]}] {r["class"]}  {status}  {r["timestamp"]}')
        print(f'  消息: {r["message"]}')
        rt = r['round_trip_ms']
        print(f'  Service 往返耗时: {colored(f"{rt:.0f} ms", "CYAN")}')
        print(f'  （详细各阶段计时由 bridge 节点打印并写入 fp_timing_logs/）')
        if r['success']:
            p = r['pose']
            print(f'\n  {colored("6D 位姿结果", "BOLD")}  frame: {p["frame"]}')
            print(f'  位置  x={p["x"]:+.6f}  y={p["y"]:+.6f}  z={p["z"]:+.6f}  (m)')
            print(f'  四元数  qx={p["qx"]:+.6f}  qy={p["qy"]:+.6f}'
                  f'  qz={p["qz"]:+.6f}  qw={p["qw"]:+.6f}')
            print(f'  欧拉角  roll={p["roll_deg"]:+.2f}°  '
                  f'pitch={p["pitch_deg"]:+.2f}°  yaw={p["yaw_deg"]:+.2f}°')
        print(colored(sep, 'CYAN'))

    def print_summary(self, results: list[dict]):
        ok = [r for r in results if r and r['success']]
        fail = [r for r in results if r and not r['success']]
        none = [r for r in results if r is None]
        sep = '═' * 60
        print(f'\n{colored("="*60, "BOLD")}')
        print(colored('  Pipeline 测试汇总', 'BOLD'))
        print(colored('='*60, 'BOLD'))
        print(f'  总次数: {len(results)}  '
              f'成功: {colored(len(ok), "GREEN")}  '
              f'失败: {colored(len(fail), "RED")}  '
              f'超时/无响应: {len(none)}')

        if ok:
            rts = [r['round_trip_ms'] for r in ok]
            print(f'\n  Service 往返耗时（成功次）:')
            print(f'    最小={min(rts):.0f}ms  最大={max(rts):.0f}ms  '
                  f'均值={sum(rts)/len(rts):.0f}ms')

            xs = [r['pose']['x'] for r in ok]
            ys = [r['pose']['y'] for r in ok]
            zs = [r['pose']['z'] for r in ok]
            print(f'\n  位置范围:')
            print(f'    x=[{min(xs):+.4f}, {max(xs):+.4f}]  '
                  f'y=[{min(ys):+.4f}, {max(ys):+.4f}]  '
                  f'z=[{min(zs):+.4f}, {max(zs):+.4f}]  (m)')
            print(f'  位置均值:')
            n = len(ok)
            print(f'    x={sum(xs)/n:+.4f}  y={sum(ys)/n:+.4f}  z={sum(zs)/n:+.4f}  (m)')

        print(colored('='*60, 'BOLD'))
        print('  详细各阶段计时（YOLO/FP加载/FP推理）已由 bridge 写入:')
        print('    /media/rykj/nvme/jetson/fp_timing_logs/fp_timing.txt')
        print('    /media/rykj/nvme/jetson/fp_timing_logs/fp_timing.jsonl')
        print(colored('='*60, 'BOLD'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name', nargs='?', default='k2c',
                        help='目标类别 (default: k2c)')
    parser.add_argument('--repeat', '-n', type=int, default=1,
                        help='测试次数 (default: 1)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='每次测试间隔秒 (default: 2.0)')
    parser.add_argument('--log-dir', default='/media/rykj/nvme/jetson/fp_timing_logs',
                        help='本脚本的结果保存目录')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = PipelineTester()

    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.log_dir, f'test_{ts}_{args.class_name}.jsonl')

    print(colored('\n  FoundationPose Pipeline 测试', 'BOLD'))
    print(f'  类别: {args.class_name}  次数: {args.repeat}  间隔: {args.interval}s')
    print(f'  结果日志: {log_path}')

    results = []
    for i in range(args.repeat):
        print(f'\n{colored(f"  ── 第 {i+1}/{args.repeat} 次 ──", "YELLOW")}')
        r = node.run_once(args.class_name, i + 1)
        results.append(r)
        if r:
            node.print_result(r)
            with open(log_path, 'a') as f:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        if i < args.repeat - 1:
            print(f'  等待 {args.interval}s...')
            time.sleep(args.interval)

    if args.repeat > 1:
        node.print_summary(results)
        # 也追加 summary 到 log
        summary = {
            'type': 'summary',
            'timestamp': datetime.datetime.now().isoformat(),
            'class': args.class_name,
            'total': len(results),
            'success': sum(1 for r in results if r and r['success']),
        }
        ok = [r for r in results if r and r['success']]
        if ok:
            rts = [r['round_trip_ms'] for r in ok]
            summary['round_trip_ms'] = {
                'min': min(rts), 'max': max(rts),
                'mean': round(sum(rts) / len(rts), 1)
            }
        with open(log_path, 'a') as f:
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
