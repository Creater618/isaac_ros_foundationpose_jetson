#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fp_pipeline_test.py
===================
调用 /foundationpose/detect，验证 YOLO + FoundationPose 全流程，保存每次计时与平均耗时。

默认 10 次（与「启动 10 次」对齐）；每次成功时解析响应中的 timing_json（各阶段 ms）。

用法：
  conda activate foundationpose
  source /opt/ros/humble/setup.bash
  source <isaac_ros_fp>/install/setup.bash
  source <niusuo_perception>/install/setup.bash

  # 默认 10 次
  python3 fp_pipeline_test.py k2c

  python3 fp_pipeline_test.py k2c --repeat 5 --interval 2.0
  python3 fp_pipeline_test.py k2c --log-dir /media/rykj/nvme/jetson/fp_timing_logs
"""

from __future__ import annotations

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

# 参与「平均」的数值字段（来自 bridge timing_json）
TIMING_AVG_KEYS = [
    'yolo_ms',
    'fp_load_ms',
    'img_convert_ms',
    'img_pub_ms',
    'trigger_ms',
    'fp_infer_ms',
    'total_ms',
]


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


def timing_to_text(t: dict, idx: int) -> str:
    """与 bridge 终端报告格式一致的可读文本。"""
    sep = '─' * 60
    loaded = '（已加载，跳过）' if t.get('fp_model_loaded_before') else ''
    lines = [
        sep,
        f'  第 {idx} 次  6D 位姿估计计时  [{t.get("class", "?")}]  {t.get("timestamp", "")}',
        sep,
        f'  ① YOLO 检测耗时          : {t.get("yolo_ms", 0):>8.1f} ms',
        f'  ② FP 模型加载耗时         : {t.get("fp_load_ms", 0):>8.1f} ms  {loaded}',
        f'  ③ 图像转换耗时            : {t.get("img_convert_ms", 0):>8.1f} ms',
        f'  ④ 图像发布耗时            : {t.get("img_pub_ms", 0):>8.1f} ms',
        f'  ⑤ FP 触发耗时             : {t.get("trigger_ms", 0):>8.1f} ms',
        f'  ⑥ FP 推理/位姿计算耗时    : {t.get("fp_infer_ms", 0):>8.1f} ms',
        sep,
        f'  总耗时（不含模型加载）    : '
        f'{t.get("total_ms", 0) - t.get("fp_load_ms", 0):>8.1f} ms',
        f'  总耗时（含模型加载）      : {t.get("total_ms", 0):>8.1f} ms',
        sep,
    ]
    if 'pose' in t:
        p = t['pose']
        lines.extend([
            f'  位姿 x={p.get("x", 0):+.6f} y={p.get("y", 0):+.6f} z={p.get("z", 0):+.6f}',
        ])
    lines.append(sep)
    return '\n'.join(lines)


def compute_timing_averages(timing_dicts: list[dict]) -> dict:
    """对多次成功的 timing 字典求算术平均。"""
    if not timing_dicts:
        return {}
    out: dict = {'runs_used': len(timing_dicts)}
    for key in TIMING_AVG_KEYS:
        vals = [float(d[key]) for d in timing_dicts if key in d]
        if vals:
            out[key] = {
                'mean_ms': round(sum(vals) / len(vals), 2),
                'min_ms': round(min(vals), 2),
                'max_ms': round(max(vals), 2),
                'n': len(vals),
            }
    # 派生：不含 FP 加载的总耗时
    excl = []
    for d in timing_dicts:
        if 'total_ms' in d and 'fp_load_ms' in d:
            excl.append(float(d['total_ms']) - float(d['fp_load_ms']))
    if excl:
        out['total_excluding_fp_load_ms'] = {
            'mean_ms': round(sum(excl) / len(excl), 2),
            'min_ms': round(min(excl), 2),
            'max_ms': round(max(excl), 2),
            'n': len(excl),
        }
    return out


def format_average_report(avg: dict, class_name: str, ts: str) -> str:
    sep = '═' * 60
    lines = [
        sep,
        f'  平均计时汇总  class={class_name}  {ts}',
        f'  成功样本数: {avg.get("runs_used", 0)}',
        sep,
    ]
    labels = {
        'yolo_ms': '① YOLO 检测',
        'fp_load_ms': '② FP 模型加载',
        'img_convert_ms': '③ 图像转换',
        'img_pub_ms': '④ 图像发布',
        'trigger_ms': '⑤ FP 触发',
        'fp_infer_ms': '⑥ FP 推理/位姿',
        'total_ms': '总耗时（含加载）',
        'total_excluding_fp_load_ms': '总耗时（不含加载）',
    }
    for key in TIMING_AVG_KEYS + ['total_excluding_fp_load_ms']:
        if key not in avg:
            continue
        s = avg[key]
        if isinstance(s, dict) and 'mean_ms' in s:
            label = labels.get(key, key)
            lines.append(
                f'  {label:22s}  均值={s["mean_ms"]:>8.1f} ms  '
                f'最小={s["min_ms"]:>8.1f}  最大={s["max_ms"]:>8.1f}  (n={s["n"]})'
            )
    lines.append(sep)
    return '\n'.join(lines)


class PipelineTester(Node):
    def __init__(self):
        super().__init__('fp_pipeline_tester')
        self.cli = self.create_client(FpDetect, '/foundationpose/detect')

    def run_once(self, class_name: str, idx: int) -> dict | None:
        if not self.cli.wait_for_service(timeout_sec=5.0):
            print(colored('  /foundationpose/detect 服务不可用', 'RED'))
            return None

        req = FpDetect.Request()
        req.class_name = class_name

        t_call = time.monotonic()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=180.0)
        t_return = time.monotonic()
        round_trip_ms = (t_return - t_call) * 1000

        resp = future.result()
        if resp is None:
            print(colored(f'  [{idx}] 服务调用超时（无响应）', 'RED'))
            return None

        timing_parsed: dict | None = None
        tj = getattr(resp, 'timing_json', '') or ''
        if tj.strip():
            try:
                timing_parsed = json.loads(tj)
            except json.JSONDecodeError:
                timing_parsed = None

        result = {
            'index': idx,
            'timestamp': datetime.datetime.now().isoformat(),
            'class': class_name,
            'success': resp.success,
            'message': resp.message,
            'round_trip_ms': round(round_trip_ms, 1),
            'timing': timing_parsed,
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
        if r.get('timing'):
            t = r['timing']
            print(f'  分阶段: YOLO={t.get("yolo_ms", "?")}ms  '
                  f'FP加载={t.get("fp_load_ms", "?")}ms  '
                  f'FP推理={t.get("fp_infer_ms", "?")}ms  '
                  f'总计={t.get("total_ms", "?")}ms')
        if r['success']:
            p = r['pose']
            print(f'\n  {colored("6D 位姿", "BOLD")}  frame: {p["frame"]}')
            print(f'  位置  x={p["x"]:+.6f}  y={p["y"]:+.6f}  z={p["z"]:+.6f}  (m)')
            print(f'  欧拉角  roll={p["roll_deg"]:+.2f}°  '
                  f'pitch={p["pitch_deg"]:+.2f}°  yaw={p["yaw_deg"]:+.2f}°')
        print(colored(sep, 'CYAN'))


def main():
    parser = argparse.ArgumentParser(description='YOLO+FoundationPose 批量计时测试')
    parser.add_argument('class_name', nargs='?', default='k2c',
                        help='目标类别 (default: k2c)')
    parser.add_argument('--repeat', '-n', type=int, default=10,
                        help='测试次数 (default: 10)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='每次测试间隔秒 (default: 2.0)')
    parser.add_argument('--log-dir', default='/media/rykj/nvme/jetson/fp_timing_logs',
                        help='日志根目录；每次运行会建子目录 batch_*')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = PipelineTester()

    os.makedirs(args.log_dir, exist_ok=True)
    batch_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = os.path.join(args.log_dir, f'batch_{batch_ts}_{args.class_name}')
    os.makedirs(batch_dir, exist_ok=True)
    summary_jsonl = os.path.join(batch_dir, 'runs.jsonl')

    print(colored('\n  FoundationPose Pipeline 批量测试', 'BOLD'))
    print(f'  类别: {args.class_name}  次数: {args.repeat}  间隔: {args.interval}s')
    print(f'  本批目录: {batch_dir}')
    print(f'  （bridge 仍会追加全局 fp_timing.txt / fp_timing.jsonl）')

    results: list[dict | None] = []
    for i in range(args.repeat):
        print(f'\n{colored(f"  ── 第 {i+1}/{args.repeat} 次 ──", "YELLOW")}')
        r = node.run_once(args.class_name, i + 1)
        results.append(r)
        if r:
            node.print_result(r)
            # 每次一条 JSON
            rec = {k: v for k, v in r.items() if k != 'timing'}
            if r.get('timing'):
                rec['timing'] = r['timing']
            with open(summary_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            # 每次一份可读计时报告
            if r.get('timing'):
                txt_path = os.path.join(batch_dir, f'run_{i+1:02d}_timing.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(timing_to_text(r['timing'], i + 1) + '\n')
                print(colored(f'  已保存: {txt_path}', 'GREEN'))
        if i < args.repeat - 1:
            print(f'  等待 {args.interval}s...')
            time.sleep(args.interval)

    ok = [r for r in results if r and r['success']]
    timings = [r['timing'] for r in ok if r.get('timing')]
    avg = compute_timing_averages(timings) if timings else {}

    end_ts = datetime.datetime.now().isoformat()
    summary_doc = {
        'type': 'batch_summary',
        'timestamp': end_ts,
        'class': args.class_name,
        'repeat_requested': args.repeat,
        'success_count': len(ok),
        'fail_count': sum(1 for r in results if r and not r['success']),
        'timeout_count': sum(1 for r in results if r is None),
        'round_trip_ms': None,
        'timing_averages': avg,
    }
    if ok:
        rts = [r['round_trip_ms'] for r in ok]
        summary_doc['round_trip_ms'] = {
            'mean': round(sum(rts) / len(rts), 1),
            'min': min(rts), 'max': max(rts), 'n': len(rts),
        }

    avg_json_path = os.path.join(batch_dir, 'average_timing.json')
    with open(avg_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_doc, f, ensure_ascii=False, indent=2)

    avg_txt = format_average_report(avg, args.class_name, end_ts)
    avg_txt_path = os.path.join(batch_dir, 'average_timing.txt')
    with open(avg_txt_path, 'w', encoding='utf-8') as f:
        f.write(avg_txt + '\n')
        if summary_doc.get('round_trip_ms'):
            rt = summary_doc['round_trip_ms']
            f.write(
                f'\n  Service 往返（成功次）  均值={rt["mean"]:.0f}ms  '
                f'最小={rt["min"]:.0f}  最大={rt["max"]:.0f}  n={rt["n"]}\n'
            )

    print(f'\n{colored("="*60, "BOLD")}')
    print(colored('  本批汇总', 'BOLD'))
    print(f'  成功 {len(ok)} / {args.repeat}')
    if avg:
        print(avg_txt)
    print(colored(f'  平均计时 JSON: {avg_json_path}', 'CYAN'))
    print(colored(f'  平均计时 TXT:  {avg_txt_path}', 'CYAN'))
    print(colored(f'  每次运行汇总:  {summary_jsonl}', 'CYAN'))
    print(colored('='*60, 'BOLD'))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
