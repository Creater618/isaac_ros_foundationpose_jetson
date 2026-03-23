# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
yolo_fp_bridge.launch.py  —  Service 触发模式
==============================================
启动桥接节点 yolo_fp_bridge.py。
FoundationPose 本体由桥接节点在收到 /foundationpose/detect 请求时自动启动。

用法：
  source /opt/ros/humble/setup.bash
  source <isaac_ros_fp>/install/setup.bash
  source <niusuo_perception>/install/setup.bash
  ros2 launch isaac_ros_foundationpose yolo_fp_bridge.launch.py

FSM 调用方式：
  ros2 service call /foundationpose/detect yolo_interfaces/srv/FpDetect \
    "{class_name: 'k2c'}"

可覆盖参数：
  mesh_dir          /media/rykj/nvme/jetson
  class_to_mesh     k2c:K2,j2:J2
  camera_info_topic /right_camera/color/camera_info
  pose_timeout      10.0    等待 FP 位姿输出的超时（秒）
  fp_wait_timeout   90.0    等待 FP 节点启动就绪的超时（秒）
"""
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('mesh_dir',
            default_value='/media/rykj/nvme/jetson'),
        DeclareLaunchArgument('class_to_mesh',
            default_value='k2c:K2,j2:J2',
            description='YOLO class name → OBJ filename mapping, e.g. k2c:K2,j2:J2'),
        DeclareLaunchArgument('camera_info_topic',
            default_value='/right_camera/color/camera_info'),
        DeclareLaunchArgument('refine_engine',
            default_value='/tmp/refine_trt_engine.plan'),
        DeclareLaunchArgument('score_engine',
            default_value='/tmp/score_trt_engine.plan'),
        DeclareLaunchArgument('refine_model',
            default_value='/tmp/refine_model.onnx'),
        DeclareLaunchArgument('score_model',
            default_value='/tmp/score_model.onnx'),
        DeclareLaunchArgument('default_texture',
            default_value='/tmp/k2c_texture.png'),
        DeclareLaunchArgument('fp_wait_timeout',
            default_value='90.0',
            description='Seconds to wait for FP node to become ready'),
        DeclareLaunchArgument('pose_timeout',
            default_value='10.0',
            description='Seconds to wait for FP pose output per request'),
    ]

    bridge_node = Node(
        package='isaac_ros_foundationpose',
        executable='yolo_fp_bridge.py',
        name='yolo_fp_bridge',
        output='screen',
        parameters=[{
            'mesh_dir':           LaunchConfiguration('mesh_dir'),
            'class_to_mesh':      LaunchConfiguration('class_to_mesh'),
            'camera_info_topic':  LaunchConfiguration('camera_info_topic'),
            'refine_engine':      LaunchConfiguration('refine_engine'),
            'score_engine':       LaunchConfiguration('score_engine'),
            'refine_model':       LaunchConfiguration('refine_model'),
            'score_model':        LaunchConfiguration('score_model'),
            'default_texture':    LaunchConfiguration('default_texture'),
            'fp_wait_timeout':    LaunchConfiguration('fp_wait_timeout'),
            'pose_timeout':       LaunchConfiguration('pose_timeout'),
        }],
    )

    return launch.LaunchDescription(args + [bridge_node])
