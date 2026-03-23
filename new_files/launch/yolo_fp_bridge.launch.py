# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""
yolo_fp_bridge.launch.py
========================
只启动桥接节点（yolo_fp_bridge.py）。
FoundationPose 本体由桥接节点在检测到目标后自动以正确的 mesh 启动。

用法：
  source /opt/ros/humble/setup.bash
  source /media/rykj/nvme/jetson/isaac_ros_foundationpose/install/setup.bash
  source /media/rykj/nvme/jetson/ga/code/ros2_ws/install/setup.bash
  ros2 launch isaac_ros_foundationpose yolo_fp_bridge.launch.py

可覆盖参数：
  mesh_dir        /media/rykj/nvme/jetson
  class_names     K2           （多个用逗号分隔，如 K2,J2）
  poll_interval   1.0
  camera_info_topic  /right_camera/color/camera_info
"""
import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mesh_dir_arg = DeclareLaunchArgument(
        'mesh_dir',
        default_value='/media/rykj/nvme/jetson',
        description='Directory to search for .obj files by class name',
    )
    class_names_arg = DeclareLaunchArgument(
        'class_names',
        default_value='k2c',
        description='Comma-separated list of YOLO class names to track (e.g. k2c,j2)',
    )
    class_to_mesh_arg = DeclareLaunchArgument(
        'class_to_mesh',
        default_value='k2c:K2',
        description='Mapping from YOLO class name to OBJ filename (e.g. k2c:K2,j2:J2)',
    )
    poll_interval_arg = DeclareLaunchArgument(
        'poll_interval',
        default_value='1.0',
        description='Polling interval in seconds for yolo_detect service',
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/right_camera/color/camera_info',
        description='Camera info topic',
    )
    refine_engine_arg = DeclareLaunchArgument(
        'refine_engine', default_value='/tmp/refine_trt_engine.plan',
    )
    score_engine_arg = DeclareLaunchArgument(
        'score_engine', default_value='/tmp/score_trt_engine.plan',
    )
    refine_model_arg = DeclareLaunchArgument(
        'refine_model', default_value='/tmp/refine_model.onnx',
    )
    score_model_arg = DeclareLaunchArgument(
        'score_model', default_value='/tmp/score_model.onnx',
    )
    default_texture_arg = DeclareLaunchArgument(
        'default_texture', default_value='/tmp/k2c_texture.png',
    )
    fp_wait_timeout_arg = DeclareLaunchArgument(
        'fp_wait_timeout', default_value='60.0',
        description='Seconds to wait for FoundationPose to become ready after launch',
    )
    pose_topic_arg = DeclareLaunchArgument(
        'pose_topic', default_value='/foundationpose/pose',
        description='PoseStamped topic published for motion control',
    )
    pose_frame_id_arg = DeclareLaunchArgument(
        'pose_frame_id', default_value='',
        description='Override frame_id in PoseStamped (empty = use camera frame)',
    )

    bridge_node = Node(
        package='isaac_ros_foundationpose',
        executable='yolo_fp_bridge.py',
        name='yolo_fp_bridge',
        output='screen',
        parameters=[{
            'mesh_dir': LaunchConfiguration('mesh_dir'),
            'class_names': LaunchConfiguration('class_names'),
            'class_to_mesh': LaunchConfiguration('class_to_mesh'),
            'poll_interval': LaunchConfiguration('poll_interval'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'refine_engine': LaunchConfiguration('refine_engine'),
            'score_engine': LaunchConfiguration('score_engine'),
            'refine_model': LaunchConfiguration('refine_model'),
            'score_model': LaunchConfiguration('score_model'),
            'default_texture': LaunchConfiguration('default_texture'),
            'fp_wait_timeout': LaunchConfiguration('fp_wait_timeout'),
            'pose_topic': LaunchConfiguration('pose_topic'),
            'pose_frame_id': LaunchConfiguration('pose_frame_id'),
        }],
    )

    return launch.LaunchDescription([
        mesh_dir_arg,
        class_names_arg,
        class_to_mesh_arg,
        poll_interval_arg,
        camera_info_topic_arg,
        refine_engine_arg,
        score_engine_arg,
        refine_model_arg,
        score_model_arg,
        default_texture_arg,
        fp_wait_timeout_arg,
        pose_topic_arg,
        pose_frame_id_arg,
        bridge_node,
    ])
