# isaac_ros_foundationpose — Jetson Native Build Patches

> **Platform**: NVIDIA Jetson (aarch64), JetPack 6 / Ubuntu 22.04 / ROS 2 Humble  
> **TensorRT**: 10.3 (native conda environment `foundationpose`, no Docker)  
> **Goal**: Run `isaac_ros_foundationpose` natively outside Docker, integrated with YOLO detection for online 6D pose estimation of K2 twist-lock connectors.

This repository contains **patches and new files** to apply on top of the official [NVIDIA Isaac ROS](https://github.com/NVIDIA-ISAAC-ROS) packages. It does **not** duplicate the full upstream source (~3 GB).

---

## Repository Structure

```
.
├── patches/
│   ├── 01_isaac_ros_common.patch          # FindTENSORRT.cmake → TRT 10.3
│   ├── 02_isaac_ros_dnn_inference.patch   # TRT 8.x → 10.x API full migration
│   ├── 03_isaac_ros_nitros.patch          # Detection3DArray dep fix
│   └── 04_isaac_ros_pose_estimation.patch # GXF fix + dynamic mesh + YOLO bridge
├── new_files/
│   ├── launch/
│   │   └── yolo_fp_bridge.launch.py       # Launch file for YOLO→FP bridge
│   └── scripts/
│       └── yolo_fp_bridge.py              # YOLO detection → FoundationPose bridge node
├── mesh/
│   └── K2.obj                             # K2 twist-lock mesh (origin-centered, with UV)
├── apply_patches.sh                       # One-shot patch apply script
├── setup_and_build.sh                     # Full workspace setup & build script
├── install_cvcuda_aarch64.sh              # CV-CUDA aarch64 installer helper
├── BUILD_JETSON_NATIVE.md                 # Detailed native build guide
└── README_CHANGES.md                      # Full changelog (Chinese)
```

---

## Quick Start

### 1. Clone upstream workspace

```bash
mkdir -p ~/isaac_ros_fp_ws/src && cd ~/isaac_ros_fp_ws/src

# Core packages
git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros
git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference
git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline
git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation
git clone --depth 1 https://github.com/ros-controls/negotiated
```

### 2. Clone this repo and apply patches

```bash
cd ~/isaac_ros_fp_ws
git clone https://github.com/Creater618/isaac_ros_foundationpose_jetson
cd isaac_ros_foundationpose_jetson
chmod +x apply_patches.sh
./apply_patches.sh ~/isaac_ros_fp_ws/src
```

### 3. Install CV-CUDA (required for GXF extensions)

See [BUILD_JETSON_NATIVE.md](BUILD_JETSON_NATIVE.md) for the CV-CUDA installation steps.

### 4. Build TRT engines

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/tmp/refine_model.onnx \
  --saveEngine=/tmp/refine_trt_engine.plan \
  --fp16

/usr/src/tensorrt/bin/trtexec \
  --onnx=/tmp/score_model.onnx \
  --saveEngine=/tmp/score_trt_engine.plan \
  --fp16
```

Download ONNX models from: https://github.com/NVlabs/FoundationPose

### 5. Build workspace

```bash
cd ~/isaac_ros_fp_ws
conda activate foundationpose
source /opt/ros/humble/setup.bash
colcon build --packages-up-to isaac_ros_foundationpose --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
```

### 6. Run

**FoundationPose standalone:**

```bash
ros2 launch isaac_ros_foundationpose isaac_ros_foundationpose_tracking.launch.py \
  mesh_file_path:=<path>/mesh/K2.obj \
  texture_path:=/tmp/k2c_texture.png \
  refine_engine_file_path:=/tmp/refine_trt_engine.plan \
  score_engine_file_path:=/tmp/score_trt_engine.plan \
  refine_model_file_path:=/tmp/refine_model.onnx \
  score_model_file_path:=/tmp/score_model.onnx \
  launch_rviz:=False
```

**Full YOLO + FoundationPose pipeline:**

```bash
# Terminal 1: YOLO service
ros2 run yolo26_seg yolo_node

# Terminal 2: Bridge node
ros2 launch isaac_ros_foundationpose yolo_fp_bridge.launch.py \
  mesh_dir:=<path>/mesh \
  class_names:=k2c \
  class_to_mesh:=k2c:K2
```

---

## Key Changes Summary

| Package | Files Changed | Purpose |
|---------|--------------|---------|
| `isaac_ros_common` | `cmake/modules/FindTENSORRT.cmake` | TRT 10.3 version parsing |
| `isaac_ros_dnn_inference` | 4 files | TRT 8.x → 10.x API migration |
| `isaac_ros_nitros` | 3 files | Detection3DArray dependency fix |
| `isaac_ros_pose_estimation` | 12 modified + 2 new | GXF fix, YOLO bridge, dynamic mesh |

For full details see [README_CHANGES.md](README_CHANGES.md).

---

## Hardware

- **Device**: NVIDIA Jetson AGX Orin (aarch64)
- **JetPack**: 6.x
- **CUDA**: 12.x
- **TensorRT**: 10.3
- **ROS 2**: Humble

## License

Patches apply to NVIDIA Isaac ROS packages which are licensed under the [Apache 2.0 License](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/LICENSE).  
New files (yolo_fp_bridge.py, yolo_fp_bridge.launch.py) are released under Apache 2.0.
