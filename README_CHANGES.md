# isaac_ros_foundationpose — 本地适配改动说明

> **平台**: NVIDIA Jetson (aarch64), JetPack / Ubuntu 20.04 / ROS 2 Humble  
> **TensorRT**: 10.3（非 Docker 原生部署，conda 环境 `foundationpose`）  
> **目标**: 在 Docker 之外的原生 conda 环境中成功编译、运行 `isaac_ros_foundationpose`，并与 YOLO26（`yolo_detect` 服务）联动实现在线 6D 位姿估计。

---

## 一、改动总览

| 子仓库 | 修改文件数 | 主要内容 |
|--------|-----------|---------|
| `isaac_ros_common` | 1 | FindTENSORRT.cmake 适配 TRT 10.3 |
| `isaac_ros_dnn_inference` | 4 | TRT 8.x → 10.3 API 全量迁移 |
| `isaac_ros_nitros` | 3 | Detection3DArray 依赖、编译修复 |
| `isaac_ros_pose_estimation` | 12 + 2新文件 | GXF 扩展修复、动态 mesh 加载、YOLO 桥接节点 |

新增独立文件：
- `/media/rykj/nvme/jetson/K2.obj` — 原点居中的 K2 扭锁网格（含 UV）
- `/tmp/refine_trt_engine.plan` — 用 TRT 10.3 重建的 refine 引擎
- `/tmp/score_trt_engine.plan`  — 用 TRT 10.3 重建的 score 引擎

---

## 二、各文件详细说明

### 2.1 `isaac_ros_common/cmake/modules/FindTENSORRT.cmake`

**问题**: TRT 10.3 版本号格式变更，`nvcaffe_parser` / `nvparsers` 库已不存在。

**修改**:
- 用 `file(STRINGS ...)` 从头文件精确解析 `NV_TENSORRT_MAJOR/MINOR/PATCH`，避免误解析
- 将 `nvcaffe_parser`、`nvparsers` 改为 `OPTIONAL` 组件，找不到不报错

---

### 2.2 `isaac_ros_dnn_inference` — TRT 10.3 API 迁移

#### `gxf_isaac_tensor_rt/CMakeLists.txt`
- 显式链接 `TENSORRT::nvinfer` 和 `TENSORRT::nvinfer_plugin`
- 解决运行时 `undefined symbol: initLibNvInferPlugins`

#### `gxf_isaac_tensor_rt/gxf/extensions/tensor_rt/tensor_rt_inference.hpp`
- `DeleteFunctor::operator()`: `ptr->destroy()` → `delete reinterpret_cast<T*>(ptr)`（TRT 10 删除了 `destroy()`）

#### `gxf_isaac_tensor_rt/gxf/extensions/tensor_rt/tensor_rt_inference.cpp`
TRT 8.x → 10.3 API 对照：

| 旧 API (TRT 8.x) | 新 API (TRT 10.x) |
|------------------|-------------------|
| `deserializeCudaEngine(data, size, nullptr)` | `deserializeCudaEngine(data, size)` |
| `getNbBindings()` | `getNbIOTensors()` |
| `getBindingName(i)` | `getIOTensorName(i)` |
| `getBindingFormatDesc(i)` | `getTensorFormatDesc(name)` |
| `getBindingIndex(name)` | `engine->getIOTensorName()` 遍历匹配 |
| `getBindingDataType(i)` | `getTensorDataType(name)` |
| `getBindingDimensions(i)` | `getTensorShape(name)` |
| `setBindingDimensions(i, dims)` | `setInputShape(name, dims)` |
| `setMaxWorkspaceSize(size)` | `setMemoryPoolLimit(kWORKSPACE, size)` |
| `buildEngineWithConfig(net, cfg)` | `buildSerializedNetwork(net, cfg)` + 反序列化 |
| `enqueueV2(buffers, stream, nullptr)` | `enqueueV3(stream)` + `setTensorAddress()` |

**额外增加**: 引擎反序列化失败时（TRT 版本不匹配）自动从 ONNX 重建并重试，避免手动删除旧引擎。

#### `isaac_ros_tensor_rt/src/tensor_rt_node.cpp`
- `std::max(shape.d[j], 1)` → `std::max(shape.d[j], static_cast<int64_t>(1))` 修复类型不匹配编译错误

---

### 2.3 `isaac_ros_nitros` — 编译修复

#### `isaac_ros_nitros/CMakeLists.txt` / `package.xml`
- 补充缺失依赖声明，解决本地编译时的链接警告

#### `isaac_ros_nitros_detection3_d_array_type/CMakeLists.txt`
- 修复 Detection3DArray 类型包依赖问题

---

### 2.4 `isaac_ros_pose_estimation` — 核心修复与新功能

#### `foundationpose_node.cpp` / `foundationpose_tracking_node.cpp`

**问题 1**: GXF 扩展包 `gxf_isaac_depth_image_proc` 找不到。  
**修复**: 在 `EXTENSIONS` 列表中补回 `{"gxf_isaac_sgm", "gxf/lib/libgxf_isaac_sgm.so"}`，`CameraMessageCompositor` 定义在此扩展中。

**问题 2**: `gxf_isaac_depth_image_proc` 包注册缺失。  
**修复**: 在两个节点的 `start()` 中通过 `SetParameterValue` 将 `mesh_file_path`、`texture_path` 等参数在运行时注入 GXF 图组件，不再依赖 YAML 内的硬编码路径。

#### `isaac_ros_foundationpose_tracking.launch.py`

**修改**:
- 改为**并发加载**三个节点（`selector_node`、`foundationpose_node`、`foundationpose_tracking_node`）到同一 `ComposableNodeContainer`
- `foundationpose_tracking_node` 的 `refine_model_file_path` / `refine_engine_file_path` 改为 `LaunchConfiguration`（之前误用了硬编码路径）
- 去除已失效的 `OnExecutionComplete` 顺序加载逻辑

#### `CMakeLists.txt`（isaac_ros_foundationpose）
新增安装规则：
```cmake
install(PROGRAMS
  scripts/yolo_fp_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)
```

#### 新文件: `scripts/yolo_fp_bridge.py`

YOLO → FoundationPose 桥接节点，功能：
1. 周期轮询 `yolo_detect` 服务（默认类名 `k2c`）
2. 根据 `class_to_mesh` 映射（默认 `k2c:K2`）在 `mesh_dir` 下查找对应 `.obj` 文件
3. 检测到新类别时，自动以正确 mesh 启动/重启 FoundationPose 子进程
4. 将 YOLO 的 RGB（bgr8→rgb8）、Depth（16UC1 mm → 32FC1 m）、Mask 和 CameraInfo 发布到 selector 输入话题
5. 调用 `/foundationpose/trigger_pose_estimation` 触发位姿估计
6. 订阅并打印 `/pose_estimation/output` 和 `/tracking/output` 的 6D 位姿结果

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mesh_dir` | `/media/rykj/nvme/jetson` | .obj 文件搜索目录 |
| `class_names` | `k2c` | YOLO 类名列表（逗号分隔） |
| `class_to_mesh` | `k2c:K2` | YOLO类名→OBJ文件名映射 |
| `camera_info_topic` | `/right_camera/color/camera_info` | 相机内参话题 |
| `poll_interval` | `1.0` | 轮询间隔（秒） |
| `refine_engine` | `/tmp/refine_trt_engine.plan` | Refine TRT 引擎路径 |
| `score_engine` | `/tmp/score_trt_engine.plan` | Score TRT 引擎路径 |
| `fp_wait_timeout` | `60.0` | 等待 FP 就绪超时（秒） |

#### 新文件: `launch/yolo_fp_bridge.launch.py`

启动桥接节点的 launch 文件，支持所有参数通过命令行覆盖。

---

## 三、K2.obj 网格文件

**原文件**: `/media/rykj/nvme/jetson/03.obj`  
**问题**: 坐标原点不在几何中心，且无 UV 纹理坐标（FoundationPose 渲染器要求 UV）

**处理**:
1. 计算顶点包围盒中心，平移顶点使包围盒中心对齐原点
   - 偏移量约: (+33.74 mm, +20.33 mm, −331.91 mm)
2. 使用球形 UV 映射为所有面添加 `vt` 坐标
3. 保存为 `/media/rykj/nvme/jetson/K2.obj`（23 MB）

---

## 四、TRT 引擎重建

由于原 `.plan` 文件由旧版 TensorRT 生成，需用当前 TRT 10.3 重建：

```bash
# Refine 引擎（约 56 秒）
/usr/src/tensorrt/bin/trtexec \
  --onnx=/tmp/refine_model.onnx \
  --saveEngine=/tmp/refine_trt_engine.plan \
  --fp16

# Score 引擎（约 50 秒）
/usr/src/tensorrt/bin/trtexec \
  --onnx=/tmp/score_model.onnx \
  --saveEngine=/tmp/score_trt_engine.plan \
  --fp16
```

---

## 五、运行方式

### 5.1 仅启动 FoundationPose（调试用）

```bash
source /opt/ros/humble/setup.bash
source /media/rykj/nvme/jetson/isaac_ros_foundationpose/install/setup.bash

ros2 launch isaac_ros_foundationpose isaac_ros_foundationpose_tracking.launch.py \
  mesh_file_path:=/media/rykj/nvme/jetson/K2.obj \
  texture_path:=/tmp/k2c_texture.png \
  refine_engine_file_path:=/tmp/refine_trt_engine.plan \
  score_engine_file_path:=/tmp/score_trt_engine.plan \
  refine_model_file_path:=/tmp/refine_model.onnx \
  score_model_file_path:=/tmp/score_model.onnx \
  launch_rviz:=False
```

### 5.2 启动完整 YOLO + FoundationPose 联动流程

```bash
# 终端 1: 启动 YOLO 服务（在 ga/code/ros2_ws 环境中）
source /opt/ros/humble/setup.bash
source /media/rykj/nvme/jetson/ga/code/ros2_ws/install/setup.bash
ros2 run yolo26_seg yolo_node  # 或对应启动命令

# 终端 2: 启动桥接节点（在 foundationpose conda 环境中）
conda activate foundationpose
source /opt/ros/humble/setup.bash
source /media/rykj/nvme/jetson/isaac_ros_foundationpose/install/setup.bash
source /media/rykj/nvme/jetson/ga/code/ros2_ws/install/setup.bash

ros2 launch isaac_ros_foundationpose yolo_fp_bridge.launch.py \
  mesh_dir:=/media/rykj/nvme/jetson \
  class_names:=k2c \
  class_to_mesh:=k2c:K2
```

---

## 六、编译方式

```bash
cd /media/rykj/nvme/jetson/isaac_ros_foundationpose
conda activate foundationpose
source /opt/ros/humble/setup.bash

# 全量编译
colcon build --packages-up-to isaac_ros_foundationpose \
  --cmake-args -DBUILD_TESTING=OFF

# 仅重编 foundationpose 本体
colcon build --packages-select isaac_ros_foundationpose \
  --cmake-args -DBUILD_TESTING=OFF

source install/setup.bash
```

---

## 七、已知问题 / 注意事项

1. **conda 与 ROS 2 Python 冲突**: 必须 `conda activate foundationpose`（Python 3.10 环境），不能用 base 环境（Python 3.13）
2. **TRT 引擎不可跨设备/版本迁移**: 引擎文件只能在生成该文件的 Jetson 上使用
3. **ExactSync 要求时间戳完全一致**: 桥接节点强制对齐三路图像时间戳
4. **FP 启动耗时较长**: FoundationPose 节点初始化（扩展加载 + GXF 图编译）约需 10-30 秒
5. **`/foundationpose/trigger_pose_estimation` 服务**: 该服务由 FoundationPose 节点提供，用于触发单次位姿估计；如未提供，需在桥接节点中注释相关调用
