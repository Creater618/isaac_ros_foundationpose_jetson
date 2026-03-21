# Jetson 原生编译（无 Docker）

在 Jetson (aarch64) 上使用 conda 或系统 Python 直接 `colcon build` 时，`gxf_isaac_foundationpose` 会依赖 **CV-CUDA (nvcv)** 的 `nvcv_types` 和 `cvcuda`，CMake 会报错找不到这两个包。按下面步骤安装后再编译即可通过。

## 1. 安装 CV-CUDA (nvcv) v0.5.0-beta

与 Isaac ROS Docker x86_64 镜像内使用的版本一致，需要 **aarch64 + CUDA 12** 的以下两个 .deb：

- `nvcv-lib-0.5.0_beta_DP-cuda12-aarch64-linux.deb`（约 15MB）
- `nvcv-dev-0.5.0_beta_DP-cuda12-aarch64-linux.deb`（约 152KB）

**下载地址：**  
https://github.com/CVCUDA/CV-CUDA/releases/tag/v0.5.0-beta  
在页面 **Assets** 中找到上述两个文件并下载（网络不稳定时可使用浏览器或代理下载）。

**安装方式一：使用脚本（推荐）**

```bash
# 若已下载好 .deb，放到 .cvcuda_deps/ 目录下后再执行
cd /media/rykj/nvme/jetson/isaac_ros_foundationpose
chmod +x install_cvcuda_aarch64.sh
./install_cvcuda_aarch64.sh
```

脚本会尝试自动下载；若下载失败，会提示你手动下载并放入 `.cvcuda_deps/`，然后重新执行脚本。安装需要 `sudo`。

**安装方式二：手动安装**

```bash
cd /path/to/isaac_ros_foundationpose
sudo dpkg -i .cvcuda_deps/nvcv-lib-0.5.0_beta_DP-cuda12-aarch64-linux.deb
sudo dpkg -i .cvcuda_deps/nvcv-dev-0.5.0_beta_DP-cuda12-aarch64-linux.deb
```

## 2. 安装其他编译依赖

```bash
sudo apt-get update
sudo apt-get install -y ros-humble-magic-enum ros-humble-vision-msgs
```

- `magic_enum`：isaac_ros_nitros 通过 isaac_ros_gxf 间接链接，缺少会报 “Target links to magic_enum::magic_enum but the target was not found”。
- `vision_msgs`：isaac_ros_nitros_detection3_d_array_type 需要，缺少会报 “vision_msgs/msg/detection3_d_array.hpp: No such file or directory”。

## 3. 克隆 negotiated 与 isaac_ros_nitros（若尚未克隆）

```bash
cd /media/rykj/nvme/jetson/isaac_ros_foundationpose/src
git clone --depth 1 https://github.com/osrf/negotiated.git
git clone --depth 1 -b release-3.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git
cd isaac_ros_nitros && git lfs pull && cd ..
```

（isaac_ros_nitros 需 LFS 拉取 GXF 的 .so；negotiated 提供 `negotiated_publisher.hpp`，非 Humble 默认包。）

## 4. 编译

```bash
source /opt/ros/humble/setup.bash
conda activate foundationpose   # 若使用 conda
cd /media/rykj/nvme/jetson/isaac_ros_foundationpose
colcon build --packages-up-to isaac_ros_foundationpose --cmake-args -DBUILD_TESTING=OFF
```

（`-DBUILD_TESTING=OFF` 避免 osrf/negotiated 因缺少测试文件而编译失败。）

若仍有 “Could not find nvcv_types” 或 “Could not find cvcuda”，可先确认安装位置：

```bash
dpkg -L nvcv-dev | grep -E 'cmake|\.cmake'
```

并将该前缀加入 `CMAKE_PREFIX_PATH` 后再执行 `colcon build`。

## 5. 可选：使用 Docker 编译

若本机网络无法稳定下载上述 .deb，可使用 Isaac ROS 官方 Docker 镜像在容器内编译（镜像内已包含 nvcv 等依赖），参见仓库内 Docker 相关文档或 `complete_setup.log` 中的镜像构建流程。
