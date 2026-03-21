#!/bin/bash
# =============================================================================
# Isaac ROS FoundationPose — Jetson (aarch64) 原生编译一键脚本
#
# 适用环境：
#   - Jetson (aarch64), JetPack 6.0, Ubuntu 22.04
#   - ROS 2 Humble 已安装（/opt/ros/humble）
#   - 可选：conda 环境 foundationpose（用于 catkin_pkg / lark）
#   - 工作目录：/media/rykj/nvme/jetson/isaac_ros_foundationpose
#
# 用法：
#   chmod +x setup_and_build.sh
#   ./setup_and_build.sh
#
# 脚本会自动完成：
#   1. 安装 conda 依赖（catkin_pkg, lark）
#   2. 安装 CV-CUDA nvcv .deb（nvcv_types / cvcuda，供 GXF 编译）
#   3. 安装 ROS 系统依赖（magic-enum, vision-msgs）
#   4. 克隆并拉取 isaac_ros_nitros（含 GXF .so via Git LFS）
#   5. 克隆 osrf/negotiated
#   6. 对源码打必要的补丁（magic_enum / vision_msgs / 依赖声明）
#   7. 执行 colcon build --packages-up-to isaac_ros_foundationpose
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
DEPS_DIR="${SCRIPT_DIR}/.cvcuda_deps"
ROS_SETUP="/opt/ros/humble/setup.bash"
CONDA_ENV="foundationpose"

# 颜色输出
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# =============================================================================
# 1. 检查前提
# =============================================================================
info "=== 1. 检查前提条件 ==="

[[ -d "${SRC_DIR}/isaac_ros_pose_estimation" ]] || \
  error "找不到 src/isaac_ros_pose_estimation，请确认 isaac_ros_pose_estimation 已克隆到 ${SRC_DIR}"

[[ -d "${SRC_DIR}/isaac_ros_common" ]] || \
  error "找不到 src/isaac_ros_common，请确认 isaac_ros_common 已克隆到 ${SRC_DIR}"

[[ -f "${ROS_SETUP}" ]] || error "找不到 ROS Humble: ${ROS_SETUP}"

source "${ROS_SETUP}"
info "ROS Humble 已加载"

# =============================================================================
# 2. conda 环境依赖（catkin_pkg, lark）
# =============================================================================
info "=== 2. 安装 conda 依赖 (catkin_pkg, lark) ==="

if command -v conda &>/dev/null; then
  CONDA_ACTIVE=$(conda info --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active_prefix_name',''))" 2>/dev/null || echo "")
  if [[ "${CONDA_ACTIVE}" == "${CONDA_ENV}" ]]; then
    pip install catkin_pkg lark 2>&1 | tail -5
    info "catkin_pkg / lark 已在 conda 环境 ${CONDA_ENV} 安装"
  else
    warn "当前 conda 环境不是 ${CONDA_ENV}，跳过 pip 安装。请手动执行：conda activate ${CONDA_ENV} && pip install catkin_pkg lark"
  fi
else
  warn "未检测到 conda，跳过 catkin_pkg / lark 安装（若报 ImportError 请手动安装）"
fi

# =============================================================================
# 3. 安装 CV-CUDA .deb（nvcv_types / cvcuda，供 gxf_isaac_foundationpose 编译）
# =============================================================================
info "=== 3. 安装 CV-CUDA nvcv .deb ==="

NVCV_BASE_URL="https://github.com/CVCUDA/CV-CUDA/releases/download/v0.5.0-beta"
NVCV_LIB_DEB="nvcv-lib-0.5.0_beta_DP-cuda12-aarch64-linux.deb"
NVCV_DEV_DEB="nvcv-dev-0.5.0_beta_DP-cuda12-aarch64-linux.deb"

if dpkg -s libnvcv0 &>/dev/null && dpkg -s nvcv0-dev &>/dev/null; then
  info "libnvcv0 / nvcv0-dev 已安装，跳过"
else
  mkdir -p "${DEPS_DIR}"
  cd "${DEPS_DIR}"

  for DEB in "${NVCV_LIB_DEB}" "${NVCV_DEV_DEB}"; do
    if [[ -f "${DEB}" ]] && [[ -s "${DEB}" ]]; then
      info "已存在: ${DEB}"
    else
      info "下载: ${DEB}"
      wget -q --show-progress -O "${DEB}" "${NVCV_BASE_URL}/${DEB}" || \
        curl -L -o "${DEB}" "${NVCV_BASE_URL}/${DEB}" || \
        error "下载失败：${DEB}。请手动下载 ${NVCV_BASE_URL}/${DEB} 并放入 ${DEPS_DIR}"
      [[ -s "${DEB}" ]] || error "文件为空，下载可能失败：${DEB}"
    fi
  done

  info "安装 nvcv .deb（需要 sudo）"
  sudo dpkg -i "${NVCV_LIB_DEB}"
  sudo dpkg -i "${NVCV_DEV_DEB}"
  cd "${SCRIPT_DIR}"
fi

# 验证 CMake 配置目录
if [[ -d /usr/local/lib/aarch64-linux-gnu/cmake/nvcv_types ]]; then
  info "nvcv_types CMake 配置已就绪: /usr/local/lib/aarch64-linux-gnu/cmake/nvcv_types"
else
  warn "未找到 nvcv_types CMake 配置，可能影响编译，请检查安装"
fi

# =============================================================================
# 4. 安装 ROS 系统依赖
# =============================================================================
info "=== 4. 安装 ROS apt 依赖 ==="

PKGS=()
dpkg -s ros-humble-magic-enum &>/dev/null || PKGS+=(ros-humble-magic-enum)
dpkg -s ros-humble-vision-msgs  &>/dev/null || PKGS+=(ros-humble-vision-msgs)

if [[ ${#PKGS[@]} -gt 0 ]]; then
  info "需要安装: ${PKGS[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y "${PKGS[@]}"
else
  info "ros-humble-magic-enum / ros-humble-vision-msgs 均已安装，跳过"
fi

# =============================================================================
# 5. 克隆 isaac_ros_nitros（含 GXF .so via Git LFS）
# =============================================================================
info "=== 5. 克隆 isaac_ros_nitros ==="

NITROS_DIR="${SRC_DIR}/isaac_ros_nitros"

if [[ -d "${NITROS_DIR}/.git" ]]; then
  info "isaac_ros_nitros 已存在，跳过克隆"
else
  cd "${SRC_DIR}"
  git clone --depth 1 -b release-3.1 \
    https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git \
    isaac_ros_nitros
fi

info "Git LFS pull（拉取 GXF .so 等大文件，约 419 MB，首次较慢）"
cd "${NITROS_DIR}"
if [[ -f "isaac_ros_gxf/gxf/core/lib/gxf_jetpack60/core/libgxf_core.so" ]] && \
   [[ -s "isaac_ros_gxf/gxf/core/lib/gxf_jetpack60/core/libgxf_core.so" ]]; then
  info "libgxf_core.so 已存在，跳过 git lfs pull"
else
  git lfs pull || warn "git lfs pull 报告错误（通常可忽略，只要文件存在即可）"
  [[ -s "isaac_ros_gxf/gxf/core/lib/gxf_jetpack60/core/libgxf_core.so" ]] || \
    error "git lfs pull 后 libgxf_core.so 仍不存在，请检查 git-lfs 是否安装及网络"
fi
cd "${SCRIPT_DIR}"

# =============================================================================
# 6. 克隆 osrf/negotiated
# =============================================================================
info "=== 6. 克隆 osrf/negotiated ==="

NEGO_DIR="${SRC_DIR}/negotiated"

if [[ -d "${NEGO_DIR}/.git" ]]; then
  info "negotiated 已存在，跳过克隆"
else
  cd "${SRC_DIR}"
  git clone --depth 1 https://github.com/osrf/negotiated.git
fi
cd "${SCRIPT_DIR}"

# =============================================================================
# 7. 打补丁（仅在未打过的情况下）
# =============================================================================
info "=== 7. 应用源码补丁 ==="

# --- 7.1 isaac_ros_nitros: 补 magic_enum find_package ---
NITROS_CMAKE="${NITROS_DIR}/isaac_ros_nitros/CMakeLists.txt"
if ! grep -q "find_package(magic_enum" "${NITROS_CMAKE}"; then
  info "补丁: isaac_ros_nitros CMakeLists.txt 增加 find_package(magic_enum)"
  sed -i 's/find_package(Eigen3 3.3 REQUIRED NO_MODULE)/find_package(magic_enum REQUIRED)\nfind_package(Eigen3 3.3 REQUIRED NO_MODULE)/' \
    "${NITROS_CMAKE}"
else
  info "已打补丁（magic_enum），跳过"
fi

# --- 7.2 isaac_ros_nitros: package.xml 增加 magic_enum 依赖 ---
NITROS_PKG="${NITROS_DIR}/isaac_ros_nitros/package.xml"
if ! grep -q "<depend>magic_enum</depend>" "${NITROS_PKG}"; then
  info "补丁: isaac_ros_nitros package.xml 增加 magic_enum depend"
  sed -i 's|<depend>isaac_ros_gxf</depend>|<depend>isaac_ros_gxf</depend>\n  <depend>magic_enum</depend>|' \
    "${NITROS_PKG}"
else
  info "已打补丁（nitros package.xml magic_enum），跳过"
fi

# --- 7.3 isaac_ros_nitros_detection3_d_array_type: vision_msgs include ---
DETECT3D_CMAKE="${NITROS_DIR}/isaac_ros_nitros_type/isaac_ros_nitros_detection3_d_array_type/CMakeLists.txt"
if ! grep -q "find_package(vision_msgs" "${DETECT3D_CMAKE}"; then
  info "补丁: isaac_ros_nitros_detection3_d_array_type CMakeLists.txt 增加 vision_msgs"
  sed -i 's/# Dependencies/# Dependencies\nfind_package(vision_msgs REQUIRED)/' \
    "${DETECT3D_CMAKE}"
  # 在 target_include_directories 里加上 vision_msgs_INCLUDE_DIRS
  sed -i 's/target_include_directories(${PROJECT_NAME} PUBLIC include\/detection3_d_array_message)/target_include_directories(${PROJECT_NAME} PUBLIC\n  include\/detection3_d_array_message\n  ${vision_msgs_INCLUDE_DIRS}\n)/' \
    "${DETECT3D_CMAKE}"
else
  info "已打补丁（detection3_d_array vision_msgs），跳过"
fi

# --- 7.4 isaac_ros_foundationpose: package.xml 增加构建依赖 ---
FP_PKG="${SRC_DIR}/isaac_ros_pose_estimation/isaac_ros_foundationpose/package.xml"
if ! grep -q "<depend>isaac_ros_nitros_image_type</depend>" "${FP_PKG}"; then
  info "补丁: isaac_ros_foundationpose package.xml 增加 image_type / tensor_list_type 依赖"
  sed -i 's|<depend>isaac_ros_nitros_detection3_d_array_type</depend>|<depend>isaac_ros_nitros_detection3_d_array_type</depend>\n  <depend>isaac_ros_nitros_image_type</depend>\n  <depend>isaac_ros_nitros_tensor_list_type</depend>|' \
    "${FP_PKG}"
else
  info "已打补丁（foundationpose image_type），跳过"
fi

if ! grep -q "^  <depend>isaac_ros_managed_nitros</depend>" "${FP_PKG}"; then
  info "补丁: isaac_ros_foundationpose package.xml 将 managed_nitros 改为 depend"
  sed -i 's|<exec_depend>isaac_ros_managed_nitros</exec_depend>|<depend>isaac_ros_managed_nitros</depend>|' \
    "${FP_PKG}"
else
  info "已打补丁（foundationpose managed_nitros），跳过"
fi

# =============================================================================
# 8. 执行 colcon build
# =============================================================================
info "=== 8. 开始编译 isaac_ros_foundationpose ==="
info "（首次编译约需 5~10 分钟，包含 CUDA 编译）"

cd "${SCRIPT_DIR}"
source "${ROS_SETUP}"
# 若已有 conda 激活，install/setup.bash 可叠加上一次的安装前缀
[[ -f "${SCRIPT_DIR}/install/setup.bash" ]] && source "${SCRIPT_DIR}/install/setup.bash" || true

colcon build \
  --packages-up-to isaac_ros_foundationpose \
  --cmake-args -DBUILD_TESTING=OFF \
  2>&1

BUILD_EXIT=$?

if [[ ${BUILD_EXIT} -eq 0 ]]; then
  echo ""
  info "=============================================="
  info "  编译成功！"
  info "=============================================="
  info "使用前请 source 工作空间："
  info "  source ${SCRIPT_DIR}/install/setup.bash"
else
  echo ""
  error "编译失败（exit code ${BUILD_EXIT}），请查看上方报错信息"
fi
