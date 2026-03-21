#!/bin/bash
# 在 Jetson (aarch64) 上安装 CV-CUDA (nvcv) 以解决 gxf_isaac_foundationpose 编译时
# 找不到 nvcv_types / cvcuda 的问题。版本与 Isaac ROS Docker x86_64 一致：v0.5.0-beta。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${SCRIPT_DIR}/.cvcuda_deps"
BASE_URL="https://github.com/CVCUDA/CV-CUDA/releases/download/v0.5.0-beta"
LIB_DEB="nvcv-lib-0.5.0_beta_DP-cuda12-aarch64-linux.deb"
DEV_DEB="nvcv-dev-0.5.0_beta_DP-cuda12-aarch64-linux.deb"

mkdir -p "$DEPS_DIR"
cd "$DEPS_DIR"

echo "=== 下载 nvcv .deb (aarch64, cuda12) ==="
for pkg in "$LIB_DEB" "$DEV_DEB"; do
  if [[ -f "$pkg" ]] && [[ -s "$pkg" ]]; then
    echo "已存在且非空: $pkg ($(du -h "$pkg" | cut -f1))"
  else
    echo "下载: $pkg"
    (wget -q --show-progress -O "$pkg" "${BASE_URL}/${pkg}" || curl -L -o "$pkg" "${BASE_URL}/${pkg}") || true
    if [[ ! -s "$pkg" ]]; then
      echo "请手动下载并放入 $DEPS_DIR："
      echo "  $BASE_URL/$LIB_DEB"
      echo "  $BASE_URL/$DEV_DEB"
      exit 1
    fi
  fi
done

echo "=== 安装 (需要 sudo) ==="
sudo dpkg -i "$LIB_DEB"
sudo dpkg -i "$DEV_DEB"

echo "=== 检查 CMake 能否找到 nvcv_types / cvcuda ==="
for name in nvcv_types cvcuda; do
  if dpkg -L nvcv-dev 2>/dev/null | grep -q "${name}"; then
    echo "nvcv-dev 提供 ${name} 相关文件"
  fi
done
if [[ -d /usr/lib/cmake/nvcv_types ]] || [[ -d /usr/lib/cmake/cvcuda ]]; then
  echo "CMake 配置目录已就绪"
else
  echo "若仍找不到，可尝试: export CMAKE_PREFIX_PATH=/usr:\$CMAKE_PREFIX_PATH"
fi

echo "=== 完成。请重新编译 ==="
echo "  source /opt/ros/humble/setup.bash"
echo "  conda activate foundationpose   # 如使用 conda"
echo "  cd $SCRIPT_DIR && colcon build --packages-up-to isaac_ros_foundationpose"
