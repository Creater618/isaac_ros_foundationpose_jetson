#!/bin/bash
# Apply all Jetson-native patches to the upstream Isaac ROS workspace.
# Usage: ./apply_patches.sh <path_to_ros_ws_src>
#   e.g: ./apply_patches.sh ~/isaac_ros_fp_ws/src

set -e

SRC="${1:-$(pwd)/../src}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/patches"

if [ ! -d "$SRC" ]; then
  echo "ERROR: Source directory not found: $SRC"
  echo "Usage: $0 <path_to_ros_ws_src>"
  exit 1
fi

echo "=== Applying Jetson-native patches to: $SRC ==="

apply_patch() {
  local repo="$1"
  local patch="$2"
  local target="$SRC/$repo"

  if [ ! -d "$target" ]; then
    echo "WARNING: $target not found, skipping $patch"
    return
  fi

  echo ""
  echo "--- Applying $patch to $repo ---"
  git -C "$target" apply --check "$PATCHES_DIR/$patch" 2>/dev/null \
    && git -C "$target" apply "$PATCHES_DIR/$patch" \
    && echo "    OK" \
    || echo "    SKIPPED (already applied or conflict)"
}

apply_patch "isaac_ros_common"        "01_isaac_ros_common.patch"
apply_patch "isaac_ros_dnn_inference" "02_isaac_ros_dnn_inference.patch"
apply_patch "isaac_ros_nitros"        "03_isaac_ros_nitros.patch"
apply_patch "isaac_ros_pose_estimation" "04_isaac_ros_pose_estimation.patch"

echo ""
echo "--- Copying new files ---"

FP_DIR="$SRC/isaac_ros_pose_estimation/isaac_ros_foundationpose"

if [ -d "$FP_DIR" ]; then
  cp "$SCRIPT_DIR/new_files/launch/yolo_fp_bridge.launch.py" "$FP_DIR/launch/"
  mkdir -p "$FP_DIR/scripts"
  cp "$SCRIPT_DIR/new_files/scripts/yolo_fp_bridge.py" "$FP_DIR/scripts/"
  chmod +x "$FP_DIR/scripts/yolo_fp_bridge.py"
  echo "    Copied yolo_fp_bridge.launch.py and yolo_fp_bridge.py"
else
  echo "    WARNING: $FP_DIR not found, skipping new file copy"
fi

echo ""
echo "=== All patches applied successfully! ==="
echo ""
echo "Next steps:"
echo "  1. Install CV-CUDA (see BUILD_JETSON_NATIVE.md)"
echo "  2. Build TRT engines with trtexec"
echo "  3. colcon build --packages-up-to isaac_ros_foundationpose --cmake-args -DBUILD_TESTING=OFF"
