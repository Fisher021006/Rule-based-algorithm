#!/usr/bin/env bash
set -euo pipefail

# 启动并进入本项目的 ROS Docker 容器。
# Mac 项目目录会挂载到容器内 /workspace/direct_control_bundle，因此你在 Mac
# 上改代码后，容器里能直接看到同一份文件。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-direct-control-ros:noetic}"
CONTAINER_NAME="${CONTAINER_NAME:-direct-control-ros}"

cd "$PROJECT_ROOT"
docker run -it --rm \
  --name "$CONTAINER_NAME" \
  -v "$PROJECT_ROOT:/workspace/direct_control_bundle" \
  -w /workspace/direct_control_bundle \
  "$IMAGE_NAME" \
  bash
