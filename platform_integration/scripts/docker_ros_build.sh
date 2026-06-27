#!/usr/bin/env bash
set -euo pipefail

# 构建本项目使用的 ROS Noetic Docker 镜像。
# 镜像里会安装 ROS、Python 依赖，并编译 common_msgs 消息包，后续算法、
# fake 车辆和 Redis-ROS Adapter 都在这个镜像对应的容器里运行。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-direct-control-ros:noetic}"

cd "$PROJECT_ROOT"
docker build -f platform_integration/docker/ros-noetic/Dockerfile -t "$IMAGE_NAME" .
