#!/usr/bin/env bash
set -euo pipefail

# 真实平台联调用启动脚本。
# 它要求 direct-control-ros 容器已经在运行，然后进入容器、加载 ROS 环境、
# 确认 roscore 可用，最后按 platform_integration/config/platform_real.yaml 启动 Redis-ROS Adapter。
# 平台下发任务后，Adapter 会启动算法并把虚拟主车轨迹回传平台。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROS_CONTAINER="${ROS_CONTAINER:-direct-control-ros}"
CONFIG_PATH="${CONFIG_PATH:-platform_integration/config/platform_real.yaml}"

if ! docker ps --format '{{.Names}}' | grep -qx "$ROS_CONTAINER"; then
  echo "[platform] missing running container: $ROS_CONTAINER"
  echo "[platform] start it first with: platform_integration/scripts/docker_ros_shell.sh"
  exit 1
fi

cd "$PROJECT_ROOT"
docker exec "$ROS_CONTAINER" bash -lc "
  set -euo pipefail
  cd /workspace/direct_control_bundle
  set +u
  source /opt/ros/noetic/setup.bash
  source /root/catkin_ws/devel/setup.bash
  set -u
  if timeout 2s rostopic list >/dev/null 2>&1; then
    echo '[platform] using existing roscore'
  else
    echo '[platform] starting roscore'
    mkdir -p /tmp/platform_redis_ros_adapter
    roscore >/tmp/platform_redis_ros_adapter/roscore.log 2>&1 &
    for _ in \$(seq 1 20); do
      if timeout 2s rostopic list >/dev/null 2>&1; then
        break
      fi
      sleep 0.5
    done
  fi
  python3 platform_integration/scripts/run_virtual_redis_adapter.py --config '$CONFIG_PATH'
"
