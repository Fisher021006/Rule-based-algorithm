#!/usr/bin/env bash
set -euo pipefail

# 从 Mac 侧启动完整的本地 Redis-ROS smoke 测试。
# 脚本会临时拉起一个 Redis 容器，再让已经运行的 ROS 容器执行内部测试脚本。
# 该测试只验证我们构造的本地闭环，不连接真实平台。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REDIS_CONTAINER="${REDIS_CONTAINER:-direct-control-redis-smoke}"
ROS_CONTAINER="${ROS_CONTAINER:-direct-control-ros}"
REDIS_PORT="${REDIS_PORT:-6379}"

cleanup() {
  docker rm -f "$REDIS_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if ! docker ps --format '{{.Names}}' | grep -qx "$ROS_CONTAINER"; then
  echo "[smoke] missing running container: $ROS_CONTAINER"
  echo "[smoke] start it first with: platform_integration/scripts/docker_ros_shell.sh"
  exit 1
fi

docker rm -f "$REDIS_CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$REDIS_CONTAINER" -p "$REDIS_PORT:6379" redis:7-alpine >/dev/null

cd "$PROJECT_ROOT"
docker exec "$ROS_CONTAINER" bash -lc \
  "cd /workspace/direct_control_bundle && REDIS_HOST=host.docker.internal REDIS_PORT=$REDIS_PORT platform_integration/scripts/run_virtual_redis_adapter_smoke_inside.sh"
