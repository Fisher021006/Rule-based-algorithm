#!/usr/bin/env bash
set -euo pipefail

# 在 ROS Docker 容器内部执行本地 Redis-ROS smoke 测试。
# 流程是：启动 roscore、启动 Adapter、用 mock 平台下发 A-B 任务、
# 等待 Adapter 收到算法控制并向 Redis 上传轨迹。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

set +u
if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # 让当前 shell 能找到 ROS Noetic 自带命令和环境变量。
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi
if [[ -f /root/catkin_ws/devel/setup.bash ]]; then
  # 让当前 shell 能找到 Docker 镜像里编译好的 common_msgs 消息包。
  # shellcheck disable=SC1091
  source /root/catkin_ws/devel/setup.bash
fi
set -u

REDIS_HOST="${REDIS_HOST:-host.docker.internal}"
REDIS_PORT="${REDIS_PORT:-6379}"
CONTROL_CHANNEL="${CONTROL_CHANNEL:-control_mock_channel}"
STATUS_CHANNEL="${STATUS_CHANNEL:-STATUSResult}"
LOG_DIR="${LOG_DIR:-/tmp/virtual_redis_adapter_smoke}"

mkdir -p "$LOG_DIR"
rm -f \
  "$LOG_DIR/adapter.log" \
  "$LOG_DIR/direct_control_node.log" \
  "$LOG_DIR/mock_platform.log" \
  "$LOG_DIR/roscore.log"

ROSCORE_PID=""
ADAPTER_PID=""

cleanup() {
  if [[ -n "$ADAPTER_PID" ]] && kill -0 "$ADAPTER_PID" >/dev/null 2>&1; then
    kill "$ADAPTER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$ROSCORE_PID" ]] && kill -0 "$ROSCORE_PID" >/dev/null 2>&1; then
    kill "$ROSCORE_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if timeout 2s rostopic list >/dev/null 2>&1; then
  echo "[smoke] using existing roscore"
else
  echo "[smoke] starting roscore"
  roscore >"$LOG_DIR/roscore.log" 2>&1 &
  ROSCORE_PID="$!"
  for _ in $(seq 1 20); do
    if timeout 2s rostopic list >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi

echo "[smoke] starting virtual Redis-ROS adapter"
python3 platform_integration/scripts/run_virtual_redis_adapter.py \
  --redis-host "$REDIS_HOST" \
  --redis-port "$REDIS_PORT" \
  --control-channel "$CONTROL_CHANNEL" \
  --status-channel "$STATUS_CHANNEL" \
  --device-id 4 \
  --vehicle-id virtual_ego \
  --vehicle-name virtual_ego \
  --cache-path "$LOG_DIR/adapter_xodr_cache.pkl" \
  --direct-cache-path "$LOG_DIR/direct_xodr_cache.pkl" \
  --rebuild-cache \
  --planner-mode local_sampling \
  --max-speed-kph 12 \
  --log-dir "$LOG_DIR" \
  >"$LOG_DIR/adapter.log" 2>&1 &
ADAPTER_PID="$!"

sleep 2
if ! kill -0 "$ADAPTER_PID" >/dev/null 2>&1; then
  echo "[smoke] adapter exited early"
  tail -120 "$LOG_DIR/adapter.log" || true
  exit 1
fi

echo "[smoke] running mock platform"
python3 platform_integration/scripts/mock_platform_ab_task.py \
  --redis-host "$REDIS_HOST" \
  --redis-port "$REDIS_PORT" \
  --control-channel "$CONTROL_CHANNEL" \
  --status-channel "$STATUS_CHANNEL" \
  --cache-path "$LOG_DIR/mock_xodr_cache.pkl" \
  --rebuild-cache \
  --publish-background \
  --timeout-s 60 \
  >"$LOG_DIR/mock_platform.log" 2>&1

echo "[smoke] topics:"
rostopic list || true
echo "[smoke] mock tail:"
tail -80 "$LOG_DIR/mock_platform.log" || true
echo "[smoke] adapter tail:"
tail -80 "$LOG_DIR/adapter.log" || true
echo "[smoke] direct_control tail:"
LATEST_DIRECT_LOG="$(ls -t "$LOG_DIR"/direct_control_node_task_*.log 2>/dev/null | head -1 || true)"
if [[ -n "$LATEST_DIRECT_LOG" ]]; then
  tail -80 "$LATEST_DIRECT_LOG" || true
else
  tail -80 "$LOG_DIR/direct_control_node.log" || true
fi
echo "[smoke] PASS"
