#!/usr/bin/env bash
set -euo pipefail

# 只验证 ROS 算法链路的 smoke 测试，不连接 Redis。
# 流程是：启动 roscore、启动 fake 车辆发布 /sensorgps 和 /actuator、
# 启动 direct_control_node.py，然后等待 /controlcmd 出现非零速度指令。

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

START_X="${START_X:--418.0}"
START_Y="${START_Y:-59.0}"
GOAL_X="${GOAL_X:--429.0}"
GOAL_Y="${GOAL_Y:-129.0}"
PLANNER_MODE="${PLANNER_MODE:-local_sampling}"
PLAN_HZ="${PLAN_HZ:-10}"
CONTROL_HZ="${CONTROL_HZ:-20}"
FAKE_HZ="${FAKE_HZ:-20}"
LOG_DIR="${LOG_DIR:-/tmp/direct_control_fake_smoke}"
FAKE_CACHE_PATH="${FAKE_CACHE_PATH:-$LOG_DIR/fake_xodr_cache.pkl}"
DIRECT_CACHE_PATH="${DIRECT_CACHE_PATH:-$LOG_DIR/direct_xodr_cache.pkl}"

mkdir -p "$LOG_DIR"

ROSCORE_PID=""
FAKE_PID=""
DIRECT_PID=""

cleanup() {
  if [[ -n "$DIRECT_PID" ]] && kill -0 "$DIRECT_PID" >/dev/null 2>&1; then
    kill "$DIRECT_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$FAKE_PID" ]] && kill -0 "$FAKE_PID" >/dev/null 2>&1; then
    kill "$FAKE_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$ROSCORE_PID" ]] && kill -0 "$ROSCORE_PID" >/dev/null 2>&1; then
    kill "$ROSCORE_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if timeout 2s rostopic list >/dev/null 2>&1; then
  echo "[smoke] using existing roscore at ${ROS_MASTER_URI:-http://127.0.0.1:11311}"
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

echo "[smoke] starting fake vehicle"
python3 platform_integration/scripts/fake_vehicle_ros_node.py \
  --start-xy "$START_X" "$START_Y" \
  --cache-path "$FAKE_CACHE_PATH" \
  --rebuild-cache \
  --hz "$FAKE_HZ" \
  --log-every 20 \
  >"$LOG_DIR/fake_vehicle.log" 2>&1 &
FAKE_PID="$!"

sleep 1
if ! kill -0 "$FAKE_PID" >/dev/null 2>&1; then
  echo "[smoke] fake vehicle exited early"
  tail -80 "$LOG_DIR/fake_vehicle.log" || true
  exit 1
fi

echo "[smoke] starting direct_control_node.py"
python3 direct_control_node.py \
  --start-xy "$START_X" "$START_Y" \
  --goal-xy "$GOAL_X" "$GOAL_Y" \
  --cache-path "$DIRECT_CACHE_PATH" \
  --rebuild-cache \
  --planner-mode "$PLANNER_MODE" \
  --plan-hz "$PLAN_HZ" \
  --control-hz "$CONTROL_HZ" \
  --max-speed-kph 12 \
  >"$LOG_DIR/direct_control.log" 2>&1 &
DIRECT_PID="$!"

sleep 2
if ! kill -0 "$DIRECT_PID" >/dev/null 2>&1; then
  echo "[smoke] direct_control_node.py exited early"
  tail -120 "$LOG_DIR/direct_control.log" || true
  exit 1
fi

echo "[smoke] waiting for non-zero /controlcmd"
python3 - <<'PY'
import sys
import time

import rospy
from common_msgs.msg import controlcmd

rospy.init_node("wait_for_direct_control_command", anonymous=True)
deadline = time.time() + 35.0
last = None
while time.time() < deadline and not rospy.is_shutdown():
    try:
        msg = rospy.wait_for_message("/controlcmd", controlcmd, timeout=1.0)
    except rospy.ROSException:
        continue
    last = msg
    print(
        "[smoke] /controlcmd acceleration_kph={:.3f} wheel_angle_deg={:.3f} brkreq={}".format(
            float(msg.acceleration),
            float(msg.wheel_angle),
            int(msg.brkreq),
        )
    )
    if float(msg.acceleration) > 0.1 and int(msg.brkreq) == 0:
        sys.exit(0)

if last is None:
    print("[smoke] no /controlcmd received before timeout", file=sys.stderr)
else:
    print("[smoke] /controlcmd stayed at hold/brake before timeout", file=sys.stderr)
sys.exit(1)
PY

echo "[smoke] topics:"
rostopic list
echo "[smoke] fake vehicle log tail:"
tail -20 "$LOG_DIR/fake_vehicle.log" || true
echo "[smoke] direct control log tail:"
tail -40 "$LOG_DIR/direct_control.log" || true
echo "[smoke] PASS"
