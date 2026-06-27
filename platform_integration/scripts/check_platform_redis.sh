#!/usr/bin/env bash
set -euo pipefail

# 检查真实平台 Redis 配置是否可连接。
# 这个脚本不会启动算法，也不会订阅平台任务，只会进入已经运行的 ROS Docker
# 容器，读取 platform_integration/config/platform_real.yaml，然后用 Adapter 的同一套 Redis 连接逻辑
# 执行 PING。适合在排查内外网、密码、白名单问题时先单独运行。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROS_CONTAINER="${ROS_CONTAINER:-direct-control-ros}"
CONFIG_PATH="${CONFIG_PATH:-platform_integration/config/platform_real.yaml}"

if ! docker ps --format '{{.Names}}' | grep -qx "$ROS_CONTAINER"; then
  echo "[redis-check] missing running container: $ROS_CONTAINER"
  echo "[redis-check] start it first with: platform_integration/scripts/docker_ros_shell.sh"
  exit 1
fi

cd "$PROJECT_ROOT"
docker exec "$ROS_CONTAINER" bash -lc "
  cd /workspace/direct_control_bundle
  export PYTHONPATH=/workspace/direct_control_bundle/platform_integration:/workspace/direct_control_bundle:\${PYTHONPATH:-}
  python3 - <<'PY'
from redis_ros_adapter.core.config import flatten_adapter_config, load_adapter_config
from redis_ros_adapter.virtual.virtual_adapter import build_arg_parser, make_redis_client

config_path = '$CONFIG_PATH'
defaults = flatten_adapter_config(load_adapter_config(config_path))
args = build_arg_parser(defaults).parse_args([])
client = make_redis_client(args)
print('[redis-check] PING ok: %s:%s db=%s' % (args.redis_host, args.redis_port, args.redis_db), flush=True)
print('[redis-check] control_channel=%s status_channel=%s device_id=%s' % (args.control_channel, args.status_channel, args.device_id), flush=True)
client.close()
PY
"
