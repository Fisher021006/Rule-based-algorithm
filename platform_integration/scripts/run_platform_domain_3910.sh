#!/usr/bin/env bash
set -euo pipefail

# 启动平台注册的 3910 域控（lxy-伴行车2）。它与其他域控共用代码和镜像，
# 但使用独立的设备 ID、Redis 通道、ROS topic、节点名和日志目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${DOMAIN_3910_CONFIG_PATH:-platform_integration/config/platform_domain_3910.yaml}"
exec "$SCRIPT_DIR/run_platform_adapter.sh"
