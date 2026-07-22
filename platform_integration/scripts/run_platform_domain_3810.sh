#!/usr/bin/env bash
set -euo pipefail

# 启动平台新注册的 3810 域控。它与原有 Adapter 共用代码和镜像，
# 但使用独立的设备 ID、Redis 通道、ROS topic、节点名和日志目录。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${DOMAIN_3810_CONFIG_PATH:-platform_integration/config/platform_domain_3810.yaml}"

exec "$SCRIPT_DIR/run_platform_adapter.sh"
