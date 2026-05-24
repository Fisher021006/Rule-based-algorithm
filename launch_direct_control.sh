#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "$SCRIPT_DIR/runtime.env" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/runtime.env"
fi

if [[ -n "${ROS_SETUP_BASH:-}" ]]; then
  # shellcheck disable=SC1090
  source "$ROS_SETUP_BASH"
fi

if [[ -n "${CATKIN_SETUP_BASH:-}" ]]; then
  # shellcheck disable=SC1090
  source "$CATKIN_SETUP_BASH"
fi

source "$SCRIPT_DIR/.venv/bin/activate"

python "$SCRIPT_DIR/direct_control_node.py" "$@"
