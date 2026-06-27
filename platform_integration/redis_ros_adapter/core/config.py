"""Redis-ROS Adapter 的配置加载工具。

这个文件只负责把 `platform_integration/config/platform_real.yaml` 或 JSON 配置文件读进来，
并转换成适配器命令行参数能直接使用的扁平字段。它不连接 Redis，
也不启动 ROS，只处理配置文件结构和默认参数之间的映射关系。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    """读取 YAML/JSON 配置文件，并保证顶层结构是字典。

    平台实测配置默认使用 YAML，保留 JSON 支持是为了后续自动生成配置
    或外部系统直接吐 JSON 配置时不需要改调用方。
    """

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"adapter config must be an object: {path}")
    return data


def load_adapter_config(path_str: str) -> dict[str, Any]:
    """按路径加载适配器配置文件。

    空路径代表不使用配置文件，直接走命令行参数和环境变量默认值；
    非空路径必须存在，否则尽早报错，避免 Adapter 带着错误 Redis
    参数启动后表现为“平台离线”。
    """

    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"adapter config not found: {path}")
    return _load_yaml_or_json(path)


def flatten_adapter_config(data: dict[str, Any]) -> dict[str, Any]:
    """把分组配置转换成 `argparse` 使用的扁平参数名。

    YAML 里为了人读起来清楚，会分成 `platform`、`adapter`、`vehicle`、
    `ros` 和 `direct_control` 几段；运行时的参数解析器需要的是
    `redis_host`、`planner_mode` 这类扁平字段。本函数就是这两种结构
    之间的唯一映射表。
    """

    platform = data.get("platform") or {}
    adapter = data.get("adapter") or {}
    vehicle = data.get("vehicle") or {}
    ros = data.get("ros") or {}
    direct_control = data.get("direct_control") or {}
    result: dict[str, Any] = {}

    mapping = {
        "redis_host": platform.get("redis_host"),
        "redis_port": platform.get("redis_port"),
        "redis_db": platform.get("redis_db"),
        "redis_password": platform.get("redis_password"),
        "redis_timeout_s": platform.get("redis_timeout_s"),
        "device_id": platform.get("device_id"),
        "device_name": platform.get("device_name"),
        "control_channel": platform.get("control_channel"),
        "status_channel": platform.get("status_channel"),
        "data_channel_hint": platform.get("data_channel_hint"),
        "mode": adapter.get("mode"),
        "heartbeat_hz": adapter.get("heartbeat_hz"),
        "publish_hz": adapter.get("publish_hz"),
        "report_hz": adapter.get("report_hz"),
        "duration_s": adapter.get("duration_s"),
        "trajectory_advance_index": adapter.get("trajectory_advance_index"),
        "log_dir": adapter.get("log_dir"),
        "vehicle_id": vehicle.get("id"),
        "vehicle_name": vehicle.get("name"),
        "vehicle_id_from_device_id": vehicle.get("id_from_device_id"),
        "vehicle_length_cm": vehicle.get("length_cm"),
        "vehicle_width_cm": vehicle.get("width_cm"),
        "vehicle_height_cm": vehicle.get("height_cm"),
        "map_path": ros.get("map_path"),
        "cache_path": ros.get("adapter_cache_path"),
        "direct_cache_path": ros.get("direct_cache_path"),
        "rebuild_cache": ros.get("rebuild_cache"),
        "sensorgps_topic": ros.get("sensorgps_topic"),
        "actuator_topic": ros.get("actuator_topic"),
        "controlcmd_topic": ros.get("controlcmd_topic"),
        "trajectory_topic": ros.get("trajectory_topic"),
        "obu_topic": ros.get("obu_topic"),
        "planner_mode": direct_control.get("planner_mode"),
        "scene_algorithm_symbol": direct_control.get("scene_algorithm_symbol"),
        "scene_main_ptc_id": direct_control.get("scene_main_ptc_id"),
        "scene_exposed_ptc_id": direct_control.get("scene_exposed_ptc_id"),
        "plan_hz": direct_control.get("plan_hz"),
        "control_hz": direct_control.get("control_hz"),
        "max_speed_kph": direct_control.get("max_speed_kph"),
    }
    for key, value in mapping.items():
        if value is not None:
            result[key] = value
    return result
