"""根据统一 JSON 配置启动离线仿真或车端直接控制。"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "launch.json"

COMMON_TOP_LEVEL_KEYS = {
    "version",
    "launch_mode",
    "dry_run",
}

OFFLINE_TOP_LEVEL_KEYS = COMMON_TOP_LEVEL_KEYS | {
    "runtime",
    "primary_vehicle_id",
    "stop_on_terminal_vehicle_ids",
    "stop_on_terminal_mode",
    "terminal_scene_statuses",
    "vehicles",
}

DIRECT_TOP_LEVEL_KEYS = COMMON_TOP_LEVEL_KEYS | {
    "confirm_real_vehicle",
    "runtime",
    "ros",
    "vehicle",
    "scene",
}

OFFLINE_RUNTIME_KEYS = {
    "map_cache",
    "dt",
    "max_steps",
    "advance_index",
    "arrive_dist",
    "arrive_speed",
    "ref_lane_count",
    "log_every",
    "skip_gif",
    "output_dir",
    "output_prefix",
    "show",
    "live_show",
    "live_refresh_every",
}

DIRECT_RUNTIME_KEYS = {
    "map_path",
    "cache_path",
    "rebuild_cache",
    "start_xy",
    "start_latlon",
    "use_current_start",
    "goal_xy",
    "goal_latlon",
    "planner_mode",
    "plan_hz",
    "control_hz",
    "arrive_dist",
    "arrive_speed",
    "obu_track_timeout_ms",
    "node_name",
    "scene_trajectory_log_path",
    "swap_latlon",
    "speed_scale",
}

DIRECT_ROS_KEYS = {
    "sensorgps_topic",
    "gps_topic",
    "actuator_topic",
    "obu_topic",
    "controlcmd_topic",
    "control_topic",
    "trajectory_topic",
}

DIRECT_VEHICLE_KEYS = {
    "self_ptc_id",
    "length_mm",
    "width_mm",
    "wheelbase_mm",
    "max_speed_kph",
    "max_acc",
    "min_acc",
}

DIRECT_SCENE_KEYS = {
    "algorithm",
    "main_ptc_id",
    "main_start_xy",
    "main_start_latlon",
    "main_goal_xy",
    "main_goal_latlon",
    "environment",
}


def build_arg_parser() -> argparse.ArgumentParser:
    """构造统一启动器参数。"""

    parser = argparse.ArgumentParser(description="根据 JSON 启动直接控制或离线仿真")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"启动配置路径，默认 {DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验配置并打印实际命令，不启动子进程",
    )
    return parser


def _load_json_config(path: Path) -> dict[str, Any]:
    """读取统一 JSON 配置并校验顶层类型。"""

    if not path.exists():
        raise FileNotFoundError(f"启动配置不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("启动配置顶层必须是 JSON object")
    version = payload.get("version", 1)
    if type(version) is not int or version != 1:
        raise ValueError(f"不支持的启动配置版本: {version}")
    launch_mode = str(payload.get("launch_mode", "") or "").strip().lower()
    if launch_mode not in {"offline_simulation", "direct_control"}:
        raise ValueError("launch_mode 必须是 offline_simulation 或 direct_control")
    return payload


def _object_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """读取可选字典配置段。"""

    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 JSON object")
    return dict(value)


def _reject_unknown_keys(section_name: str, section: dict[str, Any], allowed: set[str]) -> None:
    """拒绝未支持字段，避免拼写错误被静默忽略。"""

    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"{section_name} 存在未支持字段: {unknown}")


def _bool_value(value: Any, field_name: str) -> bool:
    """读取 JSON 布尔值，拒绝容易产生误解的字符串。"""

    if not isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是 JSON 布尔值 true 或 false")
    return value


def _resolve_input_path(raw_value: Any, config_path: Path, field_name: str) -> Path:
    """解析必须存在的输入路径，优先按项目根目录解析。"""

    value = str(raw_value or "").strip()
    if not value:
        raise ValueError(f"{field_name} 不能为空")
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        project_candidate = (PROJECT_ROOT / path).resolve()
        config_candidate = (config_path.parent / path).resolve()
        resolved = project_candidate if project_candidate.exists() else config_candidate
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} 不存在: {resolved}")
    return resolved


def _resolve_optional_path(raw_value: Any, config_path: Path) -> Path | None:
    """解析可选路径；目标不必预先存在。"""

    value = str(raw_value or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _pair(value: Any, field_name: str) -> list[float] | None:
    """校验二维坐标。"""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} 必须是 [value1, value2]")
    return [float(value[0]), float(value[1])]


def _append_value(command: list[str], flag: str, value: Any) -> None:
    """将非空单值转成命令行参数。"""

    if value is not None and value != "":
        command.extend([flag, str(value)])


def _append_pair(command: list[str], flag: str, value: Any, field_name: str) -> None:
    """将可选二维坐标转成命令行参数。"""

    parsed = _pair(value, field_name)
    if parsed is not None:
        command.extend([flag, str(parsed[0]), str(parsed[1])])


def _append_true_flag(
    command: list[str],
    flag: str,
    value: Any,
    field_name: str,
) -> None:
    """在配置为真时追加开关参数。"""

    if _bool_value(value, field_name):
        command.append(flag)


def _build_offline_command(config: dict[str, Any], config_path: Path) -> tuple[list[str], dict[str, str]]:
    """校验多车配置并构造离线仿真命令。"""

    from offline_sim.multi_runner import load_multi_vehicle_config

    load_multi_vehicle_config(config_path)
    runtime = _object_section(config, "runtime")
    _reject_unknown_keys("runtime", runtime, OFFLINE_RUNTIME_KEYS)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "simulate_scene_offline.py"),
        "--planner-mode",
        "scene_target",
        "--multi-vehicle-config",
        str(config_path),
    ]
    if "map_cache" in runtime:
        map_cache = _resolve_input_path(runtime["map_cache"], config_path, "runtime.map_cache")
        _append_value(command, "--map-cache", map_cache)
    scalar_flags = {
        "dt": "--dt",
        "max_steps": "--max-steps",
        "advance_index": "--advance-index",
        "arrive_dist": "--arrive-dist",
        "arrive_speed": "--arrive-speed",
        "ref_lane_count": "--ref-lane-count",
        "log_every": "--log-every",
        "output_prefix": "--output-prefix",
        "live_refresh_every": "--live-refresh-every",
    }
    for key, flag in scalar_flags.items():
        if key in runtime:
            _append_value(command, flag, runtime[key])
    if "output_dir" in runtime:
        output_dir = _resolve_optional_path(runtime["output_dir"], config_path)
        _append_value(command, "--output-dir", output_dir)
    for key, flag in {
        "skip_gif": "--skip-gif",
        "show": "--show",
        "live_show": "--live-show",
    }.items():
        _append_true_flag(command, flag, runtime.get(key, False), f"runtime.{key}")
    return command, os.environ.copy()


def _validate_direct_coordinates(runtime: dict[str, Any]) -> None:
    """校验直控起终点配置的互斥与完整性。"""

    start_xy = _pair(runtime.get("start_xy"), "runtime.start_xy")
    start_latlon = _pair(runtime.get("start_latlon"), "runtime.start_latlon")
    use_current_start = _bool_value(
        runtime.get("use_current_start", False),
        "runtime.use_current_start",
    )
    start_count = int(start_xy is not None) + int(start_latlon is not None) + int(use_current_start)
    if start_count != 1:
        raise ValueError(
            "直控起点必须且只能配置 start_xy、start_latlon 或 use_current_start 中的一项"
        )
    goal_xy = _pair(runtime.get("goal_xy"), "runtime.goal_xy")
    goal_latlon = _pair(runtime.get("goal_latlon"), "runtime.goal_latlon")
    if int(goal_xy is not None) + int(goal_latlon is not None) != 1:
        raise ValueError("直控终点必须且只能配置 goal_xy 或 goal_latlon 中的一项")


def _build_direct_command(config: dict[str, Any], config_path: Path) -> tuple[list[str], dict[str, str]]:
    """校验直控配置并构造 ``direct_control_node.py`` 命令。"""

    runtime = _object_section(config, "runtime")
    ros = _object_section(config, "ros")
    vehicle = _object_section(config, "vehicle")
    scene = _object_section(config, "scene")
    _reject_unknown_keys("runtime", runtime, DIRECT_RUNTIME_KEYS)
    _reject_unknown_keys("ros", ros, DIRECT_ROS_KEYS)
    _reject_unknown_keys("vehicle", vehicle, DIRECT_VEHICLE_KEYS)
    _reject_unknown_keys("scene", scene, DIRECT_SCENE_KEYS)
    _validate_direct_coordinates(runtime)

    map_path = _resolve_input_path(runtime.get("map_path"), config_path, "runtime.map_path")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "direct_control_node.py"),
        "--map-path",
        str(map_path),
    ]
    if runtime.get("cache_path"):
        cache_path = _resolve_optional_path(runtime["cache_path"], config_path)
        _append_value(command, "--cache-path", cache_path)
    _append_true_flag(
        command,
        "--rebuild-cache",
        runtime.get("rebuild_cache", False),
        "runtime.rebuild_cache",
    )
    _append_pair(command, "--start-xy", runtime.get("start_xy"), "runtime.start_xy")
    _append_pair(command, "--start-latlon", runtime.get("start_latlon"), "runtime.start_latlon")
    _append_true_flag(
        command,
        "--use-current-start",
        runtime.get("use_current_start", False),
        "runtime.use_current_start",
    )
    _append_pair(command, "--goal-xy", runtime.get("goal_xy"), "runtime.goal_xy")
    _append_pair(command, "--goal-latlon", runtime.get("goal_latlon"), "runtime.goal_latlon")

    planner_mode = str(runtime.get("planner_mode", "scene_target") or "scene_target").strip().lower()
    if planner_mode not in {"scene_target", "local_sampling"}:
        raise ValueError("runtime.planner_mode 必须是 scene_target 或 local_sampling")
    algorithm = str(scene.get("algorithm", "") or "").strip()
    if planner_mode == "scene_target" and not algorithm:
        raise ValueError("scene_target 直控模式必须配置 scene.algorithm")
    _append_value(command, "--planner-mode", planner_mode)
    _append_value(command, "--scene-algorithm-symbol", algorithm)

    runtime_scalar_flags = {
        "plan_hz": "--plan-hz",
        "control_hz": "--control-hz",
        "arrive_dist": "--arrive-dist",
        "arrive_speed": "--arrive-speed",
        "obu_track_timeout_ms": "--obu-track-timeout-ms",
        "node_name": "--node-name",
        "scene_trajectory_log_path": "--scene-trajectory-log-path",
        "speed_scale": "--speed-scale",
    }
    for key, flag in runtime_scalar_flags.items():
        if key in runtime:
            value = runtime[key]
            if key == "scene_trajectory_log_path" and value:
                value = _resolve_optional_path(value, config_path)
            _append_value(command, flag, value)
    if _bool_value(runtime.get("swap_latlon", False), "runtime.swap_latlon"):
        command.append("--swap-latlon")
    else:
        command.append("--no-swap-latlon")

    topic_values = {
        "--sensorgps-topic": ros.get("sensorgps_topic", ros.get("gps_topic")),
        "--actuator-topic": ros.get("actuator_topic"),
        "--obu-topic": ros.get("obu_topic"),
        "--controlcmd-topic": ros.get("controlcmd_topic", ros.get("control_topic")),
        "--trajectory-topic": ros.get("trajectory_topic"),
    }
    for flag, value in topic_values.items():
        _append_value(command, flag, value)

    vehicle_values = {
        "--obu-self-ptc-id": vehicle.get("self_ptc_id"),
        "--vehicle-length-mm": vehicle.get("length_mm"),
        "--vehicle-width-mm": vehicle.get("width_mm"),
        "--wheelbase-mm": vehicle.get("wheelbase_mm"),
        "--max-speed-kph": vehicle.get("max_speed_kph"),
        "--max-acc": vehicle.get("max_acc"),
        "--min-acc": vehicle.get("min_acc"),
    }
    for flag, value in vehicle_values.items():
        _append_value(command, flag, value)

    _append_value(command, "--scene-main-ptc-id", scene.get("main_ptc_id"))
    _append_pair(command, "--scene-main-start-xy", scene.get("main_start_xy"), "scene.main_start_xy")
    _append_pair(
        command,
        "--scene-main-start-latlon",
        scene.get("main_start_latlon"),
        "scene.main_start_latlon",
    )
    _append_pair(command, "--scene-main-goal-xy", scene.get("main_goal_xy"), "scene.main_goal_xy")
    _append_pair(
        command,
        "--scene-main-goal-latlon",
        scene.get("main_goal_latlon"),
        "scene.main_goal_latlon",
    )

    scene_environment = scene.get("environment", {})
    if not isinstance(scene_environment, dict):
        raise ValueError("scene.environment 必须是 JSON object")
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("ROS_SCENE_"):
            environment.pop(key, None)
    for key, value in scene_environment.items():
        if value is None or isinstance(value, (dict, list)):
            raise ValueError(f"scene.environment.{key} 必须是字符串、数字或布尔值")
        environment[str(key)] = str(value)
    return command, environment


def build_launch_command(
    config: dict[str, Any],
    config_path: Path,
) -> tuple[str, list[str], dict[str, str]]:
    """按 ``launch_mode`` 构造子进程命令。"""

    mode = str(config["launch_mode"]).strip().lower()
    if mode == "offline_simulation":
        _reject_unknown_keys("启动配置顶层", config, OFFLINE_TOP_LEVEL_KEYS)
        command, environment = _build_offline_command(config, config_path)
    else:
        _reject_unknown_keys("启动配置顶层", config, DIRECT_TOP_LEVEL_KEYS)
        command, environment = _build_direct_command(config, config_path)
    return mode, command, environment


def main() -> int:
    """统一启动器主入口。"""

    args = build_arg_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = _load_json_config(config_path)
    mode, command, environment = build_launch_command(config, config_path)
    config_dry_run = _bool_value(config.get("dry_run", False), "dry_run")
    dry_run = bool(args.dry_run or config_dry_run)
    if "confirm_real_vehicle" in config:
        _bool_value(config["confirm_real_vehicle"], "confirm_real_vehicle")
    if mode == "direct_control" and not dry_run and config.get("confirm_real_vehicle") is not True:
        raise ValueError(
            "直控实际启动前必须显式配置 confirm_real_vehicle=true；"
            "可先使用 dry_run=true 检查命令"
        )

    print(f"[scenario-launcher] config={config_path}", flush=True)
    print(f"[scenario-launcher] launch_mode={mode} dry_run={dry_run}", flush=True)
    print(f"[scenario-launcher] command={shlex.join(command)}", flush=True)
    scene_environment_keys = sorted(
        key for key in environment if key.startswith("ROS_SCENE_")
    )
    if scene_environment_keys:
        print(
            f"[scenario-launcher] scene_environment_keys={scene_environment_keys}",
            flush=True,
        )
    if dry_run:
        print("[scenario-launcher] validation passed; child process not started", flush=True)
        return 0

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def _entrypoint() -> int:
    """将配置错误转为简洁的命令行提示。"""

    try:
        return main()
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"[scenario-launcher] config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
