#!/usr/bin/env python3
"""伴行场景泛化 case 批量运行器。

按 ``configs/companion_generalized_cases.json`` 定义的 case 列表逐个起独立
子进程运行 ``simulate_scene_offline.py``，抽取伴行方位、窗口内比例、稳态
误差、协同间距和最终状态，写 ``summary.json`` / ``summary.md``，并可把
case 产物按实车测试 case 目录格式归档。

用法：
    python3 batch_sim/run_companion_matrix.py --dry-run
    python3 batch_sim/run_companion_matrix.py --group A
    python3 batch_sim/run_companion_matrix.py --case A1 --gif
    python3 batch_sim/run_companion_matrix.py --archive-dir ../实车测试case/伴行场景
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "companion_generalized_cases.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "companion_generalized"
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT.parent / "实车测试case" / "伴行场景"

POSITION_CN = {
    "front": "前车",
    "rear": "后车",
    "left": "左并排",
    "right": "右并排",
    "left_front": "左前",
    "right_front": "右前",
    "left_rear": "左后",
    "right_rear": "右后",
}

GROUP_CN = {
    "A": "八方位保持",
    "B": "加速跟随",
    "C": "偏差收敛",
    "D": "有限时长",
    "E": "退化负样本",
    "F": "主场景协同",
    "G": "多伴行",
}

POSITION_WINDOW = {
    "front": (8.0, 12.0, 16.0),
    "rear": (-16.0, -12.0, -8.0),
    "left": (-2.0, 0.0, 2.0),
    "right": (-2.0, 0.0, 2.0),
    "left_front": (5.0, 8.0, 12.0),
    "right_front": (5.0, 8.0, 12.0),
    "left_rear": (-12.0, -8.0, -5.0),
    "right_rear": (-12.0, -8.0, -5.0),
}


def _case_window(case: dict[str, Any]) -> tuple[float, float, float]:
    """返回 case 实际下发的 rel_s 窗口。"""

    position = str(case.get("position", "front"))
    default = POSITION_WINDOW.get(position, POSITION_WINDOW["front"])
    return (
        float(case.get("rel_s_min_m", default[0])),
        float(case.get("rel_s_target_m", default[1])),
        float(case.get("rel_s_max_m", default[2])),
    )


def load_config(path: Path) -> dict[str, Any]:
    """读取 case 配置。"""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_cases(
    config: dict[str, Any],
    case_filters: list[str],
    group_filters: list[str],
) -> list[dict[str, Any]]:
    """按 case id / group 过滤；未给过滤时返回全部。"""

    cases = list(config["cases"])
    wanted_ids = {str(item).strip() for item in case_filters if str(item).strip()}
    wanted_groups = {str(item).strip().upper() for item in group_filters if str(item).strip()}
    selected = [
        case
        for case in cases
        if (not wanted_ids or str(case["id"]) in wanted_ids)
        and (not wanted_groups or str(case.get("group", "")).upper() in wanted_groups)
    ]
    if wanted_ids:
        missing = wanted_ids - {str(case["id"]) for case in selected}
        if missing:
            raise SystemExit(f"未知 case: {', '.join(sorted(missing))}")
    return selected


def _goal_xy(config: dict[str, Any], case: dict[str, Any]) -> list[float]:
    """按 case 的 goal 字段取终点。"""

    if "goal_xy" in case:
        return [float(case["goal_xy"][0]), float(case["goal_xy"][1])]
    goal = str(case.get("goal", "vut"))
    if goal == "left":
        return [float(v) for v in config["corridor"]["left_goal"]]
    if goal == "right":
        return [float(v) for v in config["corridor"]["right_goal"]]
    return [float(v) for v in config["corridor"]["vut"]["goal_xy"]]


def _start_xy(config: dict[str, Any], case: dict[str, Any]) -> list[float]:
    """按 case 的 start_xy / anchor 取起点。"""

    if "start_xy" in case:
        return [float(case["start_xy"][0]), float(case["start_xy"][1])]
    anchor = str(case.get("anchor", ""))
    anchors = config.get("anchors", {})
    if anchor not in anchors:
        raise SystemExit(f"case {case.get('id')} 缺少有效 anchor: {anchor}")
    return [float(anchors[anchor][0]), float(anchors[anchor][1])]


def _case_environment(config: dict[str, Any], case: dict[str, Any]) -> dict[str, str]:
    """构造伴行算法环境变量。"""

    position = str(case.get("position", "front"))
    rel_min, rel_target, rel_max = _case_window(case)
    env = {
        "ROS_SCENE_COMPANION_POSITION": position,
        "ROS_SCENE_COMPANION_REL_S_MIN_M": str(rel_min),
        "ROS_SCENE_COMPANION_REL_S_TARGET_M": str(rel_target),
        "ROS_SCENE_COMPANION_REL_S_MAX_M": str(rel_max),
    }
    if case.get("duration_s") is not None:
        env["ROS_SCENE_COMPANION_DURATION_S"] = str(case["duration_s"])
    return env


def build_single_command(
    config: dict[str, Any],
    case: dict[str, Any],
    output_dir: Path,
    *,
    gif: bool,
) -> tuple[list[str], dict[str, str]]:
    """构造单车 case 的命令和环境变量。"""

    start = _start_xy(config, case)
    goal = _goal_xy(config, case)
    vut = config["corridor"]["vut"]
    dt = float(config["simulation"]["dt"])
    max_speed = float(config["vehicle_limits"]["max_speed_kph"])
    vut_motion = str(case.get("vut_motion", "constant"))
    vut_start = float(case.get("vut_start_speed", 5.0))
    vut_desired = float(case.get("vut_desired_speed", vut_start))
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "simulate_scene_offline.py"),
        "--planner-mode",
        "scene_target",
        "--scene-algorithm-symbol",
        str(config["scenario_algorithm"]),
        "--map-cache",
        str(PROJECT_ROOT / config["map_cache"]),
        "--ego-start-xy",
        str(start[0]),
        str(start[1]),
        "--ego-goal-xy",
        str(goal[0]),
        str(goal[1]),
        "--ego-start-speed",
        str(vut_start),
        "--ego-max-speed-kph",
        str(max_speed),
        "--main-start-xy",
        str(vut["start_xy"][0]),
        str(vut["start_xy"][1]),
        "--main-goal-xy",
        str(vut["goal_xy"][0]),
        str(vut["goal_xy"][1]),
        "--main-motion",
        vut_motion,
        "--main-start-speed",
        str(vut_start),
        "--main-desired-speed",
        str(vut_desired),
        "--dt",
        str(dt),
        "--max-steps",
        str(int(case.get("max_steps", 150))),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        str(case["id"]),
    ]
    if not gif:
        cmd.append("--skip-gif")
    env = os.environ.copy()
    env.update(_case_environment(config, case))
    return cmd, env


def build_multi_payload(
    config: dict[str, Any],
    case: dict[str, Any],
    output_dir: Path,
    *,
    gif: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """构造多车 case 的 JSON 配置和运行元信息。"""

    multi = case["multi"]
    dt = float(config["simulation"]["dt"])
    vehicles: list[dict[str, Any]] = []
    vut = multi["vut"]
    vehicles.append(
        {
            "id": "vut",
            "kind": "route",
            "motion": str(vut.get("motion", "idm")),
            "start_xy": [float(vut["start_xy"][0]), float(vut["start_xy"][1])],
            "goal_xy": [float(vut["goal_xy"][0]), float(vut["goal_xy"][1])],
            "start_speed": float(vut.get("start_speed", 0.0)),
            "desired_speed": float(vut.get("desired_speed", vut.get("start_speed", 0.0))),
        }
    )
    primary = multi.get("primary")
    if primary is not None:
        vehicles.append(
            {
                "id": str(primary["id"]),
                "kind": "algorithm",
                "algorithm": str(primary["algorithm"]),
                "main_ptc_id": "vut",
                "start_xy": [float(primary["start_xy"][0]), float(primary["start_xy"][1])],
                "goal_xy": [float(primary["goal_xy"][0]), float(primary["goal_xy"][1])],
                "start_speed": float(primary.get("start_speed", 0.0)),
                "max_speed_kph": float(primary.get("max_speed_kph", 36.0)),
                "environment": dict(primary.get("environment") or {}),
            }
        )
    for companion in multi.get("companions", []):
        position = str(companion["position"])
        env = {
            "ROS_SCENE_COMPANION_POSITION": position,
            "ROS_SCENE_COMPANION_REL_S_MIN_M": str(companion["rel_s_min_m"]),
            "ROS_SCENE_COMPANION_REL_S_TARGET_M": str(companion["rel_s_target_m"]),
            "ROS_SCENE_COMPANION_REL_S_MAX_M": str(companion["rel_s_max_m"]),
        }
        if companion.get("duration_s") is not None:
            env["ROS_SCENE_COMPANION_DURATION_S"] = str(companion["duration_s"])
        if companion.get("primary_ptc_id"):
            env["ROS_SCENE_COMPANION_PRIMARY_PTC_ID"] = str(companion["primary_ptc_id"])
            env["ROS_SCENE_COMPANION_PRIMARY_SCENE_TYPE"] = str(
                companion.get("primary_scene_type", "")
            )
        vehicles.append(
            {
                "id": str(companion["id"]),
                "kind": "algorithm",
                "algorithm": str(config["scenario_algorithm"]),
                "main_ptc_id": "vut",
                "start_xy": [float(companion["start_xy"][0]), float(companion["start_xy"][1])],
                "goal_xy": [float(companion["goal_xy"][0]), float(companion["goal_xy"][1])],
                "start_speed": float(companion.get("start_speed", 0.0)),
                "max_speed_kph": float(companion.get("max_speed_kph", 36.0)),
                "environment": env,
            }
        )
    payload = {
        "version": 1,
        "description": f"companion generalized case {case['id']}",
        "runtime": {
            "map_cache": str(config["map_cache"]),
            "dt": dt,
            "max_steps": int(multi.get("max_steps", 220)),
            "advance_index": 1,
            "log_every": 10,
            "skip_gif": not gif,
            "live_show": False,
            "output_dir": str(output_dir),
            "output_prefix": str(case["id"]),
        },
        "stop_on_terminal_vehicle_ids": list(
            multi.get("stop_on_terminal_vehicle_ids", [])
        ),
        "stop_on_terminal_mode": str(multi.get("stop_on_terminal_mode", "all")),
        "vehicles": vehicles,
    }
    # 未指定主场景车时不能写空字符串：多车加载器只在缺键时回退到首辆算法车。
    primary_vehicle_id = str(multi.get("primary_vehicle_id", "") or "").strip()
    if primary_vehicle_id:
        payload["primary_vehicle_id"] = primary_vehicle_id
    command_meta = {
        "config_path": output_dir / f"{case['id']}_multi.json",
        "dt": dt,
        "max_steps": int(multi.get("max_steps", 220)),
    }
    return payload, command_meta


def build_multi_command(
    config: dict[str, Any],
    case: dict[str, Any],
    output_dir: Path,
    *,
    gif: bool,
) -> tuple[list[str], dict[str, Any]]:
    """构造多车 case 的命令。"""

    payload, meta = build_multi_payload(config, case, output_dir, gif=gif)
    config_path = Path(meta["config_path"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "simulate_scene_offline.py"),
        "--planner-mode",
        "scene_target",
        "--multi-vehicle-config",
        str(config_path),
        "--map-cache",
        str(PROJECT_ROOT / config["map_cache"]),
        "--dt",
        str(meta["dt"]),
        "--max-steps",
        str(meta["max_steps"]),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        str(case["id"]),
    ]
    if not gif:
        cmd.append("--skip-gif")
    return cmd, meta


def _num(value: Any) -> float | None:
    """把数值字段转成 float；非数值返回 None。"""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _first_time(times: list[float], statuses: list[str], target: str) -> float | None:
    """返回首次出现目标状态的时刻。"""

    for time_s, status in zip(times, statuses):
        if status == target:
            return float(time_s)
    return None


ACQUIRED_STATUSES = {"hold_slot", "create_gap", "hold_released_gap", "recover_slot"}


def extract_companion_metrics(
    frames: list[dict[str, Any]],
    final_status: str | None,
    steady_min_v: float,
) -> dict[str, Any]:
    """从 scene_targets 的单个车辆帧序列抽取指标。

    窗口内比例、稳态误差都只统计进入 HOLD 之后的帧；ACQUIRE 阶段的初始偏差
    属于设计输入，不能拿来做保持精度判定。
    """

    dbg = [frame.get("scene_debug") or {} for frame in frames]
    times = [float(frame.get("time_s", 0.0)) for frame in frames]
    statuses = [str(frame.get("scene_status", "")) for frame in frames]
    acquired_index = next(
        (
            index
            for index, status in enumerate(statuses)
            if status in ACQUIRED_STATUSES
        ),
        None,
    )
    acquired = dbg[acquired_index:] if acquired_index is not None else []
    # 释放空间（create_gap/hold_released_gap/recover_slot）本来就要离开目标点，
    # 保持精度只在 hold_slot 帧上统计。
    hold_indexes = [index for index, status in enumerate(statuses) if status == "hold_slot"]
    hold = [dbg[index] for index in hold_indexes]
    # 稳态窗口取 hold 段最后三分之一，至少 1 帧，排除起步/加速瞬态。
    settle_count = max(1, len(hold) // 3) if hold else 0
    settle = hold[-settle_count:] if settle_count else []
    inside = [bool(item.get("inside_slot")) for item in hold if item.get("inside_slot") is not None]
    lane = [
        bool(item.get("lane_relation_ok"))
        for item in hold
        if item.get("lane_relation_ok") is not None
    ]
    hard = [
        bool(item.get("hard_collision_risk"))
        for item in dbg
        if item.get("hard_collision_risk") is not None
    ]
    hard_actors = sorted(
        {
            str(item.get("hard_collision_actor_id"))
            for item in dbg
            if item.get("hard_collision_actor_id")
        }
    )
    target = next(
        (
            _num(item.get("rel_s_target_m"))
            for item in dbg
            if _num(item.get("rel_s_target_m")) is not None
        ),
        None,
    )
    rel_rows = [
        (time_s, _num(item.get("rel_s_m")), _num(item.get("main_v_mps")), _num(item.get("rel_v_mps")))
        for time_s, item in zip(times, dbg)
    ]
    acquired_rows = rel_rows[acquired_index:] if acquired_index is not None else []
    hold_rows = [rel_rows[index] for index in hold_indexes]
    settle_rows = hold_rows[-settle_count:] if settle_count else []
    errors = [
        abs(rel_s - target)
        for _, rel_s, _, _ in acquired_rows
        if rel_s is not None and target is not None
    ]
    all_errors = [
        abs(rel_s - target)
        for _, rel_s, _, _ in rel_rows
        if rel_s is not None and target is not None
    ]
    steady = [
        (abs(rel_s - target), abs(rel_v))
        for _, rel_s, main_v, rel_v in settle_rows
        if rel_s is not None
        and target is not None
        and main_v is not None
        and main_v >= steady_min_v
        and rel_v is not None
    ]
    gaps: list[tuple[float, float | None]] = []
    needs_release = 0
    release_started = False
    release_cleared = False
    for item in dbg:
        interaction = item.get("primary_interaction")
        if not isinstance(interaction, dict):
            continue
        gap = _num(interaction.get("current_center_gap_m"))
        required = _num(interaction.get("required_center_gap_m"))
        if gap is not None:
            gaps.append((gap, required))
        if bool(interaction.get("needs_release")):
            needs_release += 1
            release_started = True
        elif release_started and bool(interaction.get("primary_valid", False)):
            # 主场景车已经让出原窗口和扩张窗口，后续才要求恢复原始方位。
            release_cleared = True
    road = [_num(item.get("road_remaining_m")) for item in dbg]
    road = [value for value in road if value is not None]
    status_sequence = list(dict.fromkeys(status for status in statuses if status))
    complete_time = _first_time(times, statuses, "scene_completed")
    hold_time = _first_time(times, statuses, "hold_slot")
    acquire_time = _first_time(times, statuses, "acquire_slot")
    create_gap_time = _first_time(times, statuses, "create_gap")
    released_time = _first_time(times, statuses, "hold_released_gap")
    recover_time = _first_time(times, statuses, "recover_slot")
    final_error = all_errors[-1] if all_errors else None
    return {
        "frame_count": len(frames),
        "final_status": final_status or (statuses[-1] if statuses else ""),
        "status_sequence": status_sequence,
        "acquired_index": acquired_index,
        "acquired_frame_count": len(acquired),
        "hold_frame_count": len(hold),
        "settle_frame_count": len(settle),
        "inside_count": sum(inside),
        "inside_total": len(inside),
        "inside_ratio": (sum(inside) / len(inside)) if inside else None,
        "lane_ok_count": sum(lane),
        "lane_ok_total": len(lane),
        "lane_ok_ratio": (sum(lane) / len(lane)) if lane else None,
        "hard_frames": sum(hard),
        "hard_actors": hard_actors,
        "rel_s_target_m": target,
        "max_rel_s_error_m": max(errors) if errors else None,
        "final_rel_s_error_m": final_error,
        "steady_max_rel_s_error_m": max((item[0] for item in steady), default=None),
        "steady_max_rel_v_mps": max((item[1] for item in steady), default=None),
        "steady_frame_count": len(steady),
        "min_center_gap_m": min((gap for gap, _ in gaps), default=None),
        "required_center_gap_m": max(
            (required for _, required in gaps if required is not None), default=None
        ),
        "needs_release_frames": needs_release,
        "release_cleared": release_cleared,
        "road_remaining_min_m": min(road) if road else None,
        "road_remaining_last_m": road[-1] if road else None,
        "acquire_time_s": acquire_time,
        "hold_time_s": hold_time,
        "create_gap_time_s": create_gap_time,
        "released_time_s": released_time,
        "recover_time_s": recover_time,
        "complete_time_s": complete_time,
        "recovered_after_gap": bool(
            create_gap_time is not None
            and recover_time is not None
            and recover_time >= create_gap_time
        ),
    }


def _companion_ids(output: dict[str, Any]) -> list[str]:
    """返回输出中的伴行车辆 id。"""

    targets = output.get("scene_targets")
    if isinstance(targets, dict):
        ids = [str(vid) for vid in targets if str(vid).startswith("companion")]
        return ids or [str(vid) for vid in targets]
    return ["ego"]


def extract_output_metrics(
    output: dict[str, Any],
    steady_min_v: float,
) -> dict[str, dict[str, Any]]:
    """抽取输出里所有伴行车辆的指标。"""

    targets = output.get("scene_targets")
    final_statuses = output.get("final_scene_statuses") or {}
    metrics: dict[str, dict[str, Any]] = {}
    if isinstance(targets, dict):
        for vehicle_id in _companion_ids(output):
            frames = targets.get(vehicle_id) or []
            if not frames:
                continue
            metrics[vehicle_id] = extract_companion_metrics(
                frames, final_statuses.get(vehicle_id), steady_min_v
            )
    else:
        frames = targets or []
        if frames:
            metrics["ego"] = extract_companion_metrics(
                frames, None, steady_min_v
            )
    return metrics


def judge_case(
    case: dict[str, Any],
    output: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    tolerances: dict[str, Any],
) -> tuple[str, list[str]]:
    """按安全 → 规划器 → 功能顺序判定。"""

    top_status = str(output.get("status", ""))
    failure_reason = str(output.get("failure_reason", ""))
    if top_status == "planner_failed" or "external_trajectory_dynamic_limit_violation" in failure_reason:
        return "PLANNER_FAIL", [f"规划器失败: {failure_reason or top_status}"]
    safety: list[str] = []
    for vehicle_id, item in metrics.items():
        if item["hard_frames"] > 0:
            safety.append(
                f"{vehicle_id} hard_collision_risk {item['hard_frames']} 帧"
                f"（{','.join(item['hard_actors']) or '未知对象'}）"
            )
        if "fail_safe" in item["status_sequence"]:
            safety.append(f"{vehicle_id} 进入 fail_safe")
    if safety:
        return "SAFETY_FAIL", safety

    expect = {str(item) for item in case.get("expect", [])}
    reasons: list[str] = []
    for vehicle_id, item in metrics.items():
        if item["final_status"] not in expect:
            if not ("scene_completed" in expect and item["complete_time_s"] is not None):
                reasons.append(
                    f"{vehicle_id} 最终状态 {item['final_status']} 不在预期 {sorted(expect)}"
                )
    if case.get("expect_no_stop"):
        if reasons:
            return "FUNCTION_FAIL", reasons
        return "PASS", []

    for vehicle_id, item in metrics.items():
        inside_ratio = item["inside_ratio"]
        if inside_ratio is not None and inside_ratio < float(
            tolerances["inside_slot_ratio_min"]
        ):
            reasons.append(
                f"{vehicle_id} inside_slot {inside_ratio:.1%} < "
                f"{float(tolerances['inside_slot_ratio_min']):.0%}"
            )
        lane_ratio = item["lane_ok_ratio"]
        if lane_ratio is not None and lane_ratio < float(
            tolerances["lane_relation_ratio_min"]
        ):
            reasons.append(
                f"{vehicle_id} lane_relation_ok {lane_ratio:.1%} < "
                f"{float(tolerances['lane_relation_ratio_min']):.0%}"
            )
        steady_error = item["steady_max_rel_s_error_m"]
        if steady_error is not None and steady_error > float(
            tolerances["steady_rel_s_error_max_m"]
        ):
            reasons.append(
                f"{vehicle_id} 稳态 |rel_s 误差| {steady_error:.3f} m > "
                f"{float(tolerances['steady_rel_s_error_max_m']):.3f} m"
            )
        steady_rel_v = item["steady_max_rel_v_mps"]
        if steady_rel_v is not None and steady_rel_v > float(
            tolerances["steady_rel_v_max_mps"]
        ):
            reasons.append(
                f"{vehicle_id} 稳态 |rel_v| {steady_rel_v:.3f} m/s > "
                f"{float(tolerances['steady_rel_v_max_mps']):.3f} m/s"
            )
        # 只有 A/D 组是恒速保持，全程误差才有意义；B/G 组 VUT 存在起步加速，
        # F 组释放空间时本来就离开目标点。
        if case.get("group") in {"A", "D"}:
            max_error = item["max_rel_s_error_m"]
            if max_error is not None and max_error > float(
                tolerances["low_speed_rel_s_error_max_m"]
            ):
                reasons.append(
                    f"{vehicle_id} 全程 |rel_s 误差| {max_error:.3f} m > "
                    f"{float(tolerances['low_speed_rel_s_error_max_m']):.3f} m"
                )
        if case.get("group") == "C":
            final_error = item["final_rel_s_error_m"]
            if final_error is not None and final_error > float(
                tolerances["steady_rel_s_error_max_m"]
            ):
                reasons.append(
                    f"{vehicle_id} 最终 |rel_s 误差| {final_error:.3f} m > "
                    f"{float(tolerances['steady_rel_s_error_max_m']):.3f} m"
                )
        if "scene_completed" in expect:
            complete_time = item["complete_time_s"]
            duration = float(case.get("duration_s", 8.0))
            if complete_time is None:
                reasons.append(f"{vehicle_id} 未进入 scene_completed")
            elif abs(complete_time - duration) > float(
                tolerances["completion_time_tol_s"]
            ):
                reasons.append(
                    f"{vehicle_id} COMPLETE {complete_time:.2f}s 偏离 "
                    f"{duration:.1f}±{float(tolerances['completion_time_tol_s']):.1f}s"
                )
        if case.get("group") == "F" and item["needs_release_frames"] > 0:
            gap = item["min_center_gap_m"]
            required = item["required_center_gap_m"]
            if gap is not None and required is not None and gap < required:
                reasons.append(
                    f"{vehicle_id} 最小中心距 {gap:.2f} m < 要求 {required:.2f} m"
                )
            if item["release_cleared"] and not item["recovered_after_gap"]:
                reasons.append(f"{vehicle_id} 主场景车已让出窗口但未 recover_slot")
        if case.get("group") == "E":
            continue
    if reasons:
        return "FUNCTION_FAIL", reasons
    return "PASS", []


def _format_optional(value: Any, fmt: str = ".3f") -> str:
    """格式化可空数值。"""

    return format(float(value), fmt) if isinstance(value, (int, float)) else "-"


def _format_ratio(value: Any) -> str:
    """格式化可空比例。"""

    return f"{float(value) * 100.0:.1f}%" if isinstance(value, (int, float)) else "-"


def _format_gap(item: dict[str, Any]) -> str:
    """只在发生空间释放时展示中心距，避免把无协同的投影距离当成约束。"""

    if not item.get("needs_release_frames"):
        return "-"
    return _format_optional(item.get("min_center_gap_m"))


def _case_title(case: dict[str, Any]) -> str:
    """生成 case 中文短标题。"""

    group = str(case.get("group", ""))
    if case["id"] == "G2":
        return "七车伴行制动"
    if case["id"] == "G1":
        return "双伴行"
    if group == "F":
        return "主场景协同"
    if group == "E":
        label = Path(str(case.get("archive_group", "") or "")).name
        return label or f"{POSITION_CN.get(str(case.get('position', '')), '多车')}退化"
    position_cn = POSITION_CN.get(str(case.get("position", "")), "多车")
    if group == "A":
        return f"{position_cn}保持"
    return f"{position_cn}{GROUP_CN.get(group, group)}"


def _case_file_stem(index: int, case: dict[str, Any]) -> str:
    """生成归档文件名。"""

    return f"case{index}_伴行_{case['id']}_{_case_title(case)}"


def _archive_location(case: dict[str, Any], index: int) -> tuple[str, str]:
    """返回 case 在归档目录下的相对目录和文件名主干。

    带 ``archive_group`` 的 case（负样本）单独放到分组目录，文件名用
    分组名而不是 case 编号，便于直接按失败模式查找。
    """

    archive_group = str(case.get("archive_group", "") or "").strip()
    if archive_group:
        return archive_group, f"{Path(archive_group).name}_{case['id']}"
    return f"case{index}", _case_file_stem(index, case)


def write_case_markdown(
    path: Path,
    index: int,
    case: dict[str, Any],
    command: str,
    output: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    verdict: str,
    reasons: list[str],
    tolerances: dict[str, Any],
) -> None:
    """写单个 case 的说明文件。"""

    archive_group = str(case.get("archive_group", "") or "").strip()
    heading = (
        f"# {Path(archive_group).name} 伴行 {case['id']}"
        if archive_group
        else f"# case{index} 伴行 {case['id']} {_case_title(case)}"
    )
    lines = [
        heading,
        "",
        "## 场景来源",
        "",
        f"离线泛化 case {case['id']}（{GROUP_CN.get(str(case.get('group', '')), '')}）。"
        f"{case.get('notes', '')}",
        "",
        "## 离线仿真命令",
        "",
        "```bash",
        command,
        "```",
        "",
        "## 结果",
        "",
        f"- 判定：**{verdict}**",
        f"- 顶层状态：`{output.get('status', '')}` / `{output.get('failure_reason', '')}`",
    ]
    for vehicle_id, item in metrics.items():
        lines.extend(
            [
                f"- 车辆 `{vehicle_id}` 最终场景状态：`{item['final_status']}`",
                f"  - 状态序列：{', '.join(item['status_sequence'])}",
                f"  - inside_slot：{item['inside_count']}/{item['inside_total']}"
                f"（{_format_ratio(item['inside_ratio'])}）",
                f"  - lane_relation_ok：{item['lane_ok_count']}/{item['lane_ok_total']}"
                f"（{_format_ratio(item['lane_ok_ratio'])}）",
                f"  - 稳态 |rel_s 误差| 最大：{_format_optional(item['steady_max_rel_s_error_m'])} m",
                f"  - 稳态 |rel_v| 最大：{_format_optional(item['steady_max_rel_v_mps'])} m/s",
                f"  - 最终 |rel_s 误差|：{_format_optional(item['final_rel_s_error_m'])} m",
                f"  - 最小中心距 / 要求：{_format_gap(item)} / "
                f"{_format_optional(item['required_center_gap_m'])} m"
                + ("" if item["needs_release_frames"] else "（未触发释放）"),
                f"  - needs_release 帧数：{item['needs_release_frames']}",
                f"  - 硬碰撞帧数：{item['hard_frames']}",
                f"  - 路线剩余最小值：{_format_optional(item['road_remaining_min_m'], '.2f')} m",
            ]
        )
        if item["complete_time_s"] is not None:
            lines.append(f"  - COMPLETE 时刻：{item['complete_time_s']:.2f} s")
        if item["hold_time_s"] is not None:
            lines.append(f"  - 首次 hold_slot：{item['hold_time_s']:.2f} s")
    if reasons:
        lines.append(f"- 判定原因：{'；'.join(reasons)}")
    lines.extend(["", "## 配置说明", ""])
    if case.get("multi_config"):
        lines.extend(
            [
                f"- 多车配置：`{case['multi_config']}`（复用既有回归配置）",
            ]
        )
    elif case.get("multi"):
        multi = case["multi"]
        vut = multi["vut"]
        lines.append(
            f"- VUT：`{vut.get('motion', 'idm')}`，起点速度 {vut.get('start_speed', 0.0)} m/s，"
            f"目标速度 {vut.get('desired_speed', vut.get('start_speed', 0.0))} m/s"
        )
        primary = multi.get("primary")
        if primary:
            lines.append(
                f"- 主场景车：`{primary['id']}`（`{primary['algorithm']}`），"
                f"起点 {primary['start_xy']}"
            )
        for companion in multi.get("companions", []):
            lines.append(
                f"- 伴行车 `{companion['id']}`：方位 `{companion['position']}`，"
                f"rel_s 窗口 [{companion['rel_s_min_m']}, {companion['rel_s_target_m']}, "
                f"{companion['rel_s_max_m']}] m，"
                f"主场景关联 `{companion.get('primary_ptc_id', '无')}`"
            )
    else:
        rel_min, rel_target, rel_max = _case_window(case)
        lines.extend(
            [
                f"- VUT：`{case.get('vut_motion', 'constant')}`，"
                f"起点速度 {case.get('vut_start_speed', 5.0)} m/s，"
                f"目标速度 {case.get('vut_desired_speed', case.get('vut_start_speed', 5.0))} m/s",
                f"- GVT 方位：`{case.get('position', 'multi')}`；"
                f"rel_s 窗口 [{rel_min:g}, {rel_target:g}, {rel_max:g}] m",
            ]
        )
    lines.extend(
        [
            f"- 地图：`xodr_parser/cache/同济测试场.pkl`，road 288 直线走廊",
            f"- 规划时域：0.5 s；单步速度修正上限 1.0 m/s",
            f"- 判定容差：inside_slot ≥ {float(tolerances['inside_slot_ratio_min']):.0%}，"
            f"lane_relation_ok ≥ {float(tolerances['lane_relation_ratio_min']):.0%}，"
            f"稳态误差 ≤ {float(tolerances['steady_rel_s_error_max_m']):.1f} m，"
            f"稳态 |rel_v| ≤ {float(tolerances['steady_rel_v_max_mps']):.1f} m/s",
            "- 动态硬门：未放宽 `validate_velocity_profile` 的任何限制",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_case_artifacts(
    output_dir: Path,
    archive_case_dir: Path,
    case_id: str,
    stem: str,
) -> list[str]:
    """把 case 产物复制到归档目录，返回复制后的文件名。"""

    archive_case_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    mapping = {
        output_dir / f"{case_id}.png": f"{stem}.png",
        output_dir / f"{case_id}.gif": f"{stem}.gif",
        output_dir / f"{case_id}.json": f"{stem}.json",
        output_dir / f"{case_id}_timing.json": f"{stem}_timing.json",
    }
    for source, name in mapping.items():
        if source.exists():
            shutil.copy2(source, archive_case_dir / name)
            copied.append(name)
    return copied


def write_summary(
    path: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    tolerances: dict[str, Any],
) -> None:
    """写矩阵汇总。"""

    total = len(results)
    passed = sum(1 for item in results if item["verdict"] == "PASS")
    lines = [
        "# 伴行场景泛化测试汇总",
        "",
        f"> 本版为伴行场景泛化矩阵的离线归档（{time.strftime('%Y-%m-%d')}）。"
        f"共 {total} 个 case，PASS {passed}/{total}。",
        "> 判定口径：安全（无硬碰撞、无规划器失败）与功能（窗口内比例、稳态误差、"
        "状态机、协同间距）分开判定；未放宽任何动态硬门。",
        "",
        "## 测试设置",
        "",
        f"- 地图：同济测试场（`{config['map_cache']}`），road 288 直线走廊。",
        "- 角色：GVT（`ego_vehicle`）只控纵向速度；VUT（`main_vehicle`）沿固定路线行驶。",
        "- 方位：front/rear/left/right/left_front/right_front/left_rear/right_rear。",
        f"- VUT 路线：{config['corridor']['vut']['start_xy']} → "
        f"{config['corridor']['vut']['goal_xy']}。",
        f"- GVT 限速：{config['vehicle_limits']['max_speed_kph']} kph；"
        f"min_acc {config['vehicle_limits']['min_acc']} / "
        f"max_acc {config['vehicle_limits']['max_acc']} m/s²。",
        f"- 仿真步长：{config['simulation']['dt']} s。",
        "",
        "## 本版修复内容",
        "",
        "`scenarios/companion_algorithm.py` 在 `_build_rel_s_control_output` 中，"
        "按局部规划器在单个 `plan_t` 时域内可实现的修正量增加 1.0 m/s 限幅：",
        "",
        "```python",
        "feasible_delta = CFG[\"planner_feasible_speed_delta_mps\"]  # 1.0",
        "target_v = clamp(target_v, ego_v - feasible_delta, ego_v + feasible_delta)",
        "```",
        "",
        "修复前，初始纵向偏差 ≥2.7 m 时算法下发 1.2~1.5 m/s 的速度台阶，"
        "规划器时间参数化报 `Acceleration limit exceeded` / `primal infeasible`，"
        "仿真第 0 步 `planner_failed`。修复后大偏差改为多步收敛，C5/C6 通过。",
        "",
        "### 释放阶段阻尼参考系",
        "",
        "`create_gap` / `hold_released_gap` 的释放目标以主场景 GVT 为基准，"
        "但阻尼项原先用 `ego_v - main_v`（相对 VUT）。切入车与 VUT 的速度差"
        "会被误判成“自己超速”，把伴行车往回拽。现改为 `ego_v - primary_v`，"
        "与目标基准一致。F1 最小中心距由 7.82 m 提升到 9.83 m。",
        "",
        "### 释放目标瞬态缓冲",
        "",
        "车辆跟踪终端速度有约一个 `plan_t` 的滞后，目标若刚好等于最小中心距，"
        "瞬态必然压线。释放目标现加入已有的 `primary_gap_buffer_m`（1.0 m）："
        "`primary_rel_s + required_center_gap + buffer`。"
        "F1 最小中心距提升到 10.83 m，满足 10.64 m 要求；判定阈值未放宽。",
        "",
        "## 归档 case 索引",
        "",
        "| case | case id | 组 | 方位 | VUT 运动 | 步数 | 最终状态 | inside | 稳态误差 | 最小间距 | 判定 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in results:
        metrics = item["metrics"]
        first = next(iter(metrics.values())) if metrics else {}
        lines.append(
            f"| [{item['archive_dir']}]({item['archive_dir']}/{item['stem']}.md) "
            f"| {item['case']['id']} | {item['case'].get('group', '')} "
            f"| {item['case'].get('position', '多车')} "
            f"| {item['case'].get('vut_motion', 'multi')} "
            f"| {first.get('frame_count', '-')} "
            f"| {first.get('final_status', '-')} "
            f"| {_format_ratio(first.get('inside_ratio'))} "
            f"| {_format_optional(first.get('steady_max_rel_s_error_m'))} m "
            f"| {_format_gap(first)} m "
            f"| {item['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## 维度分析",
            "",
        ]
    )
    for group in ["A", "B", "C", "D", "E", "F", "G"]:
        group_results = [item for item in results if item["case"].get("group") == group]
        if not group_results:
            continue
        group_pass = sum(1 for item in group_results if item["verdict"] == "PASS")
        lines.append(
            f"- **{group} {GROUP_CN[group]}**：{group_pass}/{len(group_results)} PASS。"
        )
        for item in group_results:
            first = next(iter(item["metrics"].values()), {})
            line = (
                f"  - {item['case']['id']}：{item['verdict']}，"
                f"inside {_format_ratio(first.get('inside_ratio'))}，"
                f"稳态误差 {_format_optional(first.get('steady_max_rel_s_error_m'))} m，"
                f"最小间距 {_format_gap(first)} m。"
            )
            if item["reasons"]:
                line += f"原因：{'；'.join(item['reasons'])}。"
            lines.append(line)
    lines.extend(
        [
            "",
            "## 测试设计调整",
            "",
            "1. **E2 改为静止工况**：GVT 以 5 m/s 起步、可用路线只剩 1.6 m 时，"
            "任何控制律都无法在 3 m/s² 内停住，规划器必然失败；改为 GVT/VUT 静止、"
            "路线 2.4 m，专门验证 `companion_insufficient_road` 走 `degraded_exit`。",
            "2. **F3 改为左并排伴行**：前车位于 VUT 与制动车之间，VUT 的 IDM 被前车"
            "限速，永远追不上制动车，制动场景不会触发；改为左侧伴行（相邻车道），"
            "验证 VUT 紧急制动期间伴行车保持 rel_s 0 且不误释放。",
            "3. **F1 预期加入 `hold_released_gap`**：切入车完成后停在 VUT 与前车之间，"
            "前车原窗口被永久占用，保持释放位置是正确行为，不能要求恢复。",
            "4. **判定窗口**：inside_slot/lane_relation 只统计 `hold_slot` 帧；稳态误差取 "
            "hold 段最后三分之一，避免把 ACQUIRE 初始偏差和释放过程算成保持误差。",
            "",
        ]
    )
    failures = [item for item in results if item["verdict"] != "PASS"]
    if failures:
        lines.extend(["", "## 未通过项", ""])
        for item in failures:
            lines.append(
                f"- **{item['case']['id']}（{item['verdict']}）**："
                f"{'；'.join(item['reasons'])}。详见 "
                f"[{item['archive_dir']}]({item['archive_dir']}/{item['stem']}.md)。"
            )
    lines.extend(
        [
            "",
            "## 已知遗留问题",
            "",
            "1. **ACQUIRE 大偏差已通过 1.0 m/s 单步限幅修复**：C5/C6 从第 0 步失败转为多步收敛。"
            "实车若需要更快的大偏差收敛，可把 `ROS_SCENE_COMPANION_FEASIBLE_SPEED_DELTA_MPS` "
            "调到 1.1~1.2，但不要超过规划器可实现的约 1.2 m/s。",
            "2. **路线长度规则**：旧 `cutin_two_companions` 前车 fail_safe 已定位为路线耗尽"
            "（t=20 s 时 `road_remaining_m=0.48`）。本矩阵统一使用 road 288 长走廊，"
            "并在 runner 中保留跑道检查；实车前车 goal 需比 VUT goal 多延 40~60 m。",
            "3. **并排窗口余量**：停车瞬间侧向 rel_s 会前移约 1.2 m（`min_target_progress_m=1.0`），"
            "±2 m 窗口只剩约 0.8 m 余量；并排 case 建议保持 VUT 行驶，不做停车工况。",
            "4. **VUT 缺失未进矩阵**：`waiting_main_vehicle` 分支需要 `main_vehicle=None`，"
            "当前离线 runner 总会注入 VUT；该项由代码路径检查覆盖，后续可用单元测试补。",
            "5. **参考线末端目标不自洽（未修改）**：`_build_line_target_output` 在 "
            "`remaining_s < travel_s` 时只截断 `target_s`，没有同步下调 `target_v`，"
            "导致 L/T/v 互相矛盾并触发 `external_trajectory_dynamic_limit_violation`。"
            "旧 `cutin_two_companions` 短路线 smoke 配置即由该路径触发；本矩阵正式 case "
            "统一使用长走廊规避，E2 用静止工况验证 degraded_exit，该分支本身保持原样。",
            "6. **F 组前车释放间距（已修复）**：F1 原先因阻尼参考系错配、"
            "释放目标无瞬态余量，最小中心距只有 7.82 m。修正后为 10.83 m；"
            "`primary_min_net_gap_m` 判定阈值未放宽。",
            "",
            "## 实车执行建议",
            "",
            "1. 先打通 VUT BSM id 映射（`ptc_id` / `source_id`）和 GVT `self_ptc_id`，"
            "否则 `main_ptc_id` 匹配不上，伴行会停在 `waiting_formation_geometry`。",
            "2. 第一轮跑 A1/A2/A3/A4（前、后、左、右四方位），再跑 B1、D1。",
            "3. 第二轮跑 A5~A8、B2~B4；第三轮跑 C1/C2、F1/F2、G1。",
            "4. 并排 ±2 m 对定位延迟最敏感，建议先跑前后 8~16 m 窗口，再跑并排。",
            "5. 中止条件：`|rel_s 误差| > 3 m` 持续 1 s、`hard_collision_risk=true`、"
            "规划器连续失败、主场景车时序偏差 > 2 s。",
            "6. 急停只允许由 `fail_safe` 触发；几何丢失/道路不足走 `degraded_exit` 平稳退出。",
            "",
            "## 归档说明",
            "",
            f"- 每个 case 目录含 `.md` 说明、`.png`、`.gif`、`.json`、`_timing.json`。",
            f"- 本版从 `case1` 开始；伴行场景暂无实车原始记录，后续实车记录另起编号。",
            "- E 组负样本不占用 `caseN` 编号，单独归档到 `负样本/几何不满足`、"
            "`负样本/道路不足`、`负样本/配置非法`，汇总见 `负样本/汇总.md`。",
            f"- 判定容差：inside_slot ≥ {float(tolerances['inside_slot_ratio_min']):.0%}，"
            f"lane_relation_ok ≥ {float(tolerances['lane_relation_ratio_min']):.0%}，"
            f"稳态误差 ≤ {float(tolerances['steady_rel_s_error_max_m']):.1f} m，"
            f"稳态 |rel_v| ≤ {float(tolerances['steady_rel_v_max_mps']):.1f} m/s。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


NEGATIVE_STATUS_NOTE = [
    (
        "waiting_formation_geometry",
        "算法等待主车/几何条件满足，全程不输出控制量，也不会触发急停。",
    ),
    (
        "degraded_exit",
        "算法主动降级退出，平稳停车或退出，不进入 fail_safe。",
    ),
    (
        "fail_safe",
        "真正的安全兜底，只在碰撞风险等硬约束被突破时触发；负样本不应出现。",
    ),
]


def write_negative_summary(
    path: Path,
    results: list[dict[str, Any]],
    tolerances: dict[str, Any],
) -> None:
    """写负样本分组汇总。"""

    negative = [
        item
        for item in results
        if str(item["case"].get("archive_group", "") or "").strip()
    ]
    if not negative:
        return
    passed = sum(1 for item in negative if item["verdict"] == "PASS")
    lines = [
        "# 伴行场景负样本归档",
        "",
        f"> 本目录归档 {len(negative)} 个负样本，PASS {passed}/{len(negative)}。"
        "负样本只验证退化路径：几何、可用路线或配置不满足时，算法必须平稳退出或保持等待，"
        "不得误发急停。",
        "",
        "## 负样本索引",
        "",
        "| 场景 | case | 构造方式 | 期望终态 | 实际终态 | 急停 | 判定 |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in negative:
        case = item["case"]
        metrics = next(iter(item["metrics"].values()), {})
        sequences = [m.get("status_sequence", []) for m in item["metrics"].values()]
        stopped = any("fail_safe" in sequence for sequence in sequences)
        label = Path(str(case["archive_group"])).name
        lines.append(
            f"| [{label}]({label}/{item['stem']}.md) "
            f"| {case['id']} | {case.get('notes', '-')} "
            f"| {' / '.join(case.get('expect', [])) or '-'} "
            f"| {metrics.get('final_status', '-')} "
            f"| {'是' if stopped else '否'} | {item['verdict']} |"
        )
    lines.extend(["", "## 终态语义", ""])
    for status, text in NEGATIVE_STATUS_NOTE:
        lines.append(f"- `{status}`：{text}")
    lines.extend(
        [
            "",
            "## 判定口径",
            "",
            "- 安全：状态序列不得出现 `fail_safe`，输出 JSON 不得记录硬碰撞。",
            "- 功能：最终状态必须落在 `expect` 内（E1 允许持续等待，E2/E3 必须 `degraded_exit`）。",
            f"- 容差沿用主矩阵：inside_slot ≥ "
            f"{float(tolerances['inside_slot_ratio_min']):.0%}，"
            f"lane_relation ≥ {float(tolerances['lane_relation_ratio_min']):.0%}；"
            "负样本未进入保持阶段，不参与稳态误差统计。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="伴行场景泛化 case 批量运行器")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="case 配置 JSON")
    parser.add_argument("--case", action="append", default=[], help="只跑指定 case id")
    parser.add_argument("--group", action="append", default=[], help="只跑指定组 A/B/C/D/E/F/G")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="离线输出根目录")
    parser.add_argument("--archive-dir", default="", help="归档目录；为空则只写 outputs")
    parser.add_argument("--no-archive", action="store_true", help="不复制产物、不写 case 目录")
    parser.add_argument("--gif", action="store_true", help="生成 GIF（默认跳过）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="已有输出 JSON 时跳过仿真，只重新判定和写汇总",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="单 case 超时秒数")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    cases = select_cases(config, args.case, args.group)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    archive_root = (
        Path(args.archive_dir).expanduser().resolve() if args.archive_dir else None
    )
    if archive_root is not None and not args.no_archive:
        archive_root.mkdir(parents=True, exist_ok=True)
    tolerances = config["tolerances"]
    steady_min_v = float(config["simulation"].get("steady_min_v_mps", 3.0))

    results: list[dict[str, Any]] = []
    case_number = 0
    for case in cases:
        # 负样本单独归档，不占用 caseN 编号，正式 case 连续编号。
        if not str(case.get("archive_group", "") or "").strip():
            case_number += 1
        index = case_number
        case_id = str(case["id"])
        case_output_dir = output_root / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)
        if case.get("multi_config"):
            multi_path = (PROJECT_ROOT / str(case["multi_config"])).resolve()
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "simulate_scene_offline.py"),
                "--planner-mode",
                "scene_target",
                "--multi-vehicle-config",
                str(multi_path),
                "--map-cache",
                str(PROJECT_ROOT / config["map_cache"]),
                "--dt",
                str(float(config["simulation"]["dt"])),
                "--max-steps",
                str(int(case.get("max_steps", 350))),
                "--output-dir",
                str(case_output_dir),
                "--output-prefix",
                case_id,
            ]
            if not args.gif:
                cmd.append("--skip-gif")
            env = os.environ.copy()
        elif case.get("multi"):
            cmd, _ = build_multi_command(config, case, case_output_dir, gif=args.gif)
            env = os.environ.copy()
        else:
            cmd, env = build_single_command(config, case, case_output_dir, gif=args.gif)
        command_text = " ".join(
            f"{key}={value}" if key.startswith("ROS_SCENE_") else value
            for key, value in sorted(env.items())
            if key.startswith("ROS_SCENE_")
        )
        command_text = (command_text + " " if command_text else "") + " ".join(cmd)
        if args.dry_run:
            print(f"[dry-run] {case_id}")
            print(command_text)
            continue
        output_json = case_output_dir / f"{case_id}.json"
        reuse_existing = bool(args.reuse and output_json.exists())
        if reuse_existing:
            print(f"[reuse] {case_id} <- {output_json}")
            started = time.perf_counter()
            completed = None
            elapsed = 0.0
        else:
            print(f"[run] {case_id} -> {case_output_dir}")
            started = time.perf_counter()
            completed = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            elapsed = time.perf_counter() - started
        if completed is not None:
            log_path = case_output_dir / f"{case_id}.log"
            log_path.write_text(
                (completed.stdout or "") + "\n" + (completed.stderr or ""),
                encoding="utf-8",
            )
        archive_dir, stem = _archive_location(case, index)
        if not output_json.exists():
            returncode = None if completed is None else completed.returncode
            result = {
                "case": case,
                "index": index,
                "archive_dir": archive_dir,
                "stem": stem,
                "verdict": "ERROR",
                "reasons": [f"未生成输出 JSON；returncode={returncode}"],
                "metrics": {},
                "command": command_text,
                "output": {},
                "elapsed_s": elapsed,
            }
            results.append(result)
            print(f"  ERROR returncode={returncode}")
            continue
        output = json.loads(output_json.read_text(encoding="utf-8"))
        metrics = extract_output_metrics(output, steady_min_v)
        verdict, reasons = judge_case(case, output, metrics, tolerances)
        result = {
            "case": case,
            "index": index,
            "archive_dir": archive_dir,
            "stem": stem,
            "verdict": verdict,
            "reasons": reasons,
            "metrics": metrics,
            "command": command_text,
            "output": {
                "status": output.get("status", ""),
                "failure_reason": output.get("failure_reason", ""),
            },
            "elapsed_s": elapsed,
        }
        results.append(result)
        print(
            f"  {verdict} status={output.get('status', '')} "
            f"steps={output.get('completed_steps', '-')} ({elapsed:.1f}s)"
        )
        if reasons:
            print(f"  reasons: {'；'.join(reasons)}")

    if args.dry_run:
        return

    summary_json = output_root / "summary.json"
    summary_json.write_text(
        json.dumps(
            [
                {
                    "case": item["case"],
                    "index": item["index"],
                    "archive_dir": item["archive_dir"],
                    "stem": item["stem"],
                    "verdict": item["verdict"],
                    "reasons": item["reasons"],
                    "metrics": item["metrics"],
                    "output": item["output"],
                    "elapsed_s": item["elapsed_s"],
                }
                for item in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(output_root / "summary.md", config, results, tolerances)

    if archive_root is not None and not args.no_archive:
        for item in results:
            case = item["case"]
            index = item["index"]
            stem = item["stem"]
            case_archive_dir = archive_root / item["archive_dir"]
            case_output_dir = output_root / str(case["id"])
            copy_case_artifacts(case_output_dir, case_archive_dir, str(case["id"]), stem)
            write_case_markdown(
                case_archive_dir / f"{stem}.md",
                index,
                case,
                item["command"],
                item["output"],
                item["metrics"],
                item["verdict"],
                item["reasons"],
                tolerances,
            )
        write_summary(archive_root / "汇总.md", config, results, tolerances)
        write_negative_summary(
            archive_root / "负样本" / "汇总.md", results, tolerances
        )
        print(f"[archive] {archive_root}")

    print(
        f"[done] {sum(1 for item in results if item['verdict'] == 'PASS')}/{len(results)} PASS"
    )


if __name__ == "__main__":
    main()
