"""Cut-out 切出场景算法。

设计目标：
- APPROACH 解析 VUT 主车、GVT 自车、暴露目标三车关系
- PREPARE 阶段 GVT 保持当前车道，每帧动态重算切出触发时机
- ACTIVE 一旦开始切出，锁定目标车道和终端点，固定时长执行，不再重解
- STABILIZE 完全进入目标车道后稳定保持，并记录 VUT->暴露目标 THW/TTC 命中情况
- COOLDOWN 场景失败后短时冷却，避免连续重复触发
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from planning_runtime.frame import PlanningActorState, build_planning_actor_state
from planning_runtime.scene.types import ReferenceLineView, SceneContext
from traj_planner.tools import calculate_path_cumulative_distance, get_path_point_at_s, normalize_angle, project_point_to_path


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量；读取失败时回退默认值。"""

    raw_value = os.getenv(name, "")
    if raw_value == "":
        return float(default)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return float(default)


def _env_str(name: str, default: str = "") -> str:
    """读取字符串环境变量；空值时回退默认值。"""

    value = os.getenv(name, "")
    return str(value if value != "" else default)


@dataclass(frozen=True)
class VehicleLimits:
    """场景算法实际使用的车辆尺寸与平台约束。"""

    ego_length: float
    main_length: float
    exposed_length: float
    max_v: float
    min_acc: float
    max_acc: float


CFG: dict[str, Any] = {
    # 角色和方向配置。
    # main_ptc_id 由现有 SceneTargetProvider 解析为 context.main_vehicle；
    # exposed_ptc_id 在本文件内从 planning_frame.obstacles 中解析。
    "exposed_ptc_id": _env_str("ROS_SCENE_EXPOSED_PTC_ID", ""),
    "cutout_side": _env_str("ROS_SCENE_CUTOUT_SIDE", "left"),
    # 核心场景指标：静止暴露目标用 THW，低速暴露目标用 TTC。
    "target_thw_s": _env_float("ROS_SCENE_CUTOUT_TARGET_THW_S", 2.0),
    "target_ttc_s": _env_float("ROS_SCENE_CUTOUT_TARGET_TTC_S", 2.0),
    "thw_tol_s": _env_float("ROS_SCENE_CUTOUT_THW_TOL_S", 0.25),
    "ttc_tol_s": _env_float("ROS_SCENE_CUTOUT_TTC_TOL_S", 0.35),
    "hard_min_thw_s": _env_float("ROS_SCENE_CUTOUT_HARD_MIN_THW_S", 0.8),
    "hard_min_ttc_s": _env_float("ROS_SCENE_CUTOUT_HARD_MIN_TTC_S", 1.0),
    # 暴露目标速度小于该阈值时按静止目标处理，避免浮点噪声造成误判。
    "static_speed_eps_mps": 0.05,
    "vut_metric_min_speed_mps": 0.3,
    # 切出时序。PREPARE 中搜索 hold_t；进入 ACTIVE 后锁定本次动态换道时间。
    "lane_change_time_s": _env_float("ROS_SCENE_CUTOUT_LANE_CHANGE_TIME_S", 4.0),
    "lane_change_time_min_s": _env_float("ROS_SCENE_CUTOUT_LANE_CHANGE_TIME_MIN_S", 3.0),
    "lane_change_time_low_speed_mps": _env_float("ROS_SCENE_CUTOUT_LANE_CHANGE_TIME_LOW_SPEED_MPS", 5.0),
    "lane_change_time_high_speed_mps": _env_float("ROS_SCENE_CUTOUT_LANE_CHANGE_TIME_HIGH_SPEED_MPS", 15.0),
    # VUT->暴露目标指标评估时刻。0.5 表示换道时间过半时评估，换道执行时间仍保持不变。
    "metric_eval_time_ratio": 0.5,
    "hold_t_grid": [round(0.05 * i, 2) for i in range(0, 161)],
    "activation_hold_t_max_s": 0.3,
    "prepare_plan_t_s": 1.5,
    "prepare_wait_time_buffer_s": 0.5,
    "prepare_target_extra_progress_m": 0.0,
    "prepare_timeout_s": 10.0,
    "stabilize_duration_s": 1.5,
    "abort_cooldown_s": 2.0,
    # 基础巡航与几何阈值。
    "cruise_target_v": 5.0,
    "max_heading_diff_deg": 20.0,
    "source_lane_match_d_thresh_m": 1.5,
    "target_lane_match_d_thresh_m": 1.5,
    "max_ego_to_target_line_d_m": 4.8,
    "min_target_progress_m": 3.0,
    "post_cutout_buffer_m": 6.0,
    "enter_target_d_thresh_m": 0.65,
    "execute_complete_frames": 2,
    # GVT 等待和切出自身的硬安全。第一版不用 THW/TTC 约束这些关系。
    "min_gvt_to_exposed_gap_m": 5.0,
    "source_abort_gap_m": 2.0,
    "source_release_time_ratio": 0.5,
    "source_release_time_max_s": 2.0,
    "source_release_time_min_s": 0.5,
    # 目标车道默认应无车。这里仅做硬距离防护，不作为场景主逻辑。
    "target_lane_obstacle_check_mode": "hard_only",
    "target_lane_front_min_gap_m": 7.0,
    "target_lane_rear_min_gap_m": 7.0,
    "target_lane_obstacle_d_thresh_m": 1.5,
    # 预测时对外部车辆加速度做限幅，降低 OBU 噪声对触发时机的影响。
    "prediction_min_acc": -1.5,
    "prediction_max_acc": 1.5,
}


SCENE_MEMORY_TEMPLATE: dict[str, Any] = {
    "mode": "APPROACH",
    "cooldown_until_time_s": -1.0,
    "prepare_start_time_s": -1.0,
    "prepare_locked_target_lane_id": "",
    "prepare_locked_target_lane_sequence": [],
    "active_start_time_s": -1.0,
    "active_end_time_s": -1.0,
    "active_metric_eval_time_s": -1.0,
    "locked_lane_change_time_s": -1.0,
    "metric_eval_recorded": False,
    "locked_target_line_index": None,
    "locked_target_lane_id": "",
    "locked_target_lane_sequence": [],
    "locked_source_lane_id": "",
    "locked_source_lane_sequence": [],
    "locked_source_line": None,
    "locked_source_cumulative_s": None,
    "locked_exposed_ptc_id": "",
    "locked_metric_name": "",
    "locked_expected_metric": float("inf"),
    "locked_target_xy": None,
    "locked_target_v": 0.0,
    "locked_target_a": 0.0,
    "locked_target_heading": 0.0,
    "execute_complete_hits": 0,
    "stable_until_time_s": -1.0,
    "actual_finish_metric": None,
    "last_debug": None,
}


SCENE_MEMORY: dict[str, Any] = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Public API / 外部入口 ====================


def cut_out_algorithm(context: SceneContext) -> dict[str, Any]:
    """Cut-out 场景算法主入口。"""

    global SCENE_MEMORY
    start_ms = time.time() * 1000.0
    if int(context.step_index) == 0:
        _reset_scene_memory()
    output, SCENE_MEMORY = _scene_fsm_step(context, SCENE_MEMORY)
    end_ms = time.time() * 1000.0
    print(f"Cut-out algorithm execution time: {end_ms - start_ms} ms")
    return output


# ==================== FSM / 状态机分发 ====================


def _scene_fsm_step(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行一帧 cut-out 状态机。"""

    now = float(context.time_s)
    mode = str(memory.get("mode", "APPROACH"))

    if mode == "COOLDOWN":
        if now < float(memory["cooldown_until_time_s"]):
            return _handle_cooldown(context, memory), memory
        memory["mode"] = "APPROACH"
        mode = "APPROACH"

    if mode == "APPROACH":
        return _handle_approach(context, memory, now)
    if mode == "PREPARE":
        return _handle_prepare(context, memory, now)
    if mode == "ACTIVE":
        return _handle_active(context, memory, now)
    if mode == "STABILIZE":
        return _handle_stabilize(context, memory, now)
    return _handle_unknown_mode(context, memory)


# ==================== State Handlers / 每个状态独立逻辑函数 ====================


def _handle_cooldown(
    context: SceneContext,
    memory: dict[str, Any],
) -> dict[str, Any]:
    """处理 COOLDOWN：冷却期内只沿当前车道保底行驶。"""

    return _build_cutout_safe_keep_lane_output(
        context,
        memory,
        reason="cutout_cooldown",
        scene_status="cooldown",
        scene_debug=copy.deepcopy(memory.get("last_debug")),
    )


def _handle_approach(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 APPROACH：解析三车关系和目标车道，满足条件后进入 PREPARE。"""

    scene = _build_cutout_scene(context)
    if not bool(scene["ok"]):
        _reset_prepare_memory(memory)
        _reset_execute_memory(memory)
        return _build_cutout_safe_keep_lane_output(
            context,
            memory,
            reason="cutout_wait_scene",
            scene_status="waiting_cutout_scene",
            scene_detail=str(scene["reason"]),
            scene_debug=scene["debug"],
        ), memory

    _enter_prepare_mode(memory, scene["target_line"], now)
    return _build_prepare_step_output(context, memory, scene, now, reason="cutout_prepare_enter"), memory


def _handle_prepare(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 PREPARE：GVT 仍在源车道，每帧重算 VUT->暴露目标的触发时机。"""

    scene = _build_cutout_scene(context, memory=memory)
    if not bool(scene["ok"]):
        memory["mode"] = "APPROACH"
        _reset_prepare_memory(memory)
        return _build_cutout_safe_keep_lane_output(
            context,
            memory,
            reason="cutout_scene_lost",
            scene_status="waiting_cutout_scene",
            scene_detail=str(scene["reason"]),
            scene_debug=scene["debug"],
        ), memory

    elapsed = max(0.0, float(now) - float(memory.get("prepare_start_time_s", now)))
    if elapsed >= float(CFG["prepare_timeout_s"]):
        _enter_cooldown(memory, now, scene["debug"])
        return _build_cutout_safe_keep_lane_output(
            context,
            memory,
            reason="cutout_prepare_timeout",
            scene_status="prepare_timeout",
            scene_debug=scene["debug"],
        ), memory

    trigger = _evaluate_cutout_trigger(context, scene)
    safety = _evaluate_cutout_safety(context, scene)
    debug = copy.deepcopy(scene["debug"])
    debug.update(copy.deepcopy(trigger["debug"]))
    debug.update(copy.deepcopy(safety["debug"]))
    memory["last_debug"] = copy.deepcopy(debug)

    if bool(trigger["late_abort"]):
        _enter_cooldown(memory, now, debug)
        return _build_cutout_safe_keep_lane_output(
            context,
            memory,
            reason="cutout_metric_late",
            scene_status="metric_late",
            scene_detail=str(trigger["reason"]),
            scene_debug=debug,
        ), memory

    if bool(trigger["activation_ready"]) and bool(safety["ok"]):
        active_solution = _build_active_solution(context, scene, trigger, debug)
        if active_solution is None:
            _enter_cooldown(memory, now, debug)
            return _build_cutout_safe_keep_lane_output(
                context,
                memory,
                reason="cutout_active_solution_failed",
                scene_status="infeasible_geometry",
                scene_debug=debug,
            ), memory

        _enter_active_mode(memory, scene, active_solution, trigger, now)
        return _annotate_output(
            copy.deepcopy(active_solution),
            scene_status="active_cutout",
            scene_detail="cutout_triggered",
            scene_debug=debug,
        ), memory

    if bool(trigger["activation_ready"]) and not bool(safety["ok"]):
        return _build_prepare_step_output(
            context,
            memory,
            scene,
            now,
            reason="cutout_wait_safety",
            scene_status="prepare_wait_safety",
            scene_debug=debug,
            trigger=trigger,
        ), memory

    return _build_prepare_step_output(
        context,
        memory,
        scene,
        now,
        reason="cutout_prepare_wait",
        scene_status="prepare_wait_trigger",
        scene_debug=debug,
        trigger=trigger,
    ), memory


def _handle_active(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 ACTIVE：换道已开始，固定执行锁定目标，不再重算触发时机。"""

    target_line = _resolve_locked_target_line(context, memory)
    if target_line is None:
        _enter_cooldown(memory, now, copy.deepcopy(memory.get("last_debug")))
        return _build_keep_lane_output(
            context,
            reason="cutout_active_target_line_lost",
            scene_status="infeasible_geometry",
            scene_debug=copy.deepcopy(memory.get("last_debug")),
        ), memory

    active_output = _build_locked_active_output(context, memory, target_line, now)
    if active_output is None:
        _enter_cooldown(memory, now, copy.deepcopy(memory.get("last_debug")))
        return _build_keep_lane_output(
            context,
            reason="cutout_active_output_failed",
            scene_status="infeasible_geometry",
            scene_debug=copy.deepcopy(memory.get("last_debug")),
        ), memory

    if not bool(memory.get("metric_eval_recorded", False)) and float(now) >= float(memory.get("active_metric_eval_time_s", float("inf"))):
        metric_at_eval = _compute_finish_metric_from_memory(context, memory)
        metric_debug = copy.deepcopy(metric_at_eval.get("debug", {}))
        locked_lane_change_time = float(memory.get("locked_lane_change_time_s", CFG["lane_change_time_s"]))
        metric_debug.update(
            {
                "finish_metric_eval_phase": "mid_lane_change",
                "finish_metric_eval_time_s": _metric_eval_time_s(locked_lane_change_time),
                "finish_metric_lane_change_time_s": float(locked_lane_change_time),
                "finish_metric_eval_actual_elapsed_s": float(now) - float(memory.get("active_start_time_s", now)),
            }
        )
        metric_at_eval["debug"] = metric_debug
        memory["actual_finish_metric"] = copy.deepcopy(metric_at_eval)
        memory["metric_eval_recorded"] = True

    if _is_cutout_completed(context, memory, target_line, now):
        finish_metric = copy.deepcopy(memory.get("actual_finish_metric"))
        if not finish_metric:
            finish_metric = _compute_finish_metric_from_memory(context, memory)
            finish_debug = copy.deepcopy(finish_metric.get("debug", {}))
            locked_lane_change_time = float(memory.get("locked_lane_change_time_s", CFG["lane_change_time_s"]))
            finish_debug.update(
                {
                    "finish_metric_eval_phase": "completion_fallback",
                    "finish_metric_eval_time_s": _metric_eval_time_s(locked_lane_change_time),
                    "finish_metric_lane_change_time_s": float(locked_lane_change_time),
                    "finish_metric_eval_actual_elapsed_s": float(now) - float(memory.get("active_start_time_s", now)),
                }
            )
            finish_metric["debug"] = finish_debug
        memory["actual_finish_metric"] = copy.deepcopy(finish_metric)
        memory["mode"] = "STABILIZE"
        memory["stable_until_time_s"] = float(now) + float(CFG["stabilize_duration_s"])
        return _build_stabilize_output(
            context,
            target_line,
            reason="cutout_completed",
            scene_status=str(finish_metric["scene_status"]),
            scene_debug=finish_metric["debug"],
        ), memory

    return _annotate_output(
        active_output,
        scene_status="active_cutout",
        scene_detail="active_locked_execute",
        scene_debug=copy.deepcopy(memory.get("last_debug")),
    ), memory


def _handle_stabilize(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 STABILIZE：切出完成后短时保持目标车道。"""

    target_line = _resolve_locked_target_line(context, memory)
    finish_metric = memory.get("actual_finish_metric") or {}
    if target_line is None:
        memory["mode"] = "APPROACH"
        _reset_prepare_memory(memory)
        _reset_execute_memory(memory)
        return _build_keep_lane_output(
            context,
            reason="cutout_stabilize_line_lost",
            scene_status="baseline_keep_lane",
            scene_debug=copy.deepcopy(finish_metric.get("debug")),
        ), memory

    output = _build_stabilize_output(
        context,
        target_line,
        reason="cutout_stabilize",
        scene_status=str(finish_metric.get("scene_status", "cutout_stabilize")),
        scene_debug=copy.deepcopy(finish_metric.get("debug")),
    )
    if now >= float(memory["stable_until_time_s"]):
        memory["mode"] = "APPROACH"
        _reset_prepare_memory(memory)
        _reset_execute_memory(memory)
    return output, memory


def _handle_unknown_mode(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理未知状态：回到 APPROACH 并输出保底目标。"""

    memory["mode"] = "APPROACH"
    _reset_prepare_memory(memory)
    _reset_execute_memory(memory)
    return _build_keep_lane_output(
        context,
        reason="cutout_unknown_mode",
        scene_status="baseline_keep_lane",
    ), memory


# ==================== Transition Helpers / 状态切换与记忆管理 ====================


def _enter_prepare_mode(memory: dict[str, Any], target_line: ReferenceLineView, now: float) -> None:
    """进入 PREPARE，并锁定目标车道语义。"""

    memory["mode"] = "PREPARE"
    memory["prepare_start_time_s"] = float(now)
    memory["prepare_locked_target_lane_id"] = str(target_line.lane_id)
    memory["prepare_locked_target_lane_sequence"] = _normalize_lane_sequence(target_line.lane_sequence)


def _enter_active_mode(
    memory: dict[str, Any],
    scene: dict[str, Any],
    active_solution: dict[str, Any],
    trigger: dict[str, Any],
    now: float,
) -> None:
    """进入 ACTIVE，并冻结本次切出的目标线、终端点和预期指标。"""

    target_line: ReferenceLineView = scene["target_line"]
    source_line: ReferenceLineView = scene["source_line"]
    memory["mode"] = "ACTIVE"
    memory["active_start_time_s"] = float(now)
    lane_change_time = float(active_solution["t"])
    memory["active_end_time_s"] = float(now) + lane_change_time
    memory["active_metric_eval_time_s"] = float(now) + _metric_eval_time_s(lane_change_time)
    memory["locked_lane_change_time_s"] = lane_change_time
    memory["metric_eval_recorded"] = False
    memory["locked_target_line_index"] = int(target_line.line_index)
    memory["locked_target_lane_id"] = str(target_line.lane_id)
    memory["locked_target_lane_sequence"] = _normalize_lane_sequence(target_line.lane_sequence)
    memory["locked_source_lane_id"] = str(source_line.lane_id)
    memory["locked_source_lane_sequence"] = _normalize_lane_sequence(source_line.lane_sequence)
    memory["locked_source_line"] = np.asarray(source_line.line, dtype=float).copy()
    memory["locked_source_cumulative_s"] = np.asarray(source_line.cumulative_s, dtype=float).copy()
    memory["locked_exposed_ptc_id"] = str(scene["exposed"].ptc_id)
    memory["locked_metric_name"] = str(trigger["metric_name"])
    memory["locked_expected_metric"] = float(trigger["metric_value"])
    memory["locked_target_xy"] = [float(active_solution["x"]), float(active_solution["y"])]
    memory["locked_target_v"] = float(active_solution["v"])
    memory["locked_target_a"] = float(active_solution["a"])
    memory["locked_target_heading"] = float(active_solution["theta"])
    memory["execute_complete_hits"] = 0
    memory["actual_finish_metric"] = None
    _reset_prepare_memory(memory)


def _enter_cooldown(memory: dict[str, Any], now: float, debug: dict[str, Any] | None = None) -> None:
    """进入 COOLDOWN，并保存最后一次 debug 方便离线排查。"""

    memory["mode"] = "COOLDOWN"
    memory["cooldown_until_time_s"] = float(now) + float(CFG["abort_cooldown_s"])
    memory["last_debug"] = copy.deepcopy(debug)
    _reset_prepare_memory(memory)
    _reset_execute_memory(memory)


def _reset_prepare_memory(memory: dict[str, Any]) -> None:
    """清空 PREPARE 阶段缓存。"""

    memory["prepare_start_time_s"] = -1.0
    memory["prepare_locked_target_lane_id"] = ""
    memory["prepare_locked_target_lane_sequence"] = []


def _reset_execute_memory(memory: dict[str, Any]) -> None:
    """清空 ACTIVE/STABILIZE 阶段缓存。"""

    memory["active_start_time_s"] = -1.0
    memory["active_end_time_s"] = -1.0
    memory["active_metric_eval_time_s"] = -1.0
    memory["locked_lane_change_time_s"] = -1.0
    memory["metric_eval_recorded"] = False
    memory["locked_target_line_index"] = None
    memory["locked_target_lane_id"] = ""
    memory["locked_target_lane_sequence"] = []
    memory["locked_source_lane_id"] = ""
    memory["locked_source_lane_sequence"] = []
    memory["locked_source_line"] = None
    memory["locked_source_cumulative_s"] = None
    memory["locked_exposed_ptc_id"] = ""
    memory["locked_metric_name"] = ""
    memory["locked_expected_metric"] = float("inf")
    memory["locked_target_xy"] = None
    memory["locked_target_v"] = 0.0
    memory["locked_target_a"] = 0.0
    memory["locked_target_heading"] = 0.0
    memory["execute_complete_hits"] = 0
    memory["stable_until_time_s"] = -1.0
    memory["actual_finish_metric"] = None


def _reset_scene_memory() -> None:
    """把全局场景内存重置到初始值。"""

    global SCENE_MEMORY
    SCENE_MEMORY = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Scene / Role Helpers 三车角色与几何关系 ====================


def _build_cutout_scene(
    context: SceneContext,
    memory: dict[str, Any] | None = None,
    target_line_override: ReferenceLineView | None = None,
) -> dict[str, Any]:
    """构造 cut-out 一帧场景语义，包括 VUT、GVT、暴露目标和目标车道。"""

    ego = context.ego_vehicle
    main = context.main_vehicle
    limits = _read_vehicle_limits(context, exposed_actor=None)
    if main is None:
        return _scene_failure("no_main_vehicle", {"main_valid": False})

    exposed = _resolve_exposed_actor(context, memory)
    if exposed is None:
        return _scene_failure(
            "no_exposed_actor",
            {"exposed_ptc_id": _read_exposed_ptc_id(memory)},
        )
    limits = _read_vehicle_limits(context, exposed)

    if _heading_diff_deg(float(main.heading), float(ego.heading)) > float(CFG["max_heading_diff_deg"]):
        return _scene_failure("main_heading_mismatch", {"heading_diff_deg": _heading_diff_deg(main.heading, ego.heading)})
    if _heading_diff_deg(float(exposed.heading), float(ego.heading)) > float(CFG["max_heading_diff_deg"]):
        return _scene_failure("exposed_heading_mismatch", {"heading_diff_deg": _heading_diff_deg(exposed.heading, ego.heading)})

    target_line = target_line_override
    if target_line is None:
        target_line = _resolve_prepare_target_line(context, memory) if memory is not None else None
    if target_line is None:
        target_line = _choose_cutout_target_line(context)
    if target_line is None:
        return _scene_failure("no_cutout_target_line", {"cutout_side": str(CFG["cutout_side"])})

    source_line = _extend_source_relation_line_to_exposed(_build_source_relation_line(context), exposed)
    geometry = _compute_source_geometry(context, exposed, limits, source_line)
    if not bool(geometry["same_source_lane"]):
        return _scene_failure("not_same_source_lane", geometry)
    if not bool(geometry["ordering_ok"]):
        return _scene_failure("invalid_cutout_order", geometry)
    if float(geometry["gvt_to_exposed_gap_m"]) < float(CFG["source_abort_gap_m"]):
        return _scene_failure("gvt_too_close_to_exposed", geometry)

    target_lane_check = _check_target_lane_hard_gap(context, target_line, exposed)
    debug = copy.deepcopy(geometry)
    debug.update(copy.deepcopy(target_lane_check["debug"]))
    debug["target_line_index"] = int(target_line.line_index)
    debug["target_lane_id"] = str(target_line.lane_id)
    debug["exposed_ptc_id"] = str(exposed.ptc_id)
    debug["exposed_v_mps"] = float(exposed.v)
    return {
        "ok": True,
        "reason": "ok",
        "ego": ego,
        "main": main,
        "exposed": exposed,
        "source_line": source_line,
        "target_line": target_line,
        "limits": limits,
        "geometry": geometry,
        "target_lane_check": target_lane_check,
        "debug": debug,
    }


def _scene_failure(reason: str, debug: dict[str, Any]) -> dict[str, Any]:
    """构造统一的场景解析失败结果。"""

    return {
        "ok": False,
        "reason": str(reason),
        "debug": copy.deepcopy(debug),
    }


def _build_source_relation_line(context: SceneContext) -> ReferenceLineView:
    """构造源车道关系线。

    cut-out 的 VUT 通常位于 GVT 后方，而 GVT 当前参考线是从 GVT 当前位置向前截取的。
    如果直接用当前前向参考线投影，后方 VUT 可能被压到线起点，三车顺序会失真。
    因此优先使用 ego 的完整车道中心线；若当前参考线包含后继车道段，则把后继段拼到
    完整当前车道后方，使暴露目标可以位于“当前车道 + 后继车道”的源车道走廊上。
    """

    ego = context.ego_vehicle
    lane_centerline = getattr(ego, "lane_centerline", None)
    if lane_centerline is None:
        return ego.current_ref_line

    line = np.asarray(lane_centerline, dtype=float)
    if line.ndim != 2 or len(line) < 2:
        return ego.current_ref_line

    line_xy = np.asarray(line[:, :2], dtype=float)
    forward_line = np.asarray(ego.current_ref_line.line, dtype=float)
    if forward_line.ndim == 2 and len(forward_line) >= 2 and len(_normalize_lane_sequence(ego.current_ref_line.lane_sequence)) > 1:
        current_lane_end = line_xy[-1, :2]
        distances = np.linalg.norm(forward_line[:, :2] - current_lane_end, axis=1)
        append_start_idx = int(np.argmin(distances))
        if float(distances[append_start_idx]) < float(CFG["source_lane_match_d_thresh_m"]):
            successor_points = np.asarray(forward_line[append_start_idx:, :2], dtype=float)
            if len(successor_points) >= 2:
                if np.linalg.norm(successor_points[0, :2] - current_lane_end) < 1e-6:
                    successor_points = successor_points[1:, :2]
                if len(successor_points) > 0:
                    line_xy = np.vstack([line_xy, successor_points])

    cumulative_s = calculate_path_cumulative_distance(line_xy)
    _, current_s, current_d = project_point_to_path(np.array([ego.x, ego.y], dtype=float), line_xy)
    return ReferenceLineView(
        line_index=int(ego.current_ref_line.line_index),
        lane_id=str(ego.current_ref_line.lane_id),
        lane_sequence=_normalize_lane_sequence(ego.current_ref_line.lane_sequence),
        lateral_step=0,
        line=line_xy,
        cumulative_s=cumulative_s,
        current_s=float(current_s),
        current_d=float(current_d),
    )


def _extend_source_relation_line_to_exposed(
    source_line: ReferenceLineView,
    exposed: PlanningActorState,
) -> ReferenceLineView:
    """必要时把源车道关系线延伸到后继段上的暴露目标。

    当前源线可能已经拼了后继段，但在后继段末端附近仍会出现投影被夹到线尾的情况。
    只有暴露目标位于线尾前方、且相对线尾切线横向偏差仍很小时，才追加该点。
    """

    line_xy = np.asarray(source_line.line, dtype=float)
    if line_xy.ndim != 2 or len(line_xy) < 2:
        return source_line

    exposed_xy = np.array([exposed.x, exposed.y], dtype=float)
    _, _exposed_s, exposed_d = project_point_to_path(exposed_xy, line_xy)
    if abs(float(exposed_d)) <= float(CFG["source_lane_match_d_thresh_m"]):
        return source_line

    tail = line_xy[-1, :2]
    prev = line_xy[-2, :2]
    tail_vec = tail - prev
    tail_len = float(np.linalg.norm(tail_vec))
    if tail_len < 1e-6:
        return source_line

    tail_dir = tail_vec / tail_len
    exposed_vec = exposed_xy - tail
    forward_dist = float(np.dot(exposed_vec, tail_dir))
    lateral_vec = exposed_vec - forward_dist * tail_dir
    lateral_dist = float(np.linalg.norm(lateral_vec))
    if forward_dist <= 0.0 or lateral_dist > float(CFG["source_lane_match_d_thresh_m"]):
        return source_line

    extended_line = np.vstack([line_xy, exposed_xy.reshape(1, 2)])
    cumulative_s = calculate_path_cumulative_distance(extended_line)
    return ReferenceLineView(
        line_index=int(source_line.line_index),
        lane_id=str(source_line.lane_id),
        lane_sequence=_normalize_lane_sequence(source_line.lane_sequence),
        lateral_step=int(source_line.lateral_step),
        line=extended_line,
        cumulative_s=cumulative_s,
        current_s=float(source_line.current_s),
        current_d=float(source_line.current_d),
    )


def _resolve_exposed_actor(
    context: SceneContext,
    memory: dict[str, Any] | None,
) -> PlanningActorState | None:
    """从现有 obstacles 中解析暴露目标，不新增数据输入通道。"""

    frame = context.planning_frame
    if frame is None:
        return None

    exposed_ptc_id = _read_exposed_ptc_id(memory)
    actors = [
        actor_state
        for obstacle in frame.obstacles
        for actor_state in [build_planning_actor_state(obstacle)]
        if actor_state is not None
    ]

    if exposed_ptc_id:
        for actor in actors:
            if str(actor.ptc_id) == str(exposed_ptc_id):
                return actor
        return None

    # 兜底逻辑只服务离线调试：未配置 exposed_ptc_id 时，选源车道上 GVT 前方最近目标。
    source_line = context.ego_vehicle.current_ref_line
    ego_xy = np.array([context.ego_vehicle.x, context.ego_vehicle.y], dtype=float)
    _, ego_s, _ = project_point_to_path(ego_xy, source_line.line)
    best_actor = None
    best_gap = float("inf")
    for actor in actors:
        if context.main_vehicle is not None and str(actor.ptc_id) == str(getattr(context.main_vehicle, "ptc_id", "")):
            continue
        actor_xy = np.array([actor.x, actor.y], dtype=float)
        _, actor_s, actor_d = project_point_to_path(actor_xy, source_line.line)
        if abs(float(actor_d)) > float(CFG["source_lane_match_d_thresh_m"]):
            continue
        gap = float(actor_s - ego_s)
        if gap > 0.0 and gap < best_gap:
            best_gap = gap
            best_actor = actor
    return best_actor


def _read_exposed_ptc_id(memory: dict[str, Any] | None) -> str:
    """读取暴露目标 ptc_id，优先运行期环境变量，其次锁定缓存和 CFG。"""

    env_value = _env_str("ROS_SCENE_EXPOSED_PTC_ID", "")
    if env_value:
        return env_value
    if memory is not None and str(memory.get("locked_exposed_ptc_id", "")):
        return str(memory.get("locked_exposed_ptc_id", ""))
    return str(CFG.get("exposed_ptc_id", ""))


def _compute_source_geometry(
    context: SceneContext,
    exposed: PlanningActorState,
    limits: VehicleLimits,
    source_line: ReferenceLineView,
) -> dict[str, Any]:
    """计算 VUT->GVT->暴露目标在源车道上的几何顺序。"""

    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        return {
            "same_source_lane": False,
            "ordering_ok": False,
        }

    _, vut_s, vut_d = project_point_to_path(np.array([main.x, main.y], dtype=float), source_line.line)
    _, gvt_s, gvt_d = project_point_to_path(np.array([ego.x, ego.y], dtype=float), source_line.line)
    _, exposed_s, exposed_d = project_point_to_path(np.array([exposed.x, exposed.y], dtype=float), source_line.line)

    vut_to_gvt_gap = float(gvt_s) - float(vut_s) - 0.5 * (limits.main_length + limits.ego_length)
    gvt_to_exposed_gap = float(exposed_s) - float(gvt_s) - 0.5 * (limits.exposed_length + limits.ego_length)
    same_source_lane = bool(
        abs(float(vut_d)) <= float(CFG["source_lane_match_d_thresh_m"])
        and abs(float(gvt_d)) <= float(CFG["source_lane_match_d_thresh_m"])
        and abs(float(exposed_d)) <= float(CFG["source_lane_match_d_thresh_m"])
    )
    ordering_ok = bool(vut_to_gvt_gap > 0.0 and gvt_to_exposed_gap > 0.0)

    return {
        "same_source_lane": bool(same_source_lane),
        "ordering_ok": bool(ordering_ok),
        "vut_s_on_source": float(vut_s),
        "gvt_s_on_source": float(gvt_s),
        "exposed_s_on_source": float(exposed_s),
        "vut_d_on_source": float(vut_d),
        "gvt_d_on_source": float(gvt_d),
        "exposed_d_on_source": float(exposed_d),
        "vut_to_gvt_gap_m": float(vut_to_gvt_gap),
        "gvt_to_exposed_gap_m": float(gvt_to_exposed_gap),
    }


# ==================== Target Lane / 目标车道辅助函数 ====================


def _choose_cutout_target_line(context: SceneContext) -> ReferenceLineView | None:
    """按显式 left/right 配置选择一阶相邻切出目标车道。"""

    side = str(_env_str("ROS_SCENE_CUTOUT_SIDE", str(CFG["cutout_side"]))).strip().lower()
    if side not in {"left", "right"}:
        return None
    desired_step = -1 if side == "left" else 1

    ego = context.ego_vehicle
    best_line = None
    best_score = float("inf")
    for line in ego.candidate_ref_lines:
        if int(line.lateral_step) != desired_step:
            continue

        _, ego_s_on_line, ego_d_on_line = project_point_to_path(np.array([ego.x, ego.y], dtype=float), line.line)
        if abs(float(ego_d_on_line)) > float(CFG["max_ego_to_target_line_d_m"]):
            continue
        remaining_after_target = float(line.cumulative_s[-1]) - float(ego_s_on_line)
        if remaining_after_target < float(CFG["post_cutout_buffer_m"]):
            continue

        score = abs(float(ego_d_on_line)) + 0.01 * max(0.0, -remaining_after_target)
        if score < best_score:
            best_score = score
            best_line = line
    return best_line


def _resolve_prepare_target_line(
    context: SceneContext,
    memory: dict[str, Any] | None,
) -> ReferenceLineView | None:
    """恢复 PREPARE 锁定的目标车道。"""

    if memory is None:
        return None
    line = _line_by_lane_sequence(
        context.ego_vehicle.candidate_ref_lines,
        memory.get("prepare_locked_target_lane_sequence", []),
    )
    if line is not None:
        return line
    line = _line_by_lane_id(context.ego_vehicle.candidate_ref_lines, memory.get("prepare_locked_target_lane_id", ""))
    if line is not None:
        return line
    return None


def _resolve_locked_target_line(
    context: SceneContext,
    memory: dict[str, Any],
) -> ReferenceLineView | None:
    """恢复 ACTIVE/STABILIZE 锁定的目标车道。"""

    line = _line_by_lane_sequence(
        context.ego_vehicle.candidate_ref_lines,
        memory.get("locked_target_lane_sequence", []),
    )
    if line is not None:
        return line
    line = _line_by_lane_id(context.ego_vehicle.candidate_ref_lines, memory.get("locked_target_lane_id", ""))
    if line is not None:
        return line
    return _line_by_index(context.ego_vehicle.candidate_ref_lines, memory.get("locked_target_line_index"))


def _check_target_lane_hard_gap(
    context: SceneContext,
    target_line: ReferenceLineView,
    exposed: PlanningActorState,
) -> dict[str, Any]:
    """检查目标车道前后车硬距离。

    说明：
    - 第一版测试默认目标车道无车；
    - 因此这里不计算 THW/TTC，只在存在明显近距离障碍物时拦截。
    """

    if str(CFG["target_lane_obstacle_check_mode"]).lower() == "off":
        return {"ok": True, "debug": {"target_lane_obstacle_check": "off"}}

    frame = context.planning_frame
    if frame is None:
        return {"ok": True, "debug": {"target_lane_obstacle_check": "no_frame"}}

    ego = context.ego_vehicle
    _, ego_s_on_target, _ = project_point_to_path(np.array([ego.x, ego.y], dtype=float), target_line.line)
    front_gap = float("inf")
    rear_gap = float("inf")
    front_id = ""
    rear_id = ""

    for raw_obstacle in frame.obstacles:
        actor = build_planning_actor_state(raw_obstacle)
        if actor is None:
            continue
        if context.main_vehicle is not None and str(actor.ptc_id) == str(getattr(context.main_vehicle, "ptc_id", "")):
            continue
        if str(actor.ptc_id) == str(exposed.ptc_id):
            continue

        _, obs_s, obs_d = project_point_to_path(np.array([actor.x, actor.y], dtype=float), target_line.line)
        if abs(float(obs_d)) > float(CFG["target_lane_obstacle_d_thresh_m"]):
            continue
        rel_s = float(obs_s) - float(ego_s_on_target)
        if rel_s >= 0.0 and rel_s < front_gap:
            front_gap = float(rel_s)
            front_id = str(actor.ptc_id)
        if rel_s < 0.0 and abs(rel_s) < rear_gap:
            rear_gap = float(abs(rel_s))
            rear_id = str(actor.ptc_id)

    front_ok = bool(front_gap >= float(CFG["target_lane_front_min_gap_m"]))
    rear_ok = bool(rear_gap >= float(CFG["target_lane_rear_min_gap_m"]))
    return {
        "ok": bool(front_ok and rear_ok),
        "debug": {
            "target_lane_obstacle_check": str(CFG["target_lane_obstacle_check_mode"]),
            "target_lane_front_gap_m": float(front_gap),
            "target_lane_rear_gap_m": float(rear_gap),
            "target_lane_front_id": str(front_id),
            "target_lane_rear_id": str(rear_id),
            "target_lane_front_ok": bool(front_ok),
            "target_lane_rear_ok": bool(rear_ok),
        },
    }


# ==================== THW/TTC Solver / 指标解算 ====================


def _evaluate_cutout_trigger(context: SceneContext, scene: dict[str, Any]) -> dict[str, Any]:
    """搜索 PREPARE 当前帧是否应该开始切出。

    核心原则：
    - 只用 VUT->暴露目标的 THW/TTC 做触发；
    - GVT->VUT、GVT->目标车道前后车只做安全检查，不参与触发指标。
    """

    metric_name = _select_metric_name(scene["exposed"])
    target_metric = float(CFG["target_thw_s"] if metric_name == "THW" else CFG["target_ttc_s"])
    tolerance = float(CFG["thw_tol_s"] if metric_name == "THW" else CFG["ttc_tol_s"])
    hard_min = float(CFG["hard_min_thw_s"] if metric_name == "THW" else CFG["hard_min_ttc_s"])
    metric_base = _build_vut_exposed_metric_base(context, scene)

    best_candidate = None
    best_cost = float("inf")
    first_metric = None
    lane_change_time = _dynamic_lane_change_time_s(context, scene["limits"])
    metric_eval_time = _metric_eval_time_s(lane_change_time)
    for hold_t in CFG["hold_t_grid"]:
        total_t = float(hold_t) + float(metric_eval_time)
        metric = _compute_vut_exposed_metric_from_base(metric_base, total_t)
        if first_metric is None:
            first_metric = metric
        if not bool(metric["valid"]):
            continue
        metric_value = float(metric[metric_name.lower()])
        if metric_value < hard_min:
            continue
        error = abs(metric_value - target_metric)
        if error <= tolerance:
            cost = float(error) + 0.001 * float(hold_t)
            if cost < best_cost:
                best_cost = cost
                best_candidate = {
                    "hold_t": float(hold_t),
                    "total_t": float(total_t),
                    "metric": metric,
                    "metric_value": float(metric_value),
                    "metric_error": float(metric_value - target_metric),
                }

    immediate_metric_value = float("inf")
    immediate_metric_error = float("inf")
    immediate_hard_safe = False
    immediate_hit = False
    if first_metric is not None and bool(first_metric["valid"]):
        immediate_metric_value = float(first_metric[metric_name.lower()])
        immediate_metric_error = float(immediate_metric_value - target_metric)
        immediate_hard_safe = bool(immediate_metric_value >= hard_min)
        immediate_hit = bool(immediate_hard_safe and abs(immediate_metric_error) <= tolerance)

    late_abort = bool(
        best_candidate is None
        and np.isfinite(immediate_metric_value)
        and immediate_metric_value < target_metric - tolerance
    )
    activation_ready = bool(
        best_candidate is not None
        and float(best_candidate["hold_t"]) <= float(CFG["activation_hold_t_max_s"])
        and immediate_hit
    )

    debug = {
        "trigger_metric_name": str(metric_name),
        "trigger_target_metric_s": float(target_metric),
        "trigger_tolerance_s": float(tolerance),
        "trigger_hard_min_s": float(hard_min),
        "trigger_lane_change_time_s": float(lane_change_time),
        "trigger_metric_eval_time_s": float(metric_eval_time),
        "trigger_activation_ready": bool(activation_ready),
        "trigger_late_abort": bool(late_abort),
        "trigger_best_hold_t_s": float("inf") if best_candidate is None else float(best_candidate["hold_t"]),
        "trigger_best_total_t_s": float("inf") if best_candidate is None else float(best_candidate["total_t"]),
        "trigger_best_metric_s": float("inf") if best_candidate is None else float(best_candidate["metric_value"]),
        "trigger_best_metric_error_s": float("inf") if best_candidate is None else float(best_candidate["metric_error"]),
        "trigger_immediate_metric_s": float(immediate_metric_value),
        "trigger_immediate_metric_error_s": float(immediate_metric_error),
        "trigger_immediate_metric_hit": bool(immediate_hit),
        "trigger_immediate_hard_safe": bool(immediate_hard_safe),
        "trigger_first_metric": copy.deepcopy(first_metric),
    }
    if best_candidate is not None:
        debug["trigger_best_metric_detail"] = copy.deepcopy(best_candidate["metric"])

    return {
        "activation_ready": bool(activation_ready),
        "late_abort": bool(late_abort),
        "reason": "ready" if activation_ready else ("late" if late_abort else "wait"),
        "metric_name": str(metric_name),
        "target_metric": float(target_metric),
        "tolerance": float(tolerance),
        "metric_value": float("inf") if best_candidate is None else float(best_candidate["metric_value"]),
        "hold_t": float("inf") if best_candidate is None else float(best_candidate["hold_t"]),
        "total_t": float("inf") if best_candidate is None else float(best_candidate["total_t"]),
        "lane_change_time_s": float(lane_change_time),
        "metric_eval_time_s": float(metric_eval_time),
        "debug": debug,
    }


def _compute_vut_exposed_metric(
    context: SceneContext,
    scene: dict[str, Any],
    total_t: float,
) -> dict[str, Any]:
    """预测 total_t 后 VUT 与暴露目标之间的 THW/TTC。"""

    main = context.main_vehicle
    if main is None:
        return {"valid": False, "reason": "no_main_vehicle"}

    source_line: ReferenceLineView = scene["source_line"]
    exposed: PlanningActorState = scene["exposed"]
    limits: VehicleLimits = scene["limits"]
    metric = _compute_pair_metric_on_line(
        follower_xy=np.array([main.x, main.y], dtype=float),
        follower_v=float(main.v),
        follower_a=float(main.a),
        follower_length=float(limits.main_length),
        leader_xy=np.array([exposed.x, exposed.y], dtype=float),
        leader_v=float(exposed.v),
        leader_a=float(exposed.a),
        leader_length=float(limits.exposed_length),
        line=source_line.line,
        cumulative_s=source_line.cumulative_s,
        total_t=float(total_t),
        freeze_leader=bool(_select_metric_name(exposed) == "THW"),
    )

    metric_name = _select_metric_name(exposed)
    if metric_name == "THW":
        valid = bool(
            float(metric["net_gap_m"]) > 0.0
            and float(metric["follower_v_pred"]) >= float(CFG["vut_metric_min_speed_mps"])
            and np.isfinite(float(metric["thw"]))
        )
    else:
        valid = bool(
            float(metric["net_gap_m"]) > 0.0
            and float(metric["closing_speed_mps"]) > 1e-3
            and np.isfinite(float(metric["ttc"]))
        )
    metric["valid"] = bool(valid)
    metric["metric_name"] = str(metric_name)
    metric["reason"] = "ok" if valid else "invalid_metric"
    return metric


def _build_vut_exposed_metric_base(context: SceneContext, scene: dict[str, Any]) -> dict[str, Any]:
    """预计算 VUT->暴露目标投影，避免 PREPARE 搜索 hold_t 时重复投影。"""

    source_line: ReferenceLineView = scene["source_line"]
    exposed = scene["exposed"]
    limits: VehicleLimits = scene["limits"]
    main = context.main_vehicle
    if main is None:
        return {"valid_base": False, "reason": "no_main_vehicle"}

    _, follower_s, follower_d = project_point_to_path(np.array([main.x, main.y], dtype=float), source_line.line)
    _, leader_s, leader_d = project_point_to_path(np.array([exposed.x, exposed.y], dtype=float), source_line.line)
    metric_name = _select_metric_name(exposed)
    return {
        "valid_base": True,
        "metric_name": str(metric_name),
        "freeze_leader": bool(metric_name == "THW"),
        "line_end": float(source_line.cumulative_s[-1]) - 0.5,
        "follower_s": float(follower_s),
        "follower_d": float(follower_d),
        "follower_v": float(main.v),
        "follower_a": float(main.a),
        "follower_length": float(limits.main_length),
        "leader_s": float(leader_s),
        "leader_d": float(leader_d),
        "leader_v": float(exposed.v),
        "leader_a": float(exposed.a),
        "leader_length": float(limits.exposed_length),
    }


def _compute_vut_exposed_metric_from_base(base: dict[str, Any], total_t: float) -> dict[str, Any]:
    """基于已投影的 VUT/exposed 状态计算一个未来时刻的 THW/TTC。"""

    if not bool(base.get("valid_base", False)):
        return {
            "total_t_s": float(total_t),
            "valid": False,
            "metric_name": str(base.get("metric_name", "")),
            "reason": str(base.get("reason", "invalid_base")),
            "thw": float("inf"),
            "ttc": float("inf"),
            "net_gap_m": float("inf"),
            "closing_speed_mps": 0.0,
        }

    follower_a_use = _clamp(float(base["follower_a"]), float(CFG["prediction_min_acc"]), float(CFG["prediction_max_acc"]))
    leader_a_use = 0.0 if bool(base["freeze_leader"]) else _clamp(
        float(base["leader_a"]),
        float(CFG["prediction_min_acc"]),
        float(CFG["prediction_max_acc"]),
    )
    follower_s_pred = float(base["follower_s"]) + float(base["follower_v"]) * float(total_t) + 0.5 * follower_a_use * float(total_t) ** 2
    follower_v_pred = max(0.0, float(base["follower_v"]) + follower_a_use * float(total_t))
    if bool(base["freeze_leader"]):
        leader_s_pred = float(base["leader_s"])
        leader_v_pred = 0.0
    else:
        leader_s_pred = float(base["leader_s"]) + float(base["leader_v"]) * float(total_t) + 0.5 * leader_a_use * float(total_t) ** 2
        leader_v_pred = max(0.0, float(base["leader_v"]) + leader_a_use * float(total_t))

    line_end = float(base["line_end"])
    follower_s_pred = _clamp(follower_s_pred, 0.0, line_end)
    leader_s_pred = _clamp(leader_s_pred, 0.0, line_end)
    net_gap = float(leader_s_pred) - float(follower_s_pred) - 0.5 * (float(base["leader_length"]) + float(base["follower_length"]))
    closing_speed = float(follower_v_pred) - float(leader_v_pred)
    thw_gap = float(net_gap) + float(base["leader_length"])
    thw = thw_gap / max(float(follower_v_pred), 1e-6) if thw_gap > 0.0 else float("inf")
    ttc = float(net_gap) / float(closing_speed) if net_gap > 0.0 and closing_speed > 1e-6 else float("inf")

    metric_name = str(base["metric_name"])
    if metric_name == "THW":
        valid = bool(net_gap > 0.0 and follower_v_pred >= float(CFG["vut_metric_min_speed_mps"]) and np.isfinite(float(thw)))
    else:
        valid = bool(net_gap > 0.0 and closing_speed > 1e-3 and np.isfinite(float(ttc)))

    return {
        "total_t_s": float(total_t),
        "follower_s_now": float(base["follower_s"]),
        "leader_s_now": float(base["leader_s"]),
        "follower_d": float(base["follower_d"]),
        "leader_d": float(base["leader_d"]),
        "follower_s_pred": float(follower_s_pred),
        "leader_s_pred": float(leader_s_pred),
        "follower_v_pred": float(follower_v_pred),
        "leader_v_pred": float(leader_v_pred),
        "net_gap_m": float(net_gap),
        "thw_effective_gap_m": float(thw_gap),
        "closing_speed_mps": float(closing_speed),
        "thw": float(thw),
        "ttc": float(ttc),
        "valid": bool(valid),
        "metric_name": str(metric_name),
        "reason": "ok" if valid else "invalid_metric",
    }


def _compute_pair_metric_on_line(
    *,
    follower_xy: np.ndarray,
    follower_v: float,
    follower_a: float,
    follower_length: float,
    leader_xy: np.ndarray,
    leader_v: float,
    leader_a: float,
    leader_length: float,
    line: np.ndarray,
    cumulative_s: np.ndarray,
    total_t: float,
    freeze_leader: bool = False,
) -> dict[str, Any]:
    """在同一参考线上计算 follower->leader 的预测 THW/TTC。"""

    _, follower_s, follower_d = project_point_to_path(follower_xy, line)
    _, leader_s, leader_d = project_point_to_path(leader_xy, line)
    follower_a_use = _clamp(float(follower_a), float(CFG["prediction_min_acc"]), float(CFG["prediction_max_acc"]))
    leader_a_use = 0.0 if freeze_leader else _clamp(float(leader_a), float(CFG["prediction_min_acc"]), float(CFG["prediction_max_acc"]))

    follower_s_pred = float(follower_s) + float(follower_v) * float(total_t) + 0.5 * follower_a_use * float(total_t) ** 2
    follower_v_pred = max(0.0, float(follower_v) + follower_a_use * float(total_t))
    if freeze_leader:
        leader_s_pred = float(leader_s)
        leader_v_pred = 0.0
    else:
        leader_s_pred = float(leader_s) + float(leader_v) * float(total_t) + 0.5 * leader_a_use * float(total_t) ** 2
        leader_v_pred = max(0.0, float(leader_v) + leader_a_use * float(total_t))

    line_end = float(cumulative_s[-1]) - 0.5
    follower_s_pred = _clamp(follower_s_pred, 0.0, line_end)
    leader_s_pred = _clamp(leader_s_pred, 0.0, line_end)

    net_gap = float(leader_s_pred) - float(follower_s_pred) - 0.5 * (float(leader_length) + float(follower_length))
    closing_speed = float(follower_v_pred) - float(leader_v_pred)
    thw_gap = float(net_gap) + float(leader_length)
    thw = thw_gap / max(float(follower_v_pred), 1e-6) if thw_gap > 0.0 else float("inf")
    ttc = float(net_gap) / float(closing_speed) if net_gap > 0.0 and closing_speed > 1e-6 else float("inf")

    return {
        "total_t_s": float(total_t),
        "follower_s_now": float(follower_s),
        "leader_s_now": float(leader_s),
        "follower_d": float(follower_d),
        "leader_d": float(leader_d),
        "follower_s_pred": float(follower_s_pred),
        "leader_s_pred": float(leader_s_pred),
        "follower_v_pred": float(follower_v_pred),
        "leader_v_pred": float(leader_v_pred),
        "net_gap_m": float(net_gap),
        "thw_effective_gap_m": float(thw_gap),
        "closing_speed_mps": float(closing_speed),
        "thw": float(thw),
        "ttc": float(ttc),
    }


def _select_metric_name(exposed: PlanningActorState) -> str:
    """根据暴露目标速度选择 THW 或 TTC 指标。"""

    return "THW" if float(exposed.v) <= float(CFG["static_speed_eps_mps"]) else "TTC"


# ==================== Safety Checks / 安全与可执行性检查 ====================


def _evaluate_cutout_safety(context: SceneContext, scene: dict[str, Any]) -> dict[str, Any]:
    """检查 GVT 在当前时刻是否具备切出可执行性。"""

    geometry = scene["geometry"]
    target_lane_check = scene["target_lane_check"]
    source_gap_ok = bool(float(geometry["gvt_to_exposed_gap_m"]) >= float(CFG["min_gvt_to_exposed_gap_m"]))
    target_lane_ok = bool(target_lane_check["ok"])
    lane_change_time = _dynamic_lane_change_time_s(context, scene["limits"])
    release_check = _check_pre_active_source_clearance(context, scene)
    output_feasible = _build_active_solution(
        context,
        scene,
        trigger={
            "metric_name": _select_metric_name(scene["exposed"]),
            "metric_value": float("inf"),
            "target_metric": float("inf"),
            "hold_t": 0.0,
            "total_t": _metric_eval_time_s(lane_change_time),
            "lane_change_time_s": float(lane_change_time),
        },
        debug={},
        dry_run=True,
    ) is not None
    ok = bool(source_gap_ok and target_lane_ok and release_check["ok"] and output_feasible)
    return {
        "ok": bool(ok),
        "debug": {
            "safety_source_gap_ok": bool(source_gap_ok),
            "safety_target_lane_ok": bool(target_lane_ok),
            **copy.deepcopy(release_check["debug"]),
            "safety_active_output_feasible": bool(output_feasible),
            "safety_decision_ok": bool(ok),
        },
    }


def _check_pre_active_source_clearance(context: SceneContext, scene: dict[str, Any]) -> dict[str, Any]:
    """进入 ACTIVE 前检查源车道纵向释放窗口。

    该检查不是完整轨迹碰撞检测，只判断 GVT 开始横向脱离源车道前，
    按当前纵向速度趋势是否会追近暴露目标到硬安全阈值以下。
    """

    geometry = scene["geometry"]
    ego = context.ego_vehicle
    exposed = scene["exposed"]
    gap_now = float(geometry.get("gvt_to_exposed_gap_m", float("inf")))
    if not np.isfinite(gap_now):
        return {
            "ok": False,
            "debug": {
                "safety_source_release_ok": False,
                "safety_source_release_reason": "invalid_gap",
            },
        }

    lane_change_time = _dynamic_lane_change_time_s(context, scene["limits"])
    release_t = _compute_source_release_time_s(lane_change_time)
    closing_speed = max(0.0, float(ego.v) - max(0.0, float(exposed.v)))
    predicted_min_gap = float(gap_now) - float(closing_speed) * float(release_t)
    min_gap = float(CFG["source_abort_gap_m"])
    ok = bool(predicted_min_gap >= min_gap)
    return {
        "ok": bool(ok),
        "debug": {
            "safety_source_release_ok": bool(ok),
            "safety_lane_change_time_s": float(lane_change_time),
            "safety_source_release_time_s": float(release_t),
            "safety_source_release_gap_now_m": float(gap_now),
            "safety_source_release_predicted_min_gap_m": float(predicted_min_gap),
            "safety_source_release_min_gap_m": float(min_gap),
            "safety_source_release_closing_speed_mps": float(closing_speed),
            "safety_source_release_reason": "ok" if ok else "predicted_gap_too_small",
        },
    }


def _scene_cruise_target_v(context: SceneContext, limits: VehicleLimits) -> float:
    """返回场景巡航速度上限，优先跟随 VUT 主车当前速度。"""

    main = context.main_vehicle
    if main is None:
        return _clamp(float(CFG["cruise_target_v"]), 0.0, limits.max_v)
    return _clamp(float(main.v), 0.0, limits.max_v)


def _dynamic_lane_change_time_s(context: SceneContext, limits: VehicleLimits) -> float:
    """按场景巡航速度计算本帧换道时间，低速长、高速短。"""

    cruise_v = _scene_cruise_target_v(context, limits)
    low_speed = float(CFG["lane_change_time_low_speed_mps"])
    high_speed = float(CFG["lane_change_time_high_speed_mps"])
    low_time = max(float(CFG["lane_change_time_s"]), float(CFG["lane_change_time_min_s"]))
    high_time = min(float(CFG["lane_change_time_s"]), float(CFG["lane_change_time_min_s"]))
    if high_speed <= low_speed + 1e-6:
        return high_time if cruise_v >= high_speed else low_time
    ratio = _clamp((float(cruise_v) - low_speed) / (high_speed - low_speed), 0.0, 1.0)
    return low_time + ratio * (high_time - low_time)


def _compute_source_release_time_s(lane_change_time_s: float | None = None) -> float:
    """按换道时间估算 GVT 基本脱离源车道追尾风险区的时间。"""

    lane_change_time = float(CFG["lane_change_time_s"] if lane_change_time_s is None else lane_change_time_s)
    release_t = min(
        lane_change_time * float(CFG["source_release_time_ratio"]),
        float(CFG["source_release_time_max_s"]),
    )
    return max(float(CFG["source_release_time_min_s"]), float(release_t))


def _metric_eval_time_s(lane_change_time_s: float | None = None) -> float:
    """返回 ACTIVE 开始后 VUT->暴露目标指标的评估时刻。"""

    ratio = max(0.0, min(float(CFG["metric_eval_time_ratio"]), 1.0))
    lane_change_time = float(CFG["lane_change_time_s"] if lane_change_time_s is None else lane_change_time_s)
    return lane_change_time * float(ratio)


def _compute_finish_metric(context: SceneContext, scene: dict[str, Any]) -> dict[str, Any]:
    """切出完成时计算实际 VUT->暴露目标指标，并给出成功/失败标签。"""

    if not bool(scene.get("ok", False)):
        return {
            "scene_status": "cutout_finish_metric_unavailable",
            "debug": {"finish_metric_reason": str(scene.get("reason", "scene_unavailable"))},
        }

    metric_name = _select_metric_name(scene["exposed"])
    metric = _compute_vut_exposed_metric(context, scene, total_t=0.0)
    target_metric = float(CFG["target_thw_s"] if metric_name == "THW" else CFG["target_ttc_s"])
    tolerance = float(CFG["thw_tol_s"] if metric_name == "THW" else CFG["ttc_tol_s"])
    hard_min = float(CFG["hard_min_thw_s"] if metric_name == "THW" else CFG["hard_min_ttc_s"])
    metric_value = float(metric[metric_name.lower()]) if bool(metric.get("valid", False)) else float("inf")
    hit = bool(abs(metric_value - target_metric) <= tolerance)
    hard_safe = bool(metric_value >= hard_min)
    status = "cutout_success" if hit and hard_safe else ("cutout_metric_miss" if hard_safe else "cutout_hard_metric_fail")
    debug = copy.deepcopy(scene["debug"])
    debug.update(
        {
            "finish_metric_name": str(metric_name),
            "finish_metric_value_s": float(metric_value),
            "finish_target_metric_s": float(target_metric),
            "finish_metric_error_s": float(metric_value - target_metric),
            "finish_metric_tolerance_s": float(tolerance),
            "finish_hard_min_s": float(hard_min),
            "finish_metric_hit": bool(hit),
            "finish_hard_safe": bool(hard_safe),
            "finish_metric_detail": copy.deepcopy(metric),
        }
    )
    return {"scene_status": status, "debug": debug}


def _compute_finish_metric_from_memory(
    context: SceneContext,
    memory: dict[str, Any],
) -> dict[str, Any]:
    """ACTIVE 完成后计算 VUT->暴露目标实际指标。

    GVT 完成切出后已经不在源车道，此时不能再用 GVT 当前车道恢复源车道关系。
    因此只使用 ACTIVE 开始时锁定的原车道线计算 VUT->暴露目标 THW/TTC。
    """

    main = context.main_vehicle
    if main is None:
        return {
            "scene_status": "cutout_finish_metric_unavailable",
            "debug": {"finish_metric_reason": "no_main_vehicle"},
        }

    locked_line = memory.get("locked_source_line")
    locked_cumulative_s = memory.get("locked_source_cumulative_s")
    if locked_line is None or locked_cumulative_s is None:
        return {
            "scene_status": "cutout_finish_metric_unavailable",
            "debug": {"finish_metric_reason": "no_locked_source_line"},
        }

    source_line_array = np.asarray(locked_line, dtype=float)
    source_cumulative_s = np.asarray(locked_cumulative_s, dtype=float)
    if source_line_array.ndim != 2 or len(source_line_array) < 2 or len(source_cumulative_s) < 2:
        return {
            "scene_status": "cutout_finish_metric_unavailable",
            "debug": {"finish_metric_reason": "invalid_locked_source_line"},
        }

    exposed = _resolve_exposed_actor(context, memory)
    if exposed is None:
        return {
            "scene_status": "cutout_finish_metric_unavailable",
            "debug": {
                "finish_metric_reason": "no_exposed_actor",
                "exposed_ptc_id": _read_exposed_ptc_id(memory),
            },
        }

    limits = _read_vehicle_limits(context, exposed)
    scene = {
        "ok": True,
        "source_line": ReferenceLineView(
            line_index=-1,
            lane_id=str(memory.get("locked_source_lane_id", "")),
            lane_sequence=_normalize_lane_sequence(memory.get("locked_source_lane_sequence", [])),
            lateral_step=0,
            line=source_line_array,
            cumulative_s=source_cumulative_s,
            current_s=0.0,
            current_d=0.0,
        ),
        "exposed": exposed,
        "limits": limits,
        "debug": {
            "finish_source": "locked_source_line",
            "exposed_ptc_id": str(exposed.ptc_id),
            "locked_source_lane_id": str(memory.get("locked_source_lane_id", "")),
            "locked_metric_name": str(memory.get("locked_metric_name", "")),
            "locked_expected_metric": float(memory.get("locked_expected_metric", float("inf"))),
        },
    }
    return _compute_finish_metric(context, scene)


# ==================== Output Builders / 输出封装 ====================


def _build_prepare_step_output(
    context: SceneContext,
    memory: dict[str, Any],
    scene: dict[str, Any],
    now: float,
    reason: str,
    scene_status: str = "prepare_wait_trigger",
    scene_debug: dict[str, Any] | None = None,
    trigger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 PREPARE 阶段源车道保持目标。"""

    del memory, now

    geometry = scene["geometry"]
    exposed = scene["exposed"]
    limits = _read_vehicle_limits(context, exposed)
    target_v, speed_debug = _compute_prepare_target_v(context, exposed, geometry, limits, trigger=trigger)
    plan_t = float(CFG["prepare_plan_t_s"])
    target_a = _clamp(
        (target_v - float(context.ego_vehicle.v)) / max(plan_t, 1e-3),
        limits.min_acc,
        limits.max_acc,
    )
    debug = copy.deepcopy(scene["debug"])
    if scene_debug is not None:
        debug.update(copy.deepcopy(scene_debug))
    debug.update(copy.deepcopy(speed_debug))
    debug["prepare_target_v_mps"] = float(target_v)
    debug["prepare_target_a_mps2"] = float(target_a)

    return _build_line_target_output(
        context=context,
        line=context.ego_vehicle.current_ref_line,
        target_v=target_v,
        target_a=target_a,
        plan_t=plan_t,
        extra_progress_m=float(CFG["prepare_target_extra_progress_m"]),
        reason=reason,
        scene_status=scene_status,
        scene_debug=debug,
    )


def _build_active_solution(
    context: SceneContext,
    scene: dict[str, Any],
    trigger: dict[str, Any],
    debug: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """根据当前 GVT 状态和目标车道生成本次切出的锁定终端点。"""

    target_line: ReferenceLineView = scene["target_line"]
    limits: VehicleLimits = scene["limits"]
    plan_t = float(trigger.get("lane_change_time_s", _dynamic_lane_change_time_s(context, limits)))
    ego = context.ego_vehicle
    _, ego_s_on_target, ego_d_on_target = project_point_to_path(np.array([ego.x, ego.y], dtype=float), target_line.line)
    if abs(float(ego_d_on_target)) > float(CFG["max_ego_to_target_line_d_m"]):
        return None

    target_v = _scene_cruise_target_v(context, limits)
    target_a = _clamp((target_v - float(ego.v)) / max(plan_t, 1e-3), limits.min_acc, limits.max_acc)
    travel_s = max(float(CFG["min_target_progress_m"]), 0.5 * (float(ego.v) + target_v) * plan_t)
    target_s = float(ego_s_on_target) + float(travel_s)
    target_s = _clamp(
        target_s,
        float(ego_s_on_target) + float(CFG["min_target_progress_m"]),
        float(target_line.cumulative_s[-1]) - float(CFG["post_cutout_buffer_m"]),
    )
    if target_s <= float(ego_s_on_target) + 1e-6:
        return None

    target_xy, target_heading = get_path_point_at_s(target_line.line, target_s, target_line.cumulative_s)
    if dry_run:
        return {
            "x": float(target_xy[0]),
            "y": float(target_xy[1]),
            "v": float(target_v),
            "a": float(target_a),
            "theta": float(target_heading),
            "t": float(plan_t),
            "target_line_index": int(target_line.line_index),
            "hold_t": 0.0,
            "reason": "cutout_active_dry_run",
        }

    solution_debug = copy.deepcopy(debug)
    solution_debug.update(
        {
            "active_target_s": float(target_s),
            "active_ego_s_on_target": float(ego_s_on_target),
            "active_ego_d_on_target": float(ego_d_on_target),
            "active_target_v_mps": float(target_v),
            "active_target_a_mps2": float(target_a),
            "active_plan_t_s": float(plan_t),
        }
    )
    return {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(target_v),
        "a": float(target_a),
        "theta": float(target_heading),
        "t": float(plan_t),
        "target_line_index": int(target_line.line_index),
        "hold_t": 0.0,
        "reason": "cutout_active_locked",
        "scene_debug": solution_debug,
    }


def _build_locked_active_output(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView,
    now: float,
) -> dict[str, Any] | None:
    """ACTIVE 阶段复用锁定终端点，按剩余时间推进。"""

    target_xy = memory.get("locked_target_xy")
    if target_xy is None:
        return None

    _, target_s, _ = project_point_to_path(np.asarray(target_xy, dtype=float), target_line.line)
    target_s = max(float(target_s), float(target_line.current_s) + float(CFG["min_target_progress_m"]))
    target_s = min(float(target_s), float(target_line.cumulative_s[-1]) - 0.5)
    if target_s <= float(target_line.current_s) + 1e-6:
        return None

    target_xy_projected, target_heading = get_path_point_at_s(target_line.line, target_s, target_line.cumulative_s)
    remaining_t = max(0.5, float(memory["active_end_time_s"]) - float(now))
    return {
        "x": float(target_xy_projected[0]),
        "y": float(target_xy_projected[1]),
        "v": float(memory.get("locked_target_v", 0.0)),
        "a": float(memory.get("locked_target_a", 0.0)),
        "theta": float(target_heading),
        "t": float(remaining_t),
        "target_line_index": int(target_line.line_index),
        "hold_t": 0.0,
        "reason": "cutout_active_locked",
    }


def _build_stabilize_output(
    context: SceneContext,
    target_line: ReferenceLineView,
    reason: str,
    scene_status: str,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造切出完成后目标车道稳定保持输出。"""

    target_v = _clamp(float(CFG["cruise_target_v"]), 0.0, _read_vehicle_limits(context, None).max_v)
    return _build_line_target_output(
        context=context,
        line=target_line,
        target_v=target_v,
        target_a=0.0,
        plan_t=2.0,
        reason=reason,
        scene_status=scene_status,
        scene_debug=scene_debug,
    )


def _build_cutout_safe_keep_lane_output(
    context: SceneContext,
    memory: dict[str, Any] | None,
    reason: str,
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """cut-out 场景丢失或冷却时的安全保底输出。

    如果还能从 obstacles 中解析到暴露目标，保底目标优先按暴露车跟车/停车；
    只有无法解析暴露目标时才回退到普通保持车道。
    """

    exposed = _resolve_exposed_actor(context, memory)
    if exposed is None:
        return _build_keep_lane_output(
            context,
            reason=reason,
            scene_status=scene_status,
            scene_detail=scene_detail,
            scene_debug=scene_debug,
        )

    line = _build_source_relation_line(context)
    limits = _read_vehicle_limits(context, exposed)
    _, gvt_s, gvt_d = project_point_to_path(np.array([context.ego_vehicle.x, context.ego_vehicle.y], dtype=float), line.line)
    _, exposed_s, exposed_d = project_point_to_path(np.array([exposed.x, exposed.y], dtype=float), line.line)
    gap = float(exposed_s) - float(gvt_s) - 0.5 * (limits.exposed_length + limits.ego_length)
    same_lane_ahead = bool(
        abs(float(gvt_d)) <= float(CFG["source_lane_match_d_thresh_m"])
        and abs(float(exposed_d)) <= float(CFG["source_lane_match_d_thresh_m"])
        and gap > 0.0
    )
    if not same_lane_ahead:
        return _build_keep_lane_output(
            context,
            reason=reason,
            scene_status=scene_status,
            scene_detail=scene_detail,
            scene_debug=scene_debug,
        )

    geometry = {"gvt_to_exposed_gap_m": float(gap)}
    target_v, speed_debug = _compute_prepare_target_v(context, exposed, geometry, limits, trigger=None)
    if gap <= float(CFG["source_abort_gap_m"]):
        target_v = 0.0
    plan_t = float(CFG["prepare_plan_t_s"])
    target_a = _clamp(
        (target_v - float(context.ego_vehicle.v)) / max(plan_t, 1e-3),
        limits.min_acc,
        limits.max_acc,
    )
    debug = copy.deepcopy(scene_debug or {})
    debug.update(copy.deepcopy(speed_debug))
    debug.update(
        {
            "safe_keep_lane_exposed_id": str(exposed.ptc_id),
            "safe_keep_lane_gap_m": float(gap),
            "safe_keep_lane_target_v_mps": float(target_v),
            "safe_keep_lane_target_a_mps2": float(target_a),
            "safe_keep_lane_reason": "follow_exposed",
        }
    )
    return _build_line_target_output(
        context=context,
        line=context.ego_vehicle.current_ref_line,
        target_v=target_v,
        target_a=target_a,
        plan_t=plan_t,
        extra_progress_m=float(CFG["prepare_target_extra_progress_m"]),
        reason=reason,
        scene_status=scene_status,
        scene_detail=scene_detail,
        scene_debug=debug,
    )


def _build_keep_lane_output(
    context: SceneContext,
    reason: str,
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造沿当前车道行驶的保底目标。"""

    limits = _read_vehicle_limits(context, exposed_actor=None)
    target_v = _clamp(float(CFG["cruise_target_v"]), 0.0, limits.max_v)
    return _build_line_target_output(
        context=context,
        line=context.ego_vehicle.current_ref_line,
        target_v=target_v,
        target_a=0.0,
        plan_t=2.0,
        reason=reason,
        scene_status=scene_status,
        scene_detail=scene_detail,
        scene_debug=scene_debug,
    )


def _build_line_target_output(
    *,
    context: SceneContext,
    line: ReferenceLineView,
    target_v: float,
    target_a: float,
    plan_t: float,
    reason: str,
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
    extra_progress_m: float = 2.0,
) -> dict[str, Any]:
    """在指定参考线上构造场景算法终端目标。"""

    travel_s = max(float(CFG["min_target_progress_m"]), float(target_v) * float(plan_t) + float(extra_progress_m))
    target_s = _clamp(
        float(line.current_s) + float(travel_s),
        float(line.current_s) + float(CFG["min_target_progress_m"]),
        float(line.cumulative_s[-1]) - 0.5,
    )
    target_xy, target_heading = get_path_point_at_s(line.line, target_s, line.cumulative_s)
    output = {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(target_v),
        "a": float(target_a),
        "theta": float(target_heading),
        "t": float(plan_t),
        "target_line_index": int(line.line_index),
        "hold_t": 0.0,
        "reason": str(reason),
    }
    return _annotate_output(output, scene_status, scene_detail or reason, scene_debug)


def _compute_prepare_target_v(
    context: SceneContext,
    exposed: PlanningActorState,
    geometry: dict[str, Any],
    limits: VehicleLimits,
    trigger: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """PREPARE 等待时的源车道速度策略，避免 GVT 在等待期间追近暴露目标。

    如果触发器已经算出最佳 hold_t，则按该等待时间限制 GVT 与暴露车的闭合速度。
    """

    gap = float(geometry.get("gvt_to_exposed_gap_m", float("inf")))
    exposed_v = max(0.0, float(exposed.v))
    lane_change_time = _dynamic_lane_change_time_s(context, limits)
    prepare_cruise_v = _scene_cruise_target_v(context, limits)
    if not np.isfinite(gap):
        target_v = _clamp(float(prepare_cruise_v), 0.0, limits.max_v)
        return target_v, {
            "prepare_speed_reason": "invalid_gap",
            "prepare_cruise_v_mps": float(prepare_cruise_v),
            "prepare_lane_change_time_s": float(lane_change_time),
            "prepare_speed_wait_t_s": float("inf"),
            "prepare_speed_horizon_s": float(CFG["prepare_plan_t_s"]),
            "prepare_allowed_closing_mps": float("inf"),
        }

    if gap <= float(CFG["source_abort_gap_m"]):
        return 0.0, {
            "prepare_speed_reason": "source_abort_gap",
            "prepare_cruise_v_mps": float(prepare_cruise_v),
            "prepare_lane_change_time_s": float(lane_change_time),
            "prepare_speed_wait_t_s": 0.0,
            "prepare_speed_horizon_s": float(CFG["prepare_plan_t_s"]),
            "prepare_allowed_closing_mps": 0.0,
        }

    wait_t = _extract_trigger_hold_t_s(trigger)
    if np.isfinite(wait_t):
        wait_buffer = float(CFG["prepare_wait_time_buffer_s"])
        release_t = _compute_source_release_time_s(lane_change_time)
        activation_horizon = max(float(CFG["prepare_plan_t_s"]), float(wait_t) + wait_buffer)
        release_horizon = max(float(CFG["prepare_plan_t_s"]), float(wait_t) + release_t + wait_buffer)
        activation_allowed_closing = max(0.0, gap - float(CFG["min_gvt_to_exposed_gap_m"])) / max(activation_horizon, 1e-3)
        release_allowed_closing = max(0.0, gap - float(CFG["source_abort_gap_m"])) / max(release_horizon, 1e-3)
        allowed_closing = min(activation_allowed_closing, release_allowed_closing)
        speed_horizon = max(activation_horizon, release_horizon)
    else:
        release_t = _compute_source_release_time_s(lane_change_time)
        speed_horizon = max(float(CFG["prepare_plan_t_s"]), release_t)
        activation_allowed_closing = max(0.0, gap - float(CFG["min_gvt_to_exposed_gap_m"])) / max(speed_horizon, 1e-3)
        release_allowed_closing = max(0.0, gap - float(CFG["source_abort_gap_m"])) / max(speed_horizon, 1e-3)
        allowed_closing = min(activation_allowed_closing, release_allowed_closing)
    target_v = min(float(prepare_cruise_v), exposed_v + allowed_closing)
    target_v = _clamp(target_v, 0.0, limits.max_v)
    return target_v, {
        "prepare_speed_reason": "wait_time_limited" if np.isfinite(wait_t) else "gap_limited",
        "prepare_cruise_v_mps": float(prepare_cruise_v),
        "prepare_lane_change_time_s": float(lane_change_time),
        "prepare_speed_wait_t_s": float(wait_t),
        "prepare_speed_horizon_s": float(speed_horizon),
        "prepare_allowed_closing_mps": float(allowed_closing),
        "prepare_activation_allowed_closing_mps": float(activation_allowed_closing),
        "prepare_release_allowed_closing_mps": float(release_allowed_closing),
        "prepare_release_time_s": float(release_t),
        "prepare_gap_margin_m": float(gap - float(CFG["min_gvt_to_exposed_gap_m"])),
        "prepare_exposed_v_mps": float(exposed_v),
    }


def _extract_trigger_hold_t_s(trigger: dict[str, Any] | None) -> float:
    """从触发器结果中提取最佳等待时间；无有效候选时返回 inf。"""

    if trigger is None:
        return float("inf")
    hold_t = float(trigger.get("hold_t", float("inf")))
    if not np.isfinite(hold_t) or hold_t < 0.0:
        return float("inf")
    return float(hold_t)


# ==================== Completion / 完成判定 ====================


def _is_cutout_completed(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView,
    now: float,
) -> bool:
    """判断 GVT 是否已按固定切出时长完成并进入目标车道。"""

    if float(now) < float(memory.get("active_end_time_s", float("inf"))):
        memory["execute_complete_hits"] = 0
        return False

    current_lane_id = str(context.ego_vehicle.current_ref_line.lane_id)
    target_lane_id = str(memory.get("locked_target_lane_id", ""))
    target_lane_sequence = set(_normalize_lane_sequence(memory.get("locked_target_lane_sequence", [])))
    lane_id_ok = bool((target_lane_id and current_lane_id == target_lane_id) or (target_lane_sequence and current_lane_id in target_lane_sequence))

    _, _, ego_d_on_target = project_point_to_path(np.array([context.ego_vehicle.x, context.ego_vehicle.y], dtype=float), target_line.line)
    lateral_ok = bool(abs(float(ego_d_on_target)) <= float(CFG["enter_target_d_thresh_m"]))
    if lane_id_ok or lateral_ok:
        memory["execute_complete_hits"] += 1
    else:
        memory["execute_complete_hits"] = 0
    return int(memory["execute_complete_hits"]) >= int(CFG["execute_complete_frames"])


# ==================== Utilities / 基础工具、调试与配置 ====================


def _read_vehicle_limits(
    context: SceneContext,
    exposed_actor: PlanningActorState | None,
) -> VehicleLimits:
    """统一读取车辆尺寸和平台约束。"""

    runtime_frame = context.runtime_frame
    config = {} if runtime_frame is None else runtime_frame.config
    ego_length = float(config.get("size", {}).get("L", 4.5))
    exposed_length = ego_length
    if exposed_actor is not None and float(exposed_actor.length) > 0.0:
        exposed_length = float(exposed_actor.length)
    constrain = config.get("constrain", {})
    return VehicleLimits(
        ego_length=float(ego_length),
        main_length=float(ego_length),
        exposed_length=float(exposed_length),
        max_v=float(constrain.get("max_v", 6.94)),
        min_acc=float(constrain.get("min_acc", -2.0)),
        max_acc=float(constrain.get("max_acc", 2.0)),
    )


def _annotate_output(
    output: dict[str, Any],
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """为场景输出补充状态标签和调试信息。"""

    output["scene_status"] = str(scene_status)
    if scene_detail is not None:
        output["scene_detail"] = str(scene_detail)
    if scene_debug is not None:
        output["scene_debug"] = copy.deepcopy(scene_debug)
    return output


def _clamp(value: float, lower: float, upper: float) -> float:
    """把数值裁剪到指定上下界。"""

    return float(min(max(float(value), float(lower)), float(upper)))


def _heading_diff_deg(h1: float, h2: float) -> float:
    """计算两个航向角之间的最小夹角，单位为度。"""

    return float(np.degrees(abs(normalize_angle(float(h1) - float(h2)))))


def _normalize_lane_sequence(lane_sequence: list[str] | tuple[str, ...] | None) -> list[str]:
    """把 lane sequence 规整成字符串列表。"""

    if lane_sequence is None:
        return []
    return [str(lane_id) for lane_id in lane_sequence]


def _line_by_index(
    candidate_lines: list[ReferenceLineView],
    line_index: int | None,
) -> ReferenceLineView | None:
    """按 line_index 查找参考线。"""

    if line_index is None:
        return None
    for line in candidate_lines:
        if int(line.line_index) == int(line_index):
            return line
    return None


def _line_by_lane_id(
    candidate_lines: list[ReferenceLineView],
    lane_id: str,
) -> ReferenceLineView | None:
    """按 lane_id 查找参考线。"""

    target_lane_id = str(lane_id or "")
    if not target_lane_id:
        return None
    for line in candidate_lines:
        if str(line.lane_id) == target_lane_id:
            return line
    return None


def _line_by_lane_sequence(
    candidate_lines: list[ReferenceLineView],
    target_lane_sequence: list[str] | tuple[str, ...] | None,
) -> ReferenceLineView | None:
    """按锁定 lane 链恢复语义最一致的参考线。"""

    normalized_target = _normalize_lane_sequence(target_lane_sequence)
    if not normalized_target:
        return None
    target_set = set(normalized_target)
    best_line = None
    best_score: tuple[int, int, int] | None = None
    for line in candidate_lines:
        line_sequence = _normalize_lane_sequence(line.lane_sequence)
        overlap_count = sum(1 for lane_id in line_sequence if lane_id in target_set)
        if overlap_count <= 0:
            continue
        first_match_idx = len(normalized_target)
        for idx, lane_id in enumerate(normalized_target):
            if lane_id in line_sequence:
                first_match_idx = idx
                break
        score = (overlap_count, -first_match_idx, -len(line_sequence))
        if best_score is None or score > best_score:
            best_score = score
            best_line = line
    return best_line
