"""Emergency braking scene algorithm.

设计目标：
- APPROACH 识别后方同向主车
- PREPARE 维持当前车道，等待后车满足精准 THW 触发窗口
- ACTIVE_BRAKE 按平台减速度约束输出当前车道制动目标
- STABILIZE 制动后短时低速稳定
- COOLDOWN 场景失败或结束后冷却，避免连续重复触发
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from planning_runtime.scene.types import SceneContext
from traj_planner.tools import get_path_point_at_s, normalize_angle, project_point_to_path


@dataclass(frozen=True)
class VehicleLimits:
    """场景算法实际使用的车辆尺寸与平台约束。"""

    ego_length: float
    main_length: float
    max_v: float
    min_acc: float
    max_acc: float


CFG: dict[str, Any] = {
    "cruise_target_v": 5.0,
    "prepare_target_v": 4.5,
    "prepare_wait_v_max": 4.5,
    "target_thw_s": 2.0,
    "thw_tol_s": 0.1,
    "prepare_plan_t_s": 1.5,
    "prepare_timeout_s": 8.0,
    "prepare_hard_timeout_s": 14.0,
    "prepare_abort_predicted_thw_time_s": 4.0,
    "prepare_near_thw_margin_s": 0.3,
    "prepare_min_gap_progress_m": 2.0,
    "prepare_progress_min_closing_speed_mps": 0.3,
    "brake_plan_t_s": 1.2,
    "brake_duration_s": 2.5,
    "emergency_decel_mps2": 2.5,
    "brake_target_v": 0.5,
    "stabilize_target_v": 0.8,
    "stabilize_duration_s": 1.5,
    "abort_cooldown_s": 2.0,
    "max_main_d_on_ego_m": 1.5,
    "behind_gap_max_m": 45.0,
    "main_comfort_decel_mps2": 3.0,
    "min_target_progress_floor_m": 0.3,
    "target_progress_ratio": 0.5,
    "execute_complete_frames": 2,
}


SCENE_MEMORY_TEMPLATE: dict[str, Any] = {
    "mode": "APPROACH",
    "cooldown_until_time_s": -1.0,
    "prepare_start_time_s": -1.0,
    "prepare_initial_gap_m": float("inf"),
    "prepare_best_gap_m": float("inf"),
    "active_start_time_s": -1.0,
    "active_end_time_s": -1.0,
    "active_start_v": 0.0,
    "execute_complete_hits": 0,
    "stable_until_time_s": -1.0,
    "last_metrics": None,
}


SCENE_MEMORY: dict[str, Any] = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Public API / 外部入口 ====================


def braking_algorithm(context: SceneContext) -> dict[str, Any]:
    """Emergency braking 场景算法主入口。"""
    global SCENE_MEMORY
    start_ms = time.time() * 1000.0
    if int(context.step_index) == 0:
        _reset_scene_memory()
    output, SCENE_MEMORY = _scene_fsm_step(context, SCENE_MEMORY)
    end_ms = time.time() * 1000.0
    print(f"Braking algorithm execution time: {end_ms - start_ms} ms")
    return output


# ==================== FSM / 状态机分发 ====================


def _scene_fsm_step(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行一帧 emergency braking 状态机。"""
    now = float(context.time_s)
    mode = str(memory.get("mode", "APPROACH"))

    if mode == "COOLDOWN":
        if now < float(memory["cooldown_until_time_s"]):
            return _handle_cooldown(context, memory, now)
        memory["mode"] = "APPROACH"
        mode = "APPROACH"

    if mode not in {"ACTIVE_BRAKE", "STABILIZE"} and not _has_valid_main_vehicle(context):
        return _handle_no_valid_main(context, memory)

    if mode == "APPROACH":
        return _handle_approach(context, memory, now)
    if mode == "PREPARE":
        return _handle_prepare(context, memory, now)
    if mode == "ACTIVE_BRAKE":
        return _handle_active_brake(context, memory, now)
    if mode == "STABILIZE":
        return _handle_stabilize(context, memory, now)
    return _handle_unknown_mode(context, memory)


# ==================== State Handlers / 每个状态独立逻辑函数 ====================


def _handle_cooldown(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 COOLDOWN：冷却期内沿当前车道巡航。"""
    return _build_keep_lane_output(
        context,
        reason="braking_cooldown",
        scene_status="cooldown",
    ), memory


def _handle_no_valid_main(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理非执行态没有有效主车的情况。"""
    memory["mode"] = "APPROACH"
    _reset_prepare_memory(memory)
    _reset_execute_memory(memory)
    return _build_keep_lane_output(
        context,
        reason="braking_no_valid_main",
        scene_status="waiting_valid_main",
    ), memory


def _handle_approach(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 APPROACH：识别后方同向主车，进入 PREPARE。"""
    metrics = _compute_brake_metrics(context)
    memory["last_metrics"] = copy.deepcopy(metrics)
    if not _is_brake_candidate(metrics):
        return _build_keep_lane_output(
            context,
            reason="braking_wait_candidate",
            scene_status="approach_wait_candidate",
            scene_debug=metrics,
        ), memory

    _enter_prepare_mode(memory, now, metrics)
    prepare_result = _select_prepare_thw_wait_target(context, memory, metrics, _read_vehicle_limits(context))
    activation_info = _evaluate_thw_activation(metrics)
    debug = copy.deepcopy(metrics)
    debug.update(copy.deepcopy(prepare_result.get("debug", {})))
    debug.update(activation_info)
    if bool(activation_info["thw_activation_ready"]):
        _enter_active_brake_mode(memory, context, now)
        activation_reason = str(activation_info["thw_activation_reason"])
        return _build_brake_output(
            context,
            memory,
            debug,
            reason="braking_triggered" if activation_reason == "exact_thw_ready" else "braking_triggered_late",
            scene_status="active_brake" if activation_reason == "exact_thw_ready" else "active_brake_late_thw",
        ), memory
    return _build_prepare_output(
        context,
        memory,
        debug,
        reason="braking_prepare_enter",
        scene_status="prepare_wait_trigger",
        prepare_result=prepare_result,
    ), memory


def _handle_prepare(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 PREPARE：稳定等待后车 THW 达标，并在 THW 达标时触发制动。"""
    metrics = _compute_brake_metrics(context)
    _update_prepare_progress(memory, metrics)
    timeout_info = _evaluate_prepare_timeout(memory, now, metrics)
    prepare_result = _select_prepare_thw_wait_target(context, memory, metrics, _read_vehicle_limits(context))
    activation_info = _evaluate_thw_activation(metrics)
    metrics_with_prepare_state = copy.deepcopy(metrics)
    metrics_with_prepare_state.update(timeout_info)
    metrics_with_prepare_state.update(copy.deepcopy(prepare_result.get("debug", {})))
    metrics_with_prepare_state.update(activation_info)
    memory["last_metrics"] = copy.deepcopy(metrics)
    if not _is_brake_candidate(metrics):
        memory["mode"] = "APPROACH"
        _reset_prepare_memory(memory)
        return _build_keep_lane_output(
            context,
            reason="braking_candidate_lost",
            scene_status="approach_wait_candidate",
            scene_debug=metrics_with_prepare_state,
        ), memory

    if bool(activation_info["thw_activation_ready"]):
        _enter_active_brake_mode(memory, context, now)
        activation_reason = str(activation_info["thw_activation_reason"])
        return _build_brake_output(
            context,
            memory,
            metrics_with_prepare_state,
            reason="braking_triggered" if activation_reason == "exact_thw_ready" else "braking_triggered_late",
            scene_status="active_brake" if activation_reason == "exact_thw_ready" else "active_brake_late_thw",
        ), memory

    if bool(timeout_info["prepare_should_abort"]):
        _enter_cooldown(memory, now)
        return _build_keep_lane_output(
            context,
            reason="braking_prepare_timeout",
            scene_status="prepare_timeout",
            scene_debug=metrics_with_prepare_state,
        ), memory

    return _build_prepare_output(
        context,
        memory,
        metrics_with_prepare_state,
        reason="braking_prepare_wait",
        scene_status="prepare_wait_trigger",
        prepare_result=prepare_result,
    ), memory


def _handle_active_brake(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 ACTIVE_BRAKE：输出当前车道减速目标。"""
    metrics = _compute_brake_metrics(context)
    memory["last_metrics"] = copy.deepcopy(metrics)
    if _brake_execution_completed(context, memory, now):
        memory["mode"] = "STABILIZE"
        memory["stable_until_time_s"] = float(now) + float(CFG["stabilize_duration_s"])
        return _build_stabilize_output(context, metrics), memory

    return _build_brake_output(
        context,
        memory,
        metrics,
        reason="braking_active",
        scene_status="active_brake",
    ), memory


def _handle_stabilize(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 STABILIZE：紧急制动后短时低速稳定。"""
    metrics = _compute_brake_metrics(context)
    output = _build_stabilize_output(context, metrics)
    if now >= float(memory["stable_until_time_s"]):
        _enter_cooldown(memory, now)
    return output, memory


def _handle_unknown_mode(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理未知状态：回到 APPROACH 并输出保底目标。"""
    memory["mode"] = "APPROACH"
    return _build_keep_lane_output(
        context,
        reason="braking_unknown_mode",
        scene_status="baseline_keep_lane",
    ), memory


# ==================== Transition Helpers / 状态切换与记忆管理 ====================


def _enter_prepare_mode(memory: dict[str, Any], now: float, metrics: dict[str, Any]) -> None:
    """进入 PREPARE。"""
    memory["mode"] = "PREPARE"
    memory["prepare_start_time_s"] = float(now)
    gap = float(metrics.get("net_gap_m", float("inf")))
    if np.isfinite(gap):
        memory["prepare_initial_gap_m"] = float(gap)
        memory["prepare_best_gap_m"] = float(gap)


def _enter_active_brake_mode(
    memory: dict[str, Any],
    context: SceneContext,
    now: float,
) -> None:
    """进入 ACTIVE_BRAKE，并记录制动起始状态。"""
    memory["mode"] = "ACTIVE_BRAKE"
    memory["active_start_time_s"] = float(now)
    memory["active_end_time_s"] = float(now) + float(CFG["brake_duration_s"])
    memory["active_start_v"] = float(context.ego_vehicle.v)
    memory["execute_complete_hits"] = 0
    _reset_prepare_memory(memory)


def _enter_cooldown(memory: dict[str, Any], now: float) -> None:
    """进入冷却态，并清空本轮执行上下文。"""
    memory["mode"] = "COOLDOWN"
    memory["cooldown_until_time_s"] = float(now) + float(CFG["abort_cooldown_s"])
    _reset_prepare_memory(memory)
    _reset_execute_memory(memory)


def _reset_prepare_memory(memory: dict[str, Any]) -> None:
    """清空 PREPARE 阶段缓存。"""
    memory["prepare_start_time_s"] = -1.0
    memory["prepare_initial_gap_m"] = float("inf")
    memory["prepare_best_gap_m"] = float("inf")


def _reset_execute_memory(memory: dict[str, Any]) -> None:
    """清空 ACTIVE_BRAKE / STABILIZE 执行缓存。"""
    memory["active_start_time_s"] = -1.0
    memory["active_end_time_s"] = -1.0
    memory["active_start_v"] = 0.0
    memory["execute_complete_hits"] = 0
    memory["stable_until_time_s"] = -1.0
    memory["last_metrics"] = None


def _reset_scene_memory() -> None:
    """把全局场景内存重置到初始值。"""
    global SCENE_MEMORY
    SCENE_MEMORY = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Output Builders / 输出封装 ====================


def _build_keep_lane_output(
    context: SceneContext,
    reason: str,
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造沿当前参考线巡航的保底目标。"""
    limits = _read_vehicle_limits(context)
    target_v = _clamp(float(CFG["cruise_target_v"]), 0.0, limits.max_v)
    return _build_line_target_output(
        context=context,
        target_v=target_v,
        target_a=0.0,
        plan_t=2.0,
        reason=reason,
        scene_status=scene_status,
        scene_detail=scene_detail or reason,
        scene_debug=scene_debug,
    )


def _select_prepare_thw_wait_target(
    context: SceneContext,
    memory: dict[str, Any],
    metrics: dict[str, Any],
    limits: VehicleLimits,
) -> dict[str, Any]:
    """按固定等待速度维持当前车道，等待后车 THW 进入触发窗口。"""
    min_trigger_v, min_trigger_debug = _compute_prepare_min_trigger_v(limits)
    wait_max = _clamp(float(CFG["prepare_wait_v_max"]), min_trigger_v, limits.max_v)
    target_v = _clamp(float(CFG["prepare_target_v"]), min_trigger_v, wait_max)
    target_thw = max(float(CFG["target_thw_s"]), 1e-3)
    main_v = float(metrics.get("main_v", 0.0))
    ego_v = float(metrics.get("ego_v", context.ego_vehicle.v))
    closing_speed = float(metrics.get("closing_speed_mps", 0.0))
    current_thw = float(metrics.get("thw_s", float("inf")))
    target_gap = target_thw * main_v - limits.ego_length if main_v > 1e-6 else float("inf")

    elapsed = max(0.0, float(context.time_s) - float(memory.get("prepare_start_time_s", context.time_s)))
    thw_error = current_thw - target_thw
    debug = {
        "prepare_speed_policy": "hold_thw_wait_speed",
        "prepare_solve_status": "wait_for_target_thw",
        "prepare_target_v": float(target_v),
        "prepare_raw_target_v": float(CFG["prepare_target_v"]),
        "prepare_wait_v_min": float(min_trigger_v),
        "prepare_min_trigger_v": float(min_trigger_v),
        "prepare_wait_v_max": float(wait_max),
        "prepare_elapsed_s": float(elapsed),
        "target_thw_s": float(target_thw),
        "thw_tol_s": float(CFG["thw_tol_s"]),
        "current_thw_s": float(current_thw),
        "current_thw_error_s": float(thw_error),
        "target_thw_gap_m": float(target_gap),
        "prepare_current_closing_speed_mps": float(closing_speed),
        "prepare_current_ego_v": float(ego_v),
        "prepare_current_main_v": float(main_v),
    }
    debug.update(min_trigger_debug)
    return {
        "target_v": float(target_v),
        "target_thw_s": float(target_thw),
        "solve_status": "wait_for_target_thw",
        "debug": debug,
    }


def _compute_prepare_min_trigger_v(limits: VehicleLimits) -> tuple[float, dict[str, Any]]:
    """根据制动能力自动推导 PREPARE 阶段允许的最低触发速度。"""
    desired_acc = -abs(float(CFG["emergency_decel_mps2"]))
    active_target_a = _clamp(desired_acc, limits.min_acc, limits.max_acc)
    effective_decel = abs(min(active_target_a, 0.0))
    raw_min_trigger_v = float(CFG["brake_target_v"]) + effective_decel * float(CFG["brake_plan_t_s"])
    min_trigger_v = _clamp(raw_min_trigger_v, 0.0, limits.max_v)
    return min_trigger_v, {
        "prepare_min_trigger_v": float(min_trigger_v),
        "prepare_raw_min_trigger_v": float(raw_min_trigger_v),
        "prepare_effective_brake_decel_mps2": float(effective_decel),
        "prepare_active_target_acc_mps2": float(active_target_a),
    }


def _build_prepare_output(
    context: SceneContext,
    memory: dict[str, Any],
    metrics: dict[str, Any],
    reason: str,
    scene_status: str,
    prepare_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 PREPARE 阶段当前车道目标。"""
    limits = _read_vehicle_limits(context)
    if prepare_result is None:
        prepare_result = _select_prepare_thw_wait_target(context, memory, metrics, limits)
    target_v = float(prepare_result["target_v"])
    plan_t = float(CFG["prepare_plan_t_s"])
    a_req = _clamp(
        (target_v - float(context.ego_vehicle.v)) / max(plan_t, 1e-3),
        limits.min_acc,
        limits.max_acc,
    )
    debug = copy.deepcopy(metrics)
    debug.update(copy.deepcopy(prepare_result.get("debug", {})))
    debug["prepare_target_acc_mps2"] = float(a_req)
    return _build_line_target_output(
        context=context,
        target_v=target_v,
        target_a=a_req,
        plan_t=plan_t,
        reason=reason,
        scene_status=scene_status,
        scene_debug=debug,
    )


def _build_brake_output(
    context: SceneContext,
    memory: dict[str, Any],
    metrics: dict[str, Any],
    reason: str,
    scene_status: str,
) -> dict[str, Any]:
    """构造 ACTIVE_BRAKE 阶段的当前车道制动目标。"""
    limits = _read_vehicle_limits(context)
    plan_t = float(CFG["brake_plan_t_s"])
    desired_acc = -abs(float(CFG["emergency_decel_mps2"]))
    target_a = _clamp(desired_acc, limits.min_acc, limits.max_acc)
    target_v = max(float(CFG["brake_target_v"]), float(context.ego_vehicle.v) + target_a * plan_t)
    target_v = _clamp(target_v, 0.0, limits.max_v)
    debug = copy.deepcopy(metrics)
    debug.update(
        {
            "brake_target_acc": float(target_a),
            "brake_target_v": float(target_v),
            "active_elapsed_s": max(0.0, float(context.time_s) - float(memory.get("active_start_time_s", context.time_s))),
            "active_remaining_s": max(0.0, float(memory.get("active_end_time_s", context.time_s)) - float(context.time_s)),
        }
    )
    return _build_line_target_output(
        context=context,
        target_v=target_v,
        target_a=target_a,
        plan_t=plan_t,
        reason=reason,
        scene_status=scene_status,
        scene_debug=debug,
    )


def _build_stabilize_output(
    context: SceneContext,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """构造制动后的短时低速稳定目标。"""
    limits = _read_vehicle_limits(context)
    target_v = _clamp(float(CFG["stabilize_target_v"]), 0.0, limits.max_v)
    return _build_line_target_output(
        context=context,
        target_v=target_v,
        target_a=0.0,
        plan_t=1.5,
        reason="braking_stabilize",
        scene_status="stabilize_low_speed",
        scene_debug=metrics,
    )


def _build_line_target_output(
    *,
    context: SceneContext,
    target_v: float,
    target_a: float,
    plan_t: float,
    reason: str,
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在当前参考线上按目标速度和规划时长构造终端点。"""
    line = context.ego_vehicle.current_ref_line
    min_progress = max(
        float(CFG["min_target_progress_floor_m"]),
        float(target_v) * float(plan_t) * float(CFG["target_progress_ratio"]),
    )
    travel_s = max(min_progress, float(target_v) * float(plan_t))
    target_s = _clamp(
        float(line.current_s) + float(travel_s),
        float(line.current_s) + float(min_progress),
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


# ==================== Condition Helpers / 触发与状态判断 ====================


def _has_valid_main_vehicle(context: SceneContext) -> bool:
    """判断主车是否满足 emergency braking 的基本识别条件。"""
    return context.main_vehicle is not None


def _is_brake_candidate(metrics: dict[str, Any]) -> bool:
    """判断当前主车是否是制动场景候选对象。"""
    if not bool(metrics.get("main_is_behind", False)):
        return False
    if float(metrics.get("relation_lateral_error_m", float("inf"))) > float(CFG["max_main_d_on_ego_m"]):
        return False
    gap = float(metrics.get("net_gap_m", float("inf")))
    return bool(0.0 < gap <= float(CFG["behind_gap_max_m"]))


def _evaluate_thw_activation(metrics: dict[str, Any]) -> dict[str, Any]:
    """判断当前 THW 是否达到精准触发窗口或已经越过目标 THW。"""
    target_thw = float(CFG["target_thw_s"])
    tolerance = float(CFG["thw_tol_s"])
    lower = max(0.0, target_thw - tolerance)
    upper = target_thw + tolerance
    closing_speed = float(metrics.get("closing_speed_mps", 0.0))
    thw = float(metrics.get("thw_s", float("inf")))

    ready = False
    reason = "thw_not_ready"
    if not _is_brake_candidate(metrics):
        reason = "not_candidate"
    elif not np.isfinite(thw):
        reason = "invalid_thw"
    elif closing_speed <= 1e-3:
        reason = "no_closing_speed"
    elif lower <= thw <= upper:
        ready = True
        reason = "exact_thw_ready"
    elif thw < lower:
        ready = True
        reason = "late_thw_trigger"

    return {
        "thw_activation_ready": bool(ready),
        "thw_activation_reason": str(reason),
        "target_thw_s": float(target_thw),
        "thw_tolerance_s": float(tolerance),
        "thw_activation_lower_s": float(lower),
        "thw_activation_upper_s": float(upper),
        "current_thw_s": float(thw),
        "current_thw_error_s": float(thw - target_thw),
    }


def _update_prepare_progress(memory: dict[str, Any], metrics: dict[str, Any]) -> None:
    """更新 PREPARE 阶段的缺口收敛进展。"""
    gap = float(metrics.get("net_gap_m", float("inf")))
    if not np.isfinite(gap):
        return
    if not np.isfinite(float(memory.get("prepare_initial_gap_m", float("inf")))):
        memory["prepare_initial_gap_m"] = float(gap)
    best_gap = float(memory.get("prepare_best_gap_m", float("inf")))
    memory["prepare_best_gap_m"] = float(min(best_gap, gap))


def _evaluate_prepare_timeout(
    memory: dict[str, Any],
    now: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """评估 PREPARE 是否应该超时退出，软超时后允许有进展的场景继续等待。"""
    start = float(memory.get("prepare_start_time_s", now))
    elapsed = max(0.0, float(now) - start)
    soft_timeout = float(CFG["prepare_timeout_s"])
    hard_timeout = max(soft_timeout, float(CFG["prepare_hard_timeout_s"]))
    gap = float(metrics.get("net_gap_m", float("inf")))
    closing_speed = float(metrics.get("closing_speed_mps", 0.0))
    target_thw = float(CFG["target_thw_s"])
    thw = float(metrics.get("thw_s", float("inf")))
    main_v = float(metrics.get("main_v", 0.0))
    ego_length = float(metrics.get("ego_length_m", 0.0))
    target_gap = target_thw * main_v - ego_length if main_v > 1e-6 else float("inf")
    gap_to_target = max(0.0, gap - target_gap) if np.isfinite(gap) and np.isfinite(target_gap) else float("inf")
    predicted_trigger_time = gap_to_target / closing_speed if closing_speed > 1e-3 else float("inf")
    initial_gap = float(memory.get("prepare_initial_gap_m", float("inf")))
    best_gap = float(memory.get("prepare_best_gap_m", float("inf")))
    gap_progress = initial_gap - gap if np.isfinite(initial_gap) and np.isfinite(gap) else 0.0
    best_gap_progress = initial_gap - best_gap if np.isfinite(initial_gap) and np.isfinite(best_gap) else 0.0
    near_trigger = bool(np.isfinite(thw) and thw <= target_thw + float(CFG["prepare_near_thw_margin_s"]))
    predicted_ok = bool(predicted_trigger_time <= float(CFG["prepare_abort_predicted_thw_time_s"]))
    making_progress = bool(
        best_gap_progress >= float(CFG["prepare_min_gap_progress_m"])
        and closing_speed >= float(CFG["prepare_progress_min_closing_speed_mps"])
    )

    should_abort = False
    decision = "within_soft_timeout"
    if elapsed >= hard_timeout:
        if near_trigger:
            decision = "extend_hard_near_target_thw"
        elif predicted_ok:
            decision = "extend_hard_predicted_target_thw"
        else:
            should_abort = True
            decision = "hard_timeout"
    elif elapsed >= soft_timeout:
        if near_trigger:
            decision = "extend_near_target_thw"
        elif predicted_ok:
            decision = "extend_predicted_target_thw"
        elif making_progress:
            decision = "extend_gap_progress"
        else:
            should_abort = True
            decision = "soft_timeout_no_progress"

    return {
        "prepare_elapsed_s": float(elapsed),
        "prepare_soft_timeout_s": float(soft_timeout),
        "prepare_hard_timeout_s": float(hard_timeout),
        "prepare_target_thw_gap_m": float(target_gap),
        "prepare_gap_to_target_thw_m": float(gap_to_target),
        "prepare_predicted_trigger_time_s": float(predicted_trigger_time),
        "prepare_initial_gap_m": float(initial_gap),
        "prepare_best_gap_m": float(best_gap),
        "prepare_gap_progress_m": float(gap_progress),
        "prepare_best_gap_progress_m": float(best_gap_progress),
        "prepare_near_target_thw": bool(near_trigger),
        "prepare_predicted_trigger_ok": bool(predicted_ok),
        "prepare_making_progress": bool(making_progress),
        "prepare_timeout_decision": str(decision),
        "prepare_should_abort": bool(should_abort),
    }


def _brake_execution_completed(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> bool:
    """判断主动制动是否完成。"""
    if float(now) >= float(memory.get("active_end_time_s", -1.0)):
        return True
    if float(context.ego_vehicle.v) <= float(CFG["brake_target_v"]) + 0.2:
        memory["execute_complete_hits"] += 1
    else:
        memory["execute_complete_hits"] = 0
    return int(memory["execute_complete_hits"]) >= int(CFG["execute_complete_frames"])


# ==================== Metrics / 纵向安全指标 ====================


def _project_pair_on_braking_relation_line(context: SceneContext) -> dict[str, Any]:
    """把 ego/main 投影到 braking 关系线，供 braking 计算相对关系。"""
    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        return {
            "source": "none",
            "ego_s": 0.0,
            "ego_d": 0.0,
            "main_s": 0.0,
            "main_d": float("inf"),
            "rel_s": 0.0,
            "rel_d": float("inf"),
        }

    candidate_lines = [
        ("main_forward_corridor", getattr(main.current_ref_line, "line", None)),
        ("ego_forward_corridor", getattr(ego.current_ref_line, "line", None)),
        ("ego_full_lane", getattr(ego, "lane_centerline", None)),
        ("main_full_lane", getattr(main, "lane_centerline", None)),
    ]
    projected: list[dict[str, Any]] = []
    for source, raw_line in candidate_lines:
        projection = _project_pair_on_line(context, raw_line, source)
        if projection is not None:
            projected.append(projection)

    if not projected:
        return _project_pair_by_ego_heading(context)

    return min(
        projected,
        key=lambda item: (
            max(abs(float(item["ego_d"])), abs(float(item["main_d"]))),
            abs(float(item["rel_d"])),
        ),
    )


def _project_pair_on_line(
    context: SceneContext,
    raw_line: Any,
    source: str,
) -> dict[str, Any] | None:
    """把 ego/main 投影到一条候选关系线。"""
    if raw_line is None:
        return None
    line = np.asarray(raw_line, dtype=float)
    if line.ndim != 2 or len(line) < 2:
        return None

    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        return None

    _, ego_s, ego_d = project_point_to_path(np.array([ego.x, ego.y], dtype=float), line)
    _, main_s, main_d = project_point_to_path(np.array([main.x, main.y], dtype=float), line)
    return {
        "source": str(source),
        "ego_s": float(ego_s),
        "ego_d": float(ego_d),
        "main_s": float(main_s),
        "main_d": float(main_d),
        "rel_s": float(main_s - ego_s),
        "rel_d": float(main_d - ego_d),
    }


def _project_pair_by_ego_heading(context: SceneContext) -> dict[str, Any]:
    """完整车道线不可用时，用 ego 航向局部坐标兜底。"""
    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        return {
            "source": "none",
            "ego_s": 0.0,
            "ego_d": 0.0,
            "main_s": 0.0,
            "main_d": float("inf"),
            "rel_s": 0.0,
            "rel_d": float("inf"),
        }

    rel_xy = np.array([float(main.x) - float(ego.x), float(main.y) - float(ego.y)], dtype=float)
    ego_forward = np.array([np.cos(float(ego.heading)), np.sin(float(ego.heading))], dtype=float)
    ego_left = np.array([-ego_forward[1], ego_forward[0]], dtype=float)
    rel_s = float(np.dot(rel_xy, ego_forward))
    rel_d = float(np.dot(rel_xy, ego_left))
    return {
        "source": "ego_heading_fallback",
        "ego_s": 0.0,
        "ego_d": 0.0,
        "main_s": float(rel_s),
        "main_d": float(rel_d),
        "rel_s": float(rel_s),
        "rel_d": float(rel_d),
    }


def _compute_brake_metrics(context: SceneContext) -> dict[str, Any]:
    """计算 emergency braking 使用的当前车道纵向指标。"""
    ego = context.ego_vehicle
    main = context.main_vehicle
    limits = _read_vehicle_limits(context)
    if main is None:
        return {
            "main_valid": False,
            "main_is_behind": False,
            "net_gap_m": float("inf"),
            "thw_effective_gap_m": float("inf"),
            "thw_s": float("inf"),
            "closing_speed_mps": 0.0,
            "required_decel_mps2": 0.0,
            "stop_margin_m": float("inf"),
            "ego_length_m": float(limits.ego_length),
            "main_length_m": float(limits.main_length),
        }

    projection = _project_pair_on_braking_relation_line(context)
    rel_s = float(projection["rel_s"])
    main_d_on_ego = float(projection["main_d"])
    ego_d_on_relation_line = float(projection["ego_d"])
    relation_lateral_error = max(abs(ego_d_on_relation_line), abs(main_d_on_ego))
    main_is_behind = rel_s < 0.0
    center_gap = -rel_s
    net_gap = center_gap - 0.5 * (limits.ego_length + limits.main_length)
    closing_speed = float(main.v) - float(ego.v)
    thw_effective_gap = net_gap + limits.ego_length
    thw = thw_effective_gap / float(main.v) if thw_effective_gap > 0.0 and float(main.v) > 1e-6 else float("inf")
    required_decel = (closing_speed ** 2) / max(2.0 * net_gap, 1e-3) if net_gap > 0.0 and closing_speed > 0.0 else 0.0
    main_stop_dist = (float(main.v) ** 2) / max(2.0 * float(CFG["main_comfort_decel_mps2"]), 1e-3)
    ego_stop_dist = (float(ego.v) ** 2) / max(2.0 * abs(float(CFG["emergency_decel_mps2"])), 1e-3)
    stop_margin = net_gap + ego_stop_dist - main_stop_dist
    return {
        "main_valid": True,
        "main_is_behind": bool(main_is_behind),
        "projection_source": str(projection["source"]),
        "ego_s_on_relation_line": float(projection["ego_s"]),
        "ego_d_on_relation_line": float(projection["ego_d"]),
        "main_s_on_relation_line": float(projection["main_s"]),
        "main_rel_longitudinal_m": float(rel_s),
        "main_d_on_ego": float(main_d_on_ego),
        "relation_lateral_error_m": float(relation_lateral_error),
        "net_gap_m": float(net_gap),
        "thw_effective_gap_m": float(thw_effective_gap),
        "thw_s": float(thw),
        "closing_speed_mps": float(closing_speed),
        "required_decel_mps2": float(required_decel),
        "stop_margin_m": float(stop_margin),
        "heading_diff_deg": _heading_diff_deg(float(main.heading), float(ego.heading)),
        "ego_v": float(ego.v),
        "main_v": float(main.v),
        "ego_length_m": float(limits.ego_length),
        "main_length_m": float(limits.main_length),
    }


# ==================== Utilities / 基础工具、调试与配置 ====================


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
    """把数值裁到指定上下界。"""
    return float(min(max(value, lower), upper))


def _heading_diff_deg(h1: float, h2: float) -> float:
    """计算两个航向角之间的最小夹角，单位为度。"""
    return float(np.degrees(abs(normalize_angle(float(h1) - float(h2)))))


def _read_vehicle_limits(context: SceneContext) -> VehicleLimits:
    """统一读取场景算法使用的车辆尺寸和平台约束。"""
    runtime_frame = context.runtime_frame
    config = {} if runtime_frame is None else runtime_frame.config
    ego_length = float(config.get("size", {}).get("L", 4.5))
    constrain = config.get("constrain", {})
    return VehicleLimits(
        ego_length=float(ego_length),
        main_length=float(ego_length),
        max_v=float(constrain.get("max_v", 6.94)),
        min_acc=float(constrain.get("min_acc", -2.0)),
        max_acc=float(constrain.get("max_acc", 2.0)),
    )
