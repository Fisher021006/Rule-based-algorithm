"""Emergency braking scene algorithm.

设计目标：
- APPROACH 识别后方同向主车
- PREPARE 维持当前车道，等待 TTC / 间距进入触发窗口
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
    "prepare_plan_t_s": 1.5,
    "prepare_timeout_s": 8.0,
    "brake_plan_t_s": 1.2,
    "brake_duration_s": 2.5,
    "emergency_decel_mps2": 2.5,
    "brake_target_v": 0.5,
    "stabilize_target_v": 0.8,
    "stabilize_duration_s": 1.5,
    "abort_cooldown_s": 2.0,
    "max_heading_diff_deg": 20.0,
    "search_rel_s_min": -60.0,
    "search_rel_s_max": 8.0,
    "max_main_d_on_ego_m": 1.5,
    "behind_gap_min_m": 4.0,
    "behind_gap_max_m": 45.0,
    "trigger_ttc_s": 3.0,
    "trigger_gap_m": 12.0,
    "trigger_closing_speed_mps": 0.5,
    "release_gap_m": 35.0,
    "main_comfort_decel_mps2": 3.0,
    "min_target_progress_m": 2.0,
    "execute_complete_frames": 2,
}


SCENE_MEMORY_TEMPLATE: dict[str, Any] = {
    "mode": "APPROACH",
    "cooldown_until_time_s": -1.0,
    "prepare_start_time_s": -1.0,
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

    _enter_prepare_mode(memory, now)
    return _build_prepare_output(
        context,
        metrics,
        reason="braking_prepare_enter",
        scene_status="prepare_wait_trigger",
    ), memory


def _handle_prepare(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 PREPARE：等待 TTC / 间距进入紧急制动触发窗口。"""
    metrics = _compute_brake_metrics(context)
    memory["last_metrics"] = copy.deepcopy(metrics)
    if not _is_brake_candidate(metrics):
        memory["mode"] = "APPROACH"
        _reset_prepare_memory(memory)
        return _build_keep_lane_output(
            context,
            reason="braking_candidate_lost",
            scene_status="approach_wait_candidate",
            scene_debug=metrics,
        ), memory

    if _brake_trigger_ready(metrics):
        _enter_active_brake_mode(memory, context, now)
        return _build_brake_output(
            context,
            memory,
            metrics,
            reason="braking_triggered",
            scene_status="active_brake",
        ), memory

    if _prepare_timeout(memory, now):
        _enter_cooldown(memory, now)
        return _build_keep_lane_output(
            context,
            reason="braking_prepare_timeout",
            scene_status="prepare_timeout",
            scene_debug=metrics,
        ), memory

    return _build_prepare_output(
        context,
        metrics,
        reason="braking_prepare_wait",
        scene_status="prepare_wait_trigger",
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


def _enter_prepare_mode(memory: dict[str, Any], now: float) -> None:
    """进入 PREPARE。"""
    memory["mode"] = "PREPARE"
    memory["prepare_start_time_s"] = float(now)


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


def _build_prepare_output(
    context: SceneContext,
    metrics: dict[str, Any],
    reason: str,
    scene_status: str,
) -> dict[str, Any]:
    """构造 PREPARE 阶段当前车道目标。"""
    limits = _read_vehicle_limits(context)
    target_v = _clamp(float(CFG["prepare_target_v"]), 0.0, limits.max_v)
    plan_t = float(CFG["prepare_plan_t_s"])
    a_req = _clamp(
        (target_v - float(context.ego_vehicle.v)) / max(plan_t, 1e-3),
        limits.min_acc,
        limits.max_acc,
    )
    return _build_line_target_output(
        context=context,
        target_v=target_v,
        target_a=a_req,
        plan_t=plan_t,
        reason=reason,
        scene_status=scene_status,
        scene_debug=metrics,
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
    travel_s = max(float(CFG["min_target_progress_m"]), float(target_v) * float(plan_t) + 1.0)
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


# ==================== Condition Helpers / 触发与状态判断 ====================


def _has_valid_main_vehicle(context: SceneContext) -> bool:
    """判断主车是否满足 emergency braking 的基本识别条件。"""
    if context.main_vehicle is None:
        return False
    ego = context.ego_vehicle
    main = context.main_vehicle
    same_direction = _heading_diff_deg(main.heading, ego.heading) <= float(CFG["max_heading_diff_deg"])
    rel_s_ok = float(CFG["search_rel_s_min"]) <= float(main.relative_s_to_ego) <= float(CFG["search_rel_s_max"])
    return bool(same_direction and rel_s_ok)


def _is_brake_candidate(metrics: dict[str, Any]) -> bool:
    """判断当前主车是否是制动场景候选对象。"""
    if not bool(metrics.get("main_is_behind", False)):
        return False
    if abs(float(metrics.get("main_d_on_ego", float("inf")))) > float(CFG["max_main_d_on_ego_m"]):
        return False
    gap = float(metrics.get("net_gap_m", float("inf")))
    return bool(float(CFG["behind_gap_min_m"]) <= gap <= float(CFG["behind_gap_max_m"]))


def _brake_trigger_ready(metrics: dict[str, Any]) -> bool:
    """判断 TTC / 间距是否进入紧急制动触发窗口。"""
    if not _is_brake_candidate(metrics):
        return False
    gap = float(metrics.get("net_gap_m", float("inf")))
    closing_speed = float(metrics.get("closing_speed_mps", 0.0))
    ttc = float(metrics.get("ttc_s", float("inf")))
    if gap <= float(CFG["trigger_gap_m"]):
        return True
    return bool(
        closing_speed >= float(CFG["trigger_closing_speed_mps"])
        and ttc <= float(CFG["trigger_ttc_s"])
    )


def _prepare_timeout(memory: dict[str, Any], now: float) -> bool:
    """判断 PREPARE 是否等待触发过久。"""
    start = float(memory.get("prepare_start_time_s", now))
    return bool(max(0.0, float(now) - start) >= float(CFG["prepare_timeout_s"]))


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
            "closing_speed_mps": 0.0,
            "ttc_s": float("inf"),
            "required_decel_mps2": 0.0,
            "stop_margin_m": float("inf"),
        }

    rel_xy = np.array([float(main.x) - float(ego.x), float(main.y) - float(ego.y)], dtype=float)
    ego_forward = np.array([np.cos(float(ego.heading)), np.sin(float(ego.heading))], dtype=float)
    ego_left = np.array([-ego_forward[1], ego_forward[0]], dtype=float)
    main_rel_longitudinal = float(np.dot(rel_xy, ego_forward))
    main_d_on_ego = float(np.dot(rel_xy, ego_left))
    main_is_behind = main_rel_longitudinal < 0.0
    center_gap = -main_rel_longitudinal
    net_gap = center_gap - 0.5 * (limits.ego_length + limits.main_length)
    closing_speed = float(main.v) - float(ego.v)
    ttc = net_gap / closing_speed if net_gap > 0.0 and closing_speed > 1e-3 else float("inf")
    required_decel = (closing_speed ** 2) / max(2.0 * net_gap, 1e-3) if net_gap > 0.0 and closing_speed > 0.0 else 0.0
    main_stop_dist = (float(main.v) ** 2) / max(2.0 * float(CFG["main_comfort_decel_mps2"]), 1e-3)
    ego_stop_dist = (float(ego.v) ** 2) / max(2.0 * abs(float(CFG["emergency_decel_mps2"])), 1e-3)
    stop_margin = net_gap + ego_stop_dist - main_stop_dist
    return {
        "main_valid": True,
        "main_is_behind": bool(main_is_behind),
        "main_rel_longitudinal_m": float(main_rel_longitudinal),
        "main_d_on_ego": float(main_d_on_ego),
        "net_gap_m": float(net_gap),
        "closing_speed_mps": float(closing_speed),
        "ttc_s": float(ttc),
        "required_decel_mps2": float(required_decel),
        "stop_margin_m": float(stop_margin),
        "ego_v": float(ego.v),
        "main_v": float(main.v),
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
