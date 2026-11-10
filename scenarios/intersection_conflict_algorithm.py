"""交叉口场景：控制场景车与主车的几何中心近同时进入固定圆形冲突区。

GVT 对应 ego_vehicle，VUT 对应 main_vehicle；两种 conflict_type 只标识测试类型。
成功指标为两次实际入区事件的时差，预测只用于调速。位置点沿用运行时的车身
几何中心约定，不是车头、后轴或原始 GPS 点。车身尺寸不膨胀入区半径，碰撞和
过点时差作为独立交互结果。同步判定成立后继续通行与观测，不增加提前触发器。
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from planning_runtime.scene.types import ReferenceLineView, SceneContext
from traj_planner import path as traj_path_module
from traj_planner.path import bezier_curve_kappa
from traj_planner.tools import (
    calculate_path_cumulative_distance,
    get_path_point_at_s,
    normalize_angle,
    project_point_to_path,
)


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
    """本场景使用的车身尺寸和平台限值。"""

    ego_length: float
    main_length: float
    ego_width: float
    main_width: float
    max_v: float
    min_acc: float
    max_acc: float


SUPPORTED_CONFLICT_TYPES = {"vut_straight_cross", "vut_left_oncoming"}


CFG: dict[str, Any] = {
    # 外部参数仍可单独配置，不依赖技能注册或研究模块。
    "conflict_type": _env_str("ROS_SCENE_INTERSECTION_CONFLICT_TYPE", "vut_straight_cross"),
    # 入区时差 = 场景车实际入区时刻 - 主车实际入区时刻；正值表示场景车更晚。
    "target_delta_t_s": _env_float("ROS_SCENE_INTERSECTION_TARGET_DELTA_T_S", 0.0),
    "success_time_tol_s": _env_float("ROS_SCENE_INTERSECTION_SUCCESS_TOL_S", 0.5),
    "conflict_zone_radius_m": 2.5,
    "conflict_point_match_tolerance_m": 1.2,
    "min_conflict_angle_deg": 25.0,
    "min_forward_conflict_distance_m": 2.0,
    "max_forward_conflict_distance_m": _env_float(
        "ROS_SCENE_INTERSECTION_MAX_CONFLICT_DISTANCE_M", 120.0
    ),
    "cruise_target_v": _env_float("ROS_SCENE_INTERSECTION_CRUISE_TARGET_V", 5.0),
    # VUT 到冲突点的预测：短时使用 OBU 纵向加速度，随后保持预测末速度。
    # 避免将一帧噪声或瞬时制动无限外推到整个路口。
    "vut_accel_prediction_horizon_s": 1.5,
    "vut_accel_min_mps2": -3.0,
    "vut_accel_max_mps2": 2.0,
    "sync_plan_t_min_s": 0.2,
    "sync_timeout_s": 15.0,
    "invalid_stop_plan_t_s": 1.0,
    "min_target_progress_m": 0.05,
    # 局部规划只消费当前路线走廊上的近端目标，不能直接追踪冲突区后的全局点。
    "local_target_preview_time_s": 1.5,
    "local_output_plan_t_min_s": 0.8,
    "local_output_plan_t_max_s": 4.0,
}


SCENE_MEMORY_TEMPLATE: dict[str, Any] = {
    "mode": "APPROACH",
    "sync_start_time_s": -1.0,
    "active_start_time_s": -1.0,
    "complete_start_time_s": -1.0,
    "locked_conflict": None,
    "last_metrics": None,
    "last_observation_time_s": None,
    "last_gvt_signed_distance_m": None,
    "last_vut_signed_distance_m": None,
    "last_gvt_center_xy": None,
    "last_vut_center_xy": None,
    "gvt_cross_time_s": None,
    "vut_cross_time_s": None,
    "gvt_zone_enter_time_s": None,
    "gvt_zone_exit_time_s": None,
    "vut_zone_enter_time_s": None,
    "vut_zone_exit_time_s": None,
    "actual_result_printed": False,
    "execution_complete": False,
    "execution_end_reason": "",
}


SCENE_MEMORY: dict[str, Any] = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Public API ====================


def intersection_conflict_algorithm(context: SceneContext) -> dict[str, Any]:
    """交叉口场景的统一公开入口。"""
    global SCENE_MEMORY
    start_ms = time.time() * 1000.0
    if int(context.step_index) == 0:
        _reset_scene_memory()
    output, SCENE_MEMORY = _scene_fsm_step(context, SCENE_MEMORY)
    end_ms = time.time() * 1000.0
    print(f"Intersection conflict algorithm execution time: {end_ms - start_ms} ms")
    return output


# ==================== FSM ====================


def _scene_fsm_step(context: SceneContext, memory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行统一的准备、同步、通行观测和完成四阶段。"""
    now = float(context.time_s)
    handlers = {"APPROACH": _handle_approach, "SYNC_ARRIVAL": _handle_sync_arrival,
                "CONFLICT_ACTIVE": _handle_conflict_active, "COMPLETE": _handle_complete}
    handler = handlers.get(str(memory.get("mode", "APPROACH")))
    return handler(context, memory, now) if handler else _handle_unknown_mode(context, memory)


def _handle_approach(context: SceneContext, memory: dict[str, Any], now: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """验证并锁定冲突几何；每次新尝试清空旧事件。"""
    memory.clear()
    memory.update(copy.deepcopy(SCENE_MEMORY_TEMPLATE))
    conflict = _build_conflict_context(context)
    if not bool(conflict.get("valid", False)):
        return _build_invalid_scene_output(context, str(conflict.get("reason", "intersection_invalid_scene")), conflict), memory
    memory.update(mode="SYNC_ARRIVAL", sync_start_time_s=now,
                  locked_conflict=_compact_conflict_context(conflict))
    return _handle_sync_arrival(context, memory, now)


def _handle_sync_arrival(context: SceneContext, memory: dict[str, Any], now: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """持续修正尚未发生的自车入区时刻，实际事件不再反推。"""
    metrics = _compute_conflict_metrics(context, memory.get("locked_conflict"))
    if not bool(metrics.get("valid", False)):
        # 短时丢车保留已锁几何和事件；没有新观测时不拿旧位置伪造事件。
        elapsed = max(0.0, now - float(memory["sync_start_time_s"]))
        if elapsed >= float(CFG["sync_timeout_s"]):
            memory.update(mode="CONFLICT_ACTIVE", active_start_time_s=now,
                          timing_control_end_reason="observation_unavailable")
            return _handle_conflict_active(context, memory, now)
        debug = copy.deepcopy(memory.get("last_metrics") or {})
        debug.update(observation_available=False, observation_missing_reason=metrics.get("reason"),
                     execution_complete=False, execution_phase="syncing")
        return _build_keep_lane_output(context, "intersection_waiting_observation", "sync_arrival", scene_debug=debug), memory
    metrics = _update_actual_conflict_observation(memory, metrics, now)
    elapsed = max(0.0, now - float(memory["sync_start_time_s"]))
    # 自车已经入区时，自己的入区时刻已固定；主车出区本身不是错失判据。
    if memory.get("gvt_zone_enter_time_s") is not None or elapsed >= float(CFG["sync_timeout_s"]):
        memory.update(mode="CONFLICT_ACTIVE", active_start_time_s=now)
        memory["timing_control_end_reason"] = "gvt_entered" if memory.get("gvt_zone_enter_time_s") is not None else "sync_timeout"
        return _handle_conflict_active(context, memory, now)
    memory["last_metrics"] = copy.deepcopy(metrics)
    return _build_sync_output(context, memory, metrics, "intersection_sync_arrival", "sync_arrival"), memory


def _handle_conflict_active(context: SceneContext, memory: dict[str, Any], now: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """停止追赶已发生的入区时刻，按明确巡航策略通过并继续观测。"""
    metrics = _compute_conflict_metrics(context, memory.get("locked_conflict"))
    if not bool(metrics.get("valid", False)):
        # 已锁定的事件保留；临时缺车时不伪造新的入区或通行结果。
        metrics = copy.deepcopy(memory.get("last_metrics") or metrics)
        metrics["observation_available"] = False
    else:
        metrics = _update_actual_conflict_observation(memory, metrics, now)
        metrics["observation_available"] = True
    elapsed = max(0.0, now - float(memory.get("active_start_time_s", now)))
    cleared = bool(metrics.get("observation_available", False) and all(
        memory.get(f"{actor}_zone_exit_time_s") is not None
        and float(metrics.get(f"{actor}_body_zone_clearance_m", -1.0)) > 0.0
        for actor in ("gvt", "vut")))
    timed_out = elapsed >= float(CFG["sync_timeout_s"])
    if cleared or timed_out:
        memory["last_metrics"] = copy.deepcopy(metrics)
        memory.update(mode="COMPLETE", execution_complete=True, complete_start_time_s=now,
                      execution_end_reason="both_bodies_cleared" if cleared else "observation_timeout")
        return _handle_complete(context, memory, now)
    metrics.update(execution_complete=False, execution_phase="interaction",
                   timing_control_end_reason=memory.get("timing_control_end_reason", ""))
    memory["last_metrics"] = copy.deepcopy(metrics)
    return _build_keep_lane_output(context, "intersection_observing_interaction", "conflict_active", scene_debug=metrics), memory


def _handle_complete(context: SceneContext, memory: dict[str, Any], now: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """终态只表示执行结束；同步结果不与碰撞或过点结果混合。"""
    metrics = copy.deepcopy(memory.get("last_metrics") or {})
    if not bool(metrics.get("actual_result_ready", False)):
        metrics["sync_result"] = "not_observed"
    metrics.update(execution_complete=True, execution_phase="finished",
                   execution_end_time_s=float(memory.get("complete_start_time_s", now)),
                   execution_end_reason=memory.get("execution_end_reason", ""))
    memory["last_metrics"] = copy.deepcopy(metrics)
    status = "complete" if metrics.get("sync_result") == "met" else "intersection_sync_missed"
    return _build_keep_lane_output(context, "intersection_execution_finished", status, scene_debug=metrics), memory


def _handle_unknown_mode(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """未知状态回到准备阶段，并输出明确错误。"""
    memory["mode"] = "APPROACH"
    memory["locked_conflict"] = None
    return _build_keep_lane_output(
        context,
        reason="intersection_unknown_mode",
        scene_status="baseline_keep_lane",
    ), memory


def _reset_scene_memory() -> None:
    global SCENE_MEMORY
    SCENE_MEMORY = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Output builders ====================


def _build_keep_lane_output(context: SceneContext, reason: str, scene_status: str,
                            scene_detail: str | None = None, scene_debug: dict[str, Any] | None = None) -> dict[str, Any]:
    """沿路线继续巡航；这是场景输出策略，不表示切换到了自主避障控制器。"""
    limits = _read_vehicle_limits(context)
    horizon = max(float(context.dt), float(CFG["local_target_preview_time_s"]))
    target_v = _clamp(float(CFG["cruise_target_v"]), 0.0, limits.max_v)
    target_a = _clamp((target_v - float(context.ego_vehicle.v)) / horizon, limits.min_acc, limits.max_acc)
    debug = copy.deepcopy(scene_debug or {})
    debug["post_entry_policy"] = "route_cruise"
    output = _build_conflict_target_output(
        context=context, metrics=debug,
        sync_target={"target_v_mps": target_v, "target_a_mps2": target_a},
        reason=reason, scene_status=scene_status, scene_debug=debug)
    output["scene_detail"] = scene_detail or reason
    return output


def _build_invalid_scene_output(
    context: SceneContext,
    reason: str,
    scene_debug: dict[str, Any],
) -> dict[str, Any]:
    """报告明确的场景几何错误并请求场景车停车。"""
    limits = _read_vehicle_limits(context)
    plan_t = float(CFG["invalid_stop_plan_t_s"])
    target_v = 0.0
    target_a = _clamp(
        (target_v - float(context.ego_vehicle.v)) / max(plan_t, 1e-3),
        limits.min_acc,
        limits.max_acc,
    )
    debug = copy.deepcopy(scene_debug)
    debug.setdefault("invalid_scene_reasons", [str(reason)])
    output = _build_line_target_output(
        context=context,
        target_line=context.ego_vehicle.current_ref_line,
        target_s=float(context.ego_vehicle.current_ref_line.current_s) + float(CFG["min_target_progress_m"]),
        target_v=target_v,
        target_a=target_a,
        plan_t=plan_t,
        reason=reason,
        scene_status="invalid_scene_geometry",
        scene_detail="intersection_conflict_scene_geometry_invalid",
        scene_debug=debug,
    )
    output["scene_error"] = True
    output["control_mode"] = "scene_invalid"
    return output


def _build_sync_output(
    context: SceneContext,
    memory: dict[str, Any],
    metrics: dict[str, Any],
    reason: str,
    scene_status: str,
) -> dict[str, Any]:
    """构造对准实际入区事件的速度目标。"""
    del memory
    sync_target = _solve_sync_target(context, metrics)
    debug = copy.deepcopy(metrics)
    debug.update(sync_target)
    debug.update(execution_phase="syncing", execution_complete=False)
    return _build_conflict_target_output(
        context=context,
        metrics=metrics,
        sync_target=sync_target,
        reason=reason,
        scene_status=scene_status,
        scene_debug=debug,
    )


def _build_conflict_target_output(*, context: SceneContext, metrics: dict[str, Any],
                                  sync_target: dict[str, Any], reason: str, scene_status: str,
                                  scene_debug: dict[str, Any]) -> dict[str, Any]:
    """近端距离、末速度和时域来自同一受限剖面，避免混用远端终速和近端距离。"""
    limits = _read_vehicle_limits(context)
    line, selection = _select_route_aligned_local_line(context)
    _, current_s, lateral = project_point_to_path(np.array([context.ego_vehicle.x, context.ego_vehicle.y]), line.line)
    minimum = max(float(context.dt), float(CFG["local_output_plan_t_min_s"]))
    horizon = max(minimum, min(float(CFG["local_target_preview_time_s"]), float(CFG["local_output_plan_t_max_s"])))
    desired = float(sync_target.get("desired_gvt_arrival_t_s", float("inf")))
    if np.isfinite(desired) and desired > 0.0:
        horizon = max(minimum, min(horizon, desired))
    # 生成器按固定 0.1 秒采样；先吸附时域再积分，禁止把 0.142 秒距离交给 0.1 秒轨迹。
    horizon = float(np.ceil((horizon - 1e-9) / 0.1) * 0.1)
    acceleration = float(sync_target.get("target_a_mps2", 0.0))
    distance, terminal_v, limited = _smooth_local_profile(
        float(context.ego_vehicle.v), float(context.ego_vehicle.a), acceleration, horizon, limits)
    if limited and "sync_target_feasible" in scene_debug:
        scene_debug.update(sync_target_feasible=False, sync_target_infeasible_reason="local_time_parameterization_limited")
    available = max(0.0, float(line.cumulative_s[-1]) - 0.5 - current_s)
    # 目标点落在参考线上；车身存在横向偏移（换道未完成）时，规划器实际生成的
    # 路径比参考线弧长更长，直接按弧长下发会让纵向进度超过剖面可达距离。
    # 这里用规划器同一段 Bezier 几何把期望路径长度折算成弧长，不放宽动态限值。
    path_distance = float(distance)
    arc_distance, achieved_distance = _path_distance_to_arc(
        context, line, float(current_s), float(lateral), path_distance, available)
    if achieved_distance < path_distance - 1e-6:
        # 局部走廊不足时明确标记，不把裁短后的目标继续称作原时序可行。
        scene_debug.update(sync_target_feasible=False, sync_target_infeasible_reason="local_corridor_too_short")
        terminal_v = min(terminal_v, max(0.0, 2.0 * achieved_distance / horizon - float(context.ego_vehicle.v)))
    scene_debug.update(selection)
    scene_debug.update(requested_sync_acc_mps2=acceleration, target_acc_mps2=0.0,
                       local_mean_acc_mps2=(terminal_v-float(context.ego_vehicle.v))/horizon,
                       local_output_target_distance_m=achieved_distance,
                       local_output_target_arc_m=float(arc_distance),
                       local_output_target_path_distance_m=path_distance,
                       local_output_target_s_m=current_s + float(arc_distance), local_output_plan_t_s=horizon,
                       ego_s_on_selected_local_line_m=float(current_s), ego_d_on_selected_local_line_m=float(lateral),
                       output_target_line_source="route_aligned_bounded_profile")
    output = _build_line_target_output(context=context, target_line=line,
                                    target_s=current_s + float(arc_distance), target_v=terminal_v,
                                    target_a=0.0, plan_t=horizon, reason=reason,
                                    scene_status=scene_status, scene_debug=scene_debug,
                                    min_plan_t=float(context.dt), min_target_progress=0.0)
    # 仅本场景请求最终动态限值复核，其他既有外部目标场景保持原入口行为。
    output["enforce_external_dynamic_limits"] = True
    return output


def _path_distance_to_arc(
    context: SceneContext,
    line: ReferenceLineView,
    current_s: float,
    lateral: float,
    path_distance: float,
    available_s: float,
) -> tuple[float, float]:
    """把期望路径长度折算成参考线弧长，返回 ``(弧长, 实际路径长度)``。

    规划器在车身偏离参考线时会用一段 Bezier 从当前位置平滑并回目标点，
    实际路径长度大于弧长。这里复用同一段几何反解弧长，保证下发的纵向
    进度与速度剖面可达的距离一致。路径长度对弧长单调递增，用割线迭代
    在括号内收敛，通常 2~3 次即可；横向偏移小于 5 cm 时两者差可忽略，
    直接返回，避免每帧额外构造 Bezier。
    """

    available_s = max(0.0, float(available_s))
    path_distance = max(0.0, float(path_distance))
    if abs(float(lateral)) <= 0.05:
        arc_s = min(path_distance, available_s)
        return arc_s, arc_s

    ego_xy = np.array([context.ego_vehicle.x, context.ego_vehicle.y], dtype=float)
    start_heading = float(context.ego_vehicle.heading)

    def path_length(arc_s: float) -> float:
        target_xy, _ = get_path_point_at_s(line.line, current_s + arc_s, line.cumulative_s)
        # 这里只是复用同一段几何量长度，不能占用规划器的曲率告警额度，
        # 否则真正需要关注的规划告警会被提前用光。
        warning_state = traj_path_module._curvature_warning_count
        traj_path_module._curvature_warning_count = traj_path_module.CURVATURE_WARNING_PRINT_LIMIT + 2
        try:
            path = np.asarray(
                bezier_curve_kappa(line.line, ego_xy, np.asarray(target_xy, dtype=float), start_heading,
                                   end_theta=None),
                dtype=float,
            )
        finally:
            traj_path_module._curvature_warning_count = warning_state
        if path.ndim != 2 or len(path) < 2 or not np.isfinite(path).all():
            return float("inf")
        return float(calculate_path_cumulative_distance(path)[-1])

    start_length = path_length(0.0)
    if not np.isfinite(start_length):
        return available_s, path_distance
    if start_length >= path_distance - 1e-6:
        return 0.0, start_length

    hi_arc = min(path_distance, available_s)
    hi_length = path_length(hi_arc)
    if not np.isfinite(hi_length):
        return available_s, path_distance
    if hi_length <= path_distance:
        return hi_arc, hi_length

    # 括号 [lo, hi] 始终满足 length(lo) <= path_distance < length(hi)。
    # 割线步落在括号内部并留最小收缩量，避免端点停滞；返回下界保证不超发。
    lo_arc, lo_length = 0.0, start_length
    for _ in range(4):
        if hi_length - lo_length <= 1e-3:
            break
        shrink = max(1e-4, 1e-3 * (hi_arc - lo_arc))
        if hi_arc - lo_arc <= 2.0 * shrink:
            break
        span = hi_length - lo_length
        if span <= 1e-9:
            break
        guess = lo_arc + (path_distance - lo_length) * (hi_arc - lo_arc) / span
        guess = min(max(guess, lo_arc + shrink), hi_arc - shrink)
        guess_length = path_length(guess)
        if not np.isfinite(guess_length):
            break
        if guess_length <= path_distance:
            lo_arc, lo_length = guess, guess_length
        else:
            hi_arc, hi_length = guess, guess_length
    return lo_arc, lo_length


def _build_line_target_output(
    *,
    context: SceneContext,
    target_line: ReferenceLineView,
    target_s: float,
    target_v: float,
    target_a: float,
    plan_t: float,
    reason: str,
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
    min_plan_t: float | None = None,
    min_target_progress: float | None = None,
) -> dict[str, Any]:
    """在选中的场景车参考线上构造终端目标。"""
    min_progress = float(CFG["min_target_progress_m"]) if min_target_progress is None else float(min_target_progress)
    lower_s = float(target_line.current_s) + max(0.0, min_progress)
    upper_s = float(target_line.cumulative_s[-1]) - 0.5
    if upper_s <= lower_s:
        lower_s = float(target_line.current_s)
        upper_s = max(lower_s, float(target_line.cumulative_s[-1]))
    resolved_s = _clamp(float(target_s), lower_s, upper_s)
    target_xy, target_heading = get_path_point_at_s(target_line.line, resolved_s, target_line.cumulative_s)
    output = {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(max(0.0, target_v)),
        "a": float(target_a),
        "theta": float(target_heading),
        "t": float(max(float(context.dt), float(plan_t))),
        "target_line_index": int(target_line.line_index),
        "hold_t": 0.0,
        "reason": str(reason),
    }
    if min_plan_t is not None:
        output["min_external_plan_t_s"] = float(min_plan_t)
    if min_target_progress is not None:
        output["min_external_target_progress_m"] = float(min_target_progress)
    return _annotate_output(output, scene_status, scene_detail or reason, scene_debug)


def _annotate_output(
    output: dict[str, Any],
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output["scene_status"] = str(scene_status)
    if scene_detail is not None:
        output["scene_detail"] = str(scene_detail)
    if scene_debug is not None:
        output["scene_debug"] = copy.deepcopy(scene_debug)
    return output


# ==================== Conflict construction and metrics ====================


def _build_conflict_context(context: SceneContext) -> dict[str, Any]:
    """验证双方前方冲突点，并锁定同一圆形区域的路线几何。"""
    conflict_type = str(CFG.get("conflict_type", "")).strip()
    if conflict_type not in SUPPORTED_CONFLICT_TYPES:
        return {
            "valid": False,
            "reason": "unsupported_conflict_type",
            "conflict_type": conflict_type,
            "supported_conflict_types": sorted(SUPPORTED_CONFLICT_TYPES),
            "invalid_scene_reasons": ["unsupported_conflict_type"],
        }
    if context.main_vehicle is None:
        return {
            "valid": False,
            "reason": "no_main_vehicle",
            "conflict_type": conflict_type,
            "invalid_scene_reasons": ["no_main_vehicle"],
        }

    gvt_line = _route_line(context.ego_vehicle)
    main_line = _route_line(context.main_vehicle)
    if gvt_line is None or main_line is None:
        return {
            "valid": False,
            "reason": "route_intent_missing",
            "conflict_type": conflict_type,
            "gvt_route_reason": _route_intent_reason(context.ego_vehicle),
            "main_route_reason": _route_intent_reason(context.main_vehicle),
            "invalid_scene_reasons": ["route_intent_missing"],
        }

    best_result = _find_forward_conflict_point(gvt_line, main_line)
    if not bool(best_result.get("valid", False)):
        return {
            "valid": False,
            "reason": "no_forward_route_conflict_point",
            "conflict_type": conflict_type,
            "gvt_lane_sequence": _normalize_lane_sequence(gvt_line.lane_sequence),
            "main_lane_sequence": _normalize_lane_sequence(main_line.lane_sequence),
            "invalid_scene_reasons": ["no_forward_route_conflict_point"],
        }
    best_result["gvt_line_index"] = int(gvt_line.line_index)
    best_result["gvt_lane_id"] = str(gvt_line.lane_id)
    best_result["gvt_lane_sequence"] = _normalize_lane_sequence(gvt_line.lane_sequence)
    best_result["main_line_index"] = int(main_line.line_index)
    best_result["main_lane_id"] = str(main_line.lane_id)
    best_result["main_lane_sequence"] = _normalize_lane_sequence(main_line.lane_sequence)

    conflict_xy = best_result["conflict_xy"]
    center = np.asarray(conflict_xy, dtype=float)
    radius = float(CFG["conflict_zone_radius_m"])
    if radius <= 0.0 or any(np.linalg.norm(np.array([vehicle.x, vehicle.y])-center) <= radius
                            for vehicle in (context.ego_vehicle, context.main_vehicle)):
        return {"valid": False, "reason": "initial_center_not_outside_conflict_zone"}
    zone_geometry = {actor: _route_zone_profile(line, center, radius)
                     for actor, line in (("gvt", gvt_line), ("vut", main_line))}
    if any(profile is None for profile in zone_geometry.values()):
        return {"valid": False, "reason": "route_does_not_cover_conflict_zone"}
    print(
        "[intersection_conflict] conflict_xy="
        f"({float(conflict_xy[0]):.3f}, {float(conflict_xy[1]):.3f})",
        flush=True,
    )

    return {
        "valid": True,
        "reason": "conflict_point_locked",
        "conflict_zone_radius_m": radius,
        "zone_geometry": zone_geometry,
        "conflict_type": conflict_type,
        "conflict_xy": list(conflict_xy),
        "conflict_angle_deg": float(best_result["conflict_angle_deg"]),
        "segment_distance_m": float(best_result["segment_distance_m"]),
        "gvt_line_index": int(best_result["gvt_line_index"]),
        "gvt_lane_id": str(best_result["gvt_lane_id"]),
        "gvt_lane_sequence": _normalize_lane_sequence(best_result["gvt_lane_sequence"]),
        "gvt_s_conflict_m": float(best_result["gvt_s_conflict_m"]),
        "gvt_distance_to_conflict_m": float(best_result["gvt_distance_to_conflict_m"]),
        "main_line_index": int(best_result["main_line_index"]),
        "main_lane_id": str(best_result["main_lane_id"]),
        "main_lane_sequence": _normalize_lane_sequence(best_result["main_lane_sequence"]),
        "vut_s_conflict_m": float(best_result["vut_s_conflict_m"]),
        "vut_distance_to_conflict_m": float(best_result["vut_distance_to_conflict_m"]),
    }


def _compact_conflict_context(conflict: dict[str, Any]) -> dict[str, Any]:
    """保留固定圆和锁定路线，避免后续前向裁切把已通过的冲突点夹到零。"""
    keys = ("conflict_type", "conflict_xy", "conflict_angle_deg", "gvt_line_index", "gvt_lane_id",
            "gvt_lane_sequence", "main_line_index", "main_lane_id", "main_lane_sequence",
            "conflict_zone_radius_m", "zone_geometry")
    return {key: copy.deepcopy(conflict[key]) for key in keys if key in conflict}


def _segment_circle_roots(start: np.ndarray, end: np.ndarray, center: np.ndarray, radius: float) -> list[float]:
    """返回线段与圆周的交点比例，按运动顺序排列并去除相切重复根。"""
    movement, offset = end - start, start - center
    a = float(np.dot(movement, movement))
    if a <= 1e-12:
        return []
    b, c = 2.0 * float(np.dot(offset, movement)), float(np.dot(offset, offset)) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < -1e-9:
        return []
    root = np.sqrt(max(0.0, discriminant))
    values = sorted(_clamp(float(t), 0.0, 1.0) for t in ((-b-root)/(2*a), (-b+root)/(2*a)) if -1e-9 <= t <= 1.0+1e-9)
    return [t for i, t in enumerate(values) if i == 0 or abs(t-values[i-1]) > 1e-8]


def _route_zone_profile(line: ReferenceLineView, center: np.ndarray, radius: float) -> dict[str, Any] | None:
    """求包含已选冲突点的那一段圆内路线，锁定真实入区/出区弧长。"""
    xy, cumulative = np.asarray(line.line, dtype=float)[:, :2], np.asarray(line.cumulative_s, dtype=float)
    _, conflict_s, _ = project_point_to_path(center, xy)
    roots = []
    for i in range(len(xy)-1):
        length = float(cumulative[i+1]-cumulative[i])
        roots.extend(float(cumulative[i])+fraction*length for fraction in _segment_circle_roots(xy[i], xy[i+1], center, radius))
    roots = sorted(set(round(value, 10) for value in roots))
    for entry, exit_ in zip(roots, roots[1:]):
        midpoint, _ = get_path_point_at_s(xy, 0.5*(entry+exit_), cumulative)
        if entry <= conflict_s <= exit_ and np.linalg.norm(midpoint-center) < radius+1e-8:
            return {"route_xy": xy.tolist(), "s_conflict": float(conflict_s), "s_enter": entry, "s_exit": exit_}
    return None


def _body_zone_clearance(xy: np.ndarray, heading: float, length: float, width: float,
                         center: np.ndarray, radius: float) -> float:
    """圆到有向车身矩形的净距；仅用于通行观测，不膨胀中心入区判据。"""
    delta = center-xy
    local = np.array([np.cos(heading)*delta[0]+np.sin(heading)*delta[1],
                      -np.sin(heading)*delta[0]+np.cos(heading)*delta[1]])
    return float(np.linalg.norm(np.maximum(np.abs(local)-np.array([length/2, width/2]), 0.0))-radius)


def _finite_difference(first: Any, second: Any) -> float:
    """未知或无限的到达时间不做无穷减无穷。"""
    a, b = _optional_float(first), _optional_float(second)
    return a-b if a is not None and b is not None else float("nan")


def _smooth_local_profile(speed: float, start_acc: float, requested_acc: float,
                          duration: float, limits: VehicleLimits) -> tuple[float, float, bool]:
    """让局部末状态兼容现有五次时参：初加速度来自观测，末加速度为零。

    用三次速度剖面积分得到距离，并在速度/加速度极值处检查平台限值。
    这比只约束平均加速度更严格，可避免平均值达标但五次回退峰值越限。
    """
    def violation(end_speed: float) -> float:
        change = end_speed - speed
        velocity = np.array([speed, duration*start_acc,
                             3*change-2*duration*start_acc, duration*start_acc-2*change])
        acceleration = np.array([start_acc, 6*change/duration-4*start_acc,
                                 3*start_acc-6*change/duration])
        profiles = [(velocity, acceleration)]
        if start_acc < 0.0:
            # 既有五次回退会把末速裁到 2d/T-v0；制动初态下需同时检查这条实际剖面。
            distance = duration*0.5*(speed+end_speed)+start_acc*duration*duration/12.0
            clipped_end = max(0.0, end_speed+start_acc*duration/6.0)
            delta_s = distance-speed*duration-0.5*start_acc*duration*duration
            delta_v = (clipped_end-speed)*duration-start_acc*duration*duration
            delta_a = -start_acc*duration*duration
            c3 = 10*delta_s-4*delta_v+0.5*delta_a
            c4 = -15*delta_s+7*delta_v-delta_a
            c5 = 6*delta_s-3*delta_v+0.5*delta_a
            profiles.append((np.array([speed, duration*start_acc, 3*c3/duration,
                                       4*c4/duration, 5*c5/duration]),
                             np.array([start_acc, 6*c3/duration**2,
                                       12*c4/duration**2, 20*c5/duration**2])))
        error = -float("inf")
        for velocity, acceleration in profiles:
            speed_times, acc_times = [0.0, 1.0], [0.0, 1.0]
            for root in np.polynomial.polynomial.polyroots(acceleration):
                if abs(root.imag) < 1e-8 and 0.0 < root.real < 1.0:
                    speed_times.append(float(root.real))
            for root in np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(acceleration)):
                if abs(root.imag) < 1e-8 and 0.0 < root.real < 1.0:
                    acc_times.append(float(root.real))
            speeds = np.polynomial.polynomial.polyval(speed_times, velocity)
            accs = np.polynomial.polynomial.polyval(acc_times, acceleration)
            error = max(error, float(-np.min(speeds)), float(np.max(speeds)-limits.max_v),
                        float(limits.min_acc-np.min(accs)), float(np.max(accs)-limits.max_acc))
        return error

    def within(end_speed: float) -> bool:
        return violation(end_speed) <= 1e-8

    desired = _clamp(speed+requested_acc*duration, 0.0, limits.max_v)
    terminal = desired
    if not within(terminal):
        # 裁速拐点两侧的速度/加速度均对末速呈仿射关系，各自最大越限量为凸函数。
        split = _clamp(-start_acc*duration/6.0, 0.0, limits.max_v)
        boundaries = sorted({0.0, split, limits.max_v})
        feasible, attempted = [], []
        for lower, upper in zip(boundaries, boundaries[1:]):
            wanted = _clamp(desired, lower, upper)
            if within(wanted):
                feasible.append(wanted)
                continue
            baseline = _clamp(speed+0.5*start_acc*duration, lower, upper)
            if not within(baseline):
                # 基准不可行不代表初态无解；在这一段合法末速范围内寻找可行锚点。
                left_bound, right_bound = lower, upper
                for _ in range(60):
                    left, right = (2*left_bound+right_bound)/3, (left_bound+2*right_bound)/3
                    left_error, right_error = violation(left), violation(right)
                    if min(left_error, right_error) <= 1e-8:
                        baseline = left if left_error <= right_error else right
                        break
                    if left_error <= right_error:
                        right_bound = right
                    else:
                        left_bound = left
                else:
                    baseline = min((lower, 0.5*(left_bound+right_bound), upper), key=violation)
            attempted.append(baseline)
            if within(baseline):
                low, high = 0.0, 1.0
                for _ in range(28):
                    ratio = 0.5*(low+high)
                    if within(baseline+ratio*(wanted-baseline)):
                        low = ratio
                    else:
                        high = ratio
                feasible.append(baseline+low*(wanted-baseline))
        if feasible:
            terminal = min(feasible, key=lambda value: abs(value-desired))
        elif attempted:
            # 本时域和剖面内没有找到合法末速，仍由最终检查拒绝，不放松限值。
            terminal = min(attempted, key=violation)
    distance = duration*0.5*(speed+terminal)+start_acc*duration*duration/12.0
    return max(0.0, distance), terminal, abs(terminal-desired) > 1e-5


def _compute_conflict_metrics(context: SceneContext, locked_conflict: dict[str, Any] | None) -> dict[str, Any]:
    """用同一固定圆预测中心入区；过点时差仅作独立诊断。"""
    if locked_conflict is None or context.main_vehicle is None:
        return {"valid": False, "reason": "no_locked_conflict" if locked_conflict is None else "no_main_vehicle"}
    point = np.asarray(locked_conflict["conflict_xy"], dtype=float)
    radius = float(locked_conflict.get("conflict_zone_radius_m", CFG["conflict_zone_radius_m"]))
    geometries = locked_conflict.get("zone_geometry", {})
    limits = _read_vehicle_limits(context)
    result = {"valid": True, "reason": "metrics_ok", "time_s": float(context.time_s),
              "conflict_xy": point.tolist(), "conflict_type": locked_conflict.get("conflict_type", ""),
              "conflict_angle_deg": locked_conflict.get("conflict_angle_deg", 0.0),
              "conflict_zone_radius_m": radius, "position_reference": "body_center",
              "intersection_metrics_version": 2, "timing_metric": "center_entry_delta_t_s",
              "entry_event_definition": "body_center_crosses_fixed_circle",
              "target_delta_t_s": float(CFG["target_delta_t_s"]),
              "success_time_tol_s": float(CFG["success_time_tol_s"])}
    for actor, vehicle in (("gvt", context.ego_vehicle), ("vut", context.main_vehicle)):
        geometry = geometries.get(actor)
        if geometry is None:
            line = _route_line(vehicle)
            geometry = _route_zone_profile(line, point, radius) if line is not None else None
        if geometry is None:
            return {"valid": False, "reason": "conflict_zone_route_missing"}
        xy = np.asarray([vehicle.x, vehicle.y], dtype=float)
        _, current_s, lateral_d = project_point_to_path(xy, np.asarray(geometry["route_xy"]))
        distance = float(geometry["s_conflict"]) - current_s
        enter_distance = float(geometry["s_enter"]) - current_s
        exit_distance = float(geometry["s_exit"]) - current_s
        enter_t, prediction = _predict_vut_arrival_time(distance_m=enter_distance, speed_mps=vehicle.v, accel_mps2=vehicle.a)
        cross_t, _ = _predict_vut_arrival_time(distance_m=distance, speed_mps=vehicle.v, accel_mps2=vehicle.a)
        exit_t, _ = _predict_vut_arrival_time(distance_m=exit_distance, speed_mps=vehicle.v, accel_mps2=vehicle.a)
        length = limits.ego_length if actor == "gvt" else limits.main_length
        width = limits.ego_width if actor == "gvt" else limits.main_width
        forward = np.array([np.cos(vehicle.heading), np.sin(vehicle.heading)])
        result.update({f"{actor}_center_xy": xy.tolist(), f"{actor}_s_now_m": float(current_s),
                       f"{actor}_s_conflict_m": float(geometry["s_conflict"]),
                       f"{actor}_distance_to_conflict_m": distance,
                       f"{actor}_distance_to_zone_entry_m": enter_distance,
                       f"{actor}_distance_to_zone_exit_m": exit_distance,
                       f"{actor}_d_to_conflict_line_m": float(lateral_d),
                       f"{actor}_longitudinal_to_conflict_m": float(np.dot(point - xy, forward)),
                       f"{actor}_v_mps": float(vehicle.v), f"{actor}_tta_s": cross_t,
                       f"{actor}_zone_enter_t_s": enter_t, f"{actor}_zone_exit_t_s": exit_t,
                       f"{actor}_zone_enter_prediction": str(prediction["mode"]),
                       f"{actor}_body_zone_clearance_m": _body_zone_clearance(xy, vehicle.heading, length, width, point, radius)})
    entry_delta = _finite_difference(result["gvt_zone_enter_t_s"], result["vut_zone_enter_t_s"])
    result.update(predicted_entry_delta_t_s=entry_delta, entry_delta_t_s=entry_delta,
                  zone_delta_t_s=entry_delta, delta_t_s=entry_delta,
                  delta_t_error_s=entry_delta - float(CFG["target_delta_t_s"]),
                  predicted_cross_delta_t_s=_finite_difference(result["gvt_tta_s"], result["vut_tta_s"]))
    return result


def _predict_vut_arrival_time(
    *,
    distance_m: float,
    speed_mps: float,
    accel_mps2: float,
) -> tuple[float, dict[str, float | str]]:
    """短时使用有界加速度预测，随后保持速度；到达前停车时返回无穷。"""
    raw_distance = float(distance_m)
    raw_speed = float(speed_mps)
    raw_accel = float(accel_mps2)
    distance = max(0.0, raw_distance) if np.isfinite(raw_distance) else 0.0
    speed = max(0.0, raw_speed) if np.isfinite(raw_speed) else 0.0
    if distance <= 0.0:
        return 0.0, {
            "mode": "already_at_or_past_conflict",
            "accel_used_mps2": 0.0,
            "accel_phase_distance_m": 0.0,
            "accel_phase_end_speed_mps": speed,
        }

    accel = _clamp(
        raw_accel if np.isfinite(raw_accel) else 0.0,
        float(CFG["vut_accel_min_mps2"]),
        float(CFG["vut_accel_max_mps2"]),
    )
    horizon = max(0.0, float(CFG["vut_accel_prediction_horizon_s"]))
    if horizon <= 1e-6 or abs(accel) <= 1e-6:
        return (distance / speed if speed > 1e-3 else float("inf")), {
            "mode": "constant_speed",
            "accel_used_mps2": 0.0,
            "accel_phase_distance_m": 0.0,
            "accel_phase_end_speed_mps": speed,
        }

    # 制动预测在车辆停下时结束，不能把负速度外推到预测时域末端。
    accel_duration = horizon
    if accel < 0.0:
        accel_duration = min(horizon, speed / abs(accel))
    accel_end_speed = max(0.0, speed + accel * accel_duration)
    accel_distance = max(0.0, speed * accel_duration + 0.5 * accel * accel_duration * accel_duration)

    # 目标在加速段内，解 0.5*a*t^2 + v*t - distance = 0 并取非负根。
    if distance <= accel_distance + 1e-6:
        discriminant = max(0.0, speed * speed + 2.0 * accel * distance)
        if abs(accel) > 1e-6:
            roots = [(-speed + np.sqrt(discriminant)) / accel, (-speed - np.sqrt(discriminant)) / accel]
            non_negative_roots = [root for root in roots if root >= -1e-6]
            if non_negative_roots:
                return max(0.0, min(non_negative_roots)), {
                    "mode": "constant_accel_to_conflict",
                    "accel_used_mps2": accel,
                    "accel_phase_distance_m": accel_distance,
                    "accel_phase_end_speed_mps": accel_end_speed,
                }

    if accel_end_speed <= 1e-3:
        return float("inf"), {
            "mode": "stops_before_conflict",
            "accel_used_mps2": accel,
            "accel_phase_distance_m": accel_distance,
            "accel_phase_end_speed_mps": accel_end_speed,
        }

    return accel_duration + (distance - accel_distance) / accel_end_speed, {
        "mode": "bounded_accel_then_constant_speed",
        "accel_used_mps2": accel,
        "accel_phase_distance_m": accel_distance,
        "accel_phase_end_speed_mps": accel_end_speed,
    }


def _solve_sync_target(context: SceneContext, metrics: dict[str, Any]) -> dict[str, Any]:
    """求入区时序所需的受限加速度，禁止把过去的目标夹成未来可行目标。"""
    limits = _read_vehicle_limits(context)
    desired = float(metrics.get("vut_zone_enter_t_s", float("inf"))) + float(CFG["target_delta_t_s"])
    minimum = max(float(CFG["sync_plan_t_min_s"]), float(context.dt))
    distance = float(metrics.get("gvt_distance_to_zone_entry_m",
                                 float(metrics.get("gvt_distance_to_conflict_m", 0.0)) - float(CFG["conflict_zone_radius_m"])))
    speed = max(0.0, float(context.ego_vehicle.v))
    feasible, reason = False, "invalid_desired_arrival_time"
    required_v, required_a, average = float("inf"), 0.0, float("inf")
    if np.isfinite(desired) and desired >= minimum and distance > 0.0:
        average = distance / desired
        required_v = 2.0 * average - speed
        required_a = (required_v - speed) / desired
        if required_v > limits.max_v:
            reason = "required_speed_above_limit"
        elif required_v < 0.0:
            reason = "required_speed_below_zero"
        elif required_a < limits.min_acc:
            reason = "required_decel_below_limit"
        elif required_a > limits.max_acc:
            reason = "required_acc_above_limit"
        else:
            feasible, reason = True, ""
    else:
        # 无可达时刻时只输出有界的通行速度，不宣称命中原目标。
        required_a = (float(CFG["cruise_target_v"]) - speed) / max(float(CFG["local_target_preview_time_s"]), minimum)
    acceleration = _clamp(required_a, limits.min_acc, limits.max_acc)
    return {"sync_control_mode": "entry_time_speed_sync", "desired_gvt_arrival_t_s": desired,
            "raw_required_v_mps": average, "required_terminal_v_mps": required_v,
            "target_v_mps": _clamp(required_v if np.isfinite(required_v) else float(CFG["cruise_target_v"]), 0.0, limits.max_v),
            "target_a_mps2": acceleration,
            "plan_t_s": desired, "sync_target_feasible": feasible,
            "sync_target_infeasible_reason": reason}


def _find_forward_conflict_point(
    gvt_line: ReferenceLineView,
    main_line: ReferenceLineView,
) -> dict[str, Any]:
    """搜索双方参考线前方有效的几何冲突点。"""
    gvt_path = np.asarray(gvt_line.line, dtype=float)
    main_path = np.asarray(main_line.line, dtype=float)
    if len(gvt_path) < 2 or len(main_path) < 2:
        return {"valid": False, "reason": "line_too_short"}

    best = None
    best_score = float("inf")
    max_dist = float(CFG["max_forward_conflict_distance_m"])
    min_dist = float(CFG["min_forward_conflict_distance_m"])
    match_tol = float(CFG["conflict_point_match_tolerance_m"])
    min_angle = float(CFG["min_conflict_angle_deg"])

    for i in range(len(gvt_path) - 1):
        p1 = gvt_path[i, :2]
        p2 = gvt_path[i + 1, :2]
        seg1 = p2 - p1
        seg1_len = float(np.linalg.norm(seg1))
        if seg1_len < 1e-6:
            continue
        for j in range(len(main_path) - 1):
            q1 = main_path[j, :2]
            q2 = main_path[j + 1, :2]
            seg2 = q2 - q1
            seg2_len = float(np.linalg.norm(seg2))
            if seg2_len < 1e-6:
                continue

            closest = _segment_pair_conflict_candidate(p1, p2, q1, q2)
            if closest is None:
                continue
            distance, point, u, v = closest
            if distance > match_tol:
                continue
            angle = _segment_angle_deg(seg1, seg2)
            if angle < min_angle:
                continue

            gvt_s = float(gvt_line.cumulative_s[i]) + float(u) * seg1_len
            main_s = float(main_line.cumulative_s[j]) + float(v) * seg2_len
            gvt_distance = gvt_s - float(gvt_line.current_s)
            main_distance = main_s - float(main_line.current_s)
            if gvt_distance < min_dist or main_distance < min_dist:
                continue
            if gvt_distance > max_dist or main_distance > max_dist:
                continue

            score = gvt_distance + main_distance + 5.0 * float(distance)
            if score < best_score:
                best_score = score
                best = {
                    "valid": True,
                    "conflict_xy": [float(point[0]), float(point[1])],
                    "segment_distance_m": float(distance),
                    "conflict_angle_deg": float(angle),
                    "gvt_s_conflict_m": float(gvt_s),
                    "vut_s_conflict_m": float(main_s),
                    "gvt_distance_to_conflict_m": float(gvt_distance),
                    "vut_distance_to_conflict_m": float(main_distance),
                }

    if best is None:
        return {"valid": False, "reason": "no_forward_conflict_point"}
    return best


def _segment_pair_conflict_candidate(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> tuple[float, np.ndarray, float, float] | None:
    """返回相交或近交线段的距离、中点及段内比例。"""
    exact = _segment_intersection(p1, p2, q1, q2)
    if exact is not None:
        point, u, v = exact
        return 0.0, point, u, v

    c1, c2, u, v = _closest_points_on_segments(p1, p2, q1, q2)
    if c1 is None or c2 is None:
        return None
    midpoint = 0.5 * (c1 + c2)
    return float(np.linalg.norm(c1 - c2)), midpoint, float(u), float(v)


def _segment_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    r = p2 - p1
    s = q2 - q1
    denom = _cross2d(r, s)
    if abs(denom) < 1e-9:
        return None
    qp = q1 - p1
    u = _cross2d(qp, s) / denom
    v = _cross2d(qp, r) / denom
    if -1e-6 <= u <= 1.0 + 1e-6 and -1e-6 <= v <= 1.0 + 1e-6:
        u_clamped = _clamp(float(u), 0.0, 1.0)
        v_clamped = _clamp(float(v), 0.0, 1.0)
        return p1 + u_clamped * r, u_clamped, v_clamped
    return None


def _closest_points_on_segments(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    """计算两条二维线段的最近点。"""
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a <= 1e-12 and e <= 1e-12:
        return p1, q1, 0.0, 0.0
    if a <= 1e-12:
        s = 0.0
        t = _clamp(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e <= 1e-12:
            t = 0.0
            s = _clamp(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if abs(denom) > 1e-12:
                s = _clamp((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            tnom = b * s + f
            if tnom < 0.0:
                t = 0.0
                s = _clamp(-c / a, 0.0, 1.0)
            elif tnom > e:
                t = 1.0
                s = _clamp((b - c) / a, 0.0, 1.0)
            else:
                t = tnom / e

    c1 = p1 + float(s) * d1
    c2 = q1 + float(t) * d2
    return c1, c2, float(s), float(t)


# ==================== Actual conflict observation ====================


def _update_actual_conflict_observation(memory: dict[str, Any], metrics: dict[str, Any], now: float) -> dict[str, Any]:
    """按实际中心运动与固定圆的交点锁存事件，随后复用绝对时间。"""
    if not bool(metrics.get("valid", False)):
        return metrics
    previous_time = _optional_float(memory.get("last_observation_time_s"))
    radius = float(metrics.get("conflict_zone_radius_m", CFG["conflict_zone_radius_m"]))
    for actor in ("gvt", "vut"):
        center = metrics.get(f"{actor}_center_xy")
        if center is not None:
            _update_actor_actual_observation(memory=memory, actor=actor,
                center_xy=np.asarray(center, dtype=float), previous_xy=memory.get(f"last_{actor}_center_xy"),
                conflict_xy=np.asarray(metrics["conflict_xy"], dtype=float), radius=radius,
                prev_time=previous_time, now=now)
            memory[f"last_{actor}_center_xy"] = list(center)
        signed = _signed_conflict_distance(metrics, actor)
        previous_signed = _optional_float(memory.get(f"last_{actor}_signed_distance_m"))
        if memory.get(f"{actor}_cross_time_s") is None and previous_signed is not None and previous_signed > 0.0 and signed <= 0.0:
            memory[f"{actor}_cross_time_s"] = _threshold_cross_time(previous_signed, signed, previous_time, now, 0.0)
        memory[f"last_{actor}_signed_distance_m"] = signed
        for event in ("enter", "exit"):
            observed = _optional_float(memory.get(f"{actor}_zone_{event}_time_s"))
            if observed is not None:
                metrics[f"{actor}_zone_{event}_t_s"] = observed - now
                if event == "enter":
                    metrics[f"{actor}_zone_enter_prediction"] = "observed"
    memory["last_observation_time_s"] = now
    metrics.update(_build_actual_result_metrics(memory))
    if bool(metrics["actual_entry_delta_t_ready"]):
        delta, source = float(metrics["actual_entry_delta_t_s"]), "latched"
    else:
        delta = _finite_difference(metrics.get("gvt_zone_enter_t_s"), metrics.get("vut_zone_enter_t_s"))
        source = "mixed" if any(memory.get(f"{actor}_zone_enter_time_s") is not None for actor in ("gvt", "vut")) else "predicted"
    metrics.update(entry_delta_t_s=delta, zone_delta_t_s=delta, delta_t_s=delta,
                   delta_t_error_s=delta - float(CFG["target_delta_t_s"]), zone_delta_source=source)
    if metrics["actual_result_ready"] and not memory.get("actual_result_printed", False):
        _print_actual_conflict_result(metrics)
        memory["actual_result_printed"] = True
    return metrics


def _update_actor_actual_observation(*, memory: dict[str, Any], actor: str, center_xy: np.ndarray,
                                     previous_xy: list[float] | None, conflict_xy: np.ndarray,
                                     radius: float, prev_time: float | None, now: float) -> None:
    """同一帧可记录入区和出区；首帧已经在区内时不捏造历史进入时刻。"""
    if previous_xy is None or prev_time is None or now <= prev_time:
        if np.linalg.norm(center_xy - conflict_xy) < radius - 1e-8 and memory.get(f"{actor}_zone_enter_time_s") is None:
            memory[f"{actor}_entry_censored"] = True
        return
    previous = np.asarray(previous_xy, dtype=float)
    movement = center_xy - previous
    for fraction in _segment_circle_roots(previous, center_xy, conflict_xy, radius):
        hit = previous + fraction * movement
        direction = float(np.dot(hit - conflict_xy, movement))
        event = "enter" if direction < -1e-9 else "exit" if direction > 1e-9 else ""
        if event:
            key = f"{actor}_zone_{event}_time_s"
            if memory.get(key) is None:
                memory[key] = float(prev_time) + fraction * (now - float(prev_time))


def _build_actual_result_metrics(memory: dict[str, Any]) -> dict[str, Any]:
    """同步字段全部对应入区时差，过点和占用区间另有字段。"""
    times = {f"{actor}_{event}": _optional_float(memory.get(f"{actor}_{event}_time_s"))
             for actor in ("gvt", "vut") for event in ("zone_enter", "zone_exit", "cross")}
    g_enter, v_enter = times["gvt_zone_enter"], times["vut_zone_enter"]
    entry_ready = g_enter is not None and v_enter is not None
    delta = g_enter - v_enter if entry_ready else float("nan")
    error = delta - float(CFG["target_delta_t_s"])
    result = "pending" if not entry_ready else "within_tolerance" if abs(error) <= float(CFG["success_time_tol_s"]) else "outside_tolerance"
    cross_ready = times["gvt_cross"] is not None and times["vut_cross"] is not None
    cross_delta = times["gvt_cross"] - times["vut_cross"] if cross_ready else float("nan")
    pet_ready = entry_ready and times["gvt_zone_exit"] is not None and times["vut_zone_exit"] is not None
    pet, overlap, first = float("nan"), float("nan"), "pending"
    if pet_ready:
        if times["gvt_zone_exit"] <= v_enter:
            pet, overlap, first = v_enter - times["gvt_zone_exit"], 0.0, "gvt"
        elif times["vut_zone_exit"] <= g_enter:
            pet, overlap, first = g_enter - times["vut_zone_exit"], 0.0, "vut"
        else:
            pet, overlap, first = 0.0, min(times["gvt_zone_exit"], times["vut_zone_exit"]) - max(g_enter, v_enter), "overlap"
    output = {"actual_entry_delta_t_ready": entry_ready, "actual_entry_delta_t_s": delta,
              "actual_entry_delta_t_error_s": error, "actual_entry_delta_t_result": result,
              "actual_cross_delta_t_ready": cross_ready, "actual_cross_delta_t_s": cross_delta,
              "actual_zone_delta_ready": entry_ready, "actual_zone_delta_t_s": delta,
              "actual_delta_t_ready": entry_ready, "actual_delta_t_s": delta,
              "actual_delta_t_error_s": error, "actual_delta_t_abs_error_s": abs(error),
              "actual_delta_t_result": result, "actual_result_ready": entry_ready,
              "actual_interaction_result_ready": pet_ready, "actual_pet_ready": pet_ready,
              "actual_pet_like_s": pet, "actual_zone_overlap_s": overlap,
              "actual_zone_first_actor": first,
              "sync_result": "pending" if not entry_ready else "met" if result == "within_tolerance" else "missed"}
    output.update({key + "_time_s": value if value is not None else float("nan") for key, value in times.items()})
    return output


def _signed_conflict_distance(metrics: dict[str, Any], actor: str) -> float:
    """用锁定路线的有符号弧长观测过点，不使用车头方向投影替代。"""
    value = _optional_float(metrics.get(f"{actor}_distance_to_conflict_m"))
    return value if value is not None else float("nan")


def _threshold_cross_time(
    prev_value: float | None,
    current_value: float,
    prev_time: float | None,
    now: float,
    threshold: float,
) -> float:
    if (
        prev_value is None
        or prev_time is None
        or not np.isfinite(prev_value)
        or not np.isfinite(current_value)
        or float(now) <= float(prev_time)
        or abs(float(prev_value) - float(current_value)) < 1e-6
    ):
        return float(now)

    prev_offset = float(prev_value) - float(threshold)
    current_offset = float(current_value) - float(threshold)
    if prev_offset == 0.0:
        return float(prev_time)
    if prev_offset * current_offset > 0.0:
        return float(now)

    ratio = _clamp(prev_offset / (prev_offset - current_offset), 0.0, 1.0)
    return float(prev_time) + ratio * (float(now) - float(prev_time))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


# ==================== Conditions and helpers ====================


def _print_actual_conflict_result(metrics: dict[str, Any]) -> None:
    """只打印入区同步判定；后续交互继续写入每帧指标。"""
    print("[intersection_conflict] actual_entry_result "
          f"delta_t={float(metrics.get('actual_entry_delta_t_s', float('nan'))):.3f}s "
          f"target={float(CFG['target_delta_t_s']):.3f}s "
          f"result={metrics.get('sync_result', 'pending')}", flush=True)


def _route_line(vehicle: Any | None) -> ReferenceLineView | None:
    if vehicle is None:
        return None
    route_intent = getattr(vehicle, "route_intent", None)
    if route_intent is None or not bool(getattr(route_intent, "valid", False)):
        return None
    line = getattr(route_intent, "route_line", None)
    if line is None:
        return None
    return line


def _route_intent_reason(vehicle: Any | None) -> str:
    if vehicle is None:
        return "no_vehicle"
    route_intent = getattr(vehicle, "route_intent", None)
    if route_intent is None:
        return "no_route_intent"
    if bool(getattr(route_intent, "valid", False)):
        return "route_line_ok"
    return str(getattr(route_intent, "reason", "route_intent_invalid"))


def _select_route_aligned_local_line(context: SceneContext) -> tuple[ReferenceLineView, dict[str, Any]]:
    """选择符合全局下一步车道动作的局部候选线，而非按远端坐标选线。"""

    ego = context.ego_vehicle
    lines = list(ego.candidate_ref_lines) or [ego.current_ref_line]
    route_intent = getattr(ego, "route_intent", None)
    planned_path = [str(lane_id) for lane_id in getattr(route_intent, "planned_path", []) if str(lane_id)]
    planned_actions = list(getattr(route_intent, "planned_actions", []) or [])
    current_lane_id = str(ego.lane_id or ego.current_ref_line.lane_id)

    if not planned_path or current_lane_id not in planned_path:
        return ego.current_ref_line, {
            "route_action_used": "fallback_current_line",
            "route_alignment_reason": "current_lane_not_in_planned_path",
            "selected_local_lane_sequence": _normalize_lane_sequence(ego.current_ref_line.lane_sequence),
        }

    current_path_index = max(index for index, lane_id in enumerate(planned_path) if lane_id == current_lane_id)
    next_path_index = min(current_path_index + 1, len(planned_path) - 1)
    next_action = (
        str(planned_actions[next_path_index].get("action", "successor"))
        if next_path_index < len(planned_actions) and isinstance(planned_actions[next_path_index], dict)
        else "successor"
    )
    lateral_change = next_path_index > current_path_index and next_action in {"left_change", "right_change"}
    desired_path_index = next_path_index if lateral_change else current_path_index
    desired_lane_id = planned_path[desired_path_index]

    candidates = [line for line in lines if str(line.lane_id) == desired_lane_id]
    if not candidates and desired_path_index != current_path_index:
        candidates = [line for line in lines if str(line.lane_id) == current_lane_id]
    if not candidates:
        candidates = [ego.current_ref_line]

    def route_overlap(line: ReferenceLineView) -> int:
        lane_sequence = _normalize_lane_sequence(line.lane_sequence)
        expected = planned_path[desired_path_index:]
        overlap = 0
        for candidate_lane, expected_lane in zip(lane_sequence, expected):
            if candidate_lane != expected_lane:
                break
            overlap += 1
        return overlap

    ego_xy = np.asarray([ego.x, ego.y], dtype=float)

    def score(line: ReferenceLineView) -> tuple[int, float, int]:
        _idx, _s, lateral_d = project_point_to_path(ego_xy, line.line)
        return (-route_overlap(line), float(abs(lateral_d)), int(line.line_index))

    selected_line = min(candidates, key=score)
    return selected_line, {
        "route_action_used": next_action,
        "route_alignment_reason": "next_lateral_action" if lateral_change else "current_route_lane",
        "route_current_lane_id": current_lane_id,
        "route_desired_local_lane_id": desired_lane_id,
        "route_current_path_index": int(current_path_index),
        "route_desired_path_index": int(desired_path_index),
        "selected_local_lane_sequence": _normalize_lane_sequence(selected_line.lane_sequence),
    }


def _normalize_lane_sequence(lane_sequence: list[str] | tuple[str, ...] | None) -> list[str]:
    if lane_sequence is None:
        return []
    return [str(lane_id) for lane_id in lane_sequence]


def _segment_angle_deg(seg1: np.ndarray, seg2: np.ndarray) -> float:
    h1 = float(np.arctan2(seg1[1], seg1[0]))
    h2 = float(np.arctan2(seg2[1], seg2[0]))
    angle = abs(float(np.degrees(normalize_angle(h1 - h2))))
    return min(angle, 180.0 - angle)


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _read_vehicle_limits(context: SceneContext) -> VehicleLimits:
    runtime_frame = context.runtime_frame
    config = {} if runtime_frame is None else runtime_frame.config
    size = config.get("size", {})
    constrain = config.get("constrain", {})
    ego_length = float(size.get("L", 4.5))
    ego_width = float(size.get("W", 1.8))
    main_actor = None if runtime_frame is None else runtime_frame.main_actor
    main_length = float(getattr(main_actor, "length", 0.0) or ego_length)
    main_width = float(getattr(main_actor, "width", 0.0) or ego_width)
    return VehicleLimits(
        ego_length=ego_length,
        main_length=main_length,
        ego_width=ego_width,
        main_width=main_width,
        max_v=float(constrain.get("max_v", 6.94)),
        min_acc=float(constrain.get("min_acc", -2.0)),
        max_acc=float(constrain.get("max_acc", 2.0)),
    )
