"""盲区冲突场景算法。

场景语义：
- 受控车辆是背景车 GVT，对应 ``context.ego_vehicle``。
- 被测主车是 VUT，对应 ``context.main_vehicle``。
- ``CFG["scene_side"]`` 表示 VUT 的预期换道方向，取值为 ``left`` 或 ``right``。
- VUT 尚未实际换道时，GVT 必须位于对应侧相邻车道后方，并仅通过纵向速度控制保持在 VUT 侧后方盲区。
- VUT 产生稳定横向换道趋势后，算法预测 VUT 到达目标车道时的位置，并把外部配置的完成时 THW 换算成目标净距进行控制。
- TTC 只作为调试指标输出，不作为主动规避条件；算法只保留硬碰撞几何约束。
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from planning_runtime.scene.types import ReferenceLineView, SceneContext
from traj_planner.tools import get_path_point_at_s, normalize_angle, project_point_to_path


@dataclass(frozen=True)
class VehicleLimits:
    """盲区场景使用的车辆尺寸和平台运动约束。"""

    ego_length: float
    main_length: float
    ego_width: float
    main_width: float
    max_v: float
    min_acc: float
    max_acc: float


CFG: dict[str, Any] = {
    # 严格场景几何检查默认开启：初始几何不满足盲区构造条件时，输出显式错误并令 GVT 停车。
    "strict_scene_geometry": True,
    # VUT 预期换道方向。算法不控制 VUT，只根据该方向构造同侧 GVT 盲区目标。
    "scene_side": "left",
    "cruise_target_v": 5.0,
    "prepare_plan_t_s": 1.5,
    "hold_plan_t_s": 1.2,
    "active_plan_t_s": 1.0,
    "abort_plan_t_s": 1.0,
    "complete_decel_mps2": 2.5,
    "complete_stop_v_mps": 0.05,
    "active_min_external_target_progress_m": 0.05,
    "active_terminal_speed_ignore_acc_limit": True,
    "search_rel_s_min_m": -35.0,
    "search_rel_s_max_m": -1.0,
    "blind_rel_s_min_m": -8.6,
    "blind_rel_s_max_m": -5.1,
    "blind_rel_s_target_m": -6.1,
    "rel_s_control_gain": 0.35,
    "active_rel_s_control_gain": 0.28,
    "completion_prediction_gain": 0.45,
    "prepare_max_speed_adv_mps": 3.0,
    "prepare_max_speed_lag_mps": 2.5,
    "active_max_speed_adv_mps": 2.0,
    "active_max_speed_lag_mps": 3.0,
    "max_heading_diff_deg": 20.0,
    "max_ego_d_on_target_line_m": 1.5,
    "side_lateral_min_m": 1.8,
    "side_lateral_max_m": 5.5,
    "min_road_remaining_m": 18.0,
    "min_target_progress_m": 2.0,
    "prepare_timeout_s": 12.0,
    "hold_success_duration_s": 3.0,
    "hold_max_wait_s": 60.0,
    "blind_zone_lost_max_frames": 8,
    # VUT 换道开始判定。基准值是 VUT 初始到目标侧车道中心线的横向距离。
    "lane_change_start_d_m": 0.35,
    "lane_change_lateral_v_min_mps": 0.15,
    "lane_change_heading_min_deg": 1.0,
    "lane_change_debounce_frames": 3,
    "lane_change_complete_d_m": 0.6,
    "lane_change_min_lateral_v_mps": 0.15,
    "lane_change_prediction_t_min_s": 0.1,
    "lane_change_prediction_t_max_s": 6.0,
    "lane_change_prediction_time_scale": 0.8,
    "lane_change_prediction_time_buffer_s": 0.05,
    "completion_blend_start_progress": 0.50,
    # 外部显式配置的完成时 THW。内部会把 THW 换算成同车道完成时目标净距。
    "completion_target_thw_s": 3.0,
    "completion_thw_tol_s": 0.2,
    "completion_min_net_gap_m": 1.0,
    "completion_success_frames": 3,
    "thw_v_floor_mps": 0.5,
    # 只保留硬碰撞几何保护；TTC 只写入 scene_debug。
    "hard_collision_margin_m": 0.2,
    "abort_hold_s": 1.0,
    "abort_cooldown_s": 2.0,
}


SCENE_MEMORY_TEMPLATE: dict[str, Any] = {
    "mode": "APPROACH",
    "cooldown_until_time_s": -1.0,
    "prepare_start_time_s": -1.0,
    "hold_start_time_s": -1.0,
    "active_start_time_s": -1.0,
    "complete_start_time_s": -1.0,
    "abort_start_time_s": -1.0,
    "locked_line_index": None,
    "locked_target_lane_id": "",
    "locked_target_lane_sequence": [],
    "baseline_main_abs_d_on_target": -1.0,
    "prev_main_abs_d_on_target": -1.0,
    "lane_change_start_hits": 0,
    "blind_zone_hits": 0,
    "blind_zone_miss_hits": 0,
    "hold_success_latched": False,
    "completion_success_hits": 0,
    "last_metrics": None,
}


SCENE_MEMORY: dict[str, Any] = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== 外部入口 ====================


def blind_spot_algorithm(context: SceneContext) -> dict[str, Any]:
    """盲区场景算法主入口。"""
    global SCENE_MEMORY
    start_ms = time.time() * 1000.0
    if int(context.step_index) == 0:
        _reset_scene_memory()
    output, SCENE_MEMORY = _scene_fsm_step(context, SCENE_MEMORY)
    end_ms = time.time() * 1000.0
    print(f"Blind spot algorithm execution time: {end_ms - start_ms} ms")
    return output


# ==================== 状态机 ====================


def _scene_fsm_step(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行一帧盲区场景状态机。"""
    now = float(context.time_s)
    mode = str(memory.get("mode", "APPROACH"))

    if mode == "COOLDOWN":
        if now < float(memory["cooldown_until_time_s"]):
            return _handle_cooldown(context, memory, now)
        memory["mode"] = "APPROACH"
        mode = "APPROACH"

    if mode not in {"SAFE_ABORT", "COOLDOWN"} and context.main_vehicle is None:
        return _handle_no_valid_main(context, memory)

    if mode == "APPROACH":
        return _handle_approach(context, memory, now)
    if mode == "PREPARE_BLIND_ZONE":
        return _handle_prepare_blind_zone(context, memory, now)
    if mode == "HOLD_BLIND_ZONE":
        return _handle_hold_blind_zone(context, memory, now)
    if mode == "LANE_CHANGE_ACTIVE":
        return _handle_lane_change_active(context, memory, now)
    if mode == "COMPLETE":
        return _handle_complete(context, memory, now)
    if mode == "SAFE_ABORT":
        return _handle_safe_abort(context, memory, now)
    return _handle_unknown_mode(context, memory)


# ==================== 状态处理 ====================


def _handle_cooldown(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 COOLDOWN：失败或中止后冷却，避免连续重复触发。"""
    return _build_keep_lane_output(
        context,
        reason="blind_spot_cooldown",
        scene_status="cooldown",
    ), memory


def _handle_no_valid_main(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 VUT 缺失：严格模式下标记场景输入错误。"""
    memory["mode"] = "APPROACH"
    _reset_execution_memory(memory)
    if _strict_scene_geometry_enabled():
        metrics = {
            "main_valid": False,
            "invalid_scene_reasons": ["no_main_vehicle"],
        }
        return _build_invalid_scene_output(
            context,
            reason="blind_spot_no_valid_main",
            scene_debug=metrics,
        ), memory
    return _build_keep_lane_output(
        context,
        reason="blind_spot_no_valid_main",
        scene_status="waiting_valid_main",
    ), memory


def _handle_approach(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 APPROACH：检查 GVT 是否在 VUT 预期换道侧相邻车道后方。"""
    target_line = _select_current_side_lane(context)
    metrics = _compute_blind_spot_metrics(context, target_line, memory)
    memory["last_metrics"] = copy.deepcopy(metrics)
    if not _is_approach_candidate(metrics):
        metrics["invalid_scene_reasons"] = _invalid_scene_reasons(metrics, require_search_window=True)
        if _strict_scene_geometry_enabled():
            return _build_invalid_scene_output(
                context,
                reason="blind_spot_invalid_initial_geometry",
                scene_debug=metrics,
            ), memory
        return _build_keep_lane_output(
            context,
            reason="blind_spot_wait_candidate",
            scene_status="approach_wait_candidate",
            scene_debug=metrics,
        ), memory

    _enter_prepare_blind_zone(memory, target_line, metrics, now)
    if bool(metrics["in_blind_zone"]):
        _enter_hold_blind_zone(memory, metrics, now)
        return _build_hold_output(
            context,
            target_line,
            metrics,
            memory,
            reason="blind_spot_hold_enter",
            scene_status="hold_blind_zone",
        ), memory

    return _build_prepare_output(
        context,
        target_line,
        metrics,
        memory,
        reason="blind_spot_prepare_enter",
        scene_status="prepare_blind_zone",
    ), memory


def _handle_prepare_blind_zone(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 PREPARE_BLIND_ZONE：GVT 从后方纵向接近盲区窗口。"""
    target_line = _resolve_target_line(context, memory)
    metrics = _compute_blind_spot_metrics(context, target_line, memory)
    memory["last_metrics"] = copy.deepcopy(metrics)
    if _should_hard_abort(metrics):
        _enter_safe_abort(memory, now)
        return _build_abort_output(context, target_line, metrics, reason="blind_spot_hard_collision_risk"), memory

    if not _is_prepare_candidate(metrics):
        metrics["invalid_scene_reasons"] = _invalid_scene_reasons(metrics, require_search_window=False)
        if _strict_scene_geometry_enabled():
            memory["mode"] = "APPROACH"
            _reset_execution_memory(memory)
            return _build_invalid_scene_output(
                context,
                reason="blind_spot_candidate_lost",
                scene_debug=metrics,
            ), memory
        memory["mode"] = "APPROACH"
        _reset_execution_memory(memory)
        return _build_keep_lane_output(
            context,
            reason="blind_spot_candidate_lost",
            scene_status="approach_wait_candidate",
            scene_debug=metrics,
        ), memory

    elapsed = max(0.0, now - float(memory.get("prepare_start_time_s", now)))
    if elapsed >= float(CFG["prepare_timeout_s"]):
        _enter_cooldown(memory, now)
        return _build_keep_lane_output(
            context,
            reason="blind_spot_prepare_timeout",
            scene_status="prepare_timeout",
            scene_debug=metrics,
        ), memory

    if bool(metrics["in_blind_zone"]):
        _enter_hold_blind_zone(memory, metrics, now)
        return _build_hold_output(
            context,
            target_line,
            metrics,
            memory,
            reason="blind_spot_hold_enter",
            scene_status="hold_blind_zone",
        ), memory

    return _build_prepare_output(
        context,
        target_line,
        metrics,
        memory,
        reason="blind_spot_prepare",
        scene_status="prepare_blind_zone",
    ), memory


def _handle_hold_blind_zone(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 HOLD_BLIND_ZONE：VUT 未换道时，GVT 持续保持在对应侧后方盲区。"""
    target_line = _resolve_target_line(context, memory)
    metrics = _compute_blind_spot_metrics(context, target_line, memory)
    _update_lane_change_detection(memory, metrics)
    _update_blind_zone_hold(memory, metrics, now)
    memory["last_metrics"] = copy.deepcopy(metrics)

    if _should_hard_abort(metrics):
        _enter_safe_abort(memory, now)
        return _build_abort_output(context, target_line, metrics, reason="blind_spot_hard_collision_risk"), memory

    if not _is_prepare_candidate(metrics):
        metrics["invalid_scene_reasons"] = _invalid_scene_reasons(metrics, require_search_window=False)
        if _strict_scene_geometry_enabled():
            memory["mode"] = "APPROACH"
            _reset_execution_memory(memory)
            return _build_invalid_scene_output(
                context,
                reason="blind_spot_hold_geometry_lost",
                scene_debug=metrics,
            ), memory

    if bool(metrics["lane_change_started"]):
        print(
            "主车开始换道 "
            f"rel_s={float(metrics.get('rel_s_m', float('nan'))):.3f}m "
            f"progress={float(metrics.get('lane_change_progress_ratio', 0.0)):.3f} "
            f"main_abs_d={float(metrics.get('main_abs_d_on_target_line_m', float('nan'))):.3f}m "
            f"lateral_v={float(metrics.get('main_lateral_v_toward_target_mps', float('nan'))):.3f}m/s",
            flush=True,
        )
        _enter_lane_change_active(memory, now)
        return _build_active_output(
            context,
            target_line,
            metrics,
            memory,
            reason="blind_spot_lane_change_active",
            scene_status="lane_change_active",
        ), memory

    if int(memory.get("blind_zone_miss_hits", 0)) >= int(CFG["blind_zone_lost_max_frames"]):
        memory["mode"] = "PREPARE_BLIND_ZONE"
        memory["prepare_start_time_s"] = float(now)
        return _build_prepare_output(
            context,
            target_line,
            metrics,
            memory,
            reason="blind_spot_recover_blind_zone",
            scene_status="prepare_recover_blind_zone",
        ), memory

    hold_elapsed = max(0.0, now - float(memory.get("hold_start_time_s", now)))
    status = "blind_zone_hold_success" if bool(memory.get("hold_success_latched", False)) else "hold_blind_zone"
    debug = copy.deepcopy(metrics)
    debug["hold_elapsed_s"] = float(hold_elapsed)
    debug["hold_success_duration_s"] = float(CFG["hold_success_duration_s"])
    return _build_hold_output(
        context,
        target_line,
        debug,
        memory,
        reason="blind_spot_hold",
        scene_status=status,
    ), memory


def _handle_lane_change_active(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 LANE_CHANGE_ACTIVE：VUT 换道过程中控制完成时 THW 对应的净距。"""
    target_line = _resolve_target_line(context, memory)
    metrics = _compute_blind_spot_metrics(context, target_line, memory)
    _update_lane_change_detection(memory, metrics)
    memory["last_metrics"] = copy.deepcopy(metrics)

    if _should_hard_abort(metrics):
        _enter_safe_abort(memory, now)
        return _build_abort_output(context, target_line, metrics, reason="blind_spot_hard_collision_risk"), memory

    completion_success = _is_completion_success(metrics)
    if bool(metrics.get("vut_lane_change_complete", False)):
        metrics["completion_success"] = bool(completion_success)
        _print_lane_change_completion_thw(metrics, completion_success)
        _enter_complete(memory, now)
        return _build_complete_output(
            context,
            target_line,
            metrics,
            reason="blind_spot_complete" if completion_success else "blind_spot_complete_thw_miss_stop",
            scene_status="lane_change_completion_thw_met" if completion_success else "lane_change_complete_thw_miss_stop",
        ), memory

    if not _is_active_candidate(metrics):
        metrics["invalid_scene_reasons"] = _invalid_scene_reasons(metrics, require_search_window=False)
        if _strict_scene_geometry_enabled():
            memory["mode"] = "APPROACH"
            _reset_execution_memory(memory)
            return _build_invalid_scene_output(
                context,
                reason="blind_spot_active_geometry_lost",
                scene_debug=metrics,
            ), memory

    return _build_active_output(
        context,
        target_line,
        metrics,
        memory,
        reason="blind_spot_lane_change_active",
        scene_status="lane_change_active",
    ), memory


def _handle_complete(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 COMPLETE：场景已完成后保持完成态，避免同一轮场景任务重复触发。"""
    target_line = _resolve_target_line(context, memory)
    metrics = _compute_blind_spot_metrics(context, target_line, memory)
    complete_start = float(memory.get("complete_start_time_s", now))
    metrics["complete_elapsed_s"] = max(0.0, now - complete_start)
    return _build_complete_output(
        context,
        target_line,
        metrics,
        reason="blind_spot_complete_hold",
        scene_status="complete",
    ), memory


def _handle_safe_abort(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 SAFE_ABORT：只针对硬碰撞几何或不可恢复异常进行减速中止。"""
    target_line = _resolve_target_line(context, memory)
    metrics = _compute_blind_spot_metrics(context, target_line, memory)
    abort_start = float(memory.get("abort_start_time_s", now))
    if abort_start < 0.0:
        abort_start = float(now)
        memory["abort_start_time_s"] = abort_start
    if now - abort_start >= float(CFG["abort_hold_s"]):
        _enter_cooldown(memory, now)
    return _build_abort_output(context, target_line, metrics, reason="blind_spot_safe_abort"), memory


def _handle_unknown_mode(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理未知状态：重置为 APPROACH。"""
    memory["mode"] = "APPROACH"
    _reset_execution_memory(memory)
    return _build_keep_lane_output(
        context,
        reason="blind_spot_unknown_mode",
        scene_status="baseline_keep_lane",
    ), memory


# ==================== 状态切换 ====================


def _enter_prepare_blind_zone(
    memory: dict[str, Any],
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    now: float,
) -> None:
    """锁定 GVT 当前所在的目标侧车道，并进入盲区接近阶段。"""
    memory["mode"] = "PREPARE_BLIND_ZONE"
    memory["prepare_start_time_s"] = float(now)
    _lock_target_line(memory, target_line)
    _reset_lane_change_detection(memory, metrics)
    memory["blind_zone_hits"] = 0
    memory["blind_zone_miss_hits"] = 0
    memory["hold_success_latched"] = False


def _enter_hold_blind_zone(memory: dict[str, Any], metrics: dict[str, Any], now: float) -> None:
    """进入盲区保持阶段。"""
    memory["mode"] = "HOLD_BLIND_ZONE"
    memory["hold_start_time_s"] = float(now)
    _reset_lane_change_detection(memory, metrics)
    memory["blind_zone_hits"] = 0
    memory["blind_zone_miss_hits"] = 0


def _enter_lane_change_active(memory: dict[str, Any], now: float) -> None:
    """进入 VUT 换道交互阶段。"""
    memory["mode"] = "LANE_CHANGE_ACTIVE"
    memory["active_start_time_s"] = float(now)
    memory["completion_success_hits"] = 0


def _enter_complete(memory: dict[str, Any], now: float) -> None:
    """进入场景完成阶段。"""
    memory["mode"] = "COMPLETE"
    memory["complete_start_time_s"] = float(now)


def _enter_safe_abort(memory: dict[str, Any], now: float) -> None:
    """进入硬安全中止阶段。"""
    memory["mode"] = "SAFE_ABORT"
    memory["abort_start_time_s"] = float(now)


def _enter_cooldown(memory: dict[str, Any], now: float) -> None:
    """进入冷却阶段，并清理本轮场景上下文。"""
    memory["mode"] = "COOLDOWN"
    memory["cooldown_until_time_s"] = float(now) + float(CFG["abort_cooldown_s"])
    _reset_execution_memory(memory)


def _lock_target_line(memory: dict[str, Any], target_line: ReferenceLineView | None) -> None:
    """记录 GVT 所在目标侧车道，后续状态按该车道恢复参考线。"""
    if target_line is None:
        memory["locked_line_index"] = None
        memory["locked_target_lane_id"] = ""
        memory["locked_target_lane_sequence"] = []
        return
    memory["locked_line_index"] = int(target_line.line_index)
    memory["locked_target_lane_id"] = str(target_line.lane_id)
    memory["locked_target_lane_sequence"] = _normalize_lane_sequence(target_line.lane_sequence)


def _reset_lane_change_detection(memory: dict[str, Any], metrics: dict[str, Any]) -> None:
    """记录 VUT 到目标侧车道中心线的初始横向距离，作为换道检测基准。"""
    baseline = float(metrics.get("main_abs_d_on_target_line_m", -1.0))
    memory["baseline_main_abs_d_on_target"] = baseline
    memory["prev_main_abs_d_on_target"] = baseline
    memory["lane_change_start_hits"] = 0


def _reset_execution_memory(memory: dict[str, Any]) -> None:
    """清空场景执行缓存；冷却结束时间由调用方单独维护。"""
    memory["prepare_start_time_s"] = -1.0
    memory["hold_start_time_s"] = -1.0
    memory["active_start_time_s"] = -1.0
    memory["complete_start_time_s"] = -1.0
    memory["abort_start_time_s"] = -1.0
    memory["locked_line_index"] = None
    memory["locked_target_lane_id"] = ""
    memory["locked_target_lane_sequence"] = []
    memory["baseline_main_abs_d_on_target"] = -1.0
    memory["prev_main_abs_d_on_target"] = -1.0
    memory["lane_change_start_hits"] = 0
    memory["blind_zone_hits"] = 0
    memory["blind_zone_miss_hits"] = 0
    memory["hold_success_latched"] = False
    memory["completion_success_hits"] = 0
    memory["last_metrics"] = None


def _reset_scene_memory() -> None:
    """新一轮离线/在线场景开始时重置全局状态。"""
    global SCENE_MEMORY
    SCENE_MEMORY = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== 输出构造 ====================


def _build_keep_lane_output(
    context: SceneContext,
    reason: str,
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造默认当前车道巡航目标，仅在非严格模式或冷却态使用。"""
    limits = _read_vehicle_limits(context)
    target_v = _clamp(float(CFG["cruise_target_v"]), 0.0, limits.max_v)
    return _build_line_target_output(
        context=context,
        target_line=context.ego_vehicle.current_ref_line,
        target_v=target_v,
        target_a=0.0,
        plan_t=2.0,
        reason=reason,
        scene_status=scene_status,
        scene_detail=scene_detail or reason,
        scene_debug=scene_debug,
    )


def _build_invalid_scene_output(
    context: SceneContext,
    reason: str,
    scene_debug: dict[str, Any],
) -> dict[str, Any]:
    """构造严格模式下的场景构造错误输出。

    该输出不继续制造盲区目标，而是让 GVT 在当前车道生成零速目标，并在
    ``scene_debug["invalid_scene_reasons"]`` 中给出不满足盲区场景的具体原因。
    """
    limits = _read_vehicle_limits(context)
    plan_t = 1.0
    target_v = 0.0
    target_a = _clamp(
        (target_v - float(context.ego_vehicle.v)) / max(plan_t, 1e-3),
        limits.min_acc,
        limits.max_acc,
    )
    debug = copy.deepcopy(scene_debug)
    debug["strict_scene_geometry"] = True
    debug.setdefault("invalid_scene_reasons", ["unknown_invalid_scene_geometry"])
    output = _build_line_target_output(
        context=context,
        target_line=context.ego_vehicle.current_ref_line,
        target_v=target_v,
        target_a=target_a,
        plan_t=plan_t,
        reason=reason,
        scene_status="invalid_scene_geometry",
        scene_detail="blind_spot_scene_geometry_invalid",
        scene_debug=debug,
    )
    output["scene_error"] = True
    output["control_mode"] = "scene_invalid"
    return output


def _build_prepare_output(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    memory: dict[str, Any],
    reason: str,
    scene_status: str,
) -> dict[str, Any]:
    """构造 PREPARE 阶段输出：沿目标侧车道纵向接近盲区目标位置。"""
    del memory
    target_rel_s = float(CFG["blind_rel_s_target_m"])
    return _build_rel_s_control_output(
        context=context,
        target_line=target_line,
        metrics=metrics,
        target_rel_s=target_rel_s,
        gain=float(CFG["rel_s_control_gain"]),
        max_adv=float(CFG["prepare_max_speed_adv_mps"]),
        max_lag=float(CFG["prepare_max_speed_lag_mps"]),
        plan_t=float(CFG["prepare_plan_t_s"]),
        reason=reason,
        scene_status=scene_status,
    )


def _build_hold_output(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    memory: dict[str, Any],
    reason: str,
    scene_status: str,
) -> dict[str, Any]:
    """构造 HOLD 阶段输出：保持 GVT 在 VUT 侧后方盲区。"""
    del memory
    target_rel_s = float(CFG["blind_rel_s_target_m"])
    return _build_rel_s_control_output(
        context=context,
        target_line=target_line,
        metrics=metrics,
        target_rel_s=target_rel_s,
        gain=float(CFG["rel_s_control_gain"]),
        max_adv=float(CFG["prepare_max_speed_adv_mps"]),
        max_lag=float(CFG["prepare_max_speed_lag_mps"]),
        plan_t=float(CFG["hold_plan_t_s"]),
        reason=reason,
        scene_status=scene_status,
    )


def _build_active_output(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    memory: dict[str, Any],
    reason: str,
    scene_status: str,
) -> dict[str, Any]:
    """构造 ACTIVE 阶段输出：反解 VUT 换道完成时的目标 THW 终端状态。"""
    del memory
    precise_target = _solve_completion_thw_terminal(context, target_line, metrics)
    if bool(precise_target.get("valid", False)):
        return _build_precise_completion_output(
            context=context,
            target_line=target_line,
            metrics=metrics,
            precise_target=precise_target,
            reason=reason,
            scene_status=scene_status,
        )

    fallback_target = _compute_active_target_rel_s(metrics)
    fallback_debug = copy.deepcopy(fallback_target)
    fallback_debug.update(
        {
            "completion_precise_valid": False,
            "completion_precise_invalid_reason": str(precise_target.get("reason", "unknown")),
        }
    )
    return _build_rel_s_control_output(
        context=context,
        target_line=target_line,
        metrics=metrics,
        target_rel_s=float(fallback_target["target_rel_s_m"]),
        gain=float(CFG["active_rel_s_control_gain"]),
        max_adv=float(CFG["active_max_speed_adv_mps"]),
        max_lag=float(CFG["active_max_speed_lag_mps"]),
        plan_t=float(CFG["active_plan_t_s"]),
        reason=reason,
        scene_status=scene_status,
        extra_speed_delta=float(fallback_target["prediction_speed_delta_mps"]),
        extra_debug=fallback_debug,
    )


def _build_precise_completion_output(
    *,
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    precise_target: dict[str, Any],
    reason: str,
    scene_status: str,
) -> dict[str, Any]:
    """按反解出的换道完成终端 ``s/v/t`` 构造外部目标。"""
    line = target_line if target_line is not None else context.ego_vehicle.current_ref_line
    target_s = _clamp(
        float(precise_target["target_s_m"]),
        float(line.current_s) + float(CFG["active_min_external_target_progress_m"]),
        float(line.cumulative_s[-1]) - 0.5,
    )
    target_xy, target_heading = get_path_point_at_s(line.line, target_s, line.cumulative_s)
    debug = copy.deepcopy(metrics)
    debug.update(copy.deepcopy(precise_target))
    output = {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(precise_target["target_v_mps"]),
        "a": 0.0,
        "theta": float(target_heading),
        "t": float(precise_target["plan_t_s"]),
        "target_line_index": int(line.line_index),
        "hold_t": 0.0,
        "reason": str(reason),
        "min_external_plan_t_s": float(context.dt),
        "min_external_target_progress_m": float(CFG["active_min_external_target_progress_m"]),
    }
    return _annotate_output(output, scene_status, reason, debug)


def _build_complete_output(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    reason: str,
    scene_status: str,
) -> dict[str, Any]:
    """构造 COMPLETE 阶段输出：换道交互完成后按固定减速度停车。"""
    limits = _read_vehicle_limits(context)
    line = target_line if target_line is not None else context.ego_vehicle.current_ref_line
    current_v = max(0.0, float(context.ego_vehicle.v))
    requested_decel = abs(float(CFG["complete_decel_mps2"]))
    stop_v = max(0.0, float(CFG["complete_stop_v_mps"]))

    if current_v <= stop_v or requested_decel <= 1e-6:
        target_v = 0.0
        target_a = 0.0
        plan_t = float(CFG["active_plan_t_s"])
        stop_distance = 0.0
    else:
        target_v = 0.0
        target_a = _clamp(-requested_decel, limits.min_acc, limits.max_acc)
        effective_decel = abs(min(target_a, 0.0))
        if effective_decel <= 1e-6:
            plan_t = float(CFG["active_plan_t_s"])
            stop_distance = current_v * plan_t
        else:
            plan_t = max(float(context.dt), current_v / effective_decel)
            stop_distance = (current_v ** 2) / (2.0 * effective_decel)

    target_s = _clamp(
        float(line.current_s) + max(0.3, float(stop_distance)),
        float(line.current_s) + 0.3,
        float(line.cumulative_s[-1]) - 0.5,
    )
    target_xy, target_heading = get_path_point_at_s(line.line, target_s, line.cumulative_s)
    debug = copy.deepcopy(metrics)
    debug.update(
        {
            "complete_stop_policy": "fixed_decel_to_stop",
            "complete_requested_decel_mps2": float(requested_decel),
            "complete_target_acc_mps2": float(target_a),
            "complete_target_v_mps": float(target_v),
            "complete_stop_distance_m": float(stop_distance),
            "complete_stop_plan_t_s": float(plan_t),
            "complete_stop_v_mps": float(stop_v),
        }
    )
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
        "control_mode": "emergency_stop",
        "emergency_decel_mps2": float(requested_decel),
    }
    return _annotate_output(
        output,
        scene_status=scene_status,
        scene_detail=reason,
        scene_debug=debug,
    )


def _build_abort_output(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """构造硬安全中止输出：保持当前/锁定车道并减速。"""
    limits = _read_vehicle_limits(context)
    line = target_line if target_line is not None else context.ego_vehicle.current_ref_line
    target_v = max(0.0, min(float(context.ego_vehicle.v), float(context.main_vehicle.v) if context.main_vehicle else 0.0) - 1.0)
    plan_t = float(CFG["abort_plan_t_s"])
    target_a = _clamp((target_v - float(context.ego_vehicle.v)) / max(plan_t, 1e-3), limits.min_acc, limits.max_acc)
    return _build_line_target_output(
        context=context,
        target_line=line,
        target_v=target_v,
        target_a=target_a,
        plan_t=plan_t,
        reason=reason,
        scene_status="safe_abort",
        scene_detail=reason,
        scene_debug=metrics,
    )


def _build_rel_s_control_output(
    *,
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    target_rel_s: float,
    gain: float,
    max_adv: float,
    max_lag: float,
    plan_t: float,
    reason: str,
    scene_status: str,
    extra_speed_delta: float = 0.0,
    extra_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """根据目标 ``rel_s`` 构造 GVT 速度闭环目标。"""
    limits = _read_vehicle_limits(context)
    main_v = float(metrics.get("main_v_mps", context.main_vehicle.v if context.main_vehicle else 0.0))
    rel_s = float(metrics.get("rel_s_m", float(CFG["blind_rel_s_target_m"])))
    rel_error = float(target_rel_s) - rel_s
    speed_delta = _clamp(gain * rel_error + float(extra_speed_delta), -float(max_lag), float(max_adv))
    target_v = _clamp(main_v + speed_delta, 0.0, limits.max_v)
    target_a = _clamp((target_v - float(context.ego_vehicle.v)) / max(float(plan_t), 1e-3), limits.min_acc, limits.max_acc)
    debug = copy.deepcopy(metrics)
    debug.update(
        {
            "control_target_rel_s_m": float(target_rel_s),
            "control_rel_s_error_m": float(rel_error),
            "control_speed_delta_mps": float(speed_delta),
            "control_target_v_mps": float(target_v),
            "control_target_a_mps2": float(target_a),
        }
    )
    if extra_debug is not None:
        debug.update(copy.deepcopy(extra_debug))
    return _build_line_target_output(
        context=context,
        target_line=target_line if target_line is not None else context.ego_vehicle.current_ref_line,
        target_v=target_v,
        target_a=target_a,
        plan_t=plan_t,
        reason=reason,
        scene_status=scene_status,
        scene_debug=debug,
    )


def _build_line_target_output(
    *,
    context: SceneContext,
    target_line: ReferenceLineView,
    target_v: float,
    target_a: float,
    plan_t: float,
    reason: str,
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在指定参考线上构造可被 ``ExternalTarget`` 消费的终端目标。"""
    min_progress = float(CFG["min_target_progress_m"])
    travel_s = max(min_progress, float(target_v) * float(plan_t))
    target_s = _clamp(
        float(target_line.current_s) + travel_s,
        float(target_line.current_s) + min_progress,
        float(target_line.cumulative_s[-1]) - 0.5,
    )
    target_xy, target_heading = get_path_point_at_s(target_line.line, target_s, target_line.cumulative_s)
    output = {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(target_v),
        "a": float(target_a),
        "theta": float(target_heading),
        "t": float(plan_t),
        "target_line_index": int(target_line.line_index),
        "hold_t": 0.0,
        "reason": str(reason),
    }
    return _annotate_output(output, scene_status, scene_detail or reason, scene_debug)


# ==================== 条件判断与指标计算 ====================


def _is_approach_candidate(metrics: dict[str, Any]) -> bool:
    """判断初始场景是否满足盲区构造条件。"""
    return bool(
        metrics.get("main_valid", False)
        and metrics.get("target_line_valid", False)
        and metrics.get("same_direction", False)
        and metrics.get("gvt_on_configured_side_lane", False)
        and metrics.get("rel_s_search_ok", False)
        and metrics.get("road_remaining_ok", False)
    )


def _is_prepare_candidate(metrics: dict[str, Any]) -> bool:
    """判断执行过程中 GVT 是否仍位于配置侧相邻车道后方。"""
    return bool(
        metrics.get("main_valid", False)
        and metrics.get("target_line_valid", False)
        and metrics.get("same_direction", False)
        and metrics.get("gvt_on_configured_side_lane", False)
        and metrics.get("rel_s_m", 0.0) < 0.0
        and metrics.get("road_remaining_ok", False)
    )


def _is_active_candidate(metrics: dict[str, Any]) -> bool:
    """判断换道交互阶段是否仍满足完成时 THW 控制所需的基础几何。

    VUT 完成换道后，GVT 与 VUT 会进入同一目标车道，因此这里不再要求
    ``gvt_on_configured_side_lane`` 为真，只要求两车同向、目标线有效且
    GVT 仍在 VUT 后方。
    """
    return bool(
        metrics.get("main_valid", False)
        and metrics.get("target_line_valid", False)
        and metrics.get("same_direction", False)
        and metrics.get("rel_s_m", 0.0) < 0.0
        and metrics.get("road_remaining_ok", False)
    )


def _update_blind_zone_hold(memory: dict[str, Any], metrics: dict[str, Any], now: float) -> None:
    """更新盲区保持命中/丢失计数，并锁存基础盲区保持成功状态。"""
    if bool(metrics.get("in_blind_zone", False)):
        memory["blind_zone_hits"] += 1
        memory["blind_zone_miss_hits"] = 0
    else:
        memory["blind_zone_miss_hits"] += 1

    hold_start = float(memory.get("hold_start_time_s", now))
    if hold_start < 0.0:
        hold_start = float(now)
        memory["hold_start_time_s"] = hold_start
    hold_elapsed = max(0.0, float(now) - hold_start)
    if hold_elapsed >= float(CFG["hold_success_duration_s"]) and int(memory.get("blind_zone_miss_hits", 0)) == 0:
        memory["hold_success_latched"] = True


def _update_lane_change_detection(memory: dict[str, Any], metrics: dict[str, Any]) -> None:
    """基于横向进度、横向速度和航向偏角防抖判断 VUT 是否开始向目标侧换道。"""
    baseline_abs = float(memory.get("baseline_main_abs_d_on_target", -1.0))
    if baseline_abs < 0.0:
        baseline_abs = float(metrics.get("main_abs_d_on_target_line_m", -1.0))
        memory["baseline_main_abs_d_on_target"] = baseline_abs
    prev_abs = float(memory.get("prev_main_abs_d_on_target", baseline_abs))
    current_abs = float(metrics.get("main_abs_d_on_target_line_m", baseline_abs))
    dt = max(float(metrics.get("dt_s", 0.1)), 1e-3)
    lateral_progress_m = max(0.0, baseline_abs - current_abs)
    lateral_v_toward = (prev_abs - current_abs) / dt
    heading_ok = float(metrics.get("main_heading_diff_to_target_line_deg", 0.0)) >= float(CFG["lane_change_heading_min_deg"])
    offset_ok = lateral_progress_m >= float(CFG["lane_change_start_d_m"])
    velocity_ok = lateral_v_toward >= float(CFG["lane_change_lateral_v_min_mps"])
    strong_offset_ok = lateral_progress_m >= 1.5 * float(CFG["lane_change_start_d_m"])
    started_now = bool(offset_ok and velocity_ok and (heading_ok or strong_offset_ok))
    if started_now:
        memory["lane_change_start_hits"] += 1
    else:
        memory["lane_change_start_hits"] = max(0, int(memory.get("lane_change_start_hits", 0)) - 1)
    memory["prev_main_abs_d_on_target"] = current_abs

    metrics["baseline_main_abs_d_on_target_m"] = float(baseline_abs)
    metrics["main_lateral_progress_to_target_m"] = float(lateral_progress_m)
    metrics["main_lateral_v_toward_target_mps"] = float(lateral_v_toward)
    metrics["lane_change_start_offset_ok"] = bool(offset_ok)
    metrics["lane_change_start_strong_offset_ok"] = bool(strong_offset_ok)
    metrics["lane_change_start_velocity_ok"] = bool(velocity_ok)
    metrics["lane_change_start_heading_ok"] = bool(heading_ok)
    metrics["lane_change_start_hits"] = int(memory["lane_change_start_hits"])
    metrics["lane_change_started"] = bool(int(memory["lane_change_start_hits"]) >= int(CFG["lane_change_debounce_frames"]))


def _is_completion_success(metrics: dict[str, Any]) -> bool:
    """判断 VUT 到达目标车道后，完成时 THW 是否落入配置容差。"""
    if not bool(metrics.get("vut_lane_change_complete", False)):
        return False
    thw = float(metrics.get("completion_thw_s", float("inf")))
    target_thw = float(CFG["completion_target_thw_s"])
    return bool(np.isfinite(thw) and abs(thw - target_thw) <= float(CFG["completion_thw_tol_s"]))


def _print_lane_change_completion_thw(metrics: dict[str, Any], completion_success: bool) -> None:
    """在 VUT 换道完成当帧打印实际 THW 结果。"""
    thw = float(metrics.get("completion_thw_s", float("nan")))
    target_thw = float(CFG["completion_target_thw_s"])
    rel_s = float(metrics.get("rel_s_m", float("nan")))
    headway_gap = float(metrics.get("completion_headway_gap_m", float("nan")))
    net_gap = float(metrics.get("completion_net_gap_m", float("nan")))
    ego_v = float(metrics.get("ego_v_mps", float("nan")))
    print(
        "[blind_spot] lane_change_complete "
        f"thw={thw:.3f}s target={target_thw:.3f}s "
        f"error={thw - target_thw:.3f}s "
        f"rel_s={rel_s:.3f}m headway_gap={headway_gap:.3f}m net_gap={net_gap:.3f}m "
        f"gvt_v={ego_v:.3f}m/s success={bool(completion_success)}",
        flush=True,
    )


def _should_hard_abort(metrics: dict[str, Any]) -> bool:
    """只在车身几何发生硬碰撞风险时中止危险场景。"""
    return bool(metrics.get("hard_collision_risk", False))


def _strict_scene_geometry_enabled() -> bool:
    """读取是否启用严格场景几何检查。"""
    return bool(CFG.get("strict_scene_geometry", True))


def _invalid_scene_reasons(metrics: dict[str, Any], *, require_search_window: bool) -> list[str]:
    """汇总盲区场景构造不成立的具体原因，供日志和离线 JSON 排查使用。"""
    reasons: list[str] = []
    if not bool(metrics.get("main_valid", False)):
        reasons.append("no_main_vehicle")
        return reasons
    if not bool(metrics.get("target_line_valid", False)):
        reasons.append("gvt_not_on_configured_side_adjacent_lane")
        return reasons
    if not bool(metrics.get("same_direction", False)):
        reasons.append("vut_gvt_heading_not_aligned")
    if not bool(metrics.get("gvt_on_configured_side_lane", False)):
        reasons.append("gvt_not_on_configured_side_lane")
    rel_s = float(metrics.get("rel_s_m", 0.0))
    if rel_s >= 0.0:
        reasons.append("gvt_not_behind_vut")
    if require_search_window and not bool(metrics.get("rel_s_search_ok", False)):
        reasons.append("gvt_not_in_approach_rel_s_window")
    if not bool(metrics.get("road_remaining_ok", False)):
        reasons.append("target_lane_remaining_distance_insufficient")
    if bool(metrics.get("hard_collision_risk", False)):
        reasons.append("hard_collision_risk")
    if not reasons:
        reasons.append("unknown_invalid_scene_geometry")
    return reasons


def _solve_completion_thw_terminal(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """反解 GVT 在 VUT 换道完成时满足目标 THW 所需的终端状态。

    THW 的速度基准使用后方车 GVT 在换道完成时刻的速度。
    在常加速度假设下：

    ``s_gvt_end = s_gvt + 0.5 * (v0 + v_end) * T``
    ``s_vut_front_end - s_gvt_front_end = target_thw * v_end``

    由此直接解出 ``v_end``，再按平台速度/加速度约束裁剪。
    """
    if target_line is None:
        return {"valid": False, "reason": "no_target_line"}

    raw_t = float(metrics.get("predicted_t_complete_s", float("nan")))
    if not np.isfinite(raw_t):
        return {"valid": False, "reason": "invalid_predicted_completion_time"}
    plan_t = _clamp(
        raw_t,
        float(CFG["lane_change_prediction_t_min_s"]),
        float(CFG["lane_change_prediction_t_max_s"]),
    )
    if plan_t <= 1e-3:
        return {"valid": False, "reason": "non_positive_completion_time"}

    limits = _read_vehicle_limits(context)
    gvt_s = float(metrics.get("ego_s_on_target_line_m", target_line.current_s))
    gvt_v = max(0.0, float(metrics.get("ego_v_mps", context.ego_vehicle.v)))
    vut_s_complete = float(metrics.get("predicted_main_s_complete_m", float("nan")))
    if not np.isfinite(vut_s_complete):
        vut_s_complete = float(metrics.get("main_s_on_target_line_m", 0.0)) + float(metrics.get("main_v_mps", 0.0)) * plan_t

    target_thw = max(0.0, float(CFG["completion_target_thw_s"]))
    ego_length = float(metrics.get("ego_length_m", limits.ego_length))
    main_length = float(metrics.get("main_length_m", limits.main_length))
    half_len_sum = 0.5 * (ego_length + main_length)
    headway_center_offset = 0.5 * (main_length - ego_length)
    denominator = target_thw + 0.5 * plan_t
    if denominator <= 1e-6:
        return {"valid": False, "reason": "invalid_completion_thw_denominator"}

    raw_target_v = (
        float(vut_s_complete)
        + headway_center_offset
        - float(gvt_s)
        - 0.5 * float(gvt_v) * plan_t
    ) / denominator

    min_net_gap = float(CFG["completion_min_net_gap_m"])
    if target_thw * max(raw_target_v, 0.0) - main_length < min_net_gap:
        raw_target_v = 2.0 * (
            float(vut_s_complete)
            - half_len_sum
            - min_net_gap
            - float(gvt_s)
        ) / plan_t - float(gvt_v)

    reachable_v_min = max(0.0, float(gvt_v) + limits.min_acc * plan_t)
    reachable_v_max = min(limits.max_v, float(gvt_v) + limits.max_acc * plan_t)
    if bool(CFG["active_terminal_speed_ignore_acc_limit"]):
        target_v = _clamp(raw_target_v, 0.0, limits.max_v)
    else:
        target_v = _clamp(raw_target_v, reachable_v_min, reachable_v_max)
    target_acc = (target_v - float(gvt_v)) / plan_t
    target_s = float(gvt_s) + 0.5 * (float(gvt_v) + target_v) * plan_t
    target_headway_gap = (
        float(vut_s_complete) + 0.5 * main_length
        - (target_s + 0.5 * ego_length)
    )
    target_net_gap = target_headway_gap - main_length
    achieved_thw = (
        target_headway_gap / max(target_v, float(CFG["thw_v_floor_mps"]))
        if target_headway_gap > 0.0
        else 0.0
    )
    thw_error = achieved_thw - target_thw
    feasible = bool(
        np.isfinite(raw_target_v)
        and abs(target_v - raw_target_v) <= 1e-3
        and abs(thw_error) <= float(CFG["completion_thw_tol_s"])
        and target_net_gap >= min_net_gap - 1e-3
    )
    acc_within_limit = bool(limits.min_acc <= target_acc <= limits.max_acc)

    return {
        "valid": True,
        "completion_control_mode": "precise_terminal_thw",
        "completion_thw_speed_base": "gvt_completion_speed",
        "plan_t_s": float(plan_t),
        "raw_predicted_t_complete_s": float(raw_t),
        "target_s_m": float(target_s),
        "target_v_mps": float(target_v),
        "target_acc_mps2": float(target_acc),
        "required_avg_acc_mps2": float(target_acc),
        "external_target_acc_consumed": False,
        "terminal_speed_ignore_acc_limit": bool(CFG["active_terminal_speed_ignore_acc_limit"]),
        "required_avg_acc_within_platform_limit": bool(acc_within_limit),
        "raw_target_v_mps": float(raw_target_v),
        "reachable_v_min_mps": float(reachable_v_min),
        "reachable_v_max_mps": float(reachable_v_max),
        "predicted_vut_s_complete_m": float(vut_s_complete),
        "predicted_gvt_s_complete_m": float(target_s),
        "predicted_completion_headway_gap_m": float(target_headway_gap),
        "predicted_completion_net_gap_m": float(target_net_gap),
        "predicted_completion_thw_s": float(achieved_thw),
        "predicted_completion_thw_error_s": float(thw_error),
        "completion_target_thw_s": float(target_thw),
        "completion_target_feasible": bool(feasible),
        "completion_target_infeasible_reason": "" if feasible else "speed_or_acc_constraint_limited",
    }


def _compute_active_target_rel_s(metrics: dict[str, Any]) -> dict[str, Any]:
    """VUT 换道过程中，将盲区目标 ``rel_s`` 平滑过渡到完成时 THW 对应 ``rel_s``。"""
    ego_length = float(metrics.get("ego_length_m", 4.5))
    main_length = float(metrics.get("main_length_m", 4.5))
    limits_half_sum = 0.5 * (ego_length + main_length)
    follower_v = max(float(metrics.get("ego_v_mps", 0.0)), float(CFG["thw_v_floor_mps"]))
    target_headway_gap = float(CFG["completion_target_thw_s"]) * follower_v
    target_net_gap = max(
        target_headway_gap - main_length,
        float(CFG["completion_min_net_gap_m"]),
    )
    completion_rel_s_target = -(target_net_gap + limits_half_sum)
    blind_target = float(CFG["blind_rel_s_target_m"])
    progress = float(metrics.get("lane_change_progress_ratio", 0.0))
    blend_start = _clamp(float(CFG["completion_blend_start_progress"]), 0.0, 0.95)
    blend = _clamp((progress - blend_start) / max(1.0 - blend_start, 1e-3), 0.0, 1.0)
    target_rel_s = (1.0 - blend) * blind_target + blend * completion_rel_s_target

    t_complete = float(metrics.get("predicted_t_complete_s", float("inf")))
    rel_s_complete = float(metrics.get("predicted_rel_s_complete_m", float("nan")))
    if np.isfinite(t_complete) and np.isfinite(rel_s_complete) and t_complete > 1e-3:
        prediction_error = completion_rel_s_target - rel_s_complete
        prediction_speed_delta = float(CFG["completion_prediction_gain"]) * prediction_error / t_complete
    else:
        prediction_error = 0.0
        prediction_speed_delta = 0.0

    return {
        "active_blend_to_completion": float(blend),
        "target_headway_gap_from_thw_m": float(target_headway_gap),
        "target_net_gap_from_thw_m": float(target_net_gap),
        "completion_rel_s_target_m": float(completion_rel_s_target),
        "target_rel_s_m": float(target_rel_s),
        "prediction_rel_s_error_m": float(prediction_error),
        "prediction_speed_delta_mps": float(prediction_speed_delta),
    }


def _compute_blind_spot_metrics(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    memory: dict[str, Any],
) -> dict[str, Any]:
    """计算盲区车道关系、相对位置、换道进度、完成时 THW 和硬碰撞指标。"""
    ego = context.ego_vehicle
    main = context.main_vehicle
    limits = _read_vehicle_limits(context)
    side = _normalized_scene_side()
    if main is None:
        return {
            "main_valid": False,
            "scene_side": side,
            "target_line_valid": False,
            "hard_collision_risk": False,
            "ego_length_m": float(limits.ego_length),
            "main_length_m": float(limits.main_length),
        }

    if target_line is None:
        target_line = _select_current_side_lane(context)

    if target_line is None:
        return {
            "main_valid": True,
            "scene_side": side,
            "target_line_valid": False,
            "hard_collision_risk": False,
            "ego_length_m": float(limits.ego_length),
            "main_length_m": float(limits.main_length),
            "ego_v_mps": float(ego.v),
            "main_v_mps": float(main.v),
        }

    _, ego_s, ego_d = project_point_to_path(np.array([ego.x, ego.y], dtype=float), target_line.line)
    _, main_s, main_d = project_point_to_path(np.array([main.x, main.y], dtype=float), target_line.line)
    target_xy_at_main_s, target_heading_at_main_s = get_path_point_at_s(
        target_line.line,
        _clamp(float(main_s), 0.0, float(target_line.cumulative_s[-1])),
        target_line.cumulative_s,
    )
    del target_xy_at_main_s

    rel_s = float(ego_s) - float(main_s)
    heading_diff = _heading_diff_deg(float(main.heading), float(ego.heading))
    heading_diff_to_target = _heading_diff_deg(float(main.heading), float(target_heading_at_main_s))
    side_lateral = _signed_side_lateral_offset(main, ego, side)
    ego_on_target_line = abs(float(ego_d)) <= float(CFG["max_ego_d_on_target_line_m"])
    side_lateral_ok = float(CFG["side_lateral_min_m"]) <= side_lateral <= float(CFG["side_lateral_max_m"])
    same_direction = heading_diff <= float(CFG["max_heading_diff_deg"])
    road_remaining = float(target_line.cumulative_s[-1]) - float(ego_s)
    road_remaining_ok = road_remaining >= float(CFG["min_road_remaining_m"])
    rel_s_search_ok = float(CFG["search_rel_s_min_m"]) <= rel_s <= float(CFG["search_rel_s_max_m"])
    in_blind_zone = bool(
        same_direction
        and ego_on_target_line
        and side_lateral_ok
        and float(CFG["blind_rel_s_min_m"]) <= rel_s <= float(CFG["blind_rel_s_max_m"])
    )

    center_gap = -rel_s
    net_gap = center_gap - 0.5 * (limits.ego_length + limits.main_length)
    headway_gap = center_gap + 0.5 * (limits.main_length - limits.ego_length)
    completion_thw = headway_gap / max(float(ego.v), float(CFG["thw_v_floor_mps"])) if headway_gap > 0.0 else 0.0
    closing_speed = float(ego.v) - float(main.v)
    ttc_debug = net_gap / closing_speed if net_gap > 0.0 and closing_speed > 1e-6 else float("inf")
    baseline_abs_d = float(memory.get("baseline_main_abs_d_on_target", -1.0))
    current_abs_d = abs(float(main_d))
    if baseline_abs_d <= 0.0:
        baseline_abs_d = current_abs_d
    complete_d = float(CFG["lane_change_complete_d_m"])
    lane_change_progress = _clamp(
        (baseline_abs_d - current_abs_d) / max(baseline_abs_d - complete_d, 1e-3),
        0.0,
        1.0,
    )
    predicted = _predict_lane_change_completion(
        ego_s=float(ego_s),
        main_s=float(main_s),
        ego_v=float(ego.v),
        main_v=float(main.v),
        current_abs_d=current_abs_d,
        prev_abs_d=float(memory.get("prev_main_abs_d_on_target", current_abs_d)),
        dt=float(context.dt),
    )
    hard_collision = _evaluate_hard_collision(context, limits)
    gvt_on_side_lane = bool(ego_on_target_line and side_lateral_ok)

    return {
        "main_valid": True,
        "scene_side": side,
        "target_line_valid": True,
        "target_line_index": int(target_line.line_index),
        "target_lane_id": str(target_line.lane_id),
        "target_lane_sequence": _normalize_lane_sequence(target_line.lane_sequence),
        "ego_s_on_target_line_m": float(ego_s),
        "ego_d_on_target_line_m": float(ego_d),
        "main_s_on_target_line_m": float(main_s),
        "main_d_on_target_line_m": float(main_d),
        "main_abs_d_on_target_line_m": float(current_abs_d),
        "rel_s_m": float(rel_s),
        "side_lateral_offset_m": float(side_lateral),
        "same_direction": bool(same_direction),
        "heading_diff_deg": float(heading_diff),
        "main_heading_diff_to_target_line_deg": float(heading_diff_to_target),
        "ego_on_target_line": bool(ego_on_target_line),
        "side_lateral_ok": bool(side_lateral_ok),
        "gvt_on_configured_side_lane": bool(gvt_on_side_lane),
        "rel_s_search_ok": bool(rel_s_search_ok),
        "in_blind_zone": bool(in_blind_zone),
        "road_remaining_m": float(road_remaining),
        "road_remaining_ok": bool(road_remaining_ok),
        "center_gap_m": float(center_gap),
        "completion_headway_gap_m": float(headway_gap),
        "completion_net_gap_m": float(net_gap),
        "completion_thw_s": float(completion_thw),
        "completion_target_thw_s": float(CFG["completion_target_thw_s"]),
        "completion_thw_error_s": float(completion_thw - float(CFG["completion_target_thw_s"])),
        "ttc_debug_s": float(ttc_debug),
        "closing_speed_mps": float(closing_speed),
        "lane_change_progress_ratio": float(lane_change_progress),
        "vut_lane_change_complete": bool(current_abs_d <= complete_d or lane_change_progress >= 0.85),
        "hard_collision_risk": bool(hard_collision["hard_collision_risk"]),
        "longitudinal_overlap_m": float(hard_collision["longitudinal_overlap_m"]),
        "lateral_overlap_m": float(hard_collision["lateral_overlap_m"]),
        "dt_s": float(context.dt),
        "ego_v_mps": float(ego.v),
        "main_v_mps": float(main.v),
        "ego_length_m": float(limits.ego_length),
        "main_length_m": float(limits.main_length),
        "ego_width_m": float(limits.ego_width),
        "main_width_m": float(limits.main_width),
        **predicted,
    }


def _predict_lane_change_completion(
    *,
    ego_s: float,
    main_s: float,
    ego_v: float,
    main_v: float,
    current_abs_d: float,
    prev_abs_d: float,
    dt: float,
) -> dict[str, Any]:
    """按纵向速度恒定假设，预测 VUT 到达目标侧车道中心附近时的两车纵向关系。"""
    lateral_v = (float(prev_abs_d) - float(current_abs_d)) / max(float(dt), 1e-3)
    lateral_v_eff = max(float(lateral_v), float(CFG["lane_change_min_lateral_v_mps"]))
    remaining_d = max(0.0, float(current_abs_d) - float(CFG["lane_change_complete_d_m"]))
    raw_t_complete = remaining_d / max(lateral_v_eff, 1e-3)
    adjusted_t_complete = max(
        float(dt),
        raw_t_complete * float(CFG["lane_change_prediction_time_scale"])
        - float(CFG["lane_change_prediction_time_buffer_s"]),
    )
    t_complete = _clamp(
        adjusted_t_complete,
        float(CFG["lane_change_prediction_t_min_s"]),
        float(CFG["lane_change_prediction_t_max_s"]),
    )
    ego_s_complete = float(ego_s) + float(ego_v) * t_complete
    main_s_complete = float(main_s) + float(main_v) * t_complete
    return {
        "predicted_lateral_v_toward_target_mps": float(lateral_v),
        "predicted_t_complete_raw_s": float(raw_t_complete),
        "predicted_t_complete_adjusted_s": float(adjusted_t_complete),
        "predicted_t_complete_s": float(t_complete),
        "predicted_ego_s_complete_m": float(ego_s_complete),
        "predicted_main_s_complete_m": float(main_s_complete),
        "predicted_rel_s_complete_m": float(ego_s_complete - main_s_complete),
    }


def _evaluate_hard_collision(context: SceneContext, limits: VehicleLimits) -> dict[str, Any]:
    """在 VUT 车体坐标系下检查两车矩形包络是否发生硬碰撞重叠。"""
    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        return {
            "hard_collision_risk": False,
            "longitudinal_overlap_m": 0.0,
            "lateral_overlap_m": 0.0,
        }
    rel_xy = np.array([float(ego.x) - float(main.x), float(ego.y) - float(main.y)], dtype=float)
    forward = np.array([np.cos(float(main.heading)), np.sin(float(main.heading))], dtype=float)
    left = np.array([-forward[1], forward[0]], dtype=float)
    rel_longitudinal = abs(float(np.dot(rel_xy, forward)))
    rel_lateral = abs(float(np.dot(rel_xy, left)))
    margin = float(CFG["hard_collision_margin_m"])
    longitudinal_overlap = 0.5 * (limits.ego_length + limits.main_length) + margin - rel_longitudinal
    lateral_overlap = 0.5 * (limits.ego_width + limits.main_width) + margin - rel_lateral
    hard_collision = bool(longitudinal_overlap > 0.0 and lateral_overlap > 0.0)
    return {
        "hard_collision_risk": bool(hard_collision),
        "longitudinal_overlap_m": float(max(0.0, longitudinal_overlap)),
        "lateral_overlap_m": float(max(0.0, lateral_overlap)),
    }


# ==================== 车道辅助函数 ====================


def _select_current_side_lane(context: SceneContext) -> ReferenceLineView | None:
    """当 GVT 当前参考线位于配置侧相邻车道时，将其作为盲区目标车道。"""
    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        return None
    side = _normalized_scene_side()
    side_lateral = _signed_side_lateral_offset(main, ego, side)
    if side_lateral < float(CFG["side_lateral_min_m"]):
        return None
    if abs(float(ego.current_ref_line.current_d)) > float(CFG["max_ego_d_on_target_line_m"]):
        return None
    return ego.current_ref_line


def _resolve_target_line(context: SceneContext, memory: dict[str, Any]) -> ReferenceLineView | None:
    """按已锁定的车道序列、车道 ID 或参考线下标恢复目标侧车道。"""
    line = _line_by_lane_sequence(
        context.ego_vehicle.candidate_ref_lines,
        memory.get("locked_target_lane_sequence", []),
    )
    if line is not None:
        return line
    line = _line_by_lane_id(context.ego_vehicle.candidate_ref_lines, memory.get("locked_target_lane_id", ""))
    if line is not None:
        return line
    line = _line_by_index(context.ego_vehicle.candidate_ref_lines, memory.get("locked_line_index"))
    if line is not None:
        return line
    return _select_current_side_lane(context)


def _line_by_index(candidate_lines: list[ReferenceLineView], line_index: int | None) -> ReferenceLineView | None:
    """按参考线下标查找候选参考线。"""
    if line_index is None:
        return None
    for line in candidate_lines:
        if int(line.line_index) == int(line_index):
            return line
    return None


def _line_by_lane_id(candidate_lines: list[ReferenceLineView], lane_id: str) -> ReferenceLineView | None:
    """按车道 ID 查找候选参考线。"""
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
    """按车道序列重叠关系查找语义最接近的候选参考线。"""
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


def _normalize_lane_sequence(lane_sequence: list[str] | tuple[str, ...] | None) -> list[str]:
    """把车道序列统一转成字符串列表。"""
    if lane_sequence is None:
        return []
    return [str(lane_id) for lane_id in lane_sequence]


# ==================== 通用工具 ====================


def _annotate_output(
    output: dict[str, Any],
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """给场景输出附加状态、详情和调试信息。"""
    output["scene_status"] = str(scene_status)
    if scene_detail is not None:
        output["scene_detail"] = str(scene_detail)
    if scene_debug is not None:
        output["scene_debug"] = copy.deepcopy(scene_debug)
    return output


def _normalized_scene_side() -> str:
    """读取配置侧；非法配置统一退化为 left。"""
    side = str(CFG.get("scene_side", "left")).strip().lower()
    if side not in {"left", "right"}:
        return "left"
    return side


def _signed_side_lateral_offset(main: Any, ego: Any, side: str) -> float:
    """计算 GVT 相对 VUT 的有符号横向距离；正值表示 GVT 在配置侧。"""
    rel_xy = np.array([float(ego.x) - float(main.x), float(ego.y) - float(main.y)], dtype=float)
    forward = np.array([np.cos(float(main.heading)), np.sin(float(main.heading))], dtype=float)
    left = np.array([-forward[1], forward[0]], dtype=float)
    left_offset = float(np.dot(rel_xy, left))
    return left_offset if side == "left" else -left_offset


def _clamp(value: float, lower: float, upper: float) -> float:
    """把数值裁剪到指定闭区间。"""
    return float(min(max(value, lower), upper))


def _heading_diff_deg(h1: float, h2: float) -> float:
    """计算两个航向角之间的最小绝对夹角，单位为度。"""
    return float(np.degrees(abs(normalize_angle(float(h1) - float(h2)))))


def _read_vehicle_limits(context: SceneContext) -> VehicleLimits:
    """读取车辆尺寸和平台速度/加速度约束。"""
    runtime_frame = context.runtime_frame
    config = {} if runtime_frame is None else runtime_frame.config
    size = config.get("size", {})
    constrain = config.get("constrain", {})
    ego_length = float(size.get("L", 4.5))
    ego_width = float(size.get("W", 1.8))
    return VehicleLimits(
        ego_length=float(ego_length),
        main_length=float(ego_length),
        ego_width=float(ego_width),
        main_width=float(ego_width),
        max_v=float(constrain.get("max_v", 6.94)),
        min_acc=float(constrain.get("min_acc", -2.0)),
        max_acc=float(constrain.get("max_acc", 2.0)),
    )
