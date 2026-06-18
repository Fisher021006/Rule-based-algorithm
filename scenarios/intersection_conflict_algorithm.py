"""Intersection conflict scene algorithm.

V1 scene semantics:
- Controlled vehicle is GVT, mapped to ``context.ego_vehicle``.
- Tested vehicle is VUT, mapped to ``context.main_vehicle``.
- Only two configured subtypes are supported:
  ``vut_straight_cross`` and ``vut_left_oncoming``.
- The subtype is a test-case label. The algorithm validates it by checking
  whether the selected GVT/VUT reference lines have a forward geometric
  conflict point.
- The main control metric is ``delta_t = T_gvt - T_vut`` at the conflict point.
  A small conflict-zone PET-like metric is reported for debug/success logging.
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
    """Vehicle size and platform constraints used by this scene."""

    ego_length: float
    main_length: float
    ego_width: float
    main_width: float
    max_v: float
    min_acc: float
    max_acc: float


SUPPORTED_CONFLICT_TYPES = {"vut_straight_cross", "vut_left_oncoming"}


CFG: dict[str, Any] = {
    # Semantic label configured by the test case. It is not auto-detected.
    "conflict_type": "vut_straight_cross",
    # GVT should arrive this many seconds after VUT. Negative means GVT first.
    "target_delta_t_s": 0.0,
    "success_time_tol_s": 0.5,
    "conflict_zone_radius_m": 2.5,
    "conflict_point_match_tolerance_m": 1.2,
    "min_conflict_angle_deg": 25.0,
    "min_forward_conflict_distance_m": 2.0,
    "max_forward_conflict_distance_m": 120.0,
    "post_conflict_target_buffer_m": 8.0,
    "clear_after_conflict_m": 4.0,
    "cruise_target_v": 5.0,
    "speed_floor_mps": 0.5,
    "target_speed_min_mps": 0.2,
    "sync_plan_t_min_s": 0.2,
    "sync_plan_t_max_s": 8.0,
    "sync_timeout_s": 15.0,
    "invalid_stop_plan_t_s": 1.0,
    "complete_decel_mps2": 2.5,
    "stop_complete_v_mps": 0.05,
    "stop_hold_s": 10.0,
    "min_target_progress_m": 0.05,
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
    "gvt_cross_time_s": None,
    "vut_cross_time_s": None,
    "gvt_zone_enter_time_s": None,
    "gvt_zone_exit_time_s": None,
    "vut_zone_enter_time_s": None,
    "vut_zone_exit_time_s": None,
    "actual_result_printed": False,
}


SCENE_MEMORY: dict[str, Any] = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Public API ====================


def intersection_conflict_algorithm(context: SceneContext) -> dict[str, Any]:
    """Intersection conflict scene algorithm entrypoint."""
    global SCENE_MEMORY
    start_ms = time.time() * 1000.0
    if int(context.step_index) == 0:
        _reset_scene_memory()
    output, SCENE_MEMORY = _scene_fsm_step(context, SCENE_MEMORY)
    end_ms = time.time() * 1000.0
    print(f"Intersection conflict algorithm execution time: {end_ms - start_ms} ms")
    return output


# ==================== FSM ====================


def _scene_fsm_step(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one frame of the intersection-conflict state machine."""
    now = float(context.time_s)
    mode = str(memory.get("mode", "APPROACH"))

    if mode == "APPROACH":
        return _handle_approach(context, memory, now)
    if mode == "SYNC_ARRIVAL":
        return _handle_sync_arrival(context, memory, now)
    if mode == "CONFLICT_ACTIVE":
        return _handle_conflict_active(context, memory, now)
    if mode == "COMPLETE_BRAKE":
        return _handle_complete_brake(context, memory, now)
    if mode == "COMPLETE":
        return _handle_complete(context, memory, now)
    return _handle_unknown_mode(context, memory)


def _handle_approach(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and validate conflict semantics before any timing control."""
    conflict_result = _build_conflict_context(context)
    if not bool(conflict_result.get("valid", False)):
        memory["mode"] = "APPROACH"
        memory["locked_conflict"] = None
        return _build_invalid_scene_output(
            context,
            reason=str(conflict_result.get("reason", "intersection_invalid_scene")),
            scene_debug=conflict_result,
        ), memory

    memory["mode"] = "SYNC_ARRIVAL"
    memory["sync_start_time_s"] = float(now)
    memory["locked_conflict"] = _compact_conflict_context(conflict_result)

    metrics = _compute_conflict_metrics(context, memory["locked_conflict"])
    metrics = _update_actual_conflict_observation(memory, metrics, now)
    memory["last_metrics"] = copy.deepcopy(metrics)
    return _build_sync_output(
        context,
        memory,
        metrics,
        reason="intersection_sync_enter",
        scene_status="sync_arrival",
    ), memory


def _handle_sync_arrival(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Synchronize GVT arrival time at the locked conflict point."""
    metrics = _compute_conflict_metrics(context, memory.get("locked_conflict"))

    if not bool(metrics.get("valid", False)):
        memory["last_metrics"] = copy.deepcopy(metrics)
        memory["mode"] = "APPROACH"
        return _build_invalid_scene_output(
            context,
            reason=str(metrics.get("reason", "intersection_conflict_lost")),
            scene_debug=metrics,
        ), memory

    metrics = _update_actual_conflict_observation(memory, metrics, now)
    memory["last_metrics"] = copy.deepcopy(metrics)

    if _gvt_cleared_conflict(metrics):
        _print_conflict_clear_metrics(metrics)
        _enter_complete_brake(memory, now)
        return _build_complete_brake_output(context, metrics, reason="intersection_conflict_cleared"), memory

    if _is_predicted_timing_aligned(metrics):
        memory["mode"] = "CONFLICT_ACTIVE"
        memory["active_start_time_s"] = float(now)
        return _build_active_output(
            context,
            memory,
            metrics,
            reason="intersection_conflict_triggered",
            scene_status="conflict_active",
        ), memory

    elapsed = max(0.0, now - float(memory.get("sync_start_time_s", now)))
    if elapsed >= float(CFG["sync_timeout_s"]):
        memory["mode"] = "APPROACH"
        debug = copy.deepcopy(metrics)
        debug["sync_timeout_s"] = float(CFG["sync_timeout_s"])
        debug["sync_elapsed_s"] = float(elapsed)
        debug["invalid_scene_reasons"] = ["sync_timeout"]
        return _build_invalid_scene_output(
            context,
            reason="intersection_sync_timeout",
            scene_debug=debug,
        ), memory

    return _build_sync_output(
        context,
        memory,
        metrics,
        reason="intersection_sync_arrival",
        scene_status="sync_arrival",
    ), memory


def _handle_conflict_active(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep sending GVT through the conflict point until it clears."""
    metrics = _compute_conflict_metrics(context, memory.get("locked_conflict"))
    if not bool(metrics.get("valid", False)):
        memory["last_metrics"] = copy.deepcopy(metrics)
        memory["mode"] = "APPROACH"
        return _build_invalid_scene_output(
            context,
            reason=str(metrics.get("reason", "intersection_conflict_lost")),
            scene_debug=metrics,
        ), memory

    metrics = _update_actual_conflict_observation(memory, metrics, now)
    memory["last_metrics"] = copy.deepcopy(metrics)

    if _gvt_cleared_conflict(metrics):
        _print_conflict_clear_metrics(metrics)
        _enter_complete_brake(memory, now)
        return _build_complete_brake_output(context, metrics, reason="intersection_conflict_cleared"), memory

    return _build_active_output(
        context,
        memory,
        metrics,
        reason="intersection_conflict_active",
        scene_status="conflict_active",
    ), memory


def _handle_complete_brake(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """After conflict construction, switch to emergency-stop style braking."""
    metrics = _compute_conflict_metrics(context, memory.get("locked_conflict"))
    if bool(metrics.get("valid", False)):
        metrics = _update_actual_conflict_observation(memory, metrics, now)
    memory["last_metrics"] = copy.deepcopy(metrics)
    if float(context.ego_vehicle.v) <= float(CFG["stop_complete_v_mps"]):
        memory["mode"] = "COMPLETE"
        memory["complete_start_time_s"] = float(now)
        return _build_stopped_output(context, metrics, reason="intersection_complete_stop"), memory
    return _build_complete_brake_output(context, metrics, reason="intersection_complete_brake"), memory


def _handle_complete(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hold completed state to avoid repeated triggering in one run."""
    metrics = _compute_conflict_metrics(context, memory.get("locked_conflict"))
    if bool(metrics.get("valid", False)):
        metrics = _update_actual_conflict_observation(memory, metrics, now)
    complete_start = float(memory.get("complete_start_time_s", now))
    metrics["complete_elapsed_s"] = max(0.0, float(now) - complete_start)
    memory["last_metrics"] = copy.deepcopy(metrics)
    return _build_stopped_output(context, metrics, reason="intersection_complete_hold"), memory


def _handle_unknown_mode(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reset unknown states to APPROACH."""
    memory["mode"] = "APPROACH"
    memory["locked_conflict"] = None
    return _build_keep_lane_output(
        context,
        reason="intersection_unknown_mode",
        scene_status="baseline_keep_lane",
    ), memory


def _enter_complete_brake(memory: dict[str, Any], now: float) -> None:
    memory["mode"] = "COMPLETE_BRAKE"
    memory["complete_start_time_s"] = float(now)


def _reset_scene_memory() -> None:
    global SCENE_MEMORY
    SCENE_MEMORY = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Output builders ====================


def _build_keep_lane_output(
    context: SceneContext,
    reason: str,
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a conservative keep-lane target."""
    limits = _read_vehicle_limits(context)
    target_v = _clamp(float(CFG["cruise_target_v"]), 0.0, limits.max_v)
    return _build_line_target_output(
        context=context,
        target_line=context.ego_vehicle.current_ref_line,
        target_s=float(context.ego_vehicle.current_ref_line.current_s) + max(2.0, 2.0 * target_v),
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
    """Return explicit scene construction failure and ask GVT to stop."""
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
    """Build the target that synchronizes GVT arrival at the conflict point."""
    del memory
    sync_target = _solve_sync_target(context, metrics)
    debug = copy.deepcopy(metrics)
    debug.update(sync_target)
    return _build_conflict_target_output(
        context=context,
        metrics=metrics,
        sync_target=sync_target,
        reason=reason,
        scene_status=scene_status,
        scene_debug=debug,
    )


def _build_active_output(
    context: SceneContext,
    memory: dict[str, Any],
    metrics: dict[str, Any],
    reason: str,
    scene_status: str,
) -> dict[str, Any]:
    """Build the through-conflict target after the timing window is hit."""
    del memory
    sync_target = _solve_sync_target(context, metrics)
    debug = copy.deepcopy(metrics)
    debug.update(sync_target)
    return _build_conflict_target_output(
        context=context,
        metrics=metrics,
        sync_target=sync_target,
        reason=reason,
        scene_status=scene_status,
        scene_debug=debug,
    )


def _build_conflict_target_output(
    *,
    context: SceneContext,
    metrics: dict[str, Any],
    sync_target: dict[str, Any],
    reason: str,
    scene_status: str,
    scene_debug: dict[str, Any],
) -> dict[str, Any]:
    line = _resolve_gvt_line(context, metrics)
    target_s = float(sync_target.get("target_s_m", metrics.get("gvt_s_conflict_m", line.current_s)))
    target_v = float(sync_target.get("target_v_mps", context.ego_vehicle.v))
    plan_t = float(sync_target.get("plan_t_s", 1.0))
    limits = _read_vehicle_limits(context)
    target_a = _clamp(
        (target_v - float(context.ego_vehicle.v)) / max(plan_t, 1e-3),
        limits.min_acc,
        limits.max_acc,
    )
    scene_debug["target_acc_mps2"] = float(target_a)
    return _build_line_target_output(
        context=context,
        target_line=line,
        target_s=target_s,
        target_v=target_v,
        target_a=target_a,
        plan_t=plan_t,
        reason=reason,
        scene_status=scene_status,
        scene_detail=reason,
        scene_debug=scene_debug,
        min_plan_t=float(context.dt),
        min_target_progress=float(CFG["min_target_progress_m"]),
    )


def _build_complete_brake_output(
    context: SceneContext,
    metrics: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Build emergency-stop output after GVT clears the conflict point."""
    limits = _read_vehicle_limits(context)
    requested_decel = abs(float(CFG["complete_decel_mps2"]))
    target_a = _clamp(-requested_decel, limits.min_acc, limits.max_acc)
    effective_decel = abs(min(target_a, 0.0))
    current_v = max(0.0, float(context.ego_vehicle.v))
    if effective_decel > 1e-6:
        plan_t = max(float(context.dt), current_v / effective_decel)
        stop_distance = (current_v ** 2) / (2.0 * effective_decel)
    else:
        plan_t = 1.0
        stop_distance = current_v * plan_t
    debug = copy.deepcopy(metrics)
    debug.update(
        {
            "complete_policy": "emergency_stop_after_conflict",
            "complete_decel_mps2": float(requested_decel),
            "complete_target_acc_mps2": float(target_a),
            "complete_stop_distance_m": float(stop_distance),
            "complete_plan_t_s": float(plan_t),
        }
    )
    output = _build_line_target_output(
        context=context,
        target_line=context.ego_vehicle.current_ref_line,
        target_s=float(context.ego_vehicle.current_ref_line.current_s) + max(
            float(CFG["min_target_progress_m"]),
            float(stop_distance),
        ),
        target_v=0.0,
        target_a=target_a,
        plan_t=plan_t,
        reason=reason,
        scene_status="complete_brake",
        scene_detail=reason,
        scene_debug=debug,
    )
    output.update(
        {
            "control_mode": "emergency_stop",
            "emergency_decel_mps2": abs(float(target_a)),
            "stop_hold_s": float(CFG["stop_hold_s"]),
        }
    )
    return output


def _build_stopped_output(
    context: SceneContext,
    metrics: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Build zero-speed hold output after the scene is complete."""
    debug = copy.deepcopy(metrics)
    debug["complete_policy"] = "hold_stopped_after_conflict"
    output = _build_line_target_output(
        context=context,
        target_line=context.ego_vehicle.current_ref_line,
        target_s=float(context.ego_vehicle.current_ref_line.current_s) + float(CFG["min_target_progress_m"]),
        target_v=0.0,
        target_a=0.0,
        plan_t=1.0,
        reason=reason,
        scene_status="complete",
        scene_detail=reason,
        scene_debug=debug,
    )
    output.update(
        {
            "control_mode": "emergency_stop",
            "emergency_decel_mps2": abs(float(CFG["complete_decel_mps2"])),
            "stop_hold_s": float(CFG["stop_hold_s"]),
        }
    )
    return output


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
    """Build a terminal target on the selected GVT reference line."""
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
    """Build the scene semantic context and validate forward conflict point."""
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

    best_result = None
    best_score = float("inf")
    best_priority = 999
    current_gvt_line_index = int(context.ego_vehicle.current_ref_line.line_index)
    gvt_candidate_lines = _candidate_gvt_lines(context)
    main_candidate_lines = _candidate_main_lines(context)
    for gvt_line in gvt_candidate_lines:
        for main_line in main_candidate_lines:
            result = _find_forward_conflict_point(gvt_line, main_line)
            if not bool(result.get("valid", False)):
                continue
            gvt_line_priority = 0 if int(gvt_line.line_index) == current_gvt_line_index else 1
            score = (
                float(result["gvt_distance_to_conflict_m"])
                + float(result["vut_distance_to_conflict_m"])
            )
            if gvt_line_priority < best_priority or (
                gvt_line_priority == best_priority and score < best_score
            ):
                best_priority = gvt_line_priority
                best_score = score
                best_result = result
                best_result["gvt_line_index"] = int(gvt_line.line_index)
                best_result["gvt_lane_id"] = str(gvt_line.lane_id)
                best_result["gvt_lane_sequence"] = _normalize_lane_sequence(gvt_line.lane_sequence)
                best_result["main_line_index"] = int(main_line.line_index)
                best_result["main_lane_id"] = str(main_line.lane_id)
                best_result["main_lane_sequence"] = _normalize_lane_sequence(main_line.lane_sequence)

    if best_result is None:
        return {
            "valid": False,
            "reason": "no_forward_conflict_point",
            "conflict_type": conflict_type,
            "gvt_candidate_line_count": len(gvt_candidate_lines),
            "gvt_candidate_lane_ids": [str(line.lane_id) for line in gvt_candidate_lines],
            "invalid_scene_reasons": ["no_forward_conflict_point"],
        }

    return {
        "valid": True,
        "reason": "conflict_point_locked",
        "conflict_type": conflict_type,
        "conflict_xy": list(best_result["conflict_xy"]),
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
    """Keep only stable conflict fields in memory."""
    return {
        "conflict_type": str(conflict.get("conflict_type", "")),
        "conflict_xy": list(conflict.get("conflict_xy", [0.0, 0.0])),
        "gvt_line_index": int(conflict.get("gvt_line_index", 0)),
        "gvt_lane_id": str(conflict.get("gvt_lane_id", "")),
        "gvt_lane_sequence": _normalize_lane_sequence(conflict.get("gvt_lane_sequence", [])),
        "main_line_index": int(conflict.get("main_line_index", 0)),
        "main_lane_id": str(conflict.get("main_lane_id", "")),
        "main_lane_sequence": _normalize_lane_sequence(conflict.get("main_lane_sequence", [])),
        "conflict_angle_deg": float(conflict.get("conflict_angle_deg", 0.0)),
    }


def _compute_conflict_metrics(
    context: SceneContext,
    locked_conflict: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute TTA, delta_t and PET-like metrics for the locked conflict point."""
    if locked_conflict is None:
        return {
            "valid": False,
            "reason": "no_locked_conflict",
            "invalid_scene_reasons": ["no_locked_conflict"],
        }
    if context.main_vehicle is None:
        return {
            "valid": False,
            "reason": "no_main_vehicle",
            "invalid_scene_reasons": ["no_main_vehicle"],
        }

    gvt_line = _resolve_line(
        context.ego_vehicle.candidate_ref_lines,
        locked_conflict.get("gvt_lane_sequence", []),
        str(locked_conflict.get("gvt_lane_id", "")),
        int(locked_conflict.get("gvt_line_index", 0)),
    )
    if gvt_line is None:
        gvt_line = context.ego_vehicle.current_ref_line

    main_line = _resolve_line(
        context.main_vehicle.candidate_ref_lines,
        locked_conflict.get("main_lane_sequence", []),
        str(locked_conflict.get("main_lane_id", "")),
        int(locked_conflict.get("main_line_index", 0)),
    )
    if main_line is None:
        main_line = context.main_vehicle.current_ref_line

    conflict_xy = np.asarray(locked_conflict.get("conflict_xy", [0.0, 0.0]), dtype=float)
    _, gvt_s_conflict, gvt_d_conflict = project_point_to_path(conflict_xy, gvt_line.line)
    _, main_s_conflict, main_d_conflict = project_point_to_path(conflict_xy, main_line.line)
    gvt_distance = float(gvt_s_conflict) - float(gvt_line.current_s)
    vut_distance = float(main_s_conflict) - float(main_line.current_s)
    gvt_rel_xy = conflict_xy - np.array([float(context.ego_vehicle.x), float(context.ego_vehicle.y)], dtype=float)
    gvt_forward = np.array(
        [np.cos(float(context.ego_vehicle.heading)), np.sin(float(context.ego_vehicle.heading))],
        dtype=float,
    )
    gvt_longitudinal_to_conflict = float(np.dot(gvt_rel_xy, gvt_forward))
    vut_rel_xy = conflict_xy - np.array([float(context.main_vehicle.x), float(context.main_vehicle.y)], dtype=float)
    vut_forward = np.array(
        [np.cos(float(context.main_vehicle.heading)), np.sin(float(context.main_vehicle.heading))],
        dtype=float,
    )
    vut_longitudinal_to_conflict = float(np.dot(vut_rel_xy, vut_forward))

    speed_floor = max(float(CFG["speed_floor_mps"]), 1e-3)
    gvt_v = max(0.0, float(context.ego_vehicle.v))
    vut_v = max(0.0, float(context.main_vehicle.v))
    gvt_tta = gvt_distance / max(gvt_v, speed_floor) if gvt_distance >= 0.0 else 0.0
    vut_tta = vut_distance / max(vut_v, speed_floor) if vut_distance >= 0.0 else 0.0
    delta_t = float(gvt_tta) - float(vut_tta)
    zone = _compute_pet_like(
        gvt_distance=gvt_distance,
        gvt_v=max(gvt_v, speed_floor),
        vut_distance=vut_distance,
        vut_v=max(vut_v, speed_floor),
    )
    target_delta = float(CFG["target_delta_t_s"])
    delta_error = delta_t - target_delta

    return {
        "valid": True,
        "reason": "metrics_ok",
        "conflict_type": str(locked_conflict.get("conflict_type", "")),
        "conflict_xy": [float(conflict_xy[0]), float(conflict_xy[1])],
        "conflict_angle_deg": float(locked_conflict.get("conflict_angle_deg", 0.0)),
        "gvt_line_index": int(gvt_line.line_index),
        "gvt_lane_id": str(gvt_line.lane_id),
        "gvt_lane_sequence": _normalize_lane_sequence(gvt_line.lane_sequence),
        "main_line_index": int(main_line.line_index),
        "main_lane_id": str(main_line.lane_id),
        "main_lane_sequence": _normalize_lane_sequence(main_line.lane_sequence),
        "gvt_s_now_m": float(gvt_line.current_s),
        "gvt_s_conflict_m": float(gvt_s_conflict),
        "gvt_d_to_conflict_line_m": float(gvt_d_conflict),
        "gvt_distance_to_conflict_m": float(gvt_distance),
        "gvt_longitudinal_to_conflict_m": float(gvt_longitudinal_to_conflict),
        "vut_s_now_m": float(main_line.current_s),
        "vut_s_conflict_m": float(main_s_conflict),
        "vut_d_to_conflict_line_m": float(main_d_conflict),
        "vut_distance_to_conflict_m": float(vut_distance),
        "vut_longitudinal_to_conflict_m": float(vut_longitudinal_to_conflict),
        "gvt_v_mps": float(gvt_v),
        "vut_v_mps": float(vut_v),
        "gvt_tta_s": float(gvt_tta),
        "vut_tta_s": float(vut_tta),
        "target_delta_t_s": float(target_delta),
        "delta_t_s": float(delta_t),
        "delta_t_error_s": float(delta_error),
        "success_time_tol_s": float(CFG["success_time_tol_s"]),
        **zone,
    }


def _compute_pet_like(
    *,
    gvt_distance: float,
    gvt_v: float,
    vut_distance: float,
    vut_v: float,
) -> dict[str, Any]:
    radius = max(0.0, float(CFG["conflict_zone_radius_m"]))

    gvt_enter = max(0.0, float(gvt_distance) - radius) / max(float(gvt_v), 1e-3)
    gvt_exit = max(0.0, float(gvt_distance) + radius) / max(float(gvt_v), 1e-3)
    vut_enter = max(0.0, float(vut_distance) - radius) / max(float(vut_v), 1e-3)
    vut_exit = max(0.0, float(vut_distance) + radius) / max(float(vut_v), 1e-3)

    if gvt_exit < vut_enter:
        pet_like = float(vut_enter - gvt_exit)
        first_actor = "gvt"
        overlap = 0.0
    elif vut_exit < gvt_enter:
        pet_like = float(gvt_enter - vut_exit)
        first_actor = "vut"
        overlap = 0.0
    else:
        pet_like = 0.0
        first_actor = "overlap"
        overlap = min(gvt_exit, vut_exit) - max(gvt_enter, vut_enter)

    return {
        "conflict_zone_radius_m": float(radius),
        "gvt_zone_enter_t_s": float(gvt_enter),
        "gvt_zone_exit_t_s": float(gvt_exit),
        "vut_zone_enter_t_s": float(vut_enter),
        "vut_zone_exit_t_s": float(vut_exit),
        "pet_like_s": float(pet_like),
        "zone_first_actor": str(first_actor),
        "zone_overlap_s": float(max(0.0, overlap)),
    }


def _solve_sync_target(context: SceneContext, metrics: dict[str, Any]) -> dict[str, Any]:
    """Solve GVT speed/target needed to hit configured delta_t."""
    limits = _read_vehicle_limits(context)
    target_delta = float(CFG["target_delta_t_s"])
    desired_arrival_t = float(metrics.get("vut_tta_s", float("inf"))) + target_delta
    min_t = max(float(CFG["sync_plan_t_min_s"]), float(context.dt), 1e-3)
    gvt_distance = float(metrics.get("gvt_distance_to_conflict_m", 0.0))
    target_s_conflict = float(metrics.get("gvt_s_conflict_m", context.ego_vehicle.current_ref_line.current_s))
    buffer_s = max(0.0, float(CFG["post_conflict_target_buffer_m"]))
    target_s = target_s_conflict + buffer_s

    if not np.isfinite(desired_arrival_t) or desired_arrival_t < min_t or gvt_distance <= 0.0:
        raw_v = float("inf")
        feasible = False
        infeasible_reason = "invalid_desired_arrival_time"
        target_v = _clamp(float(CFG["cruise_target_v"]), 0.0, limits.max_v)
        plan_t = max(min_t, min(float(CFG["sync_plan_t_max_s"]), buffer_s / max(target_v, 1e-3)))
    else:
        raw_v = gvt_distance / max(desired_arrival_t, 1e-3)
        target_v = _clamp(raw_v, float(CFG["target_speed_min_mps"]), limits.max_v)
        plan_t = desired_arrival_t + buffer_s / max(target_v, float(CFG["speed_floor_mps"]))
        plan_t = _clamp(plan_t, min_t, float(CFG["sync_plan_t_max_s"]))
        avg_acc = (target_v - float(context.ego_vehicle.v)) / max(desired_arrival_t, 1e-3)
        feasible = bool(
            np.isfinite(raw_v)
            and abs(target_v - raw_v) <= 1e-3
            and limits.min_acc <= avg_acc <= limits.max_acc
        )
        if target_v < raw_v - 1e-3:
            infeasible_reason = "required_speed_above_limit"
        elif target_v > raw_v + 1e-3:
            infeasible_reason = "required_speed_below_min"
        elif avg_acc < limits.min_acc:
            infeasible_reason = "required_decel_below_limit"
        elif avg_acc > limits.max_acc:
            infeasible_reason = "required_acc_above_limit"
        else:
            infeasible_reason = ""

    return {
        "sync_control_mode": "delta_t_speed_sync",
        "desired_gvt_arrival_t_s": float(desired_arrival_t),
        "raw_required_v_mps": float(raw_v),
        "target_v_mps": float(target_v),
        "target_s_m": float(target_s),
        "plan_t_s": float(plan_t),
        "sync_target_feasible": bool(feasible),
        "sync_target_infeasible_reason": str(infeasible_reason),
    }


def _find_forward_conflict_point(
    gvt_line: ReferenceLineView,
    main_line: ReferenceLineView,
) -> dict[str, Any]:
    """Find the first forward geometric conflict point between two reference lines."""
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
    """Return distance, midpoint and segment parameters for a crossing/near-crossing pair."""
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
    """Closest points between two 2D line segments."""
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


def _update_actual_conflict_observation(
    memory: dict[str, Any],
    metrics: dict[str, Any],
    now: float,
) -> dict[str, Any]:
    """Record actual crossing/zone timestamps without changing control targets."""
    if not bool(metrics.get("valid", False)):
        return metrics

    prev_time = _optional_float(memory.get("last_observation_time_s"))
    gvt_signed = _signed_conflict_distance(metrics, "gvt")
    vut_signed = _signed_conflict_distance(metrics, "vut")

    _update_actor_actual_observation(
        memory=memory,
        actor="gvt",
        signed_distance=gvt_signed,
        prev_signed_distance=_optional_float(memory.get("last_gvt_signed_distance_m")),
        prev_time=prev_time,
        now=now,
    )
    _update_actor_actual_observation(
        memory=memory,
        actor="vut",
        signed_distance=vut_signed,
        prev_signed_distance=_optional_float(memory.get("last_vut_signed_distance_m")),
        prev_time=prev_time,
        now=now,
    )

    memory["last_observation_time_s"] = float(now)
    memory["last_gvt_signed_distance_m"] = float(gvt_signed) if np.isfinite(gvt_signed) else None
    memory["last_vut_signed_distance_m"] = float(vut_signed) if np.isfinite(vut_signed) else None

    actual = _build_actual_result_metrics(memory)
    metrics.update(actual)
    if bool(actual.get("actual_result_ready", False)) and not bool(memory.get("actual_result_printed", False)):
        _print_actual_conflict_result(metrics)
        memory["actual_result_printed"] = True
    return metrics


def _update_actor_actual_observation(
    *,
    memory: dict[str, Any],
    actor: str,
    signed_distance: float,
    prev_signed_distance: float | None,
    prev_time: float | None,
    now: float,
) -> None:
    if not np.isfinite(signed_distance):
        return

    radius = max(0.0, float(CFG["conflict_zone_radius_m"]))
    enter_key = f"{actor}_zone_enter_time_s"
    exit_key = f"{actor}_zone_exit_time_s"
    cross_key = f"{actor}_cross_time_s"

    if memory.get(enter_key) is None and signed_distance <= radius:
        memory[enter_key] = _threshold_cross_time(
            prev_signed_distance,
            signed_distance,
            prev_time,
            now,
            radius,
        )

    if memory.get(cross_key) is None and signed_distance <= 0.0:
        memory[cross_key] = _threshold_cross_time(
            prev_signed_distance,
            signed_distance,
            prev_time,
            now,
            0.0,
        )
        if memory.get(enter_key) is None:
            memory[enter_key] = memory[cross_key]

    if memory.get(exit_key) is None and memory.get(enter_key) is not None and signed_distance <= -radius:
        memory[exit_key] = _threshold_cross_time(
            prev_signed_distance,
            signed_distance,
            prev_time,
            now,
            -radius,
        )


def _build_actual_result_metrics(memory: dict[str, Any]) -> dict[str, Any]:
    gvt_cross = _optional_float(memory.get("gvt_cross_time_s"))
    vut_cross = _optional_float(memory.get("vut_cross_time_s"))
    actual_delta = float("nan")
    actual_error = float("nan")
    actual_abs_error = float("nan")
    delta_ready = gvt_cross is not None and vut_cross is not None
    if delta_ready:
        actual_delta = float(gvt_cross) - float(vut_cross)
        actual_error = actual_delta - float(CFG["target_delta_t_s"])
        actual_abs_error = abs(actual_error)

    gvt_enter = _optional_float(memory.get("gvt_zone_enter_time_s"))
    gvt_exit = _optional_float(memory.get("gvt_zone_exit_time_s"))
    vut_enter = _optional_float(memory.get("vut_zone_enter_time_s"))
    vut_exit = _optional_float(memory.get("vut_zone_exit_time_s"))
    pet_ready = all(value is not None for value in [gvt_enter, gvt_exit, vut_enter, vut_exit])
    actual_pet = float("nan")
    actual_overlap = float("nan")
    first_actor = "pending"
    if pet_ready:
        if float(gvt_exit) < float(vut_enter):
            actual_pet = float(vut_enter) - float(gvt_exit)
            actual_overlap = 0.0
            first_actor = "gvt"
        elif float(vut_exit) < float(gvt_enter):
            actual_pet = float(gvt_enter) - float(vut_exit)
            actual_overlap = 0.0
            first_actor = "vut"
        else:
            actual_pet = 0.0
            actual_overlap = min(float(gvt_exit), float(vut_exit)) - max(float(gvt_enter), float(vut_enter))
            first_actor = "overlap"

    if not delta_ready:
        result = "pending"
    elif actual_abs_error <= float(CFG["success_time_tol_s"]):
        result = "within_tolerance"
    else:
        result = "outside_tolerance"

    return {
        "actual_delta_t_ready": bool(delta_ready),
        "actual_pet_ready": bool(pet_ready),
        "actual_result_ready": bool(delta_ready and pet_ready),
        "actual_delta_t_s": float(actual_delta),
        "actual_delta_t_error_s": float(actual_error),
        "actual_delta_t_abs_error_s": float(actual_abs_error),
        "actual_delta_t_result": str(result),
        "actual_pet_like_s": float(actual_pet),
        "actual_zone_overlap_s": float(max(0.0, actual_overlap)) if np.isfinite(actual_overlap) else float("nan"),
        "actual_zone_first_actor": str(first_actor),
        "gvt_cross_time_s": float(gvt_cross) if gvt_cross is not None else float("nan"),
        "vut_cross_time_s": float(vut_cross) if vut_cross is not None else float("nan"),
        "gvt_zone_enter_time_s": float(gvt_enter) if gvt_enter is not None else float("nan"),
        "gvt_zone_exit_time_s": float(gvt_exit) if gvt_exit is not None else float("nan"),
        "vut_zone_enter_time_s": float(vut_enter) if vut_enter is not None else float("nan"),
        "vut_zone_exit_time_s": float(vut_exit) if vut_exit is not None else float("nan"),
    }


def _signed_conflict_distance(metrics: dict[str, Any], actor: str) -> float:
    longitudinal_key = f"{actor}_longitudinal_to_conflict_m"
    distance_key = f"{actor}_distance_to_conflict_m"
    longitudinal = _optional_float(metrics.get(longitudinal_key))
    if longitudinal is not None:
        return float(longitudinal)
    distance = _optional_float(metrics.get(distance_key))
    return float(distance) if distance is not None else float("nan")


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


def _is_predicted_timing_aligned(metrics: dict[str, Any]) -> bool:
    delta_error = abs(float(metrics.get("delta_t_error_s", float("inf"))))
    return bool(delta_error <= float(CFG["success_time_tol_s"]))


def _print_actual_conflict_result(metrics: dict[str, Any]) -> None:
    """Print the actual crossing/PET-like result once it is observable."""
    print(
        "[intersection_conflict] actual_result "
        f"conflict_type={metrics.get('conflict_type', '')} "
        f"actual_delta_t={float(metrics.get('actual_delta_t_s', float('nan'))):.3f}s "
        f"target={float(metrics.get('target_delta_t_s', float('nan'))):.3f}s "
        f"error={float(metrics.get('actual_delta_t_error_s', float('nan'))):.3f}s "
        f"actual_pet_like={float(metrics.get('actual_pet_like_s', float('nan'))):.3f}s "
        f"zone_overlap={float(metrics.get('actual_zone_overlap_s', float('nan'))):.3f}s "
        f"gvt_cross_time={float(metrics.get('gvt_cross_time_s', float('nan'))):.3f}s "
        f"vut_cross_time={float(metrics.get('vut_cross_time_s', float('nan'))):.3f}s "
        f"result={metrics.get('actual_delta_t_result', 'pending')}",
        flush=True,
    )


def _print_conflict_clear_metrics(metrics: dict[str, Any]) -> None:
    """Print once when GVT has cleared the conflict point and braking starts."""
    print(
        "[intersection_conflict] cleared "
        f"conflict_type={metrics.get('conflict_type', '')} "
        f"gvt_longitudinal_to_conflict={float(metrics.get('gvt_longitudinal_to_conflict_m', float('nan'))):.3f}m "
        f"delta_t={float(metrics.get('delta_t_s', float('nan'))):.3f}s "
        "switching_to=complete_brake",
        flush=True,
    )


def _gvt_cleared_conflict(metrics: dict[str, Any]) -> bool:
    clear_after = float(CFG["clear_after_conflict_m"])
    distance = float(metrics.get("gvt_distance_to_conflict_m", float("inf")))
    longitudinal = float(metrics.get("gvt_longitudinal_to_conflict_m", float("inf")))
    return bool(distance <= -clear_after or longitudinal <= -clear_after)


def _candidate_main_lines(context: SceneContext) -> list[ReferenceLineView]:
    if context.main_vehicle is None:
        return []
    current = context.main_vehicle.current_ref_line
    lines = [current]
    for line in context.main_vehicle.candidate_ref_lines:
        if int(line.line_index) == int(current.line_index):
            continue
        lines.append(line)
    return lines


def _candidate_gvt_lines(context: SceneContext) -> list[ReferenceLineView]:
    current = context.ego_vehicle.current_ref_line
    lines = [current]
    for line in context.ego_vehicle.candidate_ref_lines:
        if int(line.line_index) == int(current.line_index):
            continue
        lines.append(line)
    return lines


def _resolve_gvt_line(context: SceneContext, metrics: dict[str, Any]) -> ReferenceLineView:
    line = _resolve_line(
        context.ego_vehicle.candidate_ref_lines,
        metrics.get("gvt_lane_sequence", []),
        str(metrics.get("gvt_lane_id", "")),
        int(metrics.get("gvt_line_index", 0)),
    )
    if line is not None:
        return line
    return context.ego_vehicle.current_ref_line


def _resolve_line(
    candidate_lines: list[ReferenceLineView],
    lane_sequence: list[str] | tuple[str, ...] | None,
    lane_id: str,
    line_index: int | None,
) -> ReferenceLineView | None:
    line = _line_by_lane_sequence(candidate_lines, lane_sequence)
    if line is not None:
        return line
    line = _line_by_lane_id(candidate_lines, lane_id)
    if line is not None:
        return line
    return _line_by_index(candidate_lines, line_index)


def _line_by_index(candidate_lines: list[ReferenceLineView], line_index: int | None) -> ReferenceLineView | None:
    if line_index is None:
        return None
    for line in candidate_lines:
        if int(line.line_index) == int(line_index):
            return line
    return None


def _line_by_lane_id(candidate_lines: list[ReferenceLineView], lane_id: str) -> ReferenceLineView | None:
    target_lane_id = str(lane_id or "")
    if not target_lane_id:
        return None
    for line in candidate_lines:
        if str(line.lane_id) == target_lane_id:
            return line
    return None


def _line_by_lane_sequence(
    candidate_lines: list[ReferenceLineView],
    lane_sequence: list[str] | tuple[str, ...] | None,
) -> ReferenceLineView | None:
    target_sequence = _normalize_lane_sequence(lane_sequence)
    if not target_sequence:
        return None
    target_set = set(target_sequence)
    best_line = None
    best_score: tuple[int, int, int] | None = None
    for line in candidate_lines:
        line_sequence = _normalize_lane_sequence(line.lane_sequence)
        overlap_count = sum(1 for lane_id in line_sequence if lane_id in target_set)
        if overlap_count <= 0:
            continue
        first_match_idx = len(target_sequence)
        for idx, lane_id in enumerate(target_sequence):
            if lane_id in line_sequence:
                first_match_idx = idx
                break
        score = (overlap_count, -first_match_idx, -len(line_sequence))
        if best_score is None or score > best_score:
            best_score = score
            best_line = line
    return best_line


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
