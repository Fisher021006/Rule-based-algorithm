"""基于 THW 精确解 + 退化解的 cut-in 场景算法。

设计目标：
- PREPARE 主动组织纵向状态，建立 4s 精准 cut-in 的发射态
- ACTIVE 固定 4s 执行精确 THW cut-in
- exact 不可行时，退化到当前可行 THW 边界继续执行
- 仅保留场景车自身可执行性约束，不再用主车安全约束保护被测主车
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
    """场景算法实际使用的车辆尺寸与平台约束。"""

    ego_length: float
    main_length: float
    max_v: float
    min_acc: float
    max_acc: float


CFG: dict[str, Any] = {
    "cutin_side": "auto",
    "target_thw_s": 8.0,
    "thw_tol_s": 0.1,
    "cruise_target_v": 5.0,
    "lane_change_time_s": 4.0,
    "prepare_plan_t_s": 1.5,
    "prepare_convergence_time_s": 2.0,
    "prepare_merge_lead_m": 1.0,
    "launch_delta_s_min_m": 0,
    "prepare_max_speed_adv_mps": 3.0,
    "prepare_max_speed_lag_mps": 2.0,
    "prepare_commit_debounce_frames": 1,
    "prepare_build_timeout_s": 6.0,
    "prepare_acc_margin_mps2": 0.1,
    "search_rel_s_min": -35.0,
    "search_rel_s_max": 25.0,
    "max_heading_diff_deg": 20.0,
    "target_lane_match_d_thresh": 1.5,
    "max_ego_to_target_line_d_m": 4.5,
    "min_physical_gap_m": 0.0,
    "physical_gap_tol_m": 0.2,
    "min_target_progress_m": 2.0,
    "post_merge_buffer_m": 5.0,
    "dynamic_solution_min_t_s": 0.5,
    "dead_zone_time_s": 2.0,
    "dead_zone_lateral_progress_ratio": 0.5,
    "dead_zone_pred_main_v_thresh": 0.5,
    "dead_zone_pred_main_v_min_progress": 0.25,
    "active_transient_hold_max_frames": 2,
    "active_persistent_infeasible_max_frames": 4,
    "persistent_infeasible_progress_ratio_thresh": 0.5,
    "enter_target_d_thresh_m": 0.6,
    "execute_complete_frames": 2,
    "stabilize_duration_s": 1.5,
    "abort_cooldown_s": 2.0,
    "prediction_min_acc": -1.5,
    "prediction_max_acc": 1.5,
}


SCENE_MEMORY_TEMPLATE: dict[str, Any] = {
    "mode": "APPROACH",
    "cooldown_until_time_s": -1.0,
    "prepare_start_time_s": -1.0,
    "prepare_success_hits": 0,
    "prepare_line_index": None,
    "prepare_target_lane_id": "",
    "prepare_target_lane_sequence": [],
    "locked_line_index": None,
    "locked_target_lane_id": "",
    "locked_target_lane_sequence": [],
    "lane_change_end_time_s": -1.0,
    "execution_kind": "",
    "execution_requested_thw_s": 0.0,
    "execution_degrade_reason": "",
    "last_dynamic_solution": None,
    "frozen_solution": None,
    "active_infeasible_count": 0,
    "active_start_abs_d_to_target": 0.0,
    "dead_zone_trigger_reason": "",
    "dead_zone_trigger_progress": 0.0,
    "dead_zone_trigger_pred_main_v": -1.0,
    "execute_complete_hits": 0,
    "stable_until_time_s": -1.0,
    # 完成指标在首次确认进入目标车道时锁存，不随 STABILIZE 结束清空。
    "completion_time_s": -1.0,
    "completion_lane_id": "",
    "completion_execution_kind": "",
    "completion_requested_thw_s": 0.0,
    "completion_achieved_thw_s": None,
    "completion_thw_error_s": None,
    "completion_metric_reason": "",
}


SCENE_MEMORY: dict[str, Any] = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Public API / 外部入口 ====================

def cut_in_algorithm(context: SceneContext) -> dict[str, Any]:
    """Cut-in 场景算法主入口。"""
    # 逻辑：
    # 1. 每轮新场景重置状态机内存。
    # 2. 调用状态机后打印耗时，便于离线和线上排查。
    global SCENE_MEMORY
    start_ms = time.time() * 1000.0
    if int(context.step_index) == 0:
        _reset_scene_memory()
    output, SCENE_MEMORY = _scene_fsm_step(context, SCENE_MEMORY)
    print(f"[cutin] mode={SCENE_MEMORY.get('mode', '')} scene_status={output.get('scene_status', '')} reason={output.get('scene_detail', output.get('reason', ''))} main_ptc_id={getattr(getattr(context.planning_frame, 'main_actor', None), 'ptc_id', '')} relative_s={None if context.main_vehicle is None else round(context.main_vehicle.relative_s_to_ego, 3)} relative_d={None if context.main_vehicle is None else round(context.main_vehicle.relative_d_to_ego, 3)} prepare_delta_s_target={output.get('scene_debug', {}).get('prepare_delta_s_target', '')} prepare_target_v={output.get('scene_debug', {}).get('prepare_target_v', '')} target_line_index={output.get('target_line_index', '')} prepare_state={output.get('scene_debug', {}).get('prepare_state', output.get('scene_status', ''))} feasibility_reason={output.get('scene_debug', {}).get('exact_activate_reason', output.get('scene_debug', {}).get('exact_reason', output.get('scene_detail', '')))}")
    end_ms = time.time() * 1000.0
    print(f"Cut-in algorithm execution time: {end_ms - start_ms} ms")
    return output


# ==================== FSM / 状态机分发 ====================

def _scene_fsm_step(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行一帧 cut-in 状态机，并输出本帧场景目标。"""
    # 逻辑：
    # 1. 这里只保留状态分发；每个状态的解算细节下沉到对应 handler。
    now = float(context.time_s)
    mode = str(memory.get("mode", "APPROACH"))

    # COMPLETE 是单轮场景的闭锁终态。它不再依赖 VUT 实时状态，
    # 避免主车 OBU 短时丢失后把已完成的切入重置为 APPROACH。
    if mode == "COMPLETE":
        return _handle_complete(context, memory)

    if mode == "COOLDOWN":
        if now < float(memory["cooldown_until_time_s"]):
            return _handle_cooldown(context, memory, now)
        memory["mode"] = "APPROACH"
        mode = "APPROACH"

    if mode not in {"ACTIVE", "FROZEN", "STABILIZE"} and not _has_valid_main_vehicle(context):
        return _handle_no_valid_main(context, memory)

    if mode == "APPROACH":
        return _handle_approach(context, memory, now)
    if mode == "PREPARE":
        return _handle_prepare(context, memory, now)
    if mode == "ACTIVE":
        return _handle_active(context, memory, now)
    if mode == "FROZEN":
        return _handle_frozen(context, memory, now)
    if mode == "STABILIZE":
        return _handle_stabilize(context, memory, now)
    return _handle_unknown_mode(context, memory)


# ==================== State Handlers / 每个状态独立逻辑函数 ====================

def _handle_cooldown(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 COOLDOWN：冷却期内只输出当前车道保底目标。"""
    return _build_keep_lane_output(
        context,
        reason="baseline_cooldown",
        scene_status="cooldown",
    ), memory


def _handle_no_valid_main(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理非执行态下没有有效主车的情况。"""
    memory["mode"] = "APPROACH"
    _reset_prepare_memory(memory)
    _reset_execute_memory(memory)
    return _build_keep_lane_output(
        context,
        reason="baseline_no_valid_main",
        scene_status="waiting_valid_main",
    ), memory


def _handle_approach(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 APPROACH：寻找目标车道并进入 PREPARE。"""
    target_line = _choose_target_line(context)
    if target_line is None:
        return _build_keep_lane_output(
            context,
            reason="approach_no_target_line",
            scene_status="approach_wait_target_line",
        ), memory

    _enter_prepare_mode(memory, target_line, now)
    launch_plan = _evaluate_launch_plan(context, target_line, float(CFG["lane_change_time_s"]))
    prepare_strategy = _build_prepare_strategy(context, target_line)
    return _build_prepare_output(
        context,
        memory,
        launch_plan,
        prepare_strategy,
        reason="prepare_enter",
    ), memory


def _handle_prepare(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 PREPARE：组织纵向状态，并判断 exact/degraded 是否可发射。"""
    target_line = _resolve_prepare_target_line(context, memory)
    if target_line is None:
        memory["mode"] = "APPROACH"
        _reset_prepare_memory(memory)
        return _build_keep_lane_output(
            context,
            reason="prepare_no_target_line",
            scene_status="approach_wait_target_line",
        ), memory

    launch_plan = _evaluate_launch_plan(context, target_line, float(CFG["lane_change_time_s"]))
    prepare_strategy = _build_prepare_strategy(context, target_line)

    if _prepare_should_abort_as_globally_infeasible(memory, launch_plan, now):
        _enter_cooldown(memory, now)
        return _build_keep_lane_output(
            context,
            reason="prepare_globally_infeasible",
            scene_status="globally_infeasible",
            scene_detail=str(launch_plan["summary_reason"]),
            scene_debug=launch_plan["debug"],
        ), memory

    exact_ready, exact_debug = _prepare_should_activate_exact(
        context,
        target_line,
        launch_plan,
        prepare_strategy,
    )
    if exact_ready:
        memory["prepare_success_hits"] += 1
        if int(memory["prepare_success_hits"]) >= int(CFG["prepare_commit_debounce_frames"]):
            solution = _annotate_output(
                copy.deepcopy(launch_plan["exact_result"]["solution"]),
                scene_status="exact_executable",
                scene_detail="prepare_to_active_exact",
                scene_debug=launch_plan["debug"],
            )
            _enter_active_mode(
                memory,
                target_line,
                solution,
                now,
                execution_kind="exact",
                requested_thw_s=float(CFG["target_thw_s"]),
                degrade_reason="",
            )
            return solution, memory
    else:
        memory["prepare_success_hits"] = 0

    degraded_ready, degraded_debug = _prepare_should_activate_degraded(
        context,
        target_line,
        memory,
        launch_plan,
        prepare_strategy,
        now,
    )
    if degraded_ready:
        memory["prepare_success_hits"] += 1
        if int(memory["prepare_success_hits"]) >= int(CFG["prepare_commit_debounce_frames"]):
            degraded_solution = _annotate_output(
                copy.deepcopy(launch_plan["degraded_result"]["solution"]),
                scene_status="degraded_executable",
                scene_detail=str(launch_plan["degrade_reason"]),
                scene_debug=launch_plan["debug"],
            )
            _print_degrade_event(
                prefix="prepare",
                degrade_reason=str(launch_plan["degrade_reason"]),
                target_thw=float(CFG["target_thw_s"]),
                degraded_thw=float(launch_plan["degraded_result"]["requested_thw"]),
                debug=launch_plan["debug"],
            )
            _enter_active_mode(
                memory,
                target_line,
                degraded_solution,
                now,
                execution_kind="degraded",
                requested_thw_s=float(launch_plan["degraded_result"]["requested_thw"]),
                degrade_reason=str(launch_plan["degrade_reason"]),
            )
            return degraded_solution, memory
    elif not exact_ready:
        memory["prepare_success_hits"] = 0

    prepare_debug = copy.deepcopy(launch_plan["debug"])
    prepare_debug["exact_activate_ok"] = bool(exact_ready)
    prepare_debug["degraded_activate_ok"] = bool(degraded_ready)
    prepare_debug["exact_activate_reason"] = str(exact_debug["reason"])
    prepare_debug["degraded_activate_reason"] = str(degraded_debug["reason"])
    return _build_prepare_output(
        context,
        memory,
        launch_plan,
        prepare_strategy,
        reason=str(launch_plan["prepare_reason"]),
        scene_debug=prepare_debug,
    ), memory


def _handle_active(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 ACTIVE：动态重解 exact/degraded，并在末段进入冻结执行。"""
    target_line = _resolve_target_line(context, memory)
    if target_line is None:
        _enter_cooldown(memory, now)
        return _build_keep_lane_output(
            context,
            reason="active_no_target_line",
            scene_status="infeasible_geometry",
        ), memory

    if _should_enter_dead_zone(context, memory, target_line, now):
        freeze_solution = memory.get("last_dynamic_solution")
        if freeze_solution is None:
            freeze_solution, failure_info = _select_active_solution(
                context,
                memory,
                target_line,
                _remaining_lane_change_time(memory, now),
            )
            if freeze_solution is None:
                _enter_cooldown(memory, now)
                return _build_keep_lane_output(
                    context,
                    reason="active_dead_zone_without_solution",
                    scene_status=str(failure_info["scene_status"]),
                    scene_detail=str(failure_info["reason"]),
                    scene_debug=failure_info["debug"],
                ), memory

        _enter_frozen_mode(memory, freeze_solution, now)
        frozen_output = _build_frozen_output(context, memory, target_line)
        if frozen_output is None:
            _enter_cooldown(memory, now)
            return _build_keep_lane_output(
                context,
                reason="active_frozen_build_failed",
                scene_status="infeasible_geometry",
            ), memory
        return frozen_output, memory

    active_output, failure_info = _select_active_solution(
        context,
        memory,
        target_line,
        _remaining_lane_change_time(memory, now),
    )
    if active_output is None:
        return _handle_active_infeasible(context, memory, target_line, now, failure_info), memory

    memory["active_infeasible_count"] = 0
    memory["last_dynamic_solution"] = copy.deepcopy(active_output)
    if _is_execute_completed(context, memory, target_line):
        _enter_stabilize_mode(context, memory, target_line, now)
        return _build_stabilize_output(context, target_line), memory
    return active_output, memory


def _handle_frozen(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 FROZEN：复用冻结解，不再动态重解 THW。"""
    target_line = _resolve_target_line(context, memory)
    frozen_output = _build_frozen_output(context, memory, target_line)
    if frozen_output is None:
        _enter_cooldown(memory, now)
        return _build_keep_lane_output(
            context,
            reason="frozen_solution_lost",
            scene_status="infeasible_geometry",
        ), memory

    if _is_execute_completed(context, memory, target_line):
        _enter_stabilize_mode(context, memory, target_line, now)
        if target_line is not None:
            return _build_stabilize_output(context, target_line), memory
    return frozen_output, memory


def _handle_stabilize(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 STABILIZE：换道完成后短时稳定跟随。"""
    target_line = _resolve_target_line(context, memory)
    if now >= float(memory["stable_until_time_s"]):
        _enter_complete_mode(memory)
        return _build_complete_output(context, memory), memory

    if target_line is None:
        # 进入 STABILIZE 前已经连续确认进入目标车道。此时锁定参考线
        # 短时丢失只降级为当前车道巡航，不撤销已确认的切入结果。
        return _build_keep_lane_output(
            context,
            reason="stabilize_no_target_line",
            scene_status="stabilize_following",
            scene_debug=_completion_debug(memory),
        ), memory

    return _build_stabilize_output(context, target_line), memory


def _handle_complete(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理 COMPLETE：闭锁本轮切入成功，同时保持当前车道巡航。"""

    return _build_complete_output(context, memory), memory


def _handle_unknown_mode(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """处理未知状态：回到 APPROACH 并输出保底目标。"""
    memory["mode"] = "APPROACH"
    return _build_keep_lane_output(
        context,
        reason="baseline_default",
        scene_status="baseline_keep_lane",
    ), memory


# ==================== Transition Helpers / 状态切换与记忆管理 ====================

def _enter_prepare_mode(memory: dict[str, Any], target_line: ReferenceLineView, now: float) -> None:
    """进入 PREPARE，并锁定目标车道。"""
    # 逻辑：
    # 1. PREPARE 期间固定目标车道，避免每帧换目标导致发射态漂移。
    memory["mode"] = "PREPARE"
    memory["prepare_start_time_s"] = float(now)
    memory["prepare_success_hits"] = 0
    memory["prepare_line_index"] = int(target_line.line_index)
    memory["prepare_target_lane_id"] = str(target_line.lane_id)
    memory["prepare_target_lane_sequence"] = _normalize_lane_sequence(target_line.lane_sequence)


def _enter_active_mode(
    memory: dict[str, Any],
    target_line: ReferenceLineView,
    solution: dict[str, Any],
    now: float,
    *,
    execution_kind: str,
    requested_thw_s: float,
    degrade_reason: str,
) -> None:
    """进入 ACTIVE，并锁定执行配置。"""
    # 逻辑：
    # 1. ACTIVE 之后只执行固定 4s 的 cut-in，不再重新定义场景目标。
    # 2. execution_kind 只区分 exact / degraded，degrade_reason 用于打印和调试。
    memory["mode"] = "ACTIVE"
    memory["locked_line_index"] = int(target_line.line_index)
    memory["locked_target_lane_id"] = str(target_line.lane_id)
    memory["locked_target_lane_sequence"] = _normalize_lane_sequence(target_line.lane_sequence)
    memory["lane_change_end_time_s"] = float(now) + float(CFG["lane_change_time_s"])
    memory["execution_kind"] = str(execution_kind)
    memory["execution_requested_thw_s"] = float(requested_thw_s)
    memory["execution_degrade_reason"] = str(degrade_reason)
    memory["last_dynamic_solution"] = copy.deepcopy(solution)
    memory["frozen_solution"] = None
    memory["active_infeasible_count"] = 0
    memory["active_start_abs_d_to_target"] = abs(float(solution.get("_meta", {}).get("ego_d_on_target", 0.0)))
    memory["dead_zone_trigger_reason"] = ""
    memory["dead_zone_trigger_progress"] = 0.0
    memory["dead_zone_trigger_pred_main_v"] = -1.0
    memory["execute_complete_hits"] = 0
    _reset_prepare_memory(memory)


def _enter_frozen_mode(
    memory: dict[str, Any],
    solution: dict[str, Any],
    now: float,
) -> None:
    """进入 FROZEN，并打印冻结时刻的关键信息。"""
    # 逻辑：
    # 1. 冻结后继续执行最后一个有效解，避免末段目标频繁跳变。
    frozen_solution = copy.deepcopy(solution)
    trigger_reason = str(memory.get("dead_zone_trigger_reason", ""))
    trigger_progress = float(memory.get("dead_zone_trigger_progress", 0.0))
    trigger_pred_main_v = float(memory.get("dead_zone_trigger_pred_main_v", -1.0))
    if trigger_reason == "pred_main_low_speed":
        _annotate_output(
            frozen_solution,
            scene_status="metric_invalid_low_main_speed",
            scene_detail="pred_main_low_speed",
            scene_debug={
                "dead_zone_trigger_reason": trigger_reason,
                "dead_zone_trigger_progress": float(trigger_progress),
                "dead_zone_trigger_pred_main_v": float(trigger_pred_main_v),
            },
        )
    else:
        _annotate_output(
            frozen_solution,
            scene_status=str(frozen_solution.get("scene_status", "frozen_execute")),
            scene_detail=trigger_reason or str(frozen_solution.get("scene_detail", "frozen_execute")),
            scene_debug={
                "dead_zone_trigger_reason": trigger_reason,
                "dead_zone_trigger_progress": float(trigger_progress),
                "dead_zone_trigger_pred_main_v": float(trigger_pred_main_v),
            },
        )

    memory["mode"] = "FROZEN"
    memory["frozen_solution"] = frozen_solution
    memory["last_dynamic_solution"] = copy.deepcopy(frozen_solution)
    pred_main_v_text = f" pred_main_v={trigger_pred_main_v:.3f}" if trigger_pred_main_v >= 0.0 else ""
    print(
        f"[cutin] freeze time={float(now):.2f}s trigger={trigger_reason} "
        f"progress={trigger_progress:.3f}{pred_main_v_text} {_solution_thw_text(frozen_solution)}",
        flush=True,
    )


def _enter_stabilize_mode(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView | None,
    now: float,
) -> None:
    """进入 STABILIZE，并锁存首次确认换道完成时的 THW。"""

    achieved_thw, metric_reason = _measure_completion_thw(context)
    requested_thw = float(memory.get("execution_requested_thw_s", CFG["target_thw_s"]))
    memory["mode"] = "STABILIZE"
    memory["stable_until_time_s"] = float(now) + float(CFG["stabilize_duration_s"])
    memory["completion_time_s"] = float(now)
    memory["completion_lane_id"] = str(
        memory.get("locked_target_lane_id", "")
        or ("" if target_line is None else target_line.lane_id)
        or context.ego_vehicle.lane_id
    )
    memory["completion_execution_kind"] = str(memory.get("execution_kind", ""))
    memory["completion_requested_thw_s"] = float(requested_thw)
    memory["completion_achieved_thw_s"] = (
        None if achieved_thw is None else float(achieved_thw)
    )
    memory["completion_thw_error_s"] = (
        None if achieved_thw is None else float(achieved_thw) - float(requested_thw)
    )
    memory["completion_metric_reason"] = str(metric_reason)
    thw_text = "nan" if achieved_thw is None else f"{float(achieved_thw):.3f}"
    print(
        f"[cutin] lane_change_complete time={float(now):.2f}s "
        f"lane={memory['completion_lane_id']} "
        f"execution={memory['completion_execution_kind']} "
        f"thw={thw_text} requested={requested_thw:.3f} "
        f"metric_reason={metric_reason}",
        flush=True,
    )


def _enter_complete_mode(memory: dict[str, Any]) -> None:
    """进入单轮闭锁完成态，只清理不再需要的 PREPARE 上下文。"""

    memory["mode"] = "COMPLETE"
    _reset_prepare_memory(memory)


def _enter_cooldown(memory: dict[str, Any], now: float) -> None:
    """进入冷却态，并清空本轮执行上下文。"""
    # 逻辑：
    # 1. 冷却态只保留下一轮场景启动所需的最小状态。
    memory["mode"] = "COOLDOWN"
    _reset_prepare_memory(memory)
    memory["cooldown_until_time_s"] = float(now) + float(CFG["abort_cooldown_s"])
    _reset_execute_memory(memory)


def _reset_prepare_memory(memory: dict[str, Any]) -> None:
    """清空 PREPARE 阶段缓存。"""
    # 逻辑：
    # 1. PREPARE 只缓存锁定目标车道和准备开始时间。
    memory["prepare_start_time_s"] = -1.0
    memory["prepare_success_hits"] = 0
    memory["prepare_line_index"] = None
    memory["prepare_target_lane_id"] = ""
    memory["prepare_target_lane_sequence"] = []


def _reset_execute_memory(memory: dict[str, Any]) -> None:
    """清空 ACTIVE/FROZEN/STABILIZE 共用的执行缓存。"""
    # 逻辑：
    # 1. exact/degraded 执行种类、冻结解和完成判定都统一从这里回收。
    memory["locked_line_index"] = None
    memory["locked_target_lane_id"] = ""
    memory["locked_target_lane_sequence"] = []
    memory["lane_change_end_time_s"] = -1.0
    memory["execution_kind"] = ""
    memory["execution_requested_thw_s"] = 0.0
    memory["execution_degrade_reason"] = ""
    memory["last_dynamic_solution"] = None
    memory["frozen_solution"] = None
    memory["active_infeasible_count"] = 0
    memory["active_start_abs_d_to_target"] = 0.0
    memory["dead_zone_trigger_reason"] = ""
    memory["dead_zone_trigger_progress"] = 0.0
    memory["dead_zone_trigger_pred_main_v"] = -1.0
    memory["execute_complete_hits"] = 0
    memory["stable_until_time_s"] = -1.0


def _reset_scene_memory() -> None:
    """把全局场景内存重置到初始值。"""
    # 逻辑：
    # 1. 每轮新场景从模板深拷贝，避免跨轮污染。
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
    """构造沿当前参考线前进的保底目标点。"""
    ego = context.ego_vehicle
    line = ego.current_ref_line
    limits = _read_vehicle_limits(context)
    target_v = _clamp(float(CFG["cruise_target_v"]), 0.0, limits.max_v)
    plan_t = 2.0
    travel_s = max(float(CFG["min_target_progress_m"]), target_v * plan_t + 4.0)
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
        "a": 0.0,
        "theta": float(target_heading),
        "t": float(plan_t),
        "target_line_index": int(line.line_index),
        "hold_t": 0.0,
        "reason": str(reason),
    }
    return _annotate_output(output, scene_status, scene_detail or reason, scene_debug)


def _build_complete_output(
    context: SceneContext,
    memory: dict[str, Any],
) -> dict[str, Any]:
    """构造切入完成后的当前车道巡航目标。"""

    return _build_keep_lane_output(
        context,
        reason="cutin_complete_cruise",
        scene_status="cutin_success",
        scene_detail="cutin_complete_latched",
        scene_debug=_completion_debug(memory),
    )


def _build_prepare_output(
    context: SceneContext,
    memory: dict[str, Any],
    launch_plan: dict[str, Any],
    prepare_strategy: dict[str, Any],
    reason: str,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 PREPARE 阶段的纵向组织输出。"""
    # 逻辑：
    # 1. PREPARE 的目标不是跟主车同速，而是把系统送进可切入的发射态。
    # 2. exact 不可行时，会主动建立纵向优势；degraded 可行时，则为退化切入准备起始态。
    ego = context.ego_vehicle
    line = ego.current_ref_line
    target_v = float(prepare_strategy["target_v"])
    plan_t = float(CFG["prepare_plan_t_s"])
    travel_s = max(float(CFG["min_target_progress_m"]), target_v * plan_t + 1.0)
    target_s = _clamp(
        float(line.current_s) + float(travel_s),
        float(line.current_s) + float(CFG["min_target_progress_m"]),
        float(line.cumulative_s[-1]) - 0.5,
    )
    target_xy, target_heading = get_path_point_at_s(line.line, target_s, line.cumulative_s)
    limits = _read_vehicle_limits(context)
    a_req = _clamp(
        (target_v - float(ego.v)) / max(plan_t, 1e-3),
        limits.min_acc,
        limits.max_acc,
    )

    debug = copy.deepcopy(launch_plan["debug"])
    debug.update(prepare_strategy["debug"])
    if scene_debug is not None:
        debug.update(scene_debug)

    output = {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(target_v),
        "a": float(a_req),
        "theta": float(target_heading),
        "t": float(plan_t),
        "target_line_index": int(line.line_index),
        "hold_t": 0.0,
        "reason": str(reason),
        "prepare_success_hits": int(memory.get("prepare_success_hits", 0)),
        "prepare_delta_s_target": float(prepare_strategy["delta_s_target"]),
        "prepare_delta_s_goal": float(prepare_strategy["delta_s_goal"]),
        "prepare_delta_s_error": float(prepare_strategy["delta_s_error"]),
    }
    return _annotate_output(
        output,
        scene_status=str(launch_plan["prepare_state"]),
        scene_detail=str(launch_plan["summary_reason"]),
        scene_debug=debug,
    )


def _build_projected_solution_output(
    target_line: ReferenceLineView | None,
    solution: dict[str, Any],
    *,
    remaining_t: float,
    reason: str,
    scene_status: str | None = None,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把历史解投影到当前目标线上，保持几何连续。"""
    # 逻辑：
    # 1. ACTIVE 和 FROZEN 都会复用这里，把上一次有效目标重新贴到当前目标线。
    x = float(solution["x"])
    y = float(solution["y"])
    theta = float(solution["theta"])
    target_line_index = int(solution.get("target_line_index", 0))

    if target_line is not None:
        target_s = solution.get("_meta", {}).get("target_s")
        if target_s is None:
            target_xy = np.array([float(solution["x"]), float(solution["y"])], dtype=float)
            _, target_s, _ = project_point_to_path(target_xy, target_line.line)
        target_s = max(float(target_s), float(target_line.current_s) + 1.0)
        target_s = min(float(target_s), float(target_line.cumulative_s[-1]) - 0.5)
        if target_s > float(target_line.current_s) + 1e-6:
            target_xy, target_heading = get_path_point_at_s(target_line.line, target_s, target_line.cumulative_s)
            x = float(target_xy[0])
            y = float(target_xy[1])
            theta = float(target_heading)
            target_line_index = int(target_line.line_index)

    output = copy.deepcopy(solution)
    output["x"] = float(x)
    output["y"] = float(y)
    output["theta"] = float(theta)
    output["t"] = float(max(float(CFG["dynamic_solution_min_t_s"]), remaining_t))
    output["target_line_index"] = int(target_line_index)
    output["hold_t"] = 0.0
    output["reason"] = str(reason)
    return _annotate_output(
        output,
        scene_status or str(output.get("scene_status", "active_execute")),
        scene_detail or str(output.get("scene_detail", reason)),
        scene_debug if scene_debug is not None else output.get("scene_debug"),
    )


def _build_frozen_output(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView | None,
) -> dict[str, Any] | None:
    """根据冻结解构造 FROZEN 输出。"""
    # 逻辑：
    # 1. 死区后不再重解，只执行最后一次有效解的连续投影。
    solution = memory.get("frozen_solution")
    if solution is None:
        return None
    return _build_projected_solution_output(
        target_line=target_line,
        solution=solution,
        remaining_t=_remaining_lane_change_time(memory, context.time_s),
        reason="cutin_frozen_execute",
    )


def _build_stabilize_output(
    context: SceneContext,
    target_line: ReferenceLineView,
) -> dict[str, Any]:
    """构造换道完成后的短时稳定跟随输出。"""
    # 逻辑：
    # 1. STABILIZE 只做短时跟随，不再改变场景定义。
    limits = _read_vehicle_limits(context)
    target_v = (
        _clamp(float(context.main_vehicle.v), 0.0, limits.max_v)
        if context.main_vehicle is not None
        else _clamp(float(CFG["cruise_target_v"]), 0.0, limits.max_v)
    )
    travel_s = max(8.0, target_v * 2.0 + 4.0)
    target_s = min(float(target_line.current_s) + float(travel_s), float(target_line.cumulative_s[-1]) - 0.5)
    target_xy, target_heading = get_path_point_at_s(target_line.line, target_s, target_line.cumulative_s)
    output = {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(target_v),
        "a": 0.0,
        "theta": float(target_heading),
        "t": 2.0,
        "target_line_index": int(target_line.line_index),
        "hold_t": 0.0,
        "reason": "cutin_stabilize",
    }
    return _annotate_output(output, "stabilize_following", "cutin_stabilize")


# ==================== Condition Helpers / 触发与状态判断 ====================

def _has_valid_main_vehicle(context: SceneContext) -> bool:
    """判断当前主车是否满足场景基本识别条件。"""
    # 逻辑：
    # 1. 这里只做主车对象识别，不做 cut-in 可执行性判断。
    if context.main_vehicle is None:
        return False
    ego = context.ego_vehicle
    main = context.main_vehicle
    same_direction = _heading_diff_deg(main.heading, ego.heading) <= float(CFG["max_heading_diff_deg"])
    rel_s_ok = float(CFG["search_rel_s_min"]) <= float(main.relative_s_to_ego) <= float(CFG["search_rel_s_max"])
    return bool(same_direction and rel_s_ok)


def _prepare_should_activate_exact(
    context: SceneContext,
    target_line: ReferenceLineView,
    launch_plan: dict[str, Any],
    prepare_strategy: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """判断是否满足进入 ACTIVE_EXACT 的条件。"""
    # 逻辑：
    # 1. exact 需要同时满足：精确解存在、起始窗口已收敛、启动减速度余量可接受。
    exact_result = launch_plan["exact_result"]
    if not exact_result["ok"]:
        return False, {"reason": str(exact_result["reason"])}
    window_ok, window_debug = _prepare_window_ok(
        context,
        target_line,
        prepare_strategy,
        str(launch_plan["prepare_state"]),
    )
    if not window_ok:
        return False, {"reason": "prepare_window_not_ready", **window_debug}
    activation_ok, activation_debug = _prepare_activation_ok(context, exact_result["solution"])
    if not activation_ok:
        return False, {"reason": "prepare_activation_not_ready", **activation_debug}
    return True, {"reason": "ok", **window_debug, **activation_debug}


def _prepare_should_activate_degraded(
    context: SceneContext,
    target_line: ReferenceLineView,
    memory: dict[str, Any],
    launch_plan: dict[str, Any],
    prepare_strategy: dict[str, Any],
    now: float,
) -> tuple[bool, dict[str, Any]]:
    """判断是否满足进入 ACTIVE_DEGRADED 的条件。"""
    # 逻辑：
    # 1. 主车过慢导致的退化可以直接进入 degraded。
    # 2. 自车能力受限导致的退化，优先留一段 BUILD 时间尝试 exact，超时后再退化。
    degraded_result = launch_plan["degraded_result"]
    if not degraded_result["ok"]:
        return False, {"reason": str(degraded_result["reason"])}

    window_ok, window_debug = _prepare_window_ok(
        context,
        target_line,
        prepare_strategy,
        str(launch_plan["prepare_state"]),
    )
    if not window_ok:
        return False, {"reason": "prepare_window_not_ready", **window_debug}

    activation_ok, activation_debug = _prepare_activation_ok(context, degraded_result["solution"])
    if not activation_ok:
        return False, {"reason": "prepare_activation_not_ready", **activation_debug}

    degrade_reason = str(launch_plan["degrade_reason"])
    elapsed = max(0.0, float(now) - float(memory.get("prepare_start_time_s", now)))
    interval = launch_plan.get("interval")
    upper_reason = "" if interval is None else str(interval["upper_bound_reason"])

    if degrade_reason == "main_speed_too_low_for_exact_thw":
        return True, {"reason": "degrade_low_main_ready", **window_debug, **activation_debug}
    if degrade_reason == "ego_cap_limited_for_exact_thw" and (
        elapsed >= float(CFG["prepare_build_timeout_s"]) or upper_reason == "road_end_limit"
    ):
        return True, {"reason": "degrade_cap_ready", **window_debug, **activation_debug}
    return False, {
        "reason": "degrade_building",
        "prepare_elapsed_s": float(elapsed),
        **window_debug,
        **activation_debug,
    }


def _prepare_should_abort_as_globally_infeasible(
    memory: dict[str, Any],
    launch_plan: dict[str, Any],
    now: float,
) -> bool:
    """判断 PREPARE 是否已经可以认定为全局不可执行。"""
    # 逻辑：
    # 1. 只有在明显的全局阻塞条件持续存在一段时间后，才退出 PREPARE。
    if str(launch_plan["prepare_state"]) != "globally_infeasible":
        return False
    elapsed = max(0.0, float(now) - float(memory.get("prepare_start_time_s", now)))
    return bool(elapsed >= float(CFG["prepare_build_timeout_s"]))


def _prepare_window_ok(
    context: SceneContext,
    target_line: ReferenceLineView,
    prepare_strategy: dict[str, Any],
    prepare_state: str,
) -> tuple[bool, dict[str, Any]]:
    """判断 PREPARE 是否已经收敛到可发射窗口。"""
    # 逻辑：
    # 1. 前插只在目标车道投影上已略领先，且 exact/degraded 解可执行时启动。
    # 2. PREPARE 的速度组织不会再改变这个启动判定的语义。
    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        return False, {"reason": "no_main_vehicle"}
    not_already_on_target = str(ego.current_ref_line.lane_id) != str(target_line.lane_id)
    _, ego_s_on_target, _ = project_point_to_path(
        np.array([ego.x, ego.y], dtype=float),
        target_line.line,
    )
    _, main_s_on_target, _ = project_point_to_path(
        np.array([main.x, main.y], dtype=float),
        target_line.line,
    )
    delta_s_target = float(ego_s_on_target) - float(main_s_on_target)
    launch_delta_s_min = float(CFG["launch_delta_s_min_m"])
    launch_geom_ok = delta_s_target >= launch_delta_s_min
    launch_state_ready = str(prepare_state) in {"prepare_exact_ready", "prepare_degraded_ready"}
    window_ok = bool(
        not_already_on_target
        and launch_geom_ok
        and launch_state_ready
    )
    return window_ok, {
        "prepare_delta_s_target": float(prepare_strategy["delta_s_target"]),
        "prepare_delta_s_goal": float(prepare_strategy["delta_s_goal"]),
        "prepare_delta_s_error": float(prepare_strategy["delta_s_error"]),
        "ego_s_on_target": float(ego_s_on_target),
        "main_s_on_target": float(main_s_on_target),
        "delta_s_target": float(delta_s_target),
        "launch_delta_s_min": float(launch_delta_s_min),
        "launch_geom_ok": bool(launch_geom_ok),
        "launch_state_ready": bool(launch_state_ready),
        "not_already_on_target": bool(not_already_on_target),
    }


def _prepare_activation_ok(
    context: SceneContext,
    candidate_solution: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """检查启动 cut-in 时的减速度余量是否可接受。"""
    # 逻辑：
    # 1. PREPARE 允许提前建立速度优势，但切入启动时仍要满足自车自身纵向能力约束。
    ego_v = float(context.ego_vehicle.v)
    target_v = float(candidate_solution.get("v", 0.0))
    plan_t = max(float(CFG["lane_change_time_s"]), 1e-3)
    a_req = (target_v - ego_v) / plan_t
    limits = _read_vehicle_limits(context)
    min_acc_allowed = limits.min_acc + float(CFG["prepare_acc_margin_mps2"])
    acc_margin_ok = a_req >= min_acc_allowed
    return bool(acc_margin_ok), {
        "prepare_start_a_req": float(a_req),
        "prepare_start_min_acc_allowed": float(min_acc_allowed),
        "prepare_acc_margin_ok": bool(acc_margin_ok),
    }


def _should_enter_dead_zone(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView | None,
    now: float,
) -> bool:
    """判断是否进入停止重解算的死区。"""
    # 逻辑：
    # 1. 靠近执行末端、横向进度过半或主车预测速度过低时，停止重解算并冻结目标。
    remaining_t = _remaining_lane_change_time(memory, now)
    progress_ratio = _lane_change_progress_ratio(context, memory, target_line)
    memory["dead_zone_trigger_progress"] = float(progress_ratio)
    memory["dead_zone_trigger_pred_main_v"] = -1.0

    if remaining_t <= float(CFG["dead_zone_time_s"]):
        memory["dead_zone_trigger_reason"] = "time"
        return True

    if progress_ratio >= float(CFG["dead_zone_lateral_progress_ratio"]):
        memory["dead_zone_trigger_reason"] = "lateral_progress"
        return True

    if (
        target_line is not None
        and context.main_vehicle is not None
        and memory.get("last_dynamic_solution") is not None
        and progress_ratio >= float(CFG["dead_zone_pred_main_v_min_progress"])
    ):
        pred_t = max(float(CFG["dynamic_solution_min_t_s"]), remaining_t)
        pred_main_v = float(_predict_main_vehicle_at_t(context, target_line, pred_t)["v"])
        memory["dead_zone_trigger_pred_main_v"] = float(pred_main_v)
        if pred_main_v <= float(CFG["dead_zone_pred_main_v_thresh"]):
            memory["dead_zone_trigger_reason"] = "pred_main_low_speed"
            return True

    memory["dead_zone_trigger_reason"] = ""
    return False


def _is_execute_completed(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView | None,
) -> bool:
    """判断换道是否已经稳定完成。"""
    # 逻辑：
    # 1. 优先按目标车道 ID 判断，其次退化到横向偏差阈值。
    current_lane_id = str(context.ego_vehicle.current_ref_line.lane_id)
    target_lane_id = str(memory.get("locked_target_lane_id", ""))
    target_lane_sequence = set(_normalize_lane_sequence(memory.get("locked_target_lane_sequence", [])))
    if (target_lane_id and current_lane_id == target_lane_id) or (
        target_lane_sequence and current_lane_id in target_lane_sequence
    ):
        memory["execute_complete_hits"] += 1
        return int(memory["execute_complete_hits"]) >= int(CFG["execute_complete_frames"])

    if target_line is None:
        memory["execute_complete_hits"] = 0
        return False

    _, _, ego_d_on_target = project_point_to_path(
        np.array([context.ego_vehicle.x, context.ego_vehicle.y], dtype=float),
        target_line.line,
    )
    if abs(float(ego_d_on_target)) <= float(CFG["enter_target_d_thresh_m"]):
        memory["execute_complete_hits"] += 1
    else:
        memory["execute_complete_hits"] = 0
    return int(memory["execute_complete_hits"]) >= int(CFG["execute_complete_frames"])


def _measure_completion_thw(
    context: SceneContext,
) -> tuple[float | None, str]:
    """按现有 cut-in 求解口径计算换道完成时的实际 THW。"""

    main = context.main_vehicle
    if main is None:
        return None, "main_vehicle_missing"
    if float(main.v) <= 1e-3:
        return None, "main_speed_too_low"

    ego = context.ego_vehicle
    ego_xy = np.array([ego.x, ego.y], dtype=float)
    main_xy = np.array([main.x, main.y], dtype=float)

    # 优先使用后车 VUT 的当前参考线。切入目标线往往已从 GVT
    # 当前位置向前截断，直接在它上投影会把后方 VUT 夹到起点。
    main_line = main.current_ref_line
    _, ego_s_on_main, ego_d_on_main = project_point_to_path(ego_xy, main_line.line)
    _, main_s_on_main, _ = project_point_to_path(main_xy, main_line.line)
    projected_center_distance = float(ego_s_on_main) - float(main_s_on_main)
    if (
        projected_center_distance > 1e-3
        and abs(float(ego_d_on_main)) <= float(CFG["max_ego_to_target_line_d_m"])
    ):
        center_distance = float(projected_center_distance)
        metric_reason = "main_reference_line"
    else:
        # 参考线被截断或匹配不稳定时，退回 VUT 车体纵向投影。
        # 换道完成已要求两车同向，因此该局部距离不依赖路段 s 原点。
        main_forward = np.array(
            [np.cos(float(main.heading)), np.sin(float(main.heading))],
            dtype=float,
        )
        center_distance = float(np.dot(ego_xy - main_xy, main_forward))
        metric_reason = "main_heading_fallback"

    limits = _read_vehicle_limits(context)
    net_gap = center_distance - 0.5 * (float(limits.ego_length) + float(limits.main_length))
    # 与 _project_requested_thw 保持同一口径：用后车 VUT 速度作为分母，
    # 分子为净间距加 GVT 车长。两车等长时等价于车身中心纵向距离。
    achieved_thw = (float(net_gap) + float(limits.ego_length)) / float(main.v)
    return float(achieved_thw), metric_reason


# ==================== Target Helpers / 目标车道与参考线 ====================

def _choose_target_line(context: SceneContext) -> ReferenceLineView | None:
    """从候选参考线中选择一条相邻目标车道。"""
    # 逻辑：
    # 1. 只选一阶相邻车道，且要求主车与该车道有明确几何对应关系。
    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        return None

    cutin_side = str(CFG["cutin_side"]).strip().lower()
    best_line = None
    best_score = float("inf")
    for line in ego.candidate_ref_lines:
        if abs(int(line.lateral_step)) != 1:
            continue
        if cutin_side == "left" and int(line.lateral_step) >= 0:
            continue
        if cutin_side == "right" and int(line.lateral_step) <= 0:
            continue

        _, main_s_on_line, main_d_on_line = project_point_to_path(np.array([main.x, main.y], dtype=float), line.line)
        if abs(float(main_d_on_line)) > float(CFG["target_lane_match_d_thresh"]):
            continue

        _, ego_s_on_line, ego_d_on_line = project_point_to_path(np.array([ego.x, ego.y], dtype=float), line.line)
        if abs(float(ego_d_on_line)) > float(CFG["max_ego_to_target_line_d_m"]):
            continue

        score = abs(float(main_d_on_line)) + 0.05 * abs(float(main_s_on_line) - float(ego_s_on_line))
        if score < best_score:
            best_score = score
            best_line = line
    return best_line


def _resolve_prepare_target_line(context: SceneContext, memory: dict[str, Any]) -> ReferenceLineView | None:
    """恢复 PREPARE 锁定的目标车道。"""
    # 逻辑：
    # 1. 优先按 lane_sequence 恢复，再退化到 lane_id 和 line_index。
    line = _line_by_lane_sequence(
        context.ego_vehicle.candidate_ref_lines,
        memory.get("prepare_target_lane_sequence", []),
    )
    if line is not None:
        return line
    line = _line_by_lane_id(context.ego_vehicle.candidate_ref_lines, memory.get("prepare_target_lane_id", ""))
    if line is not None:
        return line
    line = _line_by_index(context.ego_vehicle.candidate_ref_lines, memory.get("prepare_line_index"))
    if line is not None:
        return line
    return _choose_target_line(context)


def _resolve_target_line(
    context: SceneContext,
    memory: dict[str, Any],
) -> ReferenceLineView | None:
    """恢复 ACTIVE/FROZEN/STABILIZE 锁定的目标车道。"""
    # 逻辑：
    # 1. ACTIVE 之后不再重新选目标车道，只恢复最初锁定的那一条。
    line = _line_by_lane_sequence(
        context.ego_vehicle.candidate_ref_lines,
        memory.get("locked_target_lane_sequence", []),
    )
    if line is not None:
        return line
    line = _line_by_lane_id(context.ego_vehicle.candidate_ref_lines, memory.get("locked_target_lane_id", ""))
    if line is not None:
        return line
    return _line_by_index(context.ego_vehicle.candidate_ref_lines, memory.get("locked_line_index"))


def _line_by_index(candidate_lines: list[ReferenceLineView], line_index: int | None) -> ReferenceLineView | None:
    """按 line_index 恢复目标参考线。"""
    # 逻辑：
    # 1. 作为目标线恢复的最后一道兜底。
    if line_index is None:
        return None
    for line in candidate_lines:
        if int(line.line_index) == int(line_index):
            return line
    return None


def _line_by_lane_id(candidate_lines: list[ReferenceLineView], lane_id: str) -> ReferenceLineView | None:
    """按 lane_id 恢复目标参考线。"""
    # 逻辑：
    # 1. lane_id 比 line_index 更稳定，但不如 lane_sequence 鲁棒。
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
    """按锁定的 lane_sequence 恢复目标参考线。"""
    # 逻辑：
    # 1. 优先用 lane_sequence 匹配完整车道链，适合地图对象局部变化的情况。
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
    """把 lane_sequence 统一转成字符串列表。"""
    # 逻辑：
    # 1. 恢复目标线时统一按字符串匹配，避免类型差异。
    if lane_sequence is None:
        return []
    return [str(lane_id) for lane_id in lane_sequence]


# ==================== Prepare Logic / PREPARE 纵向组织与发射计划 ====================

def _evaluate_launch_plan(
    context: SceneContext,
    target_line: ReferenceLineView,
    remaining_t: float,
) -> dict[str, Any]:
    """评估当前时刻能否进入 exact/degraded cut-in。"""
    # 逻辑：
    # 1. 先构造统一求解上下文，再计算当前可行 THW 区间。
    # 2. exact 落在区间内就尝试精确解；落在区间外则按下界/上界构造退化解。
    solve_ctx_result = _build_solve_context(context, target_line, remaining_t)
    if not solve_ctx_result["ok"]:
        failure = _build_solver_failure(
            scene_status=str(solve_ctx_result["scene_status"]),
            reason=str(solve_ctx_result["reason"]),
            **solve_ctx_result["debug"],
        )
        return {
            "solve_ctx": None,
            "interval": None,
            "exact_result": failure,
            "degraded_result": failure,
            "degrade_reason": "",
            "prepare_state": "globally_infeasible",
            "prepare_reason": str(solve_ctx_result["reason"]),
            "summary_reason": str(solve_ctx_result["reason"]),
            "debug": solve_ctx_result["debug"],
        }

    solve_ctx = solve_ctx_result["ctx"]
    interval = _compute_thw_interval(solve_ctx)
    exact_result = _solve_requested_thw(
        solve_ctx,
        requested_thw=float(CFG["target_thw_s"]),
        reason="cutin_active_exact",
    )

    degraded_result = _build_solver_failure("no_solution", "degraded_not_needed")
    degrade_reason = ""
    if interval["ok"]:
        if float(CFG["target_thw_s"]) < float(interval["thw_min_feasible"]):
            degrade_reason = "main_speed_too_low_for_exact_thw"
            degraded_result = _solve_requested_thw(
                solve_ctx,
                requested_thw=float(interval["thw_min_feasible"]),
                reason="cutin_active_degraded",
            )
        elif float(CFG["target_thw_s"]) > float(interval["thw_max_feasible"]):
            degrade_reason = "ego_cap_limited_for_exact_thw"
            degraded_result = _solve_requested_thw(
                solve_ctx,
                requested_thw=float(interval["thw_max_feasible"]),
                reason="cutin_active_degraded",
            )

    if exact_result["ok"]:
        prepare_state = "prepare_exact_ready"
        prepare_reason = "prepare_exact_ready"
        summary_reason = "exact_feasible"
    elif degraded_result["ok"] and degrade_reason == "main_speed_too_low_for_exact_thw":
        prepare_state = "prepare_degraded_ready"
        prepare_reason = "prepare_degraded_low_main"
        summary_reason = degrade_reason
    elif degraded_result["ok"] and interval["upper_bound_reason"] == "road_end_limit":
        prepare_state = "prepare_degraded_ready"
        prepare_reason = "prepare_degraded_road_limit"
        summary_reason = degrade_reason
    elif degraded_result["ok"]:
        prepare_state = "prepare_build_exact"
        prepare_reason = "prepare_build_exact"
        summary_reason = degrade_reason
    elif interval["upper_bound_reason"] == "road_end_limit":
        prepare_state = "globally_infeasible"
        prepare_reason = "prepare_globally_infeasible"
        summary_reason = "road_end_limit"
    else:
        prepare_state = "prepare_build_exact"
        prepare_reason = "prepare_build_exact"
        summary_reason = str(exact_result["reason"])

    debug = copy.deepcopy(solve_ctx["debug"])
    debug.update(interval["debug"])
    debug["exact_reason"] = str(exact_result["reason"])
    debug["degrade_reason"] = str(degrade_reason)
    if exact_result["projection"] is not None:
        debug["exact_projection"] = exact_result["projection"]
    if degraded_result["projection"] is not None:
        debug["degraded_projection"] = degraded_result["projection"]
    debug["delta_s_target_now"] = float(solve_ctx["ego_s_on_target"]) - float(solve_ctx["main_prediction"]["s_on_target_line"])

    return {
        "solve_ctx": solve_ctx,
        "interval": interval,
        "exact_result": exact_result,
        "degraded_result": degraded_result,
        "degrade_reason": str(degrade_reason),
        "prepare_state": str(prepare_state),
        "prepare_reason": str(prepare_reason),
        "summary_reason": str(summary_reason),
        "debug": debug,
    }


def _build_prepare_strategy(
    context: SceneContext,
    target_line: ReferenceLineView,
) -> dict[str, Any]:
    """根据当前 launch plan 构造 PREPARE 的纵向组织策略。"""
    # 逻辑：
    # 1. PREPARE 始终以主车速度为基准，持续建立前插所需的小幅领先量。
    # 2. exact THW 仅由 ACTIVE 求解；它不能把 PREPARE 拉回主车后方。
    main = context.main_vehicle
    limits = _read_vehicle_limits(context)
    if main is None:
        target_v = _clamp(float(CFG["cruise_target_v"]), 0.0, limits.max_v)
        return {
            "delta_s_target": 0.0,
            "delta_s_goal": float(CFG["prepare_merge_lead_m"]),
            "delta_s_error": float(CFG["prepare_merge_lead_m"]),
            "target_v": float(target_v),
            "debug": {},
        }

    _, ego_s_on_target, _ = project_point_to_path(
        np.array([context.ego_vehicle.x, context.ego_vehicle.y], dtype=float),
        target_line.line,
    )
    _, main_s_on_target, _ = project_point_to_path(
        np.array([main.x, main.y], dtype=float),
        target_line.line,
    )
    delta_s_target = float(ego_s_on_target) - float(main_s_on_target)
    delta_s_goal = float(CFG["prepare_merge_lead_m"])
    delta_s_error = float(delta_s_goal) - float(delta_s_target)
    raw_speed_delta = float(delta_s_error) / max(0.1, float(CFG["prepare_convergence_time_s"]))
    speed_delta = _clamp(
        raw_speed_delta,
        -float(CFG["prepare_max_speed_lag_mps"]),
        float(CFG["prepare_max_speed_adv_mps"]),
    )
    target_v = _clamp(float(main.v) + float(speed_delta), 0.0, limits.max_v)

    return {
        "delta_s_target": float(delta_s_target),
        "delta_s_goal": float(delta_s_goal),
        "delta_s_error": float(delta_s_error),
        "target_v": float(target_v),
        "debug": {
            "prepare_delta_s_target": float(delta_s_target),
            "prepare_delta_s_goal": float(delta_s_goal),
            "prepare_delta_s_error": float(delta_s_error),
            "prepare_raw_speed_delta": float(raw_speed_delta),
            "prepare_speed_delta": float(speed_delta),
            "prepare_target_v": float(target_v),
        },
    }


# ==================== Active Logic / ACTIVE 执行态逻辑 ====================

def _select_active_solution(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView,
    remaining_t: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """在 ACTIVE 中选择本帧应执行的 exact/degraded 解。"""
    # 逻辑：
    # 1. exact 模式下优先尝试精确解，失败时允许切到 degraded。
    # 2. degraded 模式下优先维持已锁定的退化 THW，必要时再夹紧到当前可行边界。
    launch_plan = _evaluate_launch_plan(context, target_line, remaining_t)
    solve_ctx = launch_plan["solve_ctx"]
    if solve_ctx is None:
        return None, {
            "scene_status": "globally_infeasible",
            "reason": str(launch_plan["summary_reason"]),
            "debug": launch_plan["debug"],
        }

    if str(memory.get("execution_kind", "")) == "exact":
        if launch_plan["exact_result"]["ok"]:
            output = _annotate_output(
                copy.deepcopy(launch_plan["exact_result"]["solution"]),
                scene_status="exact_executable",
                scene_detail="active_exact",
                scene_debug=launch_plan["debug"],
            )
            return output, {"scene_status": "exact_executable", "reason": "active_exact", "debug": launch_plan["debug"]}

        if launch_plan["degraded_result"]["ok"]:
            memory["execution_kind"] = "degraded"
            memory["execution_requested_thw_s"] = float(launch_plan["degraded_result"]["requested_thw"])
            memory["execution_degrade_reason"] = str(launch_plan["degrade_reason"])
            _print_degrade_event(
                prefix="active",
                degrade_reason=str(launch_plan["degrade_reason"]),
                target_thw=float(CFG["target_thw_s"]),
                degraded_thw=float(launch_plan["degraded_result"]["requested_thw"]),
                debug=launch_plan["debug"],
            )
            output = _annotate_output(
                copy.deepcopy(launch_plan["degraded_result"]["solution"]),
                scene_status="degraded_executable",
                scene_detail=str(launch_plan["degrade_reason"]),
                scene_debug=launch_plan["debug"],
            )
            return output, {"scene_status": "degraded_executable", "reason": str(launch_plan["degrade_reason"]), "debug": launch_plan["debug"]}

        return None, {
            "scene_status": str(launch_plan["exact_result"]["scene_status"]),
            "reason": str(launch_plan["exact_result"]["reason"]),
            "debug": launch_plan["debug"],
        }

    requested_thw = float(memory.get("execution_requested_thw_s", CFG["target_thw_s"]))
    interval = launch_plan["interval"]
    if interval is not None and interval["ok"]:
        if str(memory.get("execution_degrade_reason", "")) == "main_speed_too_low_for_exact_thw":
            requested_thw = max(requested_thw, float(interval["thw_min_feasible"]))
        else:
            requested_thw = min(requested_thw, float(interval["thw_max_feasible"]))
        memory["execution_requested_thw_s"] = float(requested_thw)

    degraded_result = _solve_requested_thw(
        solve_ctx,
        requested_thw=float(requested_thw),
        reason="cutin_active_degraded",
    )
    if degraded_result["ok"]:
        debug = copy.deepcopy(launch_plan["debug"])
        debug["active_requested_thw"] = float(requested_thw)
        output = _annotate_output(
            copy.deepcopy(degraded_result["solution"]),
            scene_status="degraded_executable",
            scene_detail=str(memory.get("execution_degrade_reason", "degraded_active")),
            scene_debug=debug,
        )
        return output, {
            "scene_status": "degraded_executable",
            "reason": str(memory.get("execution_degrade_reason", "degraded_active")),
            "debug": debug,
        }

    if launch_plan["degraded_result"]["ok"]:
        memory["execution_requested_thw_s"] = float(launch_plan["degraded_result"]["requested_thw"])
        debug = copy.deepcopy(launch_plan["debug"])
        debug["active_requested_thw"] = float(memory["execution_requested_thw_s"])
        output = _annotate_output(
            copy.deepcopy(launch_plan["degraded_result"]["solution"]),
            scene_status="degraded_executable",
            scene_detail=str(memory.get("execution_degrade_reason", "degraded_active")),
            scene_debug=debug,
        )
        return output, {
            "scene_status": "degraded_executable",
            "reason": str(memory.get("execution_degrade_reason", "degraded_active")),
            "debug": debug,
        }

    return None, {
        "scene_status": str(degraded_result["scene_status"]),
        "reason": str(degraded_result["reason"]),
        "debug": launch_plan["debug"],
    }


def _handle_active_infeasible(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView,
    now: float,
    failure_info: dict[str, Any],
) -> dict[str, Any]:
    """处理 ACTIVE 阶段的短时无解与持续无解。"""
    # 逻辑：
    # 1. 先短时保持上一帧有效解，给执行态留一点缓冲。
    # 2. 若持续无解且还没切到一半，就回退并输出失败原因。
    memory["active_infeasible_count"] = int(memory.get("active_infeasible_count", 0)) + 1
    hold_solution = memory.get("last_dynamic_solution")
    if hold_solution is None:
        _enter_cooldown(memory, now)
        return _build_keep_lane_output(
            context,
            reason="active_solution_lost",
            scene_status=str(failure_info["scene_status"]),
            scene_detail=str(failure_info["reason"]),
            scene_debug=failure_info["debug"],
        )

    remaining_t = _remaining_lane_change_time(memory, now)
    progress_ratio = _lane_change_progress_ratio(context, memory, target_line)
    infeasible_count = int(memory["active_infeasible_count"])
    if infeasible_count < int(CFG["active_persistent_infeasible_max_frames"]):
        hold_reason = "cutin_transient_hold" if infeasible_count <= int(CFG["active_transient_hold_max_frames"]) else "cutin_persistent_hold"
        return _build_projected_solution_output(
            target_line=target_line,
            solution=hold_solution,
            remaining_t=remaining_t,
            reason=hold_reason,
            scene_status=str(failure_info["scene_status"]),
            scene_detail=str(failure_info["reason"]),
            scene_debug=failure_info["debug"],
        )

    if progress_ratio >= float(CFG["persistent_infeasible_progress_ratio_thresh"]):
        return _build_projected_solution_output(
            target_line=target_line,
            solution=hold_solution,
            remaining_t=remaining_t,
            reason="cutin_persistent_infeasible",
            scene_status=str(failure_info["scene_status"]),
            scene_detail=str(failure_info["reason"]),
            scene_debug=failure_info["debug"],
        )

    _enter_cooldown(memory, now)
    return _build_keep_lane_output(
        context,
        reason="active_persistent_failure",
        scene_status=str(failure_info["scene_status"]),
        scene_detail=str(failure_info["reason"]),
        scene_debug=failure_info["debug"],
    )


def _remaining_lane_change_time(memory: dict[str, Any], now: float) -> float:
    """计算距离当前执行结束还剩多少时间。"""
    # 逻辑：
    # 1. ACTIVE/FROZEN 都依赖这个剩余时间来决定是否重解和如何投影。
    return max(0.0, float(memory.get("lane_change_end_time_s", -1.0)) - float(now))


def _lane_change_progress_ratio(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView | None,
) -> float:
    """估计当前横向换道进度。"""
    # 逻辑：
    # 1. 用切入开始时的横向偏差做归一化，估算当前横向完成比例。
    if target_line is None:
        return 0.0
    _, _, ego_d_on_target = project_point_to_path(
        np.array([context.ego_vehicle.x, context.ego_vehicle.y], dtype=float),
        target_line.line,
    )
    start_abs_d = max(float(memory.get("active_start_abs_d_to_target", 0.0)), 1e-3)
    progress = 1.0 - abs(float(ego_d_on_target)) / start_abs_d
    return _clamp(progress, 0.0, 1.0)


# ==================== THW Solver / THW 数学解算 ====================

def _build_solve_context(
    context: SceneContext,
    target_line: ReferenceLineView,
    remaining_t: float,
) -> dict[str, Any]:
    """构造当前帧求解 THW 所需的统一上下文。"""
    # 逻辑：
    # 1. 把主车预测、目标线几何、自车投影和平台约束统一打包。
    # 2. 这样 exact / degraded / interval 计算都用同一份基础数据。
    remaining_t = float(remaining_t)
    if remaining_t < float(CFG["dynamic_solution_min_t_s"]):
        return {
            "ok": False,
            "scene_status": "no_solution",
            "reason": "remaining_time_too_short",
            "debug": {"remaining_t": float(remaining_t)},
        }

    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        return {
            "ok": False,
            "scene_status": "waiting_valid_main",
            "reason": "no_main_vehicle",
            "debug": {},
        }

    _, ego_s_on_target, ego_d_on_target = project_point_to_path(
        np.array([ego.x, ego.y], dtype=float),
        target_line.line,
    )
    if abs(float(ego_d_on_target)) > float(CFG["max_ego_to_target_line_d_m"]):
        return {
            "ok": False,
            "scene_status": "infeasible_geometry",
            "reason": "ego_too_far_from_target_line",
            "debug": {"ego_d_on_target": float(ego_d_on_target)},
        }

    main_pred = _predict_main_vehicle_at_t(context, target_line, remaining_t)
    target_line_end_s = float(target_line.cumulative_s[-1]) - 0.5
    limits = _read_vehicle_limits(context)
    ego_length = limits.ego_length
    main_length = limits.main_length
    A = float(main_pred["s_on_target_line"]) + 0.5 * (float(main_length) - float(ego_length)) - float(ego_s_on_target)

    return {
        "ok": True,
        "ctx": {
            "remaining_t": float(remaining_t),
            "ego_v": float(ego.v),
            "ego_s_on_target": float(ego_s_on_target),
            "ego_d_on_target": float(ego_d_on_target),
            "ego_length": float(ego_length),
            "main_length": float(main_length),
            "main_prediction": main_pred,
            "main_v_pred": float(main_pred["v"]),
            "A": float(A),
            "target_line_index": int(target_line.line_index),
            "target_line_end_s": float(target_line_end_s),
            "target_line": target_line,
            "min_physical_gap": float(CFG["min_physical_gap_m"]),
            "physical_gap_tol": float(CFG["physical_gap_tol_m"]),
            "min_acc": float(limits.min_acc),
            "max_acc": float(limits.max_acc),
            "max_v": float(limits.max_v),
            "debug": {
                "remaining_t": float(remaining_t),
                "ego_v": float(ego.v),
                "ego_s_on_target": float(ego_s_on_target),
                "ego_d_on_target": float(ego_d_on_target),
                "main_v_pred": float(main_pred["v"]),
                "main_s_pred": float(main_pred["s_on_target_line"]),
                "A": float(A),
                "target_line_end_s": float(target_line_end_s),
                "max_v": float(limits.max_v),
                "min_acc": float(limits.min_acc),
                "max_acc": float(limits.max_acc),
            },
        },
    }


def _compute_thw_interval(solve_ctx: dict[str, Any]) -> dict[str, Any]:
    """计算当前 solve context 的可行 THW 区间。"""
    # 逻辑：
    # 1. 把所有约束都统一投到 THW 空间，得到当前 exact/degraded 的基础判据。
    interval = _evaluate_thw_bounds(solve_ctx)
    interval["debug"].update(copy.deepcopy(solve_ctx["debug"]))
    return interval


def _evaluate_thw_bounds(solve_ctx: dict[str, Any]) -> dict[str, Any]:
    """根据当前 solve context 计算可行 THW 区间。"""
    # 逻辑：
    # 1. 下界来自：物理净距、最小前进距离、目标速度非负、最小减速度。
    # 2. 上界来自：最大速度、最大加速度、目标车道末端缓冲区。
    v_main = float(solve_ctx["main_v_pred"])
    if v_main <= 1e-6:
        return {
            "ok": False,
            "thw_min_feasible": float("inf"),
            "thw_max_feasible": float("-inf"),
            "lower_bound_reason": "main_speed_zero",
            "upper_bound_reason": "main_speed_zero",
            "debug": {
                "thw_lower_bounds": {},
                "thw_upper_bounds": {},
            },
        }

    A = float(solve_ctx["A"])
    T = float(solve_ctx["remaining_t"])
    ego_v = float(solve_ctx["ego_v"])
    line_end = float(solve_ctx["target_line_end_s"])
    s_ego = float(solve_ctx["ego_s_on_target"])
    l_ego = float(solve_ctx["ego_length"])
    min_gap = float(solve_ctx["min_physical_gap"])

    lower_bounds = {
        "zero_thw": 0.0,
        "physical_gap": (min_gap + l_ego) / v_main,
        "min_progress": (float(CFG["min_target_progress_m"]) - A) / v_main,
        "non_negative_target_v": (0.5 * T * ego_v - A) / v_main,
        "min_acc_limit": ((ego_v * T) + 0.5 * (T ** 2) * float(solve_ctx["min_acc"]) - A) / v_main,
    }
    upper_bounds = {
        "max_v_limit": (0.5 * T * (ego_v + float(solve_ctx["max_v"])) - A) / v_main,
        "max_acc_limit": ((ego_v * T) + 0.5 * (T ** 2) * float(solve_ctx["max_acc"]) - A) / v_main,
        "road_end_limit": (line_end - float(CFG["post_merge_buffer_m"]) - s_ego - A) / v_main,
    }

    lower_bound_reason, thw_min = max(lower_bounds.items(), key=lambda item: item[1])
    upper_bound_reason, thw_max = min(upper_bounds.items(), key=lambda item: item[1])
    return {
        "ok": bool(float(thw_min) <= float(thw_max)),
        "thw_min_feasible": float(thw_min),
        "thw_max_feasible": float(thw_max),
        "lower_bound_reason": str(lower_bound_reason),
        "upper_bound_reason": str(upper_bound_reason),
        "debug": {
            "thw_lower_bounds": {key: float(value) for key, value in lower_bounds.items()},
            "thw_upper_bounds": {key: float(value) for key, value in upper_bounds.items()},
            "thw_min_feasible": float(thw_min),
            "thw_max_feasible": float(thw_max),
            "thw_lower_bound_reason": str(lower_bound_reason),
            "thw_upper_bound_reason": str(upper_bound_reason),
        },
    }


def _solve_requested_thw(
    solve_ctx: dict[str, Any],
    *,
    requested_thw: float,
    reason: str,
) -> dict[str, Any]:
    """按给定 THW 直接求解一帧目标输出。"""
    # 逻辑：
    # 1. exact 和 degraded 最终都统一落到这里求目标点和目标速度。
    # 2. 返回统一结构，便于状态机直接拿来判断 exact/degraded 是否可执行。
    projection = _project_requested_thw(solve_ctx, float(requested_thw))
    if float(projection["target_gap"]) < float(solve_ctx["min_physical_gap"]):
        return _build_solver_failure(
            "metric_invalid_low_main_speed",
            "target_thw_below_physical_limit",
            projection=projection,
            requested_thw=float(requested_thw),
        )

    if float(projection["ds_needed"]) < float(CFG["min_target_progress_m"]):
        return _build_solver_failure(
            "no_solution",
            "target_progress_too_small",
            projection=projection,
            requested_thw=float(requested_thw),
        )

    if float(projection["target_s"]) > float(solve_ctx["target_line_end_s"]) - float(CFG["post_merge_buffer_m"]):
        return _build_solver_failure(
            "infeasible_geometry",
            "not_enough_post_merge_distance",
            projection=projection,
            requested_thw=float(requested_thw),
        )

    if float(projection["target_v"]) < 0.0 or float(projection["target_v"]) > float(solve_ctx["max_v"]):
        return _build_solver_failure(
            "infeasible_platform_limit",
            "target_speed_out_of_platform_limit",
            projection=projection,
            requested_thw=float(requested_thw),
        )

    if float(projection["a_req"]) < float(solve_ctx["min_acc"]) or float(projection["a_req"]) > float(solve_ctx["max_acc"]):
        return _build_solver_failure(
            "infeasible_platform_limit",
            "required_acc_out_of_platform_limit",
            projection=projection,
            requested_thw=float(requested_thw),
        )

    if float(projection["achieved_gap"]) < float(solve_ctx["min_physical_gap"]) - float(solve_ctx["physical_gap_tol"]):
        return _build_solver_failure(
            "geom_overlap",
            "achieved_gap_below_physical_limit",
            projection=projection,
            requested_thw=float(requested_thw),
        )

    if abs(float(projection["achieved_thw"]) - float(requested_thw)) > float(CFG["thw_tol_s"]):
        return _build_solver_failure(
            "no_solution",
            "thw_not_match",
            projection=projection,
            requested_thw=float(requested_thw),
        )

    target_line: ReferenceLineView = solve_ctx["target_line"]
    target_xy, target_heading = get_path_point_at_s(target_line.line, float(projection["target_s"]), target_line.cumulative_s)
    solution = {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(projection["target_v"]),
        "a": float(projection["a_req"]),
        "theta": float(target_heading),
        "t": float(solve_ctx["remaining_t"]),
        "target_line_index": int(solve_ctx["target_line_index"]),
        "hold_t": 0.0,
        "reason": str(reason),
        "_meta": {
            "requested_target_thw": float(requested_thw),
            "target_s": float(projection["target_s"]),
            "ego_s_on_target": float(solve_ctx["ego_s_on_target"]),
            "ego_d_on_target": float(solve_ctx["ego_d_on_target"]),
            "vut_pred": copy.deepcopy(solve_ctx["main_prediction"]),
            "thw_base_speed": float(solve_ctx["main_v_pred"]),
            "target_gap": float(projection["target_gap"]),
            "achieved_gap": float(projection["achieved_gap"]),
            "achieved_thw": float(projection["achieved_thw"]),
            "thw_error": float(projection["achieved_thw"] - float(requested_thw)),
            "remaining_after_merge": float(solve_ctx["target_line_end_s"] - float(projection["target_s"])),
            "remaining_t": float(solve_ctx["remaining_t"]),
            "ds_needed": float(projection["ds_needed"]),
        },
    }
    return {
        "ok": True,
        "solution": solution,
        "scene_status": "solved",
        "reason": "solved",
        "debug": {
            "requested_target_thw": float(requested_thw),
            "achieved_thw": float(projection["achieved_thw"]),
            "thw_error": float(projection["achieved_thw"] - float(requested_thw)),
            "target_gap": float(projection["target_gap"]),
            "achieved_gap": float(projection["achieved_gap"]),
            "predicted_main_v": float(solve_ctx["main_v_pred"]),
            "remaining_t": float(solve_ctx["remaining_t"]),
            "target_speed": float(projection["target_v"]),
            "required_acc": float(projection["a_req"]),
        },
        "requested_thw": float(requested_thw),
        "projection": projection,
    }


def _project_requested_thw(solve_ctx: dict[str, Any], requested_thw: float) -> dict[str, Any]:
    """给定 THW 后，直接投影出目标 gap、目标点和所需纵向速度。"""
    # 逻辑：
    # 1. 这是 exact/degraded 公共的中间层，只做几何和纵向运动学投影，不做边界判定。
    target_gap = float(requested_thw) * float(solve_ctx["main_v_pred"]) - float(solve_ctx["ego_length"])
    target_s = float(solve_ctx["main_prediction"]["s_on_target_line"]) + 0.5 * (
        float(solve_ctx["main_length"]) + float(solve_ctx["ego_length"])
    ) + float(target_gap)
    ds_needed = float(target_s) - float(solve_ctx["ego_s_on_target"])
    target_v = (2.0 * float(ds_needed) / float(solve_ctx["remaining_t"])) - float(solve_ctx["ego_v"])
    a_req = (float(target_v) - float(solve_ctx["ego_v"])) / float(solve_ctx["remaining_t"])
    achieved_gap = float(target_gap)
    achieved_thw = (float(achieved_gap) + float(solve_ctx["ego_length"])) / max(float(solve_ctx["main_v_pred"]), 1e-6)
    return {
        "requested_thw": float(requested_thw),
        "target_gap": float(target_gap),
        "target_s": float(target_s),
        "ds_needed": float(ds_needed),
        "target_v": float(target_v),
        "a_req": float(a_req),
        "achieved_gap": float(achieved_gap),
        "achieved_thw": float(achieved_thw),
    }


def _predict_main_vehicle_at_t(
    context: SceneContext,
    target_line: ReferenceLineView,
    total_t: float,
) -> dict[str, Any]:
    """把主车状态预测到给定时刻，并投影到目标参考线。"""
    # 逻辑：
    # 1. THW 求解全部围绕主车的未来状态展开。
    main = context.main_vehicle
    if main is None:
        raise ValueError("main_vehicle is required for prediction")

    main_line = main.current_ref_line
    a_use = _clamp(
        float(main.a),
        float(CFG["prediction_min_acc"]),
        float(CFG["prediction_max_acc"]),
    )
    s_pred_on_main = float(main_line.current_s) + float(main.v) * float(total_t) + 0.5 * a_use * float(total_t) ** 2
    s_pred_on_main = _clamp(s_pred_on_main, float(main_line.current_s), float(main_line.cumulative_s[-1]) - 0.5)
    xy_pred, heading_pred = get_path_point_at_s(main_line.line, s_pred_on_main, main_line.cumulative_s)
    v_pred = max(0.0, float(main.v) + a_use * float(total_t))
    _, s_pred_on_target, d_pred_on_target = project_point_to_path(xy_pred, target_line.line)
    return {
        "x": float(xy_pred[0]),
        "y": float(xy_pred[1]),
        "heading": float(heading_pred),
        "v": float(v_pred),
        "a": float(a_use),
        "s_on_main_line": float(s_pred_on_main),
        "s_on_target_line": float(s_pred_on_target),
        "d_on_target_line": float(d_pred_on_target),
    }


# ==================== Utilities / 基础工具、调试与日志 ====================


def _completion_debug(memory: dict[str, Any]) -> dict[str, Any]:
    """构造 STABILIZE/COMPLETE 共用的闭锁完成指标。"""

    achieved = memory.get("completion_achieved_thw_s")
    error = memory.get("completion_thw_error_s")
    return {
        "completion_latched": True,
        "completion_time_s": float(memory.get("completion_time_s", -1.0)),
        "completion_lane_id": str(memory.get("completion_lane_id", "")),
        "completion_execution_kind": str(memory.get("completion_execution_kind", "")),
        "completion_requested_thw_s": float(memory.get("completion_requested_thw_s", 0.0)),
        "completion_achieved_thw_s": None if achieved is None else float(achieved),
        "completion_thw_error_s": None if error is None else float(error),
        "completion_thw_within_tolerance": bool(
            error is not None and abs(float(error)) <= float(CFG["thw_tol_s"])
        ),
        "completion_metric_reason": str(memory.get("completion_metric_reason", "")),
    }


def _annotate_output(
    output: dict[str, Any],
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """为场景输出补充状态标签和调试信息。"""
    # 逻辑：
    # 1. 轨迹目标和场景状态分开记录，便于上层区分 exact/degraded/infeasible。
    output["scene_status"] = str(scene_status)
    if scene_detail is not None:
        output["scene_detail"] = str(scene_detail)
    if scene_debug is not None:
        output["scene_debug"] = copy.deepcopy(scene_debug)
    return output


def _build_solver_failure(scene_status: str, reason: str, **debug: Any) -> dict[str, Any]:
    """构造统一的求解失败结果。"""
    # 逻辑：
    # 1. exact、degraded、ACTIVE 重解都复用同一种失败结构。
    return {
        "ok": False,
        "solution": None,
        "scene_status": str(scene_status),
        "reason": str(reason),
        "debug": debug,
        "requested_thw": None,
        "projection": debug.get("projection"),
    }


def _print_degrade_event(
    *,
    prefix: str,
    degrade_reason: str,
    target_thw: float,
    degraded_thw: float,
    debug: dict[str, Any],
) -> None:
    """打印退化原因和退化后的切入 THW。"""
    # 逻辑：
    # 1. 只在进入 degraded 模式时打印一次，便于离线回放时快速定位。
    print(
        "[cutin] degrade "
        f"phase={prefix} "
        f"reason={degrade_reason} "
        f"target_thw={target_thw:.3f} "
        f"degraded_thw={degraded_thw:.3f} "
        f"thw_min={float(debug.get('thw_min_feasible', float('nan'))):.3f} "
        f"thw_max={float(debug.get('thw_max_feasible', float('nan'))):.3f} "
        f"main_v={float(debug.get('main_v_pred', debug.get('predicted_main_v', float('nan')))):.3f}",
        flush=True,
    )


def _solution_thw_text(solution: dict[str, Any] | None) -> str:
    """把解中的 THW 信息格式化成日志文本。"""
    # 逻辑：
    # 1. 日志统一输出 requested / achieved / state，方便排查 exact 和 degraded。
    if solution is None:
        return "thw=nan requested=nan"
    meta = solution.get("_meta", {})
    achieved = meta.get("achieved_thw")
    requested = meta.get("requested_target_thw", CFG["target_thw_s"])
    state = solution.get("scene_status", "")
    if achieved is None:
        return f"thw=nan requested={float(requested):.3f} state={state}"
    return f"thw={float(achieved):.3f} requested={float(requested):.3f} state={state}"


def _clamp(value: float, lower: float, upper: float) -> float:
    """把数值裁到指定上下界。"""
    # 逻辑：
    # 1. 所有速度、THW 和位置目标的边界都统一走这里。
    return float(min(max(value, lower), upper))


def _heading_diff_deg(h1: float, h2: float) -> float:
    """计算两个航向角之间的最小夹角，单位为度。"""
    # 逻辑：
    # 1. 只保留同向主车，避免误把横穿或掉头目标纳入场景。
    return float(np.degrees(abs(normalize_angle(float(h1) - float(h2)))))


def _read_vehicle_limits(context: SceneContext) -> VehicleLimits:
    """统一读取场景算法使用的车辆尺寸和平台约束。"""
    # 逻辑：
    # 1. 把 runtime_frame.config 的字段路径隔离在这里，核心求解只消费 VehicleLimits。
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
