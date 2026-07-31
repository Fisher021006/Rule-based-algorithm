"""伴行场景算法。

场景语义：
- 受控车辆是伴行背景车 GVT，对应 ``context.ego_vehicle``。
- 被伴行主车是 VUT，对应 ``context.main_vehicle``。
- GVT 由场景配置预先放在正确车道，只沿车道中心线做纵向速度控制。
- 八方位由车道关系和可配置 ``rel_s`` 窗口共同定义，不配置车道内横向偏移。
- 可选主场景协同只读取现有 OBU 车辆状态，不依赖背景车之间的意图通信。
- 普通伴行条件无法满足时保持车道退化退出，不把场景失败直接等同于停车。
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
from traj_planner.tools import (
    calculate_path_cumulative_distance,
    get_path_point_at_s,
    normalize_angle,
    project_point_to_path,
)


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量；缺失或格式非法时使用默认值。"""

    raw_value = os.getenv(name, "")
    if raw_value == "":
        return float(default)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量；缺失或格式非法时使用默认值。"""

    raw_value = os.getenv(name, "")
    if raw_value == "":
        return int(default)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return int(default)


def _env_str(name: str, default: str = "") -> str:
    """读取字符串环境变量；空字符串按未配置处理。"""

    value = os.getenv(name, "")
    return str(value if value != "" else default)


@dataclass(frozen=True)
class VehicleLimits:
    """伴行算法使用的车辆尺寸和平台运动约束。"""

    ego_length: float
    ego_width: float
    main_length: float
    main_width: float
    max_v: float
    min_acc: float
    max_acc: float


@dataclass(frozen=True)
class PositionSpec:
    """八方位对应的车道关系和纵向方向。"""

    lane_relation: str
    longitudinal_relation: str


POSITION_SPECS: dict[str, PositionSpec] = {
    # 方位由“GVT 所在车道相对 VUT 的关系”与“GVT 的纵向方向”共同表达。
    # left/right 表示相邻车道内与 VUT 近似并排，纵向允许由配置窗口细化。
    "front": PositionSpec("same", "front"),
    "rear": PositionSpec("same", "rear"),
    "left": PositionSpec("left", "side"),
    "right": PositionSpec("right", "side"),
    "left_front": PositionSpec("left", "front"),
    "right_front": PositionSpec("right", "front"),
    "left_rear": PositionSpec("left", "rear"),
    "right_rear": PositionSpec("right", "rear"),
}


DEFAULT_REL_S_WINDOWS: dict[str, tuple[float, float, float]] = {
    # 三元组依次为 (最小值, 控制目标值, 最大值)。
    # rel_s = s_GVT - s_VUT，因此正值在 VUT 前方，负值在 VUT 后方。
    "front": (8.0, 12.0, 16.0),
    "rear": (-16.0, -12.0, -8.0),
    "left": (-2.0, 0.0, 2.0),
    "right": (-2.0, 0.0, 2.0),
    "left_front": (5.0, 8.0, 12.0),
    "right_front": (5.0, 8.0, 12.0),
    "left_rear": (-12.0, -8.0, -5.0),
    "right_rear": (-12.0, -8.0, -5.0),
}


_INITIAL_POSITION = _env_str("ROS_SCENE_COMPANION_POSITION", "front").strip().lower()
_INITIAL_WINDOW = DEFAULT_REL_S_WINDOWS.get(_INITIAL_POSITION, DEFAULT_REL_S_WINDOWS["front"])


CFG: dict[str, Any] = {
    # 伴行方位和纵向位置窗口。方位负责车道语义，rel_s 窗口负责纵向语义；
    # 横向位置固定为预设车道中心线，不提供 rel_d 控制参数。
    "position": _INITIAL_POSITION,
    "rel_s_min_m": _env_float("ROS_SCENE_COMPANION_REL_S_MIN_M", _INITIAL_WINDOW[0]),
    "rel_s_target_m": _env_float("ROS_SCENE_COMPANION_REL_S_TARGET_M", _INITIAL_WINDOW[1]),
    "rel_s_max_m": _env_float("ROS_SCENE_COMPANION_REL_S_MAX_M", _INITIAL_WINDOW[2]),
    # 0 表示持续伴行到全局任务结束；正数从首次稳定进入 HOLD_SLOT 开始计时。
    "companion_duration_s": _env_float("ROS_SCENE_COMPANION_DURATION_S", 0.0),
    # 可选主场景协同。通过 ptc_id 指定主场景 GVT，并只读取其现有 OBU 状态。
    # intersection_conflict 第一版不进入专用释放逻辑，由场景配置保证互不干扰。
    "primary_ptc_id": _env_str("ROS_SCENE_COMPANION_PRIMARY_PTC_ID", ""),
    "primary_scene_type": _env_str("ROS_SCENE_COMPANION_PRIMARY_SCENE_TYPE", "").strip().lower(),
    # 纵向闭环和轨迹目标。不同阶段分别配置误差增益、相对速度阻尼和
    # 相对 VUT 的最大速度修正量，避免位置纠偏造成突兀加减速。
    "fallback_cruise_v_mps": _env_float("ROS_SCENE_COMPANION_CRUISE_V_MPS", 5.0),
    "acquire_gain": 0.45,
    "acquire_rel_v_damping": 0.25,
    "hold_gain": 0.40,
    "hold_rel_v_damping": 0.35,
    "recover_gain": 0.30,
    "recover_rel_v_damping": 0.30,
    "release_gain": 0.45,
    "release_rel_v_damping": 0.25,
    "acquire_max_speed_adv_mps": 2.5,
    "acquire_max_speed_lag_mps": 2.0,
    "hold_max_speed_adv_mps": 1.2,
    "hold_max_speed_lag_mps": 1.2,
    "release_max_speed_adv_mps": 3.0,
    "release_max_speed_lag_mps": 3.0,
    "plan_t_s": 0.5,
    "min_target_progress_m": 1.0,
    # 几何判断和状态去抖。连续命中/丢失若干帧后才切换状态，
    # 防止定位、车道匹配或边界测量抖动导致状态机来回跳变。
    "lane_center_tolerance_m": 1.5,
    "same_lane_lateral_tolerance_m": 1.8,
    "side_lane_lateral_min_m": 1.8,
    "side_lane_lateral_max_m": 6.5,
    "relation_projection_max_d_m": 8.0,
    "relation_max_jump_m": _env_float("ROS_SCENE_COMPANION_RELATION_MAX_JUMP_M", 5.0),
    "max_heading_diff_deg": 20.0,
    "slot_enter_frames": _env_int("ROS_SCENE_COMPANION_SLOT_ENTER_FRAMES", 3),
    "slot_exit_frames": _env_int("ROS_SCENE_COMPANION_SLOT_EXIT_FRAMES", 5),
    "primary_clear_frames": _env_int("ROS_SCENE_COMPANION_PRIMARY_CLEAR_FRAMES", 5),
    "min_road_remaining_m": 3.0,
    # 主场景空间释放。该逻辑只服务显式关联的主场景 GVT，
    # 不把所有普通障碍车都提升为需要改变伴行位置的协同对象。
    "cooperation_supported_scene_types": {
        "cut_in",
        "cutin",
        "cut_out",
        "cutout",
        "emergency_braking",
        "braking",
        "blind_spot",
    },
    "primary_interaction_distance_m": 45.0,
    "primary_lane_capture_d_m": 1.8,
    "primary_lane_approach_d_m": 5.0,
    "primary_prediction_horizon_s": 3.0,
    "primary_min_net_gap_m": 6.0,
    "primary_gap_buffer_m": 1.0,
    # 只有系统级不可安全执行时才消费 emergency_stop。
    "fail_safe_decel_mps2": 2.5,
}


SCENE_MEMORY_TEMPLATE: dict[str, Any] = {
    # 状态机主状态及锁定的伴行路线。锁线后优先按 lane_sequence 恢复，
    # 再按 lane_id、line_index 回退，以适应候选参考线下标变化。
    "mode": "WAIT_MAIN",
    "locked_line_index": None,
    "locked_lane_id": "",
    "locked_lane_sequence": [],
    "locked_ego_route_lane_ids": [],
    "locked_main_route_lane_ids": [],
    # 伴行窗口进入/离开和主场景空间清空均使用连续帧计数去抖。
    "slot_inside_hits": 0,
    "slot_outside_hits": 0,
    "primary_clear_hits": 0,
    "relation_invalid_hits": 0,
    "state_geometry_invalid_hits": 0,
    # mission_start_time_s 只在首次稳定伴行时写入，状态切换不会重复计时。
    "mission_start_time_s": -1.0,
    "state_start_time_s": 0.0,
    # 空间释放目标需要跨帧锁存，避免主场景车辆短暂丢失后目标立即跳回。
    "release_target_rel_s_m": None,
    "release_direction": "",
    "degraded_reason": "",
    "fail_safe_reason": "",
    "last_valid_rel_s_m": None,
    "last_valid_lane_relation": "",
    "last_metrics": None,
}


SCENE_MEMORY: dict[str, Any] = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Public API / 外部入口 ====================


def companion_algorithm(context: SceneContext) -> dict[str, Any]:
    """伴行场景算法主入口。

    每个规划周期调用一次，输入由 ``SceneContext`` 提供；函数更新内部状态机，
    返回局部规划器可消费的 ``ExternalTarget`` 字段及场景调试信息。
    """

    global SCENE_MEMORY
    start_ms = time.time() * 1000.0
    # step_index=0 视为一轮新任务，清除上一轮的锁线、计数器和终态。
    if int(context.step_index) == 0:
        _reset_scene_memory()

    previous_mode = str(SCENE_MEMORY.get("mode", "WAIT_MAIN"))
    output, SCENE_MEMORY = _scene_fsm_step(context, SCENE_MEMORY)
    current_mode = str(SCENE_MEMORY.get("mode", "WAIT_MAIN"))
    if current_mode != previous_mode:
        print(
            f"[companion] transition {previous_mode}->{current_mode} "
            f"status={output.get('scene_status', '')} reason={output.get('scene_detail', output.get('reason', ''))}",
            flush=True,
        )
    output.setdefault("scene_debug", {})["algorithm_time_ms"] = round(time.time() * 1000.0 - start_ms, 3)
    return output


# ==================== FSM / 状态机 ====================


def _scene_fsm_step(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行一帧伴行状态机。

    终态优先闭锁；运行态先校验静态配置，再分发到具体状态处理函数。
    普通伴行条件不可满足进入 ``DEGRADED_EXIT``，只有车体硬重叠才进入
    ``FAIL_SAFE``，避免把场景退化直接转换成停车并破坏主测试场景。
    """

    mode = str(memory.get("mode", "WAIT_MAIN"))
    now = float(context.time_s)

    # 三个终态在本轮任务内保持闭锁，只有新一轮 step_index=0 才重置。
    if mode == "DEGRADED_EXIT":
        return _handle_degraded_exit(context, memory), memory
    if mode == "FAIL_SAFE":
        return _handle_fail_safe(context, memory), memory
    if mode == "COMPLETE":
        return _handle_complete(context, memory), memory

    # 配置错误属于场景不可执行，不属于紧急制动条件。
    config_error = _validate_config()
    if config_error:
        _enter_degraded_exit(memory, now, config_error)
        return _handle_degraded_exit(context, memory), memory

    # 运行态采用显式分发，未知状态统一退化退出，避免输出未定义行为。
    if mode == "WAIT_MAIN":
        return _handle_wait_main(context, memory, now)
    if mode == "ACQUIRE_SLOT":
        return _handle_acquire_slot(context, memory, now)
    if mode == "HOLD_SLOT":
        return _handle_hold_slot(context, memory, now)
    if mode == "CREATE_GAP":
        return _handle_create_gap(context, memory, now)
    if mode == "HOLD_RELEASED_GAP":
        return _handle_hold_released_gap(context, memory, now)
    if mode == "RECOVER_SLOT":
        return _handle_recover_slot(context, memory, now)

    _enter_degraded_exit(memory, now, f"unknown_mode:{mode}")
    return _handle_degraded_exit(context, memory), memory


# ==================== State Handlers / 状态处理 ====================


def _handle_wait_main(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """等待指定 VUT 和合法伴行车道关系。"""

    if context.main_vehicle is None:
        # VUT 未接入时只沿当前车道巡航，不尝试凭历史状态构造伴行关系。
        return _build_keep_lane_output(
            context,
            reason="companion_wait_main",
            scene_status="waiting_main_vehicle",
            scene_debug={"main_valid": False, "position": _normalized_position()},
        ), memory

    target_line = _select_formation_line(context)
    metrics = _compute_companion_metrics(context, target_line, memory)
    memory["last_metrics"] = copy.deepcopy(metrics)
    # 所有运行态都先处理已经发生的车体硬重叠，安全终态优先级最高。
    if bool(metrics.get("hard_collision_risk", False)):
        _enter_fail_safe(memory, now, "companion_hard_collision")
        return _handle_fail_safe(context, memory), memory

    if not _is_slot_candidate(metrics):
        # 初始车道或前后方位不匹配时继续等待，第一版不主动换道获取方位。
        return _build_keep_lane_output(
            context,
            reason="companion_wait_geometry",
            scene_status="waiting_formation_geometry",
            scene_debug=metrics,
        ), memory

    _lock_formation_line(context, memory, target_line)
    if bool(metrics.get("inside_slot", False)):
        # 初始位置已经满足窗口时直接进入稳定伴行，否则先纵向追踪目标位置。
        _enter_hold_slot(memory, now)
        return _build_hold_output(context, target_line, metrics), memory

    _enter_mode(memory, "ACQUIRE_SLOT", now)
    return _build_acquire_output(context, target_line, metrics), memory


def _handle_acquire_slot(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """通过纵向速度控制进入配置的 ``rel_s`` 窗口。"""

    target_line = _resolve_formation_line(context, memory)
    metrics = _compute_companion_metrics(context, target_line, memory)
    memory["last_metrics"] = copy.deepcopy(metrics)
    if bool(metrics.get("hard_collision_risk", False)):
        _enter_fail_safe(memory, now, "companion_hard_collision")
        return _handle_fail_safe(context, memory), memory
    if not _is_slot_candidate(metrics):
        if not _geometry_loss_confirmed(memory, geometry_valid=False):
            return _build_keep_lane_output(
                context,
                reason="companion_acquire_geometry_debounce",
                scene_status="geometry_debounce",
                scene_debug=metrics,
            ), memory
        # 获取过程中几何关系丢失时回到等待态并解除锁线，等待场景重新成立。
        _enter_wait_main(memory, now)
        return _build_keep_lane_output(
            context,
            reason="companion_acquire_geometry_lost",
            scene_status="waiting_formation_geometry",
            scene_debug=metrics,
        ), memory
    _geometry_loss_confirmed(memory, geometry_valid=True)
    if float(metrics.get("road_remaining_m", 0.0)) < float(CFG["min_road_remaining_m"]):
        _enter_degraded_exit(memory, now, "companion_insufficient_road")
        return _handle_degraded_exit(context, memory), memory

    # 连续若干帧落入窗口才认为获取完成，避免边界抖动提前进入 HOLD_SLOT。
    _update_slot_hits(memory, bool(metrics.get("inside_slot", False)))
    if int(memory["slot_inside_hits"]) >= int(CFG["slot_enter_frames"]):
        _enter_hold_slot(memory, now)
        return _build_hold_output(context, target_line, metrics), memory
    return _build_acquire_output(context, target_line, metrics), memory


def _handle_hold_slot(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """稳定保持配置伴行位置，并检查是否需要为主场景释放空间。"""

    target_line = _resolve_formation_line(context, memory)
    metrics = _compute_companion_metrics(context, target_line, memory)
    # 协同判断只针对配置关联的主场景 GVT；未配置时结果为 needs_release=False。
    interaction = _compute_primary_interaction(context, target_line, metrics)
    metrics["primary_interaction"] = copy.deepcopy(interaction)
    memory["last_metrics"] = copy.deepcopy(metrics)

    if bool(metrics.get("hard_collision_risk", False)):
        _enter_fail_safe(memory, now, "companion_hard_collision")
        return _handle_fail_safe(context, memory), memory
    if not _is_slot_candidate(metrics):
        if not _geometry_loss_confirmed(memory, geometry_valid=False):
            return _build_keep_lane_output(
                context,
                reason="companion_hold_geometry_debounce",
                scene_status="geometry_debounce",
                scene_debug=metrics,
            ), memory
        _enter_wait_main(memory, now)
        return _build_keep_lane_output(
            context,
            reason="companion_hold_geometry_lost",
            scene_status="waiting_formation_geometry",
            scene_debug=metrics,
        ), memory
    _geometry_loss_confirmed(memory, geometry_valid=True)
    if bool(interaction.get("needs_release", False)):
        # 先保存释放方向和目标位置，再进入 CREATE_GAP 做纵向前移或后撤。
        _enter_create_gap(memory, now, interaction)
        return _build_create_gap_output(context, target_line, metrics, interaction), memory

    # 短时越界继续保持控制；连续越界才退回 ACQUIRE_SLOT 重新收敛。
    _update_slot_hits(memory, bool(metrics.get("inside_slot", False)))
    if int(memory["slot_outside_hits"]) >= int(CFG["slot_exit_frames"]):
        _enter_mode(memory, "ACQUIRE_SLOT", now)
        return _build_acquire_output(context, target_line, metrics), memory
    if _mission_duration_reached(memory, now):
        # 有限伴行时长只决定场景完成，不要求所有车辆在该时刻停车。
        _enter_mode(memory, "COMPLETE", now)
        return _handle_complete(context, memory), memory
    return _build_hold_output(context, target_line, metrics), memory


def _handle_create_gap(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """前移或后撤，为配置关联的主测试场景 GVT 释放纵向空间。"""

    target_line = _resolve_formation_line(context, memory)
    metrics = _compute_companion_metrics(context, target_line, memory)
    interaction = _compute_primary_interaction(context, target_line, metrics)
    metrics["primary_interaction"] = copy.deepcopy(interaction)
    memory["last_metrics"] = copy.deepcopy(metrics)

    if bool(metrics.get("hard_collision_risk", False)):
        _enter_fail_safe(memory, now, "companion_hard_collision")
        return _handle_fail_safe(context, memory), memory
    if not _is_slot_candidate(metrics):
        if not _geometry_loss_confirmed(memory, geometry_valid=False):
            return _build_keep_lane_output(
                context,
                reason="companion_release_geometry_debounce",
                scene_status="geometry_debounce",
                scene_debug=metrics,
            ), memory
        # 释放过程中连基础伴行车道都无法保持时采用非停车退化退出。
        _enter_degraded_exit(memory, now, "companion_release_geometry_lost")
        return _handle_degraded_exit(context, memory), memory
    _geometry_loss_confirmed(memory, geometry_valid=True)
    if not bool(interaction.get("primary_valid", False)):
        # 主场景 GVT 暂时不可见时不继续扩大空间，转入平滑恢复流程。
        _enter_mode(memory, "RECOVER_SLOT", now)
        return _build_recover_output(context, target_line, metrics), memory

    # 连续满足中心距要求后才进入保持释放空间，避免一帧达标立即反向恢复。
    _update_release_memory(memory, interaction)
    if bool(interaction.get("space_satisfied", False)):
        memory["primary_clear_hits"] = int(memory.get("primary_clear_hits", 0)) + 1
    else:
        memory["primary_clear_hits"] = 0
    if int(memory["primary_clear_hits"]) >= int(CFG["slot_enter_frames"]):
        _enter_mode(memory, "HOLD_RELEASED_GAP", now)
        return _build_released_gap_output(context, target_line, metrics, interaction, memory), memory
    return _build_create_gap_output(context, target_line, metrics, interaction), memory


def _handle_hold_released_gap(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """主场景车仍占据原伴行窗口时，继续保持已释放空间。"""

    target_line = _resolve_formation_line(context, memory)
    metrics = _compute_companion_metrics(context, target_line, memory)
    interaction = _compute_primary_interaction(context, target_line, metrics)
    metrics["primary_interaction"] = copy.deepcopy(interaction)
    memory["last_metrics"] = copy.deepcopy(metrics)

    if bool(metrics.get("hard_collision_risk", False)):
        _enter_fail_safe(memory, now, "companion_hard_collision")
        return _handle_fail_safe(context, memory), memory
    if not _is_slot_candidate(metrics):
        if not _geometry_loss_confirmed(memory, geometry_valid=False):
            return _build_keep_lane_output(
                context,
                reason="companion_released_geometry_debounce",
                scene_status="geometry_debounce",
                scene_debug=metrics,
            ), memory
        _enter_degraded_exit(memory, now, "companion_released_geometry_lost")
        return _handle_degraded_exit(context, memory), memory
    _geometry_loss_confirmed(memory, geometry_valid=True)

    # 主场景车辆仍可见时滚动更新释放目标；丢失时沿用内存中的最后目标。
    if bool(interaction.get("primary_valid", False)):
        _update_release_memory(memory, interaction)
    if bool(interaction.get("original_slot_occupied", False)) or bool(interaction.get("needs_release", False)):
        # 主场景车辆仍占据原伴行窗口时禁止恢复，防止切入完成后两车争抢同一位置。
        memory["primary_clear_hits"] = 0
        return _build_released_gap_output(context, target_line, metrics, interaction, memory), memory

    # 原窗口连续清空若干帧后才允许恢复原始伴行方位。
    memory["primary_clear_hits"] = int(memory.get("primary_clear_hits", 0)) + 1
    if int(memory["primary_clear_hits"]) >= int(CFG["primary_clear_frames"]):
        _enter_mode(memory, "RECOVER_SLOT", now)
        return _build_recover_output(context, target_line, metrics), memory
    return _build_released_gap_output(context, target_line, metrics, interaction, memory), memory


def _handle_recover_slot(
    context: SceneContext,
    memory: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """主场景空间解除后平滑恢复原伴行位置。"""

    target_line = _resolve_formation_line(context, memory)
    metrics = _compute_companion_metrics(context, target_line, memory)
    interaction = _compute_primary_interaction(context, target_line, metrics)
    metrics["primary_interaction"] = copy.deepcopy(interaction)
    memory["last_metrics"] = copy.deepcopy(metrics)

    if bool(metrics.get("hard_collision_risk", False)):
        _enter_fail_safe(memory, now, "companion_hard_collision")
        return _handle_fail_safe(context, memory), memory
    if not _is_slot_candidate(metrics):
        if not _geometry_loss_confirmed(memory, geometry_valid=False):
            return _build_keep_lane_output(
                context,
                reason="companion_recover_geometry_debounce",
                scene_status="geometry_debounce",
                scene_debug=metrics,
            ), memory
        _enter_wait_main(memory, now)
        return _build_keep_lane_output(
            context,
            reason="companion_recover_geometry_lost",
            scene_status="waiting_formation_geometry",
            scene_debug=metrics,
        ), memory
    _geometry_loss_confirmed(memory, geometry_valid=True)
    if bool(interaction.get("needs_release", False)) or bool(interaction.get("original_slot_occupied", False)):
        # 恢复途中冲突重新出现时立即回到空间释放，不继续追原位置。
        _enter_create_gap(memory, now, interaction)
        return _build_create_gap_output(context, target_line, metrics, interaction), memory

    _update_slot_hits(memory, bool(metrics.get("inside_slot", False)))
    if int(memory["slot_inside_hits"]) >= int(CFG["slot_enter_frames"]):
        _enter_hold_slot(memory, now)
        return _build_hold_output(context, target_line, metrics), memory
    return _build_recover_output(context, target_line, metrics), memory


def _handle_complete(context: SceneContext, memory: dict[str, Any]) -> dict[str, Any]:
    """有限时长伴行完成后保持当前车道巡航，不默认停车。"""

    line = _resolve_formation_line(context, memory) or context.ego_vehicle.current_ref_line
    # COMPLETE 仅闭锁“伴行任务已完成”，车辆继续随 VUT/默认速度巡航。
    target_v = (
        float(context.main_vehicle.v)
        if context.main_vehicle is not None
        else float(CFG["fallback_cruise_v_mps"])
    )
    return _build_line_target_output(
        context=context,
        target_line=line,
        target_v=target_v,
        plan_t=float(CFG["plan_t_s"]),
        reason="companion_complete",
        scene_status="scene_completed",
        scene_debug=copy.deepcopy(memory.get("last_metrics") or {}),
    )


def _handle_degraded_exit(context: SceneContext, memory: dict[str, Any]) -> dict[str, Any]:
    """伴行目标不可实现时保持当前车道退出，不把普通场景失败转成停车。"""

    debug = copy.deepcopy(memory.get("last_metrics") or {})
    debug["degraded_reason"] = str(memory.get("degraded_reason", "companion_degraded"))
    return _build_keep_lane_output(
        context,
        reason=str(memory.get("degraded_reason", "companion_degraded")),
        scene_status="degraded_exit",
        scene_debug=debug,
    )


def _handle_fail_safe(context: SceneContext, memory: dict[str, Any]) -> dict[str, Any]:
    """系统级不可安全执行时才进入 emergency_stop。"""

    line = _resolve_formation_line(context, memory) or context.ego_vehicle.current_ref_line
    reason = str(memory.get("fail_safe_reason", "companion_fail_safe"))
    debug = copy.deepcopy(memory.get("last_metrics") or {})
    debug["fail_safe_reason"] = reason
    debug["requested_decel_mps2"] = float(CFG["fail_safe_decel_mps2"])
    # 这里同时输出零速终端和 emergency_stop 模式，明确要求下游执行紧急减速。
    output = _build_line_target_output(
        context=context,
        target_line=line,
        target_v=0.0,
        plan_t=max(
            float(context.ego_vehicle.v) / max(float(CFG["fail_safe_decel_mps2"]), 1e-3),
            float(context.dt),
        ),
        reason=reason,
        scene_status="fail_safe",
        scene_debug=debug,
    )
    output["control_mode"] = "emergency_stop"
    output["emergency_decel_mps2"] = float(CFG["fail_safe_decel_mps2"])
    return output


# ==================== Transition Helpers / 状态切换 ====================


def _enter_mode(memory: dict[str, Any], mode: str, now: float) -> None:
    """写入新状态，并记录该状态的进入时刻。"""

    memory["mode"] = str(mode)
    memory["state_start_time_s"] = float(now)


def _enter_wait_main(memory: dict[str, Any], now: float) -> None:
    """回到等待态，并清除上一段伴行任务锁定的路线和去抖记忆。"""

    _enter_mode(memory, "WAIT_MAIN", now)
    memory["locked_line_index"] = None
    memory["locked_lane_id"] = ""
    memory["locked_lane_sequence"] = []
    memory["locked_ego_route_lane_ids"] = []
    memory["locked_main_route_lane_ids"] = []
    memory["relation_invalid_hits"] = 0
    memory["state_geometry_invalid_hits"] = 0
    memory["last_valid_rel_s_m"] = None
    memory["last_valid_lane_relation"] = ""
    _reset_slot_hits(memory)
    _reset_release_memory(memory)


def _enter_hold_slot(memory: dict[str, Any], now: float) -> None:
    """进入稳定伴行态，并在首次进入时启动有限时长任务计时。"""

    _enter_mode(memory, "HOLD_SLOT", now)
    if float(memory.get("mission_start_time_s", -1.0)) < 0.0:
        memory["mission_start_time_s"] = float(now)
    _reset_slot_hits(memory)
    _reset_release_memory(memory)


def _enter_create_gap(memory: dict[str, Any], now: float, interaction: dict[str, Any]) -> None:
    """进入主场景空间释放态，并锁存本帧计算的释放目标。"""

    _enter_mode(memory, "CREATE_GAP", now)
    memory["primary_clear_hits"] = 0
    _update_release_memory(memory, interaction)


def _enter_degraded_exit(memory: dict[str, Any], now: float, reason: str) -> None:
    """锁存非紧急退化终态及其原因。"""

    _enter_mode(memory, "DEGRADED_EXIT", now)
    memory["degraded_reason"] = str(reason)


def _enter_fail_safe(memory: dict[str, Any], now: float, reason: str) -> None:
    """锁存系统级安全终态及其原因。"""

    _enter_mode(memory, "FAIL_SAFE", now)
    memory["fail_safe_reason"] = str(reason)


def _lock_formation_line(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView | None,
) -> None:
    """记录伴行参考线及双方全局 lane 走廊。"""

    if target_line is None:
        return
    memory["locked_line_index"] = int(target_line.line_index)
    memory["locked_lane_id"] = str(target_line.lane_id)
    memory["locked_lane_sequence"] = _normalize_lane_sequence(target_line.lane_sequence)
    memory["locked_ego_route_lane_ids"] = _vehicle_route_lane_ids(
        context.ego_vehicle,
        fallback=target_line.lane_sequence,
    )
    memory["locked_main_route_lane_ids"] = _vehicle_route_lane_ids(
        context.main_vehicle,
        fallback=(
            []
            if context.main_vehicle is None
            else context.main_vehicle.current_ref_line.lane_sequence
        ),
    )


def _update_slot_hits(memory: dict[str, Any], inside_slot: bool) -> None:
    """更新进入/离开伴行窗口的连续帧计数器。"""

    if inside_slot:
        memory["slot_inside_hits"] = int(memory.get("slot_inside_hits", 0)) + 1
        memory["slot_outside_hits"] = 0
    else:
        memory["slot_inside_hits"] = 0
        memory["slot_outside_hits"] = int(memory.get("slot_outside_hits", 0)) + 1


def _reset_slot_hits(memory: dict[str, Any]) -> None:
    """清空伴行窗口进入和离开去抖计数。"""

    memory["slot_inside_hits"] = 0
    memory["slot_outside_hits"] = 0


def _geometry_loss_confirmed(
    memory: dict[str, Any],
    *,
    geometry_valid: bool,
) -> bool:
    """对车道/方位几何丢失去抖，避免单帧地图切段触发永久退化。"""

    if geometry_valid:
        memory["state_geometry_invalid_hits"] = 0
        return False
    hits = int(memory.get("state_geometry_invalid_hits", 0)) + 1
    memory["state_geometry_invalid_hits"] = hits
    return hits >= int(CFG["slot_exit_frames"])


def _update_release_memory(memory: dict[str, Any], interaction: dict[str, Any]) -> None:
    """保存有效的空间释放目标和方向，供目标车短时丢失时继续使用。"""

    target = interaction.get("release_target_rel_s_m")
    if target is not None and np.isfinite(float(target)):
        memory["release_target_rel_s_m"] = float(target)
    direction = str(interaction.get("release_direction", ""))
    if direction:
        memory["release_direction"] = direction


def _reset_release_memory(memory: dict[str, Any]) -> None:
    """清空主场景空间释放阶段的跨帧记忆。"""

    memory["primary_clear_hits"] = 0
    memory["release_target_rel_s_m"] = None
    memory["release_direction"] = ""


def _reset_scene_memory() -> None:
    """新一轮场景开始时恢复状态机初始内存。"""

    global SCENE_MEMORY
    SCENE_MEMORY = copy.deepcopy(SCENE_MEMORY_TEMPLATE)


# ==================== Output Builders / 输出构造 ====================


def _build_acquire_output(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """构造获取伴行位置的纵向控制目标，允许较大的速度修正量。"""

    return _build_rel_s_control_output(
        context=context,
        target_line=target_line,
        metrics=metrics,
        target_rel_s=float(CFG["rel_s_target_m"]),
        gain=float(CFG["acquire_gain"]),
        rel_v_damping=float(CFG["acquire_rel_v_damping"]),
        max_adv=float(CFG["acquire_max_speed_adv_mps"]),
        max_lag=float(CFG["acquire_max_speed_lag_mps"]),
        reason="companion_acquire_slot",
        scene_status="acquire_slot",
    )


def _build_hold_output(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """构造稳定伴行目标，以较小速度修正抑制位置波动。"""

    return _build_rel_s_control_output(
        context=context,
        target_line=target_line,
        metrics=metrics,
        target_rel_s=float(CFG["rel_s_target_m"]),
        gain=float(CFG["hold_gain"]),
        rel_v_damping=float(CFG["hold_rel_v_damping"]),
        max_adv=float(CFG["hold_max_speed_adv_mps"]),
        max_lag=float(CFG["hold_max_speed_lag_mps"]),
        reason="companion_hold_slot",
        scene_status="hold_slot",
    )


def _build_create_gap_output(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    interaction: dict[str, Any],
) -> dict[str, Any]:
    """构造主场景空间释放目标，按计算结果控制 GVT 前移或后撤。"""

    target_rel_s = float(
        interaction.get(
            "release_target_rel_s_m",
            CFG["rel_s_target_m"],
        )
    )
    return _build_rel_s_control_output(
        context=context,
        target_line=target_line,
        metrics=metrics,
        target_rel_s=target_rel_s,
        gain=float(CFG["release_gain"]),
        rel_v_damping=float(CFG["release_rel_v_damping"]),
        max_adv=float(CFG["release_max_speed_adv_mps"]),
        max_lag=float(CFG["release_max_speed_lag_mps"]),
        reason="companion_create_gap",
        scene_status="create_gap",
        reference_v=float(interaction.get("release_reference_v_mps", metrics.get("main_v_mps", 0.0))),
        extra_debug={"primary_interaction": copy.deepcopy(interaction)},
    )


def _build_released_gap_output(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    interaction: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    """构造释放空间保持目标；实时结果缺失时沿用上一帧有效目标。"""

    # 优先使用当前观测计算的释放目标，主场景 GVT 短时丢失时再读取锁存值。
    target_raw = interaction.get("release_target_rel_s_m")
    if target_raw is None:
        target_raw = memory.get("release_target_rel_s_m")
    if target_raw is None:
        target_raw = CFG["rel_s_target_m"]
    return _build_rel_s_control_output(
        context=context,
        target_line=target_line,
        metrics=metrics,
        target_rel_s=float(target_raw),
        gain=float(CFG["hold_gain"]),
        rel_v_damping=float(CFG["hold_rel_v_damping"]),
        max_adv=float(CFG["release_max_speed_adv_mps"]),
        max_lag=float(CFG["release_max_speed_lag_mps"]),
        reason="companion_hold_released_gap",
        scene_status="hold_released_gap",
        reference_v=float(interaction.get("release_reference_v_mps", metrics.get("main_v_mps", 0.0))),
        extra_debug={"primary_interaction": copy.deepcopy(interaction)},
    )


def _build_recover_output(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """构造平滑恢复原始伴行窗口的纵向控制目标。"""

    return _build_rel_s_control_output(
        context=context,
        target_line=target_line,
        metrics=metrics,
        target_rel_s=float(CFG["rel_s_target_m"]),
        gain=float(CFG["recover_gain"]),
        rel_v_damping=float(CFG["recover_rel_v_damping"]),
        max_adv=float(CFG["acquire_max_speed_adv_mps"]),
        max_lag=float(CFG["acquire_max_speed_lag_mps"]),
        reason="companion_recover_slot",
        scene_status="recover_slot",
    )


def _build_rel_s_control_output(
    *,
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
    target_rel_s: float,
    gain: float,
    rel_v_damping: float,
    max_adv: float,
    max_lag: float,
    reason: str,
    scene_status: str,
    reference_v: float | None = None,
    extra_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按 ``rel_s`` 误差和相对速度构造纵向终端速度。

    控制律为 ``速度修正 = Kp * rel_s误差 - Kd * rel_v``，再叠加 VUT
    或主场景参考车速度。这里只输出可达的终端速度和规划时域；
    ``target_a`` 是由速度变化反算的平均加速度，用于调试，不要求下游
    同时消费独立的终端加速度边界。
    """

    line = target_line if target_line is not None else context.ego_vehicle.current_ref_line
    limits = _read_vehicle_limits(context)
    rel_s = float(metrics.get("rel_s_m", CFG["rel_s_target_m"]))
    rel_v = float(metrics.get("rel_v_mps", 0.0))
    rel_error = float(target_rel_s) - rel_s
    # 位置误差推动 GVT 接近目标 rel_s，相对速度阻尼用于减小越过目标后的振荡。
    speed_delta = _clamp(
        float(gain) * rel_error - float(rel_v_damping) * rel_v,
        -float(max_lag),
        float(max_adv),
    )
    # 普通伴行以 VUT 速度为前馈；空间释放时可改用主场景 GVT 相关参考速度。
    base_v = float(metrics.get("main_v_mps", 0.0)) if reference_v is None else float(reference_v)
    raw_target_v = _clamp(base_v + speed_delta, 0.0, limits.max_v)
    plan_t = max(float(CFG["plan_t_s"]), float(context.dt))
    # 用车辆加速度上下限裁剪本规划周期可达速度，避免给局部规划器无解终端。
    reachable_min_v = max(0.0, float(context.ego_vehicle.v) + limits.min_acc * plan_t)
    reachable_max_v = min(limits.max_v, float(context.ego_vehicle.v) + limits.max_acc * plan_t)
    target_v = _clamp(raw_target_v, reachable_min_v, reachable_max_v)
    # 平均加速度仅记录和随输出透传；实际轨迹仍由终端位置、速度和时域生成。
    target_a = (target_v - float(context.ego_vehicle.v)) / max(plan_t, 1e-3)

    debug = copy.deepcopy(metrics)
    debug.update(
        {
            "control_target_rel_s_m": float(target_rel_s),
            "control_rel_s_error_m": float(rel_error),
            "control_rel_v_mps": float(rel_v),
            "control_reference_v_mps": float(base_v),
            "control_speed_delta_mps": float(speed_delta),
            "control_raw_target_v_mps": float(raw_target_v),
            "control_target_v_mps": float(target_v),
            "control_required_avg_acc_mps2": float(target_a),
            "control_reachable_v_min_mps": float(reachable_min_v),
            "control_reachable_v_max_mps": float(reachable_max_v),
            "external_target_acc_consumed": False,
        }
    )
    if extra_debug is not None:
        debug.update(copy.deepcopy(extra_debug))
    return _build_line_target_output(
        context=context,
        target_line=line,
        target_v=target_v,
        plan_t=plan_t,
        reason=reason,
        scene_status=scene_status,
        scene_debug=debug,
        target_a=target_a,
    )


def _build_keep_lane_output(
    context: SceneContext,
    reason: str,
    scene_status: str,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造保持当前车道的非停车保底目标。

    该输出用于等待或普通退化场景：优先跟随 VUT 当前速度，无 VUT 时使用
    配置巡航速度；它不是硬安全停车指令。
    """

    limits = _read_vehicle_limits(context)
    target_v = _clamp(float(CFG["fallback_cruise_v_mps"]), 0.0, limits.max_v)
    if context.main_vehicle is not None:
        target_v = _clamp(float(context.main_vehicle.v), 0.0, limits.max_v)
    return _build_line_target_output(
        context=context,
        target_line=context.ego_vehicle.current_ref_line,
        target_v=target_v,
        plan_t=float(CFG["plan_t_s"]),
        reason=reason,
        scene_status=scene_status,
        scene_debug=scene_debug,
    )


def _build_line_target_output(
    *,
    context: SceneContext,
    target_line: ReferenceLineView,
    target_v: float,
    plan_t: float,
    reason: str,
    scene_status: str,
    scene_debug: dict[str, Any] | None = None,
    target_a: float = 0.0,
) -> dict[str, Any]:
    """在锁定参考线上构造现有 ``ExternalTarget`` 可消费的目标。

    目标点沿参考线前推，``x/y/theta/v/t`` 交给下游局部规划器生成可执行轨迹；
    算法本身不直接下发油门、制动或转向控制量。
    """

    current_s = float(target_line.current_s)
    line_end_s = max(current_s, float(target_line.cumulative_s[-1]) - 0.05)
    remaining_s = max(0.0, line_end_s - current_s)
    # 用当前速度与目标速度的梯形积分估算规划时域内纵向行程，并确保目标点
    # 至少有最小前向距离，防止目标点与车辆当前位置重合。
    travel_s = max(
        float(CFG["min_target_progress_m"]),
        0.5 * (max(float(context.ego_vehicle.v), 0.0) + max(float(target_v), 0.0)) * float(plan_t),
    )
    target_s = current_s + min(travel_s, remaining_s)
    if remaining_s <= 1e-3:
        # 参考线已到末端时将目标速度置零，避免目标点越过可用道路范围。
        target_s = float(target_line.cumulative_s[-1])
        target_v = 0.0
    target_xy, target_heading = get_path_point_at_s(
        target_line.line,
        target_s,
        target_line.cumulative_s,
    )
    output = {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(target_v),
        "a": float(target_a),
        "theta": float(target_heading),
        "t": float(max(plan_t, context.dt)),
        "target_line_index": int(target_line.line_index),
        "hold_t": 0.0,
        "reason": str(reason),
        "control_mode": "trajectory_target",
        "min_external_target_progress_m": float(min(float(CFG["min_target_progress_m"]), remaining_s)),
    }
    return _annotate_output(
        output,
        scene_status=scene_status,
        scene_detail=reason,
        scene_debug=scene_debug,
    )


# ==================== Metrics / 指标与交互计算 ====================


def _compute_companion_metrics(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    memory: dict[str, Any],
) -> dict[str, Any]:
    """统一计算伴行车道关系、纵向位置和硬碰撞指标。

    两车先投影到 GVT 锁定路线的公共关系线，再计算
    ``rel_s = s_GVT - s_VUT``。横向方位通过车道 ID、Frenet 横向差和
    VUT 车体左向投影共同确认，避免只靠世界坐标判断弯道路段左右关系。
    """

    ego = context.ego_vehicle
    main = context.main_vehicle
    position = _normalized_position()
    spec = POSITION_SPECS.get(position)
    limits = _read_vehicle_limits(context)
    hard_collision = _evaluate_hard_collision(context, limits)
    if main is None or target_line is None or spec is None:
        return {
            "main_valid": main is not None,
            "target_line_valid": target_line is not None,
            "position": position,
            "position_valid": spec is not None,
            **hard_collision,
        }

    # 优先使用起终点规划得到的完整走廊线。该坐标轴覆盖完整 lane 段，
    # 不会在任一车辆先进入 successor 时把 s 原点重置到新车道段起点。
    relation = _resolve_relation_geometry(context, target_line, memory, update_memory=True)
    relation_line = relation["line"]
    relation_cumulative = relation["cumulative_s"]
    ego_s = float(relation["ego_s"])
    main_s = float(relation["main_s"])
    ego_d = float(relation["ego_d"])
    main_d = float(relation["main_d"])
    rel_s = float(relation["rel_s"])
    rel_v = float(ego.v) - float(main.v)
    # relation_lateral 使用共同走廊投影，正值表示 GVT 在 VUT 左侧；
    # 世界坐标侧向投影仅保留为调试量，不再承担跨段车道关系主判断。
    relation_lateral = float(ego_d) - float(main_d)
    side_lateral = _signed_left_lateral_offset(main, ego)
    heading_diff = _heading_diff_deg(float(main.heading), float(ego.heading))
    same_direction = heading_diff <= float(CFG["max_heading_diff_deg"])
    ego_centered = abs(float(ego.lane_d)) <= float(CFG["lane_center_tolerance_m"])
    main_centered = abs(float(main.lane_d)) <= float(CFG["lane_center_tolerance_m"])
    ego_route_lane_ids = _locked_or_current_route_lane_ids(
        memory,
        "locked_ego_route_lane_ids",
        ego,
        target_line.lane_sequence,
    )
    main_route_lane_ids = _locked_or_current_route_lane_ids(
        memory,
        "locked_main_route_lane_ids",
        main,
        main.current_ref_line.lane_sequence,
    )
    ego_route_ok = not ego_route_lane_ids or str(ego.lane_id) in ego_route_lane_ids
    main_route_ok = not main_route_lane_ids or str(main.lane_id) in main_route_lane_ids
    route_membership_ok = bool(ego_route_ok and main_route_ok)

    # lane ID 只表示当前 OpenDRIVE lane 段。same/left/right 改由锁定路线成员关系
    # 和共同走廊上的横向差确认，因此 predecessor/successor 异步跨段不会改变方位。
    same_lane = bool(
        route_membership_ok
        and abs(float(relation_lateral)) <= float(CFG["same_lane_lateral_tolerance_m"])
    )
    left_lane = bool(
        route_membership_ok
        and float(CFG["side_lane_lateral_min_m"])
        <= float(relation_lateral)
        <= float(CFG["side_lane_lateral_max_m"])
    )
    right_lane = bool(
        route_membership_ok
        and -float(CFG["side_lane_lateral_max_m"])
        <= float(relation_lateral)
        <= -float(CFG["side_lane_lateral_min_m"])
    )
    if not bool(relation["projection_valid"]):
        last_relation = str(memory.get("last_valid_lane_relation", ""))
        same_lane = bool(relation["relation_usable"] and last_relation == "same")
        left_lane = bool(relation["relation_usable"] and last_relation == "left")
        right_lane = bool(relation["relation_usable"] and last_relation == "right")
    elif same_lane:
        memory["last_valid_lane_relation"] = "same"
    elif left_lane:
        memory["last_valid_lane_relation"] = "left"
    elif right_lane:
        memory["last_valid_lane_relation"] = "right"
    # 配置方位只选择一种期望车道关系；不在算法内主动换道纠正错误初始车道。
    lane_relation_ok = {
        "same": same_lane,
        "left": left_lane,
        "right": right_lane,
    }[spec.lane_relation]
    rel_s_direction_ok = _rel_s_direction_ok(spec, rel_s)
    inside_rel_s = bool(
        float(CFG["rel_s_min_m"]) <= rel_s <= float(CFG["rel_s_max_m"])
    )
    # 只有方向、车道中心、车道关系和 rel_s 窗口同时成立，才算完整伴行命中。
    inside_slot = bool(
        same_direction
        and ego_centered
        and main_centered
        and lane_relation_ok
        and rel_s_direction_ok
        and inside_rel_s
    )
    road_remaining = max(
        0.0,
        float(target_line.cumulative_s[-1]) - float(target_line.current_s),
    )
    mission_start = float(memory.get("mission_start_time_s", -1.0))
    mission_elapsed = 0.0 if mission_start < 0.0 else max(0.0, float(context.time_s) - mission_start)
    return {
        "main_valid": True,
        "target_line_valid": True,
        "position": position,
        "position_valid": True,
        "expected_lane_relation": spec.lane_relation,
        "expected_longitudinal_relation": spec.longitudinal_relation,
        "target_line_index": int(target_line.line_index),
        "target_lane_id": str(target_line.lane_id),
        "target_lane_sequence": _normalize_lane_sequence(target_line.lane_sequence),
        "ego_lane_id": str(ego.lane_id),
        "main_lane_id": str(main.lane_id),
        "ego_s_on_relation_line_m": float(ego_s),
        "main_s_on_relation_line_m": float(main_s),
        "ego_d_on_relation_line_m": float(ego_d),
        "main_d_on_relation_line_m": float(main_d),
        "rel_s_m": float(rel_s),
        "rel_v_mps": float(rel_v),
        "rel_s_min_m": float(CFG["rel_s_min_m"]),
        "rel_s_target_m": float(CFG["rel_s_target_m"]),
        "rel_s_max_m": float(CFG["rel_s_max_m"]),
        "side_lateral_offset_m": float(side_lateral),
        "relation_lateral_offset_m": float(relation_lateral),
        "relation_source": str(relation["source"]),
        "relation_projection_valid": bool(relation["projection_valid"]),
        "relation_usable": bool(relation["relation_usable"]),
        "relation_raw_rel_s_m": float(relation["raw_rel_s"]),
        "relation_expected_rel_s_m": float(relation["expected_rel_s"]),
        "relation_jump_m": float(relation["jump_m"]),
        "relation_invalid_hits": int(memory.get("relation_invalid_hits", 0)),
        "ego_route_membership_ok": bool(ego_route_ok),
        "main_route_membership_ok": bool(main_route_ok),
        "route_membership_ok": bool(route_membership_ok),
        "same_lane": bool(same_lane),
        "left_adjacent_geometry": bool(left_lane),
        "right_adjacent_geometry": bool(right_lane),
        "lane_relation_ok": bool(lane_relation_ok),
        "rel_s_direction_ok": bool(rel_s_direction_ok),
        "inside_rel_s_window": bool(inside_rel_s),
        "inside_slot": bool(inside_slot),
        "same_direction": bool(same_direction),
        "heading_diff_deg": float(heading_diff),
        "ego_centered": bool(ego_centered),
        "main_centered": bool(main_centered),
        "ego_v_mps": float(ego.v),
        "main_v_mps": float(main.v),
        "road_remaining_m": float(road_remaining),
        "mission_elapsed_s": float(mission_elapsed),
        **hard_collision,
    }


def _compute_primary_interaction(
    context: SceneContext,
    target_line: ReferenceLineView | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """仅用配置和 OBU 车辆状态判断是否需要为主场景释放空间。

    函数不接收“请让行”等额外意图消息。域控配置只指定主场景 GVT 的
    ``ptc_id`` 和场景类型，算法依据该车实时位置、航向和速度判断其是否
    已经或即将占用伴行路线，并计算伴行车应前移或后撤到的 ``rel_s``。
    """

    # 未显式关联主场景，或场景类型没有第一版专用规则时，保持普通伴行。
    primary_id = str(CFG.get("primary_ptc_id", "") or "").strip()
    scene_type = str(CFG.get("primary_scene_type", "") or "").strip().lower()
    supported = scene_type in set(CFG["cooperation_supported_scene_types"])
    if not primary_id or not supported or target_line is None or context.main_vehicle is None:
        return {
            "enabled": bool(primary_id),
            "supported": bool(supported),
            "primary_valid": False,
            "primary_ptc_id": primary_id,
            "primary_scene_type": scene_type,
            "needs_release": False,
            "original_slot_occupied": False,
        }

    # 主场景 GVT 通过已有 planning_frame/OBU 障碍物状态读取，不新增车间话题。
    primary = _find_actor_by_ptc_id(context, primary_id)
    if primary is None:
        return {
            "enabled": True,
            "supported": True,
            "primary_valid": False,
            "primary_ptc_id": primary_id,
            "primary_scene_type": scene_type,
            "needs_release": False,
            "original_slot_occupied": False,
        }

    relation = _resolve_relation_geometry(context, target_line, {}, update_memory=False)
    if not bool(relation["projection_valid"]):
        return {
            "enabled": True,
            "supported": True,
            "primary_valid": False,
            "primary_ptc_id": primary_id,
            "primary_scene_type": scene_type,
            "needs_release": False,
            "original_slot_occupied": False,
            "relation_source": str(relation["source"]),
            "reason": "primary_relation_projection_invalid",
        }
    relation_line = relation["line"]
    relation_cumulative = relation["cumulative_s"]
    ego = context.ego_vehicle
    main = context.main_vehicle
    ego_s = float(relation["ego_s"])
    main_s = float(relation["main_s"])
    _, primary_s, primary_d = project_point_to_path(
        np.array([primary.x, primary.y], dtype=float),
        relation_line,
    )
    _, line_heading = get_path_point_at_s(
        relation_line,
        _clamp(float(primary_s), 0.0, float(relation_cumulative[-1])),
        relation_cumulative,
    )
    # 将主场景 GVT 速度分解到伴行路线横向，外推其短时是否会进入该路线。
    heading_error = normalize_angle(float(primary.heading) - float(line_heading))
    primary_lateral_v = float(primary.v) * float(np.sin(heading_error))
    predicted_d = float(primary_d) + primary_lateral_v * float(CFG["primary_prediction_horizon_s"])
    moving_toward_line = bool(float(primary_d) * primary_lateral_v < -0.05)
    currently_on_line = abs(float(primary_d)) <= float(CFG["primary_lane_capture_d_m"])
    predicted_on_line = bool(
        moving_toward_line
        and abs(float(predicted_d)) <= float(CFG["primary_lane_capture_d_m"])
    )
    # cut-in/cut-out 在车身进入目标线前就开始释放空间，给换道过程留出余量；
    # 其他场景只有当前或预测进入伴行路线时才触发。
    proactive_approach = bool(
        scene_type in {"cut_in", "cutin", "cut_out", "cutout"}
        and abs(float(primary_d)) <= float(CFG["primary_lane_approach_d_m"])
    )
    longitudinal_distance = abs(float(primary_s) - float(ego_s))
    longitudinal_near = longitudinal_distance <= float(CFG["primary_interaction_distance_m"])
    may_use_formation_line = bool(
        longitudinal_near
        and (currently_on_line or predicted_on_line or proactive_approach)
    )

    primary_length = float(primary.length if primary.length > 0.0 else 4.5)
    # 中心距要求 = 两车半车长之和 + 配置净间距。
    required_center_gap = (
        0.5 * (float(_read_vehicle_limits(context).ego_length) + primary_length)
        + float(CFG["primary_min_net_gap_m"])
    )
    primary_rel_s_to_main = float(primary_s) - float(main_s)
    # 已在主场景 GVT 前方则继续前移，已在其后方则继续后撤；
    # 不跨越主场景车辆寻找另一侧空间，避免两车轨迹在释放过程中交叉。
    ego_ahead = float(ego_s) >= float(primary_s)
    if ego_ahead:
        release_direction = "forward"
        release_target_rel_s = max(
            float(CFG["rel_s_target_m"]),
            primary_rel_s_to_main + required_center_gap,
        )
        current_center_gap = float(ego_s) - float(primary_s)
        reference_v = max(float(main.v), float(primary.v))
    else:
        release_direction = "backward"
        release_target_rel_s = min(
            float(CFG["rel_s_target_m"]),
            primary_rel_s_to_main - required_center_gap,
        )
        current_center_gap = float(primary_s) - float(ego_s)
        reference_v = min(float(main.v), float(primary.v))

    # 用所需中心距扩张原伴行窗口。主场景 GVT 仍落在扩张区内时，
    # 即使切入动作已经结束，也继续 HOLD_RELEASED_GAP，不立即恢复。
    expanded_min = float(CFG["rel_s_min_m"]) - required_center_gap
    expanded_max = float(CFG["rel_s_max_m"]) + required_center_gap
    original_slot_occupied = bool(
        may_use_formation_line
        and expanded_min <= primary_rel_s_to_main <= expanded_max
    )
    spacing_conflict = bool(
        may_use_formation_line
        and current_center_gap < required_center_gap + float(CFG["primary_gap_buffer_m"])
    )
    needs_release = bool(original_slot_occupied or spacing_conflict)
    space_satisfied = bool(current_center_gap >= required_center_gap)
    return {
        "enabled": True,
        "supported": True,
        "primary_valid": True,
        "primary_ptc_id": str(primary.ptc_id),
        "primary_scene_type": scene_type,
        "primary_s_on_formation_line_m": float(primary_s),
        "primary_d_on_formation_line_m": float(primary_d),
        "primary_rel_s_to_main_m": float(primary_rel_s_to_main),
        "primary_v_mps": float(primary.v),
        "primary_heading_error_deg": float(np.degrees(heading_error)),
        "primary_lateral_v_mps": float(primary_lateral_v),
        "primary_predicted_d_m": float(predicted_d),
        "currently_on_formation_line": bool(currently_on_line),
        "predicted_on_formation_line": bool(predicted_on_line),
        "proactive_approach": bool(proactive_approach),
        "may_use_formation_line": bool(may_use_formation_line),
        "longitudinal_distance_to_ego_m": float(longitudinal_distance),
        "required_center_gap_m": float(required_center_gap),
        "current_center_gap_m": float(current_center_gap),
        "original_slot_occupied": bool(original_slot_occupied),
        "spacing_conflict": bool(spacing_conflict),
        "needs_release": bool(needs_release),
        "space_satisfied": bool(space_satisfied),
        "release_direction": release_direction,
        "release_target_rel_s_m": float(release_target_rel_s),
        "release_reference_v_mps": float(reference_v),
    }


def _evaluate_hard_collision(
    context: SceneContext,
    limits: VehicleLimits,
) -> dict[str, Any]:
    """检查 GVT 与当前输入参与者是否已经发生车体硬重叠。

    这里故意不实现面向所有障碍车的预测避障：伴行车首先服务主测试场景，
    普通空间冲突由主场景协同或平台场景配置处理。只有当前车体包络已出现
    纵向和横向同时重叠时，才触发 ``FAIL_SAFE`` 紧急停车。
    """

    ego = context.ego_vehicle
    actors: list[tuple[str, float, float, float, float]] = []
    if context.main_vehicle is not None:
        actors.append(
            (
                "main",
                float(context.main_vehicle.x),
                float(context.main_vehicle.y),
                float(limits.main_length),
                float(limits.main_width),
            )
        )
    frame = context.planning_frame
    if frame is not None:
        for obstacle in frame.obstacles:
            actor = build_planning_actor_state(obstacle)
            if actor is None:
                continue
            actors.append(
                (
                    str(actor.ptc_id),
                    float(actor.x),
                    float(actor.y),
                    float(actor.length if actor.length > 0.0 else 4.5),
                    float(actor.width if actor.width > 0.0 else 1.8),
                )
            )

    # 在 GVT 车体坐标系内做轴对齐包络重叠检查，保留 0.2 m 数值裕量。
    forward = np.array([np.cos(float(ego.heading)), np.sin(float(ego.heading))], dtype=float)
    left = np.array([-forward[1], forward[0]], dtype=float)
    worst_longitudinal_overlap = 0.0
    worst_lateral_overlap = 0.0
    collision_actor_id = ""
    # VUT 可能同时出现在 main_vehicle 和 obstacles 中，按位置去重避免重复统计。
    seen: set[tuple[float, float]] = set()
    for actor_id, actor_x, actor_y, actor_length, actor_width in actors:
        key = (round(actor_x, 3), round(actor_y, 3))
        if key in seen:
            continue
        seen.add(key)
        rel_xy = np.array([actor_x - float(ego.x), actor_y - float(ego.y)], dtype=float)
        rel_longitudinal = abs(float(np.dot(rel_xy, forward)))
        rel_lateral = abs(float(np.dot(rel_xy, left)))
        longitudinal_overlap = (
            0.5 * (float(limits.ego_length) + actor_length) + 0.2 - rel_longitudinal
        )
        lateral_overlap = (
            0.5 * (float(limits.ego_width) + actor_width) + 0.2 - rel_lateral
        )
        if longitudinal_overlap > 0.0 and lateral_overlap > 0.0:
            collision_actor_id = actor_id
            worst_longitudinal_overlap = max(worst_longitudinal_overlap, longitudinal_overlap)
            worst_lateral_overlap = max(worst_lateral_overlap, lateral_overlap)
    return {
        "hard_collision_risk": bool(collision_actor_id),
        "hard_collision_actor_id": str(collision_actor_id),
        "longitudinal_overlap_m": float(worst_longitudinal_overlap),
        "lateral_overlap_m": float(worst_lateral_overlap),
    }


# ==================== Lane / Actor Helpers ====================


def _select_formation_line(context: SceneContext) -> ReferenceLineView | None:
    """选择伴行路线。

    第一版要求场景配置把 GVT 起点和路线预设在目标方位对应车道，因此只锁定
    GVT 当前主参考线，不主动选择相邻线换道，也不追随 VUT 的换道路线。
    """

    if context.main_vehicle is None:
        return None
    return context.ego_vehicle.current_ref_line


def _vehicle_route_lane_ids(
    vehicle: Any | None,
    *,
    fallback: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """读取车辆锁定的全局 lane 序列，缺失时退回局部参考线序列。"""

    if vehicle is not None:
        route_intent = getattr(vehicle, "route_intent", None)
        planned_path = _normalize_lane_sequence(
            getattr(route_intent, "planned_path", None)
        )
        if planned_path:
            return planned_path
    return _normalize_lane_sequence(fallback)


def _locked_or_current_route_lane_ids(
    memory: dict[str, Any],
    memory_key: str,
    vehicle: Any,
    fallback: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """优先使用任务开始时锁定的路线，避免运行中候选走廊切换。"""

    locked = _normalize_lane_sequence(memory.get(memory_key, []))
    if locked:
        return locked
    return _vehicle_route_lane_ids(vehicle, fallback=fallback)


def _route_corridor_candidate(vehicle: Any | None, source: str) -> tuple[str, np.ndarray] | None:
    """读取 Provider 提供的未按当前位置截断的完整路线走廊。"""

    if vehicle is None:
        return None
    route_intent = getattr(vehicle, "route_intent", None)
    if route_intent is None or not bool(getattr(route_intent, "valid", False)):
        return None
    corridor = getattr(route_intent, "corridor_line", None)
    line = None if corridor is None else np.asarray(getattr(corridor, "line", []), dtype=float)
    if line is None or line.ndim != 2 or len(line) < 2:
        return None
    return source, np.asarray(line[:, :2], dtype=float)


def _resolve_relation_geometry(
    context: SceneContext,
    target_line: ReferenceLineView,
    memory: dict[str, Any],
    *,
    update_memory: bool,
) -> dict[str, Any]:
    """选择覆盖 GVT/VUT 的稳定公共走廊，并计算连续 Frenet 关系。"""

    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        raise ValueError("companion relation geometry requires main vehicle")

    candidates: list[tuple[str, np.ndarray]] = []
    for candidate in (
        _route_corridor_candidate(ego, "ego_route_corridor"),
        _route_corridor_candidate(main, "main_route_corridor"),
    ):
        if candidate is not None:
            candidates.append(candidate)
    local_line, _ = _build_relation_line(context, target_line)
    candidates.append(("local_relation_fallback", local_line))

    last_rel_s_raw = memory.get("last_valid_rel_s_m")
    last_rel_s = None if last_rel_s_raw is None else float(last_rel_s_raw)
    rel_v = float(ego.v) - float(main.v)
    expected_rel_s = (
        0.0
        if last_rel_s is None
        else float(last_rel_s) + rel_v * max(float(context.dt), 0.0)
    )
    projection_limit = float(CFG["relation_projection_max_d_m"])
    evaluated: list[dict[str, Any]] = []
    for priority, (source, line) in enumerate(candidates):
        line_xy = np.asarray(line, dtype=float)[:, :2]
        cumulative = calculate_path_cumulative_distance(line_xy)
        _, ego_s, ego_d = project_point_to_path(
            np.array([ego.x, ego.y], dtype=float),
            line_xy,
        )
        _, main_s, main_d = project_point_to_path(
            np.array([main.x, main.y], dtype=float),
            line_xy,
        )
        raw_rel_s = float(ego_s) - float(main_s)
        candidate_expected = raw_rel_s if last_rel_s is None else expected_rel_s
        jump = abs(raw_rel_s - candidate_expected)
        max_projection_d = max(abs(float(ego_d)), abs(float(main_d)))
        projection_in_range = bool(max_projection_d <= projection_limit)
        score = (
            max_projection_d
            + 0.25 * jump
            + 1e-3 * float(priority)
            + (0.0 if projection_in_range else 1000.0)
        )
        evaluated.append(
            {
                "source": source,
                "line": line_xy,
                "cumulative_s": cumulative,
                "ego_s": float(ego_s),
                "main_s": float(main_s),
                "ego_d": float(ego_d),
                "main_d": float(main_d),
                "raw_rel_s": float(raw_rel_s),
                "jump_m": float(jump),
                "projection_in_range": projection_in_range,
                "score": float(score),
            }
        )

    selected = min(evaluated, key=lambda item: float(item["score"]))
    max_jump = float(CFG["relation_max_jump_m"])
    projection_valid = bool(
        selected["projection_in_range"]
        and (last_rel_s is None or float(selected["jump_m"]) <= max_jump)
    )
    invalid_hits = int(memory.get("relation_invalid_hits", 0))
    if update_memory:
        if projection_valid:
            invalid_hits = 0
            memory["last_valid_rel_s_m"] = float(selected["raw_rel_s"])
        else:
            invalid_hits += 1
        memory["relation_invalid_hits"] = invalid_hits

    relation_usable = bool(
        projection_valid
        or (
            last_rel_s is not None
            and invalid_hits < int(CFG["slot_exit_frames"])
        )
    )
    resolved_rel_s = (
        float(selected["raw_rel_s"])
        if projection_valid or last_rel_s is None
        else float(expected_rel_s)
    )
    selected.update(
        {
            "projection_valid": projection_valid,
            "relation_usable": relation_usable,
            "rel_s": float(resolved_rel_s),
            "expected_rel_s": float(
                selected["raw_rel_s"] if last_rel_s is None else expected_rel_s
            ),
            "invalid_hits": int(invalid_hits),
        }
    )
    return selected


def _resolve_formation_line(
    context: SceneContext,
    memory: dict[str, Any],
) -> ReferenceLineView | None:
    """从当前候选参考线中恢复先前锁定的伴行路线。

    优先匹配完整车道序列，其次匹配车道 ID 和参考线下标；都无法恢复时才
    回退到当前路线。这样可降低候选线重排导致的伴行目标跳线。
    """

    candidates = context.ego_vehicle.candidate_ref_lines
    line = _line_by_lane_sequence(candidates, memory.get("locked_lane_sequence", []))
    if line is not None:
        return line
    line = _line_by_lane_id(candidates, str(memory.get("locked_lane_id", "")))
    if line is not None:
        return line
    line = _line_by_index(candidates, memory.get("locked_line_index"))
    if line is not None:
        return line
    return _select_formation_line(context)


def _build_relation_line(
    context: SceneContext,
    target_line: ReferenceLineView,
) -> tuple[np.ndarray, np.ndarray]:
    """构造同时覆盖 GVT 前后关系的公共投影线。

    ``current_ref_line`` 往往只覆盖车辆前方，无法稳定投影位于车辆后方的 VUT。
    因此先使用完整当前车道中心线，再拼接参考线中的前向后继段，供所有车辆
    计算统一的 Frenet ``s/d``。
    """

    lane_centerline = context.ego_vehicle.lane_centerline
    if lane_centerline is None:
        line = np.asarray(target_line.line, dtype=float)[:, :2]
        return line, calculate_path_cumulative_distance(line)

    base_line = np.asarray(lane_centerline, dtype=float)
    if base_line.ndim != 2 or len(base_line) < 2:
        line = np.asarray(target_line.line, dtype=float)[:, :2]
        return line, calculate_path_cumulative_distance(line)
    line = np.asarray(base_line[:, :2], dtype=float)

    # 从与当前车道终点最近的参考线点开始拼接，重复连接点只保留一次。
    forward_line = np.asarray(target_line.line, dtype=float)
    if forward_line.ndim == 2 and len(forward_line) >= 2:
        distances = np.linalg.norm(forward_line[:, :2] - line[-1, :2], axis=1)
        append_start_idx = int(np.argmin(distances))
        if float(distances[append_start_idx]) <= float(CFG["lane_center_tolerance_m"]):
            successor = np.asarray(forward_line[append_start_idx:, :2], dtype=float)
            if len(successor) > 0 and np.linalg.norm(successor[0] - line[-1]) < 1e-6:
                successor = successor[1:]
            if len(successor) > 0:
                line = np.vstack([line, successor])
    return line, calculate_path_cumulative_distance(line)


def _find_actor_by_ptc_id(
    context: SceneContext,
    ptc_id: str,
) -> PlanningActorState | None:
    """在规划帧障碍物中按 OBU 参与者 ID 查找主场景 GVT。"""

    frame = context.planning_frame
    if frame is None:
        return None
    for obstacle in frame.obstacles:
        actor = build_planning_actor_state(obstacle)
        if actor is not None and str(actor.ptc_id) == str(ptc_id):
            return actor
    return None


def _line_by_index(
    candidate_lines: list[ReferenceLineView],
    line_index: int | None,
) -> ReferenceLineView | None:
    """按参考线下标查找候选参考线。"""

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
    """按当前车道 ID 查找候选参考线。"""

    if not lane_id:
        return None
    for line in candidate_lines:
        if str(line.lane_id) == str(lane_id):
            return line
    return None


def _line_by_lane_sequence(
    candidate_lines: list[ReferenceLineView],
    lane_sequence: list[str] | tuple[str, ...] | None,
) -> ReferenceLineView | None:
    """按车道序列重叠数量查找与锁定路线最接近的候选参考线。"""

    target = _normalize_lane_sequence(lane_sequence)
    if not target:
        return None
    target_set = set(target)
    best_line = None
    best_overlap = 0
    for line in candidate_lines:
        overlap = sum(
            1
            for lane_id in _normalize_lane_sequence(line.lane_sequence)
            if lane_id in target_set
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_line = line
    return best_line


# ==================== Conditions / 通用工具 ====================


def _validate_config() -> str:
    """校验伴行方位、纵向窗口及任务时长的静态一致性。

    返回空字符串表示配置有效；错误只触发 ``DEGRADED_EXIT``，不触发停车。
    场景车辆组合是否合理仍由虚拟域控推演和人工配置确认。
    """

    position = _normalized_position()
    if position not in POSITION_SPECS:
        return f"invalid_companion_position:{position}"
    rel_min = float(CFG["rel_s_min_m"])
    rel_target = float(CFG["rel_s_target_m"])
    rel_max = float(CFG["rel_s_max_m"])
    if not rel_min <= rel_target <= rel_max:
        return "invalid_rel_s_window"
    relation = POSITION_SPECS[position].longitudinal_relation
    if relation == "front" and rel_target <= 0.0:
        return "front_position_requires_positive_rel_s"
    if relation == "rear" and rel_target >= 0.0:
        return "rear_position_requires_negative_rel_s"
    if float(CFG["companion_duration_s"]) < 0.0:
        return "negative_companion_duration"
    return ""


def _is_slot_candidate(metrics: dict[str, Any]) -> bool:
    """判断是否具备执行伴行纵向控制的基础车道和方位几何。

    这里只要求 ``rel_s`` 的前后方向正确，不要求已经进入目标窗口；
    窗口外但几何有效的车辆可在 ``ACQUIRE_SLOT`` 中继续收敛。
    """

    return bool(
        metrics.get("main_valid", False)
        and metrics.get("target_line_valid", False)
        and metrics.get("position_valid", False)
        and metrics.get("relation_usable", False)
        and metrics.get("route_membership_ok", False)
        and metrics.get("same_direction", False)
        and metrics.get("ego_centered", False)
        and metrics.get("main_centered", False)
        and metrics.get("lane_relation_ok", False)
        and metrics.get("rel_s_direction_ok", False)
    )


def _rel_s_direction_ok(spec: PositionSpec, rel_s: float) -> bool:
    """检查 ``rel_s`` 符号是否符合前方、后方或侧方语义。"""

    if spec.longitudinal_relation == "front":
        return rel_s > 0.0
    if spec.longitudinal_relation == "rear":
        return rel_s < 0.0
    return True


def _mission_duration_reached(memory: dict[str, Any], now: float) -> bool:
    """判断正数有限伴行时长是否已到；时长为 0 时始终返回 False。"""

    duration = float(CFG["companion_duration_s"])
    start = float(memory.get("mission_start_time_s", -1.0))
    return bool(duration > 0.0 and start >= 0.0 and float(now) - start >= duration)


def _read_vehicle_limits(context: SceneContext) -> VehicleLimits:
    """从运行帧读取车辆尺寸及速度、加速度约束，并提供平台默认值。"""

    runtime_frame = context.runtime_frame
    config = {} if runtime_frame is None else runtime_frame.config
    size = config.get("size", {})
    constrain = config.get("constrain", {})
    ego_length = float(size.get("L", 4.5))
    ego_width = float(size.get("W", 1.8))
    return VehicleLimits(
        ego_length=ego_length,
        ego_width=ego_width,
        main_length=ego_length,
        main_width=ego_width,
        max_v=float(constrain.get("max_v", 6.94)),
        min_acc=float(constrain.get("min_acc", -2.0)),
        max_acc=float(constrain.get("max_acc", 2.0)),
    )


def _normalized_position() -> str:
    """读取并规范化八方位配置字符串。"""

    return str(CFG.get("position", "front")).strip().lower()


def _normalize_lane_sequence(
    lane_sequence: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """把参考线车道序列统一转换为字符串列表。"""

    if lane_sequence is None:
        return []
    return [str(lane_id) for lane_id in lane_sequence]


def _signed_left_lateral_offset(main: Any, ego: Any) -> float:
    """计算 GVT 相对 VUT 的有符号横向距离，左侧为正、右侧为负。"""

    rel_xy = np.array([float(ego.x) - float(main.x), float(ego.y) - float(main.y)], dtype=float)
    forward = np.array([np.cos(float(main.heading)), np.sin(float(main.heading))], dtype=float)
    left = np.array([-forward[1], forward[0]], dtype=float)
    return float(np.dot(rel_xy, left))


def _heading_diff_deg(h1: float, h2: float) -> float:
    """计算两个航向角的最小绝对夹角，单位为度。"""

    return float(np.degrees(abs(normalize_angle(float(h1) - float(h2)))))


def _clamp(value: float, lower: float, upper: float) -> float:
    """把数值裁剪到指定闭区间。"""

    return float(min(max(value, lower), upper))


def _annotate_output(
    output: dict[str, Any],
    scene_status: str,
    scene_detail: str | None = None,
    scene_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """给轨迹目标附加统一的场景状态、详情和调试指标。"""

    output["scene_status"] = str(scene_status)
    if scene_detail is not None:
        output["scene_detail"] = str(scene_detail)
    if scene_debug is not None:
        output["scene_debug"] = copy.deepcopy(scene_debug)
    return output
