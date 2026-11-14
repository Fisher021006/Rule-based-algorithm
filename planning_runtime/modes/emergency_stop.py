"""`emergency_stop` 模式的安全停车轨迹适配。

该模块用于场景算法明确请求安全停车时绕过普通局部规划器：
场景算法负责判断触发时机，运行时负责按制动约束生成停车轨迹。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from traj_planner.context import build_planning_context
from traj_planner.generator import build_candidate_smooth_path
from traj_planner.trajectory import (
    PROFILE_EQUALITY_TOLERANCE, attach_path_curvature,
    calculate_path_cumulative_distance, interp_pose_on_path,
)
from traj_planner.types import PlanningInput, TerminalCandidate


EMERGENCY_STOP_MODE = "emergency_stop"
ZERO_SPEED_EPS = 1e-9
DEFAULT_DT = 0.1
MIN_GEOMETRY_LOOKAHEAD_M = 2.0


def is_emergency_stop_input(planning_input: PlanningInput) -> bool:
    """判断当前外部目标是否请求安全停车模式。"""

    external_target = getattr(planning_input, "external_target", None)
    if external_target is None:
        return False
    metadata = getattr(external_target, "metadata", {}) or {}
    return str(metadata.get("control_mode", "")).strip().lower() == EMERGENCY_STOP_MODE


def build_emergency_stop_trajectory(
    planning_input: PlanningInput, *, diagnostics: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """按 `emergency_stop` 外部目标生成一条安全停车轨迹。"""

    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.clear()
    external_target = getattr(planning_input, "external_target", None)
    if external_target is None:
        diagnostics["failure_reason"] = "emergency_stop_missing_target"
        return None

    context = build_planning_context(planning_input)
    target_line_index = int(getattr(external_target, "target_line_index", 0))
    if target_line_index < 0 or target_line_index >= len(context.line_contexts):
        diagnostics["failure_reason"] = "emergency_stop_invalid_target_line"
        return None

    target_line = context.line_contexts[target_line_index]
    metadata: dict[str, Any] = dict(getattr(external_target, "metadata", {}) or {})
    ego = context.ego_state
    ego_v = float(ego.v)
    if not np.isfinite(ego_v) or ego_v < 0.0:
        diagnostics["failure_reason"] = "emergency_stop_invalid_initial_speed"
        return None
    min_acc = float(context.config["constrain"].get("min_acc", -3.0))
    requested_decel = abs(
        float(
            metadata.get(
                "emergency_decel_mps2",
                abs(float(getattr(external_target, "target_a", 0.0))) or abs(min_acc),
            )
        )
    )
    if ego_v <= ZERO_SPEED_EPS:
        stop_t = 0.0
        stop_distance = 0.0
        brake_a = 0.0
    else:
        if not np.isfinite([min_acc, requested_decel]).all() or min_acc >= 0.0 or requested_decel <= 0.0:
            diagnostics["failure_reason"] = "emergency_stop_invalid_deceleration"
            return None
        brake_a = float(max(-requested_decel, min_acc))
        stop_t = ego_v / abs(brake_a)
        stop_distance = ego_v * stop_t + 0.5 * brake_a * stop_t * stop_t

    current_s = float(target_line.current_s)
    line_end_s = float(target_line.cumulative_s[-1])
    available_s = max(0.0, line_end_s - current_s)
    diagnostics.update(
        required_stop_distance_m=float(stop_distance), available_reference_distance_m=available_s,
        requested_deceleration_mps2=float(requested_decel), applied_acceleration_mps2=brake_a,
        physical_road_end_verified=False,
    )
    # Reference shortage is not itself proof that the road/configuration is impossible.
    # It must never be disguised by clamping position while retaining positive speed.
    if stop_distance > available_s + 1e-9:
        diagnostics["failure_reason"] = "emergency_stop_insufficient_reference_distance"
        return None

    dt = float(context.config.get("local_fsm", {}).get("planner_cycle_dt", DEFAULT_DT))
    dt = max(dt, 1e-3)
    plan_t = max(float(getattr(external_target, "plan_T", 0.0)), stop_t, dt)
    total_t = max(dt, np.ceil(plan_t / dt) * dt)
    steps = int(round(total_t / dt))

    stop_path = None
    path_distance = None
    if stop_distance > 0.0:
        # Reuse the normal geometric path constructor so an off-centre vehicle does
        # not jump onto the centreline. The lookahead shapes geometry only: the stop
        # still travels exactly stop_distance, never the lookahead distance.
        geometry_length = min(available_s, stop_distance + MIN_GEOMETRY_LOOKAHEAD_M)
        terminal = TerminalCandidate(target_line_index, target_line.line_id,
            current_s + geometry_length, 0.0, plan_t, 0.0, "emergency_stop")
        candidate = build_candidate_smooth_path(context, terminal)
        stop_path = candidate.smooth_path
        if stop_path is None or len(stop_path) < 2 or not np.isfinite(stop_path).all():
            diagnostics["failure_reason"] = "emergency_stop_invalid_path"
            return None
        path_distance = calculate_path_cumulative_distance(stop_path)
        diagnostics["geometric_path_length_m"] = float(path_distance[-1])
        if stop_distance > float(path_distance[-1]) + PROFILE_EQUALITY_TOLERANCE:
            diagnostics["failure_reason"] = "emergency_stop_insufficient_path_length"
            return None
        diagnostics["terminal_distance_residual_m"] = max(0.0, stop_distance - float(path_distance[-1]))

    trajectory = []
    travel_s_values = []
    for idx in range(steps + 1):
        t = min(float(idx) * dt, total_t)
        if brake_a < -1e-6 and t < stop_t:
            v = max(0.0, ego_v + brake_a * t)
            a = brake_a
            travel_s = max(0.0, ego_v * t + 0.5 * brake_a * t * t)
        else:
            v = 0.0
            a = 0.0
            travel_s = max(0.0, stop_distance)

        if stop_path is not None and travel_s > float(path_distance[-1]) + 1e-9 and v > ZERO_SPEED_EPS:
            diagnostics["failure_reason"] = "emergency_stop_insufficient_path_length"
            return None
        if idx == 0 or stop_distance == 0.0:
            x = float(ego.x)
            y = float(ego.y)
            heading = float(ego.heading)
        else:
            x, y, heading = interp_pose_on_path(stop_path, path_distance, travel_s)
        trajectory.append([x, y, heading, float(v), float(a)])
        travel_s_values.append(float(travel_s))

    diagnostics["failure_reason"] = ""
    if stop_path is None:
        # 没有几何路径（stop_distance 为 0），不存在可估计的曲率
        traj = np.asarray(trajectory, dtype=float)
        return np.column_stack([traj, np.zeros(len(traj))])
    # 曲率按停车路径几何取，不从事后重采样的轨迹点反推
    return attach_path_curvature(trajectory, stop_path, path_distance, travel_s_values)
