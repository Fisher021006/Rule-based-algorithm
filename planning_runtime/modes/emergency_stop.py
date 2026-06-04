"""`emergency_stop` 模式的安全停车轨迹适配。

该模块用于场景算法明确请求安全停车时绕过普通局部规划器：
场景算法负责判断触发时机，运行时负责按制动约束生成停车轨迹。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from traj_planner.context import build_planning_context
from traj_planner.tools import get_path_point_at_s
from traj_planner.trajectory import calculate_trajectory_kappa
from traj_planner.types import PlanningInput


EMERGENCY_STOP_MODE = "emergency_stop"
STOP_TARGET_V_EPS = 0.05
DEFAULT_DT = 0.1


def is_emergency_stop_input(planning_input: PlanningInput) -> bool:
    """判断当前外部目标是否请求安全停车模式。"""

    external_target = getattr(planning_input, "external_target", None)
    if external_target is None:
        return False
    metadata = getattr(external_target, "metadata", {}) or {}
    return str(metadata.get("control_mode", "")).strip().lower() == EMERGENCY_STOP_MODE


def build_emergency_stop_trajectory(planning_input: PlanningInput) -> np.ndarray | None:
    """按 `emergency_stop` 外部目标生成一条安全停车轨迹。"""

    external_target = getattr(planning_input, "external_target", None)
    if external_target is None:
        return None

    context = build_planning_context(planning_input)
    target_line_index = int(getattr(external_target, "target_line_index", 0))
    if target_line_index < 0 or target_line_index >= len(context.line_contexts):
        return None

    target_line = context.line_contexts[target_line_index]
    metadata: dict[str, Any] = dict(getattr(external_target, "metadata", {}) or {})
    ego = context.ego_state
    ego_v = max(0.0, float(ego.v))
    min_acc = float(context.config["constrain"].get("min_acc", -3.0))
    requested_decel = abs(
        float(
            metadata.get(
                "emergency_decel_mps2",
                abs(float(getattr(external_target, "target_a", 0.0))) or abs(min_acc),
            )
        )
    )
    requested_decel = max(requested_decel, 1e-6)
    brake_a = float(np.clip(-requested_decel, min_acc, -1e-6))

    if ego_v <= STOP_TARGET_V_EPS:
        stop_t = 0.0
        stop_distance = 0.0
        brake_a = 0.0
    else:
        stop_t = ego_v / abs(brake_a)
        stop_distance = ego_v * stop_t + 0.5 * brake_a * stop_t * stop_t

    dt = float(context.config.get("local_fsm", {}).get("planner_cycle_dt", DEFAULT_DT))
    dt = max(dt, 1e-3)
    plan_t = max(float(getattr(external_target, "plan_T", 0.0)), stop_t, dt)
    total_t = max(dt, np.ceil(plan_t / dt) * dt)
    steps = int(round(total_t / dt))
    current_s = float(target_line.current_s)
    line_end_s = float(target_line.cumulative_s[-1])

    trajectory = []
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

        if idx == 0:
            x = float(ego.x)
            y = float(ego.y)
            heading = float(ego.heading)
        else:
            pos, heading = get_path_point_at_s(
                target_line.line,
                min(line_end_s, current_s + travel_s),
                target_line.cumulative_s,
            )
            x = float(pos[0])
            y = float(pos[1])
            heading = float(heading)
        trajectory.append([x, y, heading, float(v), float(a)])

    return np.asarray(calculate_trajectory_kappa(trajectory), dtype=float)
