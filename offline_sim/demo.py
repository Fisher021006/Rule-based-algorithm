"""可由用户直接修改的场景算法入口。

用户算法只需要返回 6 个主体参数：
``(x, y, v, a, theta, t)``。

若下游需要 ``SceneAlgorithmOutput``，由框架层统一补齐默认元信息。
"""

from __future__ import annotations

from planning_runtime.scene.types import SceneContext
from traj_planner.tools import get_path_point_at_s


def scene_algorithm(context: SceneContext) -> tuple[float, float, float, float, float, float]:
    """默认示例场景算法。

    返回：
        ``(x, y, v, a, theta, t)``
    """

    ego = context.ego_vehicle
    main = context.main_vehicle
    target_line = ego.current_ref_line
    target_v = min(max(ego.v + 0.5, 3.0), 6.5)
    plan_t = 2.0
    reason = "demo_keep_lane"

    if (
        main.lane_id == ego.lane_id
        and 0.0 < main.relative_s_to_ego < 20.0
        and main.v < ego.v + 0.3
    ):
        side_candidates = [line for line in ego.candidate_ref_lines if line.line_index != 0]
        if side_candidates:
            target_line = min(side_candidates, key=lambda line: abs(line.lateral_step))
            plan_t = 3.0
            target_v = min(max(main.v + 1.0, ego.v), 6.5)
            reason = "demo_change_lane_for_main_vehicle"
        else:
            target_v = max(main.v, 1.0)
            reason = "demo_follow_main_vehicle"

    travel_s = max(12.0, target_v * plan_t + 4.0)
    target_s = min(target_line.current_s + travel_s, float(target_line.cumulative_s[-1]) - 0.5)
    target_xy, target_heading = get_path_point_at_s(target_line.line, target_s, target_line.cumulative_s)

    return (
        float(target_xy[0]),
        float(target_xy[1]),
        float(target_v),
        0.0,
        float(target_heading),
        float(plan_t),
    )
