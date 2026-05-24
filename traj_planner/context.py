"""构建单帧规划上下文。

这个模块是“运行时原始输入”和“规划器内部推理”之间的边界层。
它把以下输入：
- 原始自车状态
- 原始障碍物
- 来自 ``drivable_lanes_planner`` 的参考线
- 缓存的全局路线信息

统一转换成标准化的 :class:`PlanningContext`。

上游：
    ``simulator/vehicle.GVT.traj_planning`` 先构造 ``PlanningInput``。
下游：
    ``behavior.py``、``sampling.py``、``evaluator.py`` 使用这里产出的
    ``PlanningContext``。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from traj_planner.state_rules import is_lane_change_centering_incomplete
from traj_planner.tools import calculate_path_cumulative_distance, project_point_to_path
from traj_planner.types import (
    LineContext,
    ObstacleSnapshot,
    PlanningContext,
    PlanningInput,
    ProjectedObstacle,
    RouteIntent,
)


def _normalize_obstacle(obstacle: Any) -> ObstacleSnapshot | None:
    """把一个“障碍物样对象”转换成 :class:`ObstacleSnapshot`。

    输入要求：
        任意对象，只要至少暴露 ``x``、``y``、``v`` 字段。
        若存在 ``heading``、``length`` 等字段会一并读取。

    输出：
        全局笛卡尔坐标系下的 ``ObstacleSnapshot``。
    """

    if obstacle is None:
        return None

    def read(name: str, default: float = 0.0) -> float:
        value = getattr(obstacle, name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    ptc_id = str(getattr(obstacle, "ptc_id", getattr(obstacle, "ptcId", "")) or "")
    if not ptc_id:
        ptc_id = f"obs_{id(obstacle)}"

    return ObstacleSnapshot(
        ptc_id=ptc_id,
        x=read("x"),
        y=read("y"),
        v=max(0.0, read("v")),
        heading=read("heading"),
        a=read("a"),
        length=max(0.0, read("length")),
        width=max(0.0, read("width")),
        height=max(0.0, read("height")),
        timestamp_ms=int(getattr(obstacle, "timestamp_ms", 0) or 0),
        raw=obstacle,
    )


def _project_front_rear_obstacles(
    ego_pos: np.ndarray,
    ego_v: float,
    line: np.ndarray,
    current_s: float,
    obstacles: list[ObstacleSnapshot],
    lateral_threshold: float = 2.5,
) -> tuple[ProjectedObstacle | None, ProjectedObstacle | None]:
    """在一条参考线上找最近的前车和后车。

    输入：
        - ``ego_pos``：自车笛卡尔坐标
        - ``current_s``：自车在该线上的投影纵向坐标
        - ``obstacles``：已标准化的障碍物快照

    输出：
        两个投影障碍物：
        - 最近前车
        - 最近后车

    过滤规则：
        横向距离超过 ``lateral_threshold`` 的障碍物不算这条线上的前/后车。
    """

    front_best: ProjectedObstacle | None = None
    rear_best: ProjectedObstacle | None = None

    for obstacle in obstacles:
        obs_pos = np.array([obstacle.x, obstacle.y], dtype=float)
        _, s_obs, d_obs = project_point_to_path(obs_pos, line)
        if s_obs < 0:
            continue
        if abs(d_obs) > lateral_threshold:
            continue

        if s_obs >= current_s:
            gap = float(s_obs - current_s)
            projected = ProjectedObstacle(
                obstacle=obstacle,
                s=float(s_obs),
                d=float(d_obs),
                gap=gap,
                relative_speed=float(ego_v - obstacle.v),
            )
            if front_best is None or gap < front_best.gap:
                front_best = projected
        else:
            gap = float(current_s - s_obs)
            projected = ProjectedObstacle(
                obstacle=obstacle,
                s=float(s_obs),
                d=float(d_obs),
                gap=gap,
                relative_speed=float(obstacle.v - ego_v),
            )
            if rear_best is None or gap < rear_best.gap:
                rear_best = projected

    return front_best, rear_best


def _build_line_contexts(planning_input: PlanningInput, obstacle_snapshots: list[ObstacleSnapshot]) -> list[LineContext]:
    """为每一条有效参考线构建 :class:`LineContext`。

    输入：
        ``GVT`` 组好的 ``PlanningInput``，以及已标准化的障碍物列表。

    输出：
        有序的 ``LineContext`` 列表。
        其中 ``line_contexts[0]`` 按约定始终是当前主参考线。
    """

    ego = planning_input.ego_state
    ego_pos = np.array([ego.x, ego.y], dtype=float)
    line_contexts: list[LineContext] = []

    for line_index, raw_line in enumerate(planning_input.refer_lines):
        line = np.asarray(raw_line, dtype=float)
        if line.ndim != 2 or len(line) < 2:
            continue
        cumulative_s = calculate_path_cumulative_distance(line)
        _, current_s, current_d = project_point_to_path(ego_pos, line)
        remaining_length = max(float(cumulative_s[-1] - current_s), 0.0)

        front_obstacle, rear_obstacle = _project_front_rear_obstacles(
            ego_pos=ego_pos,
            ego_v=float(ego.v),
            line=line,
            current_s=float(current_s),
            obstacles=obstacle_snapshots,
        )

        lane_sequence = planning_input.refer_line_lane_sequences[line_index]
        line_id = planning_input.refer_line_lane_ids[line_index]
        line_contexts.append(
            LineContext(
                line_index=line_index,
                line_id=line_id,
                lane_sequence=list(lane_sequence),
                lateral_step=int(planning_input.refer_line_lateral_steps[line_index]),
                line=line,
                cumulative_s=cumulative_s,
                current_s=float(current_s),
                current_d=float(current_d),
                remaining_length=remaining_length,
                front_obstacle=front_obstacle,
                rear_obstacle=rear_obstacle,
            )
        )

    if not line_contexts:
        raise ValueError("No valid refer lines available for planning context.")

    return line_contexts


def _extract_route_intent(planning_input: PlanningInput, line_contexts: list[LineContext]) -> RouteIntent:
    """从缓存的全局车道路径中提取本地路线意图。

    逻辑：
    1. 先确定当前 road 上全局最终想去的路线车道
    2. 如果该车道相隔多条，则把本地目标收缩成朝该方向最近的一阶相邻车道
    3. 如果收缩后的本地目标不等于当前车道，则 ``must_change=True``

    这样既保留全局路线意图，又保证“一次局部机动最多只跨一条车道”。
    """

    current_line = line_contexts[0]
    current_lane_id = current_line.line_id
    current_road_id = planning_input.current_road_id or current_lane_id.split(".")[0]
    current_road_planned_lanes = [
        lane_id for lane_id in planning_input.planned_path if lane_id.split(".")[0] == current_road_id
    ]

    if current_road_planned_lanes:
        desired_route_lane_id = current_road_planned_lanes[-1]
    else:
        desired_route_lane_id = current_lane_id

    target_lane_id = desired_route_lane_id
    target_line_index = 0
    desired_line_context = None
    for line_context in line_contexts:
        if line_context.line_id == desired_route_lane_id:
            desired_line_context = line_context
            target_line_index = line_context.line_index
            break

    if desired_line_context is not None and abs(desired_line_context.lateral_step) > 1:
        desired_sign = 1 if desired_line_context.lateral_step > 0 else -1
        adjacent_candidates = [
            line_context
            for line_context in line_contexts
            if line_context.lateral_step == desired_sign
        ]
        if adjacent_candidates:
            adjacent = adjacent_candidates[0]
            target_lane_id = adjacent.line_id
            target_line_index = adjacent.line_index

    lane_change_centering_incomplete = (
        planning_input.planner_memory.lane_change_committed
        and is_lane_change_centering_incomplete(
            current_lane_id=current_lane_id,
            target_lane_id=planning_input.planner_memory.target_lane_id,
            current_d=current_line.current_d,
            config=planning_input.config,
        )
    )
    must_change = (target_lane_id != current_lane_id and target_line_index != 0) or lane_change_centering_incomplete
    distance_to_goal = float(
        np.linalg.norm(np.asarray(planning_input.goal_xy, dtype=float) - np.array([planning_input.ego_state.x, planning_input.ego_state.y], dtype=float))
    )

    return RouteIntent(
        current_lane_id=current_lane_id,
        current_road_id=current_road_id,
        desired_route_lane_id=desired_route_lane_id,
        target_lane_id=target_lane_id,
        target_line_index=target_line_index,
        must_change=must_change,
        deadline_margin_m=float(current_line.remaining_length),
        distance_to_goal=distance_to_goal,
        current_road_planned_lanes=current_road_planned_lanes,
    )


def build_planning_context(planning_input: PlanningInput) -> PlanningContext:
    """构建单帧规划所需的完整 :class:`PlanningContext`。

    输入：
        :class:`PlanningInput`

    输出：
        :class:`PlanningContext`

    下游用途：
        这个上下文会继续传给：
        - ``behavior.decide_behavior``
        - ``sampling.build_candidate_templates``
        - ``evaluator.evaluate_candidates``
    """

    obstacle_snapshots = [
        obstacle_snapshot
        for obstacle in planning_input.obstacles
        for obstacle_snapshot in [_normalize_obstacle(obstacle)]
        if obstacle_snapshot is not None
    ]
    line_contexts = _build_line_contexts(planning_input, obstacle_snapshots)
    route_intent = _extract_route_intent(planning_input, line_contexts)

    return PlanningContext(
        ego_state=planning_input.ego_state,
        obstacle_snapshots=obstacle_snapshots,
        line_contexts=line_contexts,
        route_intent=route_intent,
        planned_path=list(planning_input.planned_path),
        planned_actions=list(planning_input.planned_actions),
        road_sequence=list(planning_input.road_sequence),
        goal_xy=np.asarray(planning_input.goal_xy, dtype=float),
        config=planning_input.config,
        planner_memory=planning_input.planner_memory,
    )
