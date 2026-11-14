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


ROUTE_LOOKAHEAD_MIN_TRIGGER_M = 18.0
ROUTE_LOOKAHEAD_SPEED_TIME_S = 3.0
ROUTE_LOOKAHEAD_BUFFER_M = 10.0
ROUTE_LOOKAHEAD_MIN_REMAIN_GAIN_M = 5.0


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
        first_lane_remaining_length = remaining_length
        if line_index < len(planning_input.refer_line_first_lane_remaining_lengths):
            first_lane_remaining_length = max(
                0.0,
                float(planning_input.refer_line_first_lane_remaining_lengths[line_index]),
            )

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
                first_lane_remaining_length=first_lane_remaining_length,
                front_obstacle=front_obstacle,
                rear_obstacle=rear_obstacle,
            )
        )

    if not line_contexts:
        raise ValueError("No valid refer lines available for planning context.")

    return line_contexts


def _road_id(lane_id: str) -> str:
    """从 lane_id 中提取 road id。"""

    return str(lane_id).split(".")[0]


def _route_lookahead_trigger_distance(planning_input: PlanningInput) -> float:
    """返回开始为未来路线换道预留窗口的距离阈值。"""

    local_fsm_cfg = planning_input.config.get("local_fsm", {})
    configured = local_fsm_cfg.get("route_lookahead_trigger_m")
    if configured is not None:
        return max(0.0, float(configured))
    return max(
        ROUTE_LOOKAHEAD_MIN_TRIGGER_M,
        ROUTE_LOOKAHEAD_SPEED_TIME_S * max(float(planning_input.ego_state.v), 0.0) + ROUTE_LOOKAHEAD_BUFFER_M,
    )


def _current_planned_index(planned_path: list[str], current_lane_id: str, current_road_id: str) -> int:
    """找到当前局部判断应从全局路径的哪个位置开始看未来。"""

    if not planned_path:
        return 0
    for idx, lane_id in enumerate(planned_path):
        if lane_id == current_lane_id:
            return idx
    for idx, lane_id in enumerate(planned_path):
        if _road_id(lane_id) == current_road_id:
            return idx
    return 0


def _line_route_progress_index(
    line_context: LineContext,
    planned_path: list[str],
    start_idx: int,
    *,
    max_initial_skip: int = 0,
) -> int:
    """返回该候选线能连续覆盖到全局路径的哪个 lane 下标。"""

    if not planned_path:
        return start_idx - 1
    lane_sequence = list(line_context.lane_sequence)
    last_start_idx = min(len(planned_path) - 1, start_idx + max(0, int(max_initial_skip)))
    for route_idx in range(start_idx, last_start_idx + 1):
        route_lane_id = planned_path[route_idx]
        try:
            sequence_idx = lane_sequence.index(route_lane_id)
        except ValueError:
            continue

        best_idx = route_idx
        for next_route_idx in range(route_idx + 1, len(planned_path)):
            next_lane_id = planned_path[next_route_idx]
            try:
                sequence_idx = lane_sequence.index(next_lane_id, sequence_idx + 1)
            except ValueError:
                break
            best_idx = next_route_idx
        return best_idx
    return start_idx - 1


def _select_route_lookahead_target(
    planning_input: PlanningInput,
    line_contexts: list[LineContext],
    current_route_progress_index: int,
    planned_start_idx: int,
) -> tuple[LineContext | None, int, str]:
    """在当前线即将无法继续覆盖全局路径时，选择一阶相邻的续接线。"""

    current_line = line_contexts[0]
    trigger_distance = _route_lookahead_trigger_distance(planning_input)
    if float(current_line.remaining_length) > trigger_distance:
        return None, current_route_progress_index, ""

    best_line: LineContext | None = None
    best_progress = current_route_progress_index
    for line_context in line_contexts[1:]:
        if abs(int(line_context.lateral_step)) != 1:
            continue
        if float(line_context.remaining_length) <= float(current_line.remaining_length) + ROUTE_LOOKAHEAD_MIN_REMAIN_GAIN_M:
            continue
        progress_idx = _line_route_progress_index(
            line_context,
            planning_input.planned_path,
            planned_start_idx,
            max_initial_skip=1,
        )
        if progress_idx <= best_progress:
            continue
        best_line = line_context
        best_progress = progress_idx

    if best_line is None:
        return None, current_route_progress_index, ""
    return best_line, best_progress, "future_route_window"


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
    planned_start_idx = _current_planned_index(planning_input.planned_path, current_lane_id, current_road_id)
    current_route_progress_index = _line_route_progress_index(
        current_line,
        planning_input.planned_path,
        planned_start_idx,
    )
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

    route_lookahead_required = False
    route_lookahead_reason = ""
    if 0 <= target_line_index < len(line_contexts):
        # 目标车道不是当前车道，它的 lane_sequence 不可能包含 planned_path[start_idx]
        # （那一格是当前车道），从 start_idx 开始查必然落空、返回 start_idx-1，
        # 于是下面 current > target 恒成立、换道意图被 current_line_preserves_future_route
        # 一律回退。目标线接手的位置在 start_idx+1，与 lookahead 的取法保持一致。
        target_route_progress_index = _line_route_progress_index(
            line_contexts[target_line_index],
            planning_input.planned_path,
            planned_start_idx,
            max_initial_skip=1,
        )
    else:
        target_route_progress_index = current_route_progress_index
    if (
        target_line_index != 0
        and current_route_progress_index > target_route_progress_index
    ):
        target_lane_id = current_lane_id
        target_line_index = 0
        target_route_progress_index = current_route_progress_index
        route_lookahead_reason = "current_line_preserves_future_route"

    lookahead_line, lookahead_progress_index, lookahead_reason = _select_route_lookahead_target(
        planning_input=planning_input,
        line_contexts=line_contexts,
        current_route_progress_index=max(current_route_progress_index, target_route_progress_index),
        planned_start_idx=planned_start_idx,
    )
    if lookahead_line is not None:
        target_lane_id = lookahead_line.line_id
        target_line_index = lookahead_line.line_index
        target_route_progress_index = lookahead_progress_index
        route_lookahead_required = True
        route_lookahead_reason = lookahead_reason

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
    if route_lookahead_required:
        must_change = True
    deadline_margin_m = float(current_line.first_lane_remaining_length)
    if target_line_index != 0 and 0 <= target_line_index < len(line_contexts):
        deadline_margin_m = min(
            deadline_margin_m,
            float(line_contexts[target_line_index].first_lane_remaining_length),
        )
        # 车道尽头只是“这条线到哪结束”，不是“路线要求你什么时候到位”。
        # 终点落在目标车道上时，必须在那之前完成换道，所以终点在当前线上的投影
        # 才是更紧的 deadline。长直道上这一项会把 deadline 从车道剩余长度拉回到
        # 实际可用的里程，短路段上不影响（min 仍取车道剩余）。
        # 放在这个分支里是因为 deadline 只在 must_change 时被读；不换道时
        # 每帧多做一次整条参考线的投影没有收益，只会抬高 context_ms。
        _, goal_s, _ = project_point_to_path(
            np.asarray(planning_input.goal_xy, dtype=float),
            current_line.line,
        )
        goal_along_current_line = float(goal_s) - float(current_line.current_s)
        if goal_along_current_line > 0.0:
            deadline_margin_m = min(deadline_margin_m, goal_along_current_line)
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
        deadline_margin_m=deadline_margin_m,
        distance_to_goal=distance_to_goal,
        current_road_planned_lanes=current_road_planned_lanes,
        route_lookahead_required=route_lookahead_required,
        route_lookahead_reason=route_lookahead_reason,
        current_route_progress_index=current_route_progress_index,
        target_route_progress_index=target_route_progress_index,
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
