"""阶段 1 行为层决策逻辑。

职责：
- 决定当前行为模式
- 选择当前局部目标车道族
- 判断当前属于 keep / DLC / MLC / 到达哪一类逻辑
- 维护少量跨帧承诺状态

非职责：
- 不直接生成轨迹
- 不直接给完整候选轨迹打分

上游：
    消费 :class:`PlanningContext`
下游：
    输出 :class:`BehaviorDecision` 给 ``sampling.py``
"""

from __future__ import annotations

import math

from traj_planner.state_rules import is_goal_reached, is_lane_change_complete
from traj_planner.types import BehaviorDecision, BehaviorMemory, BehaviorMode, CandidateTrajectory, PlanningContext


def _approach_goal_trigger_distance(current_v: float) -> float:
    """返回进入终点接近态的经验距离阈值。"""

    return max(6.0, 3.0 * max(float(current_v), 1.0))


def _approach_goal_entry_distance(config: dict, current_v: float) -> float:
    """基于正常减速能力估算应进入终点模式的距离阈值。"""

    local_fsm_cfg = config.get("local_fsm", {})
    soft_decel = max(float(local_fsm_cfg.get("arrive_soft_decel", 2.0)), 1e-3)
    cycle_dt = max(float(local_fsm_cfg.get("planner_cycle_dt", 0.1)), 1e-3)
    buffer_dist = float(local_fsm_cfg.get("arrive_emergency_buffer", 0.8))
    min_entry_dist = 2.5
    stop_dist = float(current_v) * float(current_v) / (2.0 * soft_decel)
    one_cycle_travel = float(current_v) * cycle_dt
    return max(min_entry_dist, stop_dist + one_cycle_travel + buffer_dist, _approach_goal_trigger_distance(current_v))


def _approach_goal_target(context: PlanningContext) -> tuple[int, str]:
    """返回终点模式应收敛到的目标线。

    优先选择全局路线在当前 road 上期望的终点车道；
    若当前参考线集合里找不到，则回退到当前 route_intent 的目标线；
    再找不到则回退到当前主线。
    """

    desired_lane_id = context.route_intent.desired_route_lane_id
    if desired_lane_id:
        for line_context in context.line_contexts:
            if line_context.line_id == desired_lane_id:
                return int(line_context.line_index), line_context.line_id

    fallback_idx = int(context.route_intent.target_line_index)
    if 0 <= fallback_idx < len(context.line_contexts):
        fallback_line = context.line_contexts[fallback_idx]
        return int(fallback_line.line_index), fallback_line.line_id

    return int(context.current_line.line_index), context.current_line.line_id


def _dlc_cooldown_cycles(config: dict) -> int:
    """把自由换道冷却秒数转换成规划周期数。"""

    local_fsm_cfg = config.get("local_fsm", {})
    cooldown_s = max(float(local_fsm_cfg.get("dlc_cooldown_s", 0.0)), 0.0)
    cycle_dt = max(float(local_fsm_cfg.get("planner_cycle_dt", 0.1)), 1e-3)
    return int(math.ceil(cooldown_s / cycle_dt))


def _dlc_cooldown_override_active(
    context: PlanningContext,
    current_front_ttc: float,
    current_front_stop_margin: float,
) -> bool:
    """当前车道进入明显危险态时，允许打破 DLC 冷却。"""

    local_fsm_cfg = context.config.get("local_fsm", {})
    release_ttc = float(local_fsm_cfg.get("dlc_cooldown_release_ttc", 2.5))
    release_stop_margin = float(local_fsm_cfg.get("dlc_cooldown_release_stop_margin", 1.0))
    return (
        float(current_front_ttc) < release_ttc
        or float(current_front_stop_margin) < release_stop_margin
    )


def _dlc_cooldown_active(
    context: PlanningContext,
    current_front_ttc: float,
    current_front_stop_margin: float,
) -> bool:
    """判断本帧是否应阻止新的自由换道触发。"""

    if int(context.planner_memory.dlc_cooldown_cycles) <= 0:
        return False
    return not _dlc_cooldown_override_active(context, current_front_ttc, current_front_stop_margin)


def _required_front_gap(v_ego: float) -> float:
    """换道/并线启发式中要求的最小前向安全距离。"""

    comfort_brake = 2.5
    return max(8.0, float(v_ego) * 1.5 + float(v_ego) ** 2 / (2.0 * comfort_brake))


def _required_rear_gap(v_rear: float) -> float:
    """在考虑并线前，目标车道后车至少需要满足的后向间距。"""

    return max(6.0, float(v_rear) * 1.0)


def _required_rear_brake(gap: float, rear_v: float, ego_v: float) -> float:
    """估算并入后目标车道后车所需的制动强度。

    这是一个硬安全门限，不是“整体交通效率”或“礼让”评分项。
    """

    closing_v = max(float(rear_v) - float(ego_v), 0.0)
    if closing_v <= 1e-6:
        return 0.0
    effective_gap = max(float(gap) - 4.0, 1e-3)
    return closing_v * closing_v / (2.0 * effective_gap)


def _rear_safe_for_merge(gap: float, rear_v: float, ego_v: float) -> bool:
    """仅当目标车道后车仍然安全时返回 True。"""

    if gap < _required_rear_gap(rear_v):
        return False
    return _required_rear_brake(gap, rear_v, ego_v) <= 4.0


def _effective_gap(context: PlanningContext, raw_gap: float, obs_length: float) -> float:
    """把中心点间距转成近似 bumper-to-bumper 的有效纵向间距。"""

    ego_length = float(context.config["size"]["L"])
    longitudinal_buffer = float(context.config["IDM"].get("longitudinal_buffer", 0.5))
    return max(float(raw_gap) - 0.5 * (ego_length + float(obs_length)) - longitudinal_buffer, 1e-2)


def _effective_gap_from_projected(context: PlanningContext, projected) -> float:
    """读取投影障碍物的有效纵向间距。"""

    obstacle_length = projected.obstacle.length if projected.obstacle.length > 0.0 else 4.5
    return _effective_gap(context, projected.gap, obstacle_length)


def _idm_desired_gap(v_ego: float, v_front: float, idm_cfg: dict) -> float:
    """计算 IDM 的期望安全间距。"""

    a = max(float(idm_cfg["max_acc"]), 1e-3)
    b = max(abs(float(idm_cfg["comfort_dec"])), 1e-3)
    time_gap = float(idm_cfg["time_gap"])
    min_gap = float(idm_cfg["min_gap"])
    dv = float(v_ego) - float(v_front)
    dynamic_gap = float(v_ego) * time_gap + float(v_ego) * dv / (2.0 * math.sqrt(a * b))
    return min_gap + max(0.0, dynamic_gap)


def _idm_acc_free(v_ego: float, idm_cfg: dict) -> float:
    """计算自由路况下的 IDM 加速度。"""

    desired_v = max(float(idm_cfg["desired_v"]), 1e-3)
    max_acc = float(idm_cfg["max_acc"])
    delta = float(idm_cfg["delta"])
    return max_acc * (1.0 - (max(float(v_ego), 0.0) / desired_v) ** delta)


def _idm_acc_follow(v_ego: float, gap_eff: float, v_front: float, idm_cfg: dict) -> float:
    """计算有前车跟随时的 IDM 加速度。"""

    desired_v = max(float(idm_cfg["desired_v"]), 1e-3)
    max_acc = float(idm_cfg["max_acc"])
    delta = float(idm_cfg["delta"])
    desired_gap = _idm_desired_gap(v_ego, v_front, idm_cfg)
    return max_acc * (
        1.0
        - (max(float(v_ego), 0.0) / desired_v) ** delta
        - (desired_gap / max(float(gap_eff), 1e-2)) ** 2
    )


def _line_idm_acc(context: PlanningContext, line_index: int, ego_v: float | None = None) -> float:
    """估算自车留在某条候选线上的 IDM 纵向加速度。"""

    line_context = context.line_contexts[line_index]
    ego_v = float(context.ego_state.v if ego_v is None else ego_v)
    idm_cfg = context.config["IDM"]
    if line_context.front_obstacle is None:
        return _idm_acc_free(ego_v, idm_cfg)
    return _idm_acc_follow(
        ego_v,
        _effective_gap_from_projected(context, line_context.front_obstacle),
        float(line_context.front_obstacle.obstacle.v),
        idm_cfg,
    )


def _rear_acc_following_ego(context: PlanningContext, rear_obstacle, ego_v: float) -> float:
    """估算某条线后车跟随并线后自车时的 IDM 加速度。"""

    if rear_obstacle is None:
        return 0.0
    idm_cfg = context.config["IDM"]
    rear_v = float(rear_obstacle.obstacle.v)
    ego_length = float(context.config["size"]["L"])
    gap_eff = _effective_gap(context, rear_obstacle.gap, ego_length)
    return _idm_acc_follow(rear_v, gap_eff, ego_v, idm_cfg)


def _rear_acc_following_front(context: PlanningContext, rear_obstacle, front_obstacle) -> float:
    """估算某条线后车跟随其原有前车时的 IDM 加速度。"""

    if rear_obstacle is None:
        return 0.0
    idm_cfg = context.config["IDM"]
    rear_v = float(rear_obstacle.obstacle.v)
    if front_obstacle is None:
        return _idm_acc_free(rear_v, idm_cfg)

    front_length = front_obstacle.obstacle.length if front_obstacle.obstacle.length > 0.0 else 4.5
    combined_gap = float(rear_obstacle.gap + front_obstacle.gap)
    gap_eff = _effective_gap(context, combined_gap, front_length)
    return _idm_acc_follow(rear_v, gap_eff, float(front_obstacle.obstacle.v), idm_cfg)


def _front_risk_metrics(context: PlanningContext) -> tuple[float, float, float]:
    """返回当前主线前向风险指标: ``(ttc, stop_margin, raw_gap)``。"""

    current_line = context.current_line
    if current_line.front_obstacle is None:
        return float("inf"), float("inf"), 999.0

    idm_cfg = context.config["IDM"]
    ttc_cfg = context.config["TTC"]
    ego_v = float(context.ego_state.v)
    front_v = float(current_line.front_obstacle.obstacle.v)
    raw_gap = float(current_line.front_obstacle.gap)
    gap_eff = _effective_gap_from_projected(context, current_line.front_obstacle)
    closing_v = ego_v - front_v
    if closing_v > float(ttc_cfg["closing_speed_eps"]):
        ttc = gap_eff / closing_v
    else:
        ttc = float("inf")
    launch_eval_v = max(ego_v, min(_line_preferred_speed(context, 0), 3.0))
    stop_margin = gap_eff - _idm_desired_gap(launch_eval_v, front_v, idm_cfg)
    return float(ttc), float(stop_margin), raw_gap


def _make_decision(
    context: PlanningContext,
    *,
    mode: BehaviorMode,
    allowed_line_indices: list[int],
    target_line_index: int,
    target_lane_id: str,
    must_change: bool,
    deadline_margin_m: float,
    reason: str,
    best_mobil_incentive: float = 0.0,
) -> BehaviorDecision:
    """统一构造带调试指标的行为决策。"""

    current_front_ttc, current_front_stop_margin, current_front_gap = _front_risk_metrics(context)
    return BehaviorDecision(
        mode=mode,
        allowed_line_indices=allowed_line_indices,
        target_line_index=target_line_index,
        target_lane_id=target_lane_id,
        must_change=must_change,
        deadline_margin_m=deadline_margin_m,
        reason=reason,
        best_mobil_incentive=float(best_mobil_incentive),
        current_front_ttc=float(current_front_ttc),
        current_front_stop_margin=float(current_front_stop_margin),
        current_front_gap=float(current_front_gap),
    )


def _immediate_merge_feasible(context: PlanningContext) -> bool:
    """检查当前强制换道目标是否“可以立刻并过去”。

    约束：
    - 目标线必须是一阶相邻车道
    - 前向 gap 足够
    - 并入后目标车道后车仍然安全
    """

    target_idx = context.route_intent.target_line_index
    if target_idx <= 0 or target_idx >= len(context.line_contexts):
        return False

    target_line = context.line_contexts[target_idx]
    if abs(target_line.lateral_step) != 1:
        return False
    ego_v = float(context.ego_state.v)

    front_ok = True
    if target_line.front_obstacle is not None:
        front_ok = _effective_gap_from_projected(context, target_line.front_obstacle) >= _required_front_gap(ego_v)

    rear_ok = True
    if target_line.rear_obstacle is not None:
        rear_ok = _rear_safe_for_merge(
            gap=target_line.rear_obstacle.gap,
            rear_v=target_line.rear_obstacle.obstacle.v,
            ego_v=ego_v,
        )

    return front_ok and rear_ok


def _line_preferred_speed(context: PlanningContext, line_index: int) -> float:
    """根据该线前方障碍状态估算一条“参考速度”。

    这是轻量的收益估计，不是最终速度规划结果。
    它只用于：
    - 行为层比较不同线的前向收益
    - 采样层限制终端速度范围
    """

    line_context = context.line_contexts[line_index]
    ref_v = float(context.config["constrain"].get("target_v", context.config["constrain"]["max_v"]))
    if line_context.front_obstacle is None:
        return ref_v

    front_gap = max(0.0, _effective_gap_from_projected(context, line_context.front_obstacle))
    front_v = max(0.0, float(line_context.front_obstacle.obstacle.v))
    if front_v <= 0.3:
        return min(ref_v, math.sqrt(max(0.0, 2.0 * 2.5 * front_gap)))
    return min(ref_v, max(front_v, front_v + 0.15 * max(front_gap - 4.0, 0.0)))


def _line_merge_feasible(context: PlanningContext, line_index: int) -> bool:
    """判断某条目标线是否有资格作为“一次单车道并线”的目标。

    说明：
    - 只允许一阶相邻车道
    - 前向安全和后向安全都作为行为层的硬过滤条件
    """

    if line_index <= 0 or line_index >= len(context.line_contexts):
        return False

    line_context = context.line_contexts[line_index]
    if abs(line_context.lateral_step) != 1:
        return False
    ego_v = float(context.ego_state.v)

    front_ok = True
    if line_context.front_obstacle is not None:
        front_ok = _effective_gap_from_projected(context, line_context.front_obstacle) >= _required_front_gap(ego_v)

    rear_ok = True
    if line_context.rear_obstacle is not None:
        rear_ok = _rear_safe_for_merge(
            gap=line_context.rear_obstacle.gap,
            rear_v=line_context.rear_obstacle.obstacle.v,
            ego_v=ego_v,
        )

    return front_ok and rear_ok


def _find_line_index_by_lane_id(context: PlanningContext, lane_id: str) -> int:
    """在当前帧的候选线集合中，根据 lane_id 找回 line_index。"""

    for line_context in context.line_contexts:
        if line_context.line_id == lane_id:
            return line_context.line_index
    return -1


def _committed_lane_change_incomplete(context: PlanningContext, target_lane_id: str) -> bool:
    """统一判断一个已承诺换道是否仍处于未完成状态。"""

    return not is_lane_change_complete(
        current_lane_id=context.current_line.line_id,
        target_lane_id=target_lane_id,
        current_d=context.current_line.current_d,
        config=context.config,
    )


def _committed_lane_change_completed_this_cycle(context: PlanningContext) -> bool:
    """判断上一帧承诺的换道是否已在当前帧开始时完成。"""

    memory = context.planner_memory
    if not memory.lane_change_committed or not memory.target_lane_id:
        return False
    return is_lane_change_complete(
        current_lane_id=context.current_line.line_id,
        target_lane_id=memory.target_lane_id,
        current_d=context.current_line.current_d,
        config=context.config,
    )


def _continue_committed_lane_change(context: PlanningContext) -> BehaviorDecision | None:
    """若上一帧已承诺换道且尚未完成，则统一延续到执行态。"""

    memory = context.planner_memory
    if not memory.lane_change_committed or not memory.target_lane_id:
        return None
    if not _committed_lane_change_incomplete(context, memory.target_lane_id):
        return None

    target_line_index = _find_line_index_by_lane_id(context, memory.target_lane_id)
    if target_line_index < 0:
        return None

    if context.route_intent.must_change or memory.mode in (BehaviorMode.PREPARE_MLC, BehaviorMode.EXECUTE_MLC):
        return _make_decision(
            context,
            mode=BehaviorMode.EXECUTE_MLC,
            allowed_line_indices=[target_line_index],
            target_line_index=target_line_index,
            target_lane_id=memory.target_lane_id,
            must_change=context.route_intent.must_change,
            deadline_margin_m=context.route_intent.deadline_margin_m,
            reason="lane_change_already_committed",
        )

    if memory.mode in (BehaviorMode.PREPARE_DLC, BehaviorMode.EXECUTE_DLC):
        return _make_decision(
            context,
            mode=BehaviorMode.EXECUTE_DLC,
            allowed_line_indices=[target_line_index],
            target_line_index=target_line_index,
            target_lane_id=memory.target_lane_id,
            must_change=False,
            deadline_margin_m=context.route_intent.deadline_margin_m,
            reason="lane_change_already_committed",
        )

    return None


def _lane_bias(context: PlanningContext, line_index: int) -> float:
    """返回该目标线的轻量换道偏置。"""

    line_context = context.line_contexts[line_index]
    mobil_cfg = context.config["MOBIL"]
    if line_context.lateral_step > 0:
        return float(mobil_cfg.get("bias_right", 0.0))
    if line_context.lateral_step < 0:
        return float(mobil_cfg.get("bias_left", 0.0))
    return 0.0


def _mobil_incentive_for_line(context: PlanningContext, target_line_index: int) -> float:
    """计算相邻目标线的 MOBIL 换道收益。"""

    current_line = context.current_line
    target_line = context.line_contexts[target_line_index]
    mobil_cfg = context.config["MOBIL"]
    ego_v = float(context.ego_state.v)
    politeness = float(mobil_cfg["politeness"])
    acc_threshold = float(mobil_cfg["acc_threshold"])

    a_ego_before = _line_idm_acc(context, 0, ego_v)
    a_ego_after = _line_idm_acc(context, target_line_index, ego_v)
    a_old_rear_before = _rear_acc_following_ego(context, current_line.rear_obstacle, ego_v)
    a_old_rear_after = _rear_acc_following_front(context, current_line.rear_obstacle, current_line.front_obstacle)
    a_new_rear_before = _rear_acc_following_front(context, target_line.rear_obstacle, target_line.front_obstacle)
    a_new_rear_after = _rear_acc_following_ego(context, target_line.rear_obstacle, ego_v)

    incentive = (
        (a_ego_after - a_ego_before)
        + politeness * ((a_old_rear_after - a_old_rear_before) + (a_new_rear_after - a_new_rear_before))
        - acc_threshold
        - _lane_bias(context, target_line_index)
    )
    return float(incentive)


def _select_discretionary_target(context: PlanningContext) -> tuple[int, float]:
    """为自由换道选择 MOBIL 收益最优的一阶相邻目标线。"""

    best_index = -1
    best_incentive = -float("inf")

    for line_context in context.line_contexts[1:]:
        if abs(line_context.lateral_step) != 1:
            continue
        if not _line_merge_feasible(context, line_context.line_index):
            continue

        incentive = _mobil_incentive_for_line(context, line_context.line_index)
        if incentive > best_incentive:
            best_incentive = incentive
            best_index = line_context.line_index

    if best_index < 0:
        return -1, 0.0
    return best_index, float(best_incentive)


def _must_keep_current_lane(context: PlanningContext) -> bool:
    """判断当前是否仍应优先保持在本车道内处理。"""

    current_line = context.current_line
    if current_line.front_obstacle is None:
        return False

    if context.route_intent.must_change:
        return False

    current_gap = _effective_gap_from_projected(context, current_line.front_obstacle)
    desired_gap = _required_front_gap(context.ego_state.v)
    return current_gap < desired_gap


def decide_behavior(context: PlanningContext) -> BehaviorDecision:
    """计算当前规划周期的行为模式。

    输入：
        :class:`PlanningContext`

    输出：
        :class:`BehaviorDecision`

    判定顺序：
    1. 终点接近态
    2. 强制换道（路线驱动）
    3. 已承诺的自由换道执行延续
    4. 新触发的自由换道
    5. 保持当前车道

    这个顺序是有意设计的：
    路线安全和路线连通性优先级高于局部收益。
    """

    route_intent = context.route_intent
    current_line = context.current_line
    current_v = float(context.ego_state.v)
    near_goal = route_intent.distance_to_goal <= _approach_goal_entry_distance(context.config, current_v)
    mobil_cfg = context.config["MOBIL"]
    ttc_cfg = context.config["TTC"]
    current_front_ttc, current_front_stop_margin, _ = _front_risk_metrics(context)

    if not route_intent.must_change and is_goal_reached(route_intent.distance_to_goal, current_v, context.config):
        return _make_decision(
            context,
            mode=BehaviorMode.ARRIVED,
            allowed_line_indices=[0],
            target_line_index=0,
            target_lane_id=current_line.line_id,
            must_change=False,
            deadline_margin_m=route_intent.deadline_margin_m,
            reason="goal_reached",
        )

    if near_goal:
        goal_line_index, goal_lane_id = _approach_goal_target(context)
        return _make_decision(
            context,
            mode=BehaviorMode.APPROACH_GOAL,
            allowed_line_indices=[goal_line_index],
            target_line_index=goal_line_index,
            target_lane_id=goal_lane_id,
            must_change=False,
            deadline_margin_m=float(context.line_contexts[goal_line_index].remaining_length),
            reason="goal_near_prioritize_arrival",
        )

    committed_decision = _continue_committed_lane_change(context)
    if committed_decision is not None:
        return committed_decision

    if route_intent.must_change:
        urgency_margin = max(12.0, 2.5 * current_v + 6.0)
        if route_intent.deadline_margin_m <= urgency_margin or _immediate_merge_feasible(context):
            return _make_decision(
                context,
                mode=BehaviorMode.EXECUTE_MLC,
                allowed_line_indices=[route_intent.target_line_index],
                target_line_index=route_intent.target_line_index,
                target_lane_id=route_intent.target_lane_id,
                must_change=True,
                deadline_margin_m=route_intent.deadline_margin_m,
                reason="mandatory_lane_change_execute",
            )

        return _make_decision(
            context,
            mode=BehaviorMode.PREPARE_MLC,
            allowed_line_indices=[0, route_intent.target_line_index],
            target_line_index=route_intent.target_line_index,
            target_lane_id=route_intent.target_lane_id,
            must_change=True,
            deadline_margin_m=route_intent.deadline_margin_m,
            reason="mandatory_lane_change_prepare",
        )

    dlc_cooldown_active = _dlc_cooldown_active(context, current_front_ttc, current_front_stop_margin)
    discretionary_target_idx, best_incentive = _select_discretionary_target(context)
    if (
        not dlc_cooldown_active
        and discretionary_target_idx > 0
        and best_incentive > float(mobil_cfg.get("prepare_threshold", mobil_cfg["acc_threshold"]))
    ):
        prepare_ttc_threshold = float(ttc_cfg["prepare_threshold"])
        execute_ttc_threshold = float(ttc_cfg["execute_threshold"])
        prepare_stop_threshold = float(ttc_cfg["stop_margin_prepare_threshold"])
        execute_stop_threshold = float(ttc_cfg["stop_margin_execute_threshold"])
        execute_threshold = float(mobil_cfg.get("execute_threshold", mobil_cfg["acc_threshold"]))

        should_execute = (
            best_incentive > execute_threshold
            and (
                current_front_ttc < execute_ttc_threshold
                or current_front_stop_margin < execute_stop_threshold
            )
        )
        should_prepare = (
            current_front_ttc < prepare_ttc_threshold
            or current_front_stop_margin < prepare_stop_threshold
        )

        if should_execute:
            return _make_decision(
                context,
                mode=BehaviorMode.EXECUTE_DLC,
                allowed_line_indices=[discretionary_target_idx],
                target_line_index=discretionary_target_idx,
                target_lane_id=context.line_contexts[discretionary_target_idx].line_id,
                must_change=False,
                deadline_margin_m=route_intent.deadline_margin_m,
                reason="discretionary_lane_change_execute",
                best_mobil_incentive=best_incentive,
            )

        if should_prepare:
            return _make_decision(
                context,
                mode=BehaviorMode.PREPARE_DLC,
                allowed_line_indices=[0, discretionary_target_idx],
                target_line_index=discretionary_target_idx,
                target_lane_id=context.line_contexts[discretionary_target_idx].line_id,
                must_change=False,
                deadline_margin_m=route_intent.deadline_margin_m,
                reason="discretionary_lane_change_prepare",
                best_mobil_incentive=best_incentive,
            )

    if dlc_cooldown_active:
        reason = "keep_lane_dlc_cooldown_active"
    elif _must_keep_current_lane(context):
        reason = "keep_lane_front_obstacle_present"
    else:
        reason = "keep_lane_route_allows_current"

    return _make_decision(
        context,
        mode=BehaviorMode.KEEP,
        allowed_line_indices=[0],
        target_line_index=0,
        target_lane_id=current_line.line_id,
        must_change=False,
        deadline_margin_m=route_intent.deadline_margin_m,
        reason=reason,
        best_mobil_incentive=max(0.0, best_incentive),
    )


def update_behavior_memory(
    context: PlanningContext,
    decision: BehaviorDecision,
    selected_candidate: CandidateTrajectory | None,
) -> BehaviorMemory:
    """在选出候选轨迹后更新跨帧行为记忆。

    输入：
        当前 ``PlanningContext``、本帧行为决策 ``BehaviorDecision``，
        以及选中的最优候选（若规划失败则为 ``None``）。

    输出：
        下一帧要继续携带的 :class:`BehaviorMemory`

    核心目的：
        当换道已经进入执行态后，保留一个轻量“承诺信号”，
        避免下一帧立刻反悔翻回去。
    """

    memory = BehaviorMemory(
        mode=decision.mode,
        target_lane_id=decision.target_lane_id,
        target_line_index=decision.target_line_index,
        lane_change_committed=context.planner_memory.lane_change_committed,
        dlc_cooldown_cycles=max(int(context.planner_memory.dlc_cooldown_cycles) - 1, 0),
        last_selected_line_index=context.planner_memory.last_selected_line_index,
        cycle_count=context.planner_memory.cycle_count + 1,
    )

    if selected_candidate is not None:
        memory.last_selected_line_index = selected_candidate.terminal.line_index
        if decision.mode in (
            BehaviorMode.PREPARE_MLC,
            BehaviorMode.EXECUTE_MLC,
            BehaviorMode.PREPARE_DLC,
            BehaviorMode.EXECUTE_DLC,
        ):
            memory.lane_change_committed = selected_candidate.terminal.line_index == decision.target_line_index
        else:
            memory.lane_change_committed = False

    if _committed_lane_change_completed_this_cycle(context):
        memory.lane_change_committed = False
        memory.dlc_cooldown_cycles = max(memory.dlc_cooldown_cycles, _dlc_cooldown_cycles(context.config))

    return memory
