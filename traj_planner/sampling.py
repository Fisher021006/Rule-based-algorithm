"""候选家族构造与终端状态采样。

这个模块不直接生成轨迹，只负责回答：
“这一帧应该尝试哪些终端状态？”

上游：
    输入 ``PlanningContext`` 和 ``BehaviorDecision``。
下游：
    输出 ``TerminalCandidate`` 给 ``generator.py``。

阶段 1 的采样思想：
- 先由行为层决定当前模式
- 每个模式只给出固定的 ``T`` / ``hold_t`` 模板
- 对每个 ``T``，按动力学边界生成速度采样范围
- 每个候选 ``target_v`` 唯一对应一个名义 ``target_s``
- 再用目标线前后车约束把明显不合理的 ``(T, v, s)`` 直接裁掉
"""

from __future__ import annotations

import math

from traj_planner.tools import project_point_to_path
from traj_planner.types import BehaviorDecision, BehaviorMode, PlanningContext, SamplingTemplate, TerminalCandidate


TARGET_V_STEP = 0.5
TARGET_V_STEP_MID_SPEED = 0.8
TARGET_V_STEP_HIGH_SPEED = 1.0
TARGET_V_MID_SPEED_THRESH = 3.0
TARGET_V_HIGH_SPEED_THRESH = 6.0
FRONT_REQUIRED_TIME_GAP = 1.2
FRONT_COMFORT_BRAKE = 2.5
FRONT_MIN_REQUIRED_GAP = 6.0
REAR_MIN_REQUIRED_GAP = 4.0
REAR_GAP_SPEED_RATIO = 0.8
REAR_MAX_REQUIRED_BRAKE = 4.0
APPROACH_GOAL_SOFT_DECEL = 2.0
APPROACH_GOAL_STOP_BUFFER = 0.5
APPROACH_GOAL_FORCE_STOP_DIST = 3.0
APPROACH_GOAL_FORCE_STOP_SPEED = 1.0


def _reference_speed(context: PlanningContext) -> float:
    """返回当前周期用于参考的路线巡航速度。"""

    return float(context.config["constrain"].get("target_v", context.config["constrain"]["max_v"]))


def _dynamic_speed_bounds(
    context: PlanningContext,
    plan_T: float,
    decision: BehaviorDecision,
) -> tuple[float, float]:
    """按当前速度和动力学约束给出终端速度采样区间。

    这里只考虑：
    - 当前自车速度
    - 最大加速度 / 最大减速度
    - 模板时域 ``plan_T``

    不直接在这里引入前后车约束；前后车约束在生成每个 ``(T, v, s)``
    后再逐样本裁剪。
    """

    ego_v = float(context.ego_state.v)
    constrain_cfg = context.config["constrain"]
    v_lower = max(float(constrain_cfg["min_v"]), ego_v + float(constrain_cfg["min_acc"]) * float(plan_T))
    v_upper = min(float(constrain_cfg["max_v"]), ego_v + float(constrain_cfg["max_acc"]) * float(plan_T))

    if v_upper < v_lower:
        # 若想要的速度上限比当前动力学最低可达速度更低，则退化成“尽可能慢”的单点采样。
        v_upper = v_lower

    return float(v_lower), float(v_upper)


def _sample_speed_values(v_lower: float, v_upper: float, step: float = TARGET_V_STEP) -> list[float]:
    """按固定步长采样速度区间。"""

    v_lower = max(0.0, float(v_lower))
    v_upper = max(v_lower, float(v_upper))
    step = max(float(step), 1e-3)

    speeds: list[float] = []
    value = v_lower
    while value <= v_upper + 1e-9:
        speeds.append(round(float(value), 6))
        value += step

    if not speeds:
        speeds.append(round(v_lower, 6))
    elif speeds[-1] < v_upper - 1e-6:
        speeds.append(round(v_upper, 6))

    return speeds


def _adaptive_target_v_step(context: PlanningContext) -> float:
    """根据当前自车速度调整速度采样粒度。"""

    ego_v = float(context.ego_state.v)
    if ego_v >= TARGET_V_HIGH_SPEED_THRESH:
        return TARGET_V_STEP_HIGH_SPEED
    if ego_v >= TARGET_V_MID_SPEED_THRESH:
        return TARGET_V_STEP_MID_SPEED
    return TARGET_V_STEP


def _predict_obstacle_s(line_context, obstacle, travel_time: float) -> float:
    """用常速度模型预测障碍物在给定时刻投影到候选线上的 ``s``。"""

    obs_xy = (
        float(obstacle.x) + float(obstacle.v) * math.cos(float(obstacle.heading)) * float(travel_time),
        float(obstacle.y) + float(obstacle.v) * math.sin(float(obstacle.heading)) * float(travel_time),
    )
    _, obs_s, _ = project_point_to_path(obs_xy, line_context.line)
    return float(obs_s)


def _required_front_gap_for_target_speed(target_v: float) -> float:
    """给定候选终端速度，返回目标点处要求的最小前向间距。"""

    return max(
        FRONT_MIN_REQUIRED_GAP,
        float(target_v) * FRONT_REQUIRED_TIME_GAP + float(target_v) ** 2 / (2.0 * FRONT_COMFORT_BRAKE),
    )


def _sample_front_safe(line_context, target_s: float, target_v: float, travel_time: float) -> bool:
    """检查候选终点在该线前车约束下是否安全。"""

    if line_context.front_obstacle is None:
        return True

    obs_s = _predict_obstacle_s(line_context, line_context.front_obstacle.obstacle, travel_time)
    gap = float(obs_s - target_s)
    return gap >= _required_front_gap_for_target_speed(target_v)


def _sample_rear_safe(line_context, target_s: float, target_v: float, travel_time: float) -> bool:
    """检查并入目标线后的后车安全。

    只对非当前线候选启用，因为保持当前线不会产生“切到后车前方”的问题。
    """

    if int(line_context.line_index) == 0 or line_context.rear_obstacle is None:
        return True

    rear_obstacle = line_context.rear_obstacle.obstacle
    rear_s = _predict_obstacle_s(line_context, rear_obstacle, travel_time)
    gap = float(target_s - rear_s)
    required_gap = max(REAR_MIN_REQUIRED_GAP, float(rear_obstacle.v) * REAR_GAP_SPEED_RATIO)
    if gap < required_gap:
        return False

    closing_v = max(float(rear_obstacle.v) - float(target_v), 0.0)
    if closing_v <= 1e-6:
        return True

    effective_gap = max(gap - REAR_MIN_REQUIRED_GAP, 1e-3)
    required_brake = closing_v * closing_v / (2.0 * effective_gap)
    return required_brake <= REAR_MAX_REQUIRED_BRAKE


def _goal_s_on_line(context: PlanningContext, line_context) -> float:
    """把几何终点投影到候选线，得到终点在该线上的纵向位置。"""

    _, goal_s, _ = project_point_to_path(context.goal_xy, line_context.line)
    return float(goal_s)


def _approach_goal_s_window(context: PlanningContext) -> float:
    """返回终点所在车道上的纵向允许窗口半宽。"""

    local_fsm_cfg = context.config.get("local_fsm", {})
    return max(float(local_fsm_cfg.get("approach_goal_s_window", 0.5)), 0.0)


def _approach_goal_stop_reserve(target_v: float) -> float:
    """给定候选终端速度，返回为在终点前平顺停住所需保留的距离。"""

    return float(target_v) * float(target_v) / (2.0 * APPROACH_GOAL_SOFT_DECEL) + APPROACH_GOAL_STOP_BUFFER


def build_candidate_templates(context: PlanningContext, decision: BehaviorDecision) -> list[SamplingTemplate]:
    """为当前行为模式构建采样家族。

    模板只决定：
    - 目标线
    - 换道前保持时间 ``hold_t``
    - 规划时域 ``T``

    终端速度 ``target_v`` 不再在模板里写死，而是在 ``sample_terminal_states``
    中根据动力学范围和前后车约束逐个 ``T`` 动态生成。
    """

    if decision.mode == BehaviorMode.KEEP:
        return [
            SamplingTemplate(
                name="keep_current",
                line_indices=[0],
                hold_t_values=[0.0],
                plan_T_values=[2.0, 2.5, 3.0],
                target_v_values=[],
                s_bias_values=[0.0],
            )
        ]

    if decision.mode == BehaviorMode.PREPARE_DLC:
        return [
            SamplingTemplate(
                name="prepare_dlc_keep",
                line_indices=[0],
                hold_t_values=[0.0],
                plan_T_values=[2.0, 2.5, 3.0],
                target_v_values=[],
                s_bias_values=[0.0],
            ),
            SamplingTemplate(
                name="prepare_dlc_target",
                line_indices=[decision.target_line_index],
                hold_t_values=[0.0,0.4],
                plan_T_values=[3.0,3.6],
                target_v_values=[],
                s_bias_values=[0.0],
            ),
        ]

    if decision.mode == BehaviorMode.EXECUTE_DLC:
        return [
            SamplingTemplate(
                name="execute_dlc_target",
                line_indices=[decision.target_line_index],
                hold_t_values=[0.0],
                plan_T_values=[2.8, 3.4, 4.0],
                target_v_values=[],
                s_bias_values=[0.0],
            )
        ]

    if decision.mode == BehaviorMode.PREPARE_MLC:
        return [
            SamplingTemplate(
                name="prepare_keep",
                line_indices=[0],
                hold_t_values=[0.0],
                plan_T_values=[2.0, 2.5, 3.0],
                target_v_values=[],
                s_bias_values=[0.0],
            ),
            SamplingTemplate(
                name="prepare_target",
                line_indices=[decision.target_line_index],
                hold_t_values=[0.0,0.4],
                plan_T_values=[3.0,3.6],
                target_v_values=[],
                s_bias_values=[0.0],
            ),
        ]

    if decision.mode == BehaviorMode.EXECUTE_MLC:
        return [
            SamplingTemplate(
                name="execute_target",
                line_indices=[decision.target_line_index],
                hold_t_values=[0.0],
                plan_T_values=[2.8, 3.4, 4.0],
                target_v_values=[],
                s_bias_values=[0.0],
            )
        ]

    return [
        SamplingTemplate(
            name="approach_goal",
            line_indices=[decision.target_line_index],
            hold_t_values=[0.0],
            plan_T_values=[2.0, 2.5, 3.0],
            target_v_values=[],
            s_bias_values=[0.0],
        )
    ]


def sample_terminal_states(
    context: PlanningContext,
    decision: BehaviorDecision,
    templates: list[SamplingTemplate],
) -> list[TerminalCandidate]:
    """把模板展开成具体的终端状态候选。

    当前版本中：
    - 先定模板里的 ``T`` / ``hold_t``
    - 再按动力学约束生成 ``target_v`` 采样集合
    - 每个 ``target_v`` 唯一对应一个名义 ``target_s``
    - 最后对 ``(T, target_v, target_s)`` 做前后车安全裁剪
    """

    ego_v = float(context.ego_state.v)
    candidates: list[TerminalCandidate] = []
    seen = set()

    for template in templates:
        for line_index in template.line_indices:
            line_context = context.line_contexts[line_index]
            base_s = float(line_context.current_s)
            line_end_s = float(line_context.cumulative_s[-1])

            for hold_t in template.hold_t_values:
                for plan_T in template.plan_T_values:
                    v_lower, v_upper = _dynamic_speed_bounds(context, plan_T, decision)
                    speed_samples = _sample_speed_values(v_lower, v_upper, _adaptive_target_v_step(context))


                    for target_v in speed_samples:
                        target_s = base_s + ego_v * float(hold_t) + 0.5 * (ego_v + float(target_v)) * float(plan_T)

                        if decision.mode == BehaviorMode.APPROACH_GOAL:
                            goal_s = _goal_s_on_line(context, line_context)
                            goal_window = _approach_goal_s_window(context)
                            goal_cap = goal_s + goal_window - _approach_goal_stop_reserve(float(target_v))
                            if target_s > goal_cap:
                                # 终点夹紧生效：路径被截短后末端速度必须同步归零。否则五次
                                # 多项式只能靠"负速度回退 + 静默钳位"凑出一条弧长不够却仍带
                                # 末端速度的假可行轨迹，车会卡在离终点几米处不动。
                                target_v = 0.0
                                target_s = goal_s + goal_window - _approach_goal_stop_reserve(0.0)
                            else:
                                target_s = goal_cap

                        # 为下一帧保留少量距离，避免贴着末端采样导致几何退化。
                        line_end_margin = 1.0 if decision.mode != BehaviorMode.APPROACH_GOAL else 0.1
                        if decision.mode == BehaviorMode.APPROACH_GOAL:
                            min_progress = 0.1
                        else:
                            min_progress = 2.0 if line_index == 0 else 4.0
                        target_s = max(target_s, line_context.current_s + min_progress)
                        target_s = min(target_s, max(line_end_s - line_end_margin, line_context.current_s + min_progress))

                        travel_time = float(hold_t) + float(plan_T)
                        if not _sample_front_safe(line_context, float(target_s), float(target_v), travel_time):
                            continue
                        if not _sample_rear_safe(line_context, float(target_s), float(target_v), travel_time):
                            continue

                        key = (
                            line_index,
                            round(float(target_s), 2),
                            round(float(target_v), 2),
                            round(float(plan_T), 2),
                            round(float(hold_t), 2),
                            template.name,
                        )
                        if key in seen:
                            continue
                        seen.add(key)

                        candidates.append(
                            TerminalCandidate(
                                line_index=line_index,
                                line_id=line_context.line_id,
                                target_s=float(target_s),
                                target_v=float(target_v),
                                plan_T=float(plan_T),
                                hold_t=float(hold_t),
                                template_name=template.name,
                            )
                        )

            if decision.mode == BehaviorMode.APPROACH_GOAL:
                goal_s = _goal_s_on_line(context, line_context)
                goal_window = _approach_goal_s_window(context)
                goal_dist = float(context.route_intent.distance_to_goal)
                v_lower, v_upper = _dynamic_speed_bounds(context, min(template.plan_T_values), decision)
                if goal_dist <= APPROACH_GOAL_FORCE_STOP_DIST or v_upper <= APPROACH_GOAL_FORCE_STOP_SPEED:
                    # 停车点落在终点投影处；原先退到 goal_s - goal_window 会白白缩短
                    # 最后一段路径，把小横向偏差放大成超限曲率。
                    target_s = min(goal_s + goal_window, max(goal_s, line_context.current_s + 1e-3))
                    key = (
                        line_index,
                        round(float(target_s), 2),
                        0.0,
                        round(float(min(template.plan_T_values)), 2),
                        0.0,
                        f"{template.name}_stop",
                    )
                    if key not in seen:
                        seen.add(key)
                        candidates.append(
                            TerminalCandidate(
                                line_index=line_index,
                                line_id=line_context.line_id,
                                target_s=float(target_s),
                                target_v=0.0,
                                plan_T=float(min(template.plan_T_values)),
                                hold_t=0.0,
                                template_name=f"{template.name}_stop",
                            )
                        )

    return candidates
