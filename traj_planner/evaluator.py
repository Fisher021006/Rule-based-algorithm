"""候选轨迹的硬约束过滤与软评分模块。

上游：
    从 ``generator.py`` 接收真实生成出来的 ``CandidateTrajectory`` 列表。
下游：
    向 ``planner.py`` 返回最优候选，以及带完整标注的候选集合。

设计原则：
- 安全条件是硬约束
- 候选排序采用归一化多目标代价

当前软评分主要由以下几类因素组成：
- TTC 风险
- 停车余量 / 前向净空
- MOBIL 换道效益
- 纵向效率
- 舒适性
- 与上一帧的连续性
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from traj_planner.tools import project_point_to_path
from traj_planner.types import BehaviorDecision, BehaviorMode, CandidateTrajectory, ObstacleSnapshot, PlanningContext


DT = 0.1


def _clamp01(value: float) -> float:
    """把数值截断到 ``[0, 1]``。"""

    return max(0.0, min(float(value), 1.0))


def _ego_box(context: PlanningContext) -> tuple[float, float]:
    """返回用于粗碰撞过滤的自车长宽。"""

    return float(context.config["size"]["L"]), float(context.config["size"]["W"])


def _obstacle_box(obstacle: ObstacleSnapshot) -> tuple[float, float]:
    """返回障碍物长宽，缺省时使用保守默认值。"""

    length = obstacle.length if obstacle.length > 0.0 else 4.5
    width = obstacle.width if obstacle.width > 0.0 else 1.8
    return float(length), float(width)


def _predict_obstacle_position(obstacle: ObstacleSnapshot, t: float) -> np.ndarray:
    """用常速度模型预测障碍物在未来时刻的位置。"""

    vx = float(obstacle.v) * math.cos(float(obstacle.heading))
    vy = float(obstacle.v) * math.sin(float(obstacle.heading))
    return np.array([obstacle.x + vx * t, obstacle.y + vy * t], dtype=float)


def _trajectory_dynamic_metrics(traj: np.ndarray) -> dict[str, float]:
    """计算一条六元组轨迹的动力学摘要指标。"""

    if traj is None or len(traj) == 0:
        return {
            "max_v": 0.0,
            "max_abs_acc": 0.0,
            "max_abs_kappa": 0.0,
            "jerk_cost": 0.0,
            "mean_abs_jerk": 0.0,
            "max_lat_acc": 0.0,
        }

    acc = traj[:, 4]
    jerk = np.diff(acc) / DT if len(acc) > 1 else np.array([0.0])
    speed = np.maximum(traj[:, 3], 0.0)
    kappa = np.abs(traj[:, 5])
    return {
        "max_v": float(np.max(traj[:, 3])),
        "max_abs_acc": float(np.max(np.abs(acc))),
        "max_abs_kappa": float(np.max(np.abs(traj[:, 5]))),
        "jerk_cost": float(np.sum(np.abs(jerk))),
        "mean_abs_jerk": float(np.mean(np.abs(jerk))) if len(jerk) > 0 else 0.0,
        "max_lat_acc": float(np.max(speed * speed * kappa)),
    }


def _idm_desired_gap(v_ego: float, v_front: float, context: PlanningContext) -> float:
    """计算与行为层一致的 IDM 期望安全间距。"""

    idm_cfg = context.config["IDM"]
    max_acc = max(float(idm_cfg["max_acc"]), 1e-3)
    comfort_dec = max(abs(float(idm_cfg["comfort_dec"])), 1e-3)
    time_gap = float(idm_cfg["time_gap"])
    min_gap = float(idm_cfg["min_gap"])
    dv = float(v_ego) - float(v_front)
    dynamic_gap = float(v_ego) * time_gap + float(v_ego) * dv / (2.0 * math.sqrt(max_acc * comfort_dec))
    return min_gap + max(0.0, dynamic_gap)


def _collision_metrics(context: PlanningContext, traj: np.ndarray, obstacles: Iterable[ObstacleSnapshot]) -> dict[str, float]:
    """计算候选轨迹的粗几何碰撞与净空指标。

    输出字段：
    - ``collision``：是否预测到重叠碰撞
    - ``min_clearance``：整个轨迹上的最小粗矩形净空余量
    """

    ego_length, ego_width = _ego_box(context)
    min_margin = float("inf")
    collision = False

    for idx, point in enumerate(traj):
        t = idx * DT
        ego_xy = point[:2]
        for obstacle in obstacles:
            obs_xy = _predict_obstacle_position(obstacle, t)
            rel = np.asarray(ego_xy - obs_xy, dtype=float)
            cos_h = math.cos(-float(obstacle.heading))
            sin_h = math.sin(-float(obstacle.heading))
            rel_x = rel[0] * cos_h - rel[1] * sin_h
            rel_y = rel[0] * sin_h + rel[1] * cos_h

            obs_length, obs_width = _obstacle_box(obstacle)
            longitudinal_limit = 0.5 * (ego_length + obs_length) + 0.5
            lateral_limit = 0.5 * (ego_width + obs_width) + 0.2
            margin = max(abs(rel_x) - longitudinal_limit, abs(rel_y) - lateral_limit)
            min_margin = min(min_margin, margin)
            if margin < 0.0:
                collision = True

    if min_margin == float("inf"):
        min_margin = 999.0

    return {
        "collision": collision,
        "min_clearance": float(min_margin),
    }


def _trajectory_ttc_metrics(
    context: PlanningContext,
    traj: np.ndarray,
    obstacles: Iterable[ObstacleSnapshot],
) -> dict[str, float]:
    """计算轨迹上的最小 TTC 与最小停车余量。"""

    ego_length, ego_width = _ego_box(context)
    idm_cfg = context.config["IDM"]
    ttc_cfg = context.config["TTC"]
    longitudinal_buffer = float(idm_cfg.get("longitudinal_buffer", 0.5))
    lateral_buffer = float(idm_cfg.get("lateral_buffer", 0.3))
    closing_speed_eps = float(ttc_cfg["closing_speed_eps"])
    min_ttc = float("inf")
    min_stop_margin = float("inf")

    for idx, point in enumerate(traj):
        t = idx * DT
        ego_xy = np.asarray(point[:2], dtype=float)
        ego_heading = float(point[2])
        ego_v = max(float(point[3]), 0.0)
        tangent = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=float)
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        ego_v_vec = ego_v * tangent

        for obstacle in obstacles:
            obs_xy = _predict_obstacle_position(obstacle, t)
            rel = np.asarray(obs_xy - ego_xy, dtype=float)
            s_rel = float(np.dot(rel, tangent))
            if s_rel <= 0.0:
                continue

            obs_length, obs_width = _obstacle_box(obstacle)
            d_rel = float(np.dot(rel, normal))
            lateral_gate = 0.5 * (ego_width + obs_width) + lateral_buffer
            if abs(d_rel) > lateral_gate:
                continue

            obs_v_vec = np.array(
                [
                    float(obstacle.v) * math.cos(float(obstacle.heading)),
                    float(obstacle.v) * math.sin(float(obstacle.heading)),
                ],
                dtype=float,
            )
            closing_v = float(np.dot(ego_v_vec - obs_v_vec, tangent))
            gap_eff = s_rel - 0.5 * (ego_length + obs_length) - longitudinal_buffer
            obs_v_long = float(np.dot(obs_v_vec, tangent))
            stop_margin = gap_eff - _idm_desired_gap(ego_v, obs_v_long, context)
            min_stop_margin = min(min_stop_margin, stop_margin)

            if closing_v > closing_speed_eps:
                min_ttc = min(min_ttc, gap_eff / closing_v)

    if min_ttc == float("inf"):
        min_ttc = 999.0
    if min_stop_margin == float("inf"):
        min_stop_margin = 999.0

    return {
        "min_ttc": float(min_ttc),
        "min_stop_margin": float(min_stop_margin),
    }


def _front_margin_after_merge(context: PlanningContext, candidate: CandidateTrajectory) -> float:
    """估算候选并入目标线后，对前车的停车余量。

    返回正值表示：
        在考虑自车终端速度后，仍有足够余量停在目标线前车后面。
    """

    line_context = context.line_contexts[candidate.terminal.line_index]
    if line_context.front_obstacle is None:
        return 999.0

    traj = candidate.trajectory
    final_point = traj[-1]
    duration = max(0.0, DT * (len(traj) - 1))
    obs_xy = _predict_obstacle_position(line_context.front_obstacle.obstacle, duration)
    _, obs_s, _ = project_point_to_path(obs_xy, line_context.line)
    _, ego_s, _ = project_point_to_path(final_point[:2], line_context.line)
    gap = float(obs_s - ego_s)
    required_gap = max(6.0, float(final_point[3]) * 1.2 + float(final_point[3]) ** 2 / (2.0 * 2.5))
    return gap - required_gap


def _rear_margin_after_merge(context: PlanningContext, candidate: CandidateTrajectory) -> float:
    """估算换道完成后，目标车道后车的简单后向间距余量。"""

    line_context = context.line_contexts[candidate.terminal.line_index]
    if line_context.rear_obstacle is None:
        return 999.0

    traj = candidate.trajectory
    duration = max(0.0, DT * (len(traj) - 1))
    obs_xy = _predict_obstacle_position(line_context.rear_obstacle.obstacle, duration)
    _, obs_s, _ = project_point_to_path(obs_xy, line_context.line)
    _, ego_s, _ = project_point_to_path(traj[-1, :2], line_context.line)
    gap = float(ego_s - obs_s)
    required_gap = max(4.0, line_context.rear_obstacle.obstacle.v * 0.8)
    return gap - required_gap


def _rear_required_brake_after_merge(context: PlanningContext, candidate: CandidateTrajectory) -> float:
    """估算并入后目标车道后车所需的制动强度。

    这是在引入自由换道后新增的显式后车安全硬约束，
    相比单纯的后向 gap 阈值，它考虑了相对速度，因此信息更完整。
    """

    line_context = context.line_contexts[candidate.terminal.line_index]
    if line_context.rear_obstacle is None:
        return 0.0

    traj = candidate.trajectory
    duration = max(0.0, DT * (len(traj) - 1))
    obs_xy = _predict_obstacle_position(line_context.rear_obstacle.obstacle, duration)
    _, obs_s, _ = project_point_to_path(obs_xy, line_context.line)
    _, ego_s, _ = project_point_to_path(traj[-1, :2], line_context.line)
    gap = max(float(ego_s - obs_s) - 4.0, 1e-3)
    rear_v = float(line_context.rear_obstacle.obstacle.v)
    ego_v = float(traj[-1, 3])
    closing_v = max(rear_v - ego_v, 0.0)
    if closing_v <= 1e-6:
        return 0.0
    return closing_v * closing_v / (2.0 * gap)


def _within_deadline(context: PlanningContext, decision: BehaviorDecision, candidate: CandidateTrajectory) -> bool:
    """检查强制换道场景下，保留当前线候选是否仍留有足够后续窗口。"""

    if not decision.must_change:
        return True
    if candidate.terminal.line_index == decision.target_line_index:
        return True

    current_line = context.current_line
    travelled = max(candidate.terminal.target_s - current_line.current_s, 0.0)
    remaining_after = current_line.remaining_length - travelled
    reserve_margin = max(8.0, 1.5 * float(context.ego_state.v) + 4.0)
    return remaining_after >= reserve_margin


def _goal_progress_cost(context: PlanningContext, traj: np.ndarray) -> float:
    """候选轨迹终点到全局终点的距离代价。"""

    final_xy = traj[-1, :2]
    return float(np.linalg.norm(final_xy - context.goal_xy))


def _lane_change_delay_cost(context: PlanningContext, decision: BehaviorDecision, candidate: CandidateTrajectory) -> float:
    """在强制换道场景下，对继续拖延换道的候选加罚。"""

    if not decision.must_change:
        return 0.0
    if candidate.terminal.line_index == decision.target_line_index:
        return 0.0

    travelled = max(candidate.terminal.target_s - context.current_line.current_s, 0.0)
    slack_after = max(context.current_line.remaining_length - travelled, 0.0)
    urgency_ref = max(12.0, 2.5 * float(context.ego_state.v) + 6.0)
    return 10.0 * max(0.0, 1.0 - slack_after / max(urgency_ref, 1e-3))


def _ttc_cost(context: PlanningContext, min_ttc: float) -> float:
    """把最小 TTC 映射为归一化风险代价。"""

    ttc_cfg = context.config["TTC"]
    soft_min = float(ttc_cfg["soft_min"])
    soft_max = max(float(ttc_cfg["soft_max"]), soft_min + 1e-3)
    return 1.0 - _clamp01((min_ttc - soft_min) / (soft_max - soft_min))


def _stop_margin_cost(min_stop_margin: float) -> float:
    """把最小停车余量映射为归一化风险代价。"""

    return 1.0 - _clamp01(min_stop_margin / 10.0)


def _clearance_cost(min_clearance: float) -> float:
    """把最小净空映射为归一化风险代价。"""

    return 1.0 - _clamp01((min_clearance - 0.5) / (3.0 - 0.5))


def _mobil_cost(context: PlanningContext, decision: BehaviorDecision, candidate: CandidateTrajectory) -> float:
    """根据行为层的 MOBIL 收益，对 keep/换道候选施加连续效益代价。"""

    if decision.mode not in (BehaviorMode.PREPARE_DLC, BehaviorMode.EXECUTE_DLC):
        return 0.5

    incentive = max(0.0, float(decision.best_mobil_incentive))
    score_scale = max(float(context.config["MOBIL"].get("score_scale", 1.5)), 1e-3)
    direction_score = 0.5 * math.tanh(incentive / score_scale)

    if candidate.terminal.line_index == decision.target_line_index and decision.target_line_index != 0:
        return 1.0 - (0.5 + direction_score)
    if candidate.terminal.line_index == 0:
        return 1.0 - (0.5 - direction_score)
    return 0.5


def _efficiency_cost(
    context: PlanningContext,
    decision: BehaviorDecision,
    candidate: CandidateTrajectory,
    traj: np.ndarray,
) -> float:
    """计算纵向推进和速度利用率带来的归一化效率代价。"""

    line_context = context.line_contexts[candidate.terminal.line_index]
    traj_progress = max(float(candidate.terminal.target_s - line_context.current_s), 0.0)
    progress_score = _clamp01((traj_progress - 4.0) / (18.0 - 4.0))
    target_v_ref = max(float(context.config["constrain"].get("target_v", context.config["constrain"]["max_v"])), 1e-3)
    speed_score = _clamp01(float(traj[-1, 3]) / target_v_ref)
    cost = 1.0 - (0.6 * progress_score + 0.4 * speed_score)

    if decision.must_change and candidate.terminal.line_index != decision.target_line_index:
        delay_penalty = min(1.0, _lane_change_delay_cost(context, decision, candidate) / 10.0)
        cost = min(1.0, cost + 0.5 * delay_penalty)
    return cost


def _comfort_cost(context: PlanningContext, dyn: dict[str, float]) -> float:
    """根据横纵向舒适性指标计算归一化代价。"""

    score_cfg = context.config["SCORE"]
    lat_ratio = float(score_cfg.get("comfort_lat_ratio", 0.40))
    acc_ratio = float(score_cfg.get("comfort_acc_ratio", 0.35))
    jerk_ratio = float(score_cfg.get("comfort_jerk_ratio", 0.25))
    lat_limit = max(float(context.config["IDM"].get("curve_lat_acc_limit", 1.8)), 1e-3)
    lat_cost = _clamp01(dyn["max_lat_acc"] / lat_limit)
    acc_cost = _clamp01(dyn["max_abs_acc"] / 2.5)
    jerk_cost = _clamp01(dyn["mean_abs_jerk"] / 2.0)
    return lat_ratio * lat_cost + acc_ratio * acc_cost + jerk_ratio * jerk_cost


def _consistency_cost(context: PlanningContext, decision: BehaviorDecision, candidate: CandidateTrajectory) -> float:
    """根据与上一帧所选线的一致性计算归一化代价。"""

    last_line_index = int(context.planner_memory.last_selected_line_index)
    candidate_line_index = int(candidate.terminal.line_index)
    if candidate_line_index == last_line_index:
        return 0.0
    if decision.mode in (
        BehaviorMode.PREPARE_DLC,
        BehaviorMode.EXECUTE_DLC,
        BehaviorMode.PREPARE_MLC,
        BehaviorMode.EXECUTE_MLC,
    ) and candidate_line_index in (0, decision.target_line_index) and last_line_index in (0, decision.target_line_index):
        return 0.3
    return 0.7


def evaluate_candidates(
    context: PlanningContext,
    decision: BehaviorDecision,
    candidates: list[CandidateTrajectory],
) -> tuple[CandidateTrajectory | None, list[CandidateTrajectory]]:
    """对所有已生成候选轨迹做过滤与评分。

    输入：
        ``PlanningContext``、``BehaviorDecision``、候选轨迹列表

    输出：
        ``(best_candidate, annotated_candidates)``

    硬约束：
    - 动力学上限
    - 碰撞 / 净空
    - 并入后前车余量
    - 并入后后车余量
    - 并入后后车制动需求
    - 强制换道窗口约束

    软代价：
    - 舒适性（jerk / 加速度 / 曲率）
    - 接近终点程度
    - 强制换道拖延代价
    - 自由换道前向收益偏好
    - 与上一帧所选线的一致性
    """

    best_candidate: CandidateTrajectory | None = None
    best_score = float("inf")
    max_v_allowed = float(context.config["constrain"]["max_v"]) + 0.2
    max_acc_allowed = float(context.config["constrain"]["max_acc"]) + 0.2
    max_kappa_allowed = float(context.config["constrain"]["max_kappa"]) * 1.15
    ttc_hard_min = float(context.config["TTC"]["hard_min"])
    score_cfg = context.config["SCORE"]

    for candidate in candidates:
        if candidate.trajectory is None or len(candidate.trajectory) < 2:
            candidate.feasible = False
            if not candidate.failure_reason:
                candidate.failure_reason = "empty_trajectory"
            continue

        traj = np.asarray(candidate.trajectory, dtype=float)
        dyn = _trajectory_dynamic_metrics(traj)
        collision = _collision_metrics(context, traj, context.obstacle_snapshots)
        ttc_metrics = _trajectory_ttc_metrics(context, traj, context.obstacle_snapshots)
        front_margin = _front_margin_after_merge(context, candidate)
        rear_margin = _rear_margin_after_merge(context, candidate)
        rear_required_brake = _rear_required_brake_after_merge(context, candidate)
        deadline_ok = _within_deadline(context, decision, candidate)

        metrics = {
            **dyn,
            **collision,
            **ttc_metrics,
            "front_margin_after_merge": float(front_margin),
            "rear_margin_after_merge": float(rear_margin),
            "rear_required_brake_after_merge": float(rear_required_brake),
            "deadline_ok": bool(deadline_ok),
            "goal_cost": _goal_progress_cost(context, traj),
        }
        candidate.metrics.update(metrics)

        if dyn["max_v"] > max_v_allowed:
            candidate.feasible = False
            candidate.failure_reason = "overspeed"
            continue
        if dyn["max_abs_acc"] > max_acc_allowed:
            candidate.feasible = False
            candidate.failure_reason = "over_acc"
            continue
        if dyn["max_abs_kappa"] > max_kappa_allowed:
            candidate.feasible = False
            candidate.failure_reason = "over_kappa"
            continue
        if ttc_metrics["min_ttc"] < ttc_hard_min:
            candidate.feasible = False
            candidate.failure_reason = "ttc_too_small"
            continue
        if collision["collision"]:
            candidate.feasible = False
            candidate.failure_reason = "collision"
            continue
        if collision["min_clearance"] < 0.5:
            candidate.feasible = False
            candidate.failure_reason = "clearance_too_small"
            continue
        if candidate.terminal.line_index != 0 and front_margin < 0.0:
            candidate.feasible = False
            candidate.failure_reason = "front_margin_negative"
            continue
        if candidate.terminal.line_index != 0 and rear_margin < 0.0:
            candidate.feasible = False
            candidate.failure_reason = "rear_margin_negative"
            continue
        if candidate.terminal.line_index != 0 and rear_required_brake > 4.0:
            candidate.feasible = False
            candidate.failure_reason = "rear_brake_excessive"
            continue
        if not deadline_ok:
            candidate.feasible = False
            candidate.failure_reason = "deadline_violation"
            continue

        ttc_cost = _ttc_cost(context, ttc_metrics["min_ttc"])
        stop_margin_cost = _stop_margin_cost(ttc_metrics["min_stop_margin"])
        clearance_cost = _clearance_cost(collision["min_clearance"])
        mobil_cost = _mobil_cost(context, decision, candidate)
        efficiency_cost = _efficiency_cost(context, decision, candidate, traj)
        comfort_cost = _comfort_cost(context, dyn)
        consistency_cost = _consistency_cost(context, decision, candidate)

        candidate.metrics.update(
            {
                "ttc_cost": float(ttc_cost),
                "stop_margin_cost": float(stop_margin_cost),
                "clearance_cost": float(clearance_cost),
                "mobil_cost": float(mobil_cost),
                "efficiency_cost": float(efficiency_cost),
                "comfort_cost": float(comfort_cost),
                "consistency_cost": float(consistency_cost),
            }
        )

        score = (
            float(score_cfg["ttc"]) * ttc_cost
            + float(score_cfg["stop_margin"]) * stop_margin_cost
            + float(score_cfg["clearance"]) * clearance_cost
            + float(score_cfg["mobil"]) * mobil_cost
            + float(score_cfg["efficiency"]) * efficiency_cost
            + float(score_cfg["comfort"]) * comfort_cost
            + float(score_cfg["consistency"]) * consistency_cost
        )
        if decision.mode == BehaviorMode.APPROACH_GOAL:
            score += 10.0 * _clamp01(float(traj[-1, 3]) / 2.0)
        if decision.mode == BehaviorMode.EXECUTE_MLC and candidate.terminal.line_index != decision.target_line_index:
            score += 100.0
        if decision.mode == BehaviorMode.EXECUTE_DLC and candidate.terminal.line_index != decision.target_line_index:
            score += 100.0

        candidate.feasible = True
        candidate.score = float(score)

        if score < best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate, candidates
