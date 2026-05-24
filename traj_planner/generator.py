"""把终端状态候选转换成真实六元组轨迹。

这个模块是“新阶段 1 采样框架”和“旧下半段轨迹生成器”之间的桥。

上游：
    从 ``sampling.py`` 接收 ``TerminalCandidate``。
下游：
    向 ``evaluator.py`` 返回 ``CandidateTrajectory``。

复用策略：
- 几何路径生成仍然来自 ``path.py``
- 时参仍然来自 ``trajectory.py``
- 阶段 1 只改变“候选终端状态怎么提出、怎么选优”
"""

from __future__ import annotations

import numpy as np

from traj_planner.state_rules import lane_change_completion_threshold
from traj_planner.path import bezier_curve_kappa, compute_heading_from_path
from traj_planner.tools import calculate_path_cumulative_distance, get_path_point_at_s
from traj_planner.trajectory import optimize_time_param_traj, quintic_time_param_traj
from traj_planner.types import CandidateTrajectory, PlanningContext, TerminalCandidate

MIN_LEADING_POINT_SPACING = 0.3
TIME_PARAM_DT = 0.1


def _same_line_recentering_threshold(context: PlanningContext) -> float:
    """返回同线平滑回中心的触发阈值。"""

    return max(0.05, 0.2 * lane_change_completion_threshold(context.config))


def _needs_same_line_recentering(context: PlanningContext, candidate: TerminalCandidate) -> bool:
    """判断同线候选是否仍需要一段回中心的光滑并线几何。"""

    if candidate.line_index != 0:
        return False
    return abs(context.current_line.current_d) > _same_line_recentering_threshold(context)


def _snap_plan_time(plan_T: float, dt: float = TIME_PARAM_DT) -> float:
    """把规划时域吸附到统一采样周期上。"""

    return max(float(dt), round(float(plan_T) / float(dt)) * float(dt))


def _clip_quintic_end_v(path_length: float, plan_T: float, start_v: float, end_v: float) -> float:
    """对五次多项式的终点速度做保守裁剪。"""

    return max(
        0.0,
        min(
            float(end_v),
            max(0.0, 2.0 * float(path_length) / max(float(plan_T), 1e-3) - float(start_v)),
        ),
    )


def _solve_quintic_time_param_trajectory(
    context: PlanningContext,
    smooth_path: np.ndarray,
    plan_T: float,
    start_v: float,
    start_a: float,
    end_v: float,
    end_a: float = 0.0,
) -> np.ndarray:
    """对一条几何路径运行五次多项式时参。"""

    del context

    snapped_t_plan = _snap_plan_time(plan_T)
    path_length = float(calculate_path_cumulative_distance(smooth_path)[-1]) if len(smooth_path) > 1 else 0.0
    feasible_end_v = _clip_quintic_end_v(path_length, snapped_t_plan, start_v, end_v)
    return np.asarray(
        quintic_time_param_traj(
            smooth_path,
            start_v,
            start_a,
            feasible_end_v,
            end_a,
            snapped_t_plan,
            dt=TIME_PARAM_DT,
        ),
        dtype=float,
    )


def _solve_osqp_time_param_trajectory(
    context: PlanningContext,
    smooth_path: np.ndarray,
    plan_T: float,
    start_v: float,
    start_a: float,
    end_v: float,
    end_a: float = 0.0,
) -> np.ndarray:
    """对一条几何路径运行 OSQP 时参优化。"""

    snapped_t_plan = _snap_plan_time(plan_T)
    a_max = float(context.config["constrain"]["max_acc"])
    a_min = float(context.config["constrain"]["min_acc"])
    return np.asarray(
        optimize_time_param_traj(
            smooth_path,
            snapped_t_plan,
            start_v,
            start_a,
            end_v,
            end_a,
            a_max=a_max,
            a_min=a_min,
            v_max=float(context.config["constrain"]["max_v"]),
        ),
        dtype=float,
    )


def _extract_line_segment(
    line: np.ndarray,
    cumulative_s: np.ndarray,
    start_s: float,
    end_s: float,
    start_heading: float,
    start_xy_override: np.ndarray | None = None,
) -> np.ndarray:
    """从一条参考线中截取一段前向几何路径。

    输入：
        ``line`` 及其累计弧长数组，
        ``start_s`` / ``end_s`` 定义保留区间。

    输出：
        ``np.ndarray``，shape ``(M, 3)``
        列顺序：``[x, y, heading]``

    用途：
        - 当前车道保持候选
        - “先保持一段再换道”中的前段保持路径
    """

    start_s = float(max(start_s, 0.0))
    end_s = float(min(end_s, cumulative_s[-1]))
    if end_s <= start_s:
        end_s = min(float(cumulative_s[-1]), start_s + 1.0)

    start_pos, start_path_heading = get_path_point_at_s(line, start_s, cumulative_s)
    end_pos, end_path_heading = get_path_point_at_s(line, end_s, cumulative_s)

    interior_mask = (cumulative_s > start_s) & (cumulative_s < end_s)
    interior_points = line[interior_mask, :2]
    first_point = np.asarray(start_xy_override, dtype=float) if start_xy_override is not None else np.asarray(start_pos, dtype=float)
    path_xy = [first_point]
    if len(interior_points) > 0:
        path_xy.extend(np.asarray(interior_points, dtype=float))
    path_xy.append(np.asarray(end_pos, dtype=float))
    path_xy = np.asarray(path_xy, dtype=float)

    dedup_points = [path_xy[0]]
    for point in path_xy[1:]:
        if np.linalg.norm(point - dedup_points[-1]) > 1e-6:
            dedup_points.append(point)
    path_xy = np.asarray(dedup_points, dtype=float)

    if len(path_xy) < 2:
        path_xy = np.vstack([path_xy[0], np.asarray(end_pos, dtype=float)])

    # 若真实起点与第一个参考线点过近，则跳过这些过密前导点，
    # 避免在起点附近形成极短折段并放大曲率估计。
    if len(path_xy) > 2:
        first_point = np.asarray(path_xy[0], dtype=float)
        first_valid_idx = 1
        while first_valid_idx < len(path_xy) - 1:
            if np.linalg.norm(np.asarray(path_xy[first_valid_idx], dtype=float) - first_point) >= MIN_LEADING_POINT_SPACING:
                break
            first_valid_idx += 1
        if first_valid_idx > 1:
            path_xy = np.vstack([first_point, path_xy[first_valid_idx:]])

    return compute_heading_from_path(path_xy, start_heading, end_path_heading)


def _concat_paths(first_path: np.ndarray, second_path: np.ndarray) -> np.ndarray:
    """拼接两段路径，并去掉交界处的单点重叠。"""

    if len(first_path) == 0:
        return second_path
    if len(second_path) == 0:
        return first_path
    if np.linalg.norm(first_path[-1, :2] - second_path[0, :2]) < 1e-6:
        return np.vstack([first_path, second_path[1:]])
    return np.vstack([first_path, second_path])


def build_candidate_smooth_path(context: PlanningContext, candidate: TerminalCandidate) -> CandidateTrajectory:
    """为一个终端状态候选生成时参前的几何路径。

    输入：
        ``PlanningContext`` 和 ``TerminalCandidate``

    输出：
        ``CandidateTrajectory``，其中包含：
        - ``smooth_path``：时参前的几何路径

    生成模式：
        - 同线候选：
            从当前主线截取一段前向路径
        - 直接换道候选：
            从当前姿态直接向目标线终端点连一条 Bezier 几何路径
        - 延迟换道候选：
            先沿当前线保持，再连接到目标线
    """

    ego = context.ego_state
    current_line = context.current_line
    target_line = context.line_contexts[candidate.line_index]
    start_pos = np.array([ego.x, ego.y], dtype=float)
    start_heading = float(ego.heading)
    start_v = float(ego.v)

    try:
        target_pos, _ = get_path_point_at_s(target_line.line, candidate.target_s, target_line.cumulative_s)
        target_pos = np.asarray(target_pos, dtype=float)

        if candidate.line_index == 0:
            if _needs_same_line_recentering(context, candidate):
                smooth_path = np.asarray(
                    bezier_curve_kappa(current_line.line, start_pos, target_pos, start_heading, end_theta=None),
                    dtype=float,
                )
            else:
                smooth_path = _extract_line_segment(
                    current_line.line,
                    current_line.cumulative_s,
                    current_line.current_s,
                    candidate.target_s,
                    start_heading,
                    start_xy_override=start_pos,
                )
        elif candidate.hold_t > 1e-3:
            hold_end_s = min(
                current_line.cumulative_s[-1] - 1.0,
                current_line.current_s + max(start_v * candidate.hold_t, 2.0),
            )
            hold_path = _extract_line_segment(
                current_line.line,
                current_line.cumulative_s,
                current_line.current_s,
                hold_end_s,
                start_heading,
                start_xy_override=start_pos,
            )
            hold_end_point = np.asarray(hold_path[-1, :2], dtype=float)
            hold_end_heading = float(hold_path[-1, 2])
            lane_change_path = np.asarray(
                bezier_curve_kappa(target_line.line, hold_end_point, target_pos, hold_end_heading, end_theta=None),
                dtype=float,
            )
            smooth_path = _concat_paths(hold_path, lane_change_path)
        else:
            smooth_path = np.asarray(
                bezier_curve_kappa(target_line.line, start_pos, target_pos, start_heading, end_theta=None),
                dtype=float,
            )

        if smooth_path.ndim != 2 or smooth_path.shape[0] < 2:
            return CandidateTrajectory(
                terminal=candidate,
                smooth_path=smooth_path,
                feasible=False,
                failure_reason="invalid_smooth_path",
            )

        return CandidateTrajectory(
            terminal=candidate,
            smooth_path=np.asarray(smooth_path, dtype=float),
            feasible=True,
        )
    except Exception as exc:
        return CandidateTrajectory(
            terminal=candidate,
            feasible=False,
            failure_reason=f"generator_exception:{exc}",
        )


def apply_time_param_to_candidate(
    context: PlanningContext,
    candidate: CandidateTrajectory,
    method: str = "quintic",
) -> CandidateTrajectory:
    """把已有 ``smooth_path`` 的候选转换成真实六元组轨迹。"""

    smooth_path = None if candidate.smooth_path is None else np.asarray(candidate.smooth_path, dtype=float)
    if smooth_path is None or smooth_path.ndim != 2 or smooth_path.shape[0] < 2:
        return CandidateTrajectory(
            terminal=candidate.terminal,
            smooth_path=smooth_path,
            feasible=False,
            metrics=dict(candidate.metrics),
            failure_reason=candidate.failure_reason or "invalid_smooth_path",
        )

    start_v = float(context.ego_state.v)
    start_a = float(context.ego_state.a)

    try:
        if method == "quintic":
            trajectory = _solve_quintic_time_param_trajectory(
                context=context,
                smooth_path=smooth_path,
                plan_T=candidate.terminal.plan_T,
                start_v=start_v,
                start_a=start_a,
                end_v=float(candidate.terminal.target_v),
                end_a=0.0,
            )
        elif method == "osqp":
            trajectory = _solve_osqp_time_param_trajectory(
                context=context,
                smooth_path=smooth_path,
                plan_T=candidate.terminal.plan_T,
                start_v=start_v,
                start_a=start_a,
                end_v=float(candidate.terminal.target_v),
                end_a=0.0,
            )
        else:
            raise ValueError(f"Unsupported time-param method: {method}")
    except Exception as exc:
        return CandidateTrajectory(
            terminal=candidate.terminal,
            smooth_path=smooth_path,
            feasible=False,
            metrics=dict(candidate.metrics),
            failure_reason=f"{method}_time_param_exception:{exc}",
        )

    result = CandidateTrajectory(
        terminal=candidate.terminal,
        trajectory=np.asarray(trajectory, dtype=float),
        smooth_path=smooth_path,
        feasible=True,
        metrics=dict(candidate.metrics),
    )
    result.metrics["time_param_method"] = method
    return result
