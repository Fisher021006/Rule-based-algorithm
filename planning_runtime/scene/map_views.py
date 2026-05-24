"""场景语义参考线视图辅助函数。
底层 referline/车道数据-> scene 算法能理解的参考线语义"""

from __future__ import annotations

import math

import numpy as np

from planning_runtime.frame import PlanningFrame
from planning_runtime.scene.types import ReferenceLineView
from traj_planner.tools import calculate_path_cumulative_distance, project_point_to_path
from xodr_parser.xodr_parser import XODRParser


def initial_heading_from_line(line: np.ndarray) -> float:
    """根据参考线前两个点估计初始航向角。"""

    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        raise ValueError("参考线点数不足，无法估计航向")
    vec = line[1, :2] - line[0, :2]
    return float(math.atan2(vec[1], vec[0]))


def slice_centerline_from_projection(centerline: np.ndarray, query_point: np.ndarray) -> np.ndarray:
    """从 `query_point` 在线上的投影位置开始向前截取中心线。"""

    centerline = np.asarray(centerline, dtype=float)
    query_point = np.asarray(query_point, dtype=float)
    if len(centerline) < 2:
        return centerline[:, :2].copy()

    best_idx = 0
    best_proj = centerline[0, :2].copy()
    min_dist = float("inf")

    for idx in range(len(centerline) - 1):
        p1 = centerline[idx, :2]
        p2 = centerline[idx + 1, :2]
        seg = p2 - p1
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-12:
            continue
        t = float(np.clip(np.dot(query_point - p1, seg) / seg_len_sq, 0.0, 1.0))
        proj = p1 + t * seg
        dist = float(np.linalg.norm(query_point - proj))
        if dist < min_dist:
            min_dist = dist
            best_idx = idx
            best_proj = proj

    sliced = [best_proj.tolist()]
    start_idx = best_idx + 1
    if np.linalg.norm(centerline[best_idx + 1, :2] - best_proj) < 1e-6:
        start_idx = best_idx + 2
    for idx in range(start_idx, len(centerline)):
        sliced.append(centerline[idx, :2].tolist())
    if len(sliced) < 2:
        sliced.append(centerline[-1, :2].tolist())
    return np.asarray(sliced, dtype=float)


def build_line_from_lane_sequence(
    parser: XODRParser,
    lane_sequence: list[str],
    query_point: np.ndarray,
    max_lane_segments: int,
) -> tuple[np.ndarray, list[str]]:
    """把多个车道段拼成一条面向场景算法的参考线。"""

    limited_sequence = list(lane_sequence[:max_lane_segments])
    if not limited_sequence:
        return np.empty((0, 2), dtype=float), []

    merged_points: list[list[float]] = []
    for idx, lane_id in enumerate(limited_sequence):
        centerline = np.asarray(parser.lane_centerlines.get(lane_id, []), dtype=float)
        if centerline.ndim != 2 or len(centerline) == 0:
            continue
        if idx == 0:
            points_to_add = slice_centerline_from_projection(centerline, query_point)
        else:
            points_to_add = centerline[:, :2].copy()
            if merged_points and np.linalg.norm(points_to_add[0, :2] - np.asarray(merged_points[-1])[:2]) < 1e-6:
                points_to_add = points_to_add[1:]
        for point in points_to_add:
            merged_points.append(point[:2].tolist())

    line = np.asarray(merged_points, dtype=float)
    return line, limited_sequence


def compute_lateral_steps(parser: XODRParser, lane_ids: list[str]) -> list[int]:
    """计算各候选车道相对当前主车道的横向阶数。"""

    if not lane_ids:
        return []
    current_lane_id = lane_ids[0]
    steps = {current_lane_id: 0}
    queue = [current_lane_id]

    while queue:
        lane_id = queue.pop(0)
        lane_info = parser.lanes.get(lane_id)
        if not lane_info:
            continue
        current_step = steps[lane_id]

        left_lane = lane_info.get("left_lane")
        if left_lane and left_lane not in steps:
            steps[left_lane] = current_step - 1
            queue.append(left_lane)

        right_lane = lane_info.get("right_lane")
        if right_lane and right_lane not in steps:
            steps[right_lane] = current_step + 1
            queue.append(right_lane)

    fallback_step = max(len(lane_ids), 2)
    return [int(steps.get(lane_id, fallback_step)) for lane_id in lane_ids]


def extend_lane_forward(parser: XODRParser, start_lane_id: str, max_lane_segments: int) -> list[str]:
    """从起始车道出发，沿 successor 关系向前延伸。"""

    lane_sequence = [start_lane_id]
    current_lane_id = start_lane_id

    for _ in range(max(max_lane_segments - 1, 0)):
        lane_info = parser.lanes.get(current_lane_id)
        if not lane_info:
            break
        successors = [lane_id for lane_id in lane_info.get("successor", []) if lane_id in parser.lanes]
        if not successors:
            break

        current_lane_num = int(current_lane_id.split(".")[-1])
        next_lane_id = None
        for successor_lane_id in successors:
            if int(successor_lane_id.split(".")[-1]) == current_lane_num:
                next_lane_id = successor_lane_id
                break
        if next_lane_id is None:
            next_lane_id = successors[0]

        lane_sequence.append(next_lane_id)
        current_lane_id = next_lane_id

    return lane_sequence


def build_reference_views_from_frame(
    planning_frame: PlanningFrame,
    query_point: np.ndarray,
) -> list[ReferenceLineView]:
    """根据统一规划帧构造自车参考线视图。"""

    return build_reference_views_from_refer_lines(
        refer_lines=planning_frame.refer_lines,
        refer_line_lane_ids=planning_frame.refer_line_lane_ids,
        refer_line_lane_sequences=planning_frame.refer_line_lane_sequences,
        refer_line_lateral_steps=planning_frame.refer_line_lateral_steps,
        query_point=query_point,
    )


def build_reference_views_from_refer_lines(
    refer_lines: list[np.ndarray],
    refer_line_lane_ids: list[str],
    refer_line_lane_sequences: list[list[str]],
    refer_line_lateral_steps: list[int],
    query_point: np.ndarray,
) -> list[ReferenceLineView]:
    """根据一组已知参考线及语义标签构造视图。"""

    views: list[ReferenceLineView] = []
    for line_index, raw_line in enumerate(refer_lines):
        line = np.asarray(raw_line, dtype=float)
        if line.ndim != 2 or len(line) < 2:
            continue
        line_xy = np.asarray(line[:, :2], dtype=float)
        cumulative_s = calculate_path_cumulative_distance(line_xy)
        _, current_s, current_d = project_point_to_path(query_point, line_xy)
        lane_id = (
            str(refer_line_lane_ids[line_index])
            if line_index < len(refer_line_lane_ids)
            else ""
        )
        lane_sequence = (
            list(refer_line_lane_sequences[line_index])
            if line_index < len(refer_line_lane_sequences)
            else ([lane_id] if lane_id else [])
        )
        lateral_step = (
            int(refer_line_lateral_steps[line_index])
            if line_index < len(refer_line_lateral_steps)
            else 0
        )
        views.append(
            ReferenceLineView(
                line_index=line_index,
                lane_id=lane_id,
                lane_sequence=lane_sequence,
                lateral_step=lateral_step,
                line=line_xy,
                cumulative_s=cumulative_s,
                current_s=float(current_s),
                current_d=float(current_d),
            )
        )
    if not views:
        raise ValueError("自车参考线为空，无法构造场景算法上下文")
    return views


def build_reference_views_from_lane_sequences(
    parser: XODRParser,
    lane_sequences: list[list[str]],
    query_point: np.ndarray,
    max_lane_segments: int,
    lateral_steps: list[int] | None = None,
) -> list[ReferenceLineView]:
    """根据 lane 序列集合，从给定查询点重建参考线视图。"""

    lane_ids = [lane_seq[0] for lane_seq in lane_sequences if lane_seq]
    resolved_lateral_steps = list(lateral_steps) if lateral_steps is not None else compute_lateral_steps(parser, lane_ids)
    views: list[ReferenceLineView] = []

    for line_index, lane_sequence in enumerate(lane_sequences):
        if not lane_sequence:
            continue
        line, limited_sequence = build_line_from_lane_sequence(
            parser=parser,
            lane_sequence=list(lane_sequence),
            query_point=query_point,
            max_lane_segments=max_lane_segments,
        )
        if len(line) < 2:
            continue
        cumulative_s = calculate_path_cumulative_distance(line)
        _, current_s, current_d = project_point_to_path(query_point, line)
        views.append(
            ReferenceLineView(
                line_index=line_index,
                lane_id=str(lane_sequence[0]),
                lane_sequence=limited_sequence,
                lateral_step=int(resolved_lateral_steps[line_index]) if line_index < len(resolved_lateral_steps) else 0,
                line=line,
                cumulative_s=cumulative_s,
                current_s=float(current_s),
                current_d=float(current_d),
            )
        )
    if not views:
        raise ValueError("参考线序列为空，无法构造场景算法上下文")
    return views


# Compatibility alias for the current transition period.
RuntimeFrame = PlanningFrame
build_reference_views_from_runtime_frame = build_reference_views_from_frame


def build_reference_views_from_extender_result(
    parser: XODRParser,
    extended_lane_ids: list[list[str]],
    query_point: np.ndarray,
    max_lane_segments: int,
) -> list[ReferenceLineView]:
    """根据车道延伸结果构造主车参考线视图。"""
    return build_reference_views_from_lane_sequences(
        parser=parser,
        lane_sequences=extended_lane_ids,
        query_point=query_point,
        max_lane_segments=max_lane_segments,
    )
