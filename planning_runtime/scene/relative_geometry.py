"""独立于规划器向前裁剪参考线的车辆关系几何。

地图中心线及前驱/后继连接已遵循行驶方向。
始终以自车车道为公共坐标轴（不切换到主车车道，也不使用换道路径轨迹）。
路径 ID 仅用于消除纵向连接的歧义。
在未覆盖的边界处，沿端点切线进行局部外推。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


MAX_JOIN_GAP_M = 1.0
MIN_JOIN_TANGENT_DOT = -0.5


@dataclass(frozen=True)
class RelationReference:
    line: np.ndarray
    lane_ids: tuple[str, ...]
    source: str


def _clean_line(raw: Any) -> np.ndarray | None:
    line = np.asarray(raw, dtype=float)
    if line.ndim != 2 or line.shape[1] < 2 or len(line) < 2:
        return None
    line = line[:, :2]
    if not np.isfinite(line).all():
        return None
    keep = np.r_[True, np.linalg.norm(np.diff(line, axis=0), axis=1) > 1e-8]
    line = line[keep]
    return line if len(line) >= 2 else None


def _connects(before: np.ndarray, after: np.ndarray) -> bool:
    if float(np.linalg.norm(before[-1] - after[0])) > MAX_JOIN_GAP_M:
        return False
    a, b = before[-1] - before[-2], after[1] - after[0]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) >= MIN_JOIN_TANGENT_DOT


def build_relation_reference(
    parser: Any,
    ego_view: Any,
    planned_path: Sequence[str] = (),
    max_lane_segments: int = 12,
) -> RelationReference:
    """构建一条经过自车车道、向前后延伸且无分支的中心线。

    遇到有歧义的分叉、重复 ID、缺失几何或无效连接时停止。
    显式路径只有在对应真实纵向拓扑连接时才能用于选择分支，
    不能仅凭全局路径中出现相邻车道就选择该车道。
    后向候选合并显式 predecessor 与已知 successor 的反向边，兼容缺失的反向索引；
    所有候选仍须通过几何连接校验和原有路径消歧，不修改原始地图。
    """
    centre_lines = getattr(parser, 'lane_centerlines', {})
    lane_info = getattr(parser, 'lanes', {})
    current = str(ego_view.lane_id)
    initial = _clean_line(centre_lines.get(current, []))
    if initial is None:
        fallback = _clean_line(ego_view.line)
        return RelationReference(np.empty((0, 2)) if fallback is None else fallback,
                                 (), 'planning_reference' if fallback is not None else 'heading_fallback')
    geometries = {current: initial}
    # 局部延伸逻辑可能在分叉处选择第一个后继车道。
    # 有效的显式路径比这种临时局部序列更可靠。
    paths = (tuple(planned_path), tuple(getattr(ego_view, 'lane_sequence', ()) or ()))
    visited = {current}
    before, after = [], []
    inferred_predecessors: dict[str, list[str]] = {}
    for lane_id, info in lane_info.items():
        for successor in info.get('successor', []) or []:
            inferred_predecessors.setdefault(str(successor), []).append(str(lane_id))

    def next_lane(lane_id: str, direction: str) -> str | None:
        options = []
        candidates = list(lane_info.get(lane_id, {}).get(direction, []) or [])
        if direction == 'predecessor':
            candidates.extend(inferred_predecessors.get(lane_id, []))
        for candidate in candidates:
            candidate = str(candidate)
            if candidate in visited or candidate in options:
                continue
            geometry = _clean_line(centre_lines.get(candidate, []))
            if geometry is None:
                continue
            left, right = ((geometry, geometries[lane_id]) if direction == 'predecessor'
                           else (geometries[lane_id], geometry))
            if _connects(left, right):
                geometries[candidate] = geometry
                options.append(candidate)
        offset = -1 if direction == 'predecessor' else 1
        for path in paths:
            preferred = {path[i+offset] for i, item in enumerate(path)
                         if item == lane_id and 0 <= i+offset < len(path)
                         and path[i+offset] in options}
            if len(preferred) == 1:
                return next(iter(preferred))
            if len(preferred) > 1:
                return None
        return options[0] if len(options) == 1 else None

    for direction, output in (('predecessor', before), ('successor', after)):
        lane_id = current
        for _ in range(max(0, int(max_lane_segments) - 1)):
            candidate = next_lane(lane_id, direction)
            if candidate is None:
                break
            output.append(candidate)
            visited.add(candidate)
            lane_id = candidate
    ids = tuple(reversed(before)) + (current,) + tuple(after)
    merged = []
    for lane_id in ids:
        points = geometries[lane_id]
        if merged and np.linalg.norm(merged[-1] - points[0]) <= 1e-8:
            points = points[1:]
        merged.extend(points)
    return RelationReference(np.asarray(merged, dtype=float), ids, 'lane_corridor')


def project_signed_on_reference(point: np.ndarray, line: np.ndarray) -> tuple[float, float]:
    """计算弧长位置和有符号法向偏移，仅在端点处外推。

    先在有限折线上选择最近点。仅当该点是首尾端点且查询点位于端点外侧时，
    才能沿其切线延伸 s，避免远端切线取代曲线内部的有效投影。
    d 表示法向分量，而不是到截断端点的欧氏距离。
    """
    clean = _clean_line(line)
    if clean is None:
        raise ValueError('Relationship reference needs two distinct finite points')
    point = np.asarray(point, dtype=float)[:2]
    if not np.isfinite(point).all():
        raise ValueError('Relationship query point must be finite')
    starts, segments = clean[:-1], np.diff(clean, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    raw_t = np.einsum('ij,ij->i', point-starts, segments) / lengths**2
    t = np.clip(raw_t, 0., 1.)
    feet = starts + t[:, None] * segments
    idx = int(np.argmin(np.sum((feet-point)**2, axis=1)))
    if (idx == 0 and raw_t[idx] < 0.) or (idx == len(segments)-1 and raw_t[idx] > 1.):
        t[idx] = raw_t[idx]
    tangent = segments[idx] / lengths[idx]
    foot = starts[idx] + t[idx]*segments[idx]
    offset = point-foot
    longitudinal = float(np.sum(lengths[:idx]) + t[idx]*lengths[idx])
    lateral = float(tangent[0]*offset[1] - tangent[1]*offset[0])
    return longitudinal, lateral


def relative_lane_coordinates(
    parser: Any,
    ego_vehicle: Any,
    main_xy: np.ndarray,
    planned_path: Sequence[str] = (),
    max_lane_segments: int = 12,
) -> tuple[float, float]:
    reference = build_relation_reference(parser, ego_vehicle.current_ref_line,
                                         planned_path, max_lane_segments)
    ego_xy = np.array([ego_vehicle.x, ego_vehicle.y], dtype=float)
    if reference.source == 'heading_fallback':
        delta = np.asarray(main_xy, dtype=float)[:2]-ego_xy
        c, s = np.cos(ego_vehicle.heading), np.sin(ego_vehicle.heading)
        return float(c*delta[0]+s*delta[1]), float(-s*delta[0]+c*delta[1])
    ego_s, _ = project_signed_on_reference(ego_xy, reference.line)
    main_s, main_d = project_signed_on_reference(main_xy, reference.line)
    return main_s-ego_s, main_d
