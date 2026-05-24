"""
可行驶车道延伸输出模块

功能：
1. 基于最短路径规划输出所有车道所在的路段ID
2. 根据变道约束和车道线类型筛选可行驶车道
3. 给定坐标定位路段，输出该位置可向后延伸的所有车道中心线坐标

用法：
    python drivable_lanes_extender.py --start 258,290 --end 207,365 --query 245,310
    python drivable_lanes_extender.py --start 258,290 --end 207,365 --query 245,310 --min-change-length 15
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .config import LaneChangeConfig, DrivableLanesResult
from global_planner.planner import GlobalPlanner


def _truncate_at_goal(centerline: List[List[float]], goal_point: np.ndarray) -> List[List[float]]:
    """
    在终点处截断中心线

    参数:
        centerline: 中心线点列表 [[x, y], ...]
        goal_point: 终点坐标 [x, y]

    返回:
        截断后的中心线
    """
    if len(centerline) < 2:
        return centerline

    centerline_array = np.array(centerline)
    distances = np.linalg.norm(centerline_array[:, :2] - goal_point, axis=1)
    closest_idx = int(np.argmin(distances))

    # 截取从起点到最近点（包含最近点）
    truncated = centerline[:closest_idx + 1]

    # 如果截断后太短，至少保留2个点
    if len(truncated) < 2:
        truncated = centerline[:2]

    return truncated


class DrivableLanesExtender:
    """
    可行驶车道延伸器

    功能：
    1. 基于最短路规划提取路段和车道
    2. 根据变道约束筛选可行驶车道
    3. 定位查询坐标并输出延伸的车道中心线
    """

    def __init__(self, xodr_parser, config: LaneChangeConfig = None):
        """
            xodr_parser: XODR解析器实例
            config: 变道筛选配置
        """
        self.parser = xodr_parser
        self.config = config or LaneChangeConfig()

        # 规划结果
        self.planner = None
        self.road_sequence: List[str] = []
        self.planned_path: List[str] = []
        self.all_lanes_per_road: Dict[str, List[str]] = {}
        self.filtered_drivable_lanes: Dict[str, Set[str]] = {}
        self.goal_point: Optional[np.ndarray] = None  # 终点坐标
        self.active_road_start_idx: int = 0  # 已通过路段将被永久剔除
        self._change_corridor_cache: Dict[Tuple[str, str, str], float] = {}

        # 构建KDTree用于坐标查询
        self._build_kdtree()

    def _build_kdtree(self):
        """构建KDTree用于坐标定位"""
        from scipy.spatial import KDTree

        all_points = []
        self.lane_point_indices = []  # 存储每个点对应的车道ID

        for lane_id, centerline in self.parser.lane_centerlines.items():
            lane_info = self.parser.lanes.get(lane_id)
            if not lane_info or lane_info['type'] != 'driving':
                continue
            for point in centerline:
                all_points.append(point)
                self.lane_point_indices.append(lane_id)

        if all_points:
            self.kdtree = KDTree(np.array(all_points))
            self.lane_point_indices = np.array(self.lane_point_indices)
        else:
            self.kdtree = None

    def plan_shortest_path(self, start_x: float, start_y: float,
                           end_x: float, end_y: float) -> bool:

        # 保存终点坐标
        self.goal_point = np.array([end_x, end_y])

        # 创建规划器并执行规划
        self.planner = GlobalPlanner(self.parser)
        success = self.planner.plan(start_x, start_y, end_x, end_y)
        self.active_road_start_idx = 0
        self._change_corridor_cache.clear()

        if success:
            self.road_sequence = self.planner.road_sequence.copy()
            self.planned_path = self.planner.planned_path.copy()

            # 提取每个路段的所有车道
            self._extract_all_lanes_per_road()
            # 筛选可行驶车道
            self.filter_drivable_lanes()
            # print(f"\n✅ 规划成功! 路段序列: {' -> '.join(self.road_sequence)}")
        else:
            print("\n❌ 规划失败!")

        return success

    def _extract_all_lanes_per_road(self):
        """提取每个路段的所有driving车道"""
        self.all_lanes_per_road = {}

        for road_id in self.road_sequence:
            lanes = []
            for lane_id, info in self.parser.lanes.items():
                if info['road_id'] == road_id and info['type'] == 'driving':
                    lanes.append(lane_id)
            # 按车道号排序
            lanes.sort(key=lambda x: int(x.split('.')[-1]))
            self.all_lanes_per_road[road_id] = lanes

    def filter_drivable_lanes(self) -> Dict[str, Set[str]]:
        """
        根据配置的条件筛选可行驶车道

        筛选条件：
        1. 变道长度约束
        2. 车道线类型约束（虚实线）

        返回:
            筛选后的可行驶车道字典 {road_id: set of lane_ids}
        """
        if not self.road_sequence:
            self.filtered_drivable_lanes = {}
            return self.filtered_drivable_lanes

        planned_lanes_by_road: Dict[str, Set[str]] = {road_id: set() for road_id in self.road_sequence}
        for lane_id in self.planned_path:
            road_id = lane_id.split('.')[0]
            if road_id in planned_lanes_by_road:
                planned_lanes_by_road[road_id].add(lane_id)

        # 第一步：每个 road 内部，基于计划车道做横向扩展，得到“本地可达闭包”
        local_drivable_by_road: Dict[str, Set[str]] = {}
        for road_id in self.road_sequence:
            local_drivable_by_road[road_id] = self._expand_lanes_within_road(
                road_id, planned_lanes_by_road.get(road_id, set())
            )

        # 第二步：前向传播，只保留从起点按 successor 真正能走到的车道
        forward_reachable: Dict[str, Set[str]] = {}
        first_road_id = self.road_sequence[0]
        forward_reachable[first_road_id] = local_drivable_by_road.get(first_road_id, set()).copy()

        for i in range(1, len(self.road_sequence)):
            road_id = self.road_sequence[i]
            prev_road_id = self.road_sequence[i - 1]
            entry_lanes = self._collect_successor_entries(
                forward_reachable.get(prev_road_id, set()), road_id
            )
            allowed_lanes = local_drivable_by_road.get(road_id, set())
            forward_reachable[road_id] = self._expand_lanes_within_road(
                road_id, entry_lanes & allowed_lanes, allowed_lanes=allowed_lanes
            )

        # 第三步：后向剪枝，只保留最终还能到终点车道的车道
        backward_reachable: Dict[str, Set[str]] = {}
        last_road_id = self.road_sequence[-1]
        last_goal_lanes = planned_lanes_by_road.get(last_road_id, set())
        backward_reachable[last_road_id] = self._reverse_expand_lanes_within_road(
            last_road_id, last_goal_lanes, allowed_lanes=local_drivable_by_road.get(last_road_id, set())
        )

        for i in range(len(self.road_sequence) - 2, -1, -1):
            road_id = self.road_sequence[i]
            next_road_id = self.road_sequence[i + 1]
            allowed_lanes = local_drivable_by_road.get(road_id, set())
            exit_lanes = {
                lane_id for lane_id in allowed_lanes
                if any(succ in backward_reachable.get(next_road_id, set())
                       for succ in self.parser.lanes.get(lane_id, {}).get('successor', []))
            }
            backward_reachable[road_id] = self._reverse_expand_lanes_within_road(
                road_id, exit_lanes, allowed_lanes=allowed_lanes
            )

        self.filtered_drivable_lanes = {}
        for road_id in self.road_sequence:
            self.filtered_drivable_lanes[road_id] = (
                forward_reachable.get(road_id, set()) &
                backward_reachable.get(road_id, set())
            )

        return self.filtered_drivable_lanes

    def _expand_lanes_within_road(self, road_id: str, source_lanes: Set[str],
                                  allowed_lanes: Optional[Set[str]] = None) -> Set[str]:
        """
        在单个 road 内，从 source_lanes 出发按合法换道关系做前向闭包。
        """
        if allowed_lanes is None:
            allowed = set(self.all_lanes_per_road.get(road_id, []))
        else:
            allowed = set(allowed_lanes)

        seeds = [lane_id for lane_id in source_lanes if lane_id in allowed]
        if not seeds:
            return set()

        visited = set()
        queue = list(seeds)
        reachable = set()

        while queue:
            lane_id = queue.pop(0)
            if lane_id in visited:
                continue
            visited.add(lane_id)

            if lane_id not in allowed:
                continue

            reachable.add(lane_id)
            info = self.parser.lanes.get(lane_id)
            if not info:
                continue

            for direction, neighbor_key in [('left', 'left_lane'), ('right', 'right_lane')]:
                neighbor = info.get(neighbor_key)
                if neighbor and neighbor in allowed and neighbor not in visited:
                    if self._can_change_to_lane(lane_id, direction, neighbor):
                        queue.append(neighbor)

        return reachable

    def _reverse_expand_lanes_within_road(self, road_id: str, target_lanes: Set[str],
                                          allowed_lanes: Optional[Set[str]] = None) -> Set[str]:
        """
        在单个 road 内做反向闭包：保留所有能通过合法换道到达 target_lanes 的车道。
        """
        if allowed_lanes is None:
            allowed = set(self.all_lanes_per_road.get(road_id, []))
        else:
            allowed = set(allowed_lanes)

        targets = {lane_id for lane_id in target_lanes if lane_id in allowed}
        if not targets:
            return set()

        reverse_edges: Dict[str, Set[str]] = {lane_id: set() for lane_id in allowed}

        for lane_id in allowed:
            info = self.parser.lanes.get(lane_id)
            if not info:
                continue

            for direction, neighbor_key in [('left', 'left_lane'), ('right', 'right_lane')]:
                neighbor = info.get(neighbor_key)
                if neighbor and neighbor in allowed:
                    if self._can_change_to_lane(lane_id, direction, neighbor):
                        reverse_edges.setdefault(neighbor, set()).add(lane_id)

        queue = list(targets)
        visited = set()
        reachable = set()

        while queue:
            lane_id = queue.pop(0)
            if lane_id in visited:
                continue
            visited.add(lane_id)
            reachable.add(lane_id)
            for prev_lane in reverse_edges.get(lane_id, set()):
                if prev_lane not in visited:
                    queue.append(prev_lane)

        return reachable

    def _collect_successor_entries(self, current_lanes: Set[str], next_road_id: str) -> Set[str]:
        """
        收集当前 road 的可达车道通过 successor 落到下一 road 的入口车道集合。
        """
        entries = set()
        for lane_id in current_lanes:
            info = self.parser.lanes.get(lane_id)
            if not info:
                continue
            for succ in info.get('successor', []):
                if succ.split('.')[0] == next_road_id:
                    entries.add(succ)
        return entries

    @staticmethod
    def _pair_overlap_length(source_info: dict, target_info: dict) -> float:
        """返回一对相邻车道在当前 road 上的重叠可用长度。"""

        return float(min(source_info.get('length', 0.0), target_info.get('length', 0.0)))

    def _lanes_are_adjacent_in_direction(self, source_lane: str, target_lane: str, direction: str) -> bool:
        """检查 target_lane 是否是 source_lane 在给定方向上的相邻车道。"""

        source_info = self.parser.lanes.get(source_lane)
        if not source_info:
            return False
        neighbor_key = 'left_lane' if direction == 'left' else 'right_lane'
        return str(source_info.get(neighbor_key, '')) == str(target_lane)

    def _next_adjacent_successor_pairs(
        self,
        source_lane: str,
        target_lane: str,
        direction: str,
    ) -> List[Tuple[str, str]]:
        """查找在下一规划 road 上仍保持相邻关系的 successor 车道对。"""

        source_info = self.parser.lanes.get(source_lane)
        target_info = self.parser.lanes.get(target_lane)
        if not source_info or not target_info:
            return []

        current_road_id = str(source_lane).split('.')[0]
        try:
            road_index = self.road_sequence.index(current_road_id)
        except ValueError:
            return []

        if road_index >= len(self.road_sequence) - 1:
            return []

        next_road_id = self.road_sequence[road_index + 1]
        source_successors = [
            str(succ)
            for succ in source_info.get('successor', [])
            if str(succ).split('.')[0] == next_road_id
        ]
        target_successors = [
            str(succ)
            for succ in target_info.get('successor', [])
            if str(succ).split('.')[0] == next_road_id
        ]
        if not source_successors or not target_successors:
            return []

        pairs: List[Tuple[str, str]] = []
        for source_succ in source_successors:
            for target_succ in target_successors:
                if self._lanes_are_adjacent_in_direction(source_succ, target_succ, direction):
                    pairs.append((source_succ, target_succ))
        return pairs

    def _continuous_change_corridor_length(self, source_lane: str, target_lane: str, direction: str) -> float:
        """沿连续相邻 successor 车道链累计可换道走廊长度。"""

        cache_key = (str(source_lane), str(target_lane), str(direction))
        if cache_key in self._change_corridor_cache:
            return float(self._change_corridor_cache[cache_key])

        source_info = self.parser.lanes.get(source_lane)
        target_info = self.parser.lanes.get(target_lane)
        if not source_info or not target_info:
            self._change_corridor_cache[cache_key] = 0.0
            return 0.0

        if not self._lanes_are_adjacent_in_direction(source_lane, target_lane, direction):
            self._change_corridor_cache[cache_key] = 0.0
            return 0.0

        mark_key = f'{direction}_mark_type'
        mark_type = str(source_info.get(mark_key, 'solid'))
        if mark_type not in self.config.changeable_marks:
            self._change_corridor_cache[cache_key] = 0.0
            return 0.0

        current_overlap = self._pair_overlap_length(source_info, target_info)
        next_pairs = self._next_adjacent_successor_pairs(source_lane, target_lane, direction)
        if not next_pairs:
            self._change_corridor_cache[cache_key] = float(current_overlap)
            return float(current_overlap)

        best_future = 0.0
        for next_source, next_target in next_pairs:
            best_future = max(
                best_future,
                self._continuous_change_corridor_length(next_source, next_target, direction),
            )

        total = float(current_overlap) + float(best_future)
        self._change_corridor_cache[cache_key] = total
        return total

    def _expand_drivable_lanes(self, road_id: str, initial_lanes: Set[str],
                               road_index: int) -> Set[str]:
        """
        扩展可行驶车道集合

        参数:
            road_id: 路段ID
            initial_lanes: 初始可行驶车道集合
            road_index: 路段在序列中的索引

        返回:
            扩展后的可行驶车道集合
        """
        drivable = initial_lanes.copy()
        all_lanes = self.all_lanes_per_road.get(road_id, [])

        # 迭代扩展
        expanded = True
        while expanded:
            expanded = False
            for lane_id in list(drivable):
                info = self.parser.lanes.get(lane_id)
                if not info:
                    continue

                # 检查左侧车道
                left_lane = info.get('left_lane')
                if left_lane and left_lane in all_lanes and left_lane not in drivable:
                    if self._can_change_to_lane(lane_id, 'left', left_lane):
                        drivable.add(left_lane)
                        expanded = True

                # 检查右侧车道
                right_lane = info.get('right_lane')
                if right_lane and right_lane in all_lanes and right_lane not in drivable:
                    if self._can_change_to_lane(lane_id, 'right', right_lane):
                        drivable.add(right_lane)
                        expanded = True

        # 检查连通性：确保能到达下一个路段
        if road_index < len(self.road_sequence) - 1:
            next_road_id = self.road_sequence[road_index + 1]
            next_entry_lane = self._get_entry_lane(next_road_id)
            if next_entry_lane:
                connectable = {l for l in drivable if self._can_reach(l, next_entry_lane)}
                if connectable:
                    drivable = connectable

        return drivable

    def _can_change_to_lane(self, source_lane: str, direction: str, target_lane: str) -> bool:
        """
        判断是否可以变道到目标车道。

        参数:
            source_lane: 起始车道ID
            direction: 变道方向 ('left' 或 'right')
            target_lane: 目标相邻车道ID

        返回:
            是否可以变道
        """
        source_info = self.parser.lanes.get(source_lane)
        target_info = self.parser.lanes.get(target_lane)
        if not source_info or not target_info:
            return False

        if not self._lanes_are_adjacent_in_direction(source_lane, target_lane, direction):
            return False

        current_overlap = self._pair_overlap_length(source_info, target_info)
        if current_overlap <= 1e-6:
            return False

        required_total = float(self.config.min_change_advance_length) + float(self.config.single_change_min_remaining)
        corridor_length = self._continuous_change_corridor_length(source_lane, target_lane, direction)
        return corridor_length >= required_total

    def _get_road_length(self, road_id: str) -> float:
        """获取路段长度"""
        road_info = self.parser.roads.get(road_id)
        if road_info:
            return road_info.get('length', 0)
        return 0

    def _get_entry_lane(self, road_id: str) -> Optional[str]:
        """获取路段的入口车道"""
        for lane_id in self.planned_path:
            if lane_id.split('.')[0] == road_id:
                return lane_id
        return None

    def _get_planned_lane_sequence_by_road(self) -> Dict[str, List[str]]:
        """
        按 road 聚合 planned_path，保留同一 road 内的多阶段 lane 序列。
        """
        lane_seq_by_road: Dict[str, List[str]] = {}
        for lane_id in self.planned_path:
            road_id = lane_id.split('.')[0]
            road_seq = lane_seq_by_road.setdefault(road_id, [])
            if not road_seq or road_seq[-1] != lane_id:
                road_seq.append(lane_id)
        return lane_seq_by_road

    def _can_change_along_reachable_lanes(self, start_lane: str, target_lane: str,
                                          allowed_lanes: Set[str]) -> bool:
        """
        在单个 road 内，检查是否能通过合法换道从 start_lane 到达 target_lane。
        """
        if start_lane == target_lane:
            return True
        if start_lane not in allowed_lanes or target_lane not in allowed_lanes:
            return False

        visited = set()
        queue = [start_lane]

        while queue:
            lane_id = queue.pop(0)
            if lane_id in visited:
                continue
            visited.add(lane_id)

            info = self.parser.lanes.get(lane_id)
            if not info:
                continue

            for direction, neighbor_key in [('left', 'left_lane'), ('right', 'right_lane')]:
                neighbor = info.get(neighbor_key)
                if not neighbor or neighbor not in allowed_lanes or neighbor in visited:
                    continue
                if not self._can_change_to_lane(lane_id, direction, neighbor):
                    continue
                if neighbor == target_lane:
                    return True
                queue.append(neighbor)

        return False

    def _can_reach(self, from_lane: str, to_lane: str, max_depth: int = 5) -> bool:
        """检查车道连通性"""
        if from_lane == to_lane:
            return True

        visited = set()
        queue = [(from_lane, 0)]

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)

            info = self.parser.lanes.get(current)
            if not info:
                continue

            # 检查后继车道
            for succ in info.get('successor', []):
                if succ == to_lane:
                    return True
                if succ not in visited:
                    queue.append((succ, depth + 1))

            # 检查相邻车道
            for key in ['left_lane', 'right_lane']:
                neighbor = info.get(key)
                if neighbor:
                    mark_type = info.get(f'{key.split("_")[0]}_mark_type', 'solid')
                    if mark_type in self.config.changeable_marks:
                        if neighbor == to_lane:
                            return True
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))

        return False

    @staticmethod
    def _point_to_centerline_min_dist(point: np.ndarray, centerline: np.ndarray) -> float:
        """点到中心线折线段的最小欧氏距离（非离散点最近距离）。"""
        if centerline is None or len(centerline) == 0:
            return float('inf')
        if len(centerline) == 1:
            return float(np.linalg.norm(point - centerline[0][:2]))

        min_dist = float('inf')
        for i in range(len(centerline) - 1):
            p1 = centerline[i][:2]
            p2 = centerline[i + 1][:2]
            seg = p2 - p1
            seg_len2 = float(np.dot(seg, seg))
            if seg_len2 < 1e-8:
                dist = float(np.linalg.norm(point - p1))
            else:
                t = float(np.dot(point - p1, seg) / seg_len2)
                t = float(np.clip(t, 0.0, 1.0))
                proj = p1 + t * seg
                dist = float(np.linalg.norm(point - proj))
            if dist < min_dist:
                min_dist = dist

        return min_dist

    @staticmethod
    def _lane_end_delta(point: np.ndarray, centerline: np.ndarray) -> float:
        """
        计算点相对车道末端横截面的越界量：
        >0 表示沿末端切向方向越过了车道末端。
        """
        if centerline is None or len(centerline) < 2:
            return 0.0

        lane_end = centerline[-1][:2]
        prev = centerline[-2][:2]
        tangent = lane_end - prev
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-8:
            return 0.0

        tangent = tangent / norm
        return float(np.dot(point - lane_end, tangent))

    def locate_lane_by_coordinate(self, query_x: float, query_y: float,
                                  max_distance: float = 3.0,
                                  candidate_lane_ids: Optional[Set[str]] = None) -> Tuple[
        Optional[str], Optional[str], float]:
        """
        根据坐标定位车道（必须在车道中心线上）

        参数:
            query_x, query_y: 查询坐标
            max_distance: 最大允许距离（米），超过则认为不在车道中心线上
            candidate_lane_ids: 候选车道集合；若提供，则只能定位到该集合内车道

        返回:
            (lane_id, road_id, distance) 或 (None, None, -1)
        """
        # query_point = np.array([query_x, query_y])
        # # ========== 优先在规划路径的车道中搜索 ==========
        # if self.road_sequence:
        #     best_lane_id = None
        #     best_road_id = None
        #     min_dist = float('inf')

        #     for road_id in self.road_sequence:
        #         for lane_id in self.all_lanes_per_road.get(road_id, []):
        #             if lane_id in self.parser.lane_centerlines:
        #                 centerline = self.parser.lane_centerlines[lane_id]
        #                 if len(centerline) > 0:
        #                     distances = np.linalg.norm(centerline - query_point, axis=1)
        #                     lane_min_dist = np.min(distances)
        #                     if lane_min_dist < min_dist:
        #                         min_dist = lane_min_dist
        #                         best_lane_id = lane_id
        #                         best_road_id = road_id

        #     # 在规划路径中找到车道，且距离合理（放宽到10米，允许换道时的横向偏移）
        #     if best_lane_id is not None and min_dist <= 10.0:
        #         if min_dist > max_distance:
        #             print(f"📍 坐标 ({query_x:.2f}, {query_y:.2f}) 定位到规划路径车道 {best_lane_id}，距离 {min_dist:.2f}m")
        #         return best_lane_id, best_road_id, min_dist

        if self.kdtree is None:
            return None, None, -1

        point = np.array([query_x, query_y])
        total_points = len(self.lane_point_indices)
        if total_points == 0:
            return None, None, -1

        # 若提供候选集合，则只允许定位到候选车道
        if candidate_lane_ids is not None:
            allowed = {str(lane_id) for lane_id in candidate_lane_ids}
            if not allowed:
                print("⚠️ 可行驶车道候选为空，无法定位")
                return None, None, -1

            # 先用KDTree离散点快速收窄候选，再用真实投影距离精确选车道
            k = min(800, total_points)
            _, idxs = self.kdtree.query(point, k=k)
            idxs = np.atleast_1d(idxs)

            nearby_allowed = []
            seen = set()
            for idx in idxs:
                lane_id = str(self.lane_point_indices[int(idx)])
                if lane_id in allowed and lane_id not in seen:
                    nearby_allowed.append(lane_id)
                    seen.add(lane_id)
                    if len(nearby_allowed) >= 80:
                        break

            if not nearby_allowed:
                nearby_allowed = list(allowed)

            # 末端越界阈值：超过该值判定已驶出车道横截面
            end_pass_threshold = 0.5

            best_lane_id = None
            best_dist = float('inf')
            for lane_id in nearby_allowed:
                centerline = self.parser.lane_centerlines.get(lane_id)
                if centerline is None or len(centerline) < 2:
                    continue
                lane_cl = np.asarray(centerline)

                # 已越过车道末端横截面，车道失效
                end_delta = self._lane_end_delta(point, lane_cl)
                if end_delta > end_pass_threshold:
                    continue

                dist = self._point_to_centerline_min_dist(point, lane_cl)
                if dist < best_dist:
                    best_dist = dist
                    best_lane_id = lane_id

            if best_lane_id is None:
                print(
                    f"⚠️ 坐标 ({query_x:.2f}, {query_y:.2f}) 在可行驶车道中无有效定位 "
                    f"(可能已越过候选车道末端横截面)"
                )
                return None, None, -1

            if best_dist > max_distance:
                print(
                    f"⚠️ 坐标 ({query_x:.2f}, {query_y:.2f}) 距离最近可行驶车道中心线过远 "
                    f"({best_dist:.2f}m > {max_distance}m)"
                )
                return None, None, float(best_dist)

            return best_lane_id, best_lane_id.split('.')[0], float(best_dist)

        # 未提供候选集合时，保持原始全局定位行为
        dist, idx = self.kdtree.query(point)
        if dist > max_distance:
            print(f"⚠️ 坐标 ({query_x:.2f}, {query_y:.2f}) 距离最近车道中心线过远 ({dist:.2f}m > {max_distance}m)")
            print(f"   查询坐标必须在车道中心线上！")
            return None, None, dist

        lane_id = str(self.lane_point_indices[idx])
        road_id = lane_id.split('.')[0]
        return lane_id, road_id, dist

    def plan_extended_centerlines(self, query_x: float, query_y: float) -> DrivableLanesResult:
        """
        获取从查询坐标向后延伸的所有可行驶车道中心线

        逻辑：
        1. 定位查询坐标所在的车道
        2. 递归查找所有可以通过连续合法换道到达的车道
        3. 从每个可达车道通过successor往后延伸（断开则终止）

        参数:
            query_x, query_y: 查询坐标（必须在车道中心线上）

        返回:
            DrivableLanesResult 结果对象
        """
        result = DrivableLanesResult(
            road_sequence=self.road_sequence,
            all_lanes_per_road=self.all_lanes_per_road,
            filtered_drivable_lanes=self.filtered_drivable_lanes
        )

        # 定位候选：只允许定位到“未通过路段”上的可行驶车道
        active_roads = self.road_sequence[self.active_road_start_idx:]
        drivable_candidates = set()
        for road_id in active_roads:
            drivable_candidates.update(self.filtered_drivable_lanes.get(road_id, set()))

        # 定位查询坐标所在的车道和路段（强约束：仅可行驶车道）
        current_lane_id, query_road_id, dist = self.locate_lane_by_coordinate(
            query_x, query_y, candidate_lane_ids=drivable_candidates
        )
        if query_road_id is None:
            print("❌ 无法定位查询坐标")
            return result

        # 检查路段是否在规划路径中
        if query_road_id not in self.road_sequence:
            print(f"⚠️ 坐标所在路段 {query_road_id} 不在规划路径中")
            return result

        query_road_idx = self.road_sequence.index(query_road_id)
        if query_road_idx > self.active_road_start_idx:
            self.active_road_start_idx = query_road_idx

        result.query_road_id = query_road_id
        # print(f"✅ 定位到路段: {query_road_id}, 车道: {current_lane_id}")

        # 当前路段只在全局可行的车道集合里继续展开
        all_lanes_in_road = set(
            self.filtered_drivable_lanes.get(query_road_id, set(self.all_lanes_per_road.get(query_road_id, [])))
        )

        # 递归查找所有可以通过连续合法换道到达的车道
        intersection = self._find_reachable_lanes_by_lane_change(
            current_lane_id, all_lanes_in_road
        )

        # 保证 current_lane_id 在第 0 个（也就保证 extended_centerlines[0] 对应 query 定位车道）
        others = sorted([l for l in intersection if l != current_lane_id],
                        key=lambda x: int(x.split('.')[-1]))
        result.intersection_lanes = [current_lane_id] + others
        # print(f"✅ 输出顺序第0条车道: {result.intersection_lanes[0]} (定位车道: {current_lane_id})")

        # print(f"\n📊 当前位置可行驶车道: {[int(l.split('.')[-1]) for l in result.intersection_lanes]}")

        if not result.intersection_lanes:
            print("⚠️ 无可行驶车道")
            return result

        # 从查询坐标在规划序列中的位置开始，计算向后延伸的车道
        road_idx = query_road_idx
        remaining_roads = self.road_sequence[road_idx:]

        # print(f"\n📋 从当前位置到终点的剩余路段: {' -> '.join(remaining_roads)}")

        # 从每个交集车道出发，计算向后延伸的所有可行驶车道中心线
        for start_lane in result.intersection_lanes:
            # 获取从这个车道开始的延伸路径
            extended_path, extended_centerline = self._get_extended_path(
                start_lane, query_road_id, remaining_roads, query_x, query_y
            )

            result.extended_lane_ids.append(extended_path)
            result.extended_centerlines.append(np.array(extended_centerline))

        return result

    def _find_reachable_lanes_by_lane_change(self, start_lane: str,
                                             all_lanes_in_road: Set[str]) -> Set[str]:
        """
        从起始车道出发，递归查找所有可以通过连续合法换道到达的车道

        参数:
            start_lane: 起始车道ID
            all_lanes_in_road: 该路段的所有车道

        返回:
            可达车道集合
        """
        reachable = set()
        visited = set()
        queue = [start_lane]

        while queue:
            lane_id = queue.pop(0)
            if lane_id in visited:
                continue
            visited.add(lane_id)

            # 检查该车道是否在路段中
            if lane_id not in all_lanes_in_road:
                continue

            reachable.add(lane_id)

            # 获取车道信息
            lane_info = self.parser.lanes.get(lane_id)
            if not lane_info:
                continue

            # 检查左侧车道
            left_lane = lane_info.get('left_lane')
            left_mark = lane_info.get('left_mark_type', 'solid')
            if left_lane and left_lane in all_lanes_in_road and left_lane not in visited:
                if left_mark in self.config.changeable_marks:
                    queue.append(left_lane)
                #     print(f"   ✅ {lane_id} -> 左侧 {left_lane} 可换道 (车道线: {left_mark})")
                # else:
                #     print(f"   ❌ {lane_id} -> 左侧 {left_lane} 不可换道 (车道线: {left_mark})")

            # 检查右侧车道
            right_lane = lane_info.get('right_lane')
            right_mark = lane_info.get('right_mark_type', 'solid')
            if right_lane and right_lane in all_lanes_in_road and right_lane not in visited:
                if right_mark in self.config.changeable_marks:
                    queue.append(right_lane)
                #     print(f"   ✅ {lane_id} -> 右侧 {right_lane} 可换道 (车道线: {right_mark})")
                # else:
                #     print(f"   ❌ {lane_id} -> 右侧 {right_lane} 不可换道 (车道线: {right_mark})")

        return reachable

    def _get_extended_path(self, start_lane: str, start_road_id: str,
                           remaining_roads: List[str], query_x: float,
                           query_y: float) -> Tuple[List[str], List[List[float]]]:
        """
        获取从起始车道向后延伸的路径和中心线坐标

        注意：只通过successor关系延伸，不包含换道。如果没有后继则终止。
        会在终点坐标处截断。

        参数:
            start_lane: 起始车道ID
            start_road_id: 起始路段ID
            remaining_roads: 剩余路段序列
            query_x, query_y: 查询坐标（用于定位起始位置）

        返回:
            (车道ID列表, 中心线坐标列表)
        """
        extended_lanes = [start_lane]

        current_lane = start_lane
        query_point = np.array([query_x, query_y])
        planned_lane_seq_by_road = self._get_planned_lane_sequence_by_road()

        for i, road_id in enumerate(remaining_roads):
            # 获取该路段的所有车道（不是筛选后的）
            allowed_lanes = set(self.filtered_drivable_lanes.get(road_id, set(self.all_lanes_per_road.get(road_id, []))))

            if i != 0:
                # 后续路段：只通过successor关系延伸，不换道
                lane_info = self.parser.lanes.get(current_lane)
                if lane_info:
                    # 找到后继车道（需要在该路段的车道中）
                    next_lane = None
                    for succ in lane_info.get('successor', []):
                        if succ in allowed_lanes:
                            next_lane = succ
                            break

                    if next_lane:
                        current_lane = next_lane
                        extended_lanes.append(current_lane)
                    else:
                        # 没有后继，终止延伸
                        # print(f"   ⚠️ 车道 {current_lane} 无后继，延伸终止")
                        break
                else:
                    # 无法获取车道信息，终止
                    break

        extended_centerline = self._build_extended_centerline(extended_lanes, query_point)

        # 在终点处截断
        if self.goal_point is not None and len(extended_centerline) > 0:
            extended_centerline = _truncate_at_goal(extended_centerline, self.goal_point)

        return extended_lanes, extended_centerline

    def _build_extended_centerline(self, lane_sequence: List[str], query_point: np.ndarray) -> List[List[float]]:
        """
        根据 lane 序列构建几何连续的中心线。

        约定：lane_centerlines 已在 xodr_parser 中统一为行驶方向，这里只负责：
        1. 第一条 lane 从 query 的投影点开始截取；
        2. 后续 lane 直接按既有顺序拼接；
        3. 去除相邻 lane 交界处的重复点。
        """
        if not lane_sequence:
            return []

        extended_centerline: List[List[float]] = []

        for idx, lane_id in enumerate(lane_sequence):
            centerline = self.parser.lane_centerlines.get(lane_id)
            if centerline is None or len(centerline) == 0:
                continue

            centerline = np.asarray(centerline, dtype=float)

            if idx == 0:
                first_slice = np.asarray(self._slice_centerline_from_projection(centerline, query_point), dtype=float)
                if len(first_slice) < 2:
                    continue
                extended_centerline.extend(first_slice.tolist())
                continue

            points_to_add = centerline
            if extended_centerline and np.linalg.norm(points_to_add[0, :2] - np.asarray(extended_centerline[-1])[:2]) < 1e-6:
                points_to_add = points_to_add[1:]

            if len(points_to_add) == 0:
                continue

            extended_centerline.extend(points_to_add[:, :2].tolist())

        return extended_centerline

    @staticmethod
    def _slice_centerline_from_projection(centerline: np.ndarray, query_point: np.ndarray) -> List[List[float]]:
        """
        从 query_point 在 centerline 上的投影点开始截取中心线。

        返回格式：
        [投影点, 投影点之后的离散中心线点...]
        """
        centerline = np.array(centerline, dtype=float)
        query_point = np.array(query_point, dtype=float)

        if len(centerline) == 0:
            return []
        if len(centerline) == 1:
            return [centerline[0].tolist()]

        best_idx = 0
        best_t = 0.0
        best_proj = centerline[0, :2].copy()
        min_dist = float("inf")

        for i in range(len(centerline) - 1):
            p1 = centerline[i, :2]
            p2 = centerline[i + 1, :2]
            seg = p2 - p1
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq < 1e-12:
                continue

            t = float(np.dot(query_point[:2] - p1, seg) / seg_len_sq)
            t = float(np.clip(t, 0.0, 1.0))
            proj = p1 + t * seg
            dist = float(np.linalg.norm(query_point[:2] - proj))

            if dist < min_dist:
                min_dist = dist
                best_idx = i
                best_t = t
                best_proj = proj

        sliced = [best_proj.tolist()]

        append_start_idx = best_idx + 1
        if np.linalg.norm(centerline[best_idx + 1, :2] - best_proj) < 1e-6:
            append_start_idx = best_idx + 2

        for j in range(append_start_idx, len(centerline)):
            sliced.append(centerline[j].tolist())

        return sliced

    def export_result(self, result: DrivableLanesResult, output_path: str):
        """
        导出结果到JSON文件

        参数:
            result: 结果对象
            output_path: 输出文件路径
        """
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'min_change_advance_length': self.config.min_change_advance_length,
                'single_change_min_remaining': self.config.single_change_min_remaining,
                'return_change_min_length': self.config.return_change_min_length,
                'changeable_marks': self.config.changeable_marks
            },
            'road_sequence': result.road_sequence,
            'all_lanes_per_road': {
                road_id: lanes
                for road_id, lanes in result.all_lanes_per_road.items()
            },
            'filtered_drivable_lanes': {
                road_id: sorted(list(lanes))
                for road_id, lanes in result.filtered_drivable_lanes.items()
            },
            'query_result': {
                'query_road_id': result.query_road_id,
                'intersection_lanes': result.intersection_lanes,
                'extended_lane_count': len(result.extended_centerlines),
                'extended_paths': [
                    {
                        'lane_ids': lane_ids,
                        'centerline_points_count': len(centerline),
                        'centerline': centerline
                    }
                    for lane_ids, centerline in zip(result.extended_lane_ids, result.extended_centerlines)
                ]
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 结果已保存至: {output_path}")
