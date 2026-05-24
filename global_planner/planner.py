import numpy as np
import networkx as nx

from scipy.spatial import KDTree
from typing import List, Dict, Tuple
from collections import OrderedDict

from global_planner.config import (
    CHANGEABLE_MARKS,
    LANE_CHANGE_COST,
    MIN_LANE_CHANGE_LENGTH,
    MAX_SEARCH_DISTANCE,
    RoadSegment,
    LANE_CHANGE_CONFIG
)


class GlobalPlanner:
    def __init__(self, xodr_parser):
        """
        初始化规划器

        参数:
            xodr_parser: XODR解析器实例
        """
        self.parser = xodr_parser

        # 路径规划图
        self.graph = nx.DiGraph()
        self.kdtree = None
        self.lane_indices = None

        # 规划结果
        self.start_point = None
        self.end_point = None
        self.planned_path: List[str] = []
        self.planned_actions: List[dict] = []
        self.road_sequence: List[str] = []
        self.road_segments: OrderedDict[str, RoadSegment] = OrderedDict()

        # 构建图
        self._build_graph()
        self._build_kdtree()

    def plan(self, start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
        """
        执行完整的路径规划

        参数:
            start_x, start_y: 起点坐标
            end_x, end_y: 终点坐标

        返回:
            是否规划成功
        """
        self.start_point = (start_x, start_y)
        self.end_point = (end_x, end_y)

        # 1. A*路径规划
        if not self._plan_route():
            return False

        # 2. 提取路段序列
        self._extract_road_sequence()

        # 3. 构建路段信息
        self._build_road_segments()

        # 4. 计算可行驶车道
        self._compute_drivable_lanes()

        return True

    def _build_graph(self):
        """构建车道拓扑图"""
        print("正在构建路径规划图...")

        for lane_id, lane_info in self.parser.lanes.items():
            if lane_info['type'] != 'driving':
                continue

            self.graph.add_node(lane_id, **lane_info)

            # 后继边
            for succ in lane_info['successor']:
                if succ in self.parser.lanes:
                    self.graph.add_edge(lane_id, succ,
                                        action='successor',
                                        cost=lane_info['length'])

            # 左换道边
            left_lane = lane_info['left_lane']
            if left_lane and left_lane in self.parser.lanes:
                left_mark = lane_info['left_mark_type']
                if left_mark in CHANGEABLE_MARKS and lane_info['length'] > MIN_LANE_CHANGE_LENGTH:
                    self.graph.add_edge(lane_id, left_lane,
                                        action='left_change',
                                        cost=LANE_CHANGE_COST)

            # 右换道边
            right_lane = lane_info['right_lane']
            if right_lane and right_lane in self.parser.lanes:
                right_mark = lane_info['right_mark_type']
                if right_mark in CHANGEABLE_MARKS and lane_info['length'] > MIN_LANE_CHANGE_LENGTH:
                    self.graph.add_edge(lane_id, right_lane,
                                        action='right_change',
                                        cost=LANE_CHANGE_COST)

        print(f"图构建完成! 节点: {self.graph.number_of_nodes()}, 边: {self.graph.number_of_edges()}")

    def _build_kdtree(self):
        """构建KDTree"""
        all_points = []
        self.lane_indices = []

        for lane_id, centerline in self.parser.lane_centerlines.items():
            if lane_id not in self.graph.nodes:
                continue
            for point in centerline:
                all_points.append(point)
                self.lane_indices.append(lane_id)

        if all_points:
            self.kdtree = KDTree(np.array(all_points))
            self.lane_indices = np.array(self.lane_indices)

    def _find_nearest_lane(self, point: np.ndarray) -> Tuple[str, float]:
        """查找最近车道"""
        if self.kdtree is None:
            raise ValueError("KDTree未初始化")

        dist, idx = self.kdtree.query(point)
        if dist > MAX_SEARCH_DISTANCE:
            raise ValueError(f"未找到有效车道，最近距离 {dist:.2f}m")

        return str(self.lane_indices[idx]), dist

    def _plan_route(self) -> bool:
        """A*路径规划"""
        start = np.array(self.start_point)
        end = np.array(self.end_point)

        try:
            start_lane, start_dist = self._find_nearest_lane(start)
            end_lane, end_dist = self._find_nearest_lane(end)

            # 记录起终点车道
            self.start_lane = start_lane
            self.end_lane = end_lane

            # 找到起终点在车道中心线上的最近点
            if start_lane in self.parser.lane_centerlines:
                start_cl = self.parser.lane_centerlines[start_lane]
                self.start_idx_on_lane = self._find_closest_idx(start_cl, start)
                self.start_point_on_lane = start_cl[self.start_idx_on_lane].copy()
                print(f"起点车道: {start_lane} (距离: {start_dist:.2f}m)")
                print(f"  投影点: ({self.start_point_on_lane[0]:.2f}, {self.start_point_on_lane[1]:.2f})")

            if end_lane in self.parser.lane_centerlines:
                end_cl = self.parser.lane_centerlines[end_lane]
                self.end_idx_on_lane = self._find_closest_idx(end_cl, end)
                self.end_point_on_lane = end_cl[self.end_idx_on_lane].copy()
                print(f"终点车道: {end_lane} (距离: {end_dist:.2f}m)")
                print(f"  投影点: ({self.end_point_on_lane[0]:.2f}, {self.end_point_on_lane[1]:.2f})")

        except ValueError as e:
            print(f"❌ 位置匹配失败: {e}")
            return False

        if start_lane == end_lane:
            self.planned_path = [start_lane]
            self.planned_actions = [{'lane': start_lane, 'action': 'start'}]
            return True
        # 启发式函数 估计代价
        try:
            def heuristic(u, v):
                if u not in self.parser.lane_centerlines or v not in self.parser.lane_centerlines:
                    return 0  # 中心线缺失返回0 退化Dijksatra
                u_center = self.parser.lane_centerlines[u][len(self.parser.lane_centerlines[u]) // 2]
                v_center = self.parser.lane_centerlines[v][len(self.parser.lane_centerlines[v]) // 2]
                return np.linalg.norm(u_center - v_center)
                # 代价为两个车道中心线“中点”之间的欧氏距离

            path = nx.astar_path(self.graph, start_lane, end_lane,
                                 heuristic=heuristic, weight='cost')

            actions = []
            for i, lane_id in enumerate(path):
                if i == 0:
                    actions.append({'lane': lane_id, 'action': 'start'})
                else:
                    edge_data = self.graph.edges[path[i - 1], lane_id]
                    actions.append({
                        'lane': lane_id,
                        'action': edge_data['action'],
                        'from_lane': path[i - 1]
                    })

            self.planned_path = [str(p) for p in path]
            self.planned_actions = actions
            print(f"✅ 路径规划成功! 经过 {len(path)} 个车道")
            return True

        except nx.NetworkXNoPath:
            print(f"❌ 未找到路径")
            return False

    @staticmethod
    def _find_closest_idx(centerline: np.ndarray, point: np.ndarray) -> int:
        """找到中心线上最近点的索引"""
        distances = np.linalg.norm(centerline - point, axis=1)
        return int(np.argmin(distances))

    def _extract_road_sequence(self):
        """提取路段ID序列"""
        self.road_sequence = []
        seen = set()

        for lane_id in self.planned_path:
            road_id = lane_id.split('.')[0]
            if road_id and road_id not in seen:
                self.road_sequence.append(road_id)
                seen.add(road_id)

    def _build_road_segments(self):
        """构建路段信息"""
        self.road_segments = OrderedDict()

        # 按路段分组
        road_lanes: Dict[str, List[str]] = {}
        for lane_id in self.planned_path:
            road_id = lane_id.split('.')[0]
            if road_id not in road_lanes:
                road_lanes[road_id] = []
            if lane_id not in road_lanes[road_id]:
                road_lanes[road_id].append(lane_id)

        for road_id in self.road_sequence:
            planned = road_lanes.get(road_id, [])
            all_lanes = self._get_driving_lanes(road_id)
            start_coord, end_coord, length = self._get_road_geometry(planned)

            self.road_segments[road_id] = RoadSegment(
                road_id=road_id,
                planned_lanes=planned,
                all_lanes=all_lanes,
                drivable_lanes=set(planned),
                entry_lane=planned[0] if planned else "",
                exit_lane=planned[-1] if planned else "",
                start_coord=start_coord,
                end_coord=end_coord,
                length=length
            )

    def _get_driving_lanes(self, road_id: str) -> List[str]:
        """获取路段所有driving车道"""
        lanes = []
        for lane_id, info in self.parser.lanes.items():
            if info['road_id'] == road_id and info['type'] == 'driving':
                lanes.append(lane_id)
        return sorted(lanes, key=lambda x: int(x.split('.')[-1]))

    def _get_road_geometry(self, lanes: List[str]) -> Tuple[Tuple, Tuple, float]:
        """获取路段几何信息"""
        if not lanes:
            return (0, 0), (0, 0), 0

        total_length = 0
        start = None
        end = None

        for lane_id in lanes:
            if lane_id in self.parser.lane_centerlines:
                cl = self.parser.lane_centerlines[lane_id]
                if len(cl) > 0:
                    if start is None:
                        start = (cl[0][0], cl[0][1])
                    end = (cl[-1][0], cl[-1][1])
                    for i in range(len(cl) - 1):
                        total_length += np.linalg.norm(cl[i + 1] - cl[i])

        return start or (0, 0), end or (0, 0), total_length

    def _compute_drivable_lanes(self):
        """计算可行驶车道（基于虚实线）"""
        for i, road_id in enumerate(self.road_sequence):
            seg = self.road_segments[road_id]
            drivable = set(seg.planned_lanes)

            # 扩展可变道车道
            expanded = True
            while expanded:
                expanded = False
                for lane_id in list(drivable):
                    info = self.parser.lanes.get(lane_id)
                    if not info:
                        continue

                    # 左侧
                    left = info.get('left_lane')
                    if left and left in self.parser.lanes:
                        if info.get('left_mark_type', 'solid') in CHANGEABLE_MARKS:
                            if left not in drivable:
                                drivable.add(left)
                                expanded = True

                    # 右侧
                    right = info.get('right_lane')
                    if right and right in self.parser.lanes:
                        if info.get('right_mark_type', 'solid') in CHANGEABLE_MARKS:
                            if right not in drivable:
                                drivable.add(right)
                                expanded = True

            # 检查连通性
            if i < len(self.road_sequence) - 1:
                next_entry = self.road_segments[self.road_sequence[i + 1]].entry_lane
                connectable = {l for l in drivable if self._can_reach(l, next_entry)}
                if connectable:
                    drivable = connectable

            seg.drivable_lanes = drivable

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

            for succ in info.get('successor', []):
                if succ == to_lane:
                    return True
                if succ not in visited:
                    queue.append((succ, depth + 1))

            for key in ['left_lane', 'right_lane']:
                neighbor = info.get(key)
                if neighbor:
                    mark = info.get(f'{key.split("_")[0]}_mark_type', 'solid')
                    if mark in CHANGEABLE_MARKS:
                        if neighbor == to_lane:
                            return True
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))

        return False

    def get_all_lane_centerlines(self) -> Dict[str, np.ndarray]:
        """获取所有车道中心线（用于可视化）"""
        return self.parser.lane_centerlines

    def get_drivable_centerlines(self) -> Dict[str, np.ndarray]:
        """获取所有可行驶车道的中心线"""
        result = {}
        for seg in self.road_segments.values():
            for lane_id in seg.drivable_lanes:
                if lane_id in self.parser.lane_centerlines:
                    result[lane_id] = self.parser.lane_centerlines[lane_id]
        return result


class CurvePlanner(GlobalPlanner):
    def __init__(self, xodr_parser, curve_config = None):
        super().__init__(xodr_parser)
        if curve_config is None:
            self.curve_config = LANE_CHANGE_CONFIG
        else:
            self.curve_config = curve_config

    def plan(self, start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
        self.start_point = (start_x, start_y)
        self.end_point = (end_x, end_y)

        # 1. A*路径规划
        if not self._plan_route():
            return False

        # 2. 提取路段序列
        self._extract_road_sequence()

        # 3. 构建路段信息
        self._build_road_segments()

        # 4. 计算可行驶车道
        self._compute_drivable_lanes()

        # 5. 生成轨迹
        self._generate_trajectory()

        return True

    def _generate_trajectory(self):
        all_points = []
        all_headings = []

        # 起点
        start_pt = np.array(self.start_point)
        end_pt = np.array(self.end_point)

        # 遍历规划动作
        for i, action in enumerate(self.planned_actions):
            lane_id = action['lane']
            act_type = action['action']

            if lane_id not in self.parser.lane_centerlines:
                continue

            centerline = self.parser.lane_centerlines[lane_id]
            if len(centerline) < 2:
                continue

            if act_type in ['start', 'follow', 'successor']:
                # 确定在当前车道上的起止点
                if act_type == 'start':
                    # 第一段：从起点投影到车道中心线
                    start_idx = self._find_closest_point_on_centerline(centerline, start_pt)

                    # 添加从起点到中心线的连接
                    if np.linalg.norm(start_pt - centerline[start_idx]) > 0.5:
                        all_points.append(start_pt)
                        if len(centerline) > start_idx + 1:
                            dx = centerline[start_idx + 1][0] - centerline[start_idx][0]
                            dy = centerline[start_idx + 1][1] - centerline[start_idx][1]
                        else:
                            dx = centerline[start_idx][0] - centerline[start_idx - 1][0]
                            dy = centerline[start_idx][1] - centerline[start_idx - 1][1]
                        all_headings.append(np.arctan2(dy, dx))
                else:
                    start_idx = 0

                # 检查是否需要换道（下一个动作）
                if i + 1 < len(self.planned_actions):
                    next_action = self.planned_actions[i + 1]
                    if next_action['action'] in ['left_change', 'right_change']:
                        # 需要换道：只走到换道起点
                        lc_start_ratio = self.curve_config.get('single_start_ratio', 0.3)
                        end_idx = int(len(centerline) * lc_start_ratio)
                    else:
                        # 不需要换道：走到车道终点
                        end_idx = len(centerline) - 1
                else:
                    # 最后一段：走到终点投影位置
                    end_idx = self._find_closest_point_on_centerline(centerline, end_pt)

                # 添加车道中心线上的点
                for j in range(start_idx, end_idx + 1):
                    all_points.append(centerline[j].copy())
                    # 计算航向
                    if j < len(centerline) - 1:
                        dx = centerline[j + 1][0] - centerline[j][0]
                        dy = centerline[j + 1][1] - centerline[j][1]
                    elif j > 0:
                        dx = centerline[j][0] - centerline[j - 1][0]
                        dy = centerline[j][1] - centerline[j - 1][1]
                    else:
                        dx, dy = 1, 0
                    all_headings.append(np.arctan2(dy, dx))

                # 如果是最后一段，添加到终点的连接
                if i == len(self.planned_actions) - 1:
                    if np.linalg.norm(end_pt - centerline[end_idx]) > 0.5:
                        all_points.append(end_pt)
                        all_headings.append(all_headings[-1] if all_headings else 0)

            elif act_type in ['left_change', 'right_change']:
                # 换道：生成贝塞尔曲线
                from_lane = action['from_lane']
                to_lane = lane_id

                if from_lane in self.parser.lane_centerlines and to_lane in self.parser.lane_centerlines:
                    from_cl = self.parser.lane_centerlines[from_lane]
                    to_cl = self.parser.lane_centerlines[to_lane]

                    # 生成换道曲线
                    lc_points, lc_headings = self._generate_lane_change_curve(from_cl, to_cl)
                    all_points.extend(lc_points)
                    all_headings.extend(lc_headings)

                    # 换道后继续沿新车道行驶
                    lc_end_ratio = self.curve_config.get('single_end_ratio', 0.7)
                    start_idx = int(len(to_cl) * lc_end_ratio)

                    # 检查下一个动作
                    if i + 1 < len(self.planned_actions):
                        next_action = self.planned_actions[i + 1]
                        if next_action['action'] in ['left_change', 'right_change']:
                            # 还要换道
                            next_lc_start_ratio = self.curve_config.get('single_start_ratio', 0.3)
                            end_idx = int(len(to_cl) * next_lc_start_ratio)
                        elif next_action['action'] == 'successor':
                            # 下一段是后继车道
                            end_idx = len(to_cl) - 1
                        else:
                            end_idx = len(to_cl) - 1
                    else:
                        # 最后一段
                        end_idx = self._find_closest_point_on_centerline(to_cl, end_pt)

                    # 添加换道后的车道点
                    for j in range(start_idx, end_idx + 1):
                        all_points.append(to_cl[j].copy())
                        if j < len(to_cl) - 1:
                            dx = to_cl[j + 1][0] - to_cl[j][0]
                            dy = to_cl[j + 1][1] - to_cl[j][1]
                        elif j > 0:
                            dx = to_cl[j][0] - to_cl[j - 1][0]
                            dy = to_cl[j][1] - to_cl[j - 1][1]
                        else:
                            dx, dy = 1, 0
                        all_headings.append(np.arctan2(dy, dx))

                    # 如果是最后一段，添加到终点
                    if i == len(self.planned_actions) - 1:
                        if np.linalg.norm(end_pt - to_cl[end_idx]) > 0.5:
                            all_points.append(end_pt)
                            all_headings.append(all_headings[-1] if all_headings else 0)

        if all_points:
            self.trajectory = np.array(all_points)
            self.trajectory_headings = np.array(all_headings)
        else:
            self.trajectory = np.array([])
            self.trajectory_headings = np.array([])

    @staticmethod
    def _find_closest_point_on_centerline(centerline: np.ndarray, point: np.ndarray) -> int:
        """找到中心线上距离给定点最近的点的索引"""
        distances = np.linalg.norm(centerline - point, axis=1)
        return int(np.argmin(distances))

    def _generate_lane_change_curve(self, from_cl: np.ndarray, to_cl: np.ndarray) -> Tuple[List, List]:
        """
        生成换道曲线（贝塞尔曲线）

        参数:
            from_cl: 起始车道中心线
            to_cl: 目标车道中心线

        返回:
            (点列表, 航向列表)
        """
        cfg = self.curve_config
        start_ratio = cfg.get('single_start_ratio', 0.3)
        end_ratio = cfg.get('single_end_ratio', 0.7)
        num_points = cfg.get('single_num_points', 20)

        n_from = len(from_cl)
        n_to = len(to_cl)

        # 换道起点和终点
        start_idx = int(n_from * start_ratio)
        end_idx = int(n_to * end_ratio)

        p0 = from_cl[start_idx].copy()
        p3 = to_cl[end_idx].copy() if end_idx < n_to else to_cl[-1].copy()

        # 计算起点和终点的切线方向
        if start_idx < n_from - 1:
            tangent_start = from_cl[start_idx + 1] - from_cl[start_idx]
        else:
            tangent_start = from_cl[start_idx] - from_cl[start_idx - 1]
        tangent_start = tangent_start / (np.linalg.norm(tangent_start) + 1e-6)

        if end_idx < n_to - 1:
            tangent_end = to_cl[end_idx + 1] - to_cl[end_idx]
        elif end_idx > 0:
            tangent_end = to_cl[end_idx] - to_cl[end_idx - 1]
        else:
            tangent_end = tangent_start
        tangent_end = tangent_end / (np.linalg.norm(tangent_end) + 1e-6)

        # 计算控制点（沿切线方向）
        dist = np.linalg.norm(p3 - p0)
        p1 = p0 + tangent_start * dist * 0.4
        p2 = p3 - tangent_end * dist * 0.4

        # 生成贝塞尔曲线
        points = []
        headings = []

        for i in range(num_points):
            t = i / (num_points - 1)

            # 三次贝塞尔曲线
            pt = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
            points.append(pt)

            # 切线方向（导数）
            tangent = 3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t ** 2 * (p3 - p2)
            heading = np.arctan2(tangent[1], tangent[0])
            headings.append(heading)

        return points, headings
