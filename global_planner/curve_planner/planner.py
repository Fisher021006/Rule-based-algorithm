import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from typing import List, Tuple, Dict, Optional

from xodr_parser.xodr_parser import XODRParser

from .config import (
    CHANGEABLE_MARKS,
    LANE_CHANGE_COST,
    MIN_LANE_CHANGE_LENGTH,
    MAX_SEARCH_DISTANCE,
    LANE_CHANGE_CONFIG
)


class CurvePlanner:
    """全局路径规划器"""

    def __init__(self, xodr_parser: XODRParser):
        """
        初始化路径规划器

        参数:
            xodr_parser: XODR解析器实例
        """
        self.parser = xodr_parser
        self.graph = nx.DiGraph()
        self.kdtree = None
        self.lane_indices = None

        # 构建图
        self._build_graph()
        self._build_kdtree()

    def _build_graph(self):
        """构建车道拓扑图"""
        print("正在构建路径规划图...")

        for lane_id, lane_info in self.parser.lanes.items():
            # 只处理driving类型的车道
            if lane_info['type'] != 'driving':
                continue

            # 添加节点
            self.graph.add_node(lane_id, **lane_info)

            # 添加后继边
            for succ in lane_info['successor']:
                if succ in self.parser.lanes:
                    self.graph.add_edge(lane_id, succ,
                                        action='successor',
                                        cost=lane_info['length'])

            # 添加左换道边
            left_lane = lane_info['left_lane']
            if left_lane and left_lane in self.parser.lanes:
                # 检查是否允许换道
                left_mark_type = lane_info['left_mark_type']
                if left_mark_type in CHANGEABLE_MARKS and lane_info['length'] > MIN_LANE_CHANGE_LENGTH:
                    self.graph.add_edge(lane_id, left_lane,
                                        action='left_change',
                                        cost=LANE_CHANGE_COST,
                                        mark_type=left_mark_type)

            # 添加右换道边
            right_lane = lane_info['right_lane']
            if right_lane and right_lane in self.parser.lanes:
                right_mark_type = lane_info['right_mark_type']
                if right_mark_type in CHANGEABLE_MARKS and lane_info['length'] > MIN_LANE_CHANGE_LENGTH:
                    self.graph.add_edge(lane_id, right_lane,
                                        action='right_change',
                                        cost=LANE_CHANGE_COST,
                                        mark_type=right_mark_type)

        print(f"图构建完成! 节点数: {self.graph.number_of_nodes()}, 边数: {self.graph.number_of_edges()}")

    def _build_kdtree(self):
        """构建KDTree用于快速位置匹配"""
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

    def find_nearest_lane(self, point: np.ndarray,
                          max_dist: float = None) -> Tuple[str, float]:
        """
        查找最近的车道

        参数:
            point: 查询点坐标 [x, y]
            max_dist: 最大搜索距离

        返回:
            (lane_id, distance): 最近车道ID和距离
        """
        if max_dist is None:
            max_dist = MAX_SEARCH_DISTANCE

        if self.kdtree is None:
            raise ValueError("KDTree未初始化")

        dist, idx = self.kdtree.query(point)
        if dist > max_dist:
            raise ValueError(f"未找到有效车道，最近距离 {dist:.2f} 超过阈值 {max_dist}")

        return self.lane_indices[idx], dist

    def plan_route(self, start_x: float, start_y: float,
                   goal_x: float, goal_y: float) -> Tuple[List[str], List[dict]]:
        """
        规划从起点到终点的路径

        参数:
            start_x, start_y: 起点坐标
            goal_x, goal_y: 终点坐标

        返回:
            (车道ID列表, 路径动作列表)
        """
        start_point = np.array([start_x, start_y])
        goal_point = np.array([goal_x, goal_y])

        # 查找起终点车道
        try:
            start_lane_id, start_dist = self.find_nearest_lane(start_point)
            goal_lane_id, goal_dist = self.find_nearest_lane(goal_point)
            print(f"起点匹配车道: {start_lane_id} (距离: {start_dist:.2f}m)")
            print(f"终点匹配车道: {goal_lane_id} (距离: {goal_dist:.2f}m)")
        except ValueError as e:
            print(f"❌ 位置匹配失败: {e}")
            return [], []

        # 如果起终点相同
        if start_lane_id == goal_lane_id:
            return [start_lane_id], [{'lane': start_lane_id, 'action': 'follow'}]

        # 使用A*算法搜索路径
        try:
            def heuristic(u, v):
                if u not in self.parser.lane_centerlines or v not in self.parser.lane_centerlines:
                    return 0
                u_center = self.parser.lane_centerlines[u][len(self.parser.lane_centerlines[u]) // 2]
                v_center = self.parser.lane_centerlines[v][len(self.parser.lane_centerlines[v]) // 2]
                return np.linalg.norm(u_center - v_center)

            path = nx.astar_path(self.graph, start_lane_id, goal_lane_id,
                                 heuristic=heuristic, weight='cost')

            # 构建动作列表
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

            print(f"✅ 路径规划成功! 经过 {len(path)} 个车道")
            return path, actions

        except nx.NetworkXNoPath:
            print(f"❌ 未找到从 {start_lane_id} 到 {goal_lane_id} 的路径")
            return [], []

    def get_lane_info(self, lane_id: str) -> Optional[dict]:
        """获取车道信息"""
        return self.parser.lanes.get(lane_id)

    def get_lane_centerline(self, lane_id: str) -> Optional[np.ndarray]:
        """获取车道中心线"""
        return self.parser.lane_centerlines.get(lane_id)


class LaneChangeGenerator:
    """换道曲线生成器"""

    @staticmethod
    def generate_lane_change_curve(from_lane_points: np.ndarray,
                                   to_lane_points: np.ndarray,
                                   start_ratio: float = None,
                                   end_ratio: float = None,
                                   num_points: int = None) -> np.ndarray:
        """
        生成平滑的单次换道曲线

        参数:
            from_lane_points: 起始车道中心线点
            to_lane_points: 目标车道中心线点
            start_ratio: 换道开始位置（None则使用配置值）
            end_ratio: 换道结束位置（None则使用配置值）
            num_points: 输出点数（None则使用配置值）

        返回:
            换道曲线坐标点数组
        """
        # 使用配置参数或默认值
        if start_ratio is None:
            start_ratio = LANE_CHANGE_CONFIG['single_start_ratio']
        if end_ratio is None:
            end_ratio = LANE_CHANGE_CONFIG['single_end_ratio']
        if num_points is None:
            num_points = LANE_CHANGE_CONFIG['single_num_points']

        # 计算起始车道的累计长度
        from_lengths = np.zeros(len(from_lane_points))
        for i in range(1, len(from_lane_points)):
            from_lengths[i] = from_lengths[i - 1] + np.linalg.norm(
                from_lane_points[i] - from_lane_points[i - 1]
            )

        total_length = from_lengths[-1]

        # 确定换道起止点在原车道上的位置
        start_s = total_length * start_ratio
        end_s = total_length * end_ratio

        # 插值获取换道起点位置
        start_idx = np.searchsorted(from_lengths, start_s)
        start_idx = max(1, min(start_idx, len(from_lane_points) - 1))

        # 获取起点坐标
        if start_idx > 0:
            ratio = (start_s - from_lengths[start_idx - 1]) / (
                    from_lengths[start_idx] - from_lengths[start_idx - 1] + 1e-6
            )
            start_point = from_lane_points[start_idx - 1] + ratio * (
                    from_lane_points[start_idx] - from_lane_points[start_idx - 1]
            )
        else:
            start_point = from_lane_points[0]

        # 在目标车道上找到对应终点
        to_lengths = np.zeros(len(to_lane_points))
        for i in range(1, len(to_lane_points)):
            to_lengths[i] = to_lengths[i - 1] + np.linalg.norm(
                to_lane_points[i] - to_lane_points[i - 1]
            )

        to_total_length = to_lengths[-1]
        end_s_target = to_total_length * end_ratio

        end_idx = np.searchsorted(to_lengths, end_s_target)
        end_idx = max(1, min(end_idx, len(to_lane_points) - 1))

        if end_idx > 0:
            ratio = (end_s_target - to_lengths[end_idx - 1]) / (
                    to_lengths[end_idx] - to_lengths[end_idx - 1] + 1e-6
            )
            end_point = to_lane_points[end_idx - 1] + ratio * (
                    to_lane_points[end_idx] - to_lane_points[end_idx - 1]
            )
        else:
            end_point = to_lane_points[0]

        # 计算起止点的切线方向
        if start_idx < len(from_lane_points) - 1:
            start_tangent = from_lane_points[start_idx] - from_lane_points[start_idx - 1]
        else:
            start_tangent = from_lane_points[-1] - from_lane_points[-2]
        start_tangent = start_tangent / (np.linalg.norm(start_tangent) + 1e-6)

        if end_idx < len(to_lane_points) - 1:
            end_tangent = to_lane_points[end_idx] - to_lane_points[end_idx - 1]
        else:
            end_tangent = to_lane_points[-1] - to_lane_points[-2]
        end_tangent = end_tangent / (np.linalg.norm(end_tangent) + 1e-6)

        # 使用贝塞尔曲线生成平滑曲线
        control_factor = LANE_CHANGE_CONFIG['single_control_factor']
        curve_points = LaneChangeGenerator._cubic_bezier_curve(
            start_point, start_tangent,
            end_point, end_tangent,
            num_points, control_factor
        )

        return curve_points

    @staticmethod
    def generate_multi_lane_change_curve(from_lane_points: np.ndarray,
                                         to_lane_points: np.ndarray,
                                         num_lanes: int = 2,
                                         num_points: int = None) -> np.ndarray:
        """
        生成跨多车道的平滑换道曲线（连续换道合并为一条曲线）

        参数:
            from_lane_points: 起始车道中心线点
            to_lane_points: 最终目标车道中心线点
            num_lanes: 跨越的车道数
            num_points: 输出点数（None则根据车道数自动计算）

        返回:
            换道曲线坐标点数组
        """
        # 根据跨越车道数计算点数
        if num_points is None:
            num_points = LANE_CHANGE_CONFIG['multi_num_points_per_lane'] * num_lanes

        # 根据跨越车道数调整换道区间
        base_start = LANE_CHANGE_CONFIG['single_start_ratio']
        base_end = LANE_CHANGE_CONFIG['single_end_ratio']
        start_adjust = LANE_CHANGE_CONFIG['multi_start_adjust']
        end_adjust = LANE_CHANGE_CONFIG['multi_end_adjust']

        start_ratio = max(0.1, base_start - start_adjust * (num_lanes - 1))
        end_ratio = min(0.9, base_end + end_adjust * (num_lanes - 1))

        # 计算起始车道的累计长度
        from_lengths = np.zeros(len(from_lane_points))
        for i in range(1, len(from_lane_points)):
            from_lengths[i] = from_lengths[i - 1] + np.linalg.norm(
                from_lane_points[i] - from_lane_points[i - 1]
            )

        total_length = from_lengths[-1]
        start_s = total_length * start_ratio

        # 获取起点
        start_idx = np.searchsorted(from_lengths, start_s)
        start_idx = max(1, min(start_idx, len(from_lane_points) - 1))

        if start_idx > 0 and start_idx < len(from_lengths):
            denom = from_lengths[start_idx] - from_lengths[start_idx - 1]
            if denom > 1e-6:
                ratio = (start_s - from_lengths[start_idx - 1]) / denom
            else:
                ratio = 0
            start_point = from_lane_points[start_idx - 1] + ratio * (
                    from_lane_points[start_idx] - from_lane_points[start_idx - 1]
            )
        else:
            start_point = from_lane_points[0]

        # 计算目标车道的累计长度
        to_lengths = np.zeros(len(to_lane_points))
        for i in range(1, len(to_lane_points)):
            to_lengths[i] = to_lengths[i - 1] + np.linalg.norm(
                to_lane_points[i] - to_lane_points[i - 1]
            )

        to_total_length = to_lengths[-1]
        end_s_target = to_total_length * end_ratio

        # 获取终点
        end_idx = np.searchsorted(to_lengths, end_s_target)
        end_idx = max(1, min(end_idx, len(to_lane_points) - 1))

        if end_idx > 0 and end_idx < len(to_lengths):
            denom = to_lengths[end_idx] - to_lengths[end_idx - 1]
            if denom > 1e-6:
                ratio = (end_s_target - to_lengths[end_idx - 1]) / denom
            else:
                ratio = 0
            end_point = to_lane_points[end_idx - 1] + ratio * (
                    to_lane_points[end_idx] - to_lane_points[end_idx - 1]
            )
        else:
            end_point = to_lane_points[-1]

        # 计算切线方向
        if start_idx < len(from_lane_points) - 1 and start_idx > 0:
            start_tangent = from_lane_points[start_idx] - from_lane_points[start_idx - 1]
        elif len(from_lane_points) >= 2:
            start_tangent = from_lane_points[-1] - from_lane_points[-2]
        else:
            start_tangent = np.array([1.0, 0.0])
        start_tangent = start_tangent / (np.linalg.norm(start_tangent) + 1e-6)

        if end_idx < len(to_lane_points) - 1 and end_idx > 0:
            end_tangent = to_lane_points[end_idx] - to_lane_points[end_idx - 1]
        elif len(to_lane_points) >= 2:
            end_tangent = to_lane_points[-1] - to_lane_points[-2]
        else:
            end_tangent = np.array([1.0, 0.0])
        end_tangent = end_tangent / (np.linalg.norm(end_tangent) + 1e-6)

        # 使用五次多项式生成更平滑的曲线
        control_factor = LANE_CHANGE_CONFIG['multi_control_factor']
        curve_points = LaneChangeGenerator._quintic_bezier_curve(
            start_point, start_tangent,
            end_point, end_tangent,
            num_points, control_factor
        )

        return curve_points

    @staticmethod
    def _cubic_bezier_curve(p0: np.ndarray, t0: np.ndarray,
                            p1: np.ndarray, t1: np.ndarray,
                            num_points: int,
                            control_factor: float = 3.0) -> np.ndarray:
        """
        生成三次贝塞尔曲线（保证起终点位置和切线方向）

        参数:
            p0: 起点
            t0: 起点切线方向（单位向量）
            p1: 终点
            t1: 终点切线方向（单位向量）
            num_points: 采样点数
            control_factor: 控制点距离系数，越小曲线越圆润

        返回:
            曲线采样点数组
        """
        # 计算控制点距离
        dist = np.linalg.norm(p1 - p0)
        control_dist = dist / control_factor

        # 计算控制点
        c1 = p0 + control_dist * t0
        c2 = p1 - control_dist * t1

        # 生成曲线点
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))

        for i, ti in enumerate(t):
            # 三次贝塞尔公式
            b0 = (1 - ti) ** 3
            b1 = 3 * (1 - ti) ** 2 * ti
            b2 = 3 * (1 - ti) * ti ** 2
            b3 = ti ** 3

            curve[i] = b0 * p0 + b1 * c1 + b2 * c2 + b3 * p1

        return curve

    @staticmethod
    def _quintic_bezier_curve(p0: np.ndarray, t0: np.ndarray,
                              p1: np.ndarray, t1: np.ndarray,
                              num_points: int,
                              control_factor: float = 2.5) -> np.ndarray:
        """
        生成五次贝塞尔曲线（更平滑，适合多车道换道）

        参数:
            p0: 起点
            t0: 起点切线方向（单位向量）
            p1: 终点
            t1: 终点切线方向（单位向量）
            num_points: 采样点数
            control_factor: 控制点距离系数，越小曲线越圆润

        返回:
            曲线采样点数组
        """
        dist = np.linalg.norm(p1 - p0)
        control_dist = dist / control_factor

        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))

        # 计算6个控制点（五次贝塞尔）
        c0 = p0
        c1 = p0 + control_dist * t0 * 0.4
        c2 = p0 + control_dist * t0 * 0.8
        c3 = p1 - control_dist * t1 * 0.8
        c4 = p1 - control_dist * t1 * 0.4
        c5 = p1

        for i, ti in enumerate(t):
            # 五次贝塞尔公式
            b0 = (1 - ti) ** 5
            b1 = 5 * (1 - ti) ** 4 * ti
            b2 = 10 * (1 - ti) ** 3 * ti ** 2
            b3 = 10 * (1 - ti) ** 2 * ti ** 3
            b4 = 5 * (1 - ti) * ti ** 4
            b5 = ti ** 5

            curve[i] = b0 * c0 + b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 + b5 * c5

        return curve


class TrajectoryGenerator:
    """完整轨迹生成器"""

    def __init__(self, planner: CurvePlanner):
        """
        初始化轨迹生成器

        参数:
            planner: 路径规划器实例
        """
        self.planner = planner
        self.lane_change_gen = LaneChangeGenerator()

    @staticmethod
    def _find_consecutive_lane_changes(actions: List[dict]) -> List[Tuple[int, int]]:
        """
        找出连续换道的区间

        参数:
            actions: 动作列表

        返回:
            [(start_idx, end_idx), ...] 连续换道的索引区间
        """
        groups = []
        i = 0
        while i < len(actions):
            if actions[i]['action'] in ['left_change', 'right_change']:
                start = i
                # 继续查找连续的换道
                while i < len(actions) and actions[i]['action'] in ['left_change', 'right_change']:
                    i += 1
                groups.append((start, i - 1))  # end_idx是最后一个换道的索引
            else:
                i += 1
        return groups

    def generate_full_trajectory(self, path: List[str],
                                 actions: List[dict]) -> Tuple[np.ndarray, List[dict]]:
        """
        生成完整的规划轨迹（支持连续换道合并）

        参数:
            path: 车道ID列表
            actions: 动作列表

        返回:
            (完整轨迹点数组, 轨迹段信息列表)
        """
        if not path:
            return np.array([]), []

        # 找出连续换道区间
        lc_groups = self._find_consecutive_lane_changes(actions)

        # 标记哪些索引是连续换道的一部分（非首个）
        skip_indices = set()
        lc_group_starts = {}
        for start_idx, end_idx in lc_groups:
            if end_idx > start_idx:  # 有连续换道
                for j in range(start_idx + 1, end_idx + 1):
                    skip_indices.add(j)
                lc_group_starts[start_idx] = end_idx

        all_points = []
        segments = []

        i = 0
        while i < len(path):
            lane_id = path[i]
            lane_centerline = self.planner.get_lane_centerline(lane_id)
            if lane_centerline is None or len(lane_centerline) == 0:
                i += 1
                continue

            action = actions[i]['action']

            if action in ['start', 'successor', 'follow']:
                # 直接跟随车道中心线
                segment_points = lane_centerline.copy()
                segments.append({
                    'type': 'lane_follow',
                    'lane_id': lane_id,
                    'start_idx': len(all_points),
                    'end_idx': len(all_points) + len(segment_points)
                })
                all_points.extend(segment_points.tolist())
                i += 1

            elif action in ['left_change', 'right_change']:
                from_lane_id = actions[i]['from_lane']
                from_centerline = self.planner.get_lane_centerline(from_lane_id)

                if from_centerline is None or len(from_centerline) == 0:
                    i += 1
                    continue

                # 检查是否是连续换道的开始
                if i in lc_group_starts:
                    # 连续换道：直接从起始车道换到最终目标车道
                    final_idx = lc_group_starts[i]
                    final_lane_id = path[final_idx]
                    final_centerline = self.planner.get_lane_centerline(final_lane_id)

                    if final_centerline is not None and len(final_centerline) > 0:
                        # 计算跨越的车道数，调整换道曲线长度
                        num_lanes = final_idx - i + 1

                        # 生成一条直接到最终车道的平滑曲线
                        change_curve = self.lane_change_gen.generate_multi_lane_change_curve(
                            from_centerline, final_centerline,
                            num_lanes=num_lanes,
                            num_points=50 * num_lanes
                        )

                        segments.append({
                            'type': 'multi_lane_change',
                            'from_lane': from_lane_id,
                            'to_lane': final_lane_id,
                            'lanes_crossed': num_lanes,
                            'start_idx': len(all_points),
                            'end_idx': len(all_points) + len(change_curve)
                        })
                        all_points.extend(change_curve.tolist())

                        # 换道后跟随最终车道剩余部分
                        start_ratio = min(0.3 + 0.15 * num_lanes, 0.85)
                        start_pt_idx = int(len(final_centerline) * start_ratio)
                        remaining_points = final_centerline[start_pt_idx:].copy()

                        if len(remaining_points) > 0:
                            segments.append({
                                'type': 'lane_follow',
                                'lane_id': final_lane_id,
                                'start_idx': len(all_points),
                                'end_idx': len(all_points) + len(remaining_points)
                            })
                            all_points.extend(remaining_points.tolist())

                        # 跳过连续换道中的其他车道
                        i = final_idx + 1
                    else:
                        i += 1

                elif i in skip_indices:
                    # 这是连续换道的中间部分，已经在上面处理过了，跳过
                    i += 1

                else:
                    # 单次换道
                    change_curve = self.lane_change_gen.generate_lane_change_curve(
                        from_centerline, lane_centerline,
                        start_ratio=0.3, end_ratio=0.7
                    )

                    segments.append({
                        'type': 'lane_change',
                        'from_lane': from_lane_id,
                        'to_lane': lane_id,
                        'action': action,
                        'start_idx': len(all_points),
                        'end_idx': len(all_points) + len(change_curve)
                    })
                    all_points.extend(change_curve.tolist())

                    # 换道后继续跟随目标车道的剩余部分
                    start_idx = int(len(lane_centerline) * 0.7)
                    remaining_points = lane_centerline[start_idx:].copy()

                    if len(remaining_points) > 0:
                        segments.append({
                            'type': 'lane_follow',
                            'lane_id': lane_id,
                            'start_idx': len(all_points),
                            'end_idx': len(all_points) + len(remaining_points)
                        })
                        all_points.extend(remaining_points.tolist())

                    i += 1
            else:
                i += 1

        return np.array(all_points), segments


