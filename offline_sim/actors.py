"""离线仿真器中的车辆模型辅助函数。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from global_planner.curve_planner.planner import CurvePlanner, TrajectoryGenerator
from global_planner.curve_planner.tools import project_and_cut_path
from planning_runtime.scene.map_views import slice_centerline_from_projection
from simulator.vehicle import GVT
from traj_planner.tools import calculate_path_cumulative_distance, get_path_point_at_s, project_point_to_path
from xodr_parser.xodr_parser import XODRParser


def instantiate_ego_vehicle(xodr_parser: XODRParser, max_speed_kph: float = 25.0) -> GVT:
    """创建离线仿真自车，并补齐可视化所需字段。"""

    outline = {"height": 4640.0, "width": 1780.0, "wheelbase": 2700.0}
    sport_params = {"maxSpeed": float(max_speed_kph), "maxAcc": 3.0, "minAcc": 3.0}
    ego = GVT("ego", outline, sport_params, xodr_parser)
    ego.length = ego.config["size"]["L"]
    ego.width = ego.config["size"]["W"]
    ego.height = 1.5
    ego.ptc_id = "ego"
    return ego


class IDMRouteVehicle:
    """沿固定路线运动的主车 / 障碍车模型。"""

    def __init__(
        self,
        xodr_parser: XODRParser,
        start_xy: np.ndarray,
        goal_xy: np.ndarray | None,
        start_speed: float,
        desired_speed: float,
        motion_mode: str = "idm",
        lane_change_side: str = "left",
        lane_change_start_s: float = 10.0,
        lane_change_duration_s: float = 4.0,
    ) -> None:
        self.ptc_id = "main"
        self.motion_mode = str(motion_mode)
        self.length = 4.6
        self.width = 1.8
        self.height = 1.5
        self.desired_speed = float(desired_speed)
        # 离线场景测试使用偏不保守的基础 IDM，避免主车过早消解 cut-out 风险。
        self.max_acc = 2.5
        self.comfort_decel = 3.5
        self.min_gap = 1.0
        self.time_headway = 0.6
        self.max_decel = 4.0
        self.delta = 4.0
        self.lateral_threshold = 2.5
        self.timestamp_ms = 0
        self.goal_xy = None if goal_xy is None else np.asarray(goal_xy, dtype=float)
        self.progress_s = 0.0
        self.route_lane_ids: list[str] = []
        self.static_lane_id = ""
        self.lane_change_side = str(lane_change_side or "left").strip().lower()
        self.lane_change_start_s = max(0.0, float(lane_change_start_s))
        self.lane_change_duration_s = max(0.1, float(lane_change_duration_s))
        self.lane_change_elapsed_s = 0.0
        self.lane_change_started = False
        self.lane_change_completed = False
        self.script_source_line: np.ndarray | None = None
        self.script_source_cum: np.ndarray | None = None
        self.script_target_line: np.ndarray | None = None
        self.script_target_cum: np.ndarray | None = None
        self.script_start_lane_s = 0.0

        planner = CurvePlanner(xodr_parser)
        if self.motion_mode == "static":
            lane_id, _ = planner.find_nearest_lane(np.asarray(start_xy, dtype=float))
            lane_id = str(lane_id)
            centerline = np.asarray(planner.get_lane_centerline(lane_id), dtype=float)
            if centerline.ndim != 2 or len(centerline) < 2:
                raise ValueError("静止主车所在车道中心线无效")

            sliced_path = slice_centerline_from_projection(centerline, np.asarray(start_xy, dtype=float))
            if len(sliced_path) < 2:
                raise ValueError("静止主车所在车道切片点数不足")
            headings = []
            for idx in range(len(sliced_path)):
                if idx == len(sliced_path) - 1:
                    vec = sliced_path[idx] - sliced_path[idx - 1]
                else:
                    vec = sliced_path[idx + 1] - sliced_path[idx]
                headings.append(float(math.atan2(vec[1], vec[0])))
            self.path = np.column_stack([sliced_path, np.asarray(headings, dtype=float)])
            self.path_cumulative_s = calculate_path_cumulative_distance(self.path)
            self.route_lane_ids = [lane_id]
            self.static_lane_id = lane_id
            self.x = float(start_xy[0])
            self.y = float(start_xy[1])
            self.heading = float(headings[0])
            self.v = 0.0
        elif self.motion_mode == "scripted_lane_change":
            if goal_xy is None:
                raise ValueError("scripted_lane_change 模式下主车必须提供终点用于道路延伸")
            self._init_scripted_lane_change(
                planner=planner,
                start_xy=np.asarray(start_xy, dtype=float),
                start_speed=float(start_speed),
            )
        else:
            if goal_xy is None:
                raise ValueError(f"{self.motion_mode} 模式下主车必须提供终点")
            lane_id_list, actions = planner.plan_route(start_xy[0], start_xy[1], goal_xy[0], goal_xy[1])
            if not lane_id_list:
                raise ValueError("主车全局路线规划失败")
            traj_gen = TrajectoryGenerator(planner)
            path_xy, _ = traj_gen.generate_full_trajectory(lane_id_list, actions)
            path = project_and_cut_path(path_xy, start_xy, goal_xy)
            if path.ndim != 2 or len(path) < 2:
                raise ValueError("主车路线点数不足")

            self.path = np.asarray(path, dtype=float)
            self.path_cumulative_s = calculate_path_cumulative_distance(self.path)
            self.route_lane_ids = list(lane_id_list)
            self.x = float(self.path[0, 0])
            self.y = float(self.path[0, 1])
            self.heading = float(self.path[0, 2])
            self.v = float(max(start_speed, 0.0))
        self.a = 0.0
        self.finished = False

    def _init_scripted_lane_change(
        self,
        planner: CurvePlanner,
        start_xy: np.ndarray,
        start_speed: float,
    ) -> None:
        """初始化脚本换道主车模型。

        横向运动不使用全局规划器生成的换道曲线，而是在起始车道和指定侧
        相邻车道中心线之间做平滑插值，避免离线全局换道路径出现断点。
        纵向速度仍在 ``step`` 中按 IDM 更新。
        """
        source_lane_id, _ = planner.find_nearest_lane(np.asarray(start_xy, dtype=float))
        source_lane_id = str(source_lane_id)
        lane_info = planner.parser.lanes.get(source_lane_id, {})
        if self.lane_change_side == "left":
            target_lane_id = str(lane_info.get("left_lane") or "")
        elif self.lane_change_side == "right":
            target_lane_id = str(lane_info.get("right_lane") or "")
        else:
            raise ValueError(f"不支持的 scripted_lane_change 方向: {self.lane_change_side}")
        if not target_lane_id or target_lane_id not in planner.parser.lanes:
            raise ValueError(f"起始车道 {source_lane_id} 不存在 {self.lane_change_side} 侧相邻车道")

        source_line = np.asarray(planner.get_lane_centerline(source_lane_id), dtype=float)
        target_line = np.asarray(planner.get_lane_centerline(target_lane_id), dtype=float)
        if source_line.ndim != 2 or len(source_line) < 2:
            raise ValueError(f"起始车道 {source_lane_id} 中心线无效")
        if target_line.ndim != 2 or len(target_line) < 2:
            raise ValueError(f"目标车道 {target_lane_id} 中心线无效")

        self.script_source_line = np.asarray(source_line[:, :2], dtype=float)
        self.script_target_line = np.asarray(target_line[:, :2], dtype=float)
        self.script_source_cum = calculate_path_cumulative_distance(self.script_source_line)
        self.script_target_cum = calculate_path_cumulative_distance(self.script_target_line)
        _, source_s, _ = project_point_to_path(np.asarray(start_xy, dtype=float), self.script_source_line)
        self.script_start_lane_s = float(source_s)
        self.route_lane_ids = [source_lane_id, target_lane_id]
        self.v = float(max(start_speed, 0.0))
        self.path = self._build_scripted_lane_change_path()
        self.path_cumulative_s = calculate_path_cumulative_distance(self.path)
        self.x = float(start_xy[0])
        self.y = float(start_xy[1])
        _, heading = self._scripted_pose_at_progress(0.0, 0.0, previous_xy=None)
        self.heading = float(heading)

    def _build_scripted_lane_change_path(self) -> np.ndarray:
        """生成用于主车参考线投影和可视化的连续脚本换道路径。"""
        if (
            self.script_source_line is None
            or self.script_target_line is None
            or self.script_source_cum is None
            or self.script_target_cum is None
        ):
            raise ValueError("scripted_lane_change 尚未初始化中心线")
        source_end_s = float(self.script_source_cum[-1])
        start_s = float(self.script_start_lane_s)
        end_s = source_end_s
        if end_s <= start_s + 1.0:
            raise ValueError("scripted_lane_change 起点距离车道终点过近")
        sample_s = np.linspace(start_s, end_s, max(30, int((end_s - start_s) / 0.5)))
        approx_speed = max(float(self.desired_speed), float(self.v), 1.0)
        change_start_s = start_s + float(self.lane_change_start_s)
        change_end_s = change_start_s + approx_speed * float(self.lane_change_duration_s)
        points = []
        for lane_s in sample_s:
            if lane_s <= change_start_s:
                ratio = 0.0
            elif lane_s >= change_end_s:
                ratio = 1.0
            else:
                ratio = _smoothstep((lane_s - change_start_s) / max(change_end_s - change_start_s, 1e-3))
            source_xy, _ = get_path_point_at_s(self.script_source_line, lane_s, self.script_source_cum)
            target_s = min(lane_s, float(self.script_target_cum[-1]))
            target_xy, _ = get_path_point_at_s(self.script_target_line, target_s, self.script_target_cum)
            xy = (1.0 - ratio) * source_xy + ratio * target_xy
            points.append(xy)
        points_arr = np.asarray(points, dtype=float)
        headings = _headings_from_xy(points_arr)
        return np.column_stack([points_arr, headings])

    def _scripted_pose_at_progress(
        self,
        progress_s: float,
        blend: float,
        previous_xy: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        """按当前纵向进度和横向换道比例计算主车位姿。"""
        if (
            self.script_source_line is None
            or self.script_target_line is None
            or self.script_source_cum is None
            or self.script_target_cum is None
        ):
            raise ValueError("scripted_lane_change 尚未初始化中心线")
        lane_s = min(float(self.script_start_lane_s) + float(progress_s), float(self.script_source_cum[-1]))
        source_xy, _ = get_path_point_at_s(self.script_source_line, lane_s, self.script_source_cum)
        target_s = min(lane_s, float(self.script_target_cum[-1]))
        target_xy, _ = get_path_point_at_s(self.script_target_line, target_s, self.script_target_cum)
        xy = (1.0 - float(blend)) * source_xy + float(blend) * target_xy
        if previous_xy is not None:
            delta = xy - previous_xy
            if float(np.linalg.norm(delta)) > 1e-6:
                return xy, float(math.atan2(delta[1], delta[0]))
        lookahead_s = min(lane_s + 0.5, float(self.script_source_cum[-1]))
        source_next, _ = get_path_point_at_s(self.script_source_line, lookahead_s, self.script_source_cum)
        target_next_s = min(lookahead_s, float(self.script_target_cum[-1]))
        target_next, _ = get_path_point_at_s(self.script_target_line, target_next_s, self.script_target_cum)
        next_xy = (1.0 - float(blend)) * source_next + float(blend) * target_next
        delta = next_xy - xy
        return xy, float(math.atan2(delta[1], delta[0]))

    def _find_leader(self, obstacles: list[Any]) -> tuple[float, float] | None:
        best_gap = float("inf")
        best_v = 0.0
        for obstacle in obstacles:
            if obstacle is None or obstacle is self:
                continue
            obs_pos = np.array([float(obstacle.x), float(obstacle.y)], dtype=float)
            _, obs_s, obs_d = project_point_to_path(obs_pos, self.path)
            if abs(obs_d) > self.lateral_threshold:
                continue
            if obs_s <= self.progress_s:
                continue
            obstacle_length = float(getattr(obstacle, "length", getattr(obstacle, "L", 4.5)))
            gap = float(obs_s - self.progress_s - 0.5 * self.length - 0.5 * obstacle_length)
            if gap < best_gap:
                best_gap = gap
                best_v = float(max(getattr(obstacle, "v", 0.0), 0.0))
        if not math.isfinite(best_gap):
            return None
        return max(best_gap, 0.1), best_v

    def _compute_idm_acc(self, leader: tuple[float, float] | None) -> float:
        free_term = 1.0 - (self.v / max(self.desired_speed, 1e-3)) ** self.delta
        acc = self.max_acc * free_term
        if leader is not None:
            gap, leader_v = leader
            delta_v = self.v - leader_v
            braking_term = self.v * delta_v / (2.0 * math.sqrt(self.max_acc * self.comfort_decel))
            s_star = self.min_gap + max(0.0, self.v * self.time_headway + braking_term)
            acc = self.max_acc * (free_term - (s_star / max(gap, 0.1)) ** 2)
        return float(np.clip(acc, -float(self.max_decel), self.max_acc))

    def step(self, obstacles: list[Any], dt: float, time_s: float) -> None:
        if self.motion_mode == "static":
            self.a = 0.0
            self.v = 0.0
            self.timestamp_ms = int(round(time_s * 1000.0))
            return

        if self.finished:
            self.a = 0.0
            self.v = 0.0
            self.timestamp_ms = int(round(time_s * 1000.0))
            return

        if self.motion_mode == "constant":
            self.a = 0.0
            travel_s = max(0.0, self.v * dt)
            next_v = float(self.v)
        else:
            leader = self._find_leader(obstacles)
            self.a = self._compute_idm_acc(leader)
            travel_s = max(0.0, self.v * dt + 0.5 * self.a * dt * dt)
            next_v = max(0.0, min(self.v + self.a * dt, self.desired_speed))
        self.progress_s = min(self.progress_s + travel_s, float(self.path_cumulative_s[-1]))

        if self.motion_mode == "scripted_lane_change":
            if self.progress_s >= float(self.lane_change_start_s):
                self.lane_change_started = True
                self.lane_change_elapsed_s = min(
                    float(self.lane_change_duration_s),
                    self.lane_change_elapsed_s + float(dt),
                )
            blend = _smoothstep(self.lane_change_elapsed_s / max(float(self.lane_change_duration_s), 1e-3))
            self.lane_change_completed = bool(blend >= 0.999)
            previous_xy = np.array([float(self.x), float(self.y)], dtype=float)
            pos, heading = self._scripted_pose_at_progress(self.progress_s, blend, previous_xy)
        else:
            pos, heading = get_path_point_at_s(self.path, self.progress_s, self.path_cumulative_s)
        self.x = float(pos[0])
        self.y = float(pos[1])
        self.heading = float(heading)
        self.v = next_v
        self.timestamp_ms = int(round(time_s * 1000.0))

        if self.progress_s >= float(self.path_cumulative_s[-1]) - 0.2:
            self.finished = True
            self.v = 0.0
            self.a = 0.0


def serialize_obstacle_input(obstacle: Any) -> dict[str, Any]:
    """把单个规划障碍物序列化成 JSON 可写结构。"""

    return {
        "ptc_id": str(getattr(obstacle, "ptc_id", getattr(obstacle, "ptcId", "main"))),
        "x": float(getattr(obstacle, "x")),
        "y": float(getattr(obstacle, "y")),
        "heading": float(getattr(obstacle, "heading", 0.0)),
        "v": float(getattr(obstacle, "v", 0.0)),
        "a": float(getattr(obstacle, "a", 0.0)),
        "length": float(getattr(obstacle, "length", getattr(obstacle, "L", 4.5))),
        "width": float(getattr(obstacle, "width", getattr(obstacle, "W", 1.8))),
        "timestamp_ms": int(getattr(obstacle, "timestamp_ms", 0) or 0),
    }


def _smoothstep(value: float) -> float:
    """把 0-1 进度转换成两端速度为 0 的平滑换道比例。"""
    x = float(np.clip(value, 0.0, 1.0))
    return float(x * x * (3.0 - 2.0 * x))


def _headings_from_xy(points_xy: np.ndarray) -> np.ndarray:
    """根据连续 xy 点计算航向角序列。"""
    points = np.asarray(points_xy, dtype=float)
    headings = []
    for idx in range(len(points)):
        if len(points) < 2:
            headings.append(0.0)
            continue
        if idx == len(points) - 1:
            vec = points[idx] - points[idx - 1]
        else:
            vec = points[idx + 1] - points[idx]
        headings.append(float(math.atan2(vec[1], vec[0])))
    return np.asarray(headings, dtype=float)
