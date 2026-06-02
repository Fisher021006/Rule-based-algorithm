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


def instantiate_ego_vehicle(xodr_parser: XODRParser) -> GVT:
    """创建离线仿真自车，并补齐可视化所需字段。"""

    outline = {"height": 4640.0, "width": 1780.0, "wheelbase": 2700.0}
    sport_params = {"maxSpeed": 25.0, "maxAcc": 3.0, "minAcc": 3.0}
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
    ) -> None:
        self.ptc_id = "main"
        self.motion_mode = str(motion_mode)
        self.length = 4.6
        self.width = 1.8
        self.height = 1.5
        self.desired_speed = float(desired_speed)
        self.max_acc = 2.0
        self.comfort_decel = 2.5
        self.min_gap = 2.0
        self.time_headway = 1.5
        self.delta = 4.0
        self.lateral_threshold = 2.5
        self.timestamp_ms = 0
        self.goal_xy = None if goal_xy is None else np.asarray(goal_xy, dtype=float)
        self.progress_s = 0.0
        self.route_lane_ids: list[str] = []
        self.static_lane_id = ""

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
        return float(np.clip(acc, -2.0, self.max_acc))

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
