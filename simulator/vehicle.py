"""离线仿真与 ROS 包装层共用的车辆侧规划适配器。

这个文件包含两个面向规划器的车辆抽象：
- ``VUT``：被测车路径工具
- ``GVT``：新的阶段 1 局部规划入口
"""

import math
from typing import Literal

import numpy as np

# 被测车规划相关
from global_planner.curve_planner.planner import CurvePlanner, TrajectoryGenerator
from global_planner.curve_planner.tools import project_and_cut_path

from drivable_lanes_planner.planner import DrivableLanesExtender
from traj_planner.planner import LocalPlanner


class VUT:
    def __init__(self, ptcId, outline, sportParams, xodr_parser):
        self.ptcId = ptcId

        # 尺寸参数
        self.L = outline.get("height", 3950) / 1000  # 长
        self.W = outline.get("width", 2330) / 1000  # 宽
        self.l = outline.get("wheelbase", 2500) / 1000  # 轴距

        # 约束参数
        self.max_speed = sportParams.get("maxSpeed", 25 / 3.6)
        self.max_acc = sportParams.get("maxAcc", 5.0)
        self.min_acc = -sportParams.get("minAcc", 3.0)

        # 运动参数
        self.pth = None
        self.x, self.y, self.heading = None, None, None
        self.v, self.a = 0, 0

        # 其他
        self.if_arrive = False  # 是否到达终点
        self.planner = CurvePlanner(xodr_parser)

    def path_planning(self, s_point, g_point):
        lane_id_list, action = self.planner.plan_route(s_point[0], s_point[1], g_point[0], g_point[1])
        traj_gen = TrajectoryGenerator(self.planner)
        path, segments = traj_gen.generate_full_trajectory(lane_id_list, action)
        path_with_angles = project_and_cut_path(path, s_point, g_point)
        self.pth = path_with_angles


class GVT:
    """围绕全局路线缓存和局部规划器的自车包装类。"""

    def __init__(self, ptcId, outline, sportParams, xodr_parser):
        self.ptcId = ptcId
        wheelbase_mm = outline.get("wheelbase", 2700) or 2700
        if wheelbase_mm <= 0:
            wheelbase_mm = 2700
        self.wheelbase = wheelbase_mm / 1000.0
        # 速度单位统一m/s
        protocol_max_v = 25.0 / 3.6
        vehicle_max_v = sportParams.get("maxSpeed", protocol_max_v * 3.6) / 3.6
        target_max_v = max(0.0, min(vehicle_max_v, protocol_max_v))

        # 配置参数
        self.config = {
            "size": {
                "L": outline.get("height", 2950) / 1000,
                "W": outline.get("width", 1720) / 1000,
            },
            "sg_info": {
                "sp": None,
                "gp": None,
            },
            "constrain": {
                "min_v": 0.0,
                "max_v": vehicle_max_v,
                "target_v": target_max_v,
                "min_acc": -sportParams.get("minAcc", 2.0),
                "max_acc": sportParams.get("maxAcc", 3.0),
                "max_jerk": 2.5,
                "overspeed_eps": 0.05,
                "max_acc_h": 3.0,
                "max_kappa": 0.3,
            },
            "MOBIL": {
                "politeness": 0.5,
                "acc_threshold": 0.2,
                "prepare_threshold": 0.2,
                "execute_threshold": 0.4,
                "safe_decel_threshold": 4.0,
                "bias_right": 0.1,
                "bias_left": 0.0,
                "score_scale": 1.5,
            },
            "IDM": {
                "desired_v": target_max_v,
                "time_gap": 1.5,
                "min_gap": 2.0,
                "max_acc": sportParams.get("maxAcc", 2.0),
                "comfort_dec": sportParams.get("minAcc", 1.0),
                "delta": 4.0,
                "longitudinal_buffer": 0.5,
                "lateral_buffer": 0.3,
                # 曲率耦合期望速度参数
                "curve_lat_acc_limit": 1.8,  # m/s^2  # 横向加速度
                "curve_lookahead_s": 8.0,  # m  # 预瞄8m
                "curve_kappa_eps": 1e-3,
                "curve_min_speed": 2.0,  # m/s
            },
            "TTC": {
                "hard_min": 1.5,
                "soft_min": 2.0,
                "soft_max": 6.0,
                "prepare_threshold": 6.0,
                "execute_threshold": 4.5,
                "stop_margin_prepare_threshold": 8.0,
                "stop_margin_execute_threshold": 3.0,
                "closing_speed_eps": 0.2,
            },
            "SCORE": {
                "ttc": 25.0,
                "stop_margin": 10.0,
                "clearance": 10.0,
                "mobil": 20.0,
                "efficiency": 15.0,
                "comfort": 15.0,
                "consistency": 5.0,
                "comfort_lat_ratio": 0.40,
                "comfort_acc_ratio": 0.35,
                "comfort_jerk_ratio": 0.25,
            },
            "local_fsm": {
                "enter_complete_lateral_thresh": 0.3,  # 多远认为完成换道
                "planner_cycle_dt": 0.1,  # 行为状态机默认周期
                "dlc_cooldown_s": 2.0,  # 自由换道完成后的冷却时间
                "dlc_cooldown_release_ttc": 2.5,  # 危险态豁免：TTC 低于该值时允许打破冷却
                "dlc_cooldown_release_stop_margin": 1.0,  # 危险态豁免：停车余量低于该值时允许打破冷却
                "arrive_complete_dist": 0.8,  # 宣布到达的距离阈值 (m)
                "arrive_complete_speed": 0.3,  # 宣布到达的速度阈值 (m/s)
                "arrive_soft_decel": 2.0,  # 到达阶段软减速能力 (m/s^2)
                "arrive_emergency_decel": 3.0,  # 到达阶段紧急减速度 (m/s^2)
                "arrive_emergency_min_dist": 3.0,  # 紧急制动触发最小距离阈值 (m)
                "arrive_emergency_buffer": 0.8,  # 紧急制动距离安全余量 (m)
                "arrive_full_stop_dist": 0.8,  # 该距离内强制软限速收敛到0 (m)
                "approach_goal_s_window": 0.5,  # 终点所在车道上的纵向允许窗口半宽
            },
        }

        # 运动属性
        self.x, self.y, self.heading = None, None, None
        self.v, self.a, self.kappa = 0, 0.0, 0  # TODO. 加速度默认为0
        self.Gear = 1  # 默认前进

        # 规划相关
        self.s_point = None  # 起点大地平面坐标
        self.g_point = None  # 终点大地平面坐标
        self.s_heading = None  # 是初始时刻正北顺时针增加的rad

        self.trajectory = []
        # 其他
        self.refer_lines = []
        self.refer_line_lane_ids = []
        self.refer_line_lane_sequences = []
        self.refer_line_lateral_steps = []
        self.current_road_id = None
        self.pre_traj = None
        self.pre_traj_id = 0
        self.planner_mode: Literal["scene_target", "local_sampling"] = "scene_target"
        self.scene_ref_lane_count = 5
        self.scene_runtime_step_index = 0
        self.last_planning_frame = None
        self.last_planning_input = None
        # 规控器
        self.lanePlanner = DrivableLanesExtender(xodr_parser)
        self.traj_planner = LocalPlanner(self.config)
        self.scene_target_provider = self._build_scene_target_provider(xodr_parser)

    def _build_scene_target_provider(self, xodr_parser):
        """延迟引入 scene pipeline，避免模块循环依赖。"""

        from planning_runtime.scene.provider import SceneTargetProvider

        return SceneTargetProvider(
            xodr_parser=xodr_parser,
            main_vehicle=None,
            main_lane_extender=None,
            dt=float(self.config["local_fsm"]["planner_cycle_dt"]),
            max_lane_segments=int(self.scene_ref_lane_count),
        )

    def set_planner_mode(self, planner_mode: str) -> None:
        """切换车辆侧规划入口模式。"""

        normalized = str(planner_mode or "").strip().lower()
        if normalized not in {"scene_target", "local_sampling"}:
            raise ValueError(f"unsupported planner mode: {planner_mode}")
        self.planner_mode = normalized

    def global_planning(self, s_point, g_point, s_v=0.0):
        """初始化全局路线和首帧走廊上下文。"""

        # 设定起点和终点（大地平面坐标），以及起点速度，获得参考线
        self.v = s_v
        self.s_point = s_point
        self.g_point = g_point
        self.config["sg_info"]["sp"] = s_point
        self.config["sg_info"]["gp"] = g_point

        success = self.lanePlanner.plan_shortest_path(s_point[0], s_point[1], g_point[0], g_point[1])
        if not success:
            raise ValueError("规划不出路线")

        # 获取初始参考线（使用起点坐标）
        result = self.lanePlanner.plan_extended_centerlines(self.s_point[0], self.s_point[1])
        self.refer_lines = result.extended_centerlines
        self.refer_line_lane_sequences = [list(seq) for seq in result.extended_lane_ids]
        self.refer_line_lane_ids = [seq[0] if seq else "" for seq in self.refer_line_lane_sequences]
        self.refer_line_lateral_steps = self._compute_lateral_steps(self.refer_line_lane_ids)
        self.current_road_id = result.query_road_id
        self.scene_runtime_step_index = 0
        if self.scene_target_provider is not None:
            self.scene_target_provider.history.clear()
            self.scene_target_provider.last_scene_context = None

    def _refresh_refer_lines(self) -> None:
        """按当前位置刷新当前规划走廊。"""

        result = self.lanePlanner.plan_extended_centerlines(self.x, self.y)
        if result is not None and result.extended_centerlines:
            # 过滤掉空线和单点线，避免后续曲线生成失败
            refreshed_lines = []
            refreshed_lane_sequences = []
            refreshed_lane_ids = []
            for line, lane_seq in zip(result.extended_centerlines, result.extended_lane_ids):
                if line is None or len(line) <= 1:
                    continue
                refreshed_lines.append(line)
                refreshed_lane_sequences.append(list(lane_seq))
                refreshed_lane_ids.append(lane_seq[0] if lane_seq else "")
            if refreshed_lines:
                self.refer_lines = refreshed_lines
                self.refer_line_lane_sequences = refreshed_lane_sequences
                self.refer_line_lane_ids = refreshed_lane_ids
                self.refer_line_lateral_steps = self._compute_lateral_steps(self.refer_line_lane_ids)
                self.current_road_id = result.query_road_id

    def traj_planning(self, obsList):
        """执行一帧局部规划，走阶段 1 新流水线。"""

        provider = self.scene_target_provider
        provider.set_runtime(
            step_index=int(self.scene_runtime_step_index),
            time_s=float(self.scene_runtime_step_index) * float(provider.dt),
        )
        from planning_runtime.frame import build_planning_frame
        from planning_runtime.modes.local_sampling import prepare_local_sampling_input
        from planning_runtime.modes.scene_target import prepare_scene_target_input

        try:
            planning_frame = build_planning_frame(
                ego_vehicle=self,
                provider=provider,
                obstacles=list(obsList),
            )
            if self.planner_mode == "scene_target":
                planning_input = prepare_scene_target_input(
                    planning_frame=planning_frame,
                    provider=provider,
                    planner_memory=self.traj_planner.memory,
                )
            else:
                planning_input = prepare_local_sampling_input(
                    planning_frame=planning_frame,
                    planner_memory=self.traj_planner.memory,
                )
            self.last_planning_frame = planning_frame
            self.last_planning_input = planning_input
            trajectory = self.traj_planner.plan(planning_input=planning_input)
        finally:
            self.scene_runtime_step_index += 1

        if trajectory is None:
            return None
        self.trajectory = trajectory
        return trajectory

    def _compute_lateral_steps(self, lane_ids):
        """计算候选线相对当前主线的显式横向阶数。"""

        if not lane_ids:
            return []

        current_lane_id = lane_ids[0]
        steps = {current_lane_id: 0}
        queue = [current_lane_id]
        parser_lanes = self.lanePlanner.parser.lanes

        while queue:
            lane_id = queue.pop(0)
            lane_info = parser_lanes.get(lane_id)
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

    def update_pre_traj(self, ):
        if self.pre_traj_id not in self.pre_traj:
            print(f"⚠ 当前段ID {self.pre_traj_id} 不在预轨迹中")
            return {}

        sorted_seg_ids = sorted(self.pre_traj.keys())
        current_seg_idx = sorted_seg_ids.index(self.pre_traj_id)
        preserved_seg_ids = sorted_seg_ids[current_seg_idx: current_seg_idx + 2]

        if len(preserved_seg_ids) < 2:
            print("⚠ 剩余预轨迹不足以延后一段替换，跳过本次更新")
            return {}

        anchor_seg_id = preserved_seg_ids[-1]
        anchor_seg_xyv = np.array(self.pre_traj[anchor_seg_id], copy=True)
        traj_xy = self.trajectory[:, :2]  # (21, 2)
        P_end_xy = np.array(anchor_seg_xyv[-1, :2])  # shape (2,)
        idx = np.argmin(np.linalg.norm(traj_xy - P_end_xy, axis=1)) # TODO. 最近点索引，找到规划轨迹从哪开始
        print([idx, self.trajectory.shape, anchor_seg_id])
        P_end_full = np.empty(6)
        P_end_full[:2] = P_end_xy
        P_end_full[2:] = self.trajectory[idx, 2:]  # heading, v, a, kappa
        self.a=self.trajectory[idx,4]
        P_end_full[3] = anchor_seg_xyv[-1, 2]  # 速度更改回原始速度
        new_full_seq = np.vstack([P_end_full, self.trajectory[idx + 1:]])

        if new_full_seq.shape[0] < 2:
            print("⚠ 规划轨迹不足以从下一段之后开始替换，跳过本次更新")
            return {}

        new_full_seq[:, 3] = np.clip(new_full_seq[:, 3], 0.0, self.config["constrain"]["max_v"])
        if new_full_seq.shape[0] > 2:
            new_full_seq[1, 3] = (new_full_seq[0, 3] + new_full_seq[2, 3]) / 2
        N = new_full_seq.shape[0]

        # === 分段：每段固定n点，相邻段重叠起终点 ===
        n = 2
        new_full_segments = {}
        next_seg_id = int(anchor_seg_id) + 1
        pos = 0
        new_pre_traj = {
            seg_id: np.array(self.pre_traj[seg_id], copy=True)
            for seg_id in preserved_seg_ids
        }
        while True:
            remaining = N - pos
            if remaining < n:
                break
            else:
                seg_full = new_full_seq[pos: pos + n]  # (3, 6)
                pos += n-1  # 下一段从当前段最后一个点开始
                new_pre_traj[next_seg_id] = seg_full[:, [0, 1, 3]]
                new_full_segments[next_seg_id] = seg_full
                next_seg_id += 1
        self.pre_traj = new_pre_traj
        return new_full_segments

    def cal_pre_traj_id(self):
        if not self.pre_traj:
            return  # 无预轨迹，直接返回

        current_x, current_y = self.x, self.y
        first_seg_id = sorted(self.pre_traj.keys())[0]
        proj_1_2 = np.cos(self.heading) * (self.pre_traj[first_seg_id][-1, 0] - current_x) + np.sin(self.heading) * (
                self.pre_traj[first_seg_id][-1, 1] - current_y)
        # TODO. 定位在哪一段
        if proj_1_2 > 0:
            self.pre_traj_id = first_seg_id  # 说明在第一段或者还未行驶至第一段
            # self.pre_traj = {self.pre_traj_id: self.pre_traj[self.pre_traj_id]}  # 只保留第一段
            return
        for seg_id in sorted(self.pre_traj.keys()):
            # 遍历所有 segment，找全局最近点所在的 segment
            proj_1 = np.cos(self.heading) * (self.pre_traj[seg_id][0, 0] - current_x) + np.sin(self.heading) * (
                    self.pre_traj[seg_id][0, 1] - current_y)
            proj_2 = np.cos(self.heading) * (self.pre_traj[seg_id][-1, 0] - current_x) + np.sin(self.heading) * (
                    self.pre_traj[seg_id][-1, 1] - current_y)
            if proj_1 <= 0 < proj_2:
                self.pre_traj_id = seg_id  # 前后两段之间
                # self.pre_traj = {self.pre_traj_id: self.pre_traj[self.pre_traj_id]}  # 只保留第一段
                return
        print("⚠ 未找到当前所在路段")
