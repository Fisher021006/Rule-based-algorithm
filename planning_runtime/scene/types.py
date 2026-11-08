"""定义 scene 算法专用的数据结构"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from planning_runtime.frame import PlanningFrame


@dataclass
class ReferenceLineView:
    """面向场景算法的一条参考线快照。"""

    line_index: int
    lane_id: str
    lane_sequence: list[str]
    lateral_step: int
    line: np.ndarray
    cumulative_s: np.ndarray
    current_s: float
    current_d: float


@dataclass
class SceneRouteIntent:
    """车辆由起终点派生出的全局路线意图。"""

    start_xy: np.ndarray | None
    goal_xy: np.ndarray | None
    planned_path: list[str] = field(default_factory=list)
    planned_actions: list[dict[str, Any]] = field(default_factory=list)
    road_sequence: list[str] = field(default_factory=list)
    route_line: ReferenceLineView | None = None
    corridor_line: ReferenceLineView | None = None
    valid: bool = False
    reason: str = ""


@dataclass
class SceneVehicleState:
    """面向场景算法的车辆状态快照。

    x、y 沿用运行时的车辆几何中心约定；不是车头或后轴点。自车实车入口
    已校正 GPS 安装偏移，外部主车坐标由上游直接提供，Provider 不另作偏移。
    """

    role: str
    x: float
    y: float
    heading: float
    v: float
    a: float
    t: float
    lane_id: str
    road_id: str
    s: float
    d: float
    current_ref_line: ReferenceLineView
    candidate_ref_lines: list[ReferenceLineView]
    relative_x_to_ego: float = 0.0
    relative_y_to_ego: float = 0.0
    # 两车投影到同一条完整自车车道走廊后的弧长差；后方为负。
    relative_s_to_ego: float = 0.0
    # 主车相对该中心线的带符号横向偏移（左正）；保留原口径，不减自车d。
    relative_d_to_ego: float = 0.0
    lane_s: float = 0.0
    lane_d: float = 0.0
    lane_heading: float = 0.0
    lane_centerline: np.ndarray | None = None
    route_intent: SceneRouteIntent | None = None


@dataclass
class SceneContext:
    """场景算法统一消费的一帧语义视图。"""

    frame_index: int
    time_s: float
    dt: float
    ego_vehicle: SceneVehicleState
    main_vehicle: SceneVehicleState | None
    planner_mode: str = "scene_target"
    planning_frame: PlanningFrame | None = None
    planning_input: Any | None = None
    planning_context: Any | None = None

    @property
    def step_index(self) -> int:
        """兼容旧命名：step_index 等价于 frame_index。"""

        return int(self.frame_index)

    @property
    def runtime_frame(self) -> PlanningFrame | None:
        """兼容旧命名：runtime_frame 等价于 planning_frame。"""

        return self.planning_frame


@dataclass
class SceneAlgorithmOutput:
    """框架内部统一使用的场景算法终端目标结构。"""

    x: float
    y: float
    v: float
    a: float
    theta: float
    t: float
    target_line_index: int | None = None
    hold_t: float = 0.0
    reason: str = "scene_algorithm"


# Compatibility aliases for the current transition period.
SimulationViewContext = SceneContext
