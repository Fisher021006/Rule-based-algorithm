"""阶段 1 局部规划流水线的核心数据类型定义。

这个模块只定义数据形状和接口，不包含规划算法逻辑。

数据流：
1. ``simulator/vehicle.py`` 构造 :class:`PlanningInput`
2. ``context.py`` 将其扩展成 :class:`PlanningContext`
3. ``behavior.py`` 输出 :class:`BehaviorDecision`
4. ``sampling.py`` 将行为决策展开为 :class:`TerminalCandidate`
5. ``generator.py`` 把每个候选终端状态生成 :class:`CandidateTrajectory`
6. ``evaluator.py`` 对候选轨迹评分，返回 :class:`PlanningOutput`

所有下游模块统一从这里导入类型定义，这样数据契约是显式且稳定的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from traj_planner.config import VehicleState


class BehaviorMode(str, Enum):
    """当前规划周期的高层行为模式。

    - ``KEEP`` 和 ``APPROACH_GOAL``：不离开当前主参考线
    - ``*_DLC``：自由换道，由局部收益触发
    - ``*_MLC``：强制换道，由路线意图触发
    """

    KEEP = "keep"
    PREPARE_DLC = "prepare_dlc"
    EXECUTE_DLC = "execute_dlc"
    PREPARE_MLC = "prepare_mlc"
    EXECUTE_MLC = "execute_mlc"
    APPROACH_GOAL = "approach_goal"
    ARRIVED = "arrived"


@dataclass
class BehaviorMemory:
    """跨规划周期保留的轻量行为状态。

    输入：
        在每次选出最优轨迹后，由 ``behavior.update_behavior_memory`` 更新。
    输出：
        在下一帧通过 :class:`PlanningInput` 再传回规划器。

    说明：
        ``lane_change_committed`` 是第一版防抖用的“承诺位”。
        一旦某次换道进入执行态，下一帧通常应该继续执行，而不是立刻反悔。
    """

    mode: BehaviorMode = BehaviorMode.KEEP
    target_lane_id: str = ""
    target_line_index: int = 0
    lane_change_committed: bool = False
    dlc_cooldown_cycles: int = 0
    last_selected_line_index: int = 0
    cycle_count: int = 0


@dataclass
class ObstacleSnapshot:
    """规划器内部使用的标准化障碍物记录。

    数据格式：
        全局笛卡尔坐标系，长度单位米，角度单位弧度。

    产生位置：
        由 ``context._normalize_obstacle`` 从
        ``GVT.traj_planning(obsList)`` 传入的原始障碍物对象转换而来。
    """

    ptc_id: str
    x: float
    y: float
    v: float
    heading: float = 0.0
    a: float = 0.0
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    timestamp_ms: int = 0
    raw: Any = None


@dataclass
class ProjectedObstacle:
    """投影到某一条候选参考线后的障碍物信息。

    字段含义：
        - ``s`` / ``d``：该障碍物在候选线上的类 Frenet 坐标
        - ``gap``：沿该线相对自车的纵向距离
        - ``relative_speed``：用于 gap 和制动启发式判断
    """

    obstacle: ObstacleSnapshot
    s: float
    d: float
    gap: float
    relative_speed: float


@dataclass
class LineContext:
    """当前规划周期内，单条参考线对应的上下文信息。

    输入：
        一条参考线、自车在该线上的投影结果、该线上的前后障碍物投影结果。

    输出用途：
        被行为选择、候选采样、轨迹安全评估共同使用。

    关键字段：
        ``lateral_step`` 是显式横向阶数约束：
        - 0：当前车道
        - +/-1：左右一阶相邻车道
        - +/-2：相隔两条车道
        依此类推
    """

    line_index: int
    line_id: str
    lane_sequence: list[str]
    lateral_step: int
    line: np.ndarray
    cumulative_s: np.ndarray
    current_s: float
    current_d: float
    remaining_length: float
    front_obstacle: Optional[ProjectedObstacle] = None
    rear_obstacle: Optional[ProjectedObstacle] = None


@dataclass
class RouteIntent:
    """从全局车道路径中提取出的局部路线意图。

    - ``desired_route_lane_id``：全局路线在当前 road 上最终希望到达的车道
    - ``target_lane_id``：当前这一次局部机动真正要瞄准的车道

    如果全局目标车道相隔多条，本地目标会被收缩成“朝目标方向最近的一阶相邻车道”，
    从而保证一次局部机动不会直接跨多条车道。
    """

    current_lane_id: str
    current_road_id: str
    desired_route_lane_id: str
    target_lane_id: str
    target_line_index: int
    must_change: bool
    deadline_margin_m: float
    distance_to_goal: float
    current_road_planned_lanes: list[str] = field(default_factory=list)


@dataclass
class PlanningInput:
    """从 ``GVT`` 传入 ``LocalPlanner`` 的原始输入包。

    上游：
        ``simulator/vehicle.py`` 根据自车状态、走廊输出、全局路径缓存、
        障碍物列表组装而成。

    下游：
        由 ``context.build_planning_context`` 消费并扩展。

    格式说明：
        - ``refer_lines``: ``list[np.ndarray]``，每条 shape 为 ``(N_i, 2)``
        - ``goal_xy``: ``np.ndarray``，shape 为 ``(2,)``
    """

    ego_state: VehicleState
    obstacles: list[Any]
    refer_lines: list[np.ndarray]
    refer_line_lane_ids: list[str]
    refer_line_lane_sequences: list[list[str]]
    refer_line_lateral_steps: list[int]
    current_road_id: Optional[str]
    planned_path: list[str]
    planned_actions: list[dict]
    road_sequence: list[str]
    goal_xy: np.ndarray
    config: dict
    planner_memory: BehaviorMemory
    external_target: Optional["ExternalTarget"] = None


@dataclass
class ExternalTarget:
    """由外部场景算法直接给出的终端目标。"""

    point_xy: np.ndarray
    target_line_index: int
    target_v: float
    target_a: float = 0.0
    target_heading: Optional[float] = None
    plan_T: float = 2.0
    hold_t: float = 0.0
    source: str = "external"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanningContext:
    """阶段 1 各模块共享的扩展规划上下文。

    它是单帧规划周期内的唯一事实来源：
    自车状态、路线意图、障碍物投影结果、各参考线局部信息都统一存放在这里。
    """

    ego_state: VehicleState
    obstacle_snapshots: list[ObstacleSnapshot]
    line_contexts: list[LineContext]
    route_intent: RouteIntent
    planned_path: list[str]
    planned_actions: list[dict]
    road_sequence: list[str]
    goal_xy: np.ndarray
    config: dict
    planner_memory: BehaviorMemory

    @property
    def current_line(self) -> LineContext:
        """返回当前主参考线。

        约定：
            ``line_contexts[0]`` 永远代表当前正在跟随的主线。
        """

        return self.line_contexts[0]


@dataclass
class BehaviorDecision:
    """单帧行为层的输出。

    这个对象决定“本帧允许采样哪些候选家族”，
    但它本身不是最终轨迹。
    """

    mode: BehaviorMode
    allowed_line_indices: list[int]
    target_line_index: int
    target_lane_id: str
    must_change: bool
    deadline_margin_m: float
    reason: str
    best_mobil_incentive: float = 0.0
    current_front_ttc: float = float("inf")
    current_front_stop_margin: float = float("inf")
    current_front_gap: float = 999.0


@dataclass
class SamplingTemplate:
    """``sampling.py`` 使用的离散候选家族定义。

    采样器会对 ``hold_t``、``plan_T``、``target_v``、``s_bias`` 做离散展开，
    从一个模板得到多条 :class:`TerminalCandidate`。
    """

    name: str
    line_indices: list[int]
    hold_t_values: list[float]
    plan_T_values: list[float]
    target_v_values: list[float]
    s_bias_values: list[float]


@dataclass
class TerminalCandidate:
    """一条在真正生成轨迹之前的候选终端状态。

    含义：
        - ``line_id``：选择目标参考线
        - ``target_s``：选择该线上的纵向终点
        - ``target_v`` / ``plan_T``：描述终端速度与规划时域
        - ``hold_t``：用于“先保持一段再换道”的延迟换道候选
    """

    line_index: int
    line_id: str
    target_s: float
    target_v: float
    plan_T: float
    hold_t: float
    template_name: str


@dataclass
class CandidateTrajectory:
    """某个候选终端状态生成出的真实轨迹及评估信息。

    ``trajectory``:
        ``np.ndarray``，shape ``(N, 6)``
        列顺序：``[x, y, heading, v, a, kappa]``

    ``smooth_path``:
        可选的几何路径，时参之前的中间结果，shape ``(M, 3)``
    """

    terminal: TerminalCandidate
    trajectory: Optional[np.ndarray] = None
    smooth_path: Optional[np.ndarray] = None
    feasible: bool = False
    score: float = float("inf")
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""


@dataclass
class PlanningOutput:
    """一次 LocalPlanner 周期的最终输出。

    - ``best_trajectory``：若所有候选都未通过硬约束，则可能为 ``None``
    - ``candidates``：保留下来供 debug 和离线分析使用
    """

    best_trajectory: Optional[np.ndarray]
    decision: BehaviorDecision
    candidates: list[CandidateTrajectory]
    planner_memory: BehaviorMemory
