"""规划运行时公共输入帧与构帧逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from traj_planner.config import VehicleState
from traj_planner.context import build_planning_context
from traj_planner.types import PlanningInput

if TYPE_CHECKING:
    from planning_runtime.scene.provider import SceneTargetProvider
    from simulator.vehicle import GVT


@dataclass
class PlanningActorState:
    """规划帧里的原始参与者快照。"""

    ptc_id: str
    x: float
    y: float
    heading: float
    v: float
    a: float = 0.0
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    timestamp_ms: int = 0
    lane_id_hint: str = ""
    raw: Any = None


@dataclass
class PlanningFrame:
    """规划系统统一的单帧原始世界状态。"""

    frame_index: int
    time_s: float
    dt: float
    ego_state: VehicleState
    obstacles: list[Any]
    refer_lines: list[np.ndarray]
    refer_line_lane_ids: list[str]
    refer_line_lane_sequences: list[list[str]]
    refer_line_lateral_steps: list[int]
    current_road_id: str | None
    planned_path: list[str]
    planned_actions: list[dict]
    road_sequence: list[str]
    goal_xy: np.ndarray
    config: dict[str, Any]
    main_actor: PlanningActorState | None = None

    @property
    def step_index(self) -> int:
        """兼容旧命名：step_index 等价于 frame_index。"""

        return int(self.frame_index)


def refresh_ego_refer_lines(ego_vehicle: "GVT") -> None:
    """根据自车当前位置刷新当前时刻的参考线集合。"""

    result = ego_vehicle.lanePlanner.plan_extended_centerlines(ego_vehicle.x, ego_vehicle.y)
    if result is not None and result.extended_centerlines:
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
            ego_vehicle.refer_lines = refreshed_lines
            ego_vehicle.refer_line_lane_sequences = refreshed_lane_sequences
            ego_vehicle.refer_line_lane_ids = refreshed_lane_ids
            ego_vehicle.refer_line_lateral_steps = ego_vehicle._compute_lateral_steps(ego_vehicle.refer_line_lane_ids)
            ego_vehicle.current_road_id = result.query_road_id

    if not ego_vehicle.refer_lines:
        raise ValueError("自车参考线为空，无法规划")


def build_planning_actor_state(actor: Any) -> PlanningActorState | None:
    """把一个障碍物样对象规整成规划帧参与者快照。"""

    if actor is None:
        return None

    def read(name: str, default: float = 0.0) -> float:
        value = getattr(actor, name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    ptc_id = str(getattr(actor, "ptc_id", getattr(actor, "ptcId", "")) or "")
    if not ptc_id:
        ptc_id = f"actor_{id(actor)}"

    return PlanningActorState(
        ptc_id=ptc_id,
        x=read("x"),
        y=read("y"),
        heading=read("heading"),
        v=max(0.0, read("v")),
        a=read("a"),
        length=max(0.0, read("length", getattr(actor, "L", 0.0))),
        width=max(0.0, read("width", getattr(actor, "W", 0.0))),
        height=max(0.0, read("height")),
        timestamp_ms=int(getattr(actor, "timestamp_ms", 0) or 0),
        lane_id_hint=str(getattr(actor, "static_lane_id", getattr(actor, "lane_id", "")) or ""),
        raw=actor,
    )


def build_planning_frame(
    ego_vehicle: "GVT",
    provider: "SceneTargetProvider",
    obstacles: list[Any],
) -> PlanningFrame:
    """构造规划系统统一使用的单帧输入。"""

    refresh_ego_refer_lines(ego_vehicle)
    ego_state = VehicleState(
        x=float(ego_vehicle.x),
        y=float(ego_vehicle.y),
        heading=float(ego_vehicle.heading),
        v=float(ego_vehicle.v),
        a=float(ego_vehicle.a),
        kappa=float(ego_vehicle.kappa),
    )
    main_actor = provider.resolve_main_actor(
        ego_state=ego_state,
        obstacles=obstacles,
    )
    return PlanningFrame(
        frame_index=int(provider.step_index),
        time_s=float(provider.time_s),
        dt=float(provider.dt),
        ego_state=ego_state,
        obstacles=list(obstacles),
        refer_lines=[np.asarray(line, dtype=float) for line in ego_vehicle.refer_lines],
        refer_line_lane_ids=list(ego_vehicle.refer_line_lane_ids),
        refer_line_lane_sequences=[list(seq) for seq in ego_vehicle.refer_line_lane_sequences],
        refer_line_lateral_steps=list(ego_vehicle.refer_line_lateral_steps),
        current_road_id=ego_vehicle.current_road_id,
        planned_path=list(ego_vehicle.lanePlanner.planner.planned_path),
        planned_actions=list(ego_vehicle.lanePlanner.planner.planned_actions),
        road_sequence=list(ego_vehicle.lanePlanner.planner.road_sequence),
        goal_xy=np.asarray(ego_vehicle.config["sg_info"]["gp"], dtype=float),
        config=ego_vehicle.config,
        main_actor=main_actor,
    )


def build_planning_input_from_frame(
    planning_frame: PlanningFrame,
    planner_memory: Any,
) -> PlanningInput:
    """从统一规划帧构造局部规划器消费的输入包。"""

    return PlanningInput(
        ego_state=planning_frame.ego_state,
        obstacles=list(planning_frame.obstacles),
        refer_lines=[np.asarray(line, dtype=float) for line in planning_frame.refer_lines],
        refer_line_lane_ids=list(planning_frame.refer_line_lane_ids),
        refer_line_lane_sequences=[list(seq) for seq in planning_frame.refer_line_lane_sequences],
        refer_line_lateral_steps=list(planning_frame.refer_line_lateral_steps),
        current_road_id=planning_frame.current_road_id,
        planned_path=list(planning_frame.planned_path),
        planned_actions=list(planning_frame.planned_actions),
        road_sequence=list(planning_frame.road_sequence),
        goal_xy=np.asarray(planning_frame.goal_xy, dtype=float),
        config=planning_frame.config,
        planner_memory=planner_memory,
    )


def build_planning_context_from_frame(
    planning_frame: PlanningFrame,
    planner_memory: Any,
) -> Any:
    """从统一规划帧直接构造局部规划器内部上下文。"""

    planning_input = build_planning_input_from_frame(
        planning_frame=planning_frame,
        planner_memory=planner_memory,
    )
    return build_planning_context(planning_input)


def build_planning_artifacts(
    ego_vehicle: "GVT",
    provider: "SceneTargetProvider",
    obstacles: list[Any],
) -> tuple[PlanningFrame, PlanningInput, Any]:
    """一次性构造规划帧、PlanningInput 和 PlanningContext。"""

    planning_frame = build_planning_frame(
        ego_vehicle=ego_vehicle,
        provider=provider,
        obstacles=obstacles,
    )
    planning_input = build_planning_input_from_frame(
        planning_frame=planning_frame,
        planner_memory=ego_vehicle.traj_planner.memory,
    )
    planning_context = build_planning_context(planning_input)
    return planning_frame, planning_input, planning_context


def build_planning_input_and_context(
    ego_vehicle: "GVT",
    provider: "SceneTargetProvider",
    obstacles: list[Any],
) -> tuple[PlanningInput, Any]:
    """返回统一规划帧派生出的 `PlanningInput` 与 `PlanningContext`。"""

    _, planning_input, planning_context = build_planning_artifacts(
        ego_vehicle=ego_vehicle,
        provider=provider,
        obstacles=obstacles,
    )
    return planning_input, planning_context


# Compatibility aliases for the current transition period.
RuntimeActorState = PlanningActorState
RuntimeFrame = PlanningFrame
refresh_ego_runtime_refer_lines = refresh_ego_refer_lines
build_runtime_actor_state = build_planning_actor_state
build_runtime_frame = build_planning_frame
build_planning_input_from_runtime_frame = build_planning_input_from_frame
build_runtime_planning_artifacts = build_planning_artifacts
build_runtime_planning_input = build_planning_input_and_context
