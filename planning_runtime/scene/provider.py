"""场景上下文构造与目标点适配流水线。
PlanningFrame
-> SceneContext
-> scene_algorithm
-> SceneAlgorithmOutput
-> ExternalTarget
"""

from __future__ import annotations

import importlib
import copy
import time
from typing import Any

import numpy as np

from drivable_lanes_planner.planner import DrivableLanesExtender
from planning_runtime.frame import PlanningActorState, PlanningFrame, build_planning_actor_state
from planning_runtime.scene.map_views import (
    build_reference_views_from_extender_result,
    build_reference_views_from_lane_sequences,
    build_reference_views_from_frame,
    extend_lane_forward,
)
from planning_runtime.scene.types import (
    SceneAlgorithmOutput,
    SceneContext,
    SceneVehicleState,
)
from traj_planner.config import VehicleState
from traj_planner.tools import calculate_path_cumulative_distance, get_path_point_at_s, project_point_to_path
from traj_planner.types import ExternalTarget, PlanningInput
from xodr_parser.xodr_parser import XODRParser


MAIN_ACTOR_EGO_CORRIDOR_MATCH_MAX_D = 4.5
MAIN_ACTOR_GLOBAL_LANE_MATCH_MAX_D = 5.0
DEFAULT_SCENE_ALGORITHM_SYMBOL = "scenarios.scene_algorithm.scene_algorithm"


def load_scene_algorithm(symbol_path: str):
    """按 `<module>.<symbol>` 动态加载场景算法入口。"""

    normalized = str(symbol_path or "").strip()
    if not normalized:
        raise ValueError("scene algorithm symbol path cannot be empty")
    if "." not in normalized:
        raise ValueError(
            f"invalid scene algorithm symbol path: {normalized}. "
            "expected format '<module>.<symbol>'"
        )

    module_name, symbol_name = normalized.rsplit(".", 1)
    module = importlib.import_module(module_name)
    algorithm = getattr(module, symbol_name)
    if not callable(algorithm):
        raise TypeError(f"scene algorithm symbol is not callable: {normalized}")
    return algorithm


def normalize_scene_algorithm_output(
    raw_output: Any,
    ego_vehicle: SceneVehicleState,
) -> SceneAlgorithmOutput:
    """把场景算法输出统一规整成 `SceneAlgorithmOutput`。

    支持的用户返回形式：
        - ``SceneAlgorithmOutput``
        - ``dict``，至少包含 ``x, y, v``，其余字段可缺省
        - 6 元组 ``(x, y, v, a, theta, t)``

    额外元信息如 ``target_line_index`` / ``reason`` 在缺省时由框架补齐。
    """

    if isinstance(raw_output, SceneAlgorithmOutput):
        output = raw_output
    elif isinstance(raw_output, dict):
        output = SceneAlgorithmOutput(
            x=float(raw_output["x"]),
            y=float(raw_output["y"]),
            v=float(raw_output["v"]),
            a=float(raw_output.get("a", 0.0)),
            theta=float(raw_output.get("theta", raw_output.get("heading", 0.0))),
            t=float(raw_output.get("t", raw_output.get("plan_t", 2.0))),
            target_line_index=raw_output.get("target_line_index"),
            hold_t=float(raw_output.get("hold_t", 0.0)),
            reason=str(raw_output.get("reason", "scene_algorithm")),
        )
    else:
        values = list(raw_output)
        if len(values) != 6:
            raise ValueError("scene_algorithm 必须返回 6 元组或 SceneAlgorithmOutput")
        output = SceneAlgorithmOutput(
            x=float(values[0]),
            y=float(values[1]),
            v=float(values[2]),
            a=float(values[3]),
            theta=float(values[4]),
            t=float(values[5]),
        )

    if output.target_line_index is None:
        target_point = np.array([output.x, output.y], dtype=float)
        best_line_index = 0
        best_distance = float("inf")
        for line in ego_vehicle.candidate_ref_lines:
            distances = np.linalg.norm(line.line[:, :2] - target_point, axis=1)
            distance = float(np.min(distances))
            if distance < best_distance:
                best_distance = distance
                best_line_index = line.line_index
        output.target_line_index = best_line_index
    return output


def scene_output_to_external_target(output: SceneAlgorithmOutput) -> ExternalTarget:
    """把场景输出转换成规划器消费的 `ExternalTarget`。"""

    return ExternalTarget(
        point_xy=np.array([output.x, output.y], dtype=float),
        target_line_index=int(0 if output.target_line_index is None else output.target_line_index),
        target_v=float(output.v),
        target_a=float(output.a),
        target_heading=float(output.theta),
        plan_T=max(0.5, float(output.t)),
        hold_t=max(0.0, float(output.hold_t)),
        source="scene_algorithm",
        metadata={"reason": output.reason},
    )


class SceneTargetProvider:
    """构造仿真器统一外层视图上下文，并把场景输出转换为外部目标点。"""

    def __init__(
        self,
        xodr_parser: XODRParser,
        main_vehicle: Any | None,
        main_lane_extender: DrivableLanesExtender | None,
        dt: float,
        max_lane_segments: int,
        scene_algorithm_symbol: str = DEFAULT_SCENE_ALGORITHM_SYMBOL,
    ) -> None:
        """保存统一视图上下文构造所需的长期配置与运行期缓存。

        这里保存的不是单帧输入，而是跨帧复用的环境对象：
        - 地图解析器
        - 仿真主车对象或其回退来源
        - 主车参考线延伸器
        - 当前规划周期、日志缓存等
        """

        self.parser = xodr_parser
        self.main_vehicle = main_vehicle
        self.main_lane_extender = main_lane_extender
        self.dt = float(dt)
        self.max_lane_segments = int(max_lane_segments)
        self.step_index = 0
        self.time_s = 0.0
        self.main_vehicle_ptc_id = ""
        self.scene_algorithm_symbol = ""
        self.scene_algorithm = None
        self.history: list[dict[str, Any]] = []
        self.last_scene_context: SceneContext | None = None
        self.last_scene_algorithm_timing_ms: float = 0.0
        self._main_lane_locator: DrivableLanesExtender | None = None
        self.set_scene_algorithm_symbol(scene_algorithm_symbol)

    def set_runtime(self, step_index: int, time_s: float) -> None:
        """更新当前仿真步编号和仿真时间。"""

        self.step_index = int(step_index)
        self.time_s = float(time_s)

    def set_main_vehicle_ptc_id(self, ptc_id: str) -> None:
        """指定场景算法优先关注的主车 `ptc_id`。"""

        self.main_vehicle_ptc_id = str(ptc_id or "").strip()

    def set_runtime_dt(self, dt: float) -> None:
        """更新当前运行时周期，用于写入统一视图上下文。"""

        self.dt = float(dt)

    def set_scene_algorithm_symbol(self, symbol_path: str) -> None:
        """切换当前 provider 使用的场景算法入口。"""

        normalized = str(symbol_path or "").strip() or DEFAULT_SCENE_ALGORITHM_SYMBOL
        self.scene_algorithm_symbol = normalized
        self.scene_algorithm = load_scene_algorithm(normalized)

    def resolve_main_actor(
        self,
        ego_state: VehicleState,
        obstacles: list[Any],
    ) -> PlanningActorState | None:
        """从原始障碍物中挑选主关注车，并保留仿真回退逻辑。"""

        # 先把原始障碍物统一规整成可比较的参与者快照。
        candidates = [
            actor_state
            for obstacle in obstacles
            for actor_state in [build_planning_actor_state(obstacle)]
            if actor_state is not None
        ]
        # 若用户显式指定了 ptc_id，则优先锁定该目标。
        if self.main_vehicle_ptc_id:
            for actor_state in candidates:
                if actor_state.ptc_id == self.main_vehicle_ptc_id:
                    return actor_state

        # 未指定时，默认选离自车最近的障碍物作为主关注车。
        if candidates:
            ego_xy = np.array([float(ego_state.x), float(ego_state.y)], dtype=float)
            return min(
                candidates,
                key=lambda actor_state: float(
                    np.linalg.norm(np.array([actor_state.x, actor_state.y], dtype=float) - ego_xy)
                ),
            )

        # 若这一帧没有任何障碍物，则回退到仿真器内部的主车对象。
        if self.main_vehicle is not None:
            return build_planning_actor_state(self.main_vehicle)
        return None

    def _get_main_lane_locator(self) -> DrivableLanesExtender:
        """返回用于车端 OBU 主车全局匹配的轻量车道定位器。"""

        if self._main_lane_locator is None:
            self._main_lane_locator = DrivableLanesExtender(self.parser)
        return self._main_lane_locator

    @staticmethod
    def _pick_current_view(views: list[Any]) -> Any:
        """从候选参考线中挑选与查询点最匹配的一条。"""

        return min(
            views,
            key=lambda view: (
                abs(float(view.current_d)),
                abs(float(view.current_s)),
                int(abs(getattr(view, "lateral_step", 0))),
            ),
        )

    def _project_to_full_lane(
        self,
        lane_id: str,
        x: float,
        y: float,
    ) -> tuple[float, float, float, np.ndarray | None]:
        """把坐标投影到完整 lane centerline，供场景关系判断使用。"""

        centerline = np.asarray(self.parser.lane_centerlines.get(str(lane_id), []), dtype=float)
        if centerline.ndim != 2 or len(centerline) < 2:
            return 0.0, 0.0, 0.0, None

        line_xy = np.asarray(centerline[:, :2], dtype=float)
        _, lane_s, lane_d = project_point_to_path(np.array([x, y], dtype=float), line_xy)
        cumulative_s = calculate_path_cumulative_distance(line_xy)
        _, lane_heading = get_path_point_at_s(line_xy, lane_s, cumulative_s)
        return float(lane_s), float(lane_d), float(lane_heading), line_xy

    def _build_main_views_from_ego_corridor(
        self,
        planning_frame: PlanningFrame,
        main_actor: PlanningActorState,
    ) -> list[Any] | None:
        """优先用 ego 当前走廊语义来解释 OBU 主车。"""

        if not planning_frame.refer_line_lane_sequences:
            return None
        try:
            views = build_reference_views_from_lane_sequences(
                parser=self.parser,
                lane_sequences=[list(seq) for seq in planning_frame.refer_line_lane_sequences],
                query_point=np.array([main_actor.x, main_actor.y], dtype=float),
                max_lane_segments=self.max_lane_segments,
                lateral_steps=list(planning_frame.refer_line_lateral_steps),
            )
        except ValueError:
            return None

        current_view = self._pick_current_view(views)
        if abs(float(current_view.current_d)) > MAIN_ACTOR_EGO_CORRIDOR_MATCH_MAX_D:
            return None
        return views

    def _build_main_views_from_global_lane(
        self,
        main_actor: PlanningActorState,
    ) -> list[Any]:
        """在 ego 走廊外时，退回到全局最近车道语义。"""

        lane_id_hint = str(main_actor.lane_id_hint or "")
        if not lane_id_hint:
            lane_id_hint, _, _dist = self._get_main_lane_locator().locate_lane_by_coordinate(
                float(main_actor.x),
                float(main_actor.y),
                max_distance=MAIN_ACTOR_GLOBAL_LANE_MATCH_MAX_D,
            )
            if lane_id_hint is None:
                raise ValueError(
                    f"主车无法匹配到全局车道，距离最近 driving lane 超过 {MAIN_ACTOR_GLOBAL_LANE_MATCH_MAX_D:.1f}m"
                )
        lane_sequence = extend_lane_forward(self.parser, str(lane_id_hint), self.max_lane_segments)
        return build_reference_views_from_extender_result(
            parser=self.parser,
            extended_lane_ids=[lane_sequence],
            query_point=np.array([main_actor.x, main_actor.y], dtype=float),
            max_lane_segments=self.max_lane_segments,
        )

    def _build_main_vehicle_state(
        self,
        planning_frame: PlanningFrame,
        ego_vehicle: SceneVehicleState,
    ) -> SceneVehicleState | None:
        """根据 `planning_frame.main_actor` 构造统一视图中的主车状态。"""

        main_actor = planning_frame.main_actor
        if main_actor is None:
            return None

        # 动态主车场景：根据主车当前位置实时生成候选参考线。
        if self.main_lane_extender is not None:
            result = self.main_lane_extender.plan_extended_centerlines(main_actor.x, main_actor.y)
            if not result.extended_lane_ids:
                raise ValueError("主车当前无法生成参考线")
            main_views = build_reference_views_from_extender_result(
                parser=self.parser,
                extended_lane_ids=result.extended_lane_ids,
                query_point=np.array([main_actor.x, main_actor.y], dtype=float),
                max_lane_segments=self.max_lane_segments,
            )
        else:
            # 车端 OBU 场景：先在 ego 当前走廊里解释主车，不可靠时再全局匹配。
            main_views = self._build_main_views_from_ego_corridor(planning_frame, main_actor)
            if main_views is None:
                main_views = self._build_main_views_from_global_lane(main_actor)
        main_current = self._pick_current_view(main_views)
        main_lane_s, main_lane_d, main_lane_heading, main_lane_centerline = self._project_to_full_lane(
            str(main_current.lane_id),
            float(main_actor.x),
            float(main_actor.y),
        )
        if main_lane_centerline is None:
            main_lane_s = float(main_current.current_s)
            main_lane_d = float(main_current.current_d)
            main_lane_heading = float(main_actor.heading)
        # 主车相对自车的 s / d 关系统一投影到自车当前主参考线上。
        _, main_s_on_ego, main_d_on_ego = project_point_to_path(
            np.array([main_actor.x, main_actor.y], dtype=float),
            ego_vehicle.current_ref_line.line,
        )
        return SceneVehicleState(
            role="main",
            x=float(main_actor.x),
            y=float(main_actor.y),
            heading=float(main_actor.heading),
            v=float(main_actor.v),
            a=float(main_actor.a),
            t=float(planning_frame.time_s),
            lane_id=str(main_current.lane_id),
            road_id=str(main_current.lane_id).split(".")[0],
            s=float(main_current.current_s),
            d=float(main_current.current_d),
            current_ref_line=main_current,
            candidate_ref_lines=main_views,
            relative_x_to_ego=float(main_actor.x - ego_vehicle.x),
            relative_y_to_ego=float(main_actor.y - ego_vehicle.y),
            relative_s_to_ego=float(main_s_on_ego - ego_vehicle.current_ref_line.current_s),
            relative_d_to_ego=float(main_d_on_ego),
            lane_s=float(main_lane_s),
            lane_d=float(main_lane_d),
            lane_heading=float(main_lane_heading),
            lane_centerline=main_lane_centerline,
        )

    def _build_ego_vehicle_state(self, planning_frame: PlanningFrame) -> SceneVehicleState:
        """根据原始运行时帧构造统一视图中的自车状态。"""

        ego_point = np.array([planning_frame.ego_state.x, planning_frame.ego_state.y], dtype=float)
        ego_views = build_reference_views_from_frame(
            planning_frame=planning_frame,
            query_point=ego_point,
        )
        current_view = ego_views[0]
        ego_lane_s, ego_lane_d, ego_lane_heading, ego_lane_centerline = self._project_to_full_lane(
            str(current_view.lane_id),
            float(planning_frame.ego_state.x),
            float(planning_frame.ego_state.y),
        )
        if ego_lane_centerline is None:
            ego_lane_s = float(current_view.current_s)
            ego_lane_d = float(current_view.current_d)
            ego_lane_heading = float(planning_frame.ego_state.heading)
        return SceneVehicleState(
            role="ego",
            x=float(planning_frame.ego_state.x),
            y=float(planning_frame.ego_state.y),
            heading=float(planning_frame.ego_state.heading),
            v=float(planning_frame.ego_state.v),
            a=float(planning_frame.ego_state.a),
            t=float(planning_frame.time_s),
            lane_id=str(current_view.lane_id),
            road_id=str(current_view.lane_id).split(".")[0],
            s=float(current_view.current_s),
            d=float(current_view.current_d),
            current_ref_line=current_view,
            candidate_ref_lines=ego_views,
            lane_s=float(ego_lane_s),
            lane_d=float(ego_lane_d),
            lane_heading=float(ego_lane_heading),
            lane_centerline=ego_lane_centerline,
        )

    @staticmethod
    def _normalize_algorithm_output(
        raw_output: Any,
        ego_vehicle: SceneVehicleState,
    ) -> SceneAlgorithmOutput:
        """把用户算法输出转成框架内部统一结构。"""

        return normalize_scene_algorithm_output(raw_output, ego_vehicle)

    def build_scene_context(
        self,
        planning_frame: PlanningFrame,
        planner_mode: str,
        planning_input: PlanningInput | None = None,
        planning_context: Any | None = None,
    ) -> SceneContext:
        """构造一帧完整的 `SceneContext`。

        这一步只负责把同一帧原始输入整理成仿真器统一外层视图：
        - `scene_target` 模式下，后续会直接传给场景算法
        - `local_sampling` 模式下，主要用于调试、可视化和历史记录

        它本身不执行场景算法，也不是局部规划器内部的 `PlanningContext`。
        """

        ego_vehicle = self._build_ego_vehicle_state(planning_frame)
        main_vehicle = self._build_main_vehicle_state(planning_frame, ego_vehicle)
        scene_context = SceneContext(
            frame_index=int(planning_frame.frame_index),
            time_s=float(planning_frame.time_s),
            dt=float(planning_frame.dt),
            ego_vehicle=ego_vehicle,
            main_vehicle=main_vehicle,
            planner_mode=str(planner_mode),
            planning_frame=planning_frame,
            planning_input=planning_input,
            planning_context=planning_context,
        )
        self.last_scene_context = scene_context
        return scene_context

    def generate_external_target(
        self,
        planning_frame: PlanningFrame,
        planning_input: PlanningInput | None = None,
    ) -> ExternalTarget:
        """执行场景算法，并把输出转换成规划器消费的外部目标点。

        数据流：
            ``PlanningFrame -> SceneContext -> scene_algorithm
            -> SceneAlgorithmOutput -> ExternalTarget``
        """

        scene_context = self.build_scene_context(
            planning_frame=planning_frame,
            planner_mode="scene_target",
            planning_input=planning_input,
            planning_context=None,
        )
        ego_vehicle = scene_context.ego_vehicle
        main_vehicle = scene_context.main_vehicle

        # 用户算法只需要返回 6 个主体参数，框架在这里补齐元信息。
        scene_start = time.perf_counter()
        raw_output = self.scene_algorithm(scene_context)
        self.last_scene_algorithm_timing_ms = (time.perf_counter() - scene_start) * 1000.0
        output = self._normalize_algorithm_output(raw_output, ego_vehicle)
        scene_metadata = raw_output if isinstance(raw_output, dict) else {}

        # 保留逐帧目标与主车关系，供可视化和离线排查使用。
        self.history.append(
            {
                "step_index": self.step_index,
                "time_s": self.time_s,
                "reason": output.reason,
                "target_line_index": output.target_line_index,
                "target": {
                    "x": output.x,
                    "y": output.y,
                    "v": output.v,
                    "a": output.a,
                    "theta": output.theta,
                    "t": output.t,
                },
                "ego_lane_id": ego_vehicle.lane_id,
                "main_lane_id": "" if main_vehicle is None else main_vehicle.lane_id,
                "main_relative_s_to_ego": 0.0 if main_vehicle is None else main_vehicle.relative_s_to_ego,
                "main_relative_d_to_ego": 0.0 if main_vehicle is None else main_vehicle.relative_d_to_ego,
                "scene_status": str(scene_metadata.get("scene_status", "")),
                "scene_detail": str(scene_metadata.get("scene_detail", "")),
                "scene_debug": copy.deepcopy(scene_metadata.get("scene_debug", {})),
                "scene_algorithm_symbol": self.scene_algorithm_symbol,
                "scene_algorithm_time_ms": round(float(self.last_scene_algorithm_timing_ms), 3),
            }
        )

        return scene_output_to_external_target(output)

    # Compatibility alias for the current transition period.
    build_simulation_view_context = build_scene_context
