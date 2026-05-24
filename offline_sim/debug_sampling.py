"""离线 `local_sampling` 调试辅助逻辑。"""

from __future__ import annotations

from typing import Any

import numpy as np

from planning_runtime.frame import PlanningFrame, build_planning_artifacts
from planning_runtime.scene.provider import SceneTargetProvider
from planning_runtime.scene.types import SceneContext
from traj_planner.types import PlanningInput



def record_local_sampling_result(
    ego_vehicle,
    provider: SceneTargetProvider,
    planning_frame: PlanningFrame,
    planning_input: PlanningInput,
    trajectory: np.ndarray,
) -> None:
    """把 local sampling 单帧结果写入调试上下文与历史。"""

    planning_context = ego_vehicle.traj_planner.last_context
    if planning_context is None:
        planning_context = build_planning_artifacts(
            ego_vehicle=ego_vehicle,
            provider=provider,
            obstacles=list(planning_frame.obstacles),
        )[2]
    local_context = provider.build_scene_context(
        planning_frame=planning_frame,
        planner_mode="local_sampling",
        planning_input=planning_input,
        planning_context=planning_context,
    )
    traj = np.asarray(trajectory, dtype=float)
    provider.history.append(
        {
            "step_index": provider.step_index,
            "time_s": provider.time_s,
            "reason": str(ego_vehicle.traj_planner.attribute.get("reason", "local_sampling")),
            "target_line_index": None,
            "target": {
                "x": float(traj[-1, 0]),
                "y": float(traj[-1, 1]),
                "v": float(traj[-1, 3]),
                "a": float(traj[-1, 4]),
                "theta": float(traj[-1, 2]),
                "t": float(provider.dt * max(len(traj) - 1, 1)),
            },
            "ego_lane_id": local_context.ego_vehicle.lane_id,
            "main_lane_id": "" if local_context.main_vehicle is None else local_context.main_vehicle.lane_id,
            "main_relative_s_to_ego": 0.0 if local_context.main_vehicle is None else local_context.main_vehicle.relative_s_to_ego,
            "main_relative_d_to_ego": 0.0 if local_context.main_vehicle is None else local_context.main_vehicle.relative_d_to_ego,
        }
    )
