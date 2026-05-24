"""`scene_target` 模式的输入准备逻辑。"""

from __future__ import annotations

from planning_runtime.frame import PlanningFrame, build_planning_input_from_frame
from planning_runtime.scene.provider import SceneTargetProvider
from traj_planner.types import PlanningInput


def prepare_scene_target_input(
    planning_frame: PlanningFrame,
    provider: SceneTargetProvider,
    planner_memory,
) -> PlanningInput:
    """从统一规划帧派生带 `external_target` 的 PlanningInput。"""

    planning_input = build_planning_input_from_frame(
        planning_frame=planning_frame,
        planner_memory=planner_memory,
    )
    external_target = provider.generate_external_target(
        planning_frame=planning_frame,
        planning_input=planning_input,
    )
    planning_input.external_target = external_target
    return planning_input
