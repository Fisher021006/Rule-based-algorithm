"""`local_sampling` 模式的输入准备逻辑。"""

from __future__ import annotations

from planning_runtime.frame import PlanningFrame, build_planning_input_from_frame
from traj_planner.types import PlanningInput


def prepare_local_sampling_input(
    planning_frame: PlanningFrame,
    planner_memory,
) -> PlanningInput:
    """从统一规划帧派生默认 local sampling 的 PlanningInput。"""

    return build_planning_input_from_frame(
        planning_frame=planning_frame,
        planner_memory=planner_memory,
    )
