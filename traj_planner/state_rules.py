"""规划状态机相关的共享判定规则。"""


def lane_change_completion_threshold(config: dict) -> float:
    """返回换道真正完成所需的横向误差阈值。"""

    return float(config.get("local_fsm", {}).get("enter_complete_lateral_thresh", 0.5))


def is_lane_change_complete(current_lane_id: str, target_lane_id: str, current_d: float, config: dict) -> bool:
    """统一判断一次换道是否已经完成。"""

    if not current_lane_id or not target_lane_id:
        return False
    return current_lane_id == target_lane_id and abs(float(current_d)) < lane_change_completion_threshold(config)


def is_lane_change_centering_incomplete(
    current_lane_id: str,
    target_lane_id: str,
    current_d: float,
    config: dict,
) -> bool:
    """判断是否已切到目标车道但仍未完成横向收敛。"""

    if not current_lane_id or not target_lane_id:
        return False
    return current_lane_id == target_lane_id and not is_lane_change_complete(
        current_lane_id=current_lane_id,
        target_lane_id=target_lane_id,
        current_d=current_d,
        config=config,
    )


def arrival_completion_distance(config: dict) -> float:
    """返回宣布到达所需的距离阈值。"""

    return float(config.get("local_fsm", {}).get("arrive_complete_dist", 1.5))


def arrival_completion_speed(config: dict) -> float:
    """返回宣布到达所需的速度阈值。"""

    return float(config.get("local_fsm", {}).get("arrive_complete_speed", 0.3))


def is_goal_reached(distance_to_goal: float, current_speed: float, config: dict) -> bool:
    """统一判断是否满足宣布到达的条件。"""

    return (
        float(distance_to_goal) <= arrival_completion_distance(config)
        and float(current_speed) <= arrival_completion_speed(config)
    )
