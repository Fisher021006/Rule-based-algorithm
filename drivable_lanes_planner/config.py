import numpy as np
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field

@dataclass
class LaneChangeConfig:
    """变道筛选配置类"""
    # 最小变道前进长度（米）：换一次道最少需要前进的距离
    min_change_advance_length: float = 10.0

    # 单次换道后的最小剩余道路长度（米）
    # 即换道后车道长度需要大于此值才能认为换道有效
    single_change_min_remaining: float = 10.0

    # 换道后再换回需要的最小道路长度（米）
    # 即换道后想再换回来，需要的总长度
    return_change_min_length: float = 20.0

    # 允许换道的车道线类型
    changeable_marks: List[str] = field(default_factory=lambda: [
        'broken', 'broken broken', 'broken solid', 'none'
    ])

    def __post_init__(self):
        """配置校验"""
        if self.min_change_advance_length <= 0:
            raise ValueError("最小变道前进长度必须大于0")
        if self.single_change_min_remaining <= 0:
            raise ValueError("单次换道最小剩余长度必须大于0")
        if self.return_change_min_length <= 0:
            raise ValueError("换道返回最小长度必须大于0")

@dataclass
class DrivableLanesResult:
    """可行驶车道结果数据类"""
    # 最短路径的路段ID序列
    road_sequence: List[str]

    # 每个路段的所有车道ID
    all_lanes_per_road: Dict[str, List[str]]

    # 筛选后每个路段的可行驶车道ID
    filtered_drivable_lanes: Dict[str, Set[str]]

    # 查询坐标定位的路段ID
    query_road_id: Optional[str] = None

    # 查询坐标所在路段与可行驶车道的交集
    intersection_lanes: List[str] = field(default_factory=list)

    # 从交集车道向后延伸的所有可行驶车道中心线坐标
    # 格式: List[List[List[float]]] - [车道数][点数][x,y]
    extended_centerlines: [List[List[List[float]]]] = field(default_factory=list)

    # 延伸车道的ID列表（与extended_centerlines一一对应）
    extended_lane_ids: List[List[str]] = field(default_factory=list)

    # 每条延伸线第一段 lane 的剩余长度（与extended_centerlines一一对应）
    first_lane_remaining_lengths: List[float] = field(default_factory=list)

