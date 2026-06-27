"""坐标系与航向角转换工具。

平台任务和车辆状态使用经纬度与平台航向角，算法内部使用 OpenDRIVE
投影平面坐标和数学坐标系航向角。Adapter、mock 平台和 fake 车辆都会
用到这些转换，因此集中放在这里，避免每个入口各写一套坐标逻辑。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

from utils.tools import clean_proj_string
from xodr_parser.xodr_parser import XODRParser


DEFAULT_MAP_PATH = Path(__file__).resolve().parents[3] / "xodr_parser" / "data" / "同济测试场.xodr"


def resolve_cache_path(map_path: Path, cache_path: str) -> Path:
    """根据地图路径和可选缓存路径确定 XODR 解析缓存位置。

    如果调用方显式传入缓存路径，就完全使用该路径；否则默认使用项目
    里的 `xodr_parser/cache/<地图名>.pkl`，与原算法的地图缓存习惯保持一致。
    """

    if cache_path:
        return Path(cache_path)
    return map_path.parent.parent / "cache" / f"{map_path.stem}.pkl"


def load_or_build_map(map_path_str: str, cache_path_str: str = "", rebuild_cache: bool = False) -> XODRParser:
    """加载或重新构建 OpenDRIVE 地图解析结果。

    Adapter 启动时需要地图投影参数和车道中心线。如果缓存可用就直接读缓存；
    如果缓存版本不兼容、文件不存在，或要求重建缓存，就重新解析 `.xodr`
    并保存缓存。
    """

    map_path = Path(map_path_str)
    cache_path = resolve_cache_path(map_path, cache_path_str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not rebuild_cache:
        try:
            return XODRParser.load_cache(str(cache_path))
        except (ModuleNotFoundError, AttributeError, ValueError, ImportError):
            pass
    parser = XODRParser(str(map_path))
    parser.save_cache(str(cache_path))
    return parser


class MapProjector:
    """OpenDRIVE 投影坐标与 WGS84 经纬度之间的双向转换器。

    平台下发起终点、背景车和回传轨迹都使用经纬度；规划算法和地图车道线
    使用 XODR 内部投影平面坐标。该类封装 pyproj 转换器，并提供最近车道
    航向角查询。
    """

    def __init__(self, xodr_parser: XODRParser) -> None:
        """从 XODR 的 `geo_reference` 初始化坐标转换器。"""

        self.xodr_parser = xodr_parser
        proj_crs = CRS.from_proj4(clean_proj_string(xodr_parser.geo_reference["proj"]))
        wgs84_crs = CRS.from_epsg(4326)
        self.to_xy = Transformer.from_crs(wgs84_crs, proj_crs, always_xy=True)
        self.to_latlon_transformer = Transformer.from_crs(proj_crs, wgs84_crs, always_xy=True)

    def latlon_to_xy(self, latitude: float, longitude: float) -> tuple[float, float]:
        """把平台经纬度转换为算法内部使用的平面坐标 `(x, y)`。"""

        x, y = self.to_xy.transform(float(longitude), float(latitude))
        return float(x), float(y)

    def xy_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        """把算法内部平面坐标 `(x, y)` 转换为平台需要的 `(lat, lon)`。"""

        lon, lat = self.to_latlon_transformer.transform(float(x), float(y))
        return float(lat), float(lon)

    def nearest_lane_heading(self, x: float, y: float) -> tuple[float | None, float]:
        """查找离给定点最近的车道中心线航向角和距离。

        返回的航向角是算法内部坐标系角度，单位为弧度；距离单位为米。
        该函数主要用于平台起点 `courseAngle` 不可靠时，用地图车道方向修正
        虚拟车初始朝向。
        """

        best_line = None
        best_idx = 0
        best_dist = float("inf")
        point = np.asarray([float(x), float(y)], dtype=float)
        for line in self.xodr_parser.lane_centerlines.values():
            arr = np.asarray(line, dtype=float)
            if arr.ndim != 2 or len(arr) < 2:
                continue
            distances = np.linalg.norm(arr[:, :2] - point, axis=1)
            idx = int(np.argmin(distances))
            dist = float(distances[idx])
            if dist < best_dist:
                best_line = arr
                best_idx = idx
                best_dist = dist
        if best_line is None:
            return None, float("inf")
        idx = min(best_idx, len(best_line) - 2)
        vec = best_line[idx + 1, :2] - best_line[idx, :2]
        return float(math.atan2(float(vec[1]), float(vec[0]))), best_dist


def platform_heading_deg_to_internal_rad(heading_deg: float) -> float:
    """把平台航向角转换为算法内部航向角。

    平台定义为正北 0 度、顺时针增加；算法内部是数学坐标系，+x 方向为
    0 弧度、逆时针为正。
    """

    return math.radians(90.0 - float(heading_deg))


def internal_rad_to_platform_heading_deg(theta_rad: float) -> float:
    """把算法内部航向角转换为平台航向角。

    输出范围归一化到 `[0, 360)`，用于回传车辆 `courseAngle` 或构造 mock
    平台任务。
    """

    return (90.0 - math.degrees(float(theta_rad))) % 360.0


def heading_diff_deg(a_deg: float, b_deg: float) -> float:
    """计算两个平台航向角之间的最小夹角，单位为度。"""

    return abs((float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0)
