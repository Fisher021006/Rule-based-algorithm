import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xodr_parser import XODRParser


DEFAULT_TRAJECTORY_ROOT = PROJECT_ROOT / "outputs" / "planned_trajectories"
DEFAULT_MAP_CANDIDATES = [
    PROJECT_ROOT / "xodr_parser" / "cache" / "农安测试场.pkl",
    PROJECT_ROOT / "xodr_parser" / "cache" / "通州测试场.pkl",
    PROJECT_ROOT / "xodr_parser" / "data" / "农安测试场.xodr",
    PROJECT_ROOT / "xodr_parser" / "data" / "国汽智联通州测试场map终版.xodr",
]


def resolve_default_map_path():
    for candidate in DEFAULT_MAP_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到默认地图，请通过 --map 显式传入 .pkl 或 .xodr 文件")


def resolve_latest_trajectory_file(path):
    candidate_path = Path(path)
    if not candidate_path.exists():
        raise FileNotFoundError(f"轨迹文件或目录不存在: {candidate_path}")

    if candidate_path.is_file():
        return candidate_path

    supported_files = list(candidate_path.rglob("*.json")) + list(candidate_path.rglob("*.npy"))
    supported_files = [file_path for file_path in supported_files if file_path.is_file()]
    if not supported_files:
        raise FileNotFoundError(f"目录下未找到轨迹文件(.json/.npy): {candidate_path}")
    return max(supported_files, key=lambda file_path: file_path.stat().st_mtime)


def load_trajectory_file(file_path):
    if file_path.suffix.lower() == ".npy":
        trajectory = np.load(file_path)
    else:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("trajectory", data.get("points", data))
        trajectory = np.asarray(data, dtype=float)

    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError(f"轨迹文件格式错误，期望二维数组，实际为: {trajectory.shape}")
    return trajectory


def load_map(map_path):
    if map_path.suffix.lower() == ".pkl":
        return XODRParser.load_cache(str(map_path))
    if map_path.suffix.lower() == ".xodr":
        return XODRParser(str(map_path))
    raise ValueError(f"不支持的地图格式: {map_path}")


def plot_centerlines(ax, xodr_parser, full_map):
    centerlines = xodr_parser.lane_centerlines or xodr_parser.road_centerlines
    line_width = 0.45 if full_map else 0.7
    alpha = 0.55 if full_map else 0.75

    for centerline in centerlines.values():
        if len(centerline) < 2:
            continue
        ax.plot(
            centerline[:, 0],
            centerline[:, 1],
            color="#b8b8b8",
            linewidth=line_width,
            alpha=alpha,
            zorder=1,
        )


def plot_trajectory(ax, trajectory):
    xy = trajectory[:, :2]
    speed = trajectory[:, 3] if trajectory.shape[1] >= 4 else None

    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color="#e36a1a",
        linewidth=2.2,
        alpha=0.95,
        zorder=3,
    )

    if speed is not None:
        scatter = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=speed,
            cmap="viridis",
            s=18,
            zorder=4,
        )
        return scatter

    ax.scatter(xy[:, 0], xy[:, 1], color="#e36a1a", s=18, zorder=4)
    return None


def apply_view_window(ax, xodr_parser, trajectory, full_map):
    if full_map:
        bounds = xodr_parser.get_map_bounds()
        if bounds is not None:
            ax.set_xlim(bounds["x_min"], bounds["x_max"])
            ax.set_ylim(bounds["y_min"], bounds["y_max"])
        return

    xy = trajectory[:, :2]
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    padding = max(20.0, 0.15 * max(x_max - x_min, y_max - y_min, 1.0))
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)


def main():
    parser = argparse.ArgumentParser(description="在 XODR 地图上可视化已保存的轨迹点集")
    parser.add_argument(
        "trajectory",
        nargs="?",
        default=str(DEFAULT_TRAJECTORY_ROOT),
        help="轨迹文件(.json/.npy)或轨迹目录；如果传目录则默认读取最新文件",
    )
    parser.add_argument(
        "--map",
        dest="map_path",
        default=str(resolve_default_map_path()),
        help="XODR 地图文件(.xodr)或缓存文件(.pkl)",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        help="输出图片路径，默认与轨迹文件同目录同名 .png",
    )
    parser.add_argument(
        "--full-map",
        action="store_true",
        help="绘制整张地图；默认自动缩放到轨迹附近区域",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="自定义图标题",
    )
    args = parser.parse_args()

    trajectory_path = resolve_latest_trajectory_file(args.trajectory)
    map_path = Path(args.map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"地图文件不存在: {map_path}")

    trajectory = load_trajectory_file(trajectory_path)
    xodr_parser = load_map(map_path)

    output_path = Path(args.output_path) if args.output_path else trajectory_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    plot_centerlines(ax, xodr_parser, args.full_map)
    scatter = plot_trajectory(ax, trajectory)

    start_point = trajectory[0, :2]
    end_point = trajectory[-1, :2]
    ax.scatter(start_point[0], start_point[1], color="#1f9d55", s=60, label="start", zorder=5)
    ax.scatter(end_point[0], end_point[1], color="#d64545", s=60, label="end", zorder=5)

    apply_view_window(ax, xodr_parser, trajectory, args.full_map)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(args.title or f"Trajectory Overlay: {trajectory_path.name}")
    ax.legend(loc="best")

    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
        colorbar.set_label("speed (m/s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"轨迹文件: {trajectory_path}")
    print(f"地图文件: {map_path}")
    print(f"输出图片: {output_path.resolve()}")


if __name__ == "__main__":
    main()
