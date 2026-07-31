"""多车离线仿真的通用可视化。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# 先导入 render，由它在 pyplot 导入前完成后端选择。
from offline_sim.render import _is_interactive_backend, draw_vehicle

import matplotlib.pyplot as plt

from xodr_parser.xodr_parser import XODRParser


VEHICLE_COLORS = (
    "#D62728",
    "#1F77B4",
    "#2CA02C",
    "#9467BD",
    "#FF7F0E",
    "#17BECF",
    "#8C564B",
    "#E377C2",
)


def _trace_array(trace: list[list[float]] | np.ndarray | None) -> np.ndarray | None:
    """将轨迹转成非空二维数组。"""

    if trace is None:
        return None
    array = np.asarray(trace, dtype=float)
    if array.ndim != 2 or len(array) == 0:
        return None
    return array


def render_multi_vehicle_frame(
    *,
    xodr_parser: XODRParser,
    traces: dict[str, list[list[float]] | np.ndarray],
    vehicle_metadata: dict[str, dict[str, Any]],
    frame_index: int,
    planned_trajectories: dict[str, list[list[float]] | np.ndarray] | None = None,
    status_text: str = "",
) -> Image.Image:
    """渲染多车仿真的某一帧。"""

    fig, ax = plt.subplots(figsize=(9, 9))
    for lane in xodr_parser.lane_centerlines.values():
        lane_array = np.asarray(lane, dtype=float)
        if len(lane_array) >= 2:
            ax.plot(lane_array[:, 0], lane_array[:, 1], color="#D0D6DB", linewidth=0.6)

    all_xy: list[np.ndarray] = []
    for index, (vehicle_id, trace) in enumerate(traces.items()):
        trace_array = _trace_array(trace)
        if trace_array is None:
            continue
        end = min(max(int(frame_index) + 1, 1), len(trace_array))
        visible = trace_array[:end]
        color = VEHICLE_COLORS[index % len(VEHICLE_COLORS)]
        metadata = vehicle_metadata.get(vehicle_id, {})
        kind = str(metadata.get("kind", "route"))
        label = f"{vehicle_id} ({kind})"
        ax.plot(visible[:, 0], visible[:, 1], color=color, linewidth=2.0, label=label)
        ax.scatter(visible[0, 0], visible[0, 1], color=color, s=28, marker="o")
        goal_xy = metadata.get("goal_xy")
        if goal_xy is not None:
            goal = np.asarray(goal_xy, dtype=float)
            ax.scatter(goal[0], goal[1], color=color, s=42, marker="*")
        length = float(metadata.get("length", 4.6))
        width = float(metadata.get("width", 1.8))
        draw_vehicle(
            ax,
            x=float(visible[-1, 0]),
            y=float(visible[-1, 1]),
            heading=float(visible[-1, 2]),
            length=length,
            width=width,
            color=color,
            label=vehicle_id,
        )
        all_xy.append(trace_array[:, :2])

    for index, (vehicle_id, trajectory) in enumerate((planned_trajectories or {}).items()):
        trajectory_array = _trace_array(trajectory)
        if trajectory_array is None or len(trajectory_array) < 2:
            continue
        color = VEHICLE_COLORS[index % len(VEHICLE_COLORS)]
        ax.plot(
            trajectory_array[:, 0],
            trajectory_array[:, 1],
            color=color,
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            label=f"{vehicle_id} plan",
        )

    if not all_xy:
        plt.close(fig)
        raise ValueError("多车仿真轨迹为空")
    bounds = np.vstack(all_xy)
    min_xy = np.min(bounds, axis=0) - 15.0
    max_xy = np.max(bounds, axis=0) + 15.0
    ax.set_xlim(float(min_xy[0]), float(max_xy[0]))
    ax.set_ylim(float(min_xy[1]), float(max_xy[1]))
    ax.set_aspect("equal")
    title = "Offline Multi-Vehicle Scene Simulation"
    if status_text:
        title = f"{title}\n{status_text}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    plt.close(fig)
    return image


class MultiVehicleViewer:
    """按仿真步刷新多车可视化，并可同步覆盖最新 PNG。"""

    def __init__(
        self,
        *,
        xodr_parser: XODRParser,
        vehicle_metadata: dict[str, dict[str, Any]],
        output_png: Path | None,
        refresh_every: int = 1,
    ) -> None:
        self.xodr_parser = xodr_parser
        self.vehicle_metadata = vehicle_metadata
        self.output_png = output_png
        self.refresh_every = max(int(refresh_every), 1)
        self.figure = None
        self.axis = None
        self.image_artist = None
        self.window_available = False
        if os.environ.get("OFFLINE_SIM_INTERACTIVE") == "1" and _is_interactive_backend():
            try:
                plt.ion()
                self.figure, self.axis = plt.subplots(figsize=(9, 9))
                self.axis.axis("off")
                self.figure.tight_layout(pad=0.1)
                plt.show(block=False)
                self.window_available = True
            except Exception:
                self.figure = None
                self.axis = None
                self.window_available = False

    def update(
        self,
        *,
        traces: dict[str, list[list[float]]],
        step_index: int,
        planned_trajectories: dict[str, np.ndarray] | None = None,
        status_text: str = "",
        force: bool = False,
    ) -> None:
        """渲染并显示当前多车帧。"""

        if not force and step_index >= 0 and (step_index + 1) % self.refresh_every != 0:
            return
        image = render_multi_vehicle_frame(
            xodr_parser=self.xodr_parser,
            traces=traces,
            vehicle_metadata=self.vehicle_metadata,
            frame_index=max(step_index + 1, 0),
            planned_trajectories=planned_trajectories,
            status_text=status_text,
        )
        if self.output_png is not None:
            self.output_png.parent.mkdir(parents=True, exist_ok=True)
            image.save(self.output_png)
        if not self.window_available or self.figure is None or self.axis is None:
            return
        image_array = np.asarray(image)
        if self.image_artist is None:
            self.image_artist = self.axis.imshow(image_array)
        else:
            self.image_artist.set_data(image_array)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def close(self, block: bool = False) -> None:
        """关闭多车可视化窗口。"""

        if not self.window_available or self.figure is None:
            return
        if block:
            plt.ioff()
            plt.show()
            return
        plt.close(self.figure)


def save_multi_vehicle_gif(
    *,
    xodr_parser: XODRParser,
    traces: dict[str, list[list[float]]],
    vehicle_metadata: dict[str, dict[str, Any]],
    dt: float,
    output_gif: Path,
    log_every: int = 10,
) -> None:
    """将多车仿真轨迹保存为 GIF。"""

    frame_count = max((len(trace) for trace in traces.values()), default=0)
    if frame_count <= 0:
        raise ValueError("多车仿真轨迹为空，无法生成 GIF")
    frames: list[Image.Image] = []
    for frame_index in range(frame_count):
        if frame_index == 0 or (frame_index + 1) % max(int(log_every), 1) == 0:
            print(f"[render] multi GIF frame {frame_index + 1}/{frame_count}", flush=True)
        frames.append(
            render_multi_vehicle_frame(
                xodr_parser=xodr_parser,
                traces=traces,
                vehicle_metadata=vehicle_metadata,
                frame_index=frame_index,
            )
        )
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=max(int(round(float(dt) * 1000.0)), 50),
        loop=0,
    )
