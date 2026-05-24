"""离线仿真器产物的绘图与渲染辅助函数。"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib


def _configure_matplotlib_backend() -> None:
    """根据环境变量选择 matplotlib 后端。"""

    if os.environ.get("MPLBACKEND"):
        return
    if os.environ.get("OFFLINE_SIM_INTERACTIVE") != "1":
        matplotlib.use("Agg")
        return
    backend_candidates: list[str] = []
    if sys.platform == "darwin":
        backend_candidates.append("MacOSX")
    backend_candidates.extend(["TkAgg", "QtAgg"])
    for backend_name in backend_candidates:
        try:
            matplotlib.use(backend_name)
            os.environ["OFFLINE_SIM_ACTIVE_BACKEND"] = backend_name
            return
        except Exception:
            continue
    matplotlib.use("Agg")
    os.environ["OFFLINE_SIM_ACTIVE_BACKEND"] = "Agg"


_configure_matplotlib_backend()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from xodr_parser.xodr_parser import XODRParser

if TYPE_CHECKING:
    from planning_runtime.scene.provider import SceneTargetProvider


def _is_interactive_backend() -> bool:
    """判断当前 matplotlib backend 是否支持交互窗口。"""

    backend_name = str(matplotlib.get_backend()).lower()
    try:
        from matplotlib.backends import BackendFilter, backend_registry

        interactive_backends = {
            name.lower()
            for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        }
        return backend_name in interactive_backends
    except Exception:
        return backend_name in {
            "macosx",
            "tkagg",
            "tkcairo",
            "qtagg",
            "qt5agg",
            "qt5cairo",
            "qtcairo",
            "gtk3agg",
            "gtk3cairo",
            "gtk4agg",
            "gtk4cairo",
            "wx",
            "wxagg",
            "wxcairo",
            "nbagg",
            "webagg",
            "notebook",
        }


def draw_vehicle(ax: Any, x: float, y: float, heading: float, length: float, width: float, color: str, label: str) -> None:
    """在 matplotlib 画布上绘制简化车辆框。"""

    dx = 0.5 * length * math.cos(heading)
    dy = 0.5 * length * math.sin(heading)
    lateral_x = 0.5 * width * math.cos(heading + math.pi / 2.0)
    lateral_y = 0.5 * width * math.sin(heading + math.pi / 2.0)
    corners = np.array(
        [
            [x - dx + lateral_x, y - dy + lateral_y],
            [x - dx - lateral_x, y - dy - lateral_y],
            [x + dx - lateral_x, y + dy - lateral_y],
            [x + dx + lateral_x, y + dy + lateral_y],
            [x - dx + lateral_x, y - dy + lateral_y],
        ],
        dtype=float,
    )
    ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=1.2)
    ax.text(x, y, label, color=color, fontsize=8)


def _render_single_frame(
    xodr_parser: XODRParser,
    ego_trace: np.ndarray,
    main_trace: np.ndarray,
    provider: "SceneTargetProvider",
    ego_goal_xy: np.ndarray,
    planner_mode: str,
    frame_idx: int,
    planned_trajectory: np.ndarray | None = None,
    status_text: str | None = None,
) -> Image.Image:
    """渲染输出 GIF 的单帧画面。"""

    fig, ax = plt.subplots(figsize=(8, 8))

    for lane in xodr_parser.lane_centerlines.values():
        lane = np.asarray(lane, dtype=float)
        if len(lane) >= 2:
            ax.plot(lane[:, 0], lane[:, 1], color="#D0D6DB", linewidth=0.6)

    ego_end = min(frame_idx + 1, len(ego_trace))
    main_end = min(frame_idx + 1, len(main_trace))
    ax.plot(ego_trace[:ego_end, 0], ego_trace[:ego_end, 1], color="#D62728", linewidth=2.0, label="ego trajectory")
    ax.plot(main_trace[:main_end, 0], main_trace[:main_end, 1], color="#111111", linewidth=2.0, label="main trajectory")
    if planned_trajectory is not None and len(planned_trajectory) >= 2:
        ax.plot(
            planned_trajectory[:, 0],
            planned_trajectory[:, 1],
            color="#2CA02C",
            linewidth=1.6,
            linestyle="--",
            alpha=0.95,
            label="current plan",
        )
    ax.scatter(ego_trace[0, 0], ego_trace[0, 1], color="#D62728", s=30, marker="o", label="ego start")
    ax.scatter(ego_goal_xy[0], ego_goal_xy[1], color="#D62728", s=50, marker="*", label="ego goal")
    ax.scatter(main_trace[0, 0], main_trace[0, 1], color="#111111", s=30, marker="o", label="main start")

    if provider.last_scene_context is not None:
        ego_ctx = provider.last_scene_context.ego_vehicle
        main_ctx = provider.last_scene_context.main_vehicle
        for line in ego_ctx.candidate_ref_lines:
            style = "-" if line.line_index == 0 else "--"
            color = "#1F77B4" if line.line_index == 0 else "#7FB3D5"
            ax.plot(line.line[:, 0], line.line[:, 1], linestyle=style, color=color, linewidth=1.2)
        for line in main_ctx.candidate_ref_lines:
            style = "-" if line.line_index == 0 else "--"
            color = "#FF7F0E" if line.line_index == 0 else "#F5B041"
            ax.plot(line.line[:, 0], line.line[:, 1], linestyle=style, color=color, linewidth=1.0)

    for marker in provider.history[: min(frame_idx + 1, len(provider.history))]:
        target = marker["target"]
        ax.scatter(target["x"], target["y"], color="#9467BD", s=18, marker="x")

    draw_vehicle(
        ax,
        x=float(ego_trace[ego_end - 1, 0]),
        y=float(ego_trace[ego_end - 1, 1]),
        heading=float(ego_trace[ego_end - 1, 2]),
        length=4.64,
        width=1.78,
        color="#D62728",
        label="ego",
    )
    draw_vehicle(
        ax,
        x=float(main_trace[main_end - 1, 0]),
        y=float(main_trace[main_end - 1, 1]),
        heading=float(main_trace[main_end - 1, 2]),
        length=4.6,
        width=1.8,
        color="#111111",
        label="main",
    )

    all_xy = np.vstack([ego_trace[:, :2], main_trace[:, :2]])
    min_xy = np.min(all_xy, axis=0) - 15.0
    max_xy = np.max(all_xy, axis=0) + 15.0
    ax.set_xlim(min_xy[0], max_xy[0])
    ax.set_ylim(min_xy[1], max_xy[1])
    ax.set_aspect("equal")
    title = f"Offline Scene Simulation ({planner_mode})"
    if status_text:
        title = f"{title}\n{status_text}"
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    plt.close(fig)
    return image


class LiveSimulationViewer:
    """按规划步刷新在线可视化窗口，并同步输出最新 PNG。"""

    def __init__(
        self,
        xodr_parser: XODRParser,
        provider: "SceneTargetProvider",
        ego_goal_xy: np.ndarray,
        planner_mode: str,
        output_png: Path | None,
        refresh_every: int = 1,
    ) -> None:
        self.xodr_parser = xodr_parser
        self.provider = provider
        self.ego_goal_xy = np.asarray(ego_goal_xy, dtype=float)
        self.planner_mode = planner_mode
        self.output_png = output_png
        self.refresh_every = max(int(refresh_every), 1)
        self.figure = None
        self.axis = None
        self.image_artist = None
        self.window_available = False

        if os.environ.get("OFFLINE_SIM_INTERACTIVE") == "1" and _is_interactive_backend():
            try:
                plt.ion()
                self.figure, self.axis = plt.subplots(figsize=(8, 8))
                self.axis.axis("off")
                self.figure.tight_layout(pad=0.1)
                plt.show(block=False)
                self.window_available = True
            except Exception:
                self.figure = None
                self.axis = None
                self.image_artist = None
                self.window_available = False

    def update(
        self,
        ego_trace: list[list[float]],
        main_trace: list[list[float]],
        step_index: int,
        planned_trajectory: list[list[float]] | np.ndarray | None = None,
        status_text: str | None = None,
        force: bool = False,
    ) -> None:
        """渲染当前最新规划结果。"""

        if not force and step_index >= 0 and (step_index + 1) % self.refresh_every != 0:
            return

        ego_arr = np.asarray(ego_trace, dtype=float)
        main_arr = np.asarray(main_trace, dtype=float)
        if len(ego_arr) == 0 or len(main_arr) == 0:
            return
        planned_arr = None if planned_trajectory is None else np.asarray(planned_trajectory, dtype=float)
        image = _render_single_frame(
            xodr_parser=self.xodr_parser,
            ego_trace=ego_arr,
            main_trace=main_arr,
            provider=self.provider,
            ego_goal_xy=self.ego_goal_xy,
            planner_mode=self.planner_mode,
            frame_idx=len(ego_arr) - 1,
            planned_trajectory=planned_arr,
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

    def close(self, block: bool) -> None:
        """结束在线可视化。"""

        if not self.window_available or self.figure is None:
            return
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        if block:
            plt.ioff()
            plt.show()
            return
        plt.close(self.figure)


def save_animation_gif(
    xodr_parser: XODRParser,
    ego_trace: list[list[float]],
    main_trace: list[list[float]],
    provider: "SceneTargetProvider",
    ego_goal_xy: np.ndarray,
    planner_mode: str,
    dt: float,
    output_gif: Path,
    log_every: int = 10,
) -> None:
    """把完整仿真过程保存成 GIF。"""

    ego_arr = np.asarray(ego_trace, dtype=float)
    main_arr = np.asarray(main_trace, dtype=float)
    frame_count = max(len(ego_arr), len(main_arr))
    if frame_count == 0:
        raise ValueError("轨迹为空，无法生成 GIF")

    log_every = max(int(log_every), 1)
    frames: list[Image.Image] = []
    for frame_idx in range(frame_count):
        if frame_idx == 0 or (frame_idx + 1) % log_every == 0 or frame_idx == frame_count - 1:
            print(f"[render] GIF frame {frame_idx + 1}/{frame_count}", flush=True)
        frames.append(
            _render_single_frame(
                xodr_parser=xodr_parser,
                ego_trace=ego_arr,
                main_trace=main_arr,
                provider=provider,
                ego_goal_xy=ego_goal_xy,
                planner_mode=planner_mode,
                frame_idx=frame_idx,
            )
        )
    frame_duration_ms = max(int(round(dt * 1000.0)), 50)
    print(f"[render] writing GIF -> {output_gif}", flush=True)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )


def plot_results(
    xodr_parser: XODRParser,
    ego_trace: list[list[float]],
    main_trace: list[list[float]],
    provider: "SceneTargetProvider",
    ego_goal_xy: np.ndarray,
    output_png: Path,
    show: bool,
    planner_mode: str,
) -> None:
    """渲染静态 PNG 汇总图。"""

    fig, ax = plt.subplots(figsize=(8, 8))

    for lane in xodr_parser.lane_centerlines.values():
        lane = np.asarray(lane, dtype=float)
        if len(lane) >= 2:
            ax.plot(lane[:, 0], lane[:, 1], color="#D0D6DB", linewidth=0.6)

    ego_arr = np.asarray(ego_trace, dtype=float)
    main_arr = np.asarray(main_trace, dtype=float)
    ax.plot(ego_arr[:, 0], ego_arr[:, 1], color="#D62728", linewidth=2.0, label="ego trajectory")
    ax.plot(main_arr[:, 0], main_arr[:, 1], color="#111111", linewidth=2.0, label="main trajectory")
    ax.scatter(ego_arr[0, 0], ego_arr[0, 1], color="#D62728", s=30, marker="o", label="ego start")
    ax.scatter(ego_goal_xy[0], ego_goal_xy[1], color="#D62728", s=50, marker="*", label="ego goal")
    ax.scatter(main_arr[0, 0], main_arr[0, 1], color="#111111", s=30, marker="o", label="main start")

    if provider.last_scene_context is not None:
        ego_ctx = provider.last_scene_context.ego_vehicle
        main_ctx = provider.last_scene_context.main_vehicle
        for line in ego_ctx.candidate_ref_lines:
            style = "-" if line.line_index == 0 else "--"
            color = "#1F77B4" if line.line_index == 0 else "#7FB3D5"
            ax.plot(line.line[:, 0], line.line[:, 1], linestyle=style, color=color, linewidth=1.2)
        for line in main_ctx.candidate_ref_lines:
            style = "-" if line.line_index == 0 else "--"
            color = "#FF7F0E" if line.line_index == 0 else "#F5B041"
            ax.plot(line.line[:, 0], line.line[:, 1], linestyle=style, color=color, linewidth=1.0)

    for item in provider.history:
        target = item["target"]
        ax.scatter(target["x"], target["y"], color="#9467BD", s=18, marker="x")

    draw_vehicle(
        ax,
        x=float(ego_arr[-1, 0]),
        y=float(ego_arr[-1, 1]),
        heading=float(ego_arr[-1, 2]),
        length=4.64,
        width=1.78,
        color="#D62728",
        label="ego",
    )
    draw_vehicle(
        ax,
        x=float(main_arr[-1, 0]),
        y=float(main_arr[-1, 1]),
        heading=float(main_arr[-1, 2]),
        length=4.6,
        width=1.8,
        color="#111111",
        label="main",
    )

    all_xy = np.vstack([ego_arr[:, :2], main_arr[:, :2]])
    min_xy = np.min(all_xy, axis=0) - 15.0
    max_xy = np.max(all_xy, axis=0) + 15.0
    ax.set_xlim(min_xy[0], max_xy[0])
    ax.set_ylim(min_xy[1], max_xy[1])
    ax.set_aspect("equal")
    ax.set_title(f"Offline Scene Simulation ({planner_mode})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
