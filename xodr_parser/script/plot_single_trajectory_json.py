import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Replace this path when you want to visualize another saved frame.
TRAJECTORY_JSON_PATH = PROJECT_ROOT / "outputs" / "planned_trajectories" / "82_20260316_031423/10000119_000032_1773627268218.json"

# Saved trajectory frames are 41 points at 50 ms spacing by default.
TIME_STEP_S = 0.05


def load_trajectory(json_path):
    with Path(json_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    trajectory = np.asarray(data, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] < 6:
        raise ValueError(f"Expected trajectory shape (N, 6), got {trajectory.shape}")
    return trajectory


def build_time_axis(num_points, dt):
    return np.arange(num_points, dtype=float) * dt


def default_output_path(json_path):
    json_path = Path(json_path)
    return json_path.with_name(f"{json_path.stem}_profile.png")


def plot_trajectory_profile(trajectory, json_path, output_path, dt):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    v = trajectory[:, 3]
    a = trajectory[:, 4]
    t = build_time_axis(len(trajectory), dt)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    ax_xy = axes[0]
    ax_xy.plot(x, y, color="#d96c06", linewidth=2.2)
    ax_xy.scatter(x[0], y[0], color="#1f9d55", s=60, label="start", zorder=3)
    ax_xy.scatter(x[-1], y[-1], color="#d64545", s=60, label="end", zorder=3)
    ax_xy.set_title(f"Trajectory Shape: {Path(json_path).name}")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax_xy.legend(loc="best")

    ax_v = axes[1]
    ax_v.plot(t, v, color="#1f77b4", linewidth=2.0)
    ax_v.scatter(t, v, color="#1f77b4", s=18)
    ax_v.set_title("Speed vs Time")
    ax_v.set_xlabel("time (s)")
    ax_v.set_ylabel("v (m/s)")
    ax_v.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    ax_a = axes[2]
    ax_a.plot(t, a, color="#c43d3d", linewidth=2.0)
    ax_a.scatter(t, a, color="#c43d3d", s=18)
    ax_a.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax_a.set_title("Acceleration vs Time")
    ax_a.set_xlabel("time (s)")
    ax_a.set_ylabel("a (m/s^2)")
    ax_a.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize one saved trajectory JSON file")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=str(TRAJECTORY_JSON_PATH),
        help="Path to one saved trajectory JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional PNG output path; default is <json_stem>_profile.png",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=TIME_STEP_S,
        help="Time step in seconds for v/a plots; default is 0.05",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Trajectory JSON not found: {json_path}")

    trajectory = load_trajectory(json_path)
    output_path = Path(args.output) if args.output else default_output_path(json_path)
    plot_trajectory_profile(trajectory, json_path, output_path, args.dt)

    print(f"Input JSON: {json_path.resolve()}")
    print(f"Output PNG: {output_path.resolve()}")


if __name__ == "__main__":
    main()
