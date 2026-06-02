"""场景算法 / 采样局部规划器离线仿真入口。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


DEFAULT_MAP_CACHE = PROJECT_ROOT / "xodr_parser" / "cache" / "同济测试场.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scene_offline"

DEFAULT_EGO_START = (-418.0, 59.0)
DEFAULT_EGO_GOAL = (-429.0, 129.0)
DEFAULT_MAIN_START = (-422.4995977931014, 66.86791384161317)
DEFAULT_MAIN_GOAL = (-429.0, 129.0)


def configure_matplotlib_environment(interactive: bool) -> None:
    """按是否需要 GUI 可视化配置 matplotlib 环境变量。"""

    os.environ["OFFLINE_SIM_INTERACTIVE"] = "1" if interactive else "0"
    if not interactive:
        os.environ.setdefault("MPLBACKEND", "Agg")


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="场景算法离线仿真壳，支持场景目标和局部采样规划")
    parser.add_argument(
        "--planner-mode",
        choices=["scene_target", "local_sampling"],
        default="scene_target",
        help="scene_target: 场景算法给终端点；local_sampling: 运行当前局部规划器采样链",
    )
    parser.add_argument(
        "--scene-algorithm-symbol",
        default="scenarios.scene_algorithm.scene_algorithm",
        help="scene_target 模式下使用的场景算法符号路径，例如 scenarios.cut_in_algorithm.cut_in_algorithm",
    )
    parser.add_argument("--map-cache", default=str(DEFAULT_MAP_CACHE), help="地图缓存 .pkl 路径")
    parser.add_argument("--ego-start-xy", nargs=2, type=float, default=DEFAULT_EGO_START, metavar=("X", "Y"))
    parser.add_argument("--ego-goal-xy", nargs=2, type=float, default=DEFAULT_EGO_GOAL, metavar=("X", "Y"))
    parser.add_argument("--main-start-xy", nargs=2, type=float, default=DEFAULT_MAIN_START, metavar=("X", "Y"), help="主车路线模式起点")
    parser.add_argument("--main-goal-xy", nargs=2, type=float, default=DEFAULT_MAIN_GOAL, metavar=("X", "Y"), help="主车路线模式终点")
    parser.add_argument("--main-static-xy", nargs=2, type=float, default=DEFAULT_MAIN_START, metavar=("X", "Y"), help="主车 static 模式位置")
    parser.add_argument("--ego-start-speed", type=float, default=0.0, help="自车初始速度 m/s")
    parser.add_argument("--main-start-speed", type=float, default=2.0, help="主车初始速度 m/s")
    parser.add_argument("--main-desired-speed", type=float, default=4.0, help="主车 IDM 期望速度 m/s；constant 模式忽略该参数")
    parser.add_argument("--disable-main-obstacle", action="store_true", help="不向局部规划器注入主车障碍物，用于无障碍基线测试")
    parser.add_argument(
        "--main-motion",
        choices=["idm", "constant", "static"],
        default="idm",
        help="主车运动模式：idm 表示按 IDM 运动，constant 表示按 main-start-speed 匀速沿路线行驶，static 表示静止充当障碍物",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="离散仿真步长 s")
    parser.add_argument("--advance-index", type=int, default=1, help="每轮自车执行局部轨迹中的第几个点")
    parser.add_argument("--max-steps", type=int, default=300, help="最大离散步数")
    parser.add_argument("--arrive-dist", type=float, default=1.5, help="自车到达阈值 m")
    parser.add_argument("--arrive-speed", type=float, default=0.3, help="自车到达速度阈值 m/s")
    parser.add_argument("--ref-lane-count", type=int, default=5, help="暴露给场景算法的前向 lane 段数量")
    parser.add_argument("--log-every", type=int, default=10, help="每隔多少个 step 打印一次完成进度")
    parser.add_argument("--skip-gif", action="store_true", help="只保存静态图和 JSON，不生成 GIF")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--output-prefix", default="scene_offline", help="输出文件名前缀")
    parser.add_argument("--show", action="store_true", help="仿真结束后保持最终图片窗口")
    parser.add_argument(
        "--live-show",
        action="store_true",
        help="仿真运行过程中实时刷新可视化窗口，并同步覆盖输出 *.live.png",
    )
    parser.add_argument(
        "--live-refresh-every",
        type=int,
        default=1,
        help="在线可视化每隔多少个规划 step 刷新一次，默认每步都刷",
    )
    return parser


def main() -> None:
    """脚本命令行入口。"""

    parser = build_arg_parser()
    pre_args, _ = parser.parse_known_args()
    configure_matplotlib_environment(interactive=bool(pre_args.show or pre_args.live_show))
    from offline_sim.runner import run_simulation

    args = parser.parse_args()
    summary = run_simulation(args)
    compact = {
        "planner_mode": summary["planner_mode"],
        "scene_algorithm_symbol": summary.get("scene_algorithm_symbol", ""),
        "status": summary["status"],
        "failure_reason": summary["failure_reason"],
        "ego_steps": max(len(summary["ego_trace"]) - 1, 0),
        "main_steps": max(len(summary["main_trace"]) - 1, 0),
        "output_png": summary["output_png"],
        "output_live_png": summary.get("output_live_png"),
        "output_gif": summary["output_gif"],
        "output_json": summary["output_json"],
        "output_timing_json": summary.get("output_timing_json"),
    }
    print(compact)


if __name__ == "__main__":
    main()
