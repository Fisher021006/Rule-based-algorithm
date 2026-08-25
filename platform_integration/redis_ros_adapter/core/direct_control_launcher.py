"""`direct_control_node.py` 子进程启动器。

Adapter 自己负责平台 Redis 和 ROS 桥接，但真正的规划控制算法仍然由
`direct_control_node.py` 运行。这个文件把平台准备好的任务、地图路径、
规划模式、频率等配置转换成算法脚本的命令行参数，并为每次任务创建独立
日志文件。
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from redis_ros_adapter.core.config import normalize_scene_environment
from redis_ros_adapter.core.types import PreparedTask


def _safe_filename_part(value: object, fallback: str) -> str:
    """把任务 ID、场景 ID 清洗成可用于日志文件名的片段。"""

    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._-") or fallback


def _log_algorithm_label(planner_mode: str, scene_algorithm_symbol: str) -> str:
    """根据实际启动参数生成便于筛选的日志算法标签。"""

    mode = str(planner_mode or "").strip().lower()
    if mode == "local_sampling":
        return "local_sampling"
    if mode != "scene_target":
        return _safe_filename_part(mode, "unknown_planner")

    parts = [part for part in str(scene_algorithm_symbol or "").strip().split(".") if part]
    if not parts:
        return "scene_target_default"

    # scene symbol 通常是 ``package.module.function``，日志以模块名代表场景算法。
    module_name = parts[-2] if len(parts) >= 2 else parts[-1]
    scenario_name = re.sub(r"_algorithm$", "", module_name)
    return f"scene_{_safe_filename_part(scenario_name, 'unknown')}"


class DirectControlLauncher:
    """管理单个 `direct_control_node.py` 进程的生命周期。

    每次平台开始任务时，Adapter 会调用 `start()`；平台停止任务或 Adapter
    退出时调用 `stop()`。如果连续测试同一个任务 ID，日志文件名里的时间戳
    可以避免覆盖前一次运行结果。
    """

    def __init__(self, project_root: Path, log_dir: Path) -> None:
        """记录项目根目录和算法日志目录，但不立即启动算法。"""

        self.project_root = Path(project_root)
        self.log_dir = Path(log_dir)
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.current_log_path: Path | None = None

    @property
    def running(self) -> bool:
        """判断算法子进程是否仍在运行。"""

        return self.process is not None and self.process.poll() is None

    def start(
        self,
        task: PreparedTask,
        *,
        map_path: str,
        cache_path: str,
        rebuild_cache: bool,
        planner_mode: str,
        scene_algorithm_symbol: str,
        obu_topic: str,
        obu_self_ptc_id: str,
        scene_main_ptc_id: str,
        scene_exposed_ptc_id: str,
        scene_environment: dict[str, Any] | None,
        scene_main_start_xy: list[float] | None,
        scene_main_start_latlon: list[float] | None,
        scene_main_goal_xy: list[float] | None,
        scene_main_goal_latlon: list[float] | None,
        sensorgps_topic: str,
        actuator_topic: str,
        controlcmd_topic: str,
        plan_hz: float,
        control_hz: float,
        trajectory_topic: str,
        node_name: str,
        max_speed_kph: float,
    ) -> None:
        """按平台任务启动一次 `direct_control_node.py`。

        启动前会先停止已有算法进程，避免上一个任务的 ROS 节点和日志继续
        干扰当前任务。起终点来自平台 type=1 准备任务，规划模式和频率来自
        `platform_integration/config/platform_real.yaml`。
        """

        if not task.ready or task.start is None or task.goal is None:
            raise ValueError("prepared task must contain start and goal")
        self.stop()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        task_id = _safe_filename_part(task.task_id, "unknown_task")
        case_id = _safe_filename_part(task.case_id, "unknown_case")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        algorithm_label = _log_algorithm_label(planner_mode, scene_algorithm_symbol)
        log_path = self.log_dir / f"{algorithm_label}_{timestamp}_task_{task_id}_case_{case_id}.log"
        trajectory_log_path = log_path.with_suffix(".trajectory.jsonl")
        self.current_log_path = log_path
        self.log_file = open(log_path, "w", encoding="utf-8")
        command = [
            sys.executable,
            str(self.project_root / "direct_control_node.py"),
            "--map-path",
            str(map_path),
            "--cache-path",
            str(cache_path),
            "--start-latlon",
            str(task.start.latitude),
            str(task.start.longitude),
            "--goal-latlon",
            str(task.goal.latitude),
            str(task.goal.longitude),
            "--planner-mode",
            str(planner_mode),
            "--sensorgps-topic",
            str(sensorgps_topic),
            "--actuator-topic",
            str(actuator_topic),
            "--obu-topic",
            str(obu_topic),
            "--controlcmd-topic",
            str(controlcmd_topic),
            "--plan-hz",
            str(float(plan_hz)),
            "--control-hz",
            str(float(control_hz)),
            "--trajectory-topic",
            str(trajectory_topic),
            "--node-name",
            str(node_name),
            "--max-speed-kph",
            str(float(max_speed_kph)),
        ]
        if rebuild_cache:
            command.append("--rebuild-cache")
        if scene_algorithm_symbol:
            command.extend(["--scene-algorithm-symbol", str(scene_algorithm_symbol)])
        if obu_self_ptc_id:
            command.extend(["--obu-self-ptc-id", str(obu_self_ptc_id)])
        if str(planner_mode).strip().lower() == "scene_target" and scene_algorithm_symbol:
            command.extend(["--scene-trajectory-log-path", str(trajectory_log_path)])
        if scene_main_ptc_id:
            command.extend(["--scene-main-ptc-id", str(scene_main_ptc_id)])
        if scene_main_start_xy:
            command.extend(["--scene-main-start-xy", str(scene_main_start_xy[0]), str(scene_main_start_xy[1])])
        if scene_main_start_latlon:
            command.extend(["--scene-main-start-latlon", str(scene_main_start_latlon[0]), str(scene_main_start_latlon[1])])
        if scene_main_goal_xy:
            command.extend(["--scene-main-goal-xy", str(scene_main_goal_xy[0]), str(scene_main_goal_xy[1])])
        if scene_main_goal_latlon:
            command.extend(["--scene-main-goal-latlon", str(scene_main_goal_latlon[0]), str(scene_main_goal_latlon[1])])
        env = os.environ.copy()
        normalized_scene_environment = normalize_scene_environment(scene_environment)
        env.update(normalized_scene_environment)
        # 显式字段优先于通用环境变量，保证旧配置语义稳定。
        if scene_exposed_ptc_id:
            env["ROS_SCENE_EXPOSED_PTC_ID"] = str(scene_exposed_ptc_id)
        print(f"[launcher] log_path={log_path}", file=self.log_file, flush=True)
        if str(planner_mode).strip().lower() == "scene_target" and scene_algorithm_symbol:
            print(f"[launcher] trajectory_log_path={trajectory_log_path}", file=self.log_file, flush=True)
        if normalized_scene_environment:
            print(
                "[launcher] scene_environment="
                + json.dumps(
                    normalized_scene_environment,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=self.log_file,
                flush=True,
            )
        print(f"[launcher] command={' '.join(command)}", file=self.log_file, flush=True)
        self.process = subprocess.Popen(
            command,
            cwd=str(self.project_root),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

    def stop(self) -> None:
        """停止当前算法进程并关闭日志文件句柄。"""

        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
        self.process = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def poll(self) -> int | None:
        """查询算法子进程退出码；未启动时返回 `None`。"""

        if self.process is None:
            return None
        return self.process.poll()
