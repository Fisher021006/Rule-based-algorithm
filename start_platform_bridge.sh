#!/usr/bin/env bash
# 一键启动平台任务接入桥：mux + task-manager + adapter + algorithm。
#
# 用法：
#   ./start_platform_bridge.sh                        # 默认 cut_in，限速 0（车不会动）
#   ./start_platform_bridge.sh --max-speed-kph 25     # 真正控车
#   ./start_platform_bridge.sh -a scenarios.blind_spot_algorithm.blind_spot_algorithm
#   ./start_platform_bridge.sh -p local_sampling      # A-B 自由行驶，不挂场景算法
#   ./start_platform_bridge.sh --list                 # 列出可选场景
#   ./start_platform_bridge.sh --dry-run              # 只打印命令，不启动
#   ./start_platform_bridge.sh --no-use-taskstatus-transitions  # 只认 /start_task
#   ./start_platform_bridge.sh --running-task-status-values 1,2   # 哪些 taskStatus 算执行中
#   ./start_platform_bridge.sh --scene-main-ptc-id 3690   # 锁定场景主车
#   ./start_platform_bridge.sh --scene-main-start-xy -315 176 --scene-main-goal-xy -272 124
#                                                           # 显式指定场景主车路线（平台不发主车路径）
#   ./start_platform_bridge.sh --trajectory-log-path logs/intersection.trajectory.jsonl
#                                                           # 逐帧规划结果落盘（JSONL）
#   ./start_platform_bridge.sh status                 # 查看运行中的角色
#   ./start_platform_bridge.sh stop                   # 停止所有角色
#
# 终端模式自动选择：有图形界面用 gnome-terminal（一个窗口、四个标签页），
# SSH 下用 tmux（一个会话、2x2 四窗格），都没有就后台运行并 tail 日志。
# 也可以 --mode gnome|tmux|background 强制指定。
#
# 启动后用 ./start_platform_bridge.sh attach 重新接回会话（不会重启进程）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------

ACTION="start"
MODE="auto"
PLANNER_MODE="scene_target"
SYMBOL="scenarios.cut_in_algorithm.cut_in_algorithm"
MAX_SPEED_KPH="0"
SCENE_MAIN_PTC_ID="3690"
SCENE_MAIN_START_XY=()
SCENE_MAIN_GOAL_XY=()
TRAJECTORY_LOG_PATH=""
PLAN_HZ="10"
CONTROL_HZ="20"
BRIDGE_HZ="20"
USE_CURRENT_START=1
# 同济平台实测只发 /taskstatus（周期任务状态），不发 /start_task。
# 默认让 /taskstatus 的 taskStatus 变化也能驱动 PREPARE/ACTIVE。
USE_TASKSTATUS_TRANSITIONS=1
# 同济平台用 taskStatus=1 表示“执行中”，不是 2；且从不下发“准备”值。
# 所以运行桶收 1 和 2，准备桶用 99 占位（平台永远不会发的值），等于废掉准备态。
# 注意：准备桶不能传空串，_int_set 会把空串兜回默认值 {1}，1 又会被准备桶抢走。
TASK_RUNNING_STATUS_VALUES="1,2"
TASK_PREPARE_STATUS_VALUES="99"
FORCE=0
DRY_RUN=0

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
STU_SETUP="${STU_SETUP:-$HOME/redis_stu/devel/setup.bash}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
TMUX_SESSION="bridge"

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

list_symbols() {
    echo "可选场景算法："
    local found=0
    for file in "$SCRIPT_DIR"/scenarios/*_algorithm.py; do
        [[ -e "$file" ]] || continue
        local name
        name="$(basename "$file" .py)"
        echo "  scenarios.$name.$name"
        found=1
    done
    [[ $found -eq 1 ]] || echo "  （scenarios/ 下没有找到场景算法）"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--scene-algorithm-symbol)
            SYMBOL="${2:?缺少参数值}"; shift 2 ;;
        -p|--planner-mode)
            PLANNER_MODE="${2:?缺少参数值}"; shift 2 ;;
        -s|--max-speed-kph)
            MAX_SPEED_KPH="${2:?缺少参数值}"; shift 2 ;;
        --scene-main-ptc-id)
            SCENE_MAIN_PTC_ID="${2:?缺少参数值}"; shift 2 ;;
        --scene-main-start-xy)
            SCENE_MAIN_START_XY=("${2:?缺少X}" "${3:?缺少Y}"); shift 3 ;;
        --scene-main-goal-xy)
            SCENE_MAIN_GOAL_XY=("${2:?缺少X}" "${3:?缺少Y}"); shift 3 ;;
        --trajectory-log-path)
            TRAJECTORY_LOG_PATH="${2:?缺少参数值}"; shift 2 ;;
        --plan-hz)
            PLAN_HZ="${2:?缺少参数值}"; shift 2 ;;
        --control-hz)
            CONTROL_HZ="${2:?缺少参数值}"; shift 2 ;;
        --bridge-hz)
            BRIDGE_HZ="${2:?缺少参数值}"; shift 2 ;;
        --no-use-current-start)
            USE_CURRENT_START=0; shift ;;
        --no-use-taskstatus-transitions)
            USE_TASKSTATUS_TRANSITIONS=0; shift ;;
        --running-task-status-values)
            TASK_RUNNING_STATUS_VALUES="${2:?缺少参数值}"; shift 2 ;;
        --prepare-task-status-values)
            TASK_PREPARE_STATUS_VALUES="${2:?缺少参数值}"; shift 2 ;;
        --mode)
            MODE="${2:?缺少参数值}"; shift 2 ;;
        --force)
            FORCE=1; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --list)
            list_symbols; exit 0 ;;
        -h|--help)
            usage; exit 0 ;;
        status|stop|attach)
            ACTION="$1"; shift ;;
        *)
            echo "未知参数：$1" >&2
            echo "使用 --help 查看用法" >&2
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# status / stop
# ---------------------------------------------------------------------------

PATTERN="vehicle_platform_bridge.py --role"

if [[ "$ACTION" == "status" ]]; then
    if pgrep -af "$PATTERN" >/dev/null 2>&1; then
        pgrep -af "$PATTERN"
    else
        echo "没有运行中的平台桥进程"
    fi
    exit 0
fi

if [[ "$ACTION" == "attach" ]]; then
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        exec tmux attach -t "$TMUX_SESSION"
    fi
    echo "没有可接入的 tmux 会话 $TMUX_SESSION" >&2
    exit 1
fi

if [[ "$ACTION" == "stop" ]]; then
    if pgrep -af "$PATTERN" >/dev/null 2>&1; then
        pgrep -af "$PATTERN"
        pkill -f "$PATTERN" || true
        sleep 1
        pkill -9 -f "$PATTERN" 2>/dev/null || true
        echo "已停止"
    else
        echo "没有运行中的平台桥进程"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# 预检
# ---------------------------------------------------------------------------

if [[ ! -f "$SCRIPT_DIR/vehicle_platform_bridge.py" ]]; then
    echo "错误：$SCRIPT_DIR 下找不到 vehicle_platform_bridge.py" >&2
    exit 1
fi

if [[ ! -f "$ROS_SETUP" ]]; then
    echo "错误：找不到 ROS 环境文件 $ROS_SETUP" >&2
    echo "      可以用 ROS_SETUP=/opt/ros/<版本>/setup.bash 覆盖" >&2
    exit 1
fi

if [[ ! -f "$STU_SETUP" ]]; then
    echo "错误：找不到车端消息工作空间 $STU_SETUP" >&2
    echo "      可以用 STU_SETUP=<你的devel/setup.bash> 覆盖" >&2
    exit 1
fi

if [[ "$PLANNER_MODE" == "scene_target" && -z "$SYMBOL" ]]; then
    echo "错误：scene_target 模式必须用 -a 指定场景算法" >&2
    list_symbols >&2
    exit 1
fi

case "$PLANNER_MODE" in
    scene_target|local_sampling) ;;
    *)
        echo "错误：未知规划模式 '$PLANNER_MODE'（可选 scene_target / local_sampling）" >&2
        exit 1 ;;
esac

if [[ ${#SCENE_MAIN_START_XY[@]} -gt 0 && ${#SCENE_MAIN_GOAL_XY[@]} -eq 0 ]] \
    || [[ ${#SCENE_MAIN_START_XY[@]} -eq 0 && ${#SCENE_MAIN_GOAL_XY[@]} -gt 0 ]]; then
    echo "错误：主车起点和终点必须同时提供" >&2
    exit 1
fi

if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
    VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
else
    PYTHON="python3"
    VENV_ACTIVATE=""
    echo "提示：没有找到 .venv，将使用系统 python3" >&2
fi

if pgrep -f "vehicle_platform_bridge.py --role mux" >/dev/null 2>&1 && [[ $FORCE -eq 0 ]]; then
    echo "错误：已经有 mux 在运行，/controlcmd 不能有两个发布者。" >&2
    echo "      先执行 $0 stop，或加 --force 强制再起一套。" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 组装四个角色的命令
# ---------------------------------------------------------------------------

# POSIX 安全的单引号转义，保证 tmux 用 sh 解析命令时也不会被中文路径或空格卡住。
shquote() {
    printf "'%s'" "${1//\'/\'\\\'\'}"
}

PRELUDE="cd $(shquote "$SCRIPT_DIR") && source $(shquote "$ROS_SETUP") && source $(shquote "$STU_SETUP")"
if [[ -n "$VENV_ACTIVATE" ]]; then
    PRELUDE+=" && source $(shquote "$VENV_ACTIVATE")"
fi

build_cmd() {
    local cmd="$PRELUDE && $(shquote "$PYTHON") vehicle_platform_bridge.py"
    local arg
    for arg in "$@"; do
        cmd+=" $(shquote "$arg")"
    done
    printf '%s' "$cmd"
}

CMD_MUX="$(build_cmd --role mux --bridge-hz "$BRIDGE_HZ" --max-speed-kph "$MAX_SPEED_KPH")"
TASK_ARGS=(
    --role task-manager
    --log-raw-messages 20
    --running-task-status-values "$TASK_RUNNING_STATUS_VALUES"
    --prepare-task-status-values "$TASK_PREPARE_STATUS_VALUES"
)
if [[ $USE_TASKSTATUS_TRANSITIONS -eq 1 ]]; then
    TASK_ARGS+=(--use-taskstatus-transitions)
fi
CMD_TASK="$(build_cmd "${TASK_ARGS[@]}")"
CMD_ADAPTER="$(build_cmd --role adapter --object-track-frame map)"

ALGO_ARGS=(--role algorithm --planner-mode "$PLANNER_MODE" --plan-hz "$PLAN_HZ" --control-hz "$CONTROL_HZ")
if [[ "$PLANNER_MODE" == "scene_target" ]]; then
    ALGO_ARGS+=(--scene-algorithm-symbol "$SYMBOL")
fi
if [[ "$SCENE_MAIN_PTC_ID" != "-" ]]; then
    ALGO_ARGS+=(--scene-main-ptc-id "$SCENE_MAIN_PTC_ID")
fi
if [[ ${#SCENE_MAIN_START_XY[@]} -gt 0 ]]; then
    ALGO_ARGS+=(--scene-main-start-xy "${SCENE_MAIN_START_XY[@]}")
    ALGO_ARGS+=(--scene-main-goal-xy "${SCENE_MAIN_GOAL_XY[@]}")
fi
if [[ -n "$TRAJECTORY_LOG_PATH" ]]; then
    ALGO_ARGS+=(--scene-trajectory-log-path "$TRAJECTORY_LOG_PATH")
fi
if [[ $USE_CURRENT_START -eq 1 ]]; then
    ALGO_ARGS+=(--use-current-start)
fi
CMD_ALGO="$(build_cmd "${ALGO_ARGS[@]}")"

# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

echo "============================================================"
echo " 平台任务接入桥"
echo " 规划模式 : $PLANNER_MODE"
if [[ "$PLANNER_MODE" == "scene_target" ]]; then
echo " 场景算法 : $SYMBOL"
fi
echo " 主车 ptcId: $SCENE_MAIN_PTC_ID"
if [[ ${#SCENE_MAIN_START_XY[@]} -gt 0 ]]; then
echo " 主车路线 : ${SCENE_MAIN_START_XY[*]} -> ${SCENE_MAIN_GOAL_XY[*]}"
else
echo " 主车路线 : 未配置（场景会报 route_intent_missing）"
fi
if [[ -n "$TRAJECTORY_LOG_PATH" ]]; then
echo " 轨迹日志 : $TRAJECTORY_LOG_PATH"
fi
echo " mux 限速 : ${MAX_SPEED_KPH} km/h"
if [[ $USE_TASKSTATUS_TRANSITIONS -eq 1 ]]; then
echo " 任务触发 : /start_task + /taskstatus"
echo " 状态映射 : taskStatus in {$TASK_RUNNING_STATUS_VALUES} -> ACTIVE, 0/3 -> STANDBY"
else
echo " 任务触发 : 仅 /start_task"
fi
if awk "BEGIN{exit !($MAX_SPEED_KPH > 0)}"; then
echo " ⚠ 限速大于 0，任务开始后车辆会真的移动"
else
echo " 限速为 0，任务开始后车辆不会移动（联调用）"
fi
echo "============================================================"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[mux]         $CMD_MUX"
    echo "[task-manager] $CMD_TASK"
    echo "[adapter]     $CMD_ADAPTER"
    echo "[algorithm]   $CMD_ALGO"
    exit 0
fi

if [[ "$MODE" == "auto" ]]; then
    if command -v gnome-terminal >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
        MODE="gnome"
    elif command -v tmux >/dev/null 2>&1; then
        MODE="tmux"
    else
        MODE="background"
    fi
fi

# 一个 tmux 会话里做 2x2 四窗格，四个角色的日志同屏可见。
#   ┌──────────────┬──────────────┐
#   │ mux          │ task-manager │
#   ├──────────────┼──────────────┤
#   │ adapter      │ algorithm    │
#   └──────────────┴──────────────┘
start_tmux_panes() {
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    tmux new-session -d -s "$TMUX_SESSION" -n bridge -c "$SCRIPT_DIR" "$CMD_MUX"
    tmux split-window -h -t "$TMUX_SESSION":0 -c "$SCRIPT_DIR" "$CMD_TASK"
    tmux select-pane -t "$TMUX_SESSION":0.0
    tmux split-window -v -t "$TMUX_SESSION":0 -c "$SCRIPT_DIR" "$CMD_ADAPTER"
    tmux select-pane -t "$TMUX_SESSION":0.1
    tmux split-window -v -t "$TMUX_SESSION":0 -c "$SCRIPT_DIR" "$CMD_ALGO"
    tmux select-layout -t "$TMUX_SESSION":0 tiled
    # 角色退出后保留窗格，报错信息不会一闪而过。
    tmux set-option -t "$TMUX_SESSION" remain-on-exit on || true
    tmux set-option -t "$TMUX_SESSION" pane-border-status top || true
    tmux set-option -t "$TMUX_SESSION" mouse on || true
    tmux select-pane -t "$TMUX_SESSION":0.0 -T "mux" || true
    tmux select-pane -t "$TMUX_SESSION":0.1 -T "task-manager" || true
    tmux select-pane -t "$TMUX_SESSION":0.2 -T "adapter" || true
    tmux select-pane -t "$TMUX_SESSION":0.3 -T "algorithm" || true
    tmux select-pane -t "$TMUX_SESSION":0.0
}

launch_gnome() {
    # 一个窗口、四个标签页。
    # 必须分多次调用：同一次调用里写多个 `--tab ... -- bash -c` 时，
    # gnome-terminal 的 `--` 会把后面的参数当成命令参数，结果退化成四个窗口。
    # 第一次用 --window 建窗口，之后每次 --tab 都挂到“最近打开的那个窗口”。
    local titles=(mux task-manager adapter algorithm)
    local cmds=("$CMD_MUX" "$CMD_TASK" "$CMD_ADAPTER" "$CMD_ALGO")
    local i flag
    for i in "${!titles[@]}"; do
        if [[ $i -eq 0 ]]; then flag="--window"; else flag="--tab"; fi
        gnome-terminal "$flag" --title="${titles[$i]}" -- bash -c \
            "printf '\\033]0;${titles[$i]}\\007'; ${cmds[$i]}; echo; echo '[${titles[$i]}] 已退出，按回车关闭'; exec bash" &
        sleep 0.6
    done
    echo "已在一个窗口内打开 4 个标签页：mux / task-manager / adapter / algorithm"
}

launch_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        echo "错误：没有安装 tmux，无法使用 tmux 模式（sudo apt install -y tmux）" >&2
        exit 1
    fi
    start_tmux_panes
    echo "已创建 tmux 会话 $TMUX_SESSION（2x2 四窗格），正在接入..."
    tmux attach -t "$TMUX_SESSION"
}

launch_background() {
    mkdir -p "$LOG_DIR"
    : > "$LOG_DIR/bridge_pids"
    local names=(mux task-manager adapter algorithm)
    local cmds=("$CMD_MUX" "$CMD_TASK" "$CMD_ADAPTER" "$CMD_ALGO")
    local i
    for i in "${!names[@]}"; do
        nohup bash -c "${cmds[$i]}" > "$LOG_DIR/${names[$i]}.log" 2>&1 &
        echo "$! ${names[$i]}" >> "$LOG_DIR/bridge_pids"
        sleep 0.3
    done
    echo "已在后台启动，日志目录：$LOG_DIR"
    echo "查看状态：$0 status    停止：$0 stop"
    echo "跟踪日志：tail -f $LOG_DIR/*.log"
}

case "$MODE" in
    gnome)      launch_gnome ;;
    tmux)       launch_tmux ;;
    background) launch_background ;;
    *)
        echo "错误：未知模式 $MODE（可选 gnome / tmux / background / auto）" >&2
        exit 1 ;;
esac
