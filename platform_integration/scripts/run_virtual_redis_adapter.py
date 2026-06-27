#!/usr/bin/env python3
"""虚拟平台推演 Adapter 的命令行入口包装。

直接执行本脚本时，会先把项目根目录和平台接入目录加入 Python 搜索路径，再调用
`redis_ros_adapter.virtual.virtual_adapter.main()`。这样无论在 Docker 容器里
从哪个目录启动，都能正确导入项目内模块。
"""

from pathlib import Path
import sys


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLATFORM_ROOT.parent
for path in (PROJECT_ROOT, PLATFORM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from redis_ros_adapter.virtual.virtual_adapter import main


if __name__ == "__main__":
    main()
