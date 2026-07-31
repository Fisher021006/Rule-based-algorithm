"""离线多车场景算法实例隔离。

实车和虚拟域控中，每辆算法车由独立进程运行，模块全局变量
天然隔离。只有单进程多车离线仿真需要为每辆车创建独立的
场景算法模块，避免共享 ``CFG`` 和 ``SCENE_MEMORY``。
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys
from contextlib import contextmanager
from typing import Any


_INSTANCE_COUNTER = itertools.count()


@contextmanager
def scoped_scene_environment(environment: dict[str, Any]):
    """在单辆离线算法车的初始化或规划期间隔离场景参数。"""

    overrides = {str(key): str(value) for key, value in environment.items()}
    isolated_keys = {
        key for key in os.environ if key.startswith("ROS_SCENE_")
    } | set(overrides)
    previous_values = {key: os.environ.get(key) for key in isolated_keys}
    try:
        for key in isolated_keys:
            os.environ.pop(key, None)
        os.environ.update(overrides)
        yield
    finally:
        for key in isolated_keys:
            previous = previous_values[key]
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def load_isolated_scene_algorithm(
    symbol_path: str,
    *,
    instance_key: str,
    environment: dict[str, Any],
):
    """从场景源文件加载一个仅属于指定离线车辆的算法实例。"""

    normalized = str(symbol_path or "").strip()
    if not normalized or "." not in normalized:
        raise ValueError(
            f"invalid scene algorithm symbol path: {normalized}. "
            "expected format '<module>.<symbol>'"
        )
    normalized_key = str(instance_key or "").strip()
    if not normalized_key:
        raise ValueError("offline scene algorithm instance_key cannot be empty")

    module_name, symbol_name = normalized.rsplit(".", 1)
    source_spec = importlib.util.find_spec(module_name)
    if source_spec is None or source_spec.origin is None or source_spec.loader is None:
        raise ImportError(f"cannot resolve scene algorithm module: {module_name}")

    safe_key = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in normalized_key
    )
    parent_package, _, module_leaf = module_name.rpartition(".")
    isolated_leaf = (
        f"_offline_{module_leaf}_{safe_key}_{next(_INSTANCE_COUNTER)}"
    )
    isolated_name = (
        f"{parent_package}.{isolated_leaf}"
        if parent_package
        else isolated_leaf
    )
    isolated_spec = importlib.util.spec_from_file_location(
        isolated_name,
        source_spec.origin,
        submodule_search_locations=source_spec.submodule_search_locations,
    )
    if isolated_spec is None or isolated_spec.loader is None:
        raise ImportError(f"cannot create isolated scene algorithm module: {module_name}")

    module = importlib.util.module_from_spec(isolated_spec)
    sys.modules[isolated_name] = module
    try:
        with scoped_scene_environment(environment):
            isolated_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(isolated_name, None)
        raise

    try:
        algorithm = getattr(module, symbol_name)
    except AttributeError:
        sys.modules.pop(isolated_name, None)
        raise
    if not callable(algorithm):
        sys.modules.pop(isolated_name, None)
        raise TypeError(f"scene algorithm symbol is not callable: {normalized}")
    return algorithm
