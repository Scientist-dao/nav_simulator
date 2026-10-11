"""baseline_lib.py - 板端基线库路径解析

仿真器的 baseline 后端 (既有基线导航算法, C++ 扩展) 与其相机参数夹具
来自一个不随本仓库分发的板端工程。本模块按以下顺序解析该工程根目录
并把其加入 sys.path:

  1. 环境变量 NAV_BASELINE_LIB —— 指向含 navigation/ 包的工程根;
  2. 仿真器工作区同级目录下名为 navigation* 的目录
     (兼容外层再包一层同名目录的解包形态)。

均未命中时返回 None, 调用方自行回退或报错。
"""

import glob
import os
import sys


def baseline_root():
    """返回板端基线工程根目录 (含 navigation/ 包), 找不到返回 None。"""
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))          # nav_simulator/ 的上一级
    candidates = []
    env = os.environ.get("NAV_BASELINE_LIB")
    if env:
        candidates.append(env)
    candidates += sorted(glob.glob(os.path.join(workspace, "navigation*")))
    for cand in candidates:
        base = os.path.basename(os.path.normpath(cand))
        for path in (cand, os.path.join(cand, base)):
            if os.path.isdir(os.path.join(path, "navigation")):
                return path
    return None


def add_baseline_to_syspath():
    """把基线工程根目录前插 sys.path; 返回根目录或 None。"""
    root = baseline_root()
    if root is not None and root not in sys.path:
        sys.path.insert(0, root)
    return root


def import_baseline_navigation():
    """导入基线 VisionNavigation 与 MaskDataPlus。

    不可用时抛出带指引的 RuntimeError (baseline 后端依赖)。
    """
    add_baseline_to_syspath()
    try:
        from navigation.vision_navigation import MaskDataPlus, VisionNavigation
    except ImportError as exc:
        raise RuntimeError(
            "baseline 后端需要板端基线扩展模块 (不随本仓库分发): "
            "设置环境变量 NAV_BASELINE_LIB 指向含 navigation/ 包的工程根, "
            "或在仿真器工作区同级放置该工程目录; 也可改用 "
            "--backend rowmap_cpp (构建 rowmap_navigation 工程即可)。") from exc
    return VisionNavigation, MaskDataPlus


def import_baseline_fixture():
    """导入基线相机参数夹具 build_camera_info; 不可用时返回 None。"""
    add_baseline_to_syspath()
    try:
        from vision_navigation_tests.benchmark.fixture import build_camera_info
    except ImportError:
        return None
    return build_camera_info


def import_baseline_benchmark():
    """导入基线 benchmark 夹具模块; 不可用时抛出带指引的 RuntimeError。"""
    add_baseline_to_syspath()
    try:
        from vision_navigation_tests.benchmark import fixture
    except ImportError as exc:
        raise RuntimeError(
            "baseline 后端需要板端基线工程的 benchmark 夹具 (不随本仓库分发): "
            "设置环境变量 NAV_BASELINE_LIB 指向含 vision_navigation_tests/ "
            "包的工程根, 或改用 --backend rowmap_cpp。") from exc
    return fixture
