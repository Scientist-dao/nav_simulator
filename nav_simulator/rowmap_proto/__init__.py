"""
rowmap_proto - RowMap 行结构地图导航参考实现

行地图 + 锚点信息融合 + 任务行 ID 锁定 + RTK 融合位姿,
与 C++ 基线 (VisionNavigation) 同台 A/B 的 Python 参考实现。

用法:
    sim_config.nav_backend = "rowmap"    # 默认 "baseline"
    sim = NavSimulator(field_config, sim_config)
"""

from .nav_core import RowMapNavigation
from .row_map import RowMapCore, RowModel, RowAnchor

__all__ = ["RowMapNavigation", "RowMapCore", "RowModel", "RowAnchor"]
