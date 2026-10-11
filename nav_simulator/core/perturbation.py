"""
perturbation.py - 感知扰动注入模块 (P0 仿真基建)

为 RowMap 新算法对照实验提供三类生产痛点注入 (视觉导航V2开发文档 §4.2):

  1. OcclusionWindow  时变遮挡   杂草/尘土挡镜头 → 遮挡窗内 mask 全部丢弃
  2. RowGapRegion     空间缺苗   断垄/出苗不全  → 世界 y 区间内行点删除
  3. WeakRegion       弱苗区域   苗期细弱       → score 降低 + 位置抖动 + 行宽缩小

注入层位于 field_model.get_visible_rows_vehicle() 与 MaskGenerator 之间,
即"感知前的世界状态"层: 扰动作用于行的可见几何, 而非事后篡改 mask,
保持与真实感知退化 (相机看不到/田里没有/看不清) 的物理对应。

三种注入共用确定性随机源 (seed), reset() 重建保证逐帧可复现。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# 可见行元组: (row_idx, [(X_v, Z_v), ...], class_id, width_m)
VisibleRow = Tuple[int, List[Tuple[float, float]], int, float]


@dataclass
class OcclusionWindow:
    """时变遮挡窗: [start_s, start_s+duration_s) 内感知输入被完全遮挡

    模拟杂草丛/高作物遮挡镜头、镜头沾土等场景。
    rows=None 遮挡全部行 (完全失明); 指定索引列表可做部分遮挡。
    """
    start_s: float
    duration_s: float
    rows: Optional[List[int]] = None

    def covers(self, sim_time: float) -> bool:
        return self.start_s <= sim_time < self.start_s + self.duration_s


@dataclass
class RowGapRegion:
    """空间缺苗区: 世界 y ∈ [y_start, y_end] 内无苗 (断垄)

    车辆接近时表现为行断裂: 近端段先消失, 远端段保留,
    完全失明仅在区间长于视野 (z_far) 时出现。rows=None 对全部行生效。
    """
    y_start: float
    y_end: float
    rows: Optional[List[int]] = None


@dataclass
class WeakRegion:
    """弱苗区: 世界 y ∈ [y_start, y_end] 内苗带细弱

    score_override: 检测置信度下降 (模拟弱纹理/低对比度)
    jitter_std_m:   行点横向位置噪声 (模拟残缺苗形, 逐帧独立)
    width_scale:    行宽缩放 (细苗带)
    行与区域相交即整行弱化 (行级属性, 简化建模)。
    """
    y_start: float
    y_end: float
    score_override: float = 0.55
    jitter_std_m: float = 0.03
    width_scale: float = 0.6
    rows: Optional[List[int]] = None


@dataclass
class PerturbationConfig:
    """感知扰动注入配置 (SimConfig.perturbation)"""
    occlusions: List[OcclusionWindow] = field(default_factory=list)
    gaps: List[RowGapRegion] = field(default_factory=list)
    weak_regions: List[WeakRegion] = field(default_factory=list)
    seed: Optional[int] = None      # 弱苗抖动随机种子 (可复现)
    base_score: float = 0.95        # 无扰动时默认 mask 置信度


class PerturbationApplier:
    """对可见行列表施加扰动, 输出注入后的行 + 每行 score 覆盖"""

    def __init__(self, config: PerturbationConfig):
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)
        self.reset()

    def reset(self):
        """复位随机源 (仿真 reset 后逐帧序列完全可复现)"""
        self._rng = np.random.default_rng(self.cfg.seed)

    # ── 车辆系 → 世界系 (罗盘角 heading, 与 simulator._nav_line_to_world 一致) ──
    @staticmethod
    def _to_world(xv: float, zv: float, veh_x: float, veh_y: float,
                  heading_rad: float) -> Tuple[float, float]:
        c, s = np.cos(heading_rad), np.sin(heading_rad)
        return (veh_x + xv * c + zv * s,
                veh_y - xv * s + zv * c)

    def occluded_now(self, sim_time: float) -> bool:
        """本帧是否存在完全遮挡 (全部行被遮挡)"""
        for w in self.cfg.occlusions:
            if w.covers(sim_time) and w.rows is None:
                return True
        return False

    def apply(self, visible_rows: List[VisibleRow],
              veh_x: float, veh_y: float, veh_heading: float,
              sim_time: float
              ) -> List[Tuple[VisibleRow, Optional[float]]]:
        """施加全部扰动

        Returns:
            [(row, score_override), ...]  score_override=None 表示用默认 score
        """
        out: List[Tuple[VisibleRow, Optional[float]]] = []

        for row in visible_rows:
            row_idx, pts_vehicle, class_id, width_m = row

            # ── 1. 时变遮挡: 命中窗的行整行丢弃 ──
            if self._occluded(row_idx, sim_time):
                continue

            # ── 2. 空间缺苗: 世界 y 落入 gap 区间的点删除 ──
            pts_filtered = self._apply_gaps(
                row_idx, pts_vehicle, veh_x, veh_y, veh_heading)
            if len(pts_filtered) < 2:
                continue    # 剩余点不足以构成行特征

            # ── 3. 弱苗区域: 抖动 + 行宽缩 + score 覆盖 ──
            pts_final, width_final, score_override = self._apply_weak(
                row_idx, pts_filtered, width_m,
                veh_x, veh_y, veh_heading)

            out.append(((row_idx, pts_final, class_id, width_final),
                        score_override))

        return out

    def _occluded(self, row_idx: int, sim_time: float) -> bool:
        """行是否被任一遮挡窗覆盖"""
        for w in self.cfg.occlusions:
            if w.covers(sim_time):
                if w.rows is None or row_idx in w.rows:
                    return True
        return False

    def _apply_gaps(self, row_idx: int, pts_vehicle: List[Tuple[float, float]],
                    veh_x: float, veh_y: float, veh_heading: float
                    ) -> List[Tuple[float, float]]:
        """删除落入缺苗区 (世界 y 区间) 的行点"""
        if not self.cfg.gaps:
            return pts_vehicle

        kept = []
        for xv, zv in pts_vehicle:
            _, wy = self._to_world(xv, zv, veh_x, veh_y, veh_heading)
            in_gap = False
            for g in self.cfg.gaps:
                if g.y_start <= wy <= g.y_end:
                    if g.rows is None or row_idx in g.rows:
                        in_gap = True
                        break
            if not in_gap:
                kept.append((xv, zv))
        return kept

    def apply_physical(self, visible_rows: List[VisibleRow],
                        veh_x: float, veh_y: float, veh_heading: float
                        ) -> List[VisibleRow]:
        """物理现实层: 仅空间性扰动 (gap 删点 + weak 缩行宽)

        不含时变遮挡与位置抖动 (两者属于感知退化, 不是田里真实长势):
        相机背景渲染与俯视图用它, 让画面反映"田里实际长了什么"——
        断垄区画空土、弱苗区画细苗带, 而行位置保持真值,
        与带抖动的 mask 对比即可直观测出感知偏差。
        """
        out: List[VisibleRow] = []
        for row in visible_rows:
            row_idx, pts_vehicle, class_id, width_m = row
            pts_filtered = self._apply_gaps(
                row_idx, pts_vehicle, veh_x, veh_y, veh_heading)
            if len(pts_filtered) < 2:
                continue
            weak = self._find_weak(
                row_idx, pts_filtered, veh_x, veh_y, veh_heading)
            width_final = width_m * weak.width_scale if weak else width_m
            out.append((row_idx, pts_filtered, class_id, width_final))
        return out

    def active_tags(self, sim_time: float) -> str:
        """当前激活的扰动短标签 (HUD/仪表盘显示用, ASCII)

        遮挡窗显示剩余秒数; gap/weak 为静态区域, 场景配置了即常显。
        """
        tags: List[str] = []
        for w in self.cfg.occlusions:
            if w.covers(sim_time):
                remain = w.start_s + w.duration_s - sim_time
                tags.append(f"OCCL {remain:.1f}s")
        if self.cfg.gaps:
            tags.append("GAP")
        if self.cfg.weak_regions:
            tags.append("WEAK")
        return "+".join(tags)

    def seg_in_gap(self, row_idx: int, y1: float, y2: float) -> bool:
        """世界坐标行段 (y1,y2) 是否落入缺苗区 (俯视图行绘制过滤用)"""
        for g in self.cfg.gaps:
            if g.rows is not None and row_idx not in g.rows:
                continue
            lo, hi = min(y1, y2), max(y1, y2)
            if lo <= g.y_end and hi >= g.y_start:
                return True
        return False

    def _find_weak(self, row_idx: int, pts_vehicle: List[Tuple[float, float]],
                   veh_x: float, veh_y: float, veh_heading: float
                   ) -> Optional[WeakRegion]:
        """查找与该行相交的弱苗区 (无则 None)"""
        for wr in self.cfg.weak_regions:
            for xv, zv in pts_vehicle:
                _, wy = self._to_world(xv, zv, veh_x, veh_y, veh_heading)
                if wr.y_start <= wy <= wr.y_end:
                    if wr.rows is None or row_idx in wr.rows:
                        return wr
        return None

    def _apply_weak(self, row_idx: int, pts_vehicle: List[Tuple[float, float]],
                    width_m: float, veh_x: float, veh_y: float,
                    veh_heading: float
                    ) -> Tuple[List[Tuple[float, float]], float, Optional[float]]:
        """弱苗区: 相交行施加位置抖动/行宽缩放/score 覆盖

        返回 (点列表, 行宽, score 覆盖或 None)
        """
        weak = self._find_weak(
            row_idx, pts_vehicle, veh_x, veh_y, veh_heading)

        if weak is None:
            return pts_vehicle, width_m, None

        # 逐点横向抖动 (车辆系 x 方向, 模拟苗形残缺; 逐帧独立噪声)
        pts_jittered = [
            (xv + float(self._rng.normal(0.0, weak.jitter_std_m)), zv)
            for xv, zv in pts_vehicle
        ]
        width_scaled = width_m * weak.width_scale
        return pts_jittered, width_scaled, weak.score_override
