"""
field_model.py - 田间场景几何模型

定义作物行在全局坐标系中的几何位置，支持直线、圆弧、正弦曲线等行形状。
负责将全局坐标转换到车辆坐标系，供 mask 生成器使用。

坐标系约定:
  - 全局坐标系 (World): X=东, Y=北, 原点在田块起点
  - 车辆坐标系 (Vehicle): X=横向(右正), Y=垂直(上正), Z=前向(远正)
    与导航核心 vision_navigation 的约定一致
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class RowShape(Enum):
    """作物行形状类型"""
    STRAIGHT = "straight"        # 直线
    ARC = "arc"                  # 圆弧
    SINE = "sine"                # 正弦曲线
    CLOTHOID = "clothoid"        # 回旋曲线（渐变曲率）


@dataclass
class CropRow:
    """单条作物行的几何定义（全局坐标系）"""
    # 行在全局坐标系中的折线点序列 [(x, y), ...]
    # x: 东向坐标(m), y: 北向坐标(m)
    points: List[Tuple[float, float]] = field(default_factory=list)
    # 行类别: 0=vegetation_line, 1=mulch_line
    class_id: int = 0
    # 行宽度 (m)
    width_m: float = 0.12
    # 是否有效（可用于模拟行缺失）
    active: bool = True

    @property
    def length(self) -> float:
        """行总长度"""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.points) - 1):
            dx = self.points[i+1][0] - self.points[i][0]
            dy = self.points[i+1][1] - self.points[i][1]
            total += np.hypot(dx, dy)
        return total


@dataclass
class FieldConfig:
    """田块配置"""
    num_rows: int = 3                # 行数
    row_spacing_m: float = 0.60      # 行距 (m, 与 1.72m 轮距匹配: 轮心位于行间隙)
    row_length_m: float = 200.0      # 行长度 (m)
    row_width_m: float = 0.12        # 单行宽度 (m)
    row_shape: RowShape = RowShape.STRAIGHT
    # 圆弧参数 (仅 ARC)
    arc_radius_m: float = 50.0       # 圆弧半径 (m)
    # 正弦参数 (仅 SINE)
    sine_amplitude_m: float = 0.5    # 振幅 (m)
    sine_period_m: float = 30.0      # 周期 (m)
    # 回旋曲线参数 (仅 CLOTHOID)
    clothoid_curvature_rate: float = 0.002  # 曲率变化率 (1/m²)
    # 行端截断 (模拟行尾): 非 None 时, 行在此 y 坐标处截断
    row_end_y: Optional[float] = None
    # 缺失行索引列表
    missing_rows: List[int] = field(default_factory=list)
    # 行类别 (所有行使用同一类别)
    class_id: int = 0


class FieldModel:
    """
    田间场景模型

    管理作物行的全局几何定义，提供车辆坐标系变换接口。
    车辆沿 Y 轴（北向）行驶，作物行沿 Y 方向延伸。
    """

    def __init__(self, config: FieldConfig):
        self.config = config
        self.rows: List[CropRow] = []
        self._build_rows()

    def _build_rows(self):
        """根据配置生成作物行"""
        self.rows.clear()
        cfg = self.config
        x_start = -(cfg.num_rows - 1) * cfg.row_spacing_m / 2.0

        for i in range(cfg.num_rows):
            x_base = x_start + i * cfg.row_spacing_m
            active = i not in cfg.missing_rows

            if cfg.row_shape == RowShape.STRAIGHT:
                points = self._gen_straight_row(x_base, cfg.row_length_m)
            elif cfg.row_shape == RowShape.ARC:
                points = self._gen_arc_row(x_base, cfg.row_length_m, cfg.arc_radius_m)
            elif cfg.row_shape == RowShape.SINE:
                points = self._gen_sine_row(
                    x_base, cfg.row_length_m,
                    cfg.sine_amplitude_m, cfg.sine_period_m)
            elif cfg.row_shape == RowShape.CLOTHOID:
                points = self._gen_clothoid_row(
                    x_base, cfg.row_length_m, cfg.clothoid_curvature_rate)
            else:
                points = self._gen_straight_row(x_base, cfg.row_length_m)

            # 行端截断
            if cfg.row_end_y is not None:
                points = [(x, y) for x, y in points if y <= cfg.row_end_y]

            row = CropRow(
                points=points,
                class_id=cfg.class_id,
                width_m=cfg.row_width_m,
                active=active,
            )
            self.rows.append(row)

    @staticmethod
    def _gen_straight_row(x_base: float, length: float,
                          step: float = 0.5) -> List[Tuple[float, float]]:
        """生成直线行"""
        ys = np.arange(0, length + step, step)
        return [(x_base, float(y)) for y in ys]

    @staticmethod
    def _gen_arc_row(x_base: float, length: float,
                     radius: float, step: float = 0.5
                     ) -> List[Tuple[float, float]]:
        """生成圆弧行 (向左弯曲, radius > 0; 向右弯曲, radius < 0)"""
        ys = np.arange(0, length + step, step)
        points = []
        for y in ys:
            # 圆弧: x = x_base + R - sqrt(R² - y²) (近似小角度)
            r = abs(radius)
            if r > abs(y):
                dx = r - np.sqrt(max(0.0, r * r - y * y))
            else:
                dx = r  # 超过 90 度时取最大偏移
            if radius < 0:
                dx = -dx
            points.append((x_base + dx, float(y)))
        return points

    @staticmethod
    def _gen_sine_row(x_base: float, length: float,
                      amplitude: float, period: float,
                      step: float = 0.5) -> List[Tuple[float, float]]:
        """生成正弦曲线行"""
        ys = np.arange(0, length + step, step)
        points = []
        for y in ys:
            dx = amplitude * np.sin(2.0 * np.pi * y / period)
            points.append((x_base + dx, float(y)))
        return points

    @staticmethod
    def _gen_clothoid_row(x_base: float, length: float,
                          curvature_rate: float,
                          step: float = 0.5) -> List[Tuple[float, float]]:
        """生成回旋曲线行 (曲率线性变化)"""
        ys = np.arange(0, length + step, step)
        points = []
        x_offset = 0.0
        heading = 0.0
        prev_y = 0.0
        for y in ys:
            ds = y - prev_y
            curvature = curvature_rate * y
            heading += curvature * ds
            x_offset += np.sin(heading) * ds
            points.append((x_base + x_offset, float(y)))
            prev_y = y
        return points

    def get_visible_rows_vehicle(
            self, veh_x: float, veh_y: float, veh_heading: float,
            z_near: float = 0.5, z_far: float = 20.0,
            x_range: float = 3.0,
            ) -> List[Tuple[int, List[Tuple[float, float]], int, float]]:
        """
        获取车辆坐标系下可见的作物行。

        Args:
            veh_x, veh_y: 车辆全局位置
            veh_heading: 车辆朝向（弧度，0 = 北向，顺时针为正，罗盘角；
                与 vehicle_model 运动学 / camera_renderer / C++ nav_geo.hpp
                的约定一致）
            z_near: 最近可见距离 (m)
            z_far: 最远可见距离 (m)
            x_range: 横向可见范围 (m)

        Returns:
            List of (row_index, points_in_vehicle, class_id, width_m)
            points_in_vehicle: [(X_v, Z_v), ...] 车辆坐标系中的点序列
        """
        # 世界→车辆系旋转 (罗盘角 heading, 顺时针正):
        #   vx = dE·cos(h) - dN·sin(h)
        #   vz = dE·sin(h) + dN·cos(h)
        # 为 nav_geo.hpp vcs_xz_to_enu 的正交逆变换。旧版用 cos(-h)/sin(-h)
        # (数学角) 旋转, 与运动学/ENU 链路符号相反: 位置 CTE 一阶正确
        # (小角度), 但行斜率镜像 → C++ yaw 符号反向 → 航向阻尼变正反馈
        # (蛇形摆动), mode2 ENU 线缓存非物理旋转 → FSC 评分错乱振荡。
        cos_h = np.cos(veh_heading)
        sin_h = np.sin(veh_heading)

        result = []
        for idx, row in enumerate(self.rows):
            if not row.active or len(row.points) < 2:
                continue

            # 将全局坐标转换到车辆坐标系
            pts_vehicle = []
            for gx, gy in row.points:
                # 平移到车辆为原点
                dx = gx - veh_x
                dy = gy - veh_y
                # 旋转到车辆坐标系 (车辆坐标: X=横向右正, Z=前向远正)
                # 注意: 全局 Y(北) 对应车辆 Z(前), 全局 X(东) 对应车辆 X(右)
                vx = dx * cos_h - dy * sin_h
                vz = dx * sin_h + dy * cos_h

                if z_near <= vz <= z_far and abs(vx) <= x_range:
                    pts_vehicle.append((vx, vz))

            if len(pts_vehicle) >= 2:
                result.append((idx, pts_vehicle, row.class_id, row.width_m))

        return result

    def get_all_rows_global(self) -> List[CropRow]:
        """返回所有行的全局坐标（用于俯视图渲染）"""
        return self.rows

    def get_row_x_at_y(self, row_idx: int, y: float) -> Optional[float]:
        """获取指定行在全局 y 坐标处的 x 位置"""
        if row_idx < 0 or row_idx >= len(self.rows):
            return None
        row = self.rows[row_idx]
        if not row.points:
            return None

        # 线性插值
        pts = row.points
        for i in range(len(pts) - 1):
            if pts[i][1] <= y <= pts[i + 1][1]:
                t = (y - pts[i][1]) / max(1e-6, pts[i + 1][1] - pts[i][1])
                return pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
        return None

    @property
    def center_row_index(self) -> int:
        """中心行索引"""
        return self.config.num_rows // 2

    def get_center_x_at_y(self, y: float) -> float:
        """获取中心行在全局 y 处的 x 坐标（用于计算初始偏移）"""
        x = self.get_row_x_at_y(self.center_row_index, y)
        return x if x is not None else 0.0
