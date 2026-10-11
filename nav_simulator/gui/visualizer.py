"""
visualizer.py - 仿真器可视化

提供两种显示模式:
  1. 单窗口仪表盘 (默认): Dashboard 把相机视图 / 俯视图 / 遥测 /
     传感器状态 / 历史曲线组装到一张画布, 信息密度高、布局稳定
  2. 经典双窗口 (show_dashboard=False): Camera View + Top View 两个
     独立窗口 (低分辨率屏幕 / 对照旧行为用)

TopViewRenderer 被两种模式复用:
  - 仪表盘模式: render(..., show_hud=False), 标题/HUD 由 Dashboard 统一绘制
  - 经典模式:   render(..., show_hud=True), 自带标题栏/姿态仪/图例
配色统一取自 theme.py (深色科技风)。
"""

import cv2
import numpy as np
from typing import List, Tuple

from ..core.simulator import FrameResult, NavSimulator
from ..core.field_model import FieldModel
from ..rendering.hud_renderer import draw_hud
from . import theme as T


def _sigma_color(sig: float) -> Tuple[int, int, int]:
    """锚点置信度分档色 (BGR): 绿<2cm / 黄<5cm / 红≥5cm"""
    if sig < 0.02:
        return (80, 220, 80)      # 绿: 高置信
    if sig < 0.05:
        return (40, 200, 230)     # 黄: 中置信
    return (60, 60, 230)          # 红: 低置信 (远端新锚点)


class TopViewRenderer:
    """俯视图渲染器 (深色田块主题, 精细车辆模型 + 运动学可视化)

    保持接口稳定: 无参构造 + render 的 9 个位置参数
    (外部验证脚本依赖此签名)。
    """

    def __init__(self, width: int = 520, height: int = 720,
                 scale: float = 55.0, center_back_m: float = 2.0):
        self.w = width
        self.h = height
        self.scale = scale
        # 视野中心在车辆后方多少米 (决定车前方可见纵深: 越大看得越远)
        # RowMap 锚点红黄区在 z=10~20m, 内嵌小图用大值保证 σ 分级可见
        self.center_back_m = center_back_m
        self.trail: List[Tuple[float, float]] = []
        self.max_trail = 800
        # 颠簸状态 (纯可视化扰动, 不影响导航输入)
        self._bump_phase = 0.0
        self._bump_amp = 0.0

    def _world_to_pixel(self, wx: float, wy: float,
                        cx: float, cy: float) -> Tuple[int, int]:
        px = int(self.w / 2 + (wx - cx) * self.scale)
        py = int(self.h - 60 - (wy - cy) * self.scale)
        return (px, py)

    def render(self, field_model: FieldModel,
               veh_x: float, veh_y: float, veh_heading: float,
               steering_rad: float = 0.0,
               speed_ms: float = 0.0,
               nav_line_world=None,
               roll_rad: float = 0.0,
               pitch_rad: float = 0.0,
               show_hud: bool = True,
               perturbation=None,
               occluded: bool = False,
               map_overlay: dict = None) -> np.ndarray:
        """渲染一帧俯视图

        show_hud=False 时不画标题栏/底部信息/姿态仪 (嵌入仪表盘时由
        Dashboard 统一提供面板标题与遥测数据)。
        perturbation: PerturbationApplier 实例——缺苗区行段不画、
        区域横带标注 (GAP/WEAK), 与相机背景反映同一物理现实
        occluded: 本帧感知被遮挡, 相机视野梯形变红示警
        map_overlay: RowMap 后端的行地图数据 (get_map_overlay() 输出),
        画锚点折线 + 置信度着色 (None = 基线后端不画)
        """
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 地面背景（田块纹理）
        self._draw_field_background(img)

        cx, cy = veh_x, veh_y - self.center_back_m

        # 网格
        self._draw_grid(img, cx, cy)

        # 作物行 (按真实行宽 0.12m 绘制, 便于观察轮-行间隙;
        # 缺苗区内的行段不画——地里没苗就不显示)
        row_half_w = 0.06  # 行半宽 (m)
        for row_idx, row in enumerate(field_model.get_all_rows_global()):
            if not row.active:
                continue
            for i in range(len(row.points) - 1):
                x1, y1 = row.points[i]
                x2, y2 = row.points[i + 1]
                if (perturbation is not None
                        and perturbation.seg_in_gap(row_idx, y1, y2)):
                    continue
                # 行方向法向量 (横向偏移 ±row_half_w 构建条带)
                dx, dy = x2 - x1, y2 - y1
                seg_len = np.hypot(dx, dy)
                if seg_len < 1e-6:
                    continue
                nx, ny = -dy / seg_len * row_half_w, dx / seg_len * row_half_w
                poly = np.array([
                    self._world_to_pixel(x1 + nx, y1 + ny, cx, cy),
                    self._world_to_pixel(x2 + nx, y2 + ny, cx, cy),
                    self._world_to_pixel(x2 - nx, y2 - ny, cx, cy),
                    self._world_to_pixel(x1 - nx, y1 - ny, cx, cy),
                ], dtype=np.int32)
                if np.any((poly[:, 0] >= 0) & (poly[:, 0] < self.w)):
                    cv2.fillPoly(img, [poly], T.CROP_ROW)
                    cv2.polylines(img, [poly], True, T.CROP_ROW_HI, 1,
                                  cv2.LINE_AA)

        # 缺苗/弱苗区域横带标注 (世界 y 区间, 俯视图无航向旋转呈水平带)
        if perturbation is not None:
            self._draw_perturbation_zones(img, perturbation, cx, cy)

        # 相机视野梯形 (遮挡时红色示警)
        self._draw_camera_fov(img, veh_x, veh_y, veh_heading, cx, cy,
                              occluded=occluded)

        # 导航线 (世界坐标, 与 Camera View 的红线对应)
        if nav_line_world and len(nav_line_world) >= 2:
            p1 = self._world_to_pixel(*nav_line_world[0], cx, cy)
            p2 = self._world_to_pixel(*nav_line_world[1], cx, cy)
            cv2.line(img, p1, p2, T.NAV_LINE, 2, cv2.LINE_AA)

        # 行结构地图 (RowMap 后端): 锚点折线 + 置信度着色
        if map_overlay is not None:
            self._draw_row_map(img, map_overlay, cx, cy)

        # 轨迹（渐隐）
        self._draw_trail(img, cx, cy)

        # 颠簸扰动
        bump_dx, bump_dy = self._compute_bump(speed_ms)

        # 车辆（精细模型 + 转向 + 颠簸）
        self._draw_vehicle(img, veh_x + bump_dx, veh_y + bump_dy,
                           veh_heading, steering_rad, cx, cy)

        # HUD (独立窗口模式才画; 仪表盘内嵌时由 Dashboard 统一绘制)
        if show_hud:
            self._draw_top_hud(img, veh_x, veh_y, veh_heading, speed_ms)
            self._draw_attitude_indicator(img, roll_rad, pitch_rad)

        return img

    def _draw_field_background(self, img):
        """田块地面背景 (深青灰底 + 固定种子斑块纹理, 不随帧闪烁)"""
        img[:] = T.FIELD_BG
        h, w = img.shape[:2]
        rng = np.random.RandomState(42)
        palette = [(23, 28, 21), (29, 35, 26), (32, 39, 28)]
        for _ in range(50):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            r = rng.randint(5, 25)
            c = palette[rng.randint(len(palette))]
            cv2.circle(img, (x, y), r, c, -1)

    def _draw_grid(self, img, cx, cy):
        """网格和距离刻度"""
        grid_spacing = 1.0
        major_spacing = 5.0

        x_min = cx - self.w / (2 * self.scale)
        x_max = cx + self.w / (2 * self.scale)
        y_min = cy - self.h / (2 * self.scale)
        y_max = cy + self.h / (2 * self.scale)

        x = int(np.ceil(x_min / grid_spacing)) * grid_spacing
        while x <= x_max:
            p1 = self._world_to_pixel(x, y_min, cx, cy)
            p2 = self._world_to_pixel(x, y_max, cx, cy)
            is_major = (abs(x % major_spacing) < 0.01)
            cv2.line(img, p1, p2,
                     T.FIELD_GRID_MJ if is_major else T.FIELD_GRID,
                     1, cv2.LINE_AA)
            if is_major:
                cv2.putText(img, f"{x:.0f}",
                            (p1[0] + 2, self.h - 65),
                            T.FONT, 0.3, T.TEXT_LOW, 1, cv2.LINE_AA)
            x += grid_spacing

        y = int(np.ceil(y_min / grid_spacing)) * grid_spacing
        while y <= y_max:
            p1 = self._world_to_pixel(x_min, y, cx, cy)
            p2 = self._world_to_pixel(x_max, y, cx, cy)
            is_major = (abs(y % major_spacing) < 0.01)
            cv2.line(img, p1, p2,
                     T.FIELD_GRID_MJ if is_major else T.FIELD_GRID,
                     1, cv2.LINE_AA)
            if is_major:
                cv2.putText(img, f"{y:.0f}m", (5, p1[1] - 3),
                            T.FONT, 0.3, T.TEXT_LOW, 1, cv2.LINE_AA)
            y += grid_spacing

    def _draw_camera_fov(self, img, vx, vy, heading, cx, cy,
                         occluded: bool = False):
        """相机视野范围 (遮挡时红色示警: 视野在但什么都拍不到)"""
        z_near, z_far = 0.5, 15.0
        x_half_near, x_half_far = 1.5, 3.0

        fov_local = [
            (-x_half_near, z_near), (x_half_near, z_near),
            (x_half_far, z_far), (-x_half_far, z_far),
        ]

        cos_h, sin_h = np.cos(heading), np.sin(heading)
        fov_px = []
        for lat, lon in fov_local:
            gx = vx + lat * cos_h + lon * sin_h
            gy = vy - lat * sin_h + lon * cos_h
            fov_px.append(self._world_to_pixel(gx, gy, cx, cy))

        fov_px = np.array(fov_px, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [fov_px], T.FOV_FILL)
        cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
        edge_color = T.FOV_EDGE
        edge_thick = 1
        if occluded:
            # 遮挡: 视野梯形红化 + 加粗红边 + OCCL 标签
            ov_red = img.copy()
            cv2.fillPoly(ov_red, [fov_px], (60, 60, 210))
            cv2.addWeighted(ov_red, 0.35, img, 0.65, 0, img)
            edge_color = (50, 50, 235)
            edge_thick = 2
            top_px = min(fov_px.tolist(), key=lambda p: p[1])
            cv2.putText(img, "OCCL", (top_px[0] - 22, top_px[1] - 6),
                        T.FONT, 0.38, (50, 50, 235), 1, cv2.LINE_AA)
        cv2.polylines(img, [fov_px], True, edge_color, edge_thick,
                      cv2.LINE_AA)

    def _draw_perturbation_zones(self, img, perturbation, cx, cy):
        """缺苗/弱苗区域横带标注 (P0 注入可视化)

        俯视图世界坐标直接映射屏幕 (无航向旋转), 世界 y 区间
        呈水平条带: GAP 暗棕带 (地里没苗), WEAK 黄带 (苗弱)。
        """
        h, w = img.shape[:2]

        def band_rows(y_world):
            py = int(self.h - 60 - (y_world - cy) * self.scale)
            return py

        for g in perturbation.cfg.gaps:
            py_far = band_rows(g.y_end)     # 远端 y 大 → 屏幕靠上
            py_near = band_rows(g.y_start)  # 近端 → 屏幕靠下
            top, bot = max(0, min(py_far, py_near)), min(h, max(py_far, py_near))
            if bot <= 0 or top >= h:
                continue
            overlay = img.copy()
            cv2.rectangle(overlay, (0, top), (w, bot), (48, 40, 72), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.line(img, (0, top), (w, top), (75, 62, 115), 1, cv2.LINE_AA)
            cv2.line(img, (0, bot), (w, bot), (75, 62, 115), 1, cv2.LINE_AA)
            cv2.putText(img, "GAP", (6, (top + bot) // 2 + 4),
                        T.FONT, 0.38, (115, 98, 170), 1, cv2.LINE_AA)

        for wr in perturbation.cfg.weak_regions:
            py_far = band_rows(wr.y_end)
            py_near = band_rows(wr.y_start)
            top, bot = max(0, min(py_far, py_near)), min(h, max(py_far, py_near))
            if bot <= 0 or top >= h:
                continue
            overlay = img.copy()
            cv2.rectangle(overlay, (0, top), (w, bot), (60, 190, 190), -1)
            cv2.addWeighted(overlay, 0.16, img, 0.84, 0, img)
            cv2.line(img, (0, top), (w, top), (80, 210, 210), 1, cv2.LINE_AA)
            cv2.line(img, (0, bot), (w, bot), (80, 210, 210), 1, cv2.LINE_AA)
            cv2.putText(img, "WEAK", (6, (top + bot) // 2 + 4),
                        T.FONT, 0.38, (90, 225, 225), 1, cv2.LINE_AA)

    def _draw_row_map(self, img, map_overlay: dict, cx, cy):
        """行结构地图可视化 (RowMap 后端): 锚点折线 + 置信度着色

        任务行亮绿粗线, 邻行青色细线; 锚点颜色按融合后 σ 分级
        (绿 <2cm / 黄 <5cm / 红 ≥5cm) —— 直观呈现"越走越准":
        远端新锚点 (红) 随观测次数增加逐帧变黄变绿。
        RTK 位姿估计画青色十字, 与真值车的偏移即 RTK 噪声可视化。
        """
        for row in map_overlay.get("rows", []):
            anchors = row["anchors"]
            pts = [self._world_to_pixel(wx, wy, cx, cy)
                   for wx, wy, _sig, _n in anchors]
            is_task = row["task"]
            # 锚点连线按 σ 分段着色 (段色=两端锚点较差者): 任务行呈
            # "根部绿→梢部黄红"渐变色带, 远端低置信一眼可辨
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    if is_task:
                        seg_sig = max(anchors[i][2], anchors[i + 1][2])
                        color = _sigma_color(seg_sig)
                        thick = 2
                    else:
                        color = (200, 150, 60)   # 邻行固定青细
                        thick = 1
                    cv2.line(img, pts[i], pts[i + 1], color, thick,
                             cv2.LINE_AA)
            # 锚点置信度着色 (σ=√var)
            r_pt = 4 if is_task else 2
            for (wx, wy, sig, n), p in zip(anchors, pts):
                cv2.circle(img, p, r_pt, _sigma_color(sig), -1,
                           cv2.LINE_AA)

        # RTK 位姿估计 (青十字): 与车真值位置的偏移 = RTK 噪声
        pos = map_overlay.get("pos")
        if pos is not None:
            p = self._world_to_pixel(pos[0], pos[1], cx, cy)
            cv2.drawMarker(img, p, (230, 200, 60), cv2.MARKER_CROSS,
                           12, 1, cv2.LINE_AA)

        # 图例 (底部, 与距离刻度错开)
        h, w = img.shape[:2]
        cv2.putText(img, "ROWMAP  line+anchor sigma: green<2cm "
                    "yellow<5cm red>=5cm | + = RTK pose",
                    (6, h - 72), T.FONT, 0.32, (170, 195, 170), 1,
                    cv2.LINE_AA)

    def _draw_trail(self, img, cx, cy):
        """轨迹（渐隐: 越新越亮）"""
        if len(self.trail) < 2:
            return

        n = len(self.trail)
        for i in range(n - 1):
            alpha = 0.25 + 0.75 * (i / n)
            color = tuple(int(c * alpha) for c in T.TRAIL)
            p1 = self._world_to_pixel(*self.trail[i], cx, cy)
            p2 = self._world_to_pixel(*self.trail[i + 1], cx, cy)
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)

    def _compute_bump(self, speed_ms: float) -> Tuple[float, float]:
        """计算颠簸扰动（小幅随机 + 正弦, 仅可视化用）"""
        if speed_ms < 0.1:
            return 0.0, 0.0

        self._bump_phase += 0.15 * speed_ms
        # 正弦颠簸（模拟不平地面）
        bump_y = 0.015 * np.sin(self._bump_phase * 3.7)
        bump_x = 0.008 * np.sin(self._bump_phase * 5.3)
        # 小幅随机扰动
        bump_x += np.random.normal(0, 0.003)
        bump_y += np.random.normal(0, 0.005)

        return bump_x, bump_y

    def _draw_vehicle(self, img, vx, vy, heading,
                      steering_rad, cx, cy):
        """精细车辆模型（车架 + 驾驶室 + 四轮独立转向 + 方向指示）"""
        half_len = 2.7 / 2.0
        half_wid = 1.72 / 2.0

        cos_h, sin_h = np.cos(heading), np.sin(heading)
        R = np.array([[cos_h, sin_h], [-sin_h, cos_h]])
        offset = np.array([vx, vy])

        # ── 车架（主体） ──
        body_margin = 0.08  # 车架比轮子小一圈
        corners_local = np.array([
            [-half_wid + body_margin, half_len],
            [half_wid - body_margin, half_len],
            [half_wid - body_margin, -half_len],
            [-half_wid + body_margin, -half_len],
        ])
        corners_global = corners_local @ R.T + offset
        corners_px = np.array(
            [self._world_to_pixel(gx, gy, cx, cy)
             for gx, gy in corners_global], dtype=np.int32)

        cv2.fillPoly(img, [corners_px], T.VEH_BODY)
        cv2.polylines(img, [corners_px], True, T.VEH_EDGE, 2, cv2.LINE_AA)

        # ── 驾驶室（车体上部 1/3 区域，稍深色） ──
        cab_local = np.array([
            [-half_wid + body_margin + 0.1, half_len - 0.3],
            [half_wid - body_margin - 0.1, half_len - 0.3],
            [half_wid - body_margin - 0.1, half_len * 0.2],
            [-half_wid + body_margin + 0.1, half_len * 0.2],
        ])
        cab_global = cab_local @ R.T + offset
        cab_px = np.array(
            [self._world_to_pixel(gx, gy, cx, cy)
             for gx, gy in cab_global], dtype=np.int32)
        cv2.fillPoly(img, [cab_px], (200, 105, 22))  # 车身色加深一档

        # ── 轮子 ──
        wheel_w = 0.18  # 轮子宽度 (m)
        wheel_l = 0.55  # 轮子长度 (m)
        wheel_positions = [
            (-half_wid, half_len * 0.65),   # 左前
            (half_wid, half_len * 0.65),    # 右前
            (-half_wid, -half_len * 0.65),  # 左后
            (half_wid, -half_len * 0.65),   # 右后
        ]

        for i, (wl, wo) in enumerate(wheel_positions):
            is_front = (i < 2)
            # 前轮应用转向角
            if is_front:
                wheel_heading = heading + steering_rad
            else:
                wheel_heading = heading

            cos_w, sin_w = np.cos(wheel_heading), np.sin(wheel_heading)
            R_w = np.array([[cos_w, sin_w], [-sin_w, cos_w]])

            # 轮子角点（轮子局部坐标）
            w_corners = np.array([
                [-wheel_w, wheel_l],
                [wheel_w, wheel_l],
                [wheel_w, -wheel_l],
                [-wheel_w, -wheel_l],
            ]) / 2.0

            # 轮子中心的全局位置
            wc_global = R @ np.array([wl, wo]) + offset
            # 轮子角点：先在轮子坐标系旋转，再平移到全局
            w_global = w_corners @ R_w.T + wc_global

            w_px = np.array(
                [self._world_to_pixel(gx, gy, cx, cy)
                 for gx, gy in w_global], dtype=np.int32)
            cv2.fillPoly(img, [w_px], (45, 45, 50))
            cv2.polylines(img, [w_px], True, (78, 78, 85), 1, cv2.LINE_AA)

        # ── 前轮转向指示线 ──
        if abs(steering_rad) > 0.005:
            for side in [-1, 1]:
                wc_local = np.array([side * half_wid, half_len * 0.65])
                wc_g = R @ wc_local + offset
                wc_px = self._world_to_pixel(wc_g[0], wc_g[1], cx, cy)

                # 转向方向线
                steer_dir = heading + steering_rad
                line_len = 0.8
                end_x = wc_g[0] + np.sin(steer_dir) * line_len
                end_y = wc_g[1] + np.cos(steer_dir) * line_len
                end_px = self._world_to_pixel(end_x, end_y, cx, cy)

                # 转向角越大越红 (黄→红渐变)
                intensity = min(1.0, abs(steering_rad) / 0.3)
                color = tuple(int(a + (b - a) * intensity)
                              for a, b in zip(T.WARN, T.ERR))
                cv2.line(img, wc_px, end_px, color, 2, cv2.LINE_AA)

        # ── 朝向箭头 ──
        center_px = self._world_to_pixel(vx, vy, cx, cy)
        front_x = vx + np.sin(heading) * 2.5
        front_y = vy + np.cos(heading) * 2.5
        front_px = self._world_to_pixel(front_x, front_y, cx, cy)
        cv2.arrowedLine(img, center_px, front_px, T.WARN, 2,
                        tipLength=0.25, line_type=cv2.LINE_AA)

    def _draw_attitude_indicator(self, img, roll_rad, pitch_rad):
        """车身姿态指示器 (右上角, 简化姿态仪)"""
        box_w, box_h = 90, 64
        bx, by = self.w - box_w - 10, 50

        # 半透明底
        overlay = img.copy()
        cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h),
                      T.PANEL, -1)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h),
                      T.EDGE, 1)

        # 姿态仪: 固定参考地平线 + 随 roll 倾斜/随 pitch 偏移的车身线
        cx_i = bx + box_w // 2
        cy_i = by + 24
        half = 28
        # 参考地平线 (固定)
        cv2.line(img, (cx_i - half, cy_i), (cx_i + half, cy_i),
                 T.TEXT_LOW, 1, cv2.LINE_AA)
        cv2.circle(img, (cx_i, cy_i), 2, T.TEXT_LOW, -1)
        # 车身线 (roll 倾斜 + pitch 上下偏移, 视觉增益 4x)
        pitch_px = int(np.clip(np.degrees(pitch_rad) * 4.0, -10, 10))
        roll_vis = roll_rad * 4.0  # 视觉增益: 轻微侧倾也可见
        cos_r, sin_r = np.cos(roll_vis), np.sin(roll_vis)
        ex = int(half * cos_r)
        ey = int(half * sin_r)
        cv2.line(img,
                 (cx_i - ex, cy_i - pitch_px - ey),
                 (cx_i + ex, cy_i - pitch_px + ey),
                 T.ACCENT, 2, cv2.LINE_AA)
        # 中心标记
        cv2.circle(img, (cx_i, cy_i - pitch_px), 2, T.ACCENT, -1)

        # 数值
        cv2.putText(img, f"R {np.degrees(roll_rad):+.1f}",
                    (bx + 6, by + box_h - 22),
                    T.FONT, 0.34, T.TEXT_MID, 1, cv2.LINE_AA)
        cv2.putText(img, f"P {np.degrees(pitch_rad):+.1f}",
                    (bx + 6, by + box_h - 6),
                    T.FONT, 0.34, T.TEXT_MID, 1, cv2.LINE_AA)

    def _draw_top_hud(self, img, vx, vy, heading, speed_ms):
        """俯视图 HUD (标题栏 + 速度 + 底部位置/航向 + 图例)"""
        # 标题栏
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, 40), T.PANEL, -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.putText(img, "TOP VIEW", (10, 28),
                    T.FONT, 0.55, T.TEXT_HI, 1, cv2.LINE_AA)

        # 速度（右侧）
        speed_text = f"{speed_ms:.1f} m/s"
        cv2.putText(img, speed_text, (self.w - 110, 28),
                    T.FONT, 0.55, T.ACCENT, 1, cv2.LINE_AA)

        # 底部信息栏
        overlay = img.copy()
        cv2.rectangle(overlay, (0, self.h - 50),
                      (self.w, self.h), T.PANEL, -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.line(img, (0, self.h - 50), (self.w, self.h - 50),
                 T.EDGE, 1, cv2.LINE_AA)

        cv2.putText(img, f"({vx:+.2f}, {vy:.1f}) m",
                    (10, self.h - 28),
                    T.FONT, 0.42, T.TEXT_MID, 1, cv2.LINE_AA)
        cv2.putText(img, f"{np.degrees(heading):+.1f} deg",
                    (10, self.h - 8),
                    T.FONT, 0.42, T.TEXT_MID, 1, cv2.LINE_AA)

        # 图例
        lx = self.w - 175
        cv2.line(img, (lx, self.h - 30), (lx + 18, self.h - 30),
                 T.CROP_ROW_HI, 3)
        cv2.putText(img, "Crop", (lx + 22, self.h - 26),
                    T.FONT, 0.3, T.TEXT_MID, 1, cv2.LINE_AA)
        cv2.rectangle(img, (lx, self.h - 16), (lx + 18, self.h - 6),
                      T.VEH_BODY, -1)
        cv2.putText(img, "Vehicle", (lx + 22, self.h - 8),
                    T.FONT, 0.3, T.TEXT_MID, 1, cv2.LINE_AA)
        # 导航线图例
        cv2.line(img, (lx + 90, self.h - 30), (lx + 108, self.h - 30),
                 T.NAV_LINE, 2)
        cv2.putText(img, "Nav", (lx + 112, self.h - 26),
                    T.FONT, 0.3, T.TEXT_MID, 1, cv2.LINE_AA)

    def clear_trail(self):
        self.trail.clear()

    def add_trail_point(self, x: float, y: float):
        self.trail.append((x, y))
        if len(self.trail) > self.max_trail:
            self.trail = self.trail[-self.max_trail:]


class Visualizer:
    """仿真可视化器

    show_dashboard=True (默认): 单窗口综合仪表盘 (Dashboard)
    show_dashboard=False:       经典双窗口 (Camera View + Top View)
    传 show_top_view=... 时按旧语义映射 (向后兼容)。
    """

    WINDOW_MAIN = "AGV Nav Simulator"
    WINDOW_CAMERA = "Camera View"
    WINDOW_TOP = "Top View"

    def __init__(self, simulator: NavSimulator = None,
                 scenario_name: str = "",
                 show_dashboard: bool = True,
                 show_top_view: bool = None):
        # 旧参数兼容: show_top_view=False → 仪表盘, True → 经典双窗口
        if show_top_view is not None:
            show_dashboard = not show_top_view

        self.sim = simulator
        self.scenario_name = scenario_name
        self.show_dashboard = show_dashboard

        self._dashboard = None
        if show_dashboard and simulator is not None:
            # 延迟在方法内 import: dashboard.py 顶部 import 本模块的
            # TopViewRenderer, 顶部互相 import 会循环
            from .dashboard import Dashboard
            self._dashboard = Dashboard(simulator, scenario_name)

        # 经典模式的独立俯视图渲染器 (仪表盘模式复用 Dashboard 内部的)
        self.top_renderer = None if show_dashboard else TopViewRenderer()
        self._windows_created = False

    def _ensure_windows(self, simulator: NavSimulator = None):
        if self._windows_created:
            return
        if self.show_dashboard:
            if self._dashboard is None:
                sim = simulator or self.sim
                if sim is not None:
                    from .dashboard import Dashboard
                    self._dashboard = Dashboard(sim, self.scenario_name)
            if self._dashboard is not None:
                w, h = self._dashboard.size
                cv2.namedWindow(self.WINDOW_MAIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.WINDOW_MAIN, w, h)
        else:
            cv2.namedWindow(self.WINDOW_CAMERA, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_CAMERA, 656, 800)
            cv2.namedWindow(self.WINDOW_TOP, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_TOP, 520, 720)
        self._windows_created = True

    def update(self, result: FrameResult,
               simulator: NavSimulator = None) -> int:
        """渲染一帧并取键盘输入; 返回按键码"""
        self._ensure_windows(simulator)

        if self.show_dashboard and self._dashboard is not None:
            # ── 单窗口仪表盘 ──
            img = self._dashboard.update(result)
            cv2.imshow(self.WINDOW_MAIN, img)
        elif not self.show_dashboard:
            # ── 经典双窗口 ──
            sim = simulator or self.sim
            cam_img = result.mask_image
            if cam_img is None and sim is not None:
                cam_img = np.zeros((sim.crop_h, sim.crop_w, 3),
                                   dtype=np.uint8)

            if cam_img is not None:
                cam_img = draw_hud(
                    cam_img,
                    yaw_deg=result.yaw_deg,
                    front_dist=result.front_dist,
                    rear_dist=result.rear_dist,
                    quality=result.image_quality,
                    speed_ms=result.veh_speed,
                    frame_idx=result.frame_idx,
                    sim_time=result.sim_time,
                    nav_valid=result.nav_valid,
                    num_masks=result.num_masks,
                    steering_deg=result.veh_steering_deg,
                    roll_deg=result.body_roll_deg,
                    pitch_deg=result.body_pitch_deg,
                    control_mode=result.control_mode,
                    occluded=result.occluded,
                    rows_detected=result.rows_detected,
                    rows_expected=result.rows_expected,
                    perturb_info=result.perturb_tags,
                )
                cv2.imshow(self.WINDOW_CAMERA, cam_img)

            if self.top_renderer is not None and sim is not None:
                self.top_renderer.add_trail_point(
                    result.veh_x, result.veh_y)
                # RowMap 后端: 行地图可视化 (基线无此接口)
                vn = getattr(sim, '_vn', None)
                map_overlay = (vn.get_map_overlay()
                               if hasattr(vn, 'get_map_overlay') else None)
                top_img = self.top_renderer.render(
                    sim.field_model,
                    result.veh_x, result.veh_y,
                    np.radians(result.veh_heading_deg),
                    np.radians(result.veh_steering_deg),
                    result.veh_speed,
                    result.nav_line_world,
                    np.radians(result.body_roll_deg),
                    np.radians(result.body_pitch_deg),
                    perturbation=getattr(sim, 'perturbation', None),
                    occluded=result.occluded,
                    map_overlay=map_overlay,
                )
                cv2.imshow(self.WINDOW_TOP, top_img)

        key = cv2.waitKey(1) & 0xFF
        return key

    def handle_key(self, key: int,
                   simulator: NavSimulator = None) -> bool:
        """处理按键; 返回 False 表示退出仿真"""
        if key == 255 or key == -1:
            return True

        sim = simulator or self.sim
        if sim is None:
            return True

        if key == ord('q') or key == 27:
            return False
        elif key == ord(' '):
            sim.toggle_pause()
        elif key == ord('r'):
            sim.reset()
            if self._dashboard is not None:
                self._dashboard.clear_history()
            if self.top_renderer is not None:
                self.top_renderer.clear_trail()
        elif key == ord('+') or key == ord('='):
            sim.vehicle.set_speed(
                sim.vehicle.state.speed + 0.2)
        elif key == ord('-'):
            # 软下限 0.3 m/s: 对齐基线算法的 RTK 静止门限,
            # 低于此速度内部行跟踪会冻结 (输出钉死, q 衰减至丢行)
            sim.vehicle.set_speed(
                max(0.3, sim.vehicle.state.speed - 0.2))
        elif key == ord('m') or key == ord('M'):
            self._cycle_sensor_mode(sim)
        elif key == ord('h') or key == ord('H'):
            # H 键: 强制进入掉头程序 (掉头中止机制演示/调试)。
            # 断垄场景中按 H: 行其实延续 (对岸锚点在视野内) →
            # rowmap 后端掉头中看到任务行 y 跨度增长 → 中止跳捕获
            # 段快速回 WORK; --no-abort 时则完整梨形掉头 (旧行为)
            if sim.vehicle.mode.value == "WORK":
                sim.vehicle._enter_headland()
                print("[GUI] H: 强制触发掉头 (中止机制演示)", flush=True)

        return True

    @staticmethod
    def _cycle_sensor_mode(sim: NavSimulator):
        """M 键: 循环切换传感器融合模式 (0 视觉 → 1 +IMU → 2 +RTK)

        同时更新配置 (下一帧 FrameResult 反映) 与 C++ 核心 (立即生效)。
        """
        cur = int(getattr(sim.sim_config, 'sensor_mode', 2))
        new_mode = (cur + 1) % 3
        sim.sim_config.sensor_mode = new_mode
        vn = getattr(sim, '_vn', None)
        if vn is not None:
            vn.set_sensor_mode(new_mode)

    def close(self):
        cv2.destroyAllWindows()
        self._windows_created = False
