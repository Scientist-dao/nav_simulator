"""
camera_renderer.py - 仿真相机图像渲染器

根据作物行的车辆坐标系几何，渲染仿真的相机视角图像。
用于端到端测试（配合 YOLOv8 推理），或纯可视化展示。

渲染策略:
  - 多层次土壤颜色 + 随机噪点 + 犁沟纹理
  - 天空蓝色渐变 + 地平线光晕
  - 作物行使用物理行宽投影（透视正确）
  - 光照效果：近亮远暗
  - 镜头晕影：四角自然变暗
"""

import numpy as np
import cv2
from typing import List, Tuple


# 颜色定义 (BGR)
SOIL_BASE = (55, 80, 135)           # 主土色
SOIL_LIGHT = (65, 95, 150)          # 浅土色
SOIL_DARK = (42, 62, 105)           # 深土色
SOIL_FURROW = (38, 55, 95)          # 犁沟色
VEGETATION_MAIN = (50, 170, 50)     # 作物绿
VEGETATION_DARK = (30, 130, 30)     # 深绿
VEGETATION_LIGHT = (70, 190, 70)    # 浅绿
MULCH_MAIN = (195, 195, 205)        # 覆膜白
MULCH_DARK = (165, 165, 175)        # 覆膜灰
SKY_TOP = (232, 205, 158)           # 天空顶部 (淡蓝)
SKY_HORIZON = (208, 198, 178)       # 地平线 (暖白微蓝)

# 相机几何 (与 benchmark fixture 一致, 用于地面纹理投影)
_CAM_FX = 440.84
_CAM_CX = 326.91
_CAM_CY = 379.33
_CAM_HEIGHT = 2.88                  # 相机离地高度 (m)
_CAM_PITCH_RAD = 0.1745             # 俯角 10°


class CameraRenderer:
    """
    仿真相机图像渲染器

    将车辆坐标系中的作物行渲染为仿真 BGR 图像。
    包含真实光照、纹理、晕影效果。
    """

    def __init__(self, crop_w: int, crop_h: int,
                 mask_generator=None, row_spacing: float = 0.60):
        """
        Args:
            crop_w: 裁剪图像宽度
            crop_h: 裁剪图像高度
            mask_generator: MaskGenerator 实例（复用其投影方法）
            row_spacing: 作物行距 (m), 用于梨沟纹理周期与行对齐
        """
        self.crop_w = crop_w
        self.crop_h = crop_h
        self._mask_gen = mask_generator
        self._row_spacing = row_spacing

        # 预生成晕影掩码
        self._vignette = self._make_vignette(crop_w, crop_h)

        # 预生成地面纹理噪声种子
        self._texture_seed = np.random.randint(0, 2**31)

    @staticmethod
    def _make_vignette(w: int, h: int) -> np.ndarray:
        """生成镜头晕影掩码（中心亮、四角暗）"""
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        # 归一化到 [-1, 1]
        nx = (x - cx) / cx
        ny = (y - cy) / cy
        # 距离平方
        dist_sq = nx * nx + ny * ny
        # 晕影强度：中心 1.0 → 四角 ~0.75
        vignette = 1.0 - 0.25 * np.clip(dist_sq, 0, 2) / 2.0
        return vignette.astype(np.float32)

    def render(self,
               visible_rows: List[Tuple[int, List[Tuple[float, float]], int, float]],
               veh_x: float = 0.0,
               veh_y: float = 0.0,
               veh_heading: float = 0.0,
               add_noise: bool = True,
               add_texture: bool = True,
               ) -> np.ndarray:
        """
        渲染仿真相机图像。

        Args:
            visible_rows: FieldModel.get_visible_rows_vehicle() 的输出
            add_noise: 是否添加图像噪声
            add_texture: 是否添加地面纹理

        Returns:
            BGR 图像 (crop_h, crop_w, 3), uint8
        """
        img = np.zeros((self.crop_h, self.crop_w, 3), dtype=np.uint8)

        # 1. 绘制地面 (基础色 + 噪点)
        self._draw_ground(img, add_texture)

        # 2. 透视纵向梨沟 (沿行方向汇聚, 近疏远密)
        self._draw_perspective_furrows(img)

        # 3. 横向田埂 (世界绑定, 随车前进向近处流动)
        if self._mask_gen is not None:
            self._draw_field_ridges(img, veh_x, veh_y, veh_heading)

        # 4. 世界绑定的土块斑点 (随车流动, 提供运动感)
        if self._mask_gen is not None and add_texture:
            self._draw_ground_features(img, veh_x, veh_y, veh_heading)

        # 5. 绘制天空 (地平线按 10° 俯角几何)
        self._draw_sky(img)

        # 6. 画面底部车体阴影
        self._draw_vehicle_shadow(img)

        # 7. 绘制作物行 (含侧向投影)
        if self._mask_gen is not None:
            self._draw_rows(img, visible_rows)
            # 行上植株团块 (世界间距对齐, 沿行流动)
            self._draw_plants(img, visible_rows, veh_x, veh_y, veh_heading)

        # 8. 大气透视 (远处地面融入天色)
        self._apply_atmospheric_perspective(img)

        # 7. 光照渐变（近亮远暗）
        self._apply_depth_shading(img)

        # 5. 镜头晕影
        img = (img.astype(np.float32) * self._vignette[:, :, np.newaxis]).astype(np.uint8)

        # 6. 图像噪声
        if add_noise:
            noise = np.random.normal(0, 2.5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def _draw_ground(self, img: np.ndarray, add_texture: bool):
        """绘制地面背景（多层次纹理）"""
        h, w = img.shape[:2]

        # 基础土色
        img[:] = SOIL_BASE

        if not add_texture:
            return

        # 细小噪点 (屏幕空间固定即可, 不显眼;
        # 大块纹理必须世界绑定——见 _draw_ground_features, 否则车动时画面静止)
        rng = np.random.RandomState(self._texture_seed)
        for _ in range(300):
            nx = rng.randint(0, w)
            ny = rng.randint(0, h)
            nr = rng.randint(1, 3)
            color = SOIL_DARK if rng.random() < 0.5 else SOIL_LIGHT
            cv2.circle(img, (nx, ny), nr, color, -1)

    @staticmethod
    def _vehicle_to_global(xv, zv, veh_x, veh_y, heading):
        """车辆坐标 → 全局坐标"""
        c, s = np.cos(heading), np.sin(heading)
        return veh_x + xv * c + zv * s, veh_y - xv * s + zv * c

    @staticmethod
    def _global_to_vehicle(gx, gy, veh_x, veh_y, heading):
        """全局坐标 → 车辆坐标"""
        dx, dy = gx - veh_x, gy - veh_y
        c, s = np.cos(heading), np.sin(heading)
        return dx * c - dy * s, dx * s + dy * c

    @staticmethod
    def _grid_hash(gi: int, gj: int) -> Tuple[float, float]:
        """确定性网格哈希 → [0,1) 两分量 (同一世界点每帧外观一致)"""
        h1 = ((gi * 2654435761) ^ (gj * 40503)) & 0xFFFF
        h2 = ((gi * 2246822519) ^ (gj * 3266489917)) & 0xFFFF
        return h1 / 65535.0, h2 / 65535.0

    def _draw_field_ridges(self, img: np.ndarray,
                           veh_x: float, veh_y: float, veh_heading: float):
        """横向田埂/接行痕迹 (世界绑定, 随车前进从远处流向近处)"""
        h, w = img.shape[:2]
        spacing = 8.0
        y_next = np.ceil((veh_y + 0.5) / spacing) * spacing
        for k in range(3):
            gy = y_next + k * spacing
            if gy > veh_y + 22.0:
                break
            seg_pts = []
            for gx in np.linspace(veh_x - 4.0, veh_x + 4.0, 17):
                xv, zv = self._global_to_vehicle(
                    gx, gy, veh_x, veh_y, veh_heading)
                if zv < 0.4:
                    seg_pts.append(None)
                    continue
                col, row = self._mask_gen._vehicle_xz_to_pixel(xv, zv)
                seg_pts.append((col, row, zv))
            for i in range(len(seg_pts) - 1):
                p1, p2 = seg_pts[i], seg_pts[i + 1]
                if p1 is None or p2 is None:
                    continue
                if not (0 <= p1[1] < h or 0 <= p2[1] < h):
                    continue
                z_avg = (p1[2] + p2[2]) / 2.0
                thick = max(1, int(439.0 * 0.22 / z_avg))
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]),
                         (45, 65, 110), thick)
                # 上缘受光面 (田埂隆起感)
                hi = max(1, thick // 3)
                cv2.line(img, (p1[0], p1[1] - hi), (p2[0], p2[1] - hi),
                         (70, 95, 145), max(1, hi))

    def _draw_ground_features(self, img: np.ndarray,
                              veh_x: float, veh_y: float, veh_heading: float):
        """世界绑定的土块/斑点 (随车前进流动, 提供运动感)"""
        h, w = img.shape[:2]
        step = 0.5
        gx0 = np.floor((veh_x - 4.0) / step) * step
        gy0 = np.floor(veh_y / step) * step
        gx = gx0
        while gx < veh_x + 4.0:
            gy = gy0
            while gy < veh_y + 22.0:
                gi, gj = int(round(gx / step)), int(round(gy / step))
                h1, h2 = self._grid_hash(gi, gj)
                if h1 < 0.30:  # 30% 密度
                    jx = gx + (h2 - 0.5) * 0.4
                    jy = gy + (h1 * 3.7 % 1.0 - 0.5) * 0.4
                    xv, zv = self._global_to_vehicle(
                        jx, jy, veh_x, veh_y, veh_heading)
                    if zv > 0.5:
                        col, row = self._mask_gen._vehicle_xz_to_pixel(xv, zv)
                        if 0 <= col < w and 0 <= row < h:
                            r = max(1, int(440.0 * 0.035 / zv))
                            color = SOIL_DARK if h2 < 0.5 else SOIL_LIGHT
                            cv2.circle(img, (col, row), r, color, -1)
                gy += step
            gx += step

    def _draw_plants(self, img: np.ndarray, visible_rows,
                     veh_x: float, veh_y: float, veh_heading: float):
        """作物行上的植株团块 (世界间距 0.35m 对齐, 随车沿行流动)"""
        h, w = img.shape[:2]
        spacing = 0.35
        for row_idx, pts_vehicle, class_id, width_m in visible_rows:
            if len(pts_vehicle) < 2 or class_id != 0:
                continue
            pts_g = [self._vehicle_to_global(xv, zv, veh_x, veh_y, veh_heading)
                     for xv, zv in pts_vehicle]
            for i in range(len(pts_g) - 1):
                x1, y1 = pts_g[i]
                x2, y2 = pts_g[i + 1]
                y_lo, y_hi = min(y1, y2), max(y1, y2)
                if y_hi - y_lo < 1e-6:
                    continue
                k = np.ceil(y_lo / spacing) * spacing
                while k < y_hi:
                    t = (k - y1) / (y2 - y1)
                    gx = x1 + t * (x2 - x1)
                    xv, zv = self._global_to_vehicle(
                        gx, k, veh_x, veh_y, veh_heading)
                    if zv > 0.4:
                        col, row = self._mask_gen._vehicle_xz_to_pixel(xv, zv)
                        if 0 <= col < w and 0 <= row < h:
                            r = max(1, int(440.0 * 0.055 / zv))
                            gi = int(round(k / spacing))
                            shade = ((gi * 2654435761) & 0xFFFF) / 65535.0
                            color = ((40, 180, 40) if shade < 0.5
                                     else (60, 200, 60))
                            cv2.circle(img, (col, row), r, color, -1)
                    k += spacing

    def _draw_perspective_furrows(self, img: np.ndarray):
        """透视纵向梨沟：沿行驶方向的垄沟纹理，随深度自然汇聚

        对每一像素行反推地面距离 z，以行距 (0.65m) 为周期生成正弦垄沟，
        沟正好落在作物行之间；幅值近强远弱 (远处纹理模糊)。
        """
        h, w = img.shape[:2]
        horizon = int(h * 0.36)
        cols = np.arange(w, dtype=np.float32)
        for y in range(horizon, h):
            # 该像素行对应的地面距离
            theta = _CAM_PITCH_RAD + np.arctan((y - _CAM_CY) / _CAM_FX)
            if theta <= 0.02:
                continue
            z = _CAM_HEIGHT / np.tan(theta)
            if z > 25.0:
                continue
            # 横向车辆坐标 → 以行距为周期的正弦垄沟
            xv = (cols - _CAM_CX) * z / _CAM_FX
            amp = min(0.10, 0.30 / max(z, 1.0))
            s = self._row_spacing
            stripe = np.sin(2.0 * np.pi * (xv + s / 2.0) / s)
            factor = (1.0 + amp * stripe)[:, np.newaxis]
            img[y] = np.clip(
                img[y].astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _draw_vehicle_shadow(self, img: np.ndarray):
        """画面底部中央的车体阴影 (模拟车载相机近处遮挡)"""
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.ellipse(overlay, (w // 2, h + 30), (int(w * 0.42), 70),
                    0, 0, 360, (25, 35, 55), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    def _draw_sky(self, img: np.ndarray):
        """绘制天空渐变和地平线光晕（地平线位置按 10° 俯角几何）"""
        h, w = img.shape[:2]
        # 地平线: cy - fy*tan(10°) ≈ 0.36h (crop_h=800)
        sky_rows = int(h * 0.36)

        for y in range(sky_rows):
            ratio = y / sky_rows
            # 从地平线色渐变到天空顶部色
            color = tuple(
                int(h * (1 - ratio) + t * ratio)
                for h, t in zip(SKY_HORIZON, SKY_TOP)
            )
            img[y] = color

        # 地平线光晕（柔和过渡带）
        glow_rows = 8
        for y in range(sky_rows - glow_rows, sky_rows):
            alpha = (y - (sky_rows - glow_rows)) / glow_rows * 0.3
            glow_color = tuple(min(255, int(c * 1.1)) for c in SKY_HORIZON)
            overlay = img[y:y+1].copy()
            overlay[:] = glow_color
            cv2.addWeighted(overlay, alpha, img[y:y+1], 1 - alpha, 0,
                            img[y:y+1])

    def _draw_rows(self, img: np.ndarray,
                   visible_rows: List[Tuple[int, List[Tuple[float, float]], int, float]]):
        """绘制作物行（物理行宽投影，自然透视）"""
        for row_idx, pts_vehicle, class_id, width_m in visible_rows:
            if class_id == 0:
                color_main = VEGETATION_MAIN
                color_dark = VEGETATION_DARK
                color_light = VEGETATION_LIGHT
            else:
                color_main = MULCH_MAIN
                color_dark = MULCH_DARK
                color_light = MULCH_MAIN

            half_w = width_m / 2.0
            dense_pts = self._mask_gen._densify_points(pts_vehicle, 0.25)

            # 投影左右边缘
            left_px = []
            right_px = []
            center_px = []
            for xv, zv in dense_pts:
                col_l, row_l = self._mask_gen._vehicle_xz_to_pixel(
                    xv - half_w, zv)
                col_r, row_r = self._mask_gen._vehicle_xz_to_pixel(
                    xv + half_w, zv)
                col_c, row_c = self._mask_gen._vehicle_xz_to_pixel(xv, zv)

                l_ok = (0 <= col_l < self.crop_w and 0 <= row_l < self.crop_h)
                r_ok = (0 <= col_r < self.crop_w and 0 <= row_r < self.crop_h)
                c_ok = (0 <= col_c < self.crop_w and 0 <= row_c < self.crop_h)

                if l_ok or r_ok:
                    if not l_ok:
                        col_l, row_l = col_r, row_r
                    if not r_ok:
                        col_r, row_r = col_l, row_l
                    left_px.append((col_l, row_l))
                    right_px.append((col_r, row_r))
                if c_ok:
                    center_px.append((col_c, row_c))

            if len(left_px) < 2:
                continue

            # 主填充多边形
            polygon = np.array(
                left_px + right_px[::-1], dtype=np.int32)
            cv2.fillPoly(img, [polygon], color_main)

            # 右下侧地面投影 (近宽远窄, 与左侧高光光照方向一致)
            if len(right_px) >= 2 and class_id == 0:
                outer = []
                for (lx, ly), (rx, ry) in zip(left_px, right_px):
                    off = max(2, int((rx - lx) * 0.35))
                    outer.append((min(rx + off, self.crop_w - 1),
                                  min(ry + 1, self.crop_h - 1)))
                shadow_poly = np.array(
                    right_px + outer[::-1], dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [shadow_poly], (30, 45, 70))
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

            # 左侧浅色高光（模拟光照）
            if len(left_px) >= 2 and len(center_px) >= 2:
                n = min(len(left_px), len(center_px))
                highlight = np.array(
                    left_px[:n] + center_px[:n][::-1], dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [highlight], color_light)
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

            # 中心深色线
            for i in range(len(center_px) - 1):
                cv2.line(img, center_px[i], center_px[i + 1],
                         color_dark, 2)

    def draw_hood(self, img: np.ndarray, steering_rad: float = 0.0):
        """绘制车头引擎盖 (屏幕空间固定, 刚性连接在车身/相机上)

        必须在相机姿态抖动 (warpAffine) 之后调用:
        世界随姿态旋转, 车头保持不动——与真实车载相机一致,
        转弯/侧倾时可见世界相对车头移动。

        Args:
            steering_rad: 前轮转角 (rad, 右正); 引擎盖两侧画出随转角
                          摆动的前轮——固定车载相机里唯一可见的转向动作
        """
        h, w = img.shape[:2]
        top_y = int(h * 0.86)  # 引擎盖前沿 (远端)

        # ── 引擎盖主体 (透视梯形, 与俯视图车体同色) ──
        hood = np.array([
            (int(w * 0.34), top_y), (int(w * 0.66), top_y),
            (int(w * 0.80), h), (int(w * 0.20), h),
        ], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [hood], (10, 110, 215))
        cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)

        # ── 引擎盖两侧暗边 (宽柔和阴影带, 立体感) ──
        for xa, xb in [(0.20, 0.27), (0.73, 0.80)]:
            # 引擎盖侧边线上、距侧边 offset 处的点
            def _edge_x(t, _xa=xa, _xb=xb):  # t: 0=顶 1=底
                top_x = w * (0.34 if _xa < 0.5 else 0.66)
                bot_x = w * _xa
                return int(top_x + (bot_x - top_x) * t)
            band = np.array([
                (_edge_x(0), top_y), (_edge_x(1), h),
                (_edge_x(1) + (16 if xa < 0.5 else -16), h),
                (_edge_x(0) + (6 if xa < 0.5 else -6), top_y),
            ], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [band], (5, 80, 165))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # ── 中央脊线隆起 ──
        ridge = np.array([
            (int(w * 0.475), top_y), (int(w * 0.525), top_y),
            (int(w * 0.56), h), (int(w * 0.44), h),
        ], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [ridge], (40, 155, 245))
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        # ── 前沿高光 (顶部亮边) ──
        cv2.line(img, (int(w * 0.34), top_y),
                 (int(w * 0.66), top_y), (90, 195, 255), 2)

        # ── 前格栅 (引擎盖前沿下方横条纹) ──
        for i in range(3):
            gy = top_y + 8 + i * 7
            inset = int(i * w * 0.008)
            cv2.line(img,
                     (int(w * 0.42) + inset, gy),
                     (int(w * 0.58) - inset, gy),
                     (5, 70, 140), 2)

        # ── 前轮 (引擎盖两侧探出, 随前轮转角摆动) ──
        self._draw_front_wheel(img, int(w * 0.165), int(h * 0.97),
                               steering_rad)
        self._draw_front_wheel(img, int(w * 0.835), int(h * 0.97),
                               steering_rad)

    @staticmethod
    def _draw_front_wheel(img: np.ndarray, cx: int, cy: int,
                          steering_rad: float):
        """单个前轮: 随转向角做屏幕平面旋转 (2D 近似)

        右转 (steering>0) 时轮胎在画面上顺时针摆; 宽度随 cos 轻微压缩,
        模拟转向时轮胎侧面可见的透视变化。底部超出画面被裁, 只露上半截。
        """
        h, w = img.shape[:2]
        rot = steering_rad * 1.2          # 视觉增益, 小转角也可见
        ww = max(5, int(w * 0.030 * np.cos(rot * 0.7)))  # 胎半宽
        wh = int(h * 0.060)               # 胎半高
        c, s = np.cos(rot), np.sin(rot)

        # 轮胎体 (旋转矩形)
        corners = []
        for dx, dy in [(-ww, -wh), (ww, -wh), (ww, wh), (-ww, wh)]:
            corners.append((int(cx + dx * c - dy * s),
                            int(cy + dx * s + dy * c)))
        cv2.fillPoly(img, [np.array(corners, dtype=np.int32)],
                     (35, 35, 40))

        # 轮毂 (同旋转的小矩形)
        hw, hh = max(2, ww // 3), wh * 2 // 3
        hub = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            hub.append((int(cx + dx * c - dy * s),
                        int(cy + dx * s + dy * c)))
        cv2.fillPoly(img, [np.array(hub, dtype=np.int32)],
                     (85, 85, 92))

    def _apply_atmospheric_perspective(self, img: np.ndarray):
        """大气透视：地平线附近地面逐渐融入天色 (消除色块边界感)"""
        h, w = img.shape[:2]
        horizon = int(h * 0.36)
        n = min(110, h - horizon)
        if n <= 0:
            return
        sky = np.array(SKY_HORIZON, dtype=np.float32)
        t = np.arange(n, dtype=np.float32) / n       # 0=地平线 → 1=近处
        blend = ((1.0 - t) ** 1.5) * 0.55            # 地平线处混 55% 天色
        region = img[horizon:horizon + n].astype(np.float32)
        img[horizon:horizon + n] = np.clip(
            region * (1.0 - blend[:, np.newaxis, np.newaxis])
            + sky[np.newaxis, np.newaxis, :] * blend[:, np.newaxis, np.newaxis],
            0, 255).astype(np.uint8)

    def _apply_depth_shading(self, img: np.ndarray):
        """深度光照：近亮远暗（模拟大气衰减）"""
        h, w = img.shape[:2]
        # 从底部（近）到顶部（远）逐渐变暗
        gradient = np.linspace(1.05, 0.85, h, dtype=np.float32)
        img_float = img.astype(np.float32)
        img_float *= gradient[:, np.newaxis, np.newaxis]
        np.clip(img_float, 0, 255, out=img_float)
        img[:] = img_float.astype(np.uint8)
