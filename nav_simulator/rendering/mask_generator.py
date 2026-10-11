"""
mask_generator.py - 作物行 Mask 生成器

将车辆坐标系中的作物行几何点投影到裁剪后图像的像素坐标系，
生成与 YOLOv8 分割输出格式兼容的 MaskData 列表 (与板端
MaskDataPlus 载体同构, 消费方按 .mask/.classic_class/.score
属性鸭子类型读取)。

投影逻辑与板端工程的 mask 基准实现保持一致 (保证仿真 mask 与
实车分割输出同构), 增加闭环仿真所需的动态行输入与车身姿态支持
(set_attitude: 相机随车身俯仰/侧倾, 为 IMU 姿态补偿提供真实扰动源)。
"""

from dataclasses import dataclass

import numpy as np
import cv2
from typing import List, Tuple, Optional


@dataclass
class MaskData:
    """分割 mask 载体 (鸭子类型): 与板端 MaskDataPlus 同构,
    暴露 .mask/.classic_class/.score, C++ 绑定与参考实现均按属性读取。"""
    mask: np.ndarray
    classic_class: int
    score: float


class MaskGenerator:
    """
    从车辆坐标系的作物行几何点生成 mask。

    使用相机内参 (K_final) 和外参 (R_v2c, T_v2c) 将地面上的 3D 点
    投影到 2D 裁剪图像像素坐标，然后绘制填充多边形条带作为 mask。
    set_attitude() 后外参动态生效, mask 与背景渲染同步受姿态扰动。
    """

    def __init__(self, camera_info):
        """
        Args:
            camera_info: 相机参数对象 (鸭子类型, 需含
                         K_final, R_v2c, T_v2c, T_c2v, camera_rad,
                         crop_w, crop_h 属性, 见 core/camera_params.py),
        """
        self.K_final = np.array(camera_info.K_final, dtype=float)
        # 裁剪后图像尺寸 (pixel_to_vehicle_xz 边界判断与 simulator 地平线搜索用)
        self.crop_w = int(camera_info.crop_w)
        self.crop_h = int(camera_info.crop_h)
        # nominal 外参 (姿态归零基准); R_v2c/T_v2c 为当前生效外参,
        # set_attitude() 后动态替换 (camera_renderer 复用本类投影, 自动跟随)
        self._R_v2c_nominal = np.array(camera_info.R_v2c, dtype=float)
        self._T_v2c_nominal = np.array(camera_info.T_v2c, dtype=float)
        self.R_v2c = self._R_v2c_nominal.copy()
        self.T_v2c = self._T_v2c_nominal.copy()
        self._camera_rad_nominal = float(camera_info.camera_rad)
        self._T_c2v = np.array(camera_info.T_c2v, dtype=float).reshape(3, 1)
        self._attitude = (0.0, 0.0)   # (roll_rad, pitch_rad) 当前车身姿态

    # ════════════════════════════════════════════════════════
    #  车身姿态 -> 动态相机外参
    #  ════════════════════════════════════════════════════════
    def set_attitude(self, roll_rad: float = 0.0, pitch_rad: float = 0.0):
        """更新车身姿态, 重建相机等效外参 (与 C++ set_attitude 几何一致)

        相机刚接在车身上, 车身倾斜时相机随动:
          - pitch (抬头正): 相机有效俯角 = nominal - pitch,
            按 camera_info.py 同样构造重建 R_c2v
          - roll (右侧下沉正): R_unlevel 把倾斜车身系恢复到水平地面系,
            水平地面系行点先经 R_unlevel^T 进入倾斜车身系再投影

        相机中心随刚体一起倾斜 (在水平系中位置/高度微变, 与实车一致);
        传 (0, 0) 恢复 nominal 外参。
        符号约定与基线管线实现的 set_attitude
        逐项对齐, 前后向投影自洽。
        """
        roll_rad = float(roll_rad)
        pitch_rad = float(pitch_rad)
        self._attitude = (roll_rad, pitch_rad)

        # 1. Pitch: 有效俯角下的 R_c2v (同 camera_info.py 构造)
        camera_rad_eff = self._camera_rad_nominal - pitch_rad
        c, s = np.cos(camera_rad_eff), np.sin(camera_rad_eff)
        R_c2v_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c]], dtype=float)

        # 2. Roll: unlevel 旋转 (倾斜车身系 -> 水平系, 绕 Z 前向轴)
        cr, sr = np.cos(roll_rad), np.sin(roll_rad)
        R_unlevel = np.array([
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0]], dtype=float)

        # 3. 组合有效外参: R_c2v_eff = R_unlevel @ R_c2v_pitch
        #    v_level = R_unlevel @ (R_c2v @ p_c + T_c2v)
        #    投影方向: p_c = R_c2v_eff^T @ (v_level - T_c2v_eff)
        R_c2v_eff = R_unlevel @ R_c2v_pitch
        T_c2v_eff = R_unlevel @ self._T_c2v
        self.R_v2c = R_c2v_eff.T
        self.T_v2c = -(R_c2v_eff.T @ T_c2v_eff)

    @property
    def attitude_rad(self) -> Tuple[float, float]:
        """当前车身姿态 (roll_rad, pitch_rad)"""
        return self._attitude

    def _vehicle_xz_to_pixel(self, x_m: float, z_m: float) -> Tuple[int, int]:
        """将车辆坐标系地面点 (X, 0, Z) 投影到像素 (col, row)"""
        pt_v = np.array([[x_m], [0.0], [z_m]], dtype=float)
        pt_c = self.R_v2c @ pt_v + self.T_v2c
        if pt_c[2, 0] <= 0:
            return (-1, -1)
        uv = self.K_final @ pt_c
        col = int(round(uv[0, 0] / uv[2, 0]))
        row = int(round(uv[1, 0] / uv[2, 0]))
        return (col, row)

    def pixel_to_vehicle_xz(self, col: float, row: float
                            ) -> Optional[Tuple[float, float]]:
        """像素反投影到车辆坐标系地面点 (X, Z) (与地平面 Y=0 求交)

        用于把导航线等像素结果换算回物理坐标 (数据输出 / 俯视图绘制)。
        点在相机后方或射线与地面平行时返回 None。
        """
        d = np.linalg.solve(self.K_final,
                            np.array([col, row, 1.0], dtype=float))
        R_T = self.R_v2c.T
        T = self.T_v2c.flatten()
        r_y = R_T[1, :]  # 车辆系 Y 轴在相机系中的方向
        denom = float(r_y @ d)
        if abs(denom) < 1e-9:
            return None
        t = float(r_y @ T) / denom
        if t <= 0:
            return None
        pt_v = R_T @ (t * d - T)
        return float(pt_v[0]), float(pt_v[2])

    def generate_masks(
            self,
            visible_rows: List[Tuple[int, List[Tuple[float, float]], int, float]],
            strip_width_px: int = 30,
            score: float = 0.95,
            z_subsample: float = 0.25,
            use_physical_width: bool = True,
            ) -> List[MaskData]:
        """
        从可见行生成 mask 列表。

        Args:
            visible_rows: FieldModel.get_visible_rows_vehicle() 的输出
                          [(row_idx, [(X_v, Z_v), ...], class_id, width_m), ...]
            strip_width_px: mask 条带半宽 (像素)，仅 use_physical_width=False 时生效
            score: 模拟检测置信度
            z_subsample: Z 方向子采样步长 (m)，用于在行点之间插值
            use_physical_width: True=使用物理行宽投影（近宽远窄），
                                False=使用固定像素宽度

        Returns:
            MaskData 列表
        """
        if use_physical_width:
            return self._generate_masks_physical(
                visible_rows, score, z_subsample)
        else:
            return self._generate_masks_fixed_width(
                visible_rows, strip_width_px, score, z_subsample)

    def _generate_masks_physical(
            self,
            visible_rows: List[Tuple[int, List[Tuple[float, float]], int, float]],
            score: float,
            z_subsample: float,
            ) -> List[MaskData]:
        """使用物理行宽投影生成 mask（透视正确，近宽远窄）"""
        masks = []

        for row_idx, pts_vehicle, class_id, width_m in visible_rows:
            mask = np.zeros((self.crop_h, self.crop_w), dtype=np.uint8)
            half_w = width_m / 2.0

            # 在行点之间插值，生成密集的投影点
            dense_pts = self._densify_points(pts_vehicle, z_subsample)

            # 投影左右边缘到像素坐标
            left_px = []
            right_px = []
            for xv, zv in dense_pts:
                col_l, row_l = self._vehicle_xz_to_pixel(xv - half_w, zv)
                col_r, row_r = self._vehicle_xz_to_pixel(xv + half_w, zv)
                # 至少一侧在图像内才保留
                l_ok = (0 <= col_l < self.crop_w and 0 <= row_l < self.crop_h)
                r_ok = (0 <= col_r < self.crop_w and 0 <= row_r < self.crop_h)
                if l_ok or r_ok:
                    # 如果一侧超出，用另一侧代替（保持边缘连续）
                    if not l_ok:
                        col_l, row_l = col_r, row_r
                    if not r_ok:
                        col_r, row_r = col_l, row_l
                    left_px.append((col_l, row_l))
                    right_px.append((col_r, row_r))

            if len(left_px) < 2:
                continue

            # 构建填充多边形：左边缘 + 右边缘逆序
            polygon = np.array(
                left_px + right_px[::-1], dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)

            if cv2.countNonZero(mask) > 0:
                masks.append(MaskData(mask, class_id, score))

        return masks

    def _generate_masks_fixed_width(
            self,
            visible_rows: List[Tuple[int, List[Tuple[float, float]], int, float]],
            strip_width_px: int,
            score: float,
            z_subsample: float,
            ) -> List[MaskData]:
        """使用固定像素宽度生成 mask（原始方法，兼容 benchmark）"""
        masks = []
        half = strip_width_px // 2

        for row_idx, pts_vehicle, class_id, width_m in visible_rows:
            mask = np.zeros((self.crop_h, self.crop_w), dtype=np.uint8)

            dense_pts = self._densify_points(pts_vehicle, z_subsample)

            pixel_pts = []
            for xv, zv in dense_pts:
                col, row = self._vehicle_xz_to_pixel(xv, zv)
                if 0 <= col < self.crop_w and 0 <= row < self.crop_h:
                    pixel_pts.append((col, row))

            if len(pixel_pts) < 2:
                continue

            for i in range(len(pixel_pts) - 1):
                col1, row1 = pixel_pts[i]
                col2, row2 = pixel_pts[i + 1]
                polygon = np.array([
                    [max(0, col1 - half), row1],
                    [min(self.crop_w - 1, col1 + half), row1],
                    [min(self.crop_w - 1, col2 + half), row2],
                    [max(0, col2 - half), row2],
                ], dtype=np.int32)
                cv2.fillPoly(mask, [polygon], 255)

            if cv2.countNonZero(mask) > 0:
                masks.append(MaskData(mask, class_id, score))

        return masks

    @staticmethod
    def _densify_points(pts: List[Tuple[float, float]],
                        max_step: float = 0.25
                        ) -> List[Tuple[float, float]]:
        """
        在行点之间插值，使相邻点间距不超过 max_step。

        Args:
            pts: [(X, Z), ...] 车辆坐标系点
            max_step: 最大步长 (m)

        Returns:
            密集化后的点列表
        """
        if len(pts) < 2:
            return list(pts)

        dense = [pts[0]]
        for i in range(len(pts) - 1):
            x1, z1 = pts[i]
            x2, z2 = pts[i + 1]
            dist = np.hypot(x2 - x1, z2 - z1)
            n_seg = max(1, int(np.ceil(dist / max_step)))
            for j in range(1, n_seg + 1):
                t = j / n_seg
                dense.append((x1 + t * (x2 - x1), z1 + t * (z2 - z1)))
        return dense

    def generate_noise_mask(self, area_min: int = 50, area_max: int = 200,
                            score: float = 0.45,
                            class_id: int = 0) -> Optional[MaskData]:
        """
        生成一个随机噪声 mask（模拟误检）。

        Args:
            area_min/max: 噪声区域面积范围 (像素)
            score: 低置信度
            class_id: 类别

        Returns:
            MaskData 或 None
        """
        mask = np.zeros((self.crop_h, self.crop_w), dtype=np.uint8)
        # 随机位置绘制小斑块
        cx = np.random.randint(self.crop_w // 4, 3 * self.crop_w // 4)
        cy = np.random.randint(self.crop_h // 4, 3 * self.crop_h // 4)
        area = np.random.randint(area_min, area_max)
        radius = int(np.sqrt(area / np.pi))
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        if cv2.countNonZero(mask) < area_min:
            return None
        return MaskData(mask, class_id, score)
