"""camera_params.py - 演示农机相机参数 (预计算派生值)

数值来源于真实农机的相机标定 (1080x1280 前视相机, 俯角 10 度, 安装高
2.88 m), 经 "去畸变 -> ROI 裁剪 -> 等比缩放到高 800 -> 居中裁剪到 16
对齐" 流程派生后固化为本文件字面量 —— 生成派生值的基线工程不随本仓库
分发, 这里只保留结果数值, 保证无基线工程时仿真器行为完全一致。

消费方 (字段 -> 使用处):
  - K_final / crop_w / crop_h : 裁剪后图像内参与尺寸 (mask 生成/投影/
    可视化); C++ 绑定的 camera_info 字典读取 K_final, T_c2v,
    camera_rad, crop_w, crop_h;
  - R_v2c / T_v2c             : 车辆系 -> 相机系外参 (地面点投影像素);
  - R_c2v / T_c2v             : 相机系 -> 车辆系外参;
  - vehicle_left_x / vehicle_right_x : 车体横向边界 (含宽度偏移)。
"""

import numpy as np


class DemoCameraInfo:
    """与基线工程 CameraInfo 同构的鸭子类型载体 (仅仿真器所需字段)。"""

    def __init__(self):
        # 裁剪后图像内参 (656x800 crop)
        self.K_final = np.array([
            [440.83912165410857, 0.0, 326.9140505385704],
            [0.0, 438.9204018116055, 379.3306789222569],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        # 外参: 相机系 <-> 车辆系 (俯角 10 度, 安装高 2.88 m)
        self.R_c2v = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.984807753012208, 0.17364817766693033],
            [0.0, -0.17364817766693033, 0.984807753012208],
        ], dtype=float)
        self.T_c2v = np.array([[0.0], [-2.88], [-1.5]], dtype=float)
        self.R_v2c = self.R_c2v.T
        self.T_v2c = -np.dot(self.R_v2c, self.T_c2v)
        # 安装参数
        self.camera_rad = 0.17453292519943295      # deg2rad(10.0)
        self.camera_height = 2.88                  # m
        self.camera_center_line_dist = 0.0         # m
        self.camera_front_axle_dist = -1.5         # m (相机位于前轴后方)
        # 图像尺寸 (等比缩放后居中裁剪, 16 对齐)
        self.crop_w = 656
        self.crop_h = 800
        self.scaled_w = 667
        self.scaled_h = 800
        # 车体横向边界 (轮距 1.72 m + 两侧 0.5 m 宽度偏移)
        self.vehicle_left_x = -1.3599999999999999
        self.vehicle_right_x = 1.3599999999999999


def build_camera_info() -> DemoCameraInfo:
    """构造演示农机相机参数 (与基线工程夹具数值一致)。"""
    return DemoCameraInfo()
