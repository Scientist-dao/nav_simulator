"""
simulator.py - 视觉导航闭环仿真器核心

实现完整的闭环仿真流程：
  车辆运动 → 场景可见性 → Mask 生成 → 导航算法 → 导航输出 → 车辆运动

每个仿真步对应一个相机帧（默认 15fps）。
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Callable

import sys, os

from .baseline_lib import (
    import_baseline_benchmark,
    import_baseline_fixture,
    import_baseline_navigation,
)

from .field_model import FieldModel, FieldConfig
from .vehicle_model import VehicleModel, VehicleParams, VehicleState
from .sensor_simulator import SensorSimulator, IMUConfig, RTKConfig
from .perturbation import PerturbationConfig, PerturbationApplier
from ..rendering.mask_generator import MaskGenerator
from ..rendering.camera_renderer import CameraRenderer


# sensor mode 名称 (与 C++ SensorExperimentMode 对应, GUI 显示用)
SENSOR_MODE_NAMES = {0: "VISION", 1: "VISION+IMU", 2: "VISION+IMU+RTK"}


def _load_rowmap_cpp_module():
    """加载 rowmap_navigation 工程编译的 C++ 模块

    产物模块名为 vision_navigation (部署形态: 直接替换板端基线库
    .so, 导入路径不变)。加载策略:
      - 候选目录依次找扩展产物 (VS 多配置生成器在 build-pc/Release,
      单配置生成器在 build-pc/, 部署同步目录 lib/);
      - 临时前插 sys.path 显式解析裸名, 不与基线库的
      navigation.vision_navigation 子模块路径冲突;
      - 若裸名已被基线库版本占用 sys.modules 缓存则先弹出
        (以 __file__ 路径区分)。
    """
    import glob
    import importlib

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for sub in ("build-pc" + os.sep + "Release", "build-pc", "lib"):
        d = os.path.join(root, "rowmap_navigation", sub)
        if not os.path.isdir(d):
            continue
        hits = [h for h in glob.glob(os.path.join(d, "vision_navigation.*"))
                if h.endswith((".pyd", ".so"))]
        if not hits:
            continue
        cached = sys.modules.get("vision_navigation")
        if cached is not None and "rowmap_navigation" not in str(
                getattr(cached, "__file__", "")):
            del sys.modules["vision_navigation"]
        sys.path.insert(0, d)
        try:
            return importlib.import_module("vision_navigation")
        finally:
            try:
                sys.path.remove(d)
            except ValueError:
                pass
    raise ImportError(
        "rowmap_cpp 后端未找到编译产物 vision_navigation.pyd/.so — "
        "先构建: cmake --build rowmap_navigation/build-pc --config Release")


def _draw_dashed_polyline(img: np.ndarray, pts, color, thickness: int = 2,
                           dash: int = 12, gap: int = 8):
    """沿折线画虚线 (每 dash 像素画一段, 隔 gap 像素空一段)

    用于相机视图的漏检行/未采用特征线标注。
    """
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        seg_len = float(np.hypot(x2 - x1, y2 - y1))
        if seg_len < 1e-6:
            continue
        ux, uy = (x2 - x1) / seg_len, (y2 - y1) / seg_len
        d = 0.0
        drawing = True
        while d < seg_len:
            step = dash if drawing else gap
            nd = min(d + step, seg_len)
            if drawing:
                cv2.line(img,
                         (int(x1 + ux * d), int(y1 + uy * d)),
                         (int(x1 + ux * nd), int(y1 + uy * nd)),
                         color, thickness, cv2.LINE_AA)
            d = nd
            drawing = not drawing


@dataclass
class SimConfig:
    """仿真配置"""
    fps: float = 15.0                   # 仿真帧率
    speed_ms: float = 1.0               # 车辆速度 (m/s)
    strip_width_px: int = 30            # mask 条带宽度 (像素)
    noise_probability: float = 0.0      # 噪声 mask 概率 [0, 1]
    z_near: float = 0.5                 # 最近可见距离 (m)
    z_far: float = 20.0                 # 最远可见距离 (m)
    x_range: float = 3.0                # 横向可见范围 (m)
    initial_offset_m: float = 0.0       # 初始横向偏移 (m)
    initial_heading_deg: float = 0.0    # 初始航向偏差 (度)
    test_mode: bool = True              # 导航核心 test_mode
    nav_debug: bool = False             # 导航核心 debug 输出
    max_frames: int = 0                 # 最大帧数 (0 = 无限制)
    camera_shake: bool = False          # 相机画面抖动 (默认关闭)

    # ── 传感器仿真 (RTK/IMU, 默认关闭保持向后兼容) ──
    # mode: 0=纯视觉 1=+IMU 姿态补偿 2=+RTK ENU hint (与 C++ 默认一致)
    sensor_mode: int = 2
    imu_config: Optional[IMUConfig] = None   # IMU 姿态仿真配置
    rtk_config: Optional[RTKConfig] = None   # RTK 定位仿真配置
    attitude_in_projection: bool = False    # 车身姿态进 mask 投影
                                            # (相机随车身俯仰/侧倾, 给 IMU
                                            #  补偿提供真实扰动源)
    sensor_seed: Optional[int] = None       # 传感器噪声随机种子 (可复现)
    bump_intensity: float = 1.0             # 颠簸强度倍率 (标称 1.0)
    
    # ── 感知扰动注入 (P0 仿真基建, 默认关闭保持向后兼容) ──
    # 遮挡/缺苗/弱苗三类生产痛点注入, 见 perturbation.py
    perturbation: Optional[PerturbationConfig] = None

    # ── 导航后端选择 (A/B 对比开关) ──
    # "baseline"   = C++ 基线 VisionNavigation (.pyd)
    # "rowmap"     = RowMap 参考实现 (行地图+ID 锁定, 接口同构)
    # "rowmap_cpp" = RowMap C++ 工程 (rowmap_navigation 编译产物,
    #                模块名/接口与既有板端库兼容 —— 部署形态验证)
    nav_backend: str = "baseline"


@dataclass
class FrameResult:
    """单帧仿真结果"""
    frame_idx: int = 0
    sim_time: float = 0.0
    # 车辆状态
    veh_x: float = 0.0
    veh_y: float = 0.0
    veh_heading_deg: float = 0.0
    veh_speed: float = 0.0
    veh_steering_deg: float = 0.0
    body_roll_deg: float = 0.0        # 车身侧倾角 (度)
    body_pitch_deg: float = 0.0       # 车身俯仰角 (度)
    # 导航输出
    yaw_deg: float = 0.0
    front_dist: float = 0.0
    rear_dist: float = 0.0
    image_quality: int = 0
    nav_valid: bool = False
    num_masks: int = 0
    control_mode: str = 'WORK'  # 控制器状态机模式 (WORK / 地头转向段)
    # 真值与注入状态 (P0 指标体系用)
    cte_true: float = 0.0       # 真值横向偏差 (车心到目标行中心的垂向 x 偏移)
    occluded: bool = False      # 本帧感知输入被注入遮挡
    # 行检出统计与扰动标签 (P0 可视化用)
    rows_expected: int = 0      # 画面内物理可见行数 (含断垄/弱苗后仍可见的)
    rows_detected: int = 0      # 实际生成 mask 的行数 (漏检 = expected - detected)
    perturb_tags: str = ""     # 当前激活扰动 ASCII 标签 (如 "OCCL 2.3s+GAP")
    # 传感器状态 (RTK/IMU 仿真, 供 GUI/CSV 输出)
    sensor_mode: int = 2
    imu_enabled: bool = False         # IMU 姿态仿真已启用
    rtk_enabled: bool = False         # RTK 定位仿真已启用
    imu_applied: bool = False         # C++ 本帧实际应用了 IMU 姿态补偿
    imu_roll_deg: float = 0.0         # C++ 实际应用的 roll (度, 未生效=0)
    imu_pitch_deg: float = 0.0        # C++ 实际应用的 pitch (度, 未生效=0)
    rtk_hint_valid: bool = False      # C++ 本帧 ENU hint 生效 (mode 2)
    rtk_pos_status: int = 0           # 最近包定位状态 (4=固定解)
    rtk_packets_frame: int = 0        # 本帧注入数据包数
    rtk_packets_total: int = 0        # 累计注入数据包数
    # 叠层数据
    nav_line_pixels: list = field(default_factory=list)
    nav_line_world: list = field(default_factory=list)   # 导航线世界坐标端点
    overlay_data: Optional[dict] = None
    # mask 可视化（用于显示）
    mask_image: Optional[np.ndarray] = None


class NavSimulator:
    """
    视觉导航闭环仿真器

    整合 FieldModel, VehicleModel, MaskGenerator, VisionNavigation，
    实现实时闭环仿真。
    """

    def __init__(self, field_config: FieldConfig, sim_config: SimConfig,
                 vehicle_params: VehicleParams = None):
        """
        Args:
            field_config: 田块场景配置
            sim_config: 仿真参数配置
            vehicle_params: 车辆参数
        """
        self.sim_config = sim_config
        # 全局 RNG 播种: bump 姿态噪声/图像噪声/mask 概率等走全局
        # np.random 流 (与 sensor/perturbation 的独立种子流并存)。
        # 不播种则同配置跨进程结果不同 —— GUI 手跑与 headless 跑批
        # 各自随机, 稀有失效模式 (如断垄误掉头) 无法复现定位
        if sim_config.sensor_seed is not None:
            np.random.seed(sim_config.sensor_seed)
        self.field_model = FieldModel(field_config)
        self.vehicle = VehicleModel(vehicle_params or VehicleParams())

        # 颠簸强度 (传感器场景可调大, 使姿态扰动可观察)
        self.vehicle.bump_intensity = getattr(sim_config, 'bump_intensity', 1.0)

        # 传感器仿真器 (RTK/IMU; 始终构造, 未启用时 generate_packets 返回空)
        # RowMap 兜底必须在 SensorSimulator 定型前注入: 仿真器构造时
        # 拷贝配置, 晚到的 rtk_config 不可见 (active 恒 False,
        # RTK 包永不生成 → 位姿无法建立)
        self._ensure_rowmap_rtk()
        self.sensor_sim = SensorSimulator(
            sim_config.imu_config, sim_config.rtk_config,
            seed=sim_config.sensor_seed)

        # 感知扰动注入器 (遮挡/缺苗/弱苗; 无配置时不构造)
        self.perturbation = (PerturbationApplier(sim_config.perturbation)
                             if sim_config.perturbation is not None else None)

        # 初始化导航核心（复用 benchmark 的 fixture）
        self._init_navigation()

        # 初始化 mask 生成器
        self.mask_gen = MaskGenerator(self._camera_info)
        self._horizon_row = self._find_horizon_row()

        # 初始化相机图像渲染器（用于 Camera View 仿真背景）
        self._camera_renderer = CameraRenderer(
            self.crop_w, self.crop_h, self.mask_gen,
            row_spacing=field_config.row_spacing_m)

        # 仿真状态
        self.frame_idx = 0
        self.sim_time = 0.0
        self.running = False
        self.paused = False
        self.results: List[FrameResult] = []
        self._distance_traveled = 0.0   # 累计行驶路程 (m, 弯道/掉头也准确)
        self._prev_pos: Optional[tuple] = None  # 上一帧位置 (路程累积用)

        # 场景事件回调（用于行尾等动态场景）
        self._scenario_update_fn: Optional[Callable] = None

        # 初始车辆位置：在田块起点，可设初始偏移
        self._reset_vehicle()

    def _ensure_rowmap_rtk(self):
        """RowMap 后端兜底: 场景未配 RTK 时自动启用仿真 RTK (幂等)。

        RowMap 参考实现位姿基准依赖 RTK ENU hint (设计边界: 世界系
        锚定由首包 lat/lon 建立)。场景未配置 RTK 时 _pos 恒 None
        → 观测无法反投影建图 → 永久 q=0 → 开局盲掉头循环。
        计划引入纯视觉位姿后移除。
        """
        backend = str(getattr(self.sim_config, 'nav_backend', 'baseline'))
        if backend in ("rowmap", "rowmap_cpp") \
                and self.sim_config.rtk_config is None:
            self.sim_config.rtk_config = RTKConfig(enabled=True)
            print("[ROWMAP] 位姿基准依赖 RTK hint: 场景未配置传感器, "
                  "已自动启用仿真 RTK (参考实现已知边界)")

    def _init_navigation(self):
        """初始化导航核心（与 benchmark 一致的相机参数）

        nav_backend 选档: "baseline" 用基线 C++ VisionNavigation
        (需板端基线扩展模块, 见 baseline_lib),
        "rowmap" 用行结构地图参考实现 (鸭子类型同接口),
        "rowmap_cpp" 用 rowmap_navigation 工程的 C++ 实现
        (接口与既有板端库签名兼容 —— 部署形态验证)。

        相机参数优先取基线工程夹具 (真实标定同源); 无基线工程时
        回退 core/camera_params.py 固化的同值演示参数。
        """
        fixture_build = import_baseline_fixture()
        if fixture_build is not None:
            self._camera_info = fixture_build()
        else:
            from .camera_params import build_camera_info as build_demo_camera_info
            self._camera_info = build_demo_camera_info()
        backend = str(getattr(self.sim_config, 'nav_backend', 'baseline'))

        if backend == "rowmap":
            # RowMap 位姿基准依赖 RTK hint, 场景未配 RTK 时兜底注入
            # (详见 _ensure_rowmap_rtk; __init__ 已在 SensorSimulator
            # 定型前调用过, 此处幂等保持独立调用 _init_navigation 的鲁棒性)
            self._ensure_rowmap_rtk()
            from nav_simulator.rowmap_proto import RowMapNavigation
            vn = RowMapNavigation()
            crop_h, crop_w = self._camera_info.crop_h, self._camera_info.crop_w
        elif backend == "rowmap_cpp":
            # RowMap C++ 工程: 同 RTK 兑底, 按既有板端接口签名构造
            # (相机参数与 baseline 同源; wheelbase 2.7m 同夹具
            # _WHEELBASE_LENGTH, z_front/z_rear = ±半轴距 = ±1.35
            # 与参考实现常量一致)
            self._ensure_rowmap_rtk()
            cpp_mod = _load_rowmap_cpp_module()
            vn = cpp_mod.VisionNavigation(
                wheelbase_length=2.7,
                camera_info=vars(self._camera_info),
                image_height=self._camera_info.crop_h,
                vehicle_center_point_x_thresh=0.3,
                test_mode=self.sim_config.test_mode,
            )
            crop_h, crop_w = self._camera_info.crop_h, self._camera_info.crop_w
        else:
            fixture = import_baseline_benchmark()
            vn, _, (crop_h, crop_w) = fixture.build_vision_navigation(
                test_mode=self.sim_config.test_mode)
            self._mask_cpp_cls = import_baseline_navigation()[1]

        vn.set_debug(self.sim_config.nav_debug)
        vn.set_debug_verbose(False)

        self._vn = vn
        self._nav_backend = backend
        self.crop_h = crop_h
        self.crop_w = crop_w

        # sensor mode: 0=纯视觉 1=+IMU 2=+RTK (与 C++ SensorExperimentMode
        # 对应; 无传感器数据时高层自动退化, 默认 2 不改变现有行为)
        self._vn.set_sensor_mode(int(getattr(
            self.sim_config, 'sensor_mode', 2)))

        # 全黑图像（导航算法只需 mask，图像用于 overlay 绘制参考）
        self._black_img = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

    def _reset_vehicle(self):
        """重置车辆到初始位置"""
        cfg = self.sim_config
        # 车辆在田块起点，中心行位置 + 初始偏移
        start_y = 0.0
        center_x = self.field_model.get_center_x_at_y(start_y)
        start_x = center_x + cfg.initial_offset_m

        self.vehicle.reset(
            x=start_x, y=start_y,
            heading=np.radians(cfg.initial_heading_deg),
            speed=cfg.speed_ms)
        self.frame_idx = 0
        self.sim_time = 0.0
        self.results.clear()
        self._distance_traveled = 0.0
        self._prev_pos = None
        self.sensor_sim.reset()
        if self.perturbation is not None:
            self.perturbation.reset()
        # 姿态外参复位到 nominal (attitude_in_projection 开启时)
        self.mask_gen.set_attitude(0.0, 0.0)

    def set_scenario_update_fn(self, fn: Callable):
        """设置场景更新回调（每帧调用，用于动态场景）"""
        self._scenario_update_fn = fn

    def reset(self):
        """重置仿真"""
        self._reset_vehicle()
        self._vn.force_row_reset()

    def step(self) -> FrameResult:
        """
        执行一个仿真步（一帧）。

        Returns:
            FrameResult 包含本帧所有数据
        """
        dt = 1.0 / self.sim_config.fps
        self.sim_time += dt
        self.frame_idx += 1

        # ── 1. 场景事件更新 ──
        if self._scenario_update_fn is not None:
            self._scenario_update_fn(self)

        # ── 2. 获取可见行（车辆坐标系）──
        vs = self.vehicle.state
        visible_rows = self.field_model.get_visible_rows_vehicle(
            veh_x=vs.x, veh_y=vs.y, veh_heading=vs.heading,
            z_near=self.sim_config.z_near,
            z_far=self.sim_config.z_far,
            x_range=self.sim_config.x_range,
        )

        # 路程累积 (帧间位移累加, 弯道/掉头也准确; 修复旧版直接读 y 的问题)
        if self._prev_pos is not None:
            dx, dy = vs.x - self._prev_pos[0], vs.y - self._prev_pos[1]
            self._distance_traveled += float(np.hypot(dx, dy))
        self._prev_pos = (vs.x, vs.y)

        # 真值 CTE: 车心到目标行 (中心行) 的横向偏差 (指标体系用;
        # 默认场景车骑中心行, 行形状任意时 get_center_x_at_y 均适用)
        cte_true = vs.x - self.field_model.get_center_x_at_y(vs.y)

        # ── 2.5 车身姿态进相机投影（可选）──
        # 相机随车身俯仰/侧倾: mask 与渲染背景同步受扰动 (真实车载相机),
        # 为 IMU 姿态补偿 (mode>=1) 提供可验证的扰动源; 关闭时投影外参
        # 恒为 nominal (相机理想水平安装)
        if self.sim_config.attitude_in_projection:
            self.mask_gen.set_attitude(vs.roll, vs.pitch)

        # ── 2.6 感知扰动注入 + 生成 mask (遮挡/缺苗/弱苗, P0) ──
        # 注入后每行可携带独立 score (弱苗降置信度), 逐行生成 mask;
        # 遮挡窗内注入结果为空列表 → mask 为空 → 感知完全失效。
        # 逐行生成同时统计 rows_detected (真正产出 mask 的行),
        # 与物理可见行数 rows_expected 之差即漏检行, 供可视化标注
        occluded = False
        perturb_tags = ""
        physical_rows = visible_rows      # 相机背景/俯视图用物理现实行
        detected_row_ids = set()          # 本帧实际检出 (生成 mask) 的行号
        if self.perturbation is not None:
            occluded = self.perturbation.occluded_now(self.sim_time)
            perturb_tags = self.perturbation.active_tags(self.sim_time)
            physical_rows = self.perturbation.apply_physical(
                visible_rows, vs.x, vs.y, vs.heading)
            injected = self.perturbation.apply(
                visible_rows, vs.x, vs.y, vs.heading, self.sim_time)
            masks_py = []
            for row, score_ov in injected:
                n_before = len(masks_py)
                masks_py.extend(self.mask_gen.generate_masks(
                    [row],
                    strip_width_px=self.sim_config.strip_width_px,
                    score=(score_ov if score_ov is not None
                           else self.perturbation.cfg.base_score)))
                if len(masks_py) > n_before:
                    detected_row_ids.add(row[0])
        else:
            # 无扰动路径同样逐行生成 (与多行批量调用等价: 物理宽度
            # 投影对每行独立), 保持检出统计口径一致
            masks_py = []
            for row in visible_rows:
                n_before = len(masks_py)
                masks_py.extend(self.mask_gen.generate_masks(
                    [row], strip_width_px=self.sim_config.strip_width_px))
                if len(masks_py) > n_before:
                    detected_row_ids.add(row[0])

        # ── 3. 生成 mask ──
        # (扰动注入模式下已在 2.6 步逐行生成, 此处仅补充噪声 mask)

        # 添加噪声 mask
        if self.sim_config.noise_probability > 0:
            if np.random.random() < self.sim_config.noise_probability:
                noise_mask = self.mask_gen.generate_noise_mask()
                if noise_mask is not None:
                    masks_py.append(noise_mask)

        # ── 4. 转换为 C++ 格式 (仅 baseline 后端需要;
        # rowmap/rowmap_cpp 直接吃鸭子类型 mask) ──
        masks_cpp = ([
            self._mask_cpp_cls(m.mask, m.classic_class, float(m.score))
            for m in masks_py
        ] if self._nav_backend == "baseline" else [])

        # ── 4.5 传感器数据注入 (RTK/IMU, 在导航解算前) ──
        # 50Hz 数据包按仿真时间戳均匀落在帧间 (末包 = 本帧时刻),
        # 模拟真实接收机异步到达; 与 frame_time 同源保证匹配窗 (0.1s) 命中
        rtk_packets_frame = 0
        if self.sensor_sim.active:
            packets = self.sensor_sim.generate_packets(
                self.sim_time, vs.x, vs.y, vs.heading, vs.speed,
                vs.roll, vs.pitch)
            for pkt in packets:
                self._vn.feed_rtk_data(pkt)
            rtk_packets_frame = len(packets)
            # 首包时刻把车辆真值位置喂给 RowMap 后端作为可视化换算原点
            # (地图活在 RTK 基准系, 俯视图需要田块系; 纯显示用不进算法,
            # C++ 基线无此接口自动跳过)
            if packets and hasattr(self._vn, 'set_rtk_world_origin'):
                self._vn.set_rtk_world_origin(vs.x, vs.y)

        # ── 5. 调用导航算法 ──
        # frame_time 用仿真时间: 与传感器包 recv_time 同源, 且不受渲染
        # 卡顿污染 (旧版墙钟在 GUI 掉帧时造成时序失真);
        # rowmap/rowmap_cpp 后端直接吃鸭子类型 mask (C++ 绑定按
        # .mask/.classic_class/.score 属性提取, 基线库 MaskDataPlus 亦兼容)
        if self._nav_backend in ("rowmap", "rowmap_cpp"):
            nav_masks = masks_py
        else:
            nav_masks = masks_cpp
        # 后端扩展 hint (鸭子类型, C++ 基线无此方法零影响): 控制模式
        # (CAPTURE/掉头段解锁重捕获引导的行跨度限制) 与车速 (航向 EMA
        # 系数自适应 —— 高速下固定 α 滞后导致投影错位/关联失败)
        if hasattr(self._vn, "set_control_hint"):
            self._vn.set_control_hint(self.vehicle.mode.value)
        if hasattr(self._vn, "set_speed_hint"):
            self._vn.set_speed_hint(self.vehicle.state.speed)
        yaw, front_dist, rear_dist, nav_line_px, quality = \
            self._vn.get_navigation_data(
                self._black_img, nav_masks, self.sim_time)

        # C5 行尾概率 hint (鸭子类型, C++ 基线无此接口零影响):
        # 导航侧 P(end) 喂控制器掉头触发门调制 (c5b 集成)
        if hasattr(self._vn, "get_headland_prob"):
            self.vehicle.set_headland_prob(
                self._vn.get_headland_prob())

        overlay_data = self._vn.get_overlay_data()

        nav_valid = (int(quality) > 0)

        # ── 5.5 传感器诊断 (C++ 本帧实际应用状态, 供 GUI/CSV) ──
        imu_applied = False
        imu_roll_deg = 0.0
        imu_pitch_deg = 0.0
        rtk_hint_valid = False
        rtk_pos_status = 0
        if self.sensor_sim.active:
            att = self._vn.get_attitude_diag()
            imu_applied = bool(att.get("fresh") and att.get("has_data"))
            imu_roll_deg = float(att.get("applied_roll_deg", 0.0))
            imu_pitch_deg = float(att.get("applied_pitch_deg", 0.0))
            rtkd = self._vn.get_rtk_position_diag()
            rtk_hint_valid = bool(rtkd.get("rtk_hint_valid"))
            rtk_pos_status = int(rtkd.get("rtk_pos_status", 0))

        # 导航线反投影到世界坐标 (数据输出 / 俯视图绘制)
        nav_line_world = self._nav_line_to_world(nav_line_px, vs)

        # 导航线弦方向角 + 弦中点距离 (车辆系) —— 弯道曲率前馈信号
        line_angle_deg, line_z_mid = self._nav_line_angle(nav_line_px)

        # ── 6. 生成 Camera View 可视化图像 ──
        mask_image = self._render_mask_image(
            masks_py, nav_line_px, overlay_data, physical_rows,
            detected_row_ids=detected_row_ids, occluded=occluded)

        # 相机随车身姿态抖动 (可选, 仅影响可视化, 不污染导航输入)
        if self.sim_config.camera_shake:
            mask_image = self._apply_camera_shake(
                mask_image, vs.roll, vs.pitch)

        # 车头引擎盖 (固定在车身/相机上, 若开启抖动必须在抖动后绘制:
        # 世界随姿态动、车头不动, 与真实车载相机一致; 前轮随转角摆动)
        self._camera_renderer.draw_hood(mask_image, vs.steering)

        # ── 7. 构建结果 ──
        result = FrameResult(
            frame_idx=self.frame_idx,
            sim_time=self.sim_time,
            veh_x=vs.x, veh_y=vs.y,
            veh_heading_deg=np.degrees(vs.heading),
            veh_speed=vs.speed,
            veh_steering_deg=np.degrees(vs.steering),
            body_roll_deg=np.degrees(vs.roll),
            body_pitch_deg=np.degrees(vs.pitch),
            yaw_deg=yaw,
            front_dist=front_dist,
            rear_dist=rear_dist,
            image_quality=int(quality),
            nav_valid=nav_valid,
            num_masks=len(masks_py),
            cte_true=float(cte_true),
            occluded=bool(occluded),
            rows_expected=len(physical_rows),
            rows_detected=len(detected_row_ids),
            perturb_tags=perturb_tags,
            sensor_mode=int(getattr(self.sim_config, 'sensor_mode', 2)),
            imu_enabled=self.sensor_sim.imu_cfg.enabled,
            rtk_enabled=self.sensor_sim.rtk_cfg.enabled,
            imu_applied=imu_applied,
            imu_roll_deg=imu_roll_deg,
            imu_pitch_deg=imu_pitch_deg,
            rtk_hint_valid=rtk_hint_valid,
            rtk_pos_status=rtk_pos_status,
            rtk_packets_frame=rtk_packets_frame,
            rtk_packets_total=self.sensor_sim.packets_sent,
            nav_line_pixels=nav_line_px,
            nav_line_world=nav_line_world,
            overlay_data=overlay_data,
            mask_image=mask_image,
        )

        # ── 8. 更新车辆运动（闭环）──
        # 掉头中止信号 (rowmap 后端独有): 掉头期间任一已有行延续锚点落图
        row_alive = (self._vn.get_row_alive()
                     if hasattr(self._vn, "get_row_alive") else False)
        self.vehicle.update(
            yaw_deg=yaw,
            cte_front=front_dist,
            cte_rear=rear_dist,
            dt=dt,
            nav_valid=nav_valid,
            line_angle_deg=line_angle_deg,
            line_z_mid=line_z_mid,
            quality=quality,
            row_alive=row_alive,
        )
        result.control_mode = self.vehicle.mode.value

        # 历史列表只存轻量数值帧: mask_image (~1.5MB)/overlay_data/
        # nav_line_pixels 逐帧累积, 长跑 (23k 帧 GUI 实测 2026-10-09)
        # 耗尽进程内存 OOM。GUI/录制消费的是 step() 返回的当帧
        # result (图像完整), 指标统计/数据工厂只读数值字段。
        # 注意必须在 control_mode 回填之后再 append (evaluate 按
        # 它过滤作业帧)。
        self.results.append(replace(
            result, mask_image=None, overlay_data=None,
            nav_line_pixels=[]))

        return result

    def run(self, num_frames: int = 0,
            callback: Optional[Callable[[FrameResult], bool]] = None):
        """
        运行仿真。

        Args:
            num_frames: 运行帧数 (0 = 直到 callback 返回 False)
            callback: 每帧回调，返回 False 停止仿真
        """
        self.running = True
        max_frames = num_frames or self.sim_config.max_frames

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            result = self.step()

            if callback is not None:
                if not callback(result):
                    break

            if max_frames > 0 and self.frame_idx >= max_frames:
                break

        self.running = False

    def stop(self):
        """停止仿真"""
        self.running = False

    def toggle_pause(self):
        """暂停/继续"""
        self.paused = not self.paused

    # 导航线远端端点的固定钳制距离 (端点在地平线以上时)
    _NAV_LINE_Z_FAR = 12.0

    def _find_horizon_row(self) -> int:
        """二分搜索地平线行号: 可反投影到地面的最小像素行 (初始化一次)"""
        col = self.mask_gen.crop_w // 2
        lo, hi = 0, self.mask_gen.crop_h - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.mask_gen.pixel_to_vehicle_xz(col, mid) is not None:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _px_to_veh_clamped(self, px, py):
        """像素点 → 车辆系地面坐标, 地平线以上时按固定远距钳制

        射线不与地面相交 (端点在地平线上) 时取同列地平线处的射线
        方向, 按 _NAV_LINE_Z_FAR 截断, 保证远端点也有坐标。
        完全失败返回 None。
        """
        res = self.mask_gen.pixel_to_vehicle_xz(px, py)
        if res is not None:
            return res
        res_h = self.mask_gen.pixel_to_vehicle_xz(px, self._horizon_row + 2)
        if res_h is None or res_h[1] <= 0:
            return None
        t = min(self._NAV_LINE_Z_FAR / res_h[1], 1.0)
        return (res_h[0] * t, res_h[1] * t)

    def _nav_line_angle(self, nav_line_px) -> tuple:
        """导航线弦方向角 (度, 右正) 与弦中点前向距离 z_mid (m)

        两点反投影到车辆系后取 atan2(dx, dz)。弯道上前视行段的弦
        方向指示前方曲率; 配合 z_mid 可还原几何曲率 κ≈tan(φ)/z_mid
        (圆的性质: 弦向角 = 弧中点切线转角), 供曲率前馈使用。
        任一端点完全反投影失败时返回 (0, 默认 4m)。
        """
        if not nav_line_px or len(nav_line_px) < 2:
            return 0.0, 4.0
        p0 = self._px_to_veh_clamped(*nav_line_px[0])
        p1 = self._px_to_veh_clamped(*nav_line_px[1])
        if p0 is None or p1 is None:
            return 0.0, 4.0
        dx, dz = p1[0] - p0[0], p1[1] - p0[1]
        if abs(dz) < 1e-3:
            return 0.0, 4.0
        z_mid = max((p0[1] + p1[1]) * 0.5, 0.5)
        return float(np.degrees(np.arctan2(dx, dz))), float(z_mid)

    def _nav_line_to_world(self, nav_line_px, vs) -> list:
        """导航线像素端点 → 世界坐标端点 (供数据输出和俯视图绘制)

        端点位于地平线以上时射线不与地面相交, 此时取同列地平线处的
        射线方向, 按固定远距 _NAV_LINE_Z_FAR 钳制, 保证始终输出两端点。
        """
        pts = []
        if not nav_line_px:
            return pts
        c, s = np.cos(vs.heading), np.sin(vs.heading)
        for px, py in nav_line_px[:2]:
            res = self._px_to_veh_clamped(px, py)
            if res is None:
                continue
            xv, zv = res
            pts.append((vs.x + xv * c + zv * s,
                        vs.y - xv * s + zv * c))
        return pts

    def _apply_camera_shake(self, img: np.ndarray,
                            roll_rad: float, pitch_rad: float) -> np.ndarray:
        """
        对 Camera View 图像施加车身姿态引起的相机抖动。

        roll  → 图像微旋转 + 水平微移
        pitch → 图像垂直微移
        叠加亚像素级高频振动。仅作用于可视化图像,
        导航算法的 mask 输入保持干净。
        """
        h, w = img.shape[:2]
        roll_deg = np.degrees(roll_rad)
        pitch_deg = np.degrees(pitch_rad)

        # 侧倾表现为画面旋转 (4x 视觉增益: 物理侧倾仅 0.1-0.5°,
        # 不放大几乎不可见; 车头在 warp 后绘制保持固定, 形成参照)
        angle = roll_deg * 4.0
        dx = roll_deg * 1.5 + np.random.normal(0, 0.4)
        dy = pitch_deg * 6.0 + np.random.normal(0, 0.4)

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _render_mask_image(self, masks_py, nav_line_px, overlay_data,
                           background_rows=None, detected_row_ids=None,
                           occluded=False):
        """渲染 Camera View 可视化图像 (所见即所算)

        背景用物理现实行 (断垄区空土/弱苗区细苗带), 遮挡时叠加杂草
        覆盖层; 特征线按是否被采用着色 (亮绿实线/黄虚线), 漏检行
        红虚线+MISS 标注——画面所见与算法所算逐项对得上:
          背景里看得到的行 → 应该有绿色 mask;
          没有的 → 红虚线 MISS (漏检/被遮挡);
          C++ 特征线亮绿 → 参与了导航线解算。
        """
        # 1. 背景渲染: 物理现实行 (gap/weak 后的田里真貌)
        if background_rows is not None:
            vs = self.vehicle.state
            img = self._camera_renderer.render(
                background_rows, veh_x=vs.x, veh_y=vs.y,
                veh_heading=vs.heading,
                add_noise=False, add_texture=True)
        else:
            img = np.zeros((self.crop_h, self.crop_w, 3), dtype=np.uint8)
            img[:] = (60, 85, 140)  # 土棕色背景

        # 2. 遮挡覆盖层 (杂草/尘土糊镜头; 在 mask 之下, 完全遮挡时
        #    mask 为空, 画面只剩杂草块与警示框)
        if occluded:
            self._draw_occlusion(img)

        # 3. 叠加半透明 mask（保持原有色彩标识）
        colors = [
            (56, 255, 56),    # class 0: vegetation_line → 绿色
            (255, 255, 56),   # class 1: mulch_line → 青色
            (255, 56, 56),    # class 2: road → 蓝色
        ]
        for m in masks_py:
            cls = m.classic_class
            color = colors[cls % len(colors)]
            colored = np.zeros_like(img)
            colored[m.mask > 0] = color
            img = cv2.addWeighted(img, 1.0, colored, 0.35, 0)

        # 4. 绘制导航线（醒目红色粗线）
        if nav_line_px and len(nav_line_px) >= 2:
            pt1 = (int(nav_line_px[0][0]), int(nav_line_px[0][1]))
            pt2 = (int(nav_line_px[1][0]), int(nav_line_px[1][1]))
            cv2.line(img, pt1, pt2, (0, 0, 255), 3)

        # 5. 特征线: 被采用参与导航解算的行 → 亮绿粗实线+端点;
        #    检出但未采用的行 → 黄色虚线
        if overlay_data:
            center_lines = overlay_data.get("center_lines", [])
            line_used = overlay_data.get("line_used", [])
            for i, line in enumerate(center_lines):
                if len(line) < 2:
                    continue
                pts = [(int(p[0]), int(p[1])) for p in line]
                used = (i < len(line_used) and line_used[i])
                if used:
                    for j in range(len(pts) - 1):
                        cv2.line(img, pts[j], pts[j + 1], (60, 255, 60), 3)
                    cv2.circle(img, pts[0], 4, (60, 255, 60), -1)
                    cv2.circle(img, pts[-1], 4, (60, 255, 60), -1)
                else:
                    _draw_dashed_polyline(img, pts, (0, 200, 255), 2)

        # 6. 漏检行标注: 背景里物理可见但未生成 mask 的行 → 红虚线
        #    (完全遮挡时背景被覆盖层挡住, 逐行标注无意义, 跳过)
        if (not occluded and detected_row_ids is not None
                and background_rows is not None):
            for row_idx, pts_vehicle, _cls, _w in background_rows:
                if row_idx in detected_row_ids:
                    continue
                px_pts = []
                for xv, zv in pts_vehicle:
                    col, row = self.mask_gen._vehicle_xz_to_pixel(xv, zv)
                    if 0 <= col < self.crop_w and 0 <= row < self.crop_h:
                        px_pts.append((col, row))
                if len(px_pts) < 2:
                    continue
                _draw_dashed_polyline(img, px_pts, (0, 0, 255), 2)
                # MISS 标注锚点取行中段: 底部近端被引擎盖遮挡
                # (draw_hood 在本函数之后绘制, 覆盖 y>0.86h),
                # 远端各线汇聚重叠, 中段最清晰
                y_lo = int(self.crop_h * 0.45)
                y_hi = int(self.crop_h * 0.82)
                band = [p for p in px_pts if y_lo <= p[1] <= y_hi]
                anchor = (band[len(band) // 2] if band
                          else px_pts[len(px_pts) // 2])
                cv2.putText(img, "MISS", (anchor[0] + 8, anchor[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 255), 1, cv2.LINE_AA)

        # 7. 绘制车辆前轴参考线（车身宽度指示）
        if overlay_data:
            afl = overlay_data.get("axle_front_left", [])
            afr = overlay_data.get("axle_front_right", [])
            if len(afl) >= 2 and len(afr) >= 2:
                # 左边界线（白色）
                p1 = (int(afl[0][0]), int(afl[0][1]))
                p2 = (int(afl[1][0]), int(afl[1][1]))
                cv2.line(img, p1, p2, (255, 255, 255), 2)
                # 右边界线（白色）
                p1 = (int(afr[0][0]), int(afr[0][1]))
                p2 = (int(afr[1][0]), int(afr[1][1]))
                cv2.line(img, p1, p2, (255, 255, 255), 2)
                # 前轴连接线（黄色虚线效果）
                cv2.line(img,
                         (int(afl[0][0]), int(afl[0][1])),
                         (int(afr[0][0]), int(afr[0][1])),
                         (0, 255, 255), 1)

            # 8. 绘制车辆中轴线
            vl = overlay_data.get("vehicle_line", None)
            if vl and len(vl) >= 2:
                p1 = (int(vl[0][0]), int(vl[0][1]))
                p2 = (int(vl[1][0]), int(vl[1][1]))
                cv2.line(img, p1, p2, (255, 255, 255), 1)

        # 9. 图例 (天空区一行小字, 白天背景用深色字)
        cv2.putText(img,
                    "FEAT: green=used  yellow dash=not used  "
                    "red dash=row missed",
                    (10, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (35, 35, 35), 1, cv2.LINE_AA)

        return img

    def _draw_occlusion(self, img: np.ndarray):
        """遮挡覆盖层: 模拟杂草/尘土糊住镜头

        遮挡物贴在镜头上, 不随车动 (画面内固定), 用固定 seed 生成;
        叠加红色警示边框与 OCCLUDED 横幅, 一眼可辨"当前看不见"。
        """
        h, w = img.shape[:2]
        rng = np.random.RandomState(20260826)
        overlay = img.copy()
        palette = [(35, 60, 30), (50, 90, 40), (60, 70, 35), (30, 45, 25)]
        # 大团杂草块 (覆盖画面主体)
        for _ in range(16):
            cx, cy = rng.randint(0, w), rng.randint(int(h * 0.30), h)
            r = rng.randint(int(w * 0.12), int(w * 0.30))
            cv2.circle(overlay, (cx, cy), r,
                       palette[rng.randint(len(palette))], -1)
        # 尘土颗粒
        for _ in range(400):
            nx, ny = rng.randint(0, w), rng.randint(int(h * 0.25), h)
            cv2.circle(overlay, (nx, ny), rng.randint(1, 3),
                       palette[rng.randint(len(palette))], -1)
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

        # 红色警示边框 + 顶部横幅
        cv2.rectangle(img, (3, 3), (w - 4, h - 4), (0, 0, 255), 4)
        banner = "OCCLUDED - NO PERCEPTION"
        (tw, th), _ = cv2.getTextSize(
            banner, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bx = (w - tw) // 2
        by = int(h * 0.24)
        ov = img.copy()
        cv2.rectangle(ov, (bx - 14, by - th - 10),
                      (bx + tw + 14, by + 10), (0, 0, 160), -1)
        cv2.addWeighted(ov, 0.7, img, 0.3, 0, img)
        cv2.putText(img, banner, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)

    def get_summary(self) -> Dict:
        """获取仿真统计摘要"""
        if not self.results:
            return {}

        yaws = [r.yaw_deg for r in self.results if r.nav_valid]
        fronts = [r.front_dist for r in self.results if r.nav_valid]
        qualities = [r.image_quality for r in self.results]
        valid_count = sum(1 for r in self.results if r.nav_valid)

        return {
            "total_frames": len(self.results),
            "valid_frames": valid_count,
            "valid_rate": valid_count / max(1, len(self.results)),
            "yaw_mean": float(np.mean(yaws)) if yaws else float('nan'),
            "yaw_std": float(np.std(yaws)) if yaws else float('nan'),
            "front_dist_mean": float(np.mean(fronts)) if fronts else float('nan'),
            "front_dist_std": float(np.std(fronts)) if fronts else float('nan'),
            "quality_mean": float(np.mean(qualities)),
            "final_position": (self.vehicle.state.x, self.vehicle.state.y),
            "distance_traveled": self._distance_traveled,
        }

