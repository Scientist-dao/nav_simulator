"""
nav_core.py - RowMap 导航核心 (参考实现): 接口与 VisionNavigation (C++) 对齐

simulator 通过 SimConfig.nav_backend = "rowmap" 切换本核心, 调用面
(方法签名/返回值/诊断 dict) 与 C++ 基线完全一致, 同台 A/B 无侵入。

算法流水 (每帧 get_navigation_data):
  1. 位姿: RTK 包 (feed_rtk_data 累积) → ENU 位置 + 双天线航向 EMA。
     首包 lat/lon 为基准原点 —— 地图活在"RTK 基准系"里, 基准偏移
     不影响相对量 (CTE/yaw), 无需绝对定位。
  2. 观测提取: mask 按图像行扫描取列均值 → 反投影车辆系 (外参含
     IMU 姿态补偿, mode>=1) → 变换世界系点串, 方差随距离增长
     (像素噪声 + 航向误差放大 + RTK 位置噪声)。
  3. 地图更新: 行级关联 (最近行, 门限 0.35m) + 锚点 1/var 融合;
     无观测帧 (遮挡/断垄) 跳过 —— 地图保持。
  4. 解算: 任务行前段锚点 (按航向方向选段, 支持沿行双向作业) →
     车辆系 → 加权二次曲线拟合 (c≈0 时自动退化为直线, 直行/弯道
     统一表达) → CTE/yaw。解算输入不含本帧图像,
     感知中断时由地图+RTK 外推。
  5. quality: 本帧观测命中任务行=95; 中断但地图充足=70 (nav_valid
     保持, 不触发地头误掉头 —— D2 机理修复); 前向段拟合失败但
     地图存在=50 (重捕获引导: 查表修回行带, 不写地图); 地图
     不足=0。

当前范围: sensor mode 2 (RTK 主路径, 痛点场景全部 mode 2)。
mode 0/1 无位置源时 quality=0 诚实降级 (视觉里程计位姿留后续版本)。
"""

import collections
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from nav_simulator.core.baseline_lib import import_baseline_fixture
from nav_simulator.rendering.mask_generator import MaskGenerator

from .row_map import RowMapCore


def _build_camera_info():
    """相机参数: 优先基线工程夹具, 无基线工程时回退本仓库固化参数。"""
    fixture_build = import_baseline_fixture()
    if fixture_build is not None:
        return fixture_build()
    from nav_simulator.core.camera_params import build_camera_info
    return build_camera_info()


class RowMapNavigation:
    """RowMap 导航核心参考实现 —— VisionNavigation 的鸭子类型替身"""

    # ── 参数 (参考实现阶段集中常量) ──
    K_PIX_SIGMA = 0.005       # 像素/分割横向噪声: 每 m 前向距离的 σ (m)
    SIGMA_RTK = 0.02          # RTK 位置噪声 (m, 与 RTKConfig 默认一致)
    SIGMA_HEADING_MEAS = math.radians(0.5)   # 双天线航向单包噪声 (rad)
    HEADING_EMA_ALPHA = 0.15  # 航向 EMA 平滑系数 (稳态 σ ≈ 0.13°;
                              # α 过大时 heading 噪声直接透传到 yaw 输出,
                              # 是 GUI 上航向角跳动的主源)
    OUTPUT_EMA_ALPHA = 0.4    # 输出端 (yaw/front/rear) 单极点平滑系数
                              # (15fps 下时间常数 ≈ 0.17s, 压噪声 ~60%)
    SCAN_STEP_PX = 8          # mask 图像行扫描采样步长 (px)
    Z_MIN_OBS = 0.3           # 观测前向距离范围 (m)
    Z_MAX_OBS = 20.0
    LOOKAHEAD_M = 10.0        # 解算取任务行前段长度 (m; 加长提高斜率估计
                              # 基线 → yaw 噪声下降, 远端锚点 var 大自动降权)
    Z_FRONT = 1.35            # 前轴参考距离 (半轴距, 与控制器符号约定对齐)
    Z_REAR = -1.35
    QUALITY_TRACKING = 95     # 本帧观测命中任务行
    QUALITY_DEAD_RECKON = 70  # 感知中断但地图充足 (遮挡/断垄外推)
    QUALITY_RECAPTURE = 50    # 前向拟合失败但地图存在: 查表引导回行带
    RECAPTURE_ANCHORS = 8     # 引导取最近锚点邻域的个数 (≈4m 拟合段)
    MIN_SOLVE_ANCHORS = 2     # 解算最少锚点数
    SEG_BACK_M = 0.5          # 前向段向车身后多取的长度 (m)
    QNET_MAX_DX = 0.15        # C3 网络 δx 纠偏安全限幅 (m)
    # ── C4 BEV 伪观测参数 ──
    BEV_P_THR = 0.6           # P 图提取伪观测的置信门槧
    BEV_PSEUDO_VAR = 0.0625   # 伪观测 var (σ=0.25m, 诚实保守: 比远端
                              # 手写视觉 var 大 ~2.5 倍, 融合中低权重)
    BEV_GATE_M = 0.2          # 伪观测守门: 与任务行均值偏差上限 (m);
                              # < 行距 0.6 的一半且 < GATE_ABS_M 0.35,
                              # 保证 update 关联到任务行而非邻行
    BEV_AHEAD_M = 18.0        # 伪观测只填前方缺口 (BEV 窗 32m, 取前 18m)
    BEV_BACK_M = 1.0          # 尾部回退 (m)
    BEV_VAR_GROW_FRAMES = 150 # 失明时长 var 增长系数 (150 帧 = 10s 后 ×2)
    # ── C4 BEV 伪观测参数 ──
    BEV_P_THR = 0.6           # P 图提取伪观测的置信门槧
    BEV_PSEUDO_VAR = 0.0625   # 伪观测 var (σ=0.25m, 诚实保守: 比远端
                              # 手写视觉 var 大 ~2.5 倍, 融合中低权重)
    BEV_GATE_M = 0.2          # 伪观测守门: 与任务行均值偏差上限 (m);
                              # < 行距 0.6 的一半且 < GATE_ABS_M 0.35,
                              # 保证 update 关联到任务行而非邻行
    BEV_AHEAD_M = 18.0        # 伪观测只填前方缺口 (BEV 窗 32m, 取前 18m)
    BEV_BACK_M = 1.0          # 尾部回退 (m)
    BEV_VAR_GROW_FRAMES = 150 # 失明时长 var 增长系数 (150 帧 = 10s 后 ×2)

    _METRES_PER_DEG = 111320.0

    def __init__(self):
        # 投影器: nominal 外参 + IMU 姿态补偿 (与 mask 生成同参数,
        # 不复用 simulator 实例 —— 那是真值姿态, 会偷看上帝视角)
        self._proj = MaskGenerator(_build_camera_info())
        # ── 位姿状态 (RTK 基准系) ──
        self._pos: Optional[Tuple[float, float]] = None    # (east, north)
        self._heading: Optional[float] = None              # rad 罗盘角 (EMA)
        self._heading_var = self.SIGMA_HEADING_MEAS ** 2
        self._origin: Optional[Tuple[float, float]] = None  # 首包 lat/lon 基准
        self._pos_status = 0
        # ── IMU 姿态 ──
        self._att: Tuple[float, float] = (0.0, 0.0)        # (roll, pitch) rad
        self._att_has = False      # 历史上收到过姿态
        self._att_fresh = False    # 自上次解算后收到新姿态
        self._att_applied_last = False  # 最近一帧解算实际应用了姿态补偿
                                     # (诊断用: get_attitude_diag 在解算后
                                     # 调用, _att_fresh 已被消费清零)
        # ── 地图与帧计数 ──
        self._map = RowMapCore()
        self._frame = 0
        self._sensor_mode = 2
        self._debug = False
        # ── 输出平滑状态 (单极点 EMA; 无效帧直通重置) ──
        self._out_valid_prev = False
        self._yaw_s = 0.0
        self._front_s = 0.0
        self._rear_s = 0.0
        # ── 输出缓存 (诊断/overlay) ──
        self._quality = 0
        self._nav_line_px: List[Tuple[int, int]] = []
        self._overlay: Dict = {}
        self._last_matched_task = False
        # ── 可视化基准换算 (纯显示用, 不进算法路径) ──
        # 地图坐标活在 RTK 基准系 (首包 lat/lon 为原点); 俯视图需要
        # 田块系坐标。首包时刻由 simulator 把车辆真值起点喂进来
        # (set_rtk_world_origin), get_map_overlay 输出时平移。
        self._world_origin: Optional[Tuple[float, float]] = None
        # ── 仿真器 hint (鸭子类型扩展, C++ 基线不提供) ──
        self._ctrl_hint = "WORK"   # 上一帧控制模式 (WORK/TURN_*/CAPTURE)
        self._speed_hint = 1.0     # 车速 m/s (航向 EMA α 自适应用)
        # ── C3 band 级 dump (旁路记录, 默认关; 训练数据工厂用) ──
        # 开启后 _extract_points 把每个观测 band 的 14 维特征 +
        # 世界坐标 (RTK 基准系) 记入缓冲, 由外部 collect 取走算
        # 真值标签 (点到真值行的带符号垂距)。不影响正常解算路径。
        self._band_dump_on = False
        self._band_dump_buf: List[Dict] = []
        # ── C5 行尾 dump (旁路记录, 默认关) ──
        # 每帧解算后记录地图历史形态特征 (锚点密度衰减/观测命中
        # 统计/quality 窗口), 标签由仿真器侧行真值起止计算。
        # 观测/quality 历史窗是特征的一部分, 常驻维护 (开销可忽略)。
        self._headland_dump_on = False
        self._headland_dump_buf: List[Dict] = []
        self._obs_hist = collections.deque(maxlen=20)   # task_obs 历史
        self._qual_hist = collections.deque(maxlen=8)   # quality 历史
        # ── C3 质量网络 (可开关, 默认 None = 手写方差公式) ──
        # runtime: callable(feats (N,14)) → (δx (N,), σ² (N,));
        # 由 dl/c3_runtime.py 构造后注入 (numpy 前向, 无 torch 依赖)
        self._qnet = None
        # ── C4 BEV 时序证据 (可开关, 默认 None = 纯视觉观测) ──
        # runtime: dl/c4_runtime.BEVRuntime (step + 隐状态自维护);
        # 视觉失明帧 (任务行未被真观测命中) 从 P 图提伪观测喂
        # map.update 正门 —— 遮挡/断垄期间维持行结构 (c4c A/B)
        self._bev = None
        self._bev_blind_n = 0   # 连续失明帧数 (伪观测 var 增长用)
        # ── C5 行尾感知网络 (可开关, 默认 None = 固定触发阈值) ──
        # runtime: callable(feats (N,22)) → P(end) (N,);
        # 每帧 (含 q=0 帧) 前向, get_headland_prob() 取走喂
        # 控制器掉头触发门调制 (c5b 集成)
        self._hnet = None
        self._p_end: Optional[float] = None
        # ── 掉头中止检测 (鸭子类型扩展, 几何事实不依赖网络) ──
        # 掉头期间任一"掉头开始时已存在"的行 y 跨度增长 ≥2m = 对岸
        # 锚点落图 = 行未结束, 掉头理由消失。断垄掉头时任务行的对
        # 岸延续观测会因掉头段位姿漂移 (≈一行行距) 落进邻行 → 只盯
        # 任务行会漏判, 故扩成任一行。真行尾所有行同止不触发; 掉头
        # 中新见的行 (新 id) 不在快照内, 天然不触发
        self._hd_span0: Optional[Dict[int, Tuple[float, float]]] = None
        self._row_alive = False

    # ════════════════════════════════════════════════════════
    #  接口对齐 VisionNavigation (simulator 调用面)
    # ════════════════════════════════════════════════════════
    def set_debug(self, enabled: bool):
        self._debug = bool(enabled)

    def set_debug_verbose(self, enabled: bool):
        pass    # 参考实现无 verbose 通道

    def set_sensor_mode(self, mode: int):
        self._sensor_mode = int(mode)

    def set_control_hint(self, mode: str):
        """仿真器喂上一帧控制模式 (鸭子类型扩展): CAPTURE/掉头段
        解锁重捕获引导的行跨度限制 —— 捕获阶段车常在行带外,
        查表回家是唯一生路; WORK 段维持限制 (防引导锁在行延长线
        上抑制行尾掉头语义)"""
        self._ctrl_hint = str(mode)

    def set_speed_hint(self, speed: float):
        """仿真器喂车速 (鸭子类型扩展): 航向 EMA α 随速度自适应 ——
        高速时固定 α 的时间常数 (帧数恒定) 意味着秒级滞后成倍放大,
        反投影错位导致观测关联失败 (幽灵行/任务行锚点停长)"""
        self._speed_hint = max(0.0, float(speed))

    def set_quality_net(self, runtime):
        """接入 C3 异方差观测质量网络 (可开关; 消融 A/B 用)

        runtime: callable(feats (N,14) ndarray) → (δx (N,), σ² (N,))
        (dl/c3_runtime.py 从 npz 权重构造, numpy 前向)。接入后逐
        band 用网络输出替代手写三项式 var, xv 加 δx 纠偏 (限幅
        QNET_MAX_DX); 传 None 恢复手写公式。fuse() 接口不变,
        地图核心零改动。
        """
        self._qnet = runtime

    def set_bev_net(self, runtime):
        """接入 C4 BEV 时序证据聚合网络 (可开关; 消融 A/B 用)

        runtime: dl/c4_runtime.BEVRuntime (numpy 前向, 自维护隐
        状态)。接入后每帧用本帧真观测点喂 BEV 前向; 视觉失明帧
        (任务行未被真观测命中) 从 P 图提取伪观测走 map.update()
        正门 —— 遮挡/断垄期间前方锚点继续生长, 拟合不中断;
        有真观测帧不掺假。传 None 恢复纯视觉。
        """
        self._bev = runtime

    def set_headland_net(self, runtime):
        """接入 C5 行尾/地头感知网络 (可开关; 消融 A/B 用)

        runtime: callable(feats (N,22) ndarray) → P(end) (N,)
        (dl/c5_runtime.py 从 npz 权重构造, numpy 前向)。接入后每帧
        (含 q=0 帧 —— 行尾越过后正是关键样本) 前向得行终止概率,
        get_headland_prob() 取走; simulator 喂给控制器掉头触发门
        调制 (P 高→提前掉头, P 低→保守防断垄/遮挡误掉头)。传 None
        恢复固定触发阈值。
        """
        self._hnet = runtime

    def get_headland_prob(self) -> Optional[float]:
        """C5 行尾概率 P(end) (None = 未接入; 每帧解算后更新)"""
        return self._p_end

    def set_headland_net(self, runtime):
        """接入 C5 行尾/地头感知网络 (可开关; 消融 A/B 用)

        runtime: callable(feats (N,22) ndarray) → P(end) (N,)
        (dl/c5_runtime.py 从 npz 权重构造, numpy 前向)。接入后每帧
        (含 q=0 帧 —— 行尾越过后正是关键样本) 前向得行终止概率,
        get_headland_prob() 取走; simulator 喂给控制器掉头触发门
        调制 (P 高→提前掉头, P 低→保守防断垄/遮挡误掉头)。传 None
        恢复固定触发阈值。
        """
        self._hnet = runtime

    def get_headland_prob(self) -> Optional[float]:
        """C5 行尾概率 P(end) (None = 未接入; 每帧解算后更新)"""
        return self._p_end

    def get_row_alive(self) -> bool:
        """掉头中止信号: 掉头期间任一已有行 y 跨度增长 ≥2m (对岸
        延续锚点落图)。True = 行未结束, 控制器应中止掉头程序跳捕获
        段。WORK 帧恒 False; baseline/C++ 后端无此接口 (hasattr 兜底)"""
        return self._row_alive

    def force_row_reset(self):
        """行状态复位 (仿真 reset 调用): 地图/锁定/位姿全部清空

        位姿也清的理由: R 键重置是完整重开 —— RTK 基准原点/航向 EMA/
        输出平滑状态若保留, 重播首帧会从旧状态爬升。仿真器会随帧
        重新喂包, 全部状态自然重建。
        """
        self._map.reset()
        self._pos = None
        self._heading = None
        self._heading_var = self.SIGMA_HEADING_MEAS ** 2
        self._origin = None
        self._pos_status = 0
        self._att = (0.0, 0.0)
        self._att_has = False
        self._att_fresh = False
        self._att_applied_last = False
        self._frame = 0
        self._out_valid_prev = False
        self._quality = 0
        self._nav_line_px = []
        self._overlay = {}
        self._world_origin = None
        self._obs_hist.clear()
        self._qual_hist.clear()
        # C4 隐状态同步清零 (旧地块的时序记忆对新地块是污染)
        if self._bev is not None:
            self._bev.reset()
        self._bev_blind_n = 0
        self._hd_span0 = None
        self._row_alive = False

    def feed_rtk_data(self, pkt: Dict):
        """RTK/IMU 数据包注入 (dict 字段与 sensor_simulator 约定一致)

        位置以首包 lat/lon 为基准逆投影 ENU —— 地图坐标活在同一基准系,
        相对量 (CTE/yaw) 不受基准偏移影响。航向 (双天线) EMA 平滑。
        pos_status != 4 (非固定解) 时位置保持 (hint 失效语义)。
        """
        lat = pkt.get("latitude")
        lon = pkt.get("longitude")
        if lat is None or lon is None:
            return
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return
        if self._origin is None:
            self._origin = (lat, lon)

        self._pos_status = int(pkt.get("pos_status", 0))
        if self._pos_status == 4:
            lat0, lon0 = self._origin
            cos_lat0 = math.cos(math.radians(lat0))
            east = (lon - lon0) * self._METRES_PER_DEG * cos_lat0
            north = (lat - lat0) * self._METRES_PER_DEG
            self._pos = (east, north)

        h = pkt.get("heading")
        if h is not None and math.isfinite(h):
            h_rad = math.radians(h)
            if self._heading is None:
                self._heading = h_rad
                self._heading_var = self.SIGMA_HEADING_MEAS ** 2
            else:
                # EMA (样本先 wrap 到当前估计 ±180°, 防绕圈跳变);
                # α 随车速自适应 (v=1 基准不变, 高速放大压滞后, 上限 0.6)
                dh = (h_rad - self._heading + math.pi) % (2 * math.pi) - math.pi
                a = min(0.6, self.HEADING_EMA_ALPHA
                        * max(1.0, self._speed_hint))
                self._heading += a * dh
                self._heading_var = (a / (2.0 - a)) * self.SIGMA_HEADING_MEAS ** 2

        roll = pkt.get("roll")
        pitch = pkt.get("pitch")
        if roll is not None and pitch is not None \
                and math.isfinite(roll) and math.isfinite(pitch):
            self._att = (math.radians(roll), math.radians(pitch))
            self._att_has = True
            self._att_fresh = True

    def get_attitude_diag(self) -> Dict:
        applied = self._att_applied_last
        return {
            # fresh 语义与 C++ 对齐: 本帧解算实际应用了姿态补偿
            # (解算内 _att_fresh 已被消费, 用事后记录的 _att_applied_last)
            "fresh": applied,
            "has_data": self._att_has,
            "applied_roll_deg": (math.degrees(self._att[0]) if applied else 0.0),
            "applied_pitch_deg": (math.degrees(self._att[1]) if applied else 0.0),
        }

    def get_rtk_position_diag(self) -> Dict:
        return {
            "rtk_hint_valid": bool(self._pos is not None
                                    and self._pos_status == 4),
            "rtk_pos_status": self._pos_status,
        }

    def get_overlay_data(self) -> Dict:
        """GUI 叠层数据: center_lines (各行锚点投影像素) + line_used"""
        return self._overlay

    # ════════════════════════════════════════════════════════
    #  主入口: 每帧导航解算
    # ════════════════════════════════════════════════════════
    def get_navigation_data(self, image, masks, frame_time
                            ) -> Tuple[float, float, float, List, int]:
        """Returns: (yaw_deg, front_dist, rear_dist, nav_line_px, quality)

        符号约定与 C++/vehicle_model 一致:
          front/rear 右偏为正 (车在目标线右侧 → 正 → 左转纠偏);
          yaw 偏左为负 (罗盘系, 车航向 − 线方向)。
        """
        self._frame += 1

        # ── 1. 位姿可用性 (RTK 主路径, mode 2) ──
        # mode 0/1 无位置源 (视觉里程计位姿留后续版本) → 诚实降级 q=0,
        # 与文档 当前范围声明一致; 否则 mode 矩阵 A/B 会误读 "作弊" 数据
        if (self._sensor_mode < 2
                or self._pos is None or self._heading is None):
            self._att_fresh = False
            self._quality = 0
            return (0.0, 0.0, 0.0, [], 0)

        px, py = self._pos
        c, s = math.cos(self._heading), math.sin(self._heading)

        # ── 2. 投影器姿态补偿 (mode>=1 且有 IMU 数据) ──
        att_applied = (self._sensor_mode >= 1 and self._att_has)
        if att_applied:
            self._proj.set_attitude(*self._att)
        else:
            self._proj.set_attitude(0.0, 0.0)
        self._att_fresh = False
        self._att_applied_last = att_applied

        # ── 3. 观测提取: mask → 车辆系 → 世界系 ──
        # C3 质量网络激活时: 逐 band 网络前向得 (δx 纠偏, σ²),
        # 替代手写三项式 var —— fuse() 接口不变, 地图核心零改动
        obs_rows = []
        for m in masks:
            score = float(getattr(m, "score", 1.0))
            pts_v, band_meta = self._extract_points(m.mask, score=score)
            if len(pts_v) < 2:
                continue
            if self._qnet is not None and band_meta:
                feats = np.asarray(
                    [self._band_features(band_meta, i, score)
                     for i in range(len(band_meta))],
                    dtype=np.float64)
                dx_net, var_net = self._qnet(feats)
                # δx 安全限幅: 网络离谱输出不至污染观测
                dx_net = np.clip(dx_net, -self.QNET_MAX_DX,
                                 self.QNET_MAX_DX)
                pts_v = [(xv + float(dx_net[i]), zv)
                         for i, (xv, zv) in enumerate(pts_v)]
                var_list = [float(v) for v in var_net]
            else:
                var_list = None
            pts_w = []
            for i, (xv, zv) in enumerate(pts_v):
                wx = px + xv * c + zv * s
                wy = py - xv * s + zv * c
                if var_list is not None:
                    var = var_list[i]
                else:
                    var = ((self.K_PIX_SIGMA * zv) ** 2
                           + self._heading_var * zv * zv
                           + self.SIGMA_RTK ** 2)
                pts_w.append((wx, wy, var))
            obs_rows.append(pts_w)

        # ── 4. 地图更新 + 任务行锁定 ──
        matched = self._map.update(obs_rows, self._frame) if obs_rows else []
        task_id = self._map.lock_task_row(px, py)
        task_obs = (task_id is not None and task_id in matched)
        # C4 伪观测 (可开关): BEV 前向维护隐状态; 失明帧从 P 图
        # 提伪观测喂地图 (matched 不计入 —— task_obs 保持纯视觉
        # 语义, quality 分级 95/70 不被伪观测抬高)
        if self._bev is not None:
            self._bev_frame(px, py, obs_rows, task_obs)
        self._last_matched_task = task_obs

        # ── 4.5 掉头中止检测: 任一已有行延续 = 掉头理由消失 ──
        # 掉头中 (ctrl_hint 非 WORK) 对岸锚点落图: 断垄时三条行同时
        # 延伸, 但掉头段位姿漂移会让任务行的延续观测落进邻行 (S3:
        # +0.35m ≈ 一行行距) → 只盯任务行漏判。快照掉头开始时全部
        # 行的 y 跨度, 之后任一行增长 ≥2m 即行存活。掉头中新见的行
        # (新 id) 不在快照内不触发; 真地头所有行同止不触发。纯地图
        # 几何事实, 不依赖网络
        if self._ctrl_hint != "WORK":
            if self._hd_span0 is None:
                self._hd_span0 = {rid: (r.y_min, r.y_max)
                                  for rid, r in self._map.rows.items()}
            else:
                for rid, (ymin0, ymax0) in self._hd_span0.items():
                    r = self._map.rows.get(rid)
                    if (r is not None
                            and (r.y_max > ymax0 + 2.0
                                 or r.y_min < ymin0 - 2.0)):
                        self._row_alive = True
                        break
        else:
            self._hd_span0 = None
            self._row_alive = False

        # ── 5. 解算: 任务行前段 → 车辆系 → 加权二次曲线拟合 ──
        task = self._map.task_row()
        fit = None
        seg_world = []
        if task is not None:
            # 前向段按航向的 +y 分量选段方向 (锚点按 y 有序): 反向
            # 再入 (掉头后沿 -y 开) 时前方是 y 较小侧, 固定 +y 段会
            # 取到身后锚点导致拟合错向
            if math.cos(self._heading) >= 0.0:
                seg_world = task.segment(py - self.SEG_BACK_M,
                                         py + self.LOOKAHEAD_M)
            else:
                seg_world = task.segment(py - self.LOOKAHEAD_M,
                                         py + self.SEG_BACK_M)
            if len(seg_world) >= self.MIN_SOLVE_ANCHORS:
                pts_v = []
                for ay, ax, avar in seg_world:
                    dx, dy = ax - px, ay - py
                    xv = dx * c - dy * s
                    zv = dx * s + dy * c
                    pts_v.append((xv, zv, avar))
                fit = self._fit_curve(pts_v)

        # ── 6. 输出与 quality 分级 ──
        guide = None
        if fit is None:
            # 前向段拟合失败分两种, 语义完全不同:
            # a) 车在行纵向跨度内但横向偏出/前向锚点不足 → 重捕获引导
            #    (查表回家, 观测恢复后自动升级 95/70)
            # b) 车已越过行尾/行头 (>2m) → 引导反而有害: 会把车锁在
            #    行延长线上蛇形且 q=50 恒有效抑制行尾掉头。不引导,
            #    q=0 让控制器丢行计数 → 路程掉头语义接管
            if task is not None:
                in_span = ((task.y_min - 2.0) <= py
                           <= (task.y_max + 2.0))
                # CAPTURE/掉头段 (ctrl_hint != WORK) 解锁 span 限制:
                # 捕获阶段车常在行带外, 最近锚点邻域引导是唯一生路;
                # WORK 段维持 span 限制 (行延长线外不引导, 让位掉头)
                if in_span or self._ctrl_hint != "WORK":
                    seg_g = task.nearest_segment(
                        py, n=self.RECAPTURE_ANCHORS)
                    if len(seg_g) >= self.MIN_SOLVE_ANCHORS:
                        pts_g = []
                        for ay, ax, avar in seg_g:
                            dx, dy = ax - px, ay - py
                            pts_g.append((dx * c - dy * s,
                                          dx * s + dy * c, avar))
                        guide = self._fit_curve(pts_g)
        if guide is None and fit is None:
            # 地图不足 (开局未建图 / 行尾外无锚点): hold 无效输出。
            # 事件日志: 质量跌 0 会驱动控制器丢行计数 → 掉头,
            # 打印现场快照供 GUI 控制台定位根因 (时间/位姿/地图状态)
            if self._quality > 0:
                print(f"[ROWMAP] quality->0 @ t={frame_time:.2f}s "
                      f"pos=({px:.2f},{py:.2f}) status={self._pos_status} "
                      f"rows={len(self._map.rows)} "
                      f"task={self._map.task_row_id} "
                      f"task_anchors={len(task.anchors) if task else 0} "
                      f"seg_len={len(seg_world)} "
                      f"obs_rows={len(obs_rows)}")
            self._quality = 0
            self._nav_line_px = []
            self._overlay = {}
            self._out_valid_prev = False
            # C5 历史/dump (旁路): q=0 帧正是行尾语义的关键样本
            self._obs_hist.append(bool(task_obs))
            self._qual_hist.append(0)
            feat = self._headland_features(px, py, task_obs,
                                           False, False,
                                           len(obs_rows), 0)
            self._update_p_end(feat)
            if self._headland_dump_on:
                self._dump_headland(feat, px, py, task_obs)
            return (0.0, 0.0, 0.0, [], 0)

        if guide is not None:
            if self._quality == 0:
                print(f"[ROWMAP] recapture guide engaged @ "
                      f"t={frame_time:.2f}s pos=({px:.2f},{py:.2f}) "
                      f"task={self._map.task_row_id}")
            fc = guide
            quality_out = self.QUALITY_RECAPTURE
        else:
            fc = fit
            quality_out = (self.QUALITY_TRACKING if task_obs
                           else self.QUALITY_DEAD_RECKON)
        front_dist = -self._curve_x(fc, self.Z_FRONT)
        rear_dist = -self._curve_x(fc, self.Z_REAR)
        # yaw 取曲线在车体中心 (z=0) 处的切线方向 —— 行方向的一般化
        # (直线拟合时切线恒为 b, 与旧版语义一致)
        yaw_deg = -math.degrees(math.atan(self._curve_slope(fc, 0.0)))

        # 输出端单极点平滑: 压 RTK 噪声透传的输出跳动 (yaw/front)。
        # 无效→有效恢复首帧直通重置 (避免从 0 爬升拖慢恢复)
        if self._out_valid_prev:
            a_out = self.OUTPUT_EMA_ALPHA
            self._yaw_s += a_out * (yaw_deg - self._yaw_s)
            self._front_s += a_out * (front_dist - self._front_s)
            self._rear_s += a_out * (rear_dist - self._rear_s)
        else:
            self._yaw_s = yaw_deg
            self._front_s = front_dist
            self._rear_s = rear_dist
            self._out_valid_prev = True

        # 导航线像素: 拟合曲线 z∈{0.5, 12} 端点投影 (与观测同一外参)
        nav_line_px = []
        for z_end in (0.5, 12.0):
            col, row = self._proj._vehicle_xz_to_pixel(
                self._curve_x(fc, z_end), z_end)
            if 0 <= col < self._proj.crop_w and 0 <= row < self._proj.crop_h:
                nav_line_px.append((col, row))
        self._nav_line_px = nav_line_px

        # quality 分级: 95 跟踪 / 70 外推 / 50 重捕获引导 (guide 帧的
        # task_obs 必为 False —— 观测命中任务行时前向段必有锚点)
        self._quality = quality_out

        # C5 历史/dump (旁路) + 行尾概率前向 (可开关)
        self._obs_hist.append(bool(task_obs))
        self._qual_hist.append(int(quality_out))
        feat = self._headland_features(
            px, py, task_obs, fit is not None, guide is not None,
            len(obs_rows), int(quality_out))
        self._update_p_end(feat)
        if self._headland_dump_on:
            self._dump_headland(feat, px, py, task_obs)

        # overlay: 各行锚点段 (车前方) 投影像素 + 是否任务行
        self._overlay = self._build_overlay(px, py, c, s)

        return (self._yaw_s, self._front_s, self._rear_s,
                list(nav_line_px), int(self._quality))

    # ════════════════════════════════════════════════════════
    #  内部实现
    # ════════════════════════════════════════════════════════
    def _extract_points(self, mask: np.ndarray, score: float = 1.0
                        ) -> Tuple[List[Tuple[float, float]],
                                   Optional[List[tuple]]]:
        """mask → 车辆系观测点串 [(xv, zv), ...] (按图像行自底向上扫描)

        每采样带取 mask 像素列均值作为行中心 (行条带横向连续),
        反投影到车辆系地面 (外参已含 IMU 姿态补偿 → 姿态残差进入
        观测噪声, 由 var 模型表达 —— mode 1 的验证路径)。

        Returns: (pts, band_meta)。band_meta = [(row_c, ncols,
        span_px, xv, zv), ...] 仅在质量网络或 band dump 任一激活时
       组装 (否则 None —— 旁路全关时零额外开销), 供
        _band_features 组装 14 维特征。
        """
        h, w = mask.shape
        need_meta = (self._qnet is not None or self._band_dump_on)
        bands = []   # (row_c, ncols, span_px, xv, zv)
        pts = []
        r = h - 1
        while r >= 0:
            band = mask[max(0, r - self.SCAN_STEP_PX + 1): r + 1]
            cols = np.flatnonzero(band.any(axis=0))
            if len(cols):
                col = float(cols.mean())
                row_c = r - (self.SCAN_STEP_PX - 1) / 2.0
                res = self._proj.pixel_to_vehicle_xz(col, row_c)
                if res is not None:
                    xv, zv = res
                    if self.Z_MIN_OBS <= zv <= self.Z_MAX_OBS:
                        pts.append((xv, zv))
                        if need_meta:
                            bands.append((row_c, len(cols),
                                          float(cols.max() - cols.min()),
                                          xv, zv))
            r -= self.SCAN_STEP_PX
        if self._band_dump_on and bands:
            self._dump_bands(bands, score)
        return pts, (bands if need_meta else None)

    # ── C3 band dump (旁路, 默认关) ─────────────────────────

    def enable_band_dump(self):
        """开启 band 级特征 dump (训练数据工厂用, 不影响解算)"""
        self._band_dump_on = True

    def collect_band_dump(self) -> Tuple[List[Dict], Dict]:
        """取走 band dump 缓冲。Returns: (records, meta);
        meta 含 world_origin (RTK 基准系 → 田块全局系的平移,
        由 simulator set_rtk_world_origin 喂入; 未喂时为 None,
        标签计算需跳过该 run)"""
        buf = self._band_dump_buf
        self._band_dump_buf = []
        meta = {"world_origin": self._world_origin,
                "frames": self._frame}
        return buf, meta

    # ── C5 行尾 dump (旁路, 默认关) ─────────────────────

    def enable_headland_dump(self):
        """开启行尾特征 dump (C5 训练数据工厂用, 不影响解算)"""
        self._headland_dump_on = True

    def collect_headland_dump(self) -> Tuple[List[Dict], Dict]:
        """取走行尾 dump 缓冲。Returns: (records, meta);
        meta.world_origin 语义同 band dump (帧级 py 为 RTK 基准
        系估计值, 标签用仿真器真值位姿另算)"""
        buf = self._headland_dump_buf
        self._headland_dump_buf = []
        return buf, {"world_origin": self._world_origin,
                     "frames": self._frame}

    def _headland_features(self, px, py, task_obs, has_fit,
                           has_guide, n_obs_masks, quality_now) -> tuple:
        """22 维行尾特征组装 (文档 §3.7 输入清单; C5 dump/推理共用,
        训练与部署严格同源)

        核心信号: 任务行锚点密度沿 y 的衰减 (真行尾平滑衰减到
        零且前方从未覆盖; 断垄前方曾有覆盖) + 行尾相对距离 +
        观测命中/quality 历史窗。无任务行时任务行相关槽位置零
        + has_task=0 指示。
        """
        # deque 不支持切片, 转 list 取近帧窗 (20 元素, 开销可忽略)
        hist = list(self._obs_hist)
        obs5 = (sum(hist[-5:]) / min(5, len(hist))
                if hist else 0.0)
        obs15 = (sum(hist[-15:]) / min(15, len(hist))
                 if hist else 0.0)
        qual_mean = (sum(self._qual_hist) / len(self._qual_hist)
                     if self._qual_hist else 0.0)
        task = self._map.task_row()
        if task is not None and task.anchors:
            ys = [a.y for a in task.anchors]      # 按 y 升序
            y_min, y_max = ys[0], ys[-1]
            # 前方分段锚点数 (沿 +y 航向; 反向作业第一版不覆盖)
            n_05 = sum(1 for y in ys if py <= y < py + 5.0)
            n_510 = sum(1 for y in ys if py + 5.0 <= y < py + 10.0)
            n_1015 = sum(1 for y in ys if py + 10.0 <= y < py + 15.0)
            n_1520 = sum(1 for y in ys if py + 15.0 <= y < py + 20.0)
            # 前方 [py, py+20] 最大连续空窗 (含端点): 真行尾 →
            # 锚点衰减后空窗占满; 断垄 → 远端有锚点压缩空窗
            front = [y for y in ys if py <= y <= py + 20.0]
            edges = [py] + front + [py + 20.0]
            gap_front = max((edges[i + 1] - edges[i]
                             for i in range(len(edges) - 1)),
                            default=20.0)
            gap_behind = max(0.0, py - y_max)   # 车越过行尾距离
            return (
                1.0,                            # 0 任务行存在
                math.log1p(len(ys)),            # 1 锚点总数
                (y_max - y_min) / 100.0,        # 2 行总跨度
                (y_min - py) / 50.0,            # 3 行头相对距离
                (y_max - py) / 50.0,            # 4 行尾相对距离 (核心)
                math.log1p(n_05),               # 5 前方 0-5m 锚点数
                math.log1p(n_510),              # 6
                math.log1p(n_1015),             # 7
                math.log1p(n_1520),             # 8
                gap_front / 20.0,               # 9 前方最大空窗
                gap_behind / 20.0,              # 10 车越过行尾距离
                obs5,                           # 11 近 5 帧命中率
                obs15,                          # 12 近 15 帧命中率
                qual_mean / 95.0,               # 13 quality 均值窗
                quality_now / 95.0,             # 14 本帧 quality
                math.log1p(n_obs_masks),        # 15 本帧 mask 数
                1.0 if has_guide else 0.0,      # 16 guide 状态
                1.0 if has_fit else 0.0,        # 17 fit 状态
                self._speed_hint,               # 18 车速
                math.sqrt(self._heading_var),   # 19 航向 σ
                y_min / 100.0,                  # 20 行头绝对 y
                y_max / 100.0,                  # 21 行尾绝对 y
            )
        return (
            0.0,                            # 0 无任务行
            0.0, 0.0, 0.0, 0.0,             # 1-4 任务行几何全零
            0.0, 0.0, 0.0, 0.0,             # 5-8 前方锚点数全零
            1.0,                            # 9 前方空窗 = 满
            0.0,                            # 10
            obs5, obs15, qual_mean / 95.0, quality_now / 95.0,
            math.log1p(n_obs_masks),
            1.0 if has_guide else 0.0,
            1.0 if has_fit else 0.0,
            self._speed_hint,
            math.sqrt(self._heading_var),
            0.0, 0.0,                       # 20-21
        )

    def _update_p_end(self, feat: tuple):
        """C5 行尾概率前向 (可开关; 未接入时保持 None)"""
        if self._hnet is None:
            return
        p = self._hnet(np.asarray([feat], dtype=np.float64))
        self._p_end = float(np.asarray(p).reshape(-1)[0])

    def _dump_headland(self, feat: tuple, px, py, task_obs):
        """行尾 dump 缓冲追加 (特征由 _headland_features 组装)"""
        self._headland_dump_buf.append({
            "frame": self._frame, "feat": feat,
            "px": px, "py": py, "task_obs": bool(task_obs),
        })

    def _band_features(self, band_meta, i: int, score: float
                       ) -> tuple:
        """14 维 band 特征 (文档 §3.3 清单; C3 网络前向/dump 共用,
        保证训练与推理特征严格同源)

        band_meta: _extract_points 组装的 [(row_c, ncols, span_px,
        xv, zv), ...]; i: 目标 band 序号 (自底向上, 0=近端)。
        """
        row_c, ncols, span, xv, zv = band_meta[i]
        n = len(band_meta)
        # 相邻 band 横向跳变 (单 band 取 0): 条带断裂/错检指示
        if n > 1:
            j = i - 1 if i > 0 else i + 1
            jump = abs(xv - band_meta[j][3])
        else:
            jump = 0.0
        applied = self._att_applied_last
        ch, cw = self._proj.crop_h, self._proj.crop_w
        return (
            zv / self.Z_MAX_OBS,            # 0 前向距离
            row_c / ch,                     # 1 band 像素行中心
            ncols / cw,                     # 2 band 命中列数
            span / cw,                      # 3 条带像素宽度
            float(score),                   # 4 分割置信度
            math.degrees(self._att[0]) if applied else 0.0,
                                            # 5 应用 roll 补偿 (度)
            math.degrees(self._att[1]) if applied else 0.0,
                                            # 6 应用 pitch 补偿 (度)
            math.sqrt(self._heading_var),   # 7 航向 σ (rad)
            jump,                           # 8 相邻 band 跳变 (m)
            1.0 if i == 0 else 0.0,         # 9 近端 band 指示
            1.0 if i == n - 1 else 0.0,     # 10 远端 band 指示
            n / 100.0,                      # 11 本 mask band 总数
            abs(xv) / 3.0,                  # 12 横向位置幅值 (m)
            1.0 if self._pos_status == 4 else 0.0,  # 13 RTK 固定解
        )

    def _dump_bands(self, bands, score: float):
        """band 级特征 dump (特征组装见 _band_features) + 世界坐标"""
        px, py = self._pos
        c, s = math.cos(self._heading), math.sin(self._heading)
        n = len(bands)
        for i in range(n):
            xv, zv = bands[i][3], bands[i][4]
            feat = self._band_features(bands, i, score)
            # 世界坐标 (RTK 基准系, 与观测进地图同一变换);
            # 位姿字段供 C4 按帧聚合重构 BEV 序列复用
            wx = px + xv * c + zv * s
            wy = py - xv * s + zv * c
            self._band_dump_buf.append({
                "frame": self._frame, "feat": feat,
                "xv": xv, "zv": zv, "wx": wx, "wy": wy,
                "px": px, "py": py, "h": self._heading,
                "score": float(score),
            })

    # ── C4 BEV 时序证据 (可开关, 默认关) ─────────────────────

    def _bev_frame(self, px, py, obs_rows, task_obs):
        """C4 每帧钩子: BEV 前向维护隐状态 + 失明帧伪观测

        每帧无条件前向 (真观测喂输入通道, 失明时 obs 空 —— 隐状态
        靠时序记忆保持行证据); 仅当任务行未被真观测命中时才从
        P 图提伪观测。伪观测走地图伪通道 (绕过行关联直喂任务行,
        锚点带 pseudo 标记不参与关联打分 —— 防漂移伪锚点劫持对岸
        真观测的行身份)。伪观测 var 随失明时长增长 (记忆的位置会随
        航向误差积分漂移, 越久越不确定)。
        """
        task = self._map.task_row()
        if task_obs:
            self._bev_blind_n = 0
        else:
            self._bev_blind_n += 1
        obs_flat = [(wx, wy) for row in obs_rows
                    for wx, wy, _ in row]
        p, (ox, oy) = self._bev.step(obs_flat, px, py, self._heading)
        if task_obs or task is None or not task.anchors:
            return    # 有真观测不掺假 / 无任务行无锚定目标
        pts = self._bev_pseudo_obs(p, ox, oy, task, py)
        if pts:
            self._map.update([pts], self._frame, pseudo=True)

    def _bev_pseudo_obs(self, p, ox, oy, task, py):
        """P 图 → 任务行伪观测点串 [(wx, wy, var), ...]

        沿 y 每 2px (1m) 一层, 在任务行预测 x ±2px 邻域找 P 峰值,
        峰值 ±1px 加权质心作亚像素 x; 全串与任务行均值偏差
        守门 (BEV_GATE_M, 防 P 图错检把地图带偏 —— 宁可本帧纯
        记忆也不喂错点)。预测线与守门均用真锚点 (x_at_real):
        伪锚点随失明漂移, 预测若跟着伪锚点走会形成"漂移→预测
        漂→伪观测更漂"的自反馈; 锚定真锚点几何后, 伪锚点偏差被
        BEV_GATE_M 封顶。
        """
        grid = self._bev.grid
        res = self._bev.res
        y_lo = py - self.BEV_BACK_M
        y_hi = py + self.BEV_AHEAD_M
        var = self.BEV_PSEUDO_VAR * (
            1.0 + self._bev_blind_n / self.BEV_VAR_GROW_FRAMES)
        pts = []
        for gi in range(0, grid, 2):
            wy = oy + gi * res
            if not (y_lo <= wy <= y_hi):
                continue
            wx_pred = task.x_at_real(wy)
            if wx_pred is None:
                continue
            cj = int(round((wx_pred - ox) / res))
            lo = max(0, cj - 2)
            hi = min(grid, cj + 3)
            if hi <= lo:
                continue
            seg = p[gi, lo:hi]
            jmax = lo + int(np.argmax(seg))
            if p[gi, jmax] < self.BEV_P_THR:
                continue
            # 亚像素质心 (峰值 ±1px 的 P 加权)
            w_lo, w_hi = max(lo, jmax - 1), min(hi, jmax + 2)
            w = p[gi, w_lo:w_hi]
            cols = np.arange(w_lo, w_hi, dtype=np.float64)
            cj_f = float((w * cols).sum() / w.sum())
            pts.append((ox + cj_f * res, wy, var))
        if not pts:
            return []
        dev = sum(abs(wx - task.x_at_real(wy))
                  for wx, wy, _ in pts) / len(pts)
        if dev > self.BEV_GATE_M:
            return []
        return pts

    @staticmethod
    def _fit_curve(pts_v: List[Tuple[float, float, float]]
                   ) -> Optional[Tuple[float, float, float, float]]:
        """车辆系点串加权拟合二次曲线 x = a + b·u + c·u² (u = z − zc)

        权重 1/var; n≤3 点或正规方程病态时退化为直线 (c=0)。3×3
        正规方程用手写高斯消元求解 —— 与 C++ fit_curve 逐操作对齐,
        保证对拍一致 (不用 numpy 求解器, 舍入序不同会引入 ULP 差)。
        Returns: (a, b, c, zc) 或 None (点数不足); x(z) 由 _curve_x 求值。
        """
        n = len(pts_v)
        if n < 2:
            return None
        # 加权矩 (标量累加, 与 C++ 同序)
        sw = 0.0
        wz = 0.0
        wx = 0.0
        for x, z, var in pts_v:
            w = 1.0 / max(var, 1e-8)
            sw += w
            wz += w * z
            wx += w * x
        zbar = wz / sw
        xbar = wx / sw

        def _linear() -> Tuple[float, float, float, float]:
            denom = 0.0
            num = 0.0
            for x, z, var in pts_v:
                w = 1.0 / max(var, 1e-8)
                dz = z - zbar
                denom += w * dz * dz
                num += w * dz * (x - xbar)
            if abs(denom) < 1e-9:
                return (xbar, 0.0, 0.0, 0.0)
            b = num / denom
            return (xbar - b * zbar, b, 0.0, 0.0)

        if n <= 3:
            return _linear()

        # 二次: 中心化矩 u = z − zbar (正规方程加增广列)
        S = sw
        Su = 0.0
        Suu = 0.0
        Suuu = 0.0
        Suuuu = 0.0
        Sx = wx
        Sxu = 0.0
        Sxuu = 0.0
        for x, z, var in pts_v:
            w = 1.0 / max(var, 1e-8)
            u = z - zbar
            u2 = u * u
            Su += w * u
            Suu += w * u2
            Suuu += w * u2 * u
            Suuuu += w * u2 * u2
            Sxu += w * u * x
            Sxuu += w * u2 * x
        m = [[S, Su, Suu, Sx],
             [Su, Suu, Suuu, Sxu],
             [Suu, Suuu, Suuuu, Sxuu]]
        scale = max(abs(S), abs(Suu), abs(Suuu), abs(Suuu), 1e-30)
        eps = 1e-12 * scale
        # 高斯消元 (部分主元), 与 C++ 逐操作一致
        for k in range(3):
            p = k
            for r in range(k + 1, 3):
                if abs(m[r][k]) > abs(m[p][k]):
                    p = r
            if abs(m[p][k]) < eps:
                return _linear()      # 病态 (锚点 z 聚集) → 直线回退
            if p != k:
                m[p], m[k] = m[k], m[p]
            for r in range(k + 1, 3):
                f = m[r][k] / m[k][k]
                for cc in range(k, 4):
                    m[r][cc] -= f * m[k][cc]
        c2 = m[2][3] / m[2][2]
        c1 = (m[1][3] - m[1][2] * c2) / m[1][1]
        c0 = (m[0][3] - m[0][2] * c2 - m[0][1] * c1) / m[0][0]
        return (c0, c1, c2, zbar)

    @staticmethod
    def _curve_x(fc: Tuple[float, float, float, float], z: float) -> float:
        """曲线求值 x(z) = a + b·(z−zc) + c·(z−zc)²"""
        a, b, c, zc = fc
        u = z - zc
        return a + b * u + c * u * u

    @staticmethod
    def _curve_slope(fc: Tuple[float, float, float, float],
                     z: float) -> float:
        """曲线切线斜率 dx/dz (z=0 处即车体中心下方行方向)"""
        a, b, c, zc = fc
        u = z - zc
        return b + 2.0 * c * u

    def _build_overlay(self, px: float, py: float,
                       c: float, s: float) -> Dict:
        """各行地图锚点段投影像素 (GUI 特征线叠层)

        center_lines: 每行 (锚点 >= MIN_DISPLAY_ANCHORS) 的像素点列表;
        line_used: 该行是否任务行 (亮绿实线) —— 与基线 overlay 语义一致。
        """
        center_lines = []
        line_used = []
        for rid, row in self._map.rows.items():
            if len(row.anchors) < 3:
                continue    # 噪声碎行不上图
            seg = row.segment(py - 0.5, py + self.Z_MAX_OBS)
            if len(seg) < 2:
                continue
            px_list = []
            for ay, ax, _ in seg:
                dx, dy = ax - px, ay - py
                xv = dx * c - dy * s
                zv = dx * s + dy * c
                if zv < 0.2:
                    continue    # 车后点不投影
                col, row_px = self._proj._vehicle_xz_to_pixel(xv, zv)
                if 0 <= col < self._proj.crop_w \
                        and 0 <= row_px < self._proj.crop_h:
                    px_list.append((int(col), int(row_px)))
            if len(px_list) >= 2:
                center_lines.append(px_list)
                line_used.append(rid == self._map.task_row_id)
        return {"center_lines": center_lines, "line_used": line_used}

    # ── 诊断辅助 (GUI/调试, simulator 不依赖) ──
    def map_stats(self) -> Dict:
        return self._map.stats()

    def get_map_overlay(self) -> Dict:
        """俯视图地图可视化数据 (GUI 后端感知绘制用)

        Returns: {"rows": [{"id": int, "task": bool,
                             "anchors": [(wx, wy, sigma_m, n_obs), ...]}],
                  "pos": (east, north) or None,
                  "heading": rad or None}
        锚点含融合后 σ (=√var) 与观测次数 n —— 置信度直接可视化。
        坐标系: 若已设 world_origin (首包真值起点) 则平移到田块系,
        否则保持 RTK 基准系 (退化显示, 原点=车辆首包位置)。
        """
        ox, oy = self._world_origin or (0.0, 0.0)
        rows = []
        for rid, row in self._map.rows.items():
            if len(row.anchors) < 2:
                continue
            rows.append({
                "id": rid,
                "task": rid == self._map.task_row_id,
                "anchors": [(a.x + ox, a.y + oy, math.sqrt(a.var), a.n)
                            for a in row.anchors],
            })
        pos = None
        if self._pos is not None:
            pos = (self._pos[0] + ox, self._pos[1] + oy)
        return {
            "rows": rows,
            "pos": pos,
            "heading": self._heading,
        }

    def set_rtk_world_origin(self, world_x: float, world_y: float):
        """设置 RTK 基准系→田块系换算原点 (纯可视化用)

        simulator 在首包 RTK 注入时把车辆真值位置 (田块系) 喂入。
        只影响 get_map_overlay 的显示坐标, 算法路径 (解算/建图)
        继续活在 RTK 基准系 —— 相对量 (CTE/yaw) 与基准无关。
        """
        if self._world_origin is None:
            self._world_origin = (float(world_x), float(world_y))

    @property
    def task_locked(self) -> bool:
        return self._map.task_row_id is not None
