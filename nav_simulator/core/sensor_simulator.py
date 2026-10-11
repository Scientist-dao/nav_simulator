"""
sensor_simulator.py - IMU/RTK 传感器仿真器

生成与实车 RTK 接收机一致的传感器数据包, 经 VisionNavigation.feed_rtk_data()
注入导航核心, 端到端验证 sensor mode 1/2 的融合路径:
  - mode 1: IMU 姿态补偿 (set_attitude -> camera_rad_eff / R_unlevel)
  - mode 2: + RTK ENU 车轴位姿 hint (RegionManager FSC) + RTK 速度

数据模型 (与 C++ 端约定对齐, 详见 C++ 工程的传感器模式文档):
  - 时间基准: 数据包 recv_time 与图像 frame_time 同用仿真时间 sim_time,
    满足算法侧最近时间匹配 (窗口 0.1 s) 的前提; 每帧最后一个包恰好落在
    帧时刻, 保证匹配窗必然命中
  - 更新率: 传感器以 rate_hz (默认 50 Hz) 独立于相机帧率 (默认 15 fps)
    采样, 帧间生成多个数据包逐个注入, 模拟真实接收机异步到达 + 算法侧
    环形缓冲 (128 样本) 的行为
  - IMU 姿态: 车辆真值姿态 (VehicleState.roll/pitch) + 每包独立高斯
    测量噪声 + 可选安装偏置。符号约定与 C++ set_attitude 注释一致:
    roll 右侧下沉为正 / pitch 抬头为正, 直接传值, 单位: 度
  - 位置: 车辆 ENU 真值 + 各向同性高斯噪声, 按平地球模型逆投影为经纬度
    (与 core/nav_geo.hpp 的 to_local_enu 正反互逆, 200 m 内误差 < 0.1 mm)
  - ENU hint 生效条件 (mode 2): pos_status == 4 (RTK 固定解) 且
    diff_age_s <= 2.0 且 heading/lat/lon 非 NaN。退化场景 (浮点解/单点
    解) 通过切换 RTKConfig.pos_status 模拟, 对应 hint 自动失效

IMU-only 部署 (rtk.enabled=False): roll/pitch 正常, 位置/速度填真值但
pos_status=0 (不可用) -> mode 2 下 ENU hint 不生效, 仅姿态补偿工作。
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


# 平地球模型常数 (与 C++ 端 nav_geo 实现一致)
_METRES_PER_DEG = 111320.0


@dataclass
class IMUConfig:
    """IMU 姿态仿真配置 (mode>=1 的姿态补偿路径)"""
    enabled: bool = False
    roll_noise_std_deg: float = 0.15    # 侧倾测量噪声 (度)
    pitch_noise_std_deg: float = 0.15   # 俯仰测量噪声 (度)
    bias_roll_deg: float = 0.0          # 安装偏置 (度): 模拟零位未标定
    bias_pitch_deg: float = 0.0


@dataclass
class RTKConfig:
    """RTK 定位仿真配置 (mode 2 的 ENU hint + 速度路径)

    噪声代价实测 (sensor_full_stack 稳态段, 2026-08): 默认 0.5°/2cm 噪声下
    mode 2 输出 yaw std ~0.32° / front std ~9mm (mode 0/1 约 0.05°/1mm);
    降为 0.1°/5mm 后收敛到 0.11°/2mm —— 输出噪声随本参数近似线性缩放。
    机理: 逐包独立噪声经 ENU hint 进入 C++ 世界坐标一致性评分 (σ=0.12m
    量级)。真实接收机输出为内部滤波后的时间相关值, 此处独立高斯为
    逐样本噪声上限建模 (低成本单机双天线, 无 INS 融合)。
    """
    enabled: bool = False
    rate_hz: float = 50.0               # 数据更新率 (Hz)
    position_noise_std_m: float = 0.02  # 水平位置噪声 (m, 东/北各向同性)
    heading_noise_std_deg: float = 0.5  # 双天线航向噪声 (度)
    speed_noise_std_ms: float = 0.05    # 速度噪声 (m/s)
    pos_status: int = 4                 # 4=RTK 固定解 (ENU hint 生效)
    drop_rate: float = 0.0              # 丢包率 [0,1): 模拟无线链路丢包
    origin_lat_deg: float = 30.0        # 田块原点纬度
    origin_lon_deg: float = 120.0       # 田块原点经度
    origin_alt_m: float = 10.0          # 田块海拔 (m)


class SensorSimulator:
    """
    IMU + RTK 传感器仿真器

    每仿真帧调用一次 generate_packets(): 生成 (上一帧时刻, 本帧时刻] 区间
    内按 rate_hz 均匀分布的数据包 (末包时间戳 = 本帧时刻), 由调用方逐包
    feed_rtk_data() 注入导航核心。包位置在帧间线性插值。
    """

    def __init__(self, imu_config: IMUConfig = None,
                 rtk_config: RTKConfig = None, seed: int = None):
        self.imu_cfg = imu_config or IMUConfig()
        self.rtk_cfg = rtk_config or RTKConfig()
        self._rng = np.random.default_rng(seed)
        # 帧间插值基准 (上一帧末的仿真时刻与车辆位置)
        self._last_time: Optional[float] = None
        self._last_pos: Optional[tuple] = None
        # 统计 (GUI 传感器面板 / CSV 输出)
        self.packets_sent = 0
        self.packets_dropped = 0
        self.last_rx_time: float = float('nan')

    @property
    def active(self) -> bool:
        """是否启用任一传感器"""
        return self.imu_cfg.enabled or self.rtk_cfg.enabled

    def reset(self):
        """复位帧间基准 (仿真重置时调用; 统计清零, 随机流不重置)"""
        self._last_time = None
        self._last_pos = None
        self.packets_sent = 0
        self.packets_dropped = 0
        self.last_rx_time = float('nan')

    def generate_packets(self, t_now: float,
                         veh_x: float, veh_y: float,
                         veh_heading: float, veh_speed: float,
                         veh_roll_rad: float, veh_pitch_rad: float
                         ) -> List[Dict]:
        """
        生成 (上一注入时刻, t_now] 的数据包列表 (时间戳升序, 末包 = t_now)。

        Args:
            t_now: 本帧仿真时间 (s), 与 get_navigation_data 的 frame_time 同源
            veh_x, veh_y: 车辆全局位置 (ENU, m)
            veh_heading: 车辆朝向 (rad, 0=北, 顺时针正, 罗盘角)
            veh_speed: 车速 (m/s)
            veh_roll_rad, veh_pitch_rad: 车身真值姿态 (rad, 右倾正/抬头正)

        Returns:
            feed_rtk_data 兼容的 dict 列表; 未启用时返回空列表
        """
        if not self.active:
            self._last_time = t_now
            self._last_pos = (veh_x, veh_y)
            return []

        # ── 时间网格: 上一帧时刻 -> 本帧时刻, 按 rate_hz 取包数, 均匀分布 ──
        # 首帧只发一包; 之后每帧 rate_hz/fps 包 (50Hz/15fps ~ 3-4 包)
        if self._last_time is None or t_now <= self._last_time:
            times = [t_now]
        else:
            interval = t_now - self._last_time
            n = max(1, int(round(interval * self.rtk_cfg.rate_hz)))
            times = [self._last_time + (k + 1) * interval / n
                     for k in range(n)]

        # ── 位置插值基准 ──
        t_prev = self._last_time if self._last_time is not None else t_now
        span = max(t_now - t_prev, 1e-9)
        if self._last_pos is None:
            self._last_pos = (veh_x, veh_y)
        x0, y0 = self._last_pos

        packets = []
        for t_k in times:
            # 丢包模拟: 独立概率丢弃 (算法侧将回退缓冲内更旧样本或超窗退化)
            if self.rtk_cfg.drop_rate > 0 \
                    and self._rng.random() < self.rtk_cfg.drop_rate:
                self.packets_dropped += 1
                continue
            a = min(max((t_k - t_prev) / span, 0.0), 1.0)
            px = x0 + (veh_x - x0) * a
            py = y0 + (veh_y - y0) * a
            packets.append(self._make_packet(
                t_k, px, py, veh_heading, veh_speed,
                veh_roll_rad, veh_pitch_rad))
            self.packets_sent += 1
            self.last_rx_time = t_k

        self._last_time = t_now
        self._last_pos = (veh_x, veh_y)
        return packets

    def _make_packet(self, t: float, x: float, y: float,
                     heading_rad: float, speed_ms: float,
                     roll_rad: float, pitch_rad: float) -> Dict:
        """组装单包数据 (字段与 feed_rtk_data 的 dict 解析对齐)"""
        rtk = self.rtk_cfg
        imu = self.imu_cfg
        # 车辆模型 heading 为罗盘角 (0=北, 顺时针正), 与 RTK 双天线
        # 航向 / NMEA HDT/COG 及 C++ nav_geo.hpp (Forward unit in ENU:
        # (sin h, cos h)) 的约定一致, 直接换算度数即可, 无需符号转换。
        heading_deg = float(np.degrees(heading_rad))

        # ── 位置: ENU 真值 + 各向同性高斯噪声 -> 平地球逆投影为经纬度 ──
        # 逆变换: lat = lat0 + north/111320, lon = lon0 + east/(111320*cos(lat0))
        # 与 nav_geo.hpp to_local_enu 正反互逆, 保证算法侧 ENU 解算无模型误差
        if rtk.enabled:
            ne, nn = self._rng.normal(0.0, rtk.position_noise_std_m, 2)
            pos_status = rtk.pos_status
            speed = max(0.0, speed_ms
                        + self._rng.normal(0.0, rtk.speed_noise_std_ms))
            heading_meas = (heading_deg
                            + self._rng.normal(0.0, rtk.heading_noise_std_deg))
            sat_count = int(self._rng.integers(21, 27))
            diff_age = float(self._rng.uniform(0.2, 0.6))
        else:
            # IMU-only: 位置/速度填真值, pos_status=0 -> ENU hint 不可用
            ne, nn = 0.0, 0.0
            pos_status = 0
            speed = speed_ms
            heading_meas = None       # 单天线无航向 (ENU hint 前置条件失败)
            sat_count = 8
            diff_age = 0.0

        cos_lat0 = np.cos(np.radians(rtk.origin_lat_deg))
        pkt = {
            "recv_time": float(t),
            "latitude": float(rtk.origin_lat_deg + (y + nn) / _METRES_PER_DEG),
            "longitude": float(rtk.origin_lon_deg
                               + (x + ne) / (_METRES_PER_DEG * cos_lat0)),
            "altitude": float(rtk.origin_alt_m),
            "speed_ms": float(speed),
            "speed_dir": heading_deg,
            "pos_status": int(pos_status),
            "sat_count": sat_count,
            "diff_age_s": diff_age,
            # 可选字段: None = 该传感器不可用 (C++ 端解析为 NaN)
            "heading": heading_meas,
            "roll": None,
            "pitch": None,
        }

        # ── IMU 姿态: 真值 + 偏置 + 每包独立测量噪声 (度) ──
        # 符号约定与 C++ set_attitude 一致 (roll 右侧下沉正 / pitch 抬头正),
        # 无需任何符号换算
        if imu.enabled:
            pkt["roll"] = float(
                np.degrees(roll_rad) + imu.bias_roll_deg
                + self._rng.normal(0.0, imu.roll_noise_std_deg))
            pkt["pitch"] = float(
                np.degrees(pitch_rad) + imu.bias_pitch_deg
                + self._rng.normal(0.0, imu.pitch_noise_std_deg))

        return pkt

    def get_status(self) -> Dict:
        """传感器状态摘要 (GUI 传感器面板 / CSV 输出)"""
        return {
            "imu_enabled": self.imu_cfg.enabled,
            "rtk_enabled": self.rtk_cfg.enabled,
            "rate_hz": self.rtk_cfg.rate_hz if self.rtk_cfg.enabled else 0.0,
            "pos_status": (self.rtk_cfg.pos_status
                           if self.rtk_cfg.enabled else 0),
            "packets_sent": self.packets_sent,
            "packets_dropped": self.packets_dropped,
            "drop_rate": self.rtk_cfg.drop_rate,
            "last_rx_time": self.last_rx_time,
        }
