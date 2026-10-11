"""
presets.py - 预定义仿真场景

提供常用的田间场景预设，每个场景返回 (FieldConfig, SimConfig, description)。
"""

from typing import Dict, Tuple, Optional, Callable
from ..core.field_model import FieldConfig, RowShape
from ..core.simulator import SimConfig
from ..core.sensor_simulator import IMUConfig, RTKConfig
from ..core.perturbation import (
    PerturbationConfig, OcclusionWindow, RowGapRegion, WeakRegion)


# ── 场景注册表 ──
SCENARIOS: Dict[str, dict] = {}


def _register(name: str, description: str):
    """场景注册装饰器"""
    def decorator(fn):
        SCENARIOS[name] = {
            "factory": fn,
            "description": description,
        }
        return fn
    return decorator


def get_scenario(name: str) -> dict:
    """获取场景配置"""
    if name not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS.keys()))
        raise ValueError(
            f"Unknown scenario: '{name}'. Available: {available}")
    return SCENARIOS[name]


def list_scenarios() -> Dict[str, str]:
    """列出所有场景及其描述"""
    return {k: v["description"] for k, v in SCENARIOS.items()}


# ══════════════════════════════════════════════════════════════
# 基础场景
# ══════════════════════════════════════════════════════════════

@_register("straight_3row", "3行直线田, 车辆在中心行正上方")
def straight_3row():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        initial_heading_deg=0.0,
        speed_ms=1.0,
    )
    return field, sim


@_register("straight_offset", "3行直线田, 车辆初始右偏0.25m")
def straight_offset():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        initial_heading_deg=0.0,
        speed_ms=1.0,
    )
    return field, sim


@_register("straight_5row", "5行直线田, 密集行")
def straight_5row():
    field = FieldConfig(
        num_rows=5,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        initial_heading_deg=0.0,
        speed_ms=1.0,
    )
    return field, sim


@_register("straight_angle", "3行直线田, 车辆初始5度偏角")
def straight_angle():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        initial_heading_deg=5.0,
        speed_ms=1.0,
    )
    return field, sim


# ══════════════════════════════════════════════════════════════
# 曲线场景
# ══════════════════════════════════════════════════════════════

@_register("curved_gentle", "缓弯曲行 (半径100m), 曲线跟踪")
def curved_gentle():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.ARC,
        arc_radius_m=100.0,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
    )
    return field, sim


@_register("curved_medium", "急弯曲行 (半径50m), 弯道跟踪")
def curved_medium():
    """R=50m: 1m/s 作业速度下可全程跟踪的急弯 (3 种子 x 30m 实测
    100% valid)。感知稳定边界约在 R=40~50m, 更急见 curved_sharp"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.ARC,
        arc_radius_m=50.0,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
    )
    return field, sim


@_register("curved_sharp", "极弯曲行 (半径30m), 感知极限测试")
def curved_sharp():
    """R=30m: 超出感知算法稳定边界 (弦参考失真 + holdover 耗尽),
    任何控制/速度策略都会在 ~8-11m 后丢行 (多组对照实验证实)。
    用途: 演示丢行后的系统行为 (holdover→失效→缓慢回正)"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.ARC,
        arc_radius_m=30.0,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=0.8,
    )
    return field, sim


@_register("sine_wave", "正弦曲线行, S形跟踪")
def sine_wave():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.SINE,
        sine_amplitude_m=0.4,
        sine_period_m=40.0,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
    )
    return field, sim


# ══════════════════════════════════════════════════════════════
# 鲁棒性场景
# ══════════════════════════════════════════════════════════════

@_register("row_end", "行尾场景, 行在100m处终止")
def row_end():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        row_end_y=100.0,  # 行在 100m 处终止
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
        max_frames=1500,  # 足够到达行尾
    )
    return field, sim


@_register("row_missing_side", "缺失侧边行, 只有中间行")
def row_missing_side():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        missing_rows=[0, 2],  # 缺失左右行
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
    )
    return field, sim


@_register("row_missing_center", "缺失中心行, 只有两侧行")
def row_missing_center():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        missing_rows=[1],  # 缺失中间行
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
    )
    return field, sim


@_register("noisy_field", "噪声场景, 随机误检 mask")
def noisy_field():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
        noise_probability=0.15,  # 15% 概率出现噪声
    )
    return field, sim


# ══════════════════════════════════════════════════════════════
# 综合场景
# ══════════════════════════════════════════════════════════════

@_register("mulch_rows", "覆膜行场景 (class_id=1)")
def mulch_rows():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=1,  # mulch_line
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
    )
    return field, sim


@_register("wide_spacing", "宽行距 (0.9m) 测试")
def wide_spacing():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.90,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
    )
    return field, sim


@_register("narrow_spacing", "窄行距 (0.45m) 测试")
def narrow_spacing():
    field = FieldConfig(
        num_rows=5,
        row_spacing_m=0.45,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
    )
    return field, sim


@_register("high_speed", "高速行驶 (2.5m/s) 测试")
def high_speed():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=300.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=2.5,
    )
    return field, sim


@_register("clothoid", "回旋曲线 (渐变曲率) 测试")
def clothoid():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.CLOTHOID,
        clothoid_curvature_rate=0.001,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
    )
    return field, sim


# ══════════════════════════════════════════════════════════
# 传感器融合场景 (RTK/IMU 仿真)
# 公共设定: 3 倍颠簸 + 姿态进投影 (相机随车身倾斜) 提供真实扰动源,
# 固定种子保证可复现; GUI 单窗口仪表盘 SENSORS 面板实时显示链路状态,
# M 键可运行中切换融合模式做 A/B 对照
# ══════════════════════════════════════════════════════════

@_register("sensor_full_stack", "全传感器融合 (IMU+RTK mode2), 姿态扰动+补偿")
def sensor_full_stack():
    """RTK 固定解 + IMU 姿态, mode 2 完整链路。
    姿态进投影使相机随车身俯仰/侧倾 (扰动源), C++ set_attitude
    补偿; RTK 50Hz ENU hint 提供车轴位姿先验。"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True),
        attitude_in_projection=True,
        bump_intensity=3.0,
        sensor_seed=42,
    )
    return field, sim


@_register("sensor_imu_only", "纯 IMU 姿态补偿 (mode1), 无 RTK")
def sensor_imu_only():
    """IMU-only 部署: 无 RTK (pos_status=0, ENU hint 不可用),
    仅姿态补偿路径工作。对应实车低成本单天线方案。"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=1,
        imu_config=IMUConfig(enabled=True),
        attitude_in_projection=True,
        bump_intensity=3.0,
        sensor_seed=42,
    )
    return field, sim


@_register("sensor_rtk_float", "RTK 浮点解退化 (ENU hint 失效)")
def sensor_rtk_float():
    """pos_status=5 (RTK FLOAT): C++ _rtk_usable_for_position
    拒绝 ENU hint, 定位回退纯视觉; IMU 姿态补偿仍工作。
    分层退化演示: 传感器越差, 退化越平滑。"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True, pos_status=5),
        attitude_in_projection=True,
        bump_intensity=3.0,
        sensor_seed=42,
    )
    return field, sim


@_register("sensor_rtk_drop", "RTK 链路丢包 30%, 间歇性 hint 失效")
def sensor_rtk_drop():
    """drop_rate=0.3: 30% 数据包丢弃, 匹配窗内无新鲜样本时
    hint 间歇性失效, 观察融合输出的抖动与恢复。"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True, drop_rate=0.3),
        attitude_in_projection=True,
        bump_intensity=3.0,
        sensor_seed=42,
    )
    return field, sim


# ════════════════════════════════════════════════════════════
# 痛点注入场景 (P0 仿真基建, 视觉导航V2开发文档 §4.2)
# 公共设定: 与 sensor_full_stack 同级的生产配置 (mode 2 全传感器
# + 姿态投影 + 3 倍频簸), 对应基线缺陷 D1/D2 的复现战场;
# 除 deadlock_center 外均 initial_offset=0.25 (先进入稳态再考验扰动,
# 收敛约 6.5s/6.5m, 扰动区从 20m 起避开捕获动态)
# ════════════════════════════════════════════════════════════

@_register("deadlock_center", "对称死锁点: 初始恰偏半行距 0.30m (D1)")
def deadlock_center():
    """车起始停在两行正中间 (偏 0.30m = 半行距): 基线对称对模式
    把导航线定在车正下方, 判定已对准永不纠偏 (实测 90 帧不收敛);
    同时该几何位形下两侧轮心距邻行苗带仅 4cm, 是最大伤害位。
    预期: 基线压苗率~100%; RowMap 行锁定后应直接收敛。"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.30,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True),
        attitude_in_projection=True,
        bump_intensity=3.0,
        sensor_seed=42,
        max_frames=450,       # 30s 足够分辨死锁 (正常收敛仅需 ~7s)
    )
    return field, sim


@_register("occlusion_burst", "时变遮挡: 3 段杂草遮挡镜头 4/5/3s (D2)")
def occlusion_burst():
    """车已收敛后遭遇三段完全遮挡 (杂草丛挡镜头 → mask 全空):
    20~24s (4s), 35~40s (5s), 55~58s (3s)。
    基线 holdover 仅 0.67s (10 帧@15fps), 窗内输出清零后方向盘
    回正赌运气; 预期: 窗内丢线率~80%+, 漂移取决于车况。"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True),
        attitude_in_projection=True,
        bump_intensity=3.0,
        sensor_seed=42,
        max_frames=900,       # 60s, 车走 60m, 覆盖全部三段遮挡
        perturbation=PerturbationConfig(
            occlusions=[
                OcclusionWindow(start_s=20.0, duration_s=4.0),
                OcclusionWindow(start_s=35.0, duration_s=5.0),
                OcclusionWindow(start_s=55.0, duration_s=3.0),
            ],
            seed=7,
        ),
    )
    return field, sim


@_register("row_gap", "断垄: 全部行 30~38m 无苗, 行断裂+近端真空 (D2)")
def row_gap():
    """全部行在 y∈[30,38] (8m) 无苗 (断垄/出苗不全):
    车接近时先看到行断裂 (近段先消失远段保留), 进入区间后仅剩
    远端弱信息。与 occlusion_burst 互补: 一个是完全失明 (镜头被挡),
    一个是田里真没苗。考验短记忆下的远端维持与重捕获。"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True),
        attitude_in_projection=True,
        bump_intensity=3.0,
        sensor_seed=42,
        max_frames=900,       # 60s, 车走 60m, 完整穿越断垄区
        perturbation=PerturbationConfig(
            gaps=[RowGapRegion(y_start=30.0, y_end=38.0)],
            seed=7,
        ),
    )
    return field, sim


@_register("weak_seedling", "弱苗区: 20~50m 苗细弱+位置抖动+低置信度")
def weak_seedling():
    """y∈[20,50] (30m) 弱苗带: 行宽×0.6 + 逐帧横向抖动 σ=3cm +
    检测 score 0.55 (低于正常 0.95)。考验观测质量评估与融合
    降权 (RowMap C3 质量评估 MLP 的目标场景); 基线手工规则对
    低置信度 mask 的处理粗糙, 预期输出噪声明显增大。"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True),
        attitude_in_projection=True,
        bump_intensity=3.0,
        sensor_seed=42,
        max_frames=900,       # 60s, 车走 60m, 弱苗区占 20~50m
        perturbation=PerturbationConfig(
            weak_regions=[
                WeakRegion(y_start=20.0, y_end=50.0,
                           score_override=0.55,
                           jitter_std_m=0.03,
                           width_scale=0.6),
            ],
            seed=7,
        ),
    )
    return field, sim


# ══════════════════════════════════════════════════════════
#  深度学习加难场景 (C4 BEV ConvGRU 主战场; 与 dl/c4_dump.py
#  的训练数据同源 —— A/B 消融与训练集必须用同一场景定义)
# ══════════════════════════════════════════════════════════

@_register("hard_gap15", "加难: 15m 长断垄 (25~40m 全部行无苗)")
def hard_gap15():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True),
        attitude_in_projection=True,
        bump_intensity=3.0,
        max_frames=900,
        perturbation=PerturbationConfig(
            gaps=[RowGapRegion(y_start=25.0, y_end=40.0)], seed=7),
    )
    return field, sim


@_register("hard_gap25", "极限: 25m 超长断垄 (25~50m 全部行无苗), 掉头中止+行关联门限压力场景")
def hard_gap25():
    """25m 断垄 > 前视外推极限 (LOOKAHEAD 10m + 锚点支撑): 断垄尾
    必触发掉头; 对岸 17~22m 检出是边缘事件, 检出时任一行延续锚点
    落图 → 掉头中止回行向穿越; 未检出时完成掉头干净倒回 (信息论
    极限)。同时考验关联门限不被对岸检出劫持 (端外钳位)"""
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True),
        attitude_in_projection=True,
        bump_intensity=3.0,
        max_frames=900,
        perturbation=PerturbationConfig(
            gaps=[RowGapRegion(y_start=25.0, y_end=50.0)], seed=7),
    )
    return field, sim


@_register("hard_occl10", "加难: 10s 持续遮挡 (20~30s, 150 帧@15fps 完全失明)")
def hard_occl10():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True),
        attitude_in_projection=True,
        bump_intensity=3.0,
        max_frames=900,
        perturbation=PerturbationConfig(
            occlusions=[OcclusionWindow(start_s=20.0, duration_s=10.0)],
            seed=7),
    )
    return field, sim


@_register("weak_curved", "加难: 弱苗 + 弯道叠加 (R=50m 圆弧 + 20~60m 弱苗带)")
def weak_curved():
    field = FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.ARC,
        arc_radius_m=50.0,
        class_id=0,
    )
    sim = SimConfig(
        initial_offset_m=0.25,
        speed_ms=1.0,
        sensor_mode=2,
        imu_config=IMUConfig(enabled=True),
        rtk_config=RTKConfig(enabled=True),
        attitude_in_projection=True,
        bump_intensity=3.0,
        max_frames=900,
        perturbation=PerturbationConfig(
            weak_regions=[WeakRegion(y_start=20.0, y_end=60.0,
                                     score_override=0.55,
                                     jitter_std_m=0.03,
                                     width_scale=0.6)],
            seed=7),
    )
    return field, sim
