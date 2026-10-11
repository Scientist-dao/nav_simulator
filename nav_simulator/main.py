#!/usr/bin/env python3
"""
main.py - 视觉导航实时仿真器入口

用法:
  # 基本用法 (mask 模式)
  python nav_simulator/main.py --scenario straight_3row

  # 指定行距和速度
  python nav_simulator/main.py --scenario curved_gentle --row-spacing 0.75 --speed 1.5

  # 传感器仿真 (RTK/IMU, 单窗口仪表盘显示)
  python nav_simulator/main.py --scenario sensor_full_stack
  python nav_simulator/main.py --scenario straight_3row --imu --rtk \
      --attitude-projection --bump 3.0

  # 无头模式 (不显示 GUI, 只记录数据)
  python nav_simulator/main.py --scenario straight_3row --headless --record output/

  # 列出所有可用场景
  python nav_simulator/main.py --list

键盘控制 (GUI 模式):
  空格   - 暂停/继续
  R      - 重置车辆到起始位置
  +/-    - 增加/减少车速
  M      - 切换传感器融合模式 (纯视觉 / +IMU / +RTK)
  Q/ESC  - 退出
"""

import argparse
import dataclasses
import json
import os
import sys
import time

# 确保能导入 nav_simulator 包
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.dirname(_SIM_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from nav_simulator.core.simulator import (
    NavSimulator, SimConfig, SENSOR_MODE_NAMES)
from nav_simulator.core.field_model import FieldConfig, RowShape
from nav_simulator.core.vehicle_model import VehicleParams
from nav_simulator.core.sensor_simulator import IMUConfig, RTKConfig
from nav_simulator.scenarios.presets import get_scenario, list_scenarios, SCENARIOS
from nav_simulator.gui.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="视觉导航实时仿真器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scenario", "-s", default="straight_3row",
                        help="场景名称 (用 --list 查看所有可用场景)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="列出所有可用场景")
    parser.add_argument("--speed", type=float, default=None,
                        help="覆盖车辆速度 (m/s)")
    parser.add_argument("--rows", type=int, default=None,
                        help="覆盖作物行数 (奇/偶均可; 偶数行时地块"
                             "几何中心是苗缝, 配合 --offset 摆位形)")
    parser.add_argument("--row-spacing", type=float, default=None,
                        help="覆盖行距 (m)")
    parser.add_argument("--offset", type=float, default=None,
                        help="覆盖初始横向偏移 (m)")
    parser.add_argument("--fps", type=float, default=None,
                        help="覆盖仿真帧率")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式 (不显示 GUI)")
    parser.add_argument("--record", "-r", default=None, metavar="DIR",
                        help="记录仿真数据到指定目录")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="最大仿真帧数")
    parser.add_argument("--config", "-c", default=None,
                        help="自定义配置文件路径 (JSON)")
    parser.add_argument("--nav-debug", action="store_true",
                        help="开启导航核心 debug 输出")
    # ── 传感器仿真 (RTK/IMU) ──
    parser.add_argument("--sensor-mode", type=int, default=None,
                        choices=[0, 1, 2],
                        help="传感器融合模式: 0=纯视觉 1=+IMU 2=+RTK (默认 2)")
    parser.add_argument("--imu", action="store_true",
                        help="启用 IMU 姿态仿真 (真值+噪声, 50Hz)")
    parser.add_argument("--rtk", action="store_true",
                        help="启用 RTK 定位仿真 (ENU hint, 50Hz)")
    parser.add_argument("--attitude-projection", action="store_true",
                        help="车身姿态进相机投影 (扰动源: 相机随车身俯仰/侧倾)")
    parser.add_argument("--bump", type=float, default=None, metavar="X",
                        help="颠簸强度倍率 (标称 1.0, 传感器场景常用 3.0)")
    parser.add_argument("--sensor-seed", type=int, default=None,
                        help="传感器噪声随机种子 (可复现)")
    # ── 导航后端 (A/B) ──
    parser.add_argument("--backend", default="baseline",
                        choices=["baseline", "rowmap", "rowmap_cpp"],
                        help="导航后端: baseline (C++ 基线), "
                             "rowmap (行结构地图参考实现) 或 "
                             "rowmap_cpp (RowMap C++ 工程, 部署形态)")
    # ── 三网络开关 (仅 rowmap 后端; 省路径用默认权重) ──
    parser.add_argument("--qnet", nargs="?", const="", default=None,
                        metavar="PATH",
                        help="C3 质量网络 npz (默认 models/c3_quality_weights.npz)")
    parser.add_argument("--hnet", nargs="?", const="", default=None,
                        metavar="PATH",
                        help="C5 行尾感知网络 npz (默认 models/c5_headland_weights.npz)")
    parser.add_argument("--bev", nargs="?", const="", default=None,
                        metavar="PATH",
                        help="C4 BEV ConvGRU npz (默认 models/c4_bev_weights.npz)")
    parser.add_argument("--no-abort", action="store_true",
                        help="关闭掉头中止机制 (rowmap 后端: 掉头中识别"
                             "到任务行延续仍继续掉头 —— 消融对比用)")
    # ── GUI ──
    parser.add_argument("--classic-gui", action="store_true",
                        help="经典双窗口 (默认单窗口综合仪表盘)")
    return parser.parse_args()


def load_config_file(path: str) -> dict:
    """加载 JSON 配置文件"""
    if not os.path.exists(path):
        print(f"Warning: config file not found: {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def apply_sensor_config(sim_config: SimConfig, sensors: dict):
    """应用 JSON 配置的 sensors 段到 SimConfig

    结构 (全部可选):
      sensors: {mode, attitude_in_projection, bump_intensity, seed,
                imu: {enabled, roll_noise_std_deg, ...},
                rtk: {enabled, rate_hz, pos_status, ...}}
    """
    if not sensors:
        return
    if sensors.get("mode") is not None:
        sim_config.sensor_mode = int(sensors["mode"])
    if sensors.get("attitude_in_projection") is not None:
        sim_config.attitude_in_projection = bool(
            sensors["attitude_in_projection"])
    if sensors.get("bump_intensity") is not None:
        sim_config.bump_intensity = float(sensors["bump_intensity"])
    if sensors.get("seed") is not None:
        sim_config.sensor_seed = int(sensors["seed"])

    imu = sensors.get("imu")
    if imu:
        sim_config.imu_config = IMUConfig(
            enabled=bool(imu.get("enabled", True)),
            roll_noise_std_deg=float(imu.get("roll_noise_std_deg", 0.15)),
            pitch_noise_std_deg=float(imu.get("pitch_noise_std_deg", 0.15)),
            bias_roll_deg=float(imu.get("bias_roll_deg", 0.0)),
            bias_pitch_deg=float(imu.get("bias_pitch_deg", 0.0)),
        )
    rtk = sensors.get("rtk")
    if rtk:
        sim_config.rtk_config = RTKConfig(
            enabled=bool(rtk.get("enabled", True)),
            rate_hz=float(rtk.get("rate_hz", 50.0)),
            position_noise_std_m=float(rtk.get("position_noise_std_m", 0.02)),
            heading_noise_std_deg=float(rtk.get("heading_noise_std_deg", 0.5)),
            speed_noise_std_ms=float(rtk.get("speed_noise_std_ms", 0.05)),
            pos_status=int(rtk.get("pos_status", 4)),
            drop_rate=float(rtk.get("drop_rate", 0.0)),
            origin_lat_deg=float(rtk.get("origin_lat_deg", 30.0)),
            origin_lon_deg=float(rtk.get("origin_lon_deg", 120.0)),
            origin_alt_m=float(rtk.get("origin_alt_m", 10.0)),
        )


def main():
    args = parse_args()

    # ── 列出场景 ──
    if args.list:
        print("\n可用场景:")
        print("-" * 60)
        for name, desc in sorted(list_scenarios().items()):
            print(f"  {name:25s}  {desc}")
        print("-" * 60)
        print(f"共 {len(SCENARIOS)} 个场景\n")
        return

    # ── 获取场景配置 ──
    try:
        scenario_info = get_scenario(args.scenario)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    field_config, sim_config = scenario_info["factory"]()
    print(f"场景: {args.scenario}")
    print(f"描述: {scenario_info['description']}")

    # ── 自定义配置文件覆盖（优先级低于命令行） ──
    vehicle_params = None
    if args.config:
        cfg = load_config_file(args.config)
        sim_cfg = cfg.get("simulation", {})
        for k, v in sim_cfg.items():
            if hasattr(sim_config, k):
                setattr(sim_config, k, v)
        # sensors 段 (RTK/IMU 仿真)
        apply_sensor_config(sim_config, cfg.get("sensors", {}))
        # 车辆参数 (旧版读取后未应用, 此处修复)
        veh_cfg = cfg.get("vehicle", {})
        if veh_cfg:
            valid = {f.name for f in dataclasses.fields(VehicleParams)}
            kwargs = {k: v for k, v in veh_cfg.items() if k in valid}
            if kwargs:
                vehicle_params = VehicleParams(**kwargs)
        field_cfg = cfg.get("field", {})
        for k, v in field_cfg.items():
            if hasattr(field_config, k):
                if k == "row_shape" and isinstance(v, str):
                    v = RowShape(v)
                setattr(field_config, k, v)

    # ── 命令行覆盖（最高优先级） ──
    if args.speed is not None:
        sim_config.speed_ms = args.speed
    if args.rows is not None:
        field_config.num_rows = args.rows
    if args.row_spacing is not None:
        field_config.row_spacing_m = args.row_spacing
    if args.offset is not None:
        sim_config.initial_offset_m = args.offset
    if args.fps is not None:
        sim_config.fps = args.fps
    if args.max_frames is not None:
        sim_config.max_frames = args.max_frames
    if args.nav_debug:
        sim_config.nav_debug = True
    if args.backend:
        sim_config.nav_backend = args.backend
    if args.sensor_mode is not None:
        sim_config.sensor_mode = args.sensor_mode
    if args.imu:
        imu_cfg = sim_config.imu_config or IMUConfig()
        imu_cfg.enabled = True
        sim_config.imu_config = imu_cfg
    if args.rtk:
        rtk_cfg = sim_config.rtk_config or RTKConfig()
        rtk_cfg.enabled = True
        sim_config.rtk_config = rtk_cfg
    if args.attitude_projection:
        sim_config.attitude_in_projection = True
    if args.bump is not None:
        sim_config.bump_intensity = args.bump
    if args.sensor_seed is not None:
        sim_config.sensor_seed = args.sensor_seed

    # ── 打印参数 ──
    print(f"\n仿真参数:")
    print(f"  帧率:       {sim_config.fps} fps")
    print(f"  车速:       {sim_config.speed_ms} m/s")
    print(f"  初始偏移:   {sim_config.initial_offset_m} m")
    print(f"  初始航向:   {sim_config.initial_heading_deg} deg")
    print(f"  行数:       {field_config.num_rows}")
    print(f"  行距:       {field_config.row_spacing_m} m")
    print(f"  行形状:     {field_config.row_shape.value}")
    print(f"  噪声概率:   {sim_config.noise_probability}")
    imu_on = sim_config.imu_config is not None and sim_config.imu_config.enabled
    rtk_on = sim_config.rtk_config is not None and sim_config.rtk_config.enabled
    if imu_on or rtk_on or sim_config.attitude_in_projection:
        mode_name = SENSOR_MODE_NAMES.get(
            sim_config.sensor_mode, f"MODE {sim_config.sensor_mode}")
        print(f"  传感器:     IMU={'ON' if imu_on else 'off'}"
              f" RTK={'ON' if rtk_on else 'off'}"
              f"  mode={sim_config.sensor_mode} ({mode_name})")
        if sim_config.attitude_in_projection:
            print(f"  姿态投影:   ON (相机随车身俯仰/侧倾)")
        if sim_config.bump_intensity != 1.0:
            print(f"  颠簸强度:   x{sim_config.bump_intensity:.1f}")
        if sim_config.sensor_seed is not None:
            print(f"  噪声种子:   {sim_config.sensor_seed}")
    print()

    # ── 初始化仿真器 ──
    print("正在初始化导航核心...", flush=True)
    simulator = NavSimulator(field_config, sim_config,
                             vehicle_params=vehicle_params)
    print(f"导航核心初始化完成 (crop: {simulator.crop_w}x{simulator.crop_h})",
          flush=True)

    # ── 三网络注入 (rowmap/rowmap_cpp 后端; 语义与 run_benchmark 一致) ──
    nets_on = []
    if args.backend == "rowmap":
        if args.qnet is not None:
            from nav_simulator.dl.c3_runtime import load_quality_net
            p = args.qnet or os.path.join(
                _WORKSPACE_DIR, "models", "c3_quality_weights.npz")
            simulator._vn.set_quality_net(load_quality_net(p))
            nets_on.append("C3")
        if args.hnet is not None:
            from nav_simulator.dl.c5_runtime import load_headland_net
            p = args.hnet or os.path.join(
                _WORKSPACE_DIR, "models", "c5_headland_weights.npz")
            simulator._vn.set_headland_net(load_headland_net(p))
            nets_on.append("C5")
        if args.bev is not None:
            from nav_simulator.dl.c4_runtime import load_bev_net
            p = args.bev or os.path.join(
                _WORKSPACE_DIR, "models", "c4_bev_weights.npz")
            simulator._vn.set_bev_net(load_bev_net(p))
            nets_on.append("C4")
    elif args.backend == "rowmap_cpp":
        # C++ 侧吃 .wts 部署权重 (mlp.hpp 小端二进制): npz 缺失或
        # 过期 (比 .wts 新) 时经 tools/export_weights.py 自动导出到
        # npz 同名 .wts; 加载失败仅告警不中断 (门禁另有专项对拍)
        def _export_wts(npz_path):
            wts_path = os.path.splitext(npz_path)[0] + ".wts"
            if (not os.path.isfile(wts_path)
                    or os.path.getmtime(wts_path)
                    < os.path.getmtime(npz_path)):
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "rm_export_weights", os.path.join(
                        _WORKSPACE_DIR, "rowmap_navigation",
                        "tools", "export_weights.py"))
                exp = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(exp)
                exp.npz_to_wts(npz_path, wts_path)
            return wts_path

        if args.qnet is not None:
            p = args.qnet or os.path.join(
                _WORKSPACE_DIR, "models", "c3_quality_weights.npz")
            if not os.path.isfile(p):
                print(f"警告: C3 权重缺失 ({p}), 跳过注入", flush=True)
            elif simulator._vn.set_quality_net(_export_wts(p)):
                nets_on.append("C3")
            else:
                print(f"警告: C3 .wts 加载失败 ({p})", flush=True)
        if args.hnet is not None:
            p = args.hnet or os.path.join(
                _WORKSPACE_DIR, "models", "c5_headland_weights.npz")
            if not os.path.isfile(p):
                print(f"警告: C5 权重缺失 ({p}), 跳过注入", flush=True)
            elif simulator._vn.set_headland_net(_export_wts(p)):
                nets_on.append("C5")
            else:
                print(f"警告: C5 .wts 加载失败 ({p})", flush=True)
        if args.bev is not None:
            print("警告: C4 BEV 网络尚未 C++ 化, --bev 对 rowmap_cpp "
                  "后端忽略", flush=True)
    elif args.qnet is not None or args.hnet is not None or args.bev is not None:
        print("警告: --qnet/--hnet/--bev 仅支持 --backend rowmap/"
              "rowmap_cpp, 已忽略",
              flush=True)
    if nets_on:
        print(f"感知网络: {'+'.join(nets_on)} 已注入", flush=True)

    # 掉头中止机制开关 (rowmap/rowmap_cpp 后端; 默认开)
    if args.no_abort:
        if args.backend in ("rowmap", "rowmap_cpp"):
            simulator.vehicle.row_alive_abort = False
            print("掉头中止机制: 已关闭 (消融对比)", flush=True)
        else:
            print("警告: --no-abort 仅支持 --backend rowmap/rowmap_cpp, "
                  "已忽略", flush=True)

    # ── 数据记录 ──
    record_dir = args.record
    record_file = None
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        record_path = os.path.join(record_dir, f"sim_{args.scenario}_{int(time.time())}.csv")
        record_file = open(record_path, "w")
        record_file.write(
            "frame,sim_time,veh_x,veh_y,heading_deg,speed_ms,steering_deg,"
            "roll_deg,pitch_deg,yaw_deg,front_dist,rear_dist,"
            "quality,nav_valid,num_masks,"
            "cte_true,occluded,rows_detected,rows_expected,"
            "sensor_mode,imu_applied,imu_roll_deg,imu_pitch_deg,"
            "rtk_hint_valid,rtk_pos_status,rtk_pkts_frame,"
            "nav_wx1,nav_wy1,nav_wx2,nav_wy2\n")
        print(f"数据记录到: {record_path}")

    # ── 可视化 ──
    visualizer = None
    if not args.headless:
        visualizer = Visualizer(
            simulator, scenario_name=args.scenario,
            show_dashboard=not args.classic_gui)
        print("\n键盘控制: 空格=暂停  R=重置  +/-=调速"
              "  M=切传感器模式  Q/ESC=退出")
        print("开始仿真...\n", flush=True)

    # ── 主循环 ──
    # GUI 模式无限跑: 场景 max_frames 只约束 headless/跑批
    # (occlusion_burst 900 帧 = 60s, 跑满自动退出会误以为崩溃;
    # 观察地图生长/遮挡行为需要长时间运行, Q 键退出)
    if visualizer:
        sim_config.max_frames = 0
    frame_interval = 1.0 / sim_config.fps
    last_frame_time = time.time()

    def frame_callback(result):
        nonlocal last_frame_time

        # 数据记录
        if record_file:
            # 导航线世界坐标端点 (无则留空)
            nl = result.nav_line_world
            nav_cols = (f"{nl[0][0]:.4f},{nl[0][1]:.4f},"
                        f"{nl[1][0]:.4f},{nl[1][1]:.4f}"
                        if len(nl) >= 2 else ",,,")
            record_file.write(
                f"{result.frame_idx},{result.sim_time:.3f},"
                f"{result.veh_x:.4f},{result.veh_y:.4f},"
                f"{result.veh_heading_deg:.3f},{result.veh_speed:.3f},"
                f"{result.veh_steering_deg:.3f},"
                f"{result.body_roll_deg:.3f},{result.body_pitch_deg:.3f},"
                f"{result.yaw_deg:.4f},{result.front_dist:.4f},"
                f"{result.rear_dist:.4f},{result.image_quality},"
                f"{int(result.nav_valid)},{result.num_masks},"
                f"{result.cte_true:.4f},{int(result.occluded)},"
                f"{result.rows_detected},{result.rows_expected},"
                f"{result.sensor_mode},{int(result.imu_applied)},"
                f"{result.imu_roll_deg:.3f},{result.imu_pitch_deg:.3f},"
                f"{int(result.rtk_hint_valid)},{result.rtk_pos_status},"
                f"{result.rtk_packets_frame},"
                f"{nav_cols}\n")

        # 可视化
        if visualizer:
            key = visualizer.update(result, simulator)
            if not visualizer.handle_key(key, simulator):
                return False

        # 控制台输出 (每 30 帧)
        if result.frame_idx % 30 == 0:
            status = "OK" if result.nav_valid else "LOST"
            print(f"  [{result.frame_idx:5d}] "
                  f"yaw={result.yaw_deg:+7.3f}  "
                  f"front={result.front_dist:+7.4f}m  "
                  f"rear={result.rear_dist:+7.4f}m  "
                  f"q={result.image_quality:3d}  "
                  f"pos=({result.veh_x:+.2f},{result.veh_y:.1f})  "
                  f"[{status}]",
                  flush=True)

        # 帧率控制
        if not args.headless:
            elapsed = time.time() - last_frame_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.time()

        return True

    try:
        simulator.run(callback=frame_callback)
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        if visualizer:
            visualizer.close()
        if record_file:
            record_file.close()

    # ── 输出统计 ──
    summary = simulator.get_summary()
    if summary:
        print("\n" + "=" * 60)
        print("仿真统计:")
        print(f"  总帧数:     {summary['total_frames']}")
        print(f"  有效帧率:   {summary['valid_rate']*100:.1f}%")
        print(f"  Yaw 均值:   {summary['yaw_mean']:.4f} deg")
        print(f"  Yaw 标准差: {summary['yaw_std']:.4f} deg")
        print(f"  前轴偏差:   {summary['front_dist_mean']:.4f} m "
              f"(std={summary['front_dist_std']:.4f})")
        print(f"  平均质量:   {summary['quality_mean']:.1f}")
        print(f"  最终位置:   ({summary['final_position'][0]:.2f}, "
              f"{summary['final_position'][1]:.1f})")
        print(f"  行驶距离:   {summary['distance_traveled']:.1f} m")
        # 传感器链路统计
        ss = simulator.sensor_sim
        if ss.active:
            drop = (ss.packets_dropped /
                    max(1, ss.packets_sent + ss.packets_dropped) * 100.0)
            print(f"  传感器包:   发送 {ss.packets_sent}"
                  f" 丢弃 {ss.packets_dropped} ({drop:.1f}%)")
        print("=" * 60)


if __name__ == "__main__":
    main()
