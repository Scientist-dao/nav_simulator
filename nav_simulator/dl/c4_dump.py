"""
c4_dump.py - C4 BEV 时序证据聚合 (ConvGRU) 训练数据工厂 (dump 管线)

跑批仿真, 同时开启 band dump (观测点) + headland dump (帧位姿,
遮挡帧也有记录 —— 恰是 ConvGRU 需要保持时序记忆的帧), 存轻量
源数据 (观测点坐标/帧位姿/真值行折线/基准原点), BEV 网格构造
是纯确定性函数, 留给训练侧 Dataset 在线生成 (128×128 网格
直接存盘会爆体积)。

场景: 标准多样性 (直线/弯道/正弦/弱苗/断垄/遮挡) + 加难组合
(文档 §3.6 要求: 15m 长断垄 / 10s 持续遮挡 / 弱苗+弯道叠加 ——
标准场景纯几何 q=70 已 0% 丢线, 学习版必须在更难的地方证明价值)。

用法 (工作区根目录):
  .venv/Scripts/python.exe -X utf8 -m nav_simulator.dl.c4_dump

输出: datasets/c4_bev_v1.npz
  {name}_obs_xy    (M,2) f32   观测点 (wx,wy) RTK 基准系
  {name}_obs_frame (M,)   i32
  {name}_pose      (F,3) f32   每帧 (px,py,heading) RTK 基准系
  {name}_row_pts   (P,2) f32   真值行折线点 (田块全局系, 行拼接)
  {name}_row_bounds(R+1,) i32  行 i 的点区间 [bounds[i], bounds[i+1])
  {name}_origin    (2,)   f32  world_origin (基准系→全局系平移)
  scen_list        (S,)   <str>  场景名
  scen_seeds       (S,)   i32    每组对应的 sensor seed
"""

import argparse
import os
import sys
import time

import numpy as np

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_SIM_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from nav_simulator.core.field_model import FieldModel
from nav_simulator.core.simulator import NavSimulator
from nav_simulator.scenarios.presets import get_scenario

# 标准 (多样性) + 加难 (C4 主战场)
STD_SCENARIOS = [
    "straight_3row", "curved_gentle", "curved_medium", "sine_wave",
    "weak_seedling", "row_gap", "occlusion_burst",
]

# 加难场景已注册进 presets.py (与 run_benchmark A/B 消融同源)
HARD_SCENARIOS = ["hard_gap15", "hard_occl10", "weak_curved"]


def dump_scenario(name: str, seed: int):
    """跑一次仿真 → 源数据 dict (观测点/帧位姿/行折线/origin)"""
    info = get_scenario(name)
    field_config, sim_config = info["factory"]()
    if sim_config.max_frames == 0:
        sim_config.max_frames = 900
    sim_config.sensor_seed = seed
    sim_config.sensor_mode = 2
    sim_config.nav_backend = "rowmap"
    if sim_config.perturbation is not None:
        sim_config.perturbation.seed = 1000 + seed

    sim = NavSimulator(field_config, sim_config)
    sim._vn.enable_band_dump()
    sim._vn.enable_headland_dump()
    sim.run(num_frames=sim_config.max_frames)
    band_recs, band_meta = sim._vn.collect_band_dump()
    head_recs, _ = sim._vn.collect_headland_dump()
    origin = band_meta.get("world_origin")
    sim = None
    if origin is None:
        return None

    # 观测点 (RTK 基准系)
    obs_xy = np.asarray([[r["wx"], r["wy"]] for r in band_recs],
                        dtype=np.float32).reshape(-1, 2)
    obs_frame = np.asarray([r["frame"] for r in band_recs],
                           dtype=np.int32)
    # 帧位姿: headland 每帧一条 (含遮挡/q=0 帧), 帧号 1-based
    # (存 pose_frame 供训练侧建帧号→位姿映射 —— headland dump
    #  在极端情况下可能跳帧, 不能用数组下标直接对齐)
    pose_frame = np.asarray([r["frame"] for r in head_recs],
                            dtype=np.int32)
    pose = np.asarray([[r["px"], r["py"], 0.0] for r in head_recs],
                      dtype=np.float32).reshape(-1, 3)
    # heading 从 band 记录补 (headland 未存) —— 用该帧任一 band 的 h
    h_by_frame = {}
    for r in band_recs:
        h_by_frame.setdefault(r["frame"], r["h"])
    for i, r in enumerate(head_recs):
        h = h_by_frame.get(r["frame"])
        if h is not None:
            pose[i, 2] = h

    # 真值行折线 (田块全局系; FieldModel 从 field_config 重建, 只读)
    rows = [row for row in FieldModel(field_config).rows
            if row.active and len(row.points) >= 2]
    pts_all, bounds = [], [0]
    for row in rows:
        pts_all.extend(row.points)
        bounds.append(len(pts_all))
    row_pts = np.asarray(pts_all, dtype=np.float32).reshape(-1, 2)

    return {
        "obs_xy": obs_xy, "obs_frame": obs_frame,
        "pose": pose, "pose_frame": pose_frame,
        "row_pts": row_pts,
        "row_bounds": np.asarray(bounds, dtype=np.int32),
        "origin": np.asarray(origin, dtype=np.float32),
        "n_frames": int(sim_config.max_frames),
        "n_obs": int(len(obs_xy)),
    }


def main():
    ap = argparse.ArgumentParser(description="C4 BEV data factory")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--out", default=os.path.join(
        _WORKSPACE_DIR, "datasets", "c4_bev_v1.npz"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    scen_names, scen_seeds, payloads = [], [], {}
    names = STD_SCENARIOS + HARD_SCENARIOS
    total_runs = len(names) * len(seeds)
    done = 0
    t_start = time.time()

    for name in names:
        for seed in seeds:
            done += 1
            print(f"[{done:3d}/{total_runs}] {name:20s} seed={seed} ... ",
                  end="", flush=True)
            t0 = time.time()
            data = dump_scenario(name, seed)
            if data is None:
                print("skip (no origin)", flush=True)
                continue
            key = f"{name}#s{seed}"
            scen_names.append(key)
            scen_seeds.append(seed)
            payloads[f"{key}::obs_xy"] = data["obs_xy"]
            payloads[f"{key}::obs_frame"] = data["obs_frame"]
            payloads[f"{key}::pose"] = data["pose"]
            payloads[f"{key}::pose_frame"] = data["pose_frame"]
            payloads[f"{key}::row_pts"] = data["row_pts"]
            payloads[f"{key}::row_bounds"] = data["row_bounds"]
            payloads[f"{key}::origin"] = data["origin"]
            print(f"{data['n_obs']:6d} obs / {data['n_frames']} frames "
                  f"in {time.time()-t0:.1f}s", flush=True)

    if not payloads:
        print("ERROR: no data collected")
        sys.exit(1)

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.out,
        scen_list=np.asarray(scen_names),
        scen_seeds=np.asarray(scen_seeds, dtype=np.int32),
        **payloads,
    )
    print(f"\nsaved: {args.out}")
    print(f"scenarios: {len(scen_names)} runs")
    print(f"wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
