"""
c5_dump.py - C5 行尾/地头感知网络 训练数据工厂 (dump 管线)

跑批仿真, 从 RowMapNavigation 的行尾 dump 旁路
(nav_core.enable_headland_dump) 收集 (22 维地图历史特征, 行终止
标签) 样本对, 存为 npz 供 c5_train.py 训练。

标签定义 (以仿真器真值位姿计算):
    margin = row_end_true_y − veh_y_true   (车沿 +y 正向行驶帧)
    label  = 1  若 margin ≤ P_END_M (行终止点在前方 P 米内或已越过)
    label  = 0  否则 (行仍向前延伸 —— 断垄/遮挡/弱苗均属此类:
                 正是基线误判行尾的地方, 必须大量喂作负样本)

正样本量产: row_end_y ∈ {30, 50, 70} 变体 (手工构造 FieldConfig,
车从 0 出发走完整行 + 行尾后 60s 的丢行/掉头过程)。掉头后反向
作业帧 (cos(heading) < 0) 丢弃 —— 特征的"前方"语义按 +y 定义,
第一版不覆盖反向。

用法 (工作区根目录):
  .venv/Scripts/python.exe -X utf8 -m nav_simulator.dl.c5_dump

输出: datasets/c5_headland_v1.npz {features (N,22), label (N,),
       margin (N,), frame, scen_id} + _meta.json
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_SIM_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from nav_simulator.core.field_model import FieldConfig, RowShape
from nav_simulator.core.simulator import NavSimulator, SimConfig
from nav_simulator.scenarios.presets import get_scenario

P_END_M = 5.0          # 行终止判定窗 (前方 P 米内)
ROW_END_VARIANTS = [30.0, 50.0, 70.0]     # 行尾 y 位置变体 (m)
TAIL_S = 60.0          # 行尾后继续仿真时长 (s, 覆盖丢行/掉头过程)
FPS = 15.0

# 负样本场景: 断垄/遮挡/弱苗是基线误判行尾的战场, 必须大量负样本
NEG_SCENARIOS = [
    "straight_3row", "row_gap", "occlusion_burst",
    "weak_seedling", "curved_gentle", "sine_wave",
]
NEG_FRAMES = 450       # 负样本场景帧数上限 (straight 足量, 扰动场景
                       # 自带 900 也压到 450 —— 负样本过剩, 无需跑满)

FEATURE_NAMES = [
    "has_task", "n_anchors", "span_m", "y_min_rel", "y_max_rel",
    "n_a_0_5", "n_a_5_10", "n_a_10_15", "n_a_15_20",
    "gap_front", "gap_behind",
    "obs_rate_5", "obs_rate_15", "qual_mean_8", "qual_now",
    "n_masks", "is_guide", "is_fit", "speed", "heading_sigma",
    "y_min_abs", "y_max_abs",
]


def _base_field_config() -> FieldConfig:
    """row_end 变体场景的田块 (3 行直线, 与 straight_3row 同参)"""
    return FieldConfig(
        num_rows=3,
        row_spacing_m=0.60,
        row_length_m=200.0,
        row_shape=RowShape.STRAIGHT,
        class_id=0,
    )


def _run_and_collect(field_config, sim_config):
    """跑一次仿真 → (features, labels, margins, frames) 或 None

    标签/正向过滤用 sim.results 真值位姿; frame 对齐: dump 的
    frame 号 f (1-based) ↔ results[f-1]。
    """
    sim = NavSimulator(field_config, sim_config)
    sim._vn.enable_headland_dump()
    n_frames = sim_config.max_frames
    sim.run(num_frames=n_frames)
    recs, meta = sim._vn.collect_headland_dump()

    res = sim.results
    row_end_true = field_config.row_end_y or field_config.row_length_m
    sim = None
    if not recs:
        return None

    feats, labels, margins, frames = [], [], [], []
    for r in recs:
        fi = r["frame"] - 1
        if fi < 0 or fi >= len(res):
            continue
        # 正向作业帧过滤 (反向掉头/作业帧的"前方"语义未定义)
        if math.cos(math.radians(res[fi].veh_heading_deg)) < 0.0:
            continue
        margin = row_end_true - res[fi].veh_y
        feats.append(r["feat"])
        labels.append(1.0 if margin <= P_END_M else 0.0)
        margins.append(margin)
        frames.append(r["frame"])
    if not feats:
        return None
    return (np.asarray(feats, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
            np.asarray(margins, dtype=np.float32),
            np.asarray(frames, dtype=np.int32),
            len(recs), int(labels.count(1.0)))


def _make_sim_config(seed: int, max_frames: int) -> SimConfig:
    # nav_backend 必须 rowmap: headland dump 是 rowmap 参考实现独有
    # 旁路 (baseline C++ _vn 无 enable_headland_dump 接口)
    return SimConfig(
        initial_offset_m=0.0,
        speed_ms=1.0,
        sensor_mode=2,
        sensor_seed=seed,
        nav_backend="rowmap",
        max_frames=max_frames,
    )


def main():
    ap = argparse.ArgumentParser(description="C5 headland data factory")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--out", default=os.path.join(
        _WORKSPACE_DIR, "datasets", "c5_headland_v1.npz"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    all_f, all_l, all_m, all_fr, all_sc, all_sd = [], [], [], [], [], []
    scen_table = []
    total_runs = (len(ROW_END_VARIANTS) + len(NEG_SCENARIOS)) * len(seeds)
    done = 0
    t_start = time.time()

    # ── 正样本: row_end 变体 (手工构造场景) ──
    for y_end in ROW_END_VARIANTS:
        scen_id = len(scen_table)
        scen_table.append(f"row_end@{y_end:.0f}m")
        n_frames = int(y_end * FPS) + int(TAIL_S * FPS)
        for seed in seeds:
            done += 1
            print(f"[{done:3d}/{total_runs}] row_end@{y_end:4.0f}m   "
                  f"seed={seed} ... ", end="", flush=True)
            t0 = time.time()
            field = _base_field_config()
            field.row_end_y = y_end
            out = _run_and_collect(field, _make_sim_config(seed, n_frames))
            if out is None:
                print("no data", flush=True)
                continue
            f_, l_, m_, fr_, n_rec, n_pos = out
            all_f.append(f_); all_l.append(l_)
            all_m.append(m_); all_fr.append(fr_)
            all_sc.append(np.full(len(l_), scen_id, dtype=np.int16))
            all_sd.append(np.full(len(l_), seed, dtype=np.int16))
            print(f"{len(l_):5d} rec ({n_pos:3d} pos, "
                  f"dropped {n_rec-len(l_)}) in {time.time()-t0:.1f}s",
                  flush=True)

    # ── 负样本: 预置场景 (断垄/遮挡/弱苗是关键负样本) ──
    for name in NEG_SCENARIOS:
        scen_id = len(scen_table)
        scen_table.append(name)
        for seed in seeds:
            done += 1
            print(f"[{done:3d}/{total_runs}] {name:20s} seed={seed} ... ",
                  end="", flush=True)
            t0 = time.time()
            info = get_scenario(name)
            field_config, sim_config = info["factory"]()
            sim_config.sensor_seed = seed
            sim_config.sensor_mode = 2
            sim_config.nav_backend = "rowmap"
            if sim_config.perturbation is not None:
                sim_config.perturbation.seed = 1000 + seed
            sim_config.max_frames = min(sim_config.max_frames or NEG_FRAMES,
                                        NEG_FRAMES)
            out = _run_and_collect(field_config, sim_config)
            if out is None:
                print("no data", flush=True)
                continue
            f_, l_, m_, fr_, n_rec, n_pos = out
            all_f.append(f_); all_l.append(l_)
            all_m.append(m_); all_fr.append(fr_)
            all_sc.append(np.full(len(l_), scen_id, dtype=np.int16))
            all_sd.append(np.full(len(l_), seed, dtype=np.int16))
            print(f"{len(l_):5d} rec ({n_pos} pos) "
                  f"in {time.time()-t0:.1f}s", flush=True)

    if not all_f:
        print("ERROR: no samples collected")
        sys.exit(1)

    feats = np.concatenate(all_f)
    labels = np.concatenate(all_l)
    margins = np.concatenate(all_m)
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=feats, label=labels, margin=margins,
        frame=np.concatenate(all_fr),
        scen_id=np.concatenate(all_sc),
        dump_seed=np.concatenate(all_sd),
    )
    n = len(labels)
    meta = {
        "feat_names": FEATURE_NAMES,
        "scenarios": scen_table,
        "n_samples": int(n),
        "n_pos": int(labels.sum()),
        "pos_rate": float(labels.mean()),
        "p_end_m": P_END_M,
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nsaved: {args.out}")
    print(f"       {meta_path}")
    print(f"total: {n} samples, pos={int(labels.sum())} "
          f"({labels.mean():.2%})")
    print(f"wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
