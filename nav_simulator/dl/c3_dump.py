"""
c3_dump.py - C3 异方差观测质量网络 训练数据工厂 (dump 管线)

跑批多场景 headless 仿真, 从 RowMapNavigation 的 band dump 旁路
(nav_core.enable_band_dump) 收集 (14 维特征, 真值横向误差) 样本对,
存为 npz 供 c3_train.py 训练。

标签定义: 观测点世界坐标 (RTK 基准系, 经 world_origin 平移到田块
全局系) 到最近真值行的带符号垂距 —— 与观测进地图时的实际几何误差
同源 (分割噪声/姿态补偿残差/RTK 噪声全部贡献在内, 这正是锚点
融合时 var 要表达的量)。符号约定: 行切线方向的左法向为正。

场景覆盖 (多样性优先): 直线基础/初始偏差/曲率族 (gentle~sharp,
sine, clothoid)/痛点注入 (weak_seedling 主战场, row_gap,
occlusion_burst, noisy_field 离群样本)/传感器全开/缺行/覆膜/行距。

用法 (工作区根目录):
  .venv/Scripts/python.exe -X utf8 -m nav_simulator.dl.c3_dump
  # 可选: --seeds 1,2 --frames 300 --out datasets/c3_bands_v1.npz

输出: npz {features (N,14) f32, err (N,) f32, zv/scen_id/frame,
          score} + 同名 _meta.json (特征名/场景名表/统计)
"""

import argparse
import json
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

# 场景清单: 覆盖特征空间各维度 (距离/曲率/置信度/扰动/行距)
SCENARIOS = [
    "straight_3row", "straight_offset", "straight_angle",
    "curved_gentle", "curved_medium", "curved_sharp", "sine_wave",
    "clothoid",
    "weak_seedling", "row_gap", "occlusion_burst", "noisy_field",
    "sensor_full_stack", "row_missing_center",
    "mulch_rows", "wide_spacing", "narrow_spacing",
]

FEAT_NAMES = [
    "z_norm",        # 0  前向距离 / Z_MAX_OBS
    "row_c_norm",    # 1  band 像素行中心 / crop_h
    "ncols_norm",    # 2  band 命中列数 / crop_w
    "span_norm",     # 3  条带像素宽度 / crop_w
    "score",         # 4  mask 分割置信度
    "roll_deg",      # 5  应用 roll 补偿 (度)
    "pitch_deg",     # 6  应用 pitch 补偿 (度)
    "heading_sigma", # 7  航向 σ (rad)
    "jump_m",        # 8  相邻 band 横向跳变 (m)
    "is_near_end",   # 9  近端 band 指示
    "is_far_end",    # 10 远端 band 指示
    "n_bands",       # 11 本 mask band 总数 / 100
    "abs_xv_norm",   # 12 横向位置幅值 / 3m
    "rtk_fixed",     # 13 RTK 固定解指示
]


# ── 真值标签: 点到最近行的带符号垂距 ─────────────────────────

class RowTruthField:
    """真值行折线的向量化最近邻查询 (一次性建索引, 批量查询)"""

    def __init__(self, field_model):
        pts_all, rid_all, tan_all = [], [], []
        for rid, row in enumerate(field_model.rows):
            if not row.active or len(row.points) < 2:
                continue
            p = np.asarray(row.points, dtype=np.float64)   # (M, 2)
            # 切线: 中心差分 (端点单侧), 归一化
            t = np.empty_like(p)
            t[1:-1] = p[2:] - p[:-2]
            t[0] = p[1] - p[0]
            t[-1] = p[-1] - p[-2]
            t /= np.maximum(1e-9, np.linalg.norm(t, axis=1,
                                                 keepdims=True))
            pts_all.append(p)
            rid_all.append(np.full(len(p), rid, dtype=np.int32))
            tan_all.append(t)
        self.pts = np.concatenate(pts_all)      # (P, 2)
        self.rid = np.concatenate(rid_all)      # (P,)
        self.tan = np.concatenate(tan_all)      # (P, 2)
        # 左法向 (切线逆时针 90°): err>0 = 观测在行左侧
        self.nrm = np.stack([-self.tan[:, 1], self.tan[:, 0]], axis=1)

    def signed_dist_batch(self, gx, gy, chunk=4096):
        """批量带符号垂距。Returns: (err (N,), row_idx (N,))

        先全局最近点 (0.5m 折线步长, 量化误差对 cm 级标签可忽略),
        再以该点处行切线的左法向投影定符号。
        """
        n = len(gx)
        err = np.empty(n, dtype=np.float64)
        ridx = np.empty(n, dtype=np.int32)
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            q = np.stack([gx[i0:i1], gy[i0:i1]], axis=1)   # (B, 2)
            # (B, P) 距离平方 → 每观测最近点
            d2 = ((q[:, None, :] - self.pts[None, :, :]) ** 2).sum(-1)
            j = np.argmin(d2, axis=1)
            delta = q - self.pts[j]                          # (B, 2)
            # 带符号: 左法向投影 (垂距的符号化; 切向分量天然被丢弃)
            err[i0:i1] = (delta * self.nrm[j]).sum(-1)
            ridx[i0:i1] = self.rid[j]
        return err, ridx


# ── 单场景 dump ──────────────────────────────────────────────

def dump_scenario(scenario_name: str, seed: int,
                  frames_override=None):
    """跑一次 headless 仿真并收集 band 样本。

    Returns: dict of arrays {features, err, zv, frame, score,
    row_idx, scen_id} 或 None (无有效样本 / origin 缺失)。
    """
    info = get_scenario(scenario_name)
    field_config, sim_config = info["factory"]()
    sim_config.sensor_seed = seed
    sim_config.sensor_mode = 2
    sim_config.nav_backend = "rowmap"
    if frames_override:
        sim_config.max_frames = frames_override
    if sim_config.max_frames == 0:
        sim_config.max_frames = 450
    # 扰动序列 seed 随外部种子变化 (弱苗抖动/断垄随机化)
    if sim_config.perturbation is not None:
        sim_config.perturbation.seed = 1000 + seed

    sim = NavSimulator(field_config, sim_config)
    sim._vn.enable_band_dump()
    sim.run(num_frames=sim_config.max_frames)
    recs, meta = sim._vn.collect_band_dump()
    origin = meta.get("world_origin")
    sim = None
    if not recs:
        print(f"    (no band records)")
        return None
    if origin is None:
        # RTK 未激活 → 观测世界坐标无法换算到田块系 (理论上
        # rowmap 后端必有仿真 RTK 兜底, 此为防御分支)
        print(f"    (world_origin missing, skip labels)")
        return None

    ox, oy = origin
    feats = np.asarray([r["feat"] for r in recs], dtype=np.float64)
    wx = np.asarray([r["wx"] for r in recs], dtype=np.float64)
    wy = np.asarray([r["wy"] for r in recs], dtype=np.float64)
    # RTK 基准系 → 田块全局系 (两系轴向同为 ENU, 纯平移);
    # FieldModel 从 field_config 重建 (几何只读, 与 sim 内实例等价)
    truth = RowTruthField(FieldModel(field_config))
    err, row_idx = truth.signed_dist_batch(wx + ox, wy + oy)

    return {
        "features": feats.astype(np.float32),
        "err": err.astype(np.float32),
        "zv": np.asarray([r["zv"] for r in recs], dtype=np.float32),
        "frame": np.asarray([r["frame"] for r in recs], dtype=np.int32),
        "score": np.asarray([r["score"] for r in recs], dtype=np.float32),
        "row_idx": row_idx,
        "n": len(recs),
    }


# ── 主流程 ───────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="C3 band data factory")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(
        _WORKSPACE_DIR, "datasets", "c3_bands_v1.npz"))
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    all_feats, all_err, all_zv = [], [], []
    all_frame, all_score, all_scen = [], [], []
    scen_table = []          # scen_id → name
    t_start = time.time()
    total_runs = len(scenarios) * len(seeds)
    done = 0

    for scenario in scenarios:
        scen_id = len(scen_table)
        scen_table.append(scenario)
        for seed in seeds:
            done += 1
            print(f"[{done:3d}/{total_runs}] {scenario:20s} "
                  f"seed={seed} ... ", end="", flush=True)
            t0 = time.time()
            data = dump_scenario(scenario, seed, args.frames)
            if data is None:
                continue
            all_feats.append(data["features"])
            all_err.append(data["err"])
            all_zv.append(data["zv"])
            all_frame.append(data["frame"])
            all_score.append(data["score"])
            all_scen.append(np.full(data["n"], scen_id, dtype=np.int16))
            print(f"{data['n']:6d} bands in {time.time()-t0:.1f}s "
                  f"(err rms={np.sqrt(np.mean(data['err']**2)):.4f}m)",
                  flush=True)

    if not all_feats:
        print("ERROR: no samples collected")
        sys.exit(1)

    features = np.concatenate(all_feats)
    err = np.concatenate(all_err)
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=features,
        err=err,
        zv=np.concatenate(all_zv),
        frame=np.concatenate(all_frame),
        score=np.concatenate(all_score),
        scen_id=np.concatenate(all_scen),
    )
    meta = {
        "feat_names": FEAT_NAMES,
        "scenarios": scen_table,
        "n_samples": int(len(err)),
        "err_stats": {
            "mean": float(err.mean()), "std": float(err.std()),
            "rms": float(np.sqrt(np.mean(err ** 2))),
            "abs_p95": float(np.percentile(np.abs(err), 95)),
        },
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nsaved: {args.out}")
    print(f"       {meta_path}")
    print(f"total: {len(err)} samples, "
          f"err rms={np.sqrt(np.mean(err**2)):.4f}m, "
          f"|err| p95={meta['err_stats']['abs_p95']:.4f}m")
    print(f"wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
