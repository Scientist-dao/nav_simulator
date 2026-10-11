#!/usr/bin/env python3
"""
run_benchmark.py - A/B 跑批评测脚本 (P0 仿真基建)

对指定场景 × 传感器种子 × 融合模式组合批量运行 headless 仿真,
输出视觉导航V2开发文档 §4.1 指标体系的汇总报告 (JSON + 控制台)。

用法 (工作区根目录):
  # 基线痛点数据 (P0 交付物: 基线在 4 个痛点场景的丑数据)
  .venv/Scripts/python.exe -m nav_simulator.run_benchmark

  # 自定义组合
  .venv/Scripts/python.exe -m nav_simulator.run_benchmark \
      --scenarios deadlock_center,occlusion_burst --seeds 1,2,3 --modes 0,1,2

  # RowMap 参考实现 A/B (同台同指标对比)
  .venv/Scripts/python.exe -m nav_simulator.run_benchmark \
      --backend rowmap

输出: results/{p0_baseline|p1_rowmap}/benchmark_*.json + summary.csv
"""

import argparse
import json
import os
import sys
import time

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.dirname(_SIM_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from nav_simulator.core.simulator import NavSimulator
from nav_simulator.core.metrics import evaluate, AgronomyParams
from nav_simulator.scenarios.presets import get_scenario, list_scenarios

# P0 默认跑批组合: 4 个痛点场景 (痛点注入) + sensor_full_stack (生产对照)
DEFAULT_SCENARIOS = [
    "deadlock_center", "occlusion_burst", "row_gap", "weak_seedling",
    "sensor_full_stack",
]


def parse_args():
    p = argparse.ArgumentParser(description="A/B benchmark runner")
    p.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS),
                   help="comma-separated scenario names")
    p.add_argument("--seeds", default="1,2,3",
                   help="comma-separated sensor seeds")
    p.add_argument("--modes", default="2",
                   help="comma-separated sensor modes (0/1/2)")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="output dir (default results/p0_baseline, "
                        "p1_rowmap or p2_rowmap_cpp)")
    p.add_argument("--backend", default="baseline",
                   choices=["baseline", "rowmap", "rowmap_cpp"],
                   help="navigation backend: baseline (C++), "
                        "rowmap (Python reference) or rowmap_cpp (C++)")
    p.add_argument("--qnet", nargs="?", const="", default=None,
                   metavar="PATH",
                   help="C3 quality net weights npz (C3c A/B; 仅 "
                        "rowmap 后端; 省略路径默认 "
                        "models/c3_quality_weights.npz)")
    p.add_argument("--hnet", nargs="?", const="", default=None,
                   metavar="PATH",
                   help="C5 headland net weights npz (C5b A/B; 仅 "
                        "rowmap 后端; 省略路径默认 "
                        "models/c5_headland_weights.npz)")
    p.add_argument("--bev", nargs="?", const="", default=None,
                   metavar="PATH",
                   help="C4 BEV ConvGRU weights npz (C4c A/B; 仅 "
                        "rowmap 后端; 省略路径默认 "
                        "models/c4_bev_weights.npz)")
    p.add_argument("--frames", type=int, default=None,
                   help="override max frames per run")
    return p.parse_args()


def run_one(scenario_name: str, seed: int, mode: int, frames_override=None,
            backend: str = "baseline", qnet_path: str = None,
            hnet_path: str = None, bev_path: str = None):
    """跑一次 headless 仿真并计算指标

    Returns: (run_record, wall_time_s)  run_record 含全部指标;
    qnet/hnet/bev 路径非空时 (仅 rowmap 后端) 分别注入 C3 观测
    质量网络 (δx+σ²)、C5 行尾感知网络 (掉头触发门调制) 与 C4
    BEV 时序证据 (失明帧伪观测)。
    """
    scenario_info = get_scenario(scenario_name)
    field_config, sim_config = scenario_info["factory"]()

    # 覆盖: 种子 / 模式 / 帧数 / 导航后端 (其余场景设定保持原样)
    sim_config.sensor_seed = seed
    sim_config.sensor_mode = mode
    sim_config.nav_backend = backend
    if frames_override:
        sim_config.max_frames = frames_override
    if sim_config.max_frames == 0:
        sim_config.max_frames = 450   # 无限制场景给个安全上限

    t0 = time.time()
    sim = NavSimulator(field_config, sim_config)
    backend_label = backend
    if qnet_path and backend == "rowmap":
        from nav_simulator.dl.c3_runtime import load_quality_net
        rt = load_quality_net(qnet_path)
        if rt is None:
            raise FileNotFoundError(
                f"C3 qnet weights not found: {qnet_path}")
        sim._vn.set_quality_net(rt)
        backend_label = "rowmap+qnet"
    if hnet_path and backend == "rowmap":
        from nav_simulator.dl.c5_runtime import load_headland_net
        hn = load_headland_net(hnet_path)
        if hn is None:
            raise FileNotFoundError(
                f"C5 hnet weights not found: {hnet_path}")
        sim._vn.set_headland_net(hn)
        backend_label = (backend_label + "+hnet"
                         if "+" in backend_label else "rowmap+hnet")
    if bev_path and backend == "rowmap":
        from nav_simulator.dl.c4_runtime import load_bev_net
        bv = load_bev_net(bev_path)
        if bv is None:
            raise FileNotFoundError(
                f"C4 bev weights not found: {bev_path}")
        sim._vn.set_bev_net(bv)
        backend_label = (backend_label + "+bev"
                         if "+" in backend_label else "rowmap+bev")
    sim.run(num_frames=sim_config.max_frames)
    wall = time.time() - t0

    occlusion_windows = (sim_config.perturbation.occlusions
                         if sim_config.perturbation is not None else None)
    metrics = evaluate(sim.results, AgronomyParams(),
                       occlusion_windows=occlusion_windows)

    record = {
        "scenario": scenario_name,
        "backend": backend_label,
        "seed": seed,
        "mode": mode,
        "frames": sim_config.max_frames,
        "wall_time_s": round(wall, 1),
        **metrics,
    }
    sim = None   # 及时释放 (每 run 都持有 C++ 核心)
    return record, wall


def aggregate(records):
    """按 (backend, scenario, mode) 聚合种子均值 (标量指标)"""
    groups = {}
    for r in records:
        key = (r.get("backend", "baseline"), r["scenario"], r["mode"])
        groups.setdefault(key, []).append(r)

    agg = []
    for (backend, scenario, mode), runs in sorted(groups.items()):
        row = {"backend": backend, "scenario": scenario, "mode": mode,
               "runs": len(runs)}
        for k, v in runs[0].items():
            if isinstance(v, (int, float)) and k not in (
                    "seed", "mode", "frames", "wall_time_s"):
                vals = [r[k] for r in runs
                        if isinstance(r.get(k), (int, float))
                        and r[k] == r[k]]          # NaN 过滤
                if vals:
                    row[k] = round(sum(vals) / len(vals), 4)
        agg.append(row)
    return agg


def print_table(agg):
    """控制台汇总表 (ASCII, GBK 终端安全)"""
    # (显示名, 记录 key, 列宽)
    cols = [
        ("backend", "backend", 20),
        ("scenario", "scenario", 18), ("mode", "mode", 4),
        ("crush_rate", "crush_rate", 11),
        ("crush_excl", "crush_rate_occlusion_excl", 11),
        ("loss_rate", "line_loss_rate", 10),
        ("loss_occl", "line_loss_rate_occlusion", 10),
        ("cte_rmse", "cte_rmse_m", 9),
        ("settle_s", "settle_time_s", 9),
        ("yaw_std", "yaw_steady_std_deg", 8),
        ("front_std", "front_steady_std_m", 10),
    ]
    header = " ".join(name[:w].ljust(w) for name, _, w in cols)
    print("\n" + "=" * len(header))
    print("AGGREGATED (mean over seeds)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in agg:
        line = (f"{row.get('backend', 'baseline')[:20].ljust(20)} "
                f"{row['scenario'][:18].ljust(18)} "
                f"{str(row['mode']).ljust(4)}")
        for name, key, w in cols[2:]:
            v = row.get(key)
            if isinstance(v, float):
                s = f"{v:.4f}"
            elif v is None:
                s = "-"
            else:
                s = str(v)
            line += " " + s[:w].ljust(w)
        print(line)
    print("=" * len(header) + "\n")


def main():
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    modes = [int(m) for m in args.modes.split(",") if m.strip()]

    # 场景名校验
    known = set(list_scenarios().keys())
    for s in scenarios:
        if s not in known:
            print(f"ERROR: unknown scenario '{s}'")
            print(f"available: {', '.join(sorted(known))}")
            sys.exit(1)

    default_out = os.path.join(
        _WORKSPACE_DIR, "results",
        {"rowmap": "p1_rowmap",
         "rowmap_cpp": "p2_rowmap_cpp"}.get(args.backend, "p0_baseline"))
    qnet_path = None
    if args.qnet is not None:
        if args.backend != "rowmap":
            print("ERROR: --qnet 仅支持 --backend rowmap")
            sys.exit(1)
        qnet_path = (args.qnet or os.path.join(
            _WORKSPACE_DIR, "models", "c3_quality_weights.npz"))
        if not os.path.isfile(qnet_path):
            print(f"ERROR: qnet weights not found: {qnet_path} "
                  f"(先跑 c3_train.py)")
            sys.exit(1)
        default_out = os.path.join(_WORKSPACE_DIR, "results", "c3_ab")
    hnet_path = None
    if args.hnet is not None:
        if args.backend != "rowmap":
            print("ERROR: --hnet 仅支持 --backend rowmap")
            sys.exit(1)
        hnet_path = (args.hnet or os.path.join(
            _WORKSPACE_DIR, "models", "c5_headland_weights.npz"))
        if not os.path.isfile(hnet_path):
            print(f"ERROR: hnet weights not found: {hnet_path} "
                  f"(先跑 c5_train.py)")
            sys.exit(1)
        if args.qnet is None:
            default_out = os.path.join(_WORKSPACE_DIR, "results", "c5_ab")
    bev_path = None
    if args.bev is not None:
        if args.backend != "rowmap":
            print("ERROR: --bev 仅支持 --backend rowmap")
            sys.exit(1)
        bev_path = (args.bev or os.path.join(
            _WORKSPACE_DIR, "models", "c4_bev_weights.npz"))
        if not os.path.isfile(bev_path):
            print(f"ERROR: bev weights not found: {bev_path} "
                  f"(先跑 c4_train.py)")
            sys.exit(1)
        if args.qnet is None and args.hnet is None:
            default_out = os.path.join(_WORKSPACE_DIR, "results", "c4_ab")
    out_dir = args.out or default_out
    os.makedirs(out_dir, exist_ok=True)

    total = len(scenarios) * len(seeds) * len(modes)
    print(f"benchmark [{args.backend}]: {len(scenarios)} scenarios "
          f"x {len(seeds)} seeds x {len(modes)} modes = {total} runs")
    print(f"output: {out_dir}\n")

    records = []
    done = 0
    t_start = time.time()
    for scenario in scenarios:
        for mode in modes:
            for seed in seeds:
                done += 1
                print(f"[{done:3d}/{total}] {scenario:20s} "
                      f"mode={mode} seed={seed} ... ",
                      end="", flush=True)
                record, wall = run_one(
                    scenario, seed, mode, frames_override=args.frames,
                    backend=args.backend, qnet_path=qnet_path,
                    hnet_path=hnet_path, bev_path=bev_path)
                records.append(record)
                print(f"done in {wall:.1f}s  "
                      f"crush={record['crush_rate']:.2%}  "
                      f"loss={record['line_loss_rate']:.2%}",
                      flush=True)

    # 保存明细 + 聚合
    stamp = time.strftime("%Y%m%d_%H%M%S")
    detail_path = os.path.join(out_dir, f"benchmark_{stamp}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "created": stamp,
            "backend": args.backend,
            "scenarios": scenarios,
            "seeds": seeds,
            "modes": modes,
            "agronomy": {
                "row_spacing_m": 0.60, "wheel_track_m": 1.72,
                "tire_width_m": 0.30, "crop_row_width_m": 0.12},
            "runs": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\ndetail saved: {detail_path}")

    agg = aggregate(records)
    agg_path = os.path.join(out_dir, f"summary_{stamp}.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"summary saved: {agg_path}")

    # CSV 汇总 (论文/Excel 用)
    csv_path = os.path.join(out_dir, f"summary_{stamp}.csv")
    if agg:
        keys = [k for k in agg[0].keys()]
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in agg:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"csv saved: {csv_path}")

    print_table(agg)
    print(f"total wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
