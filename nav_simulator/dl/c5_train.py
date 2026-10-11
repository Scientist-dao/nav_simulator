"""
c5_train.py - C5 行尾/地头感知网络 训练 + 工作点评估

模型: MLP 22 → 32 → 16 → 1 (p_end logit), ~1K 参数, CPU 微秒级。
损失: BCE with pos_weight (行尾临近帧远少于行中帧, 加权平衡)。

评估 (部署视角 —— 误掉头代价 >> 漏预警, 后者有里程条件兜底):
  - 约束负样本 FP rate ≤ MAX_FP_RATE 的前提下最大化召回,
    扫描验证集得部署工作点 (threshold, recall, fp_rate);
  - 分场景混淆矩阵: row_gap/occlusion 的断垄/穿越段必须压住
    (单帧几何上"前方无观测", 靠地图历史形态区分 —— C5 的卖点);
  - 按种子划分 train/val (同场景不同传感器噪声, 检噪声泛化)。

用法 (工作区根目录, 先跑 c5_dump.py 产数据):
  .venv/Scripts/python.exe -X utf8 -m nav_simulator.dl.c5_train

输出:
  models/c5_headland.pt           (state dict + 归一化统计)
  models/c5_headland_weights.npz  (部署裸权重)
  results/c5/metrics.json         (工作点 + 分场景混淆矩阵)
"""

import argparse
import json
import os
import sys

import numpy as np

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_SIM_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

import torch
import torch.nn as nn

MAX_FP_RATE = 0.005     # 部署约束: 负样本误报率 ≤ 0.5%/帧


class HeadlandMLP(nn.Module):
    """22 维地图历史特征 → 行终止概率 logit"""

    def __init__(self, in_dim: int = 22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.SiLU(),
            nn.Linear(32, 16), nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser(description="C5 headland net training")
    ap.add_argument("--data", default=os.path.join(
        _WORKSPACE_DIR, "datasets", "c5_headland_v1.npz"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-seed", type=int, default=2,
                    help="数据工厂侧做验证的 sensor seed")
    ap.add_argument("--out-dir", default=os.path.join(
        _WORKSPACE_DIR, "results", "c5"))
    ap.add_argument("--model-dir", default=os.path.join(
        _WORKSPACE_DIR, "models"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    data = np.load(args.data)
    meta_path = os.path.splitext(args.data)[0] + "_meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    feats = data["features"].astype(np.float32)
    label = data["label"].astype(np.float32)
    scen = data["scen_id"]
    dump_seed = data["dump_seed"]
    n = len(label)
    print(f"loaded {n} samples "
          f"(pos={int(label.sum())} = {label.mean():.2%}) "
          f"from {args.data}")

    # ── 按数据工厂种子划分: val-seed 的全部样本做验证 ──
    # (同场景不同传感器噪声 → 检噪声泛化; 场景两边均有)
    val_mask = dump_seed == args.val_seed
    val_idx = np.flatnonzero(val_mask)
    tr_idx = np.flatnonzero(~val_mask)
    if len(val_idx) == 0 or len(tr_idx) == 0:
        # 该 seed 不在数据里: 回退随机 90/10
        rng = np.random.RandomState(args.seed)
        idx = rng.permutation(n)
        n_val = max(1, int(n * 0.1))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        print(f"WARNING: val seed {args.val_seed} not in data, "
              f"fallback to random split")

    mu = feats[tr_idx].mean(axis=0)
    sd = feats[tr_idx].std(axis=0) + 1e-6
    feats_n = (feats - mu) / sd

    xt = torch.from_numpy(feats_n[tr_idx])
    lt = torch.from_numpy(label[tr_idx])
    xv = torch.from_numpy(feats_n[val_idx])
    lv = torch.from_numpy(label[val_idx])

    n_pos_tr = float(lt.sum())
    n_neg_tr = float(len(lt) - n_pos_tr)
    pos_weight = torch.tensor(min(50.0, max(1.0, n_neg_tr / max(1.0, n_pos_tr))))
    print(f"train {len(lt)} (pos {int(n_pos_tr)}) / val {len(lv)} "
          f"(pos {int(lv.sum())}), pos_weight={float(pos_weight):.1f}")

    model = HeadlandMLP(feats.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state = float("inf"), None
    n_tr = len(lt)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_tr)
        tot, nb = 0.0, 0
        for i0 in range(0, n_tr, args.batch):
            b = perm[i0:i0 + args.batch]
            logit = model(xt[b])
            loss = crit(logit, lt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(crit(model(xv), lv))
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in
                          model.state_dict().items()}
            marker = " *"
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d}: train={tot/nb:.4f} "
                  f"val={val_loss:.4f}{marker}")

    model.load_state_dict(best_state)

    # ── 工作点扫描: FP ≤ MAX_FP_RATE 约束下最大召回 ──
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(xv)).numpy()
    lv_np = label[val_idx]
    ths = np.linspace(0.05, 0.95, 91)
    best = None
    for th in ths:
        pred = pv >= th
        fp = float((pred & (lv_np < 0.5)).sum()
                   / max(1, (lv_np < 0.5).sum()))
        rec = float((pred & (lv_np >= 0.5)).sum()
                    / max(1, (lv_np >= 0.5).sum()))
        if fp <= MAX_FP_RATE:
            if best is None or rec > best[1]:
                best = (float(th), rec, fp)
    if best is None:
        # 约束不可满足: 取 FP 最小的阈值 (报告并提示需更多负样本)
        fp_rates = []
        for th in ths:
            pred = pv >= th
            fp = float((pred & (lv_np < 0.5)).sum()
                       / max(1, (lv_np < 0.5).sum()))
            fp_rates.append((fp, float(th)))
        fp_rates.sort()
        th = fp_rates[0][1]
        pred = pv >= th
        best = (th,
                float((pred & (lv_np >= 0.5)).sum()
                      / max(1, (lv_np >= 0.5).sum())),
                fp_rates[0][0])
        print("WARNING: MAX_FP_RATE not satisfiable, "
              "reporting min-FP point")
    th, recall, fp_rate = best
    print(f"\noperating point: threshold={th:.2f} "
          f"recall={recall:.2%} fp={fp_rate:.3%}")

    # ── 分场景混淆矩阵 (验证集, 工作点阈值) ──
    scen_val = scen[val_idx]
    scen_names = meta.get("scenarios", [])
    per_scen = {}
    pred = pv >= th
    for sid in np.unique(scen_val):
        m = scen_val == sid
        name = scen_names[sid] if sid < len(scen_names) else str(sid)
        pos = int((lv_np[m] >= 0.5).sum())
        neg = int((lv_np[m] < 0.5).sum())
        tp = int((pred[m] & (lv_np[m] >= 0.5)).sum())
        fp_n = int((pred[m] & (lv_np[m] < 0.5)).sum())
        per_scen[name] = {
            "pos": pos, "neg": neg, "tp": tp, "fp": fp_n,
            "recall": (tp / pos) if pos else None,
            "fp_rate": (fp_n / neg) if neg else None,
        }
        print(f"  {name:20s} pos={pos:4d} tp={tp:4d} "
              f"({(tp/pos if pos else 0):5.1%})  "
              f"neg={neg:4d} fp={fp_n:3d} "
              f"({(fp_n/neg if neg else 0):6.2%})")

    # ── 落盘 ──
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    metrics = {
        "n_train": int(len(lt)), "n_val": int(len(lv)),
        "best_val_loss": best_val,
        "operating_point": {"threshold": th, "recall": recall,
                            "fp_rate": fp_rate},
        "per_scenario": per_scen,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    torch.save({
        "state_dict": model.state_dict(),
        "feat_mean": mu, "feat_std": sd,
        "in_dim": int(feats.shape[1]),
        "threshold": th,
    }, os.path.join(args.model_dir, "c5_headland.pt"))

    # 变量名用 state —— sd 是标准化参数 feat_std, 不能覆盖
    state = model.state_dict()
    np.savez(os.path.join(args.model_dir, "c5_headland_weights.npz"),
             W1=state["net.0.weight"].numpy(), b1=state["net.0.bias"].numpy(),
             W2=state["net.2.weight"].numpy(), b2=state["net.2.bias"].numpy(),
             W3=state["net.4.weight"].numpy(), b3=state["net.4.bias"].numpy(),
             feat_mean=mu, feat_std=sd, threshold=np.float64(th))
    print(f"\nsaved: {args.model_dir}\\c5_headland.pt / "
          f"c5_headland_weights.npz")
    print(f"saved: {args.out_dir}\\metrics.json")


if __name__ == "__main__":
    main()
