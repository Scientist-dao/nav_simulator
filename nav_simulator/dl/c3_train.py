"""
c3_train.py - C3 异方差观测质量网络 训练 + 可靠性图评估

模型: MLP 14 → 32 → 16 → 2 输出 [δx 系统纠偏, log σ²] (~1K 参数,
CPU 前向微秒级)。损失 = 异方差高斯 NLL:

    L = (err − δx)² / (2σ²) + ½·log σ²

σ 报小 (吹牛) 第一项指数重罚, σ 报大 (过谦) 第二项缓罚 —— 网络唯一
出路是把 σ 学到与真实残差一致 (校准)。δx 学条件均值 (系统偏差,
如弱苗抖动/姿态残差), σ² 学条件方差。

评估: 可靠性图 (预测 σ 分桶 vs 实际残差 RMS, 完美校准=对角线),
与手写方差公式 (nav_core 三项式) 同图对比 —— 论文核心证据。

用法 (工作区根目录, 先跑 c3_dump.py 产数据):
  .venv/Scripts/python.exe -X utf8 -m nav_simulator.dl.c3_train
  # 可选: --data datasets/c3_bands_v1.npz --epochs 60

输出:
  models/c3_quality.pt           (state dict + 归一化统计, 恢复用)
  models/c3_quality_weights.npz  (部署用裸权重: C++ 前向/Python 接入)
  results/c3/reliability.png     (可靠性图: C3 vs 手写公式)
  results/c3/calibration.csv     (分桶数据)
  results/c3/metrics.json        (汇总指标)
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

# 与 nav_core.py 手写方差公式同参数 (可靠性图基线)
K_PIX_SIGMA = 0.005
SIGMA_RTK = 0.02
# 特征索引 (与 c3_dump.FEAT_NAMES 对齐)
I_ZV, I_HSIG = 0, 7
Z_SCALE = 20.0    # 特征 0 = zv / 20


class QualityMLP(nn.Module):
    """14 维 band 特征 → (δx, log σ²)"""

    def __init__(self, in_dim: int = 14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.SiLU(),
            nn.Linear(32, 16), nn.SiLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def hetero_nll(pred: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
    """异方差高斯 NLL (均值见 docstring; err 带符号真值误差)"""
    dx = pred[:, 0]
    log_var = pred[:, 1]
    var = torch.exp(log_var).clamp(1e-8, 1e4)
    return ((err - dx) ** 2 / (2.0 * var)
            + 0.5 * log_var).mean()


def handcrafted_sigma(features: np.ndarray) -> np.ndarray:
    """手写方差公式 (nav_core 三项式) 的 σ —— 可靠性图基线

    var = (K_PIX_SIGMA·zv)² + heading_var·zv² + SIGMA_RTK²
    (特征里 zv 和 hsig 都是归一化过的, 先还原)
    """
    zv = features[:, I_ZV] * Z_SCALE
    hsig = features[:, I_HSIG]
    var = (K_PIX_SIGMA * zv) ** 2 + (hsig ** 2) * zv ** 2 \
        + SIGMA_RTK ** 2
    return np.sqrt(var)


def reliability_bins(sigma_pred: np.ndarray, resid: np.ndarray,
                     n_bins: int = 10):
    """σ 分桶校准: 每桶 (σ_pred RMS, 实际残差 RMS)"""
    edges = np.quantile(sigma_pred, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    rows = []
    for i in range(n_bins):
        m = (sigma_pred >= edges[i]) & (sigma_pred < edges[i + 1])
        if m.sum() < 10:
            continue
        rows.append((float(np.sqrt(np.mean(sigma_pred[m] ** 2))),
                     float(np.sqrt(np.mean(resid[m] ** 2))),
                     int(m.sum())))
    return rows


def main():
    ap = argparse.ArgumentParser(description="C3 quality net training")
    ap.add_argument("--data", default=os.path.join(
        _WORKSPACE_DIR, "datasets", "c3_bands_v1.npz"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(
        _WORKSPACE_DIR, "results", "c3"))
    ap.add_argument("--model-dir", default=os.path.join(
        _WORKSPACE_DIR, "models"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # ── 数据 ──
    data = np.load(args.data)
    feats = data["features"].astype(np.float32)   # (N, 14)
    err = data["err"].astype(np.float32)          # (N,) 带符号
    n = len(err)
    print(f"loaded {n} samples from {args.data}")
    print(f"err: mean={err.mean():+.4f} std={err.std():.4f} "
          f"rms={np.sqrt(np.mean(err**2)):.4f} m")

    idx = rng.permutation(n)
    n_val = max(1, int(n * 0.1))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    # 输入标准化 (训练集统计)
    mu = feats[tr_idx].mean(axis=0)
    sd = feats[tr_idx].std(axis=0) + 1e-6
    feats_n = (feats - mu) / sd

    dev = "cpu"
    xt = torch.from_numpy(feats_n[tr_idx])
    et = torch.from_numpy(err[tr_idx])
    xv = torch.from_numpy(feats_n[val_idx])
    ev = torch.from_numpy(err[val_idx])
    print(f"train {len(et)} / val {len(ev)} (device={dev})")

    # ── 模型与初始化 ──
    model = QualityMLP(feats.shape[1]).to(dev)
    # log σ² 输出头 bias 初始化为先验 log(err.var()):
    # 一开始 σ ≈ 全局 RMS, 避免从 1 出发的冷启动大梯度
    with torch.no_grad():
        model.net[-1].bias[1] = float(np.log(max(1e-8, err.var())))
        model.net[-1].bias[0] = 0.0

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)

    # ── 训练 (验证 NLL 早停) ──
    best_val, best_state = float("inf"), None
    n_tr = len(et)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_tr)
        tot, nb = 0.0, 0
        for i0 in range(0, n_tr, args.batch):
            b = perm[i0:i0 + args.batch]
            pred = model(xt[b])
            loss = hetero_nll(pred, et[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            val_nll = float(hetero_nll(model(xv), ev))
        marker = ""
        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.clone() for k, v in
                          model.state_dict().items()}
            marker = " *"
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d}: train_nll={tot/nb:.4f} "
                  f"val_nll={val_nll:.4f}{marker}")

    model.load_state_dict(best_state)
    print(f"best val_nll={best_val:.4f}")

    # ── 评估: 纠偏效果 + 可靠性图 ──
    model.eval()
    with torch.no_grad():
        pv = model(xv).numpy()
    dx_val, logvar_val = pv[:, 0], pv[:, 1]
    sigma_val = np.exp(0.5 * logvar_val)
    resid = err[val_idx] - dx_val

    rms_raw = float(np.sqrt(np.mean(err[val_idx] ** 2)))
    rms_corr = float(np.sqrt(np.mean(resid ** 2)))
    print(f"\nresidual RMS: raw={rms_raw:.4f} → "
          f"δx-corrected={rms_corr:.4f} m "
          f"({(1-rms_corr/rms_raw)*100:.1f}% reduction)")

    # 手写公式基线 (同一验证集)
    sigma_hand = handcrafted_sigma(feats[val_idx])
    rows_c3 = reliability_bins(sigma_val, resid)
    rows_hand = reliability_bins(sigma_hand, err[val_idx])

    # 校准误差汇总 (各桶 |σ_pred − σ_actual| 加权平均, 占实际 RMS 比例)
    def miscalib(rows):
        if not rows:
            return float("nan")
        w = np.array([r[2] for r in rows], dtype=float)
        e = np.array([abs(r[0] - r[1]) for r in rows])
        return float((w * e).sum() / w.sum())

    mc_c3 = miscalib(rows_c3)
    mc_hand = miscalib(rows_hand)
    print(f"calibration err: C3={mc_c3:.4f} vs hand={mc_hand:.4f} m "
          f"(lower=better)")

    # ── 落盘 ──
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # 可靠性图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
        lim = max(0.15, float(np.percentile(np.r_[sigma_val,
                                                  sigma_hand], 99)) * 1.1)
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
        ax.plot([r[0] for r in rows_hand], [r[1] for r in rows_hand],
                "s-", color="#c44", label="handcrafted formula")
        ax.plot([r[0] for r in rows_c3], [r[1] for r in rows_c3],
                "o-", color="#285", label="C3 MLP (heteroscedastic)")
        ax.set_xlabel("predicted σ (m)")
        ax.set_ylabel("actual residual RMS (m)")
        ax.set_title("C3 observation quality calibration")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "reliability.png"))
        plt.close(fig)
        print(f"saved: {args.out_dir}\\reliability.png")
    except ImportError:
        print("matplotlib missing, skip reliability figure")

    with open(os.path.join(args.out_dir, "calibration.csv"),
              "w", encoding="utf-8") as f:
        f.write("model,sigma_pred_rms,resid_rms,n\n")
        for r in rows_hand:
            f.write(f"handcrafted,{r[0]:.5f},{r[1]:.5f},{r[2]}\n")
        for r in rows_c3:
            f.write(f"c3_mlp,{r[0]:.5f},{r[1]:.5f},{r[2]}\n")

    metrics = {
        "n_train": int(n_tr), "n_val": int(n_val),
        "best_val_nll": best_val,
        "residual_rms_raw": rms_raw,
        "residual_rms_corrected": rms_corr,
        "bias_correction_gain_pct": (1 - rms_corr / rms_raw) * 100,
        "calibration_err_c3": mc_c3,
        "calibration_err_handcrafted": mc_hand,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 训练态 (恢复用)
    torch.save({
        "state_dict": model.state_dict(),
        "feat_mean": mu, "feat_std": sd,
        "in_dim": int(feats.shape[1]),
    }, os.path.join(args.model_dir, "c3_quality.pt"))

    # 部署态裸权重 (Python 参考实现接入 / C++ 前向共用)
    # 变量名用 state —— sd 是标准化参数 feat_std, 不能覆盖
    state = model.state_dict()
    np.savez(os.path.join(args.model_dir, "c3_quality_weights.npz"),
             W1=state["net.0.weight"].numpy(), b1=state["net.0.bias"].numpy(),
             W2=state["net.2.weight"].numpy(), b2=state["net.2.bias"].numpy(),
             W3=state["net.4.weight"].numpy(), b3=state["net.4.bias"].numpy(),
             feat_mean=mu, feat_std=sd)
    print(f"saved: {args.model_dir}\\c3_quality.pt / "
          f"c3_quality_weights.npz")
    print(f"saved: {args.out_dir}\\metrics.json / calibration.csv")


if __name__ == "__main__":
    main()
