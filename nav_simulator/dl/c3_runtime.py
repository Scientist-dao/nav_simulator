"""
c3_runtime.py - C3 异方差观测质量网络的 numpy 推理前向 (部署态)

从 c3_train.py 落盘的裸权重 (models/c3_quality_weights.npz) 构造
callable, 供 Python 参考实现 nav_core.set_quality_net() 注入 —— 无
torch 依赖, 逐层点积 + SiLU, 与 C++ 部署前向同一套算式 (后续
RKNN/C++ 侧照抄权重即可)。

用法:
  from nav_simulator.dl.c3_runtime import load_quality_net
  vn.set_quality_net(load_quality_net("models/c3_quality_weights.npz"))
  # 消融 A/B: vn.set_quality_net(None) 恢复手写三项式 var

自测 (随机权重 + torch 前向一致性, 验证前向算式无笔误):
  .venv/Scripts/python.exe -X utf8 -m nav_simulator.dl.c3_runtime --selftest
"""

import argparse
import os
import sys

import numpy as np

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_SIM_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

# 与训练侧 clamp 一致 (hetero_nll 里 var clamp 1e-8..1e4)
VAR_MIN, VAR_MAX = 1e-8, 1e4
DEFAULT_WEIGHTS = os.path.join(
    _WORKSPACE_DIR, "models", "c3_quality_weights.npz")


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class QualityNetRuntime:
    """MLP 14 → 32 → 16 → 2 前向: feats → (δx, σ²)

    callable 接口与 nav_core.set_quality_net 的约定一致:
        __call__(feats (N,14) float64 ndarray) → (δx (N,), σ² (N,))
    (float64 输入自动转 float32 计算, 输出转 float64 —— 参考实现侧
    世界坐标是 double, 不让网络内部精度决定输出 dtype)
    """

    def __init__(self, W1, b1, W2, b2, W3, b3,
                 feat_mean, feat_std):
        self.W1 = np.ascontiguousarray(W1, dtype=np.float32)
        self.b1 = np.ascontiguousarray(b1, dtype=np.float32)
        self.W2 = np.ascontiguousarray(W2, dtype=np.float32)
        self.b2 = np.ascontiguousarray(b2, dtype=np.float32)
        self.W3 = np.ascontiguousarray(W3, dtype=np.float32)
        self.b3 = np.ascontiguousarray(b3, dtype=np.float32)
        self.feat_mean = np.ascontiguousarray(feat_mean,
                                              dtype=np.float32)
        self.feat_std = np.ascontiguousarray(feat_std,
                                             dtype=np.float32)

    def __call__(self, feats: np.ndarray):
        x = np.atleast_2d(np.asarray(feats, dtype=np.float32))
        x = (x - self.feat_mean) / self.feat_std
        h = _silu(x @ self.W1.T + self.b1)
        h = _silu(h @ self.W2.T + self.b2)
        out = h @ self.W3.T + self.b3          # (N, 2)
        dx = out[:, 0].astype(np.float64)
        var = np.exp(out[:, 1]).clip(VAR_MIN, VAR_MAX)
        return dx, var.astype(np.float64)


def load_quality_net(npz_path: str = DEFAULT_WEIGHTS):
    """权重文件 → callable(feats) → (δx, σ²); 文件缺失返回 None"""
    if not os.path.isfile(npz_path):
        return None
    z = np.load(npz_path)
    return QualityNetRuntime(
        z["W1"], z["b1"], z["W2"], z["b2"], z["W3"], z["b3"],
        z["feat_mean"], z["feat_std"])


# ── 自测: 与 torch 前向一致性 ─────────────────────────────

def _selftest(weights_path: str):
    import torch
    import torch.nn as nn

    rng = np.random.RandomState(0)
    if os.path.isfile(weights_path):
        print(f"selftest vs real weights: {weights_path}")
        z = np.load(weights_path)
        in_dim = z["W1"].shape[1]
    else:
        print("no real weights yet, selftest with random weights")
        in_dim = 14
        z = {
            "W1": rng.randn(32, in_dim).astype(np.float32) * 0.3,
            "b1": rng.randn(32).astype(np.float32) * 0.1,
            "W2": rng.randn(16, 32).astype(np.float32) * 0.3,
            "b2": rng.randn(16).astype(np.float32) * 0.1,
            "W3": rng.randn(2, 16).astype(np.float32) * 0.3,
            "b3": rng.randn(2).astype(np.float32) * 0.1,
            "feat_mean": rng.randn(in_dim).astype(np.float32) * 0.2,
            "feat_std": (rng.rand(in_dim).astype(np.float32) + 0.5),
        }

    rt = QualityNetRuntime(
        z["W1"], z["b1"], z["W2"], z["b2"], z["W3"], z["b3"],
        z["feat_mean"], z["feat_std"])

    net = nn.Sequential()
    net.add_module("0", nn.Linear(in_dim, 32))
    net.add_module("1", nn.SiLU())
    net.add_module("2", nn.Linear(32, 16))
    net.add_module("3", nn.SiLU())
    net.add_module("4", nn.Linear(16, 2))
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(
            np.asarray(z["W1"], dtype=np.float32)))
        net[0].bias.copy_(torch.from_numpy(
            np.asarray(z["b1"], dtype=np.float32)))
        net[2].weight.copy_(torch.from_numpy(
            np.asarray(z["W2"], dtype=np.float32)))
        net[2].bias.copy_(torch.from_numpy(
            np.asarray(z["b2"], dtype=np.float32)))
        net[4].weight.copy_(torch.from_numpy(
            np.asarray(z["W3"], dtype=np.float32)))
        net[4].bias.copy_(torch.from_numpy(
            np.asarray(z["b3"], dtype=np.float32)))

    feats = rng.randn(512, in_dim).astype(np.float64) * 2.0
    dx_rt, var_rt = rt(feats)
    # torch 侧喂标准化后的输入 (net 不含标准化层, runtime 内置)
    feats_n = ((feats.astype(np.float32)
                - np.asarray(z["feat_mean"], np.float32))
               / np.asarray(z["feat_std"], np.float32))
    with torch.no_grad():
        out_t = net(torch.from_numpy(feats_n))
    dx_t = out_t[:, 0].numpy().astype(np.float64)
    var_t = np.exp(out_t[:, 1].numpy().astype(np.float64))

    d_dx = float(np.abs(dx_rt - dx_t).max())
    d_var = float(np.abs(var_rt - var_t).max() / var_t.max())
    print(f"max |Δδx| = {d_dx:.3e}  (rel Δσ² = {d_var:.3e})")
    ok = d_dx < 1e-5 and d_var < 1e-5
    print("PASS: numpy runtime ↔ torch forward consistent"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C3 runtime selftest")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest(args.weights))
    rt = load_quality_net(args.weights)
    print("loaded" if rt is not None
          else f"weights not found: {args.weights}")
