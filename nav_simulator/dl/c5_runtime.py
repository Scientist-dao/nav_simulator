"""
c5_runtime.py - C5 行尾/地头感知网络的 numpy 推理前向 (部署态)

从 c5_train.py 落盘的裸权重 (models/c5_headland_weights.npz) 构造
callable: 22 维地图历史特征 → P(行终止)。供 nav_core.
set_headland_net() 注入; 每帧 (含 q=0 帧) 前向, 概率由
get_headland_prob() 取走喂控制器掉头触发门调制。

用法:
  from nav_simulator.dl.c5_runtime import load_headland_net
  vn.set_headland_net(load_headland_net())

自测 (随机权重 + torch 前向一致性):
  .venv/Scripts/python.exe -X utf8 -m nav_simulator.dl.c5_runtime --selftest
"""

import argparse
import os
import sys

import numpy as np

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_SIM_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

DEFAULT_WEIGHTS = os.path.join(
    _WORKSPACE_DIR, "models", "c5_headland_weights.npz")


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class HeadlandNetRuntime:
    """MLP 22 → 32 → 16 → 1 前向: feats → P(end)

    callable 接口与 nav_core.set_headland_net 的约定一致:
        __call__(feats (N,22) ndarray) → P (N,) float64
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

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(feats, dtype=np.float32))
        x = (x - self.feat_mean) / self.feat_std
        h = _silu(x @ self.W1.T + self.b1)
        h = _silu(h @ self.W2.T + self.b2)
        logit = (h @ self.W3.T + self.b3)[:, 0]
        return (1.0 / (1.0 + np.exp(-logit))).astype(np.float64)


def load_headland_net(npz_path: str = DEFAULT_WEIGHTS):
    """权重文件 → callable(feats) → P(end); 文件缺失返回 None"""
    if not os.path.isfile(npz_path):
        return None
    z = np.load(npz_path)
    return HeadlandNetRuntime(
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
        in_dim = 22
        z = {
            "W1": rng.randn(32, in_dim).astype(np.float32) * 0.3,
            "b1": rng.randn(32).astype(np.float32) * 0.1,
            "W2": rng.randn(16, 32).astype(np.float32) * 0.3,
            "b2": rng.randn(16).astype(np.float32) * 0.1,
            "W3": rng.randn(1, 16).astype(np.float32) * 0.3,
            "b3": rng.randn(1).astype(np.float32) * 0.1,
            "feat_mean": rng.randn(in_dim).astype(np.float32) * 0.2,
            "feat_std": (rng.rand(in_dim).astype(np.float32) + 0.5),
        }

    rt = HeadlandNetRuntime(
        z["W1"], z["b1"], z["W2"], z["b2"], z["W3"], z["b3"],
        z["feat_mean"], z["feat_std"])

    net = nn.Sequential(
        nn.Linear(in_dim, 32), nn.SiLU(),
        nn.Linear(32, 16), nn.SiLU(),
        nn.Linear(16, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(
            np.asarray(z["W1"], np.float32)))
        net[0].bias.copy_(torch.from_numpy(
            np.asarray(z["b1"], np.float32)))
        net[2].weight.copy_(torch.from_numpy(
            np.asarray(z["W2"], np.float32)))
        net[2].bias.copy_(torch.from_numpy(
            np.asarray(z["b2"], np.float32)))
        net[4].weight.copy_(torch.from_numpy(
            np.asarray(z["W3"], np.float32)))
        net[4].bias.copy_(torch.from_numpy(
            np.asarray(z["b3"], np.float32)))

    feats = rng.randn(512, in_dim).astype(np.float64) * 2.0
    p_rt = rt(feats)
    feats_n = ((feats.astype(np.float32)
                - np.asarray(z["feat_mean"], np.float32))
               / np.asarray(z["feat_std"], np.float32))
    with torch.no_grad():
        logit = net(torch.from_numpy(feats_n))
    p_t = torch.sigmoid(logit[:, 0]).numpy().astype(np.float64)

    d = float(np.abs(p_rt - p_t).max())
    print(f"max |ΔP| = {d:.3e}")
    ok = d < 1e-5
    print("PASS: numpy runtime ↔ torch forward consistent"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C5 runtime selftest")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest(args.weights))
    rt = load_headland_net(args.weights)
    print("loaded" if rt is not None
          else f"weights not found: {args.weights}")
