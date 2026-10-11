"""
c4_runtime.py - C4 BEV 时序证据聚合 (手搓 ConvGRU) numpy 推理前向

从 c4_train.py 落盘的裸权重 (models/c4_bev_weights.npz) 构造
逐帧前向 runtime: 每帧喂 (观测点, 位姿) → 维护隐状态 (窗平移
对齐) → 输出 P(row) 网格。conv 用 cv2.filter2D (与 torch
conv2d 同为 cross-correlation, 算式一致, 无需翻转核)。

用法:
  from nav_simulator.dl.c4_runtime import load_bev_net
  vn.set_bev_net(load_bev_net())   # None 时权重缺失
  # 每帧 nav_core 内部调用 runtime.step(obs_xy, px, py, heading)

自测 (随机权重 + torch 前向一致性):
  .venv/Scripts/python.exe -X utf8 -m nav_simulator.dl.c4_runtime --selftest
"""

import argparse
import os
import sys

import cv2
import numpy as np

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_SIM_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

DEFAULT_WEIGHTS = os.path.join(
    _WORKSPACE_DIR, "models", "c4_bev_weights.npz")

# 与 c4_train.py 同源的网格常量
GRID = 64
RES = 0.5
HALF = GRID * RES / 2.0
IN_CH = 4

# 观测高斯晕染核 (7×7, σ=1.5px) —— 与训练侧 build_input 一致
_K = 7
_kx = np.arange(_K) - _K // 2
_kg = np.exp(-(_kx ** 2) / (2 * 1.5 ** 2))
GAUSS_K = np.outer(_kg, _kg).astype(np.float32)
GAUSS_K /= GAUSS_K.max()


def quant_origin(px, py):
    """窗 origin (西南角) 0.5m 量化对齐 (训练/推理一致)"""
    return (np.floor((px - HALF) / RES) * RES,
            np.floor((py - HALF) / RES) * RES)


def build_input_np(obs_xy_w, px, py, heading):
    """4 通道 BEV 输入 (numpy 版, 与 c4_train.build_input 同式)"""
    ox, oy = quant_origin(px, py)
    ch = np.zeros((IN_CH, GRID, GRID), dtype=np.float32)
    obs = np.asarray(obs_xy_w, dtype=np.float64).reshape(-1, 2)
    if len(obs):
        fi = (obs[:, 1] - oy) / RES
        fj = (obs[:, 0] - ox) / RES
        base_i = np.round(fi).astype(np.int64)
        base_j = np.round(fj).astype(np.int64)
        ok = ((base_i >= _K // 2) & (base_i < GRID - _K // 2)
              & (base_j >= _K // 2) & (base_j < GRID - _K // 2))
        if ok.any():
            bi, bj = base_i[ok], base_j[ok]
            off = np.arange(_K) - _K // 2
            di = off[None, :, None] + np.zeros((1, _K, _K), np.int64)
            dj = off[None, None, :] + np.zeros((1, _K, _K), np.int64)
            ii = (bi[:, None, None] + di).ravel()
            jj = (bj[:, None, None] + dj).ravel()
            w = np.broadcast_to(GAUSS_K, (len(bi), _K, _K)).ravel()
            np.add.at(ch[0], (ii, jj), w)
            ch[0] = np.minimum(ch[0], 2.0)
    ci = int(round((py - oy) / RES))
    cj = int(round((px - ox) / RES))
    if 0 <= ci < GRID and 0 <= cj < GRID:
        ch[1, ci, cj] = 1.0
    ch[2] = np.float32(np.sin(heading))
    ch[3] = np.float32(np.cos(heading))
    return ch, (float(ox), float(oy))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _conv2d(x, W, b):
    """x (C,G,G), W (O,C,3,3), b (O,) → (O,G,G)。
    cv2.filter2D = cross-correlation, 与 torch conv2d 同算式;
    输入零填充 (pad=1), 输出取 filter2D 结果内部区域
    (filter2D 同尺寸输出, [1:-1] 恰对应 torch 的 pad=1 valid 位)。"""
    O = W.shape[0]
    G = x.shape[1]
    xp = np.zeros((x.shape[0], G + 2, G + 2), dtype=np.float32)
    xp[:, 1:-1, 1:-1] = x
    out = np.empty((O, G, G), dtype=np.float32)
    for o in range(O):
        acc = np.zeros((G + 2, G + 2), dtype=np.float32)
        for c in range(x.shape[0]):
            acc += cv2.filter2D(xp[c], cv2.CV_32F, W[o, c])
        out[o] = acc[1:-1, 1:-1] + b[o]
    return out


def _shift_h_np(h, di, dj):
    """隐状态按窗移动平移 (零填充, 不环绕) —— 与 c4_train.shift_h
    同式: out[i,j] = h[i+di, j+dj]"""
    if di == 0 and dj == 0:
        return h
    out = np.zeros_like(h)
    H, W = h.shape[1], h.shape[2]
    i0, i1 = max(0, -di), min(H, H - di)
    j0, j1 = max(0, -dj), min(W, W - dj)
    if i1 <= i0 or j1 <= j0:
        return out
    out[:, i0:i1, j0:j1] = h[:, i0 + di:i1 + di, j0 + dj:j1 + dj]
    return out


class BEVRuntime:
    """ConvGRU 逐帧前向: step(obs_xy, px, py, heading) → P 图

    状态: 隐状态 h (hid,G,G) + 上帧窗 origin。reset() 清零
    (新地块/重捕获时调用; 正常连续作业逐帧 step 不重置)。
    """

    def __init__(self, gz_W, gz_b, hh_W, hh_b, head_W, head_b,
                 hid, in_ch):
        self.gz_W = np.ascontiguousarray(gz_W, np.float32)
        self.gz_b = np.ascontiguousarray(gz_b, np.float32)
        self.hh_W = np.ascontiguousarray(hh_W, np.float32)
        self.hh_b = np.ascontiguousarray(hh_b, np.float32)
        self.head_W = np.ascontiguousarray(head_W, np.float32)
        self.head_b = np.ascontiguousarray(head_b, np.float32)
        self.hid = int(hid)
        self.in_ch = int(in_ch)
        self.grid = GRID          # 网格边长 (px)
        self.res = RES            # 分辨率 (m/px)
        self.reset()

    def reset(self):
        self.h = np.zeros((self.hid, GRID, GRID), dtype=np.float32)
        self.prev_origin = None

    def step(self, obs_xy_w, px, py, heading):
        """一帧前向。Returns: (p (G,G) float32, (ox, oy) 窗 origin)"""
        x, origin = build_input_np(obs_xy_w, px, py, heading)
        if self.prev_origin is not None:
            # 窗 origin 北移 (oy 增大) → 内容 i 减小 → di 为正
            di = int(round((origin[1] - self.prev_origin[1]) / RES))
            dj = int(round((origin[0] - self.prev_origin[0]) / RES))
            self.h = _shift_h_np(self.h, di, dj)
        xh = np.concatenate([x, self.h], axis=0)
        gz = _conv2d(xh, self.gz_W, self.gz_b)
        z_g = _sigmoid(gz[:self.hid])
        r_g = _sigmoid(gz[self.hid:])
        xr = np.concatenate([x, r_g * self.h], axis=0)
        hh = np.tanh(_conv2d(xr, self.hh_W, self.hh_b))
        self.h = (1.0 - z_g) * self.h + z_g * hh
        head = _conv2d(self.h, self.head_W, self.head_b)
        self.prev_origin = origin
        return _sigmoid(head[0]), origin


def load_bev_net(npz_path: str = DEFAULT_WEIGHTS):
    """权重文件 → BEVRuntime; 文件缺失返回 None"""
    if not os.path.isfile(npz_path):
        return None
    z = np.load(npz_path)
    return BEVRuntime(
        z["gz_W"], z["gz_b"], z["hh_W"], z["hh_b"],
        z["head_W"], z["head_b"],
        int(z["hid"]), int(z["in_ch"]))


# ── 自测: 与 torch 前向一致性 (随机权重, 3 帧递归) ──────────

def _selftest(weights_path: str):
    import torch
    sys.path.insert(0, _SIM_DIR)
    from nav_simulator.dl.c4_train import BEVAggregator

    rng = np.random.RandomState(0)
    hid = 8
    if os.path.isfile(weights_path):
        print(f"selftest vs real weights: {weights_path}")
        z = np.load(weights_path)
    else:
        print("no real weights yet, selftest with random weights")
        z = {
            "gz_W": (rng.randn(2 * hid, IN_CH + hid, 3, 3)
                     * 0.2).astype(np.float32),
            "gz_b": (rng.randn(2 * hid) * 0.1).astype(np.float32),
            "hh_W": (rng.randn(hid, IN_CH + hid, 3, 3)
                     * 0.2).astype(np.float32),
            "hh_b": (rng.randn(hid) * 0.1).astype(np.float32),
            "head_W": (rng.randn(1, hid, 1, 1)
                       * 0.3).astype(np.float32),
            "head_b": (rng.randn(1) * 0.1).astype(np.float32),
            "hid": hid, "in_ch": IN_CH,
        }

    rt = BEVRuntime(z["gz_W"], z["gz_b"], z["hh_W"], z["hh_b"],
                    z["head_W"], z["head_b"], z["hid"], z["in_ch"])

    model = BEVAggregator(hid=int(z["hid"]))
    sd = model.state_dict()
    sd["cell.conv_gz.weight"] = torch.from_numpy(
        np.asarray(z["gz_W"], np.float32))
    sd["cell.conv_gz.bias"] = torch.from_numpy(
        np.asarray(z["gz_b"], np.float32))
    sd["cell.conv_hh.weight"] = torch.from_numpy(
        np.asarray(z["hh_W"], np.float32))
    sd["cell.conv_hh.bias"] = torch.from_numpy(
        np.asarray(z["hh_b"], np.float32))
    sd["head.weight"] = torch.from_numpy(
        np.asarray(z["head_W"], np.float32))
    sd["head.bias"] = torch.from_numpy(
        np.asarray(z["head_b"], np.float32))
    model.load_state_dict(sd)
    model.eval()

    # 3 帧递归序列 (含窗移动: py 每帧 +1.1m)
    xs, origins = [], []
    obs_all = []
    for t in range(3):
        px, py = 1.3 + 0.2 * t, 5.0 + 1.1 * t
        obs = rng.randn(20, 2) * 0.3 + (0.0, py + 2)
        obs_all.append((obs, px, py, 0.05))
    from nav_simulator.dl.c4_train import build_input as _bi
    for obs, px, py, h in obs_all:
        x, (ox, oy) = _bi(obs, px, py, h)
        xs.append(x)
        origins.append((ox, oy))
    x_t = torch.from_numpy(np.stack(xs)).unsqueeze(0)
    o_t = torch.from_numpy(
        np.asarray(origins, np.float32)).unsqueeze(0)
    with torch.no_grad():
        logit = model(x_t, o_t)
    # torch 输出 (B,K,G,G): 取末帧与 runtime 逐帧 step 的末帧对齐
    p_torch = torch.sigmoid(logit)[0, -1].numpy()

    rt.reset()
    for obs, px, py, h in obs_all:
        p_rt, _ = rt.step(obs, px, py, h)

    d = float(np.abs(p_rt - p_torch).max())
    print(f"max |ΔP| = {d:.3e}  (after 3 recurrent frames)")
    ok = d < 1e-4
    print("PASS: numpy runtime ↔ torch ConvGRU consistent"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C4 runtime selftest")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest(args.weights))
    rt = load_bev_net(args.weights)
    print("loaded" if rt is not None
          else f"weights not found: {args.weights}")
