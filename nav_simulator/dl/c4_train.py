"""
c4_train.py - C4 BEV 时序证据聚合 (手搓 ConvGRU) 训练 + 评估

网格: 世界对齐局部窗 32m×32m @ 0.5m = 64×64 (训练/推理一致;
0.25m 网格 CPU 训练预算不可行, 0.5m 对行线定位足够 —— 观测高斯
晕染后行结构清晰)。通道: 观测证据 + 车辆位置标记 + 航向 sin/cos。

模型: 手搓 ConvGRU cell (conv2d + sigmoid/tanh 手工搭门, 无 RNN
专用算子 —— ONNX 全普通算子, RKNN 直接吃) + 1×1 head 出 P(row)。
隐藏 8 通道, ~15K 参数。

窗平移: 窗 origin 0.5m 量化对齐世界系 (相邻帧窗移动 = 整数像素),
隐状态按窗移动量平移 (零填充切片, 不用 roll —— 行在 y 向贯穿
窗口, 环绕伪影会把窗外行证据搬到对侧)。

损失: BCE (正像素 ~5%, 加权) + Dice。

评估 (消融 A4 核心): 同一网络两种模式 —— 递归 (h 逐帧传递) vs
单帧 (h 每帧清零, 等价无时序 CNN)。加难场景 (10s 遮挡/15m 断垄)
上递归模式应显著胜出: 遮挡期间无观测输入, 单帧 P 图必空, 递归
靠时序记忆保持行证据。

用法 (工作区根目录, 先跑 c4_dump.py 产数据):
  .venv/Scripts/python.exe -X utf8 -m nav_simulator.dl.c4_train

输出:
  models/c4_bev.pt / c4_bev_weights.npz
  results/c4/metrics.json  (递归 vs 单帧 Dice/召回对比)
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_SIM_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 网格常量 ──
GRID = 64           # 64×64
RES = 0.5           # 0.5 m/px
SPAN = GRID * RES   # 32 m
HALF = SPAN / 2.0   # 16 m
IN_CH = 4           # 观测证据/车辆位置/sin/cos

# 观测高斯晕染核 (7×7, σ=1.5px)
_K = 7
_KSIG = 1.5
_kx = np.arange(_K) - _K // 2
_kg = np.exp(-(_kx ** 2) / (2 * _KSIG ** 2))
GAUSS_K = np.outer(_kg, _kg).astype(np.float32)   # (7,7)
GAUSS_K /= GAUSS_K.max()

# 目标栅格膨胀核 (3×3 十字: 行线粗 1px → 判定容忍 ±1px)
CROSS_K = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                   dtype=np.float32)


def quant_origin(px, py):
    """窗 origin (西南角) 0.5m 量化对齐: 相邻帧移动=整数像素"""
    return (math.floor((px - HALF) / RES) * RES,
            math.floor((py - HALF) / RES) * RES)


def world_to_pix(gx, gy, ox, oy):
    """全局坐标 → 网格 (i 行=北向, j 列=东向); 返回 float 索引"""
    return (gy - oy) / RES, (gx - ox) / RES


# ── 单帧输入/目标构造 ──

def build_input(obs_xy_g, px, py, heading):
    """构 4 通道 BEV 输入。obs_xy_g: (M,2) 全局系观测点"""
    ox, oy = quant_origin(px, py)
    ch = np.zeros((IN_CH, GRID, GRID), dtype=np.float32)
    if len(obs_xy_g):
        fi, fj = world_to_pix(obs_xy_g[:, 0], obs_xy_g[:, 1], ox, oy)
        # 高斯晕染 scatter (kernel 展开索引)
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
            ch[0] = np.minimum(ch[0], 2.0)   # 叠加限幅
    # 车辆位置标记
    ci, cj = world_to_pix(px, py, ox, oy)
    ci, cj = int(round(ci)), int(round(cj))
    if 0 <= ci < GRID and 0 <= cj < GRID:
        ch[1, ci, cj] = 1.0
    # 航向 (罗盘角: 0=北, 顺时针正)
    ch[2] = math.sin(heading)
    ch[3] = math.cos(heading)
    return ch, (ox, oy)


def build_target(row_pts, row_bounds, ox, oy):
    """真值行栅格 (0/1, 行点 scatter + 十字膨胀)"""
    t = np.zeros((GRID, GRID), dtype=np.float32)
    fi, fj = world_to_pix(row_pts[:, 0], row_pts[:, 1], ox, oy)
    ii = np.round(fi).astype(np.int64)
    jj = np.round(fj).astype(np.int64)
    ok = (ii >= 0) & (ii < GRID) & (jj >= 0) & (jj < GRID)
    if ok.any():
        np.add.at(t, (ii[ok], jj[ok]), 1.0)
    t = cv2.filter2D(t, -1, CROSS_K, borderType=cv2.BORDER_CONSTANT)
    return (t > 0).astype(np.float32)


# ── 数据集 (在线构造序列) ──

class RunData:
    """单个 run 的预加载数据 (按帧分桶观测 + 帧号→位姿映射 + 行折线)"""

    def __init__(self, key, npz):
        self.key = key
        pose = npz[f"{key}::pose"]               # (F, 3) 基准系
        pose_fr = npz[f"{key}::pose_frame"]      # (F,) 1-based 帧号
        self.pose_by_frame = {int(fr): pose[i]
                              for i, fr in enumerate(pose_fr)}
        self.n_frames = int(pose_fr.max())
        obs_xy = npz[f"{key}::obs_xy"]              # (M, 2) 基准系
        obs_fr = npz[f"{key}::obs_frame"]           # (M,) 1-based
        self.origin = npz[f"{key}::origin"]         # 基准系→全局系
        self.row_pts = npz[f"{key}::row_pts"]       # (P, 2) 全局系
        self.row_bounds = npz[f"{key}::row_bounds"]
        # 按帧分桶 (全局系观测点)
        self.obs_by_frame = {}
        if len(obs_xy):
            gx = obs_xy[:, 0] + self.origin[0]
            gy = obs_xy[:, 1] + self.origin[1]
            for fr in np.unique(obs_fr):
                m = obs_fr == fr
                self.obs_by_frame[int(fr)] = np.stack(
                    [gx[m], gy[m]], axis=1)

    def pose_at(self, frame: int):
        """帧号 → (px, py, h); 缺帧时取最近已记录帧 (防御, 实际连续)"""
        if frame in self.pose_by_frame:
            return self.pose_by_frame[frame]
        if not self.pose_by_frame:
            return (0.0, 0.0, 0.0)
        best = min(self.pose_by_frame.keys(),
                   key=lambda f: abs(f - frame))
        return self.pose_by_frame[best]


def build_sequence(r: "RunData", f0: int, k: int):
    """从帧 f0 (1-based) 起构 k 帧序列 (输入/目标/窗 origin)
    —— Dataset 与逐 run 评估共用, 保证同源"""
    xs, ys, oxs = [], [], []
    for f in range(f0, f0 + k):
        px, py, h = r.pose_at(f)
        obs = r.obs_by_frame.get(f, np.zeros((0, 2), np.float32))
        x, (ox, oy) = build_input(obs, px, py, h)
        y = build_target(r.row_pts, r.row_bounds, ox, oy)
        xs.append(x)
        ys.append(y)
        oxs.append((ox, oy))
    return (torch.from_numpy(np.stack(xs)),
            torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.asarray(oxs, dtype=np.float32)))


class BEVDataset(torch.utils.data.Dataset):
    """随机序列采样: __getitem__ 返回 (K, IN_CH, G, G) + (K, G, G)"""

    def __init__(self, runs, k: int, seed: int):
        self.runs = runs
        self.k = k
        self.rng = np.random.RandomState(seed)
        # 序列起点池: 每 run 的 [1, n_frames-K] (帧号 1-based)
        self.starts = [(ri, f0) for ri, r in enumerate(runs)
                       for f0 in range(1, r.n_frames - k + 1)]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        ri, f0 = self.starts[idx]
        x, y, origins = build_sequence(self.runs[ri], f0, self.k)
        return (x, y, origins, self.k)


# ── 模型 ──

class ConvGRUCell(nn.Module):
    """手搓 conv GRU cell (ONNX 全普通算子)"""

    def __init__(self, in_ch: int, hid_ch: int):
        super().__init__()
        # 更新/重置门共享一个 conv (算子数减半)
        self.conv_gz = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, 3, padding=1)
        self.conv_hh = nn.Conv2d(in_ch + hid_ch, hid_ch, 3, padding=1)

    def forward(self, x, h):
        gz = torch.sigmoid(self.conv_gz(torch.cat([x, h], dim=1)))
        z, r = gz.chunk(2, dim=1)
        hh = torch.tanh(self.conv_hh(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * hh


def shift_h(h: torch.Tensor, di: int, dj: int) -> torch.Tensor:
    """隐状态按窗移动平移 (零填充, 不环绕)。

    窗 origin 移动 d(oy, ox) → 内容平移 di=round(d_oy/RES),
    dj=round(d_ox/RES): out[i,j] = h[i+di, j+dj]。
    """
    if di == 0 and dj == 0:
        return h
    B, C, H, W = h.shape
    out = torch.zeros_like(h)
    i0, i1 = max(0, -di), min(H, H - di)
    j0, j1 = max(0, -dj), min(W, W - dj)
    if i1 <= i0 or j1 <= j0:
        return out
    out[:, :, i0:i1, j0:j1] = \
        h[:, :, i0 + di:i1 + di, j0 + dj:j1 + dj]
    return out


class BEVAggregator(nn.Module):
    """ConvGRU 聚合器: 序列 → 每帧 P(row) logit"""

    def __init__(self, in_ch: int = IN_CH, hid: int = 8):
        super().__init__()
        self.hid = hid
        self.cell = ConvGRUCell(in_ch, hid)
        self.head = nn.Conv2d(hid, 1, 1)

    def forward(self, x, origins, reset_every: int = 0):
        """x: (B, K, C, H, W); origins: (B, K, 2) 每帧窗 origin;
        reset_every>0 时每 N 帧清零 h (N=1 即单帧基线)。
        Returns: (B, K, H, W) logit"""
        B, K = x.shape[:2]
        h = x.new_zeros(B, self.hid, x.shape[3], x.shape[4])
        outs = []
        prev_ox = origins[:, 0, 0]
        prev_oy = origins[:, 0, 1]
        for t in range(K):
            if t > 0:
                # 每样本独立平移 (batch 内窗移动不同)。
                # 窗 origin 北移 (oy 增大) → 内容 i 减小 → di 为正
                di_b = torch.round(
                    (origins[:, t, 1] - prev_oy) / RES).long()
                dj_b = torch.round(
                    (origins[:, t, 0] - prev_ox) / RES).long()
                hs = []
                for b in range(B):
                    hs.append(shift_h(h[b:b + 1],
                                       int(di_b[b]), int(dj_b[b])))
                h = torch.cat(hs, dim=0)
            if reset_every and t % reset_every == 0:
                h = h * 0.0
            h = self.cell(x[:, t], h)
            outs.append(self.head(h))
            prev_ox, prev_oy = origins[:, t, 0], origins[:, t, 1]
        return torch.stack(outs, dim=1).squeeze(2)


# ── 损失 ──

def seg_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE (正像素加权) + Dice"""
    pos_frac = target.mean(dim=(1, 2, 3), keepdim=True).clamp(0.05, 0.5)
    w = torch.where(target > 0.5,
                    1.0 / (2.0 * pos_frac),
                    1.0 / (2.0 * (1.0 - pos_frac)))
    bce = F.binary_cross_entropy_with_logits(logit, target, weight=w)
    p = torch.sigmoid(logit)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    dice = 1.0 - 2.0 * inter / denom
    return bce + dice.mean()


# ── 评估 ──

def evaluate(model, loader, dev):
    """递归 vs 单帧两种模式: Dice / 行像素召回 (P>0.5) —— 训练中
    快速早停信号 (固定 val 序列子集)"""
    model.eval()
    stats = {"recur": [0.0, 0.0, 0], "single": [0.0, 0.0, 0]}
    with torch.no_grad():
        for x, y, origins, k_eff in loader:
            x, y, origins = x.to(dev), y.to(dev), origins.to(dev)
            for mode, reset in (("recur", 0), ("single", 1)):
                logit = model(x, origins, reset_every=reset)
                p = torch.sigmoid(logit)
                pred = (p > 0.5).float()
                inter = (pred * y).sum(dim=(1, 2, 3))
                denom = (pred.sum(dim=(1, 2, 3))
                         + y.sum(dim=(1, 2, 3)) + 1e-6)
                dice = (2.0 * inter / denom).sum().item()
                tp = inter.sum().item()
                gt = y.sum().item()
                s = stats[mode]
                s[0] += dice
                s[1] += tp / max(gt, 1.0)
                s[2] += 1
    out = {}
    for mode, (d, r, n) in stats.items():
        out[f"{mode}_dice"] = d / max(n, 1)
        out[f"{mode}_row_recall"] = r / max(n, 1)
    return out


def evaluate_runs(model, runs, k, dev, max_seqs=6):
    """逐 run 评估 (消融 A4 证据): 每 run 均匀滑窗采样序列,
    分场景输出递归 vs 单帧 Dice/召回 —— 加难场景 (遮挡/断垄)
    上递归应显著胜出"""
    model.eval()
    per_run = {}
    agg = {"recur": [0.0, 0.0], "single": [0.0, 0.0]}
    with torch.no_grad():
        for r in runs:
            stats = {"recur": [0.0, 0.0, 0], "single": [0.0, 0.0, 0]}
            stride = max(k, (r.n_frames - k) // max_seqs + 1)
            n_seq = 0
            f0 = 1
            while f0 + k - 1 <= r.n_frames and n_seq < max_seqs:
                x, y, origins = build_sequence(r, f0, k)
                x = x.unsqueeze(0).to(dev)
                origins = origins.unsqueeze(0).to(dev)
                y_dev = y.unsqueeze(0).to(dev)
                for mode, reset in (("recur", 0), ("single", 1)):
                    logit = model(x, origins, reset_every=reset)
                    p = torch.sigmoid(logit)
                    pred = (p > 0.5).float()
                    inter = (pred * y_dev).sum(dim=(1, 2, 3))
                    denom = (pred.sum(dim=(1, 2, 3))
                             + y_dev.sum(dim=(1, 2, 3)) + 1e-6)
                    s = stats[mode]
                    s[0] += (2.0 * inter / denom).mean().item()
                    s[1] += (inter.sum().item()
                             / max(y_dev.sum().item(), 1.0))
                    s[2] += 1
                n_seq += 1
                f0 += stride
            if stats["recur"][2] == 0:
                continue
            entry = {}
            for mode, (d, rc, n) in stats.items():
                entry[f"{mode}_dice"] = d / n
                entry[f"{mode}_row_recall"] = rc / n
                agg[mode][0] += d / n
                agg[mode][1] += rc / n
            per_run[r.key] = entry
    n_runs = len(per_run)
    summary = {m: {"dice": v[0] / n_runs, "row_recall": v[1] / n_runs}
               for m, v in agg.items()} if n_runs else {}
    return {"per_run": per_run, "summary": summary}


# ── 主流程 ──

def main():
    ap = argparse.ArgumentParser(description="C4 ConvGRU training")
    ap.add_argument("--data", default=os.path.join(
        _WORKSPACE_DIR, "datasets", "c4_bev_v1.npz"))
    ap.add_argument("--k", type=int, default=24,
                    help="训练序列长度 (帧, BPTT 步数)")
    ap.add_argument("--hid", type=int, default=8)
    ap.add_argument("--batches", type=int, default=200,
                    help="每 epoch batch 数")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-seed", type=int, default=2)
    ap.add_argument("--out-dir", default=os.path.join(
        _WORKSPACE_DIR, "results", "c4"))
    ap.add_argument("--model-dir", default=os.path.join(
        _WORKSPACE_DIR, "models"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cpu"

    print(f"loading {args.data} ...")
    npz = np.load(args.data)
    scen_list = [str(s) for s in npz["scen_list"]]
    scen_seeds = npz["scen_seeds"]
    runs_tr, runs_val = [], []
    for key, sd in zip(scen_list, scen_seeds):
        r = RunData(key, npz)
        (runs_val if int(sd) == args.val_seed else runs_tr).append(r)
        print(f"  {key:28s} {r.n_frames:4d} frames, "
              f"{len(r.obs_by_frame)} obs-frames")
    print(f"train runs: {len(runs_tr)}, val runs: {len(runs_val)}")

    ds_tr = BEVDataset(runs_tr, args.k, seed=args.seed)
    ds_val = BEVDataset(runs_val, args.k, seed=1234)
    # 验证 loader 固定前 48 个序列 (确定性)
    val_subset = torch.utils.data.Subset(
        ds_val, list(range(0, len(ds_val), max(1, len(ds_val) // 48)))[:48])
    loader_val = torch.utils.data.DataLoader(
        val_subset, batch_size=args.val_batch, shuffle=False)
    loader_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=args.batch, shuffle=True,
        num_workers=0, drop_last=True)

    model = BEVAggregator(hid=args.hid).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)

    best_val, best_state = -1.0, None
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        it = iter(loader_tr)
        tot, nb = 0.0, 0
        for _ in range(args.batches):
            try:
                x, y, origins, _ = next(it)
            except StopIteration:
                it = iter(loader_tr)
                x, y, origins, _ = next(it)
            logit = model(x, origins)
            loss = seg_loss(logit, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()

        metrics = evaluate(model, loader_val, dev)
        marker = ""
        if metrics["recur_dice"] > best_val:
            best_val = metrics["recur_dice"]
            best_state = {k: v.clone() for k, v in
                          model.state_dict().items()}
            marker = " *"
        print(f"epoch {epoch+1:3d}: loss={tot/nb:.4f} "
              f"recur_dice={metrics['recur_dice']:.3f} "
              f"single_dice={metrics['single_dice']:.3f} "
              f"recur_rec={metrics['recur_row_recall']:.3f}"
              f"{marker}  ({time.time()-t_start:.0f}s)")

    model.load_state_dict(best_state)
    metrics = evaluate(model, loader_val, dev)
    print("\nfinal (best val, fixed subset):")
    for k_, v_ in metrics.items():
        print(f"  {k_:22s} {v_:.3f}")

    # 逐 run 评估 (分场景消融证据: 加难场景递归 vs 单帧)
    print("\nper-run evaluation (recur vs single):")
    ev = evaluate_runs(model, runs_val, args.k, dev)
    for key, entry in ev["per_run"].items():
        print(f"  {key:28s} dice {entry['single_dice']:.3f} → "
              f"{entry['recur_dice']:.3f}   recall "
              f"{entry['single_row_recall']:.3f} → "
              f"{entry['recur_row_recall']:.3f}")

    # ── 落盘 ──
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "params": n_params, "k": args.k, "grid": GRID,
            "res": RES, "final": metrics,
            "per_run": ev["per_run"],
            "summary": ev["summary"],
            "temporal_gain_dice": metrics["recur_dice"]
                                  - metrics["single_dice"],
            "temporal_gain_recall": metrics["recur_row_recall"]
                                    - metrics["single_row_recall"],
        }, f, ensure_ascii=False, indent=2)

    torch.save({
        "state_dict": model.state_dict(),
        "grid": GRID, "res": RES, "hid": args.hid,
        "in_ch": IN_CH,
    }, os.path.join(args.model_dir, "c4_bev.pt"))

    sd = model.state_dict()
    np.savez(os.path.join(args.model_dir, "c4_bev_weights.npz"),
             gz_W=sd["cell.conv_gz.weight"].numpy(),
             gz_b=sd["cell.conv_gz.bias"].numpy(),
             hh_W=sd["cell.conv_hh.weight"].numpy(),
             hh_b=sd["cell.conv_hh.bias"].numpy(),
             head_W=sd["head.weight"].numpy(),
             head_b=sd["head.bias"].numpy(),
             grid=np.int64(GRID), res=np.float64(RES),
             hid=np.int64(args.hid), in_ch=np.int64(IN_CH))
    print(f"\nsaved: {args.model_dir}\\c4_bev.pt / c4_bev_weights.npz")
    print(f"saved: {args.out_dir}\\metrics.json")


if __name__ == "__main__":
    main()
