"""
metrics.py - 导航性能评价指标模块 (P0 仿真基建)

实现视觉导航V2开发文档 §4.1 的指标体系。核心思想:
压苗率 (农艺损失指标) 为一等公民, 区别于文献普遍只报 RMSE 的做法。

指标分四组:
  农艺损失   crush_rate            |CTE真值| > d_crush 的作业帧占比
  基础精度   cte_rmse / cte_final / settle_time_s / steady 噪声
  丢线行为   line_loss_rate        无有效导航输出帧占比
  扰动恢复   遮挡窗内漂移 / 窗后重捕获延迟 (需注入窗口信息)

压苗预算 d_crush 推导 (开发文档 §1.4, 参数为仿真器默认值):
  车骑目标行时轮心 ±track/2=0.86, 最近苗行中心在 ±0.60 (隔一行),
  轮心距苗行中心 0.26, 减苗半宽 0.06 与胎半宽 0.15 → d_crush = 0.05 m。
  通式: d_crush = min_k |track/2 - k·spacing| - crop_w/2 - tire_w/2
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict

from .perturbation import OcclusionWindow


@dataclass
class AgronomyParams:
    """农艺/车辆几何参数 (压苗预算与压苗率判定用)"""
    row_spacing_m: float = 0.60      # 行距 (中心线间距)
    wheel_track_m: float = 1.72      # 轮距
    tire_width_m: float = 0.30       # 胎宽
    crop_row_width_m: float = 0.12   # 苗带宽 (根茎带)


def crush_budget_m(p: AgronomyParams) -> float:
    """压苗预算: |CTE| 超过此值时至少一侧轮胎压到苗带 (骑行姿态下)

    轮心到最近苗行中心的最小距离 - 苗半宽 - 胎半宽。
    默认参数: |0.86-0.60| - 0.06 - 0.15 = 0.05 m。
    """
    half_track = p.wheel_track_m / 2.0
    best = float('inf')
    k = 1
    while k * p.row_spacing_m <= half_track + p.row_spacing_m:
        best = min(best, abs(half_track - k * p.row_spacing_m))
        k += 1
    return best - p.crop_row_width_m / 2.0 - p.tire_width_m / 2.0


def evaluate(results,
             agro: Optional[AgronomyParams] = None,
             occlusion_windows: Optional[List[OcclusionWindow]] = None,
             settle_band_m: float = 0.05,
             include_headland: bool = False
             ) -> Dict:
    """对一次仿真的 FrameResult 序列计算全部指标

    Args:
        results: NavSimulator.results (需含 cte_true / occluded 字段)
        agro: 农艺参数 (默认取仿真器默认值)
        occlusion_windows: 遮挡注入窗 (计算扰动恢复指标; None 跳过)
        settle_band_m: 收敛判定带宽 (|CTE| 持续小于此值视为收敛)
        include_headland: 是否把地头转向帧计入压苗率 (默认只统计 WORK)

    Returns:
        dict, 全部为 JSON 可序列化的标量/列表
    """
    agro = agro or AgronomyParams()
    d_crush = crush_budget_m(agro)

    # 作业帧过滤 (地头转向不是对行作业, 压苗判定无意义)
    if include_headland:
        work = list(results)
    else:
        work = [r for r in results if r.control_mode == 'WORK']
    if not work:
        return {"frames_total": len(results), "frames_work": 0}

    times = np.array([r.sim_time for r in work])
    cte = np.array([r.cte_true for r in work])
    valid = np.array([bool(r.nav_valid) for r in work])
    occluded = np.array([bool(r.occluded) for r in work])
    yaw = np.array([r.yaw_deg for r in work])
    front = np.array([r.front_dist for r in work])
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / 15.0

    # ── 农艺损失: 压苗率 ──
    crush_mask = np.abs(cte) > d_crush
    crush_rate = float(np.mean(crush_mask))

    # ── 基础精度 ──
    cte_abs = np.abs(cte)
    settle_idx = _settle_index(cte_abs, settle_band_m)
    settle_time_s = (float(times[settle_idx]) if settle_idx is not None
                     else float('nan'))
    steady = slice(settle_idx, None) if settle_idx is not None else slice(0, 0)
    n_steady = int(np.count_nonzero(~occluded[steady] & valid[steady]))

    m = {
        "frames_total": len(results),
        "frames_work": len(work),
        "d_crush_m": round(d_crush, 4),
        # 农艺损失
        "crush_rate": round(crush_rate, 4),
        "crush_rate_occlusion_excl": round(
            float(np.mean(crush_mask[~occluded])) if np.any(~occluded)
            else float('nan'), 4),
        # 基础精度
        "cte_rmse_m": round(float(np.sqrt(np.mean(cte ** 2))), 4),
        "cte_max_abs_m": round(float(np.max(cte_abs)), 4),
        "cte_final_m": round(float(cte[-1]), 4),
        "settle_time_s": round(settle_time_s, 2),
        "cte_steady_rmse_m": (round(float(np.sqrt(np.mean(
            cte[steady] ** 2))), 4) if n_steady > 10 else float('nan')),
        "yaw_steady_std_deg": (round(float(np.std(
            yaw[steady][~occluded[steady] & valid[steady]])), 4)
            if n_steady > 10 else float('nan')),
        "front_steady_std_m": (round(float(np.std(
            front[steady][~occluded[steady] & valid[steady]])), 4)
            if n_steady > 10 else float('nan')),
        # 丢线行为
        "line_loss_rate": round(float(1.0 - np.mean(valid)), 4),
        "line_loss_rate_occlusion": (round(float(
            1.0 - np.mean(valid[occluded])), 4)
            if np.any(occluded) else float('nan')),
        # 误触发掉头: WORK 中连续丢行超阈值 → 控制器判行尾进掉头程序。
        # 遮挡/断垄场景下任何掉头都是误判 (行还长着), 是最刺眼的失效模式
        # (被挡 1.67s 即在田中间掉头彻底迷路)。
        "headland_triggers": _count_headland_triggers(results),
    }

    # ── 扰动恢复指标 (需要注入窗口信息) ──
    if occlusion_windows:
        m["occlusion_stats"] = [
            _occlusion_window_stats(w, times, cte, valid, occluded, dt)
            for w in occlusion_windows
        ]

    return m


def _count_headland_triggers(results) -> int:
    """统计 WORK → 地头转向 (HD_*) 的切换次数

    直线/短行程场景下不会自然到达行尾, 任何切换都是丢线误判触发
    (vehicle_model.headland_trigger_frames=25 帧连续丢行即判行尾)。
    """
    count = 0
    prev_work = True
    for r in results:
        is_work = (r.control_mode == 'WORK')
        if prev_work and not is_work:
            count += 1
        prev_work = is_work
    return count


def _settle_index(cte_abs: np.ndarray, band: float) -> Optional[int]:
    """收敛帧号: 最后一次 |CTE|>band 的下一帧 (之后全部在带内)

    全程未收敛返回 None。
    """
    outside = np.nonzero(cte_abs > band)[0]
    if len(outside) == 0:
        return 0
    idx = int(outside[-1]) + 1
    return idx if idx < len(cte_abs) else None


def _occlusion_window_stats(w: OcclusionWindow, times, cte, valid, occluded,
                            dt: float) -> Dict:
    """单个遮挡窗的恢复指标

    drift_max_m:     窗内最大 |CTE| (闭眼行走漂移能力)
    valid_in_window: 窗内有效输出占比 (基线 holdover≈0.67s 后应塌到 0)
    reacquire_delay_s: 窗结束后恢复有效输出的延迟 (窗内从未失效则=0)
    """
    in_win = (times >= w.start_s) & (times < w.start_s + w.duration_s)
    after_win = times >= w.start_s + w.duration_s

    stats = {
        "start_s": w.start_s,
        "duration_s": w.duration_s,
        "frames_in_window": int(np.count_nonzero(in_win)),
    }

    if np.any(in_win):
        stats["drift_max_m"] = round(
            float(np.max(np.abs(cte[in_win]))), 4)
        stats["valid_in_window"] = round(
            float(np.mean(valid[in_win])), 4)
        # 重捕获: 窗内失效过才需要重捕获, 取窗结束后首个有效帧延迟
        lost_in_win = not bool(np.all(valid[in_win]))
    else:
        stats["drift_max_m"] = float('nan')
        stats["valid_in_window"] = float('nan')
        lost_in_win = False

    delay = 0.0
    if lost_in_win and np.any(after_win):
        va = valid[after_win]
        first_valid = np.nonzero(va)[0]
        delay = (float('nan') if len(first_valid) == 0
                 else float(first_valid[0]) * dt)
    stats["reacquire_delay_s"] = round(delay, 2) if delay == delay else delay
    return stats
