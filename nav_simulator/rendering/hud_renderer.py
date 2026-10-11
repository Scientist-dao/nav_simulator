"""
hud_renderer.py - 导航数据 HUD 叠加渲染

在相机视角图像上叠加导航参数文字和图形信息。
现代化半透明面板设计。
"""

import cv2
import numpy as np
from typing import Optional


# 颜色定义 (BGR)
COLOR_GREEN = (0, 230, 0)
COLOR_RED = (0, 0, 230)
COLOR_YELLOW = (0, 230, 230)
COLOR_CYAN = (230, 230, 0)
COLOR_WHITE = (240, 240, 240)
COLOR_GRAY = (140, 140, 140)
COLOR_DARK = (25, 25, 25)
COLOR_PANEL = (35, 35, 35)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hud(img: np.ndarray,
             yaw_deg: float,
             front_dist: float,
             rear_dist: float,
             quality: int,
             speed_ms: float,
             frame_idx: int,
             sim_time: float,
             nav_valid: bool,
             num_masks: int = 0,
             steering_deg: float = 0.0,
             roll_deg: float = 0.0,
             pitch_deg: float = 0.0,
             control_mode: str = 'WORK',
             extra_info: Optional[dict] = None,
             occluded: bool = False,
             rows_detected: int = 0,
             rows_expected: int = 0,
             perturb_info: str = "",
             ) -> np.ndarray:
    """
    在图像上绘制现代化 HUD 面板。

    Args:
        img: BGR 图像
        yaw_deg: 航向角 (度)
        front_dist: 前轴偏差 (m)
        rear_dist: 后轴偏差 (m)
        quality: 图像质量 (0-100)
        speed_ms: 车速 (m/s)
        frame_idx: 帧序号
        sim_time: 仿真时间 (s)
        nav_valid: 导航是否有效
        num_masks: mask 数量
        steering_deg: 转向角 (度)
        control_mode: 控制状态机当前模式 (WORK / 地头转向各段)
        extra_info: 额外信息字典
        occluded: 本帧感知被注入遮挡 (P0 可视化)
        rows_detected/rows_expected: 检出行数/画面可见行数
        perturb_info: 当前激活扰动标签 (如 "OCCL 2.3s+GAP")

    Returns:
        叠加后的图像
    """
    out = img.copy()
    h, w = out.shape[:2]

    # ── 顶部半透明面板 ──
    panel_h = 140
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    # 面板底部分隔线（渐变）
    for i in range(3):
        alpha = 0.5 - i * 0.15
        cv2.line(out, (0, panel_h + i), (w, panel_h + i),
                 tuple(int(c * alpha) for c in COLOR_GRAY), 1)

    # ── 导航状态指示器 ──
    # 遮挡优先级最高: 感知输入已断 (导航输出可能还靠记忆撑着,
    # 但状态必须提示"当前看不见"); 警示横幅已内置于相机图像本体
    # (simulator._draw_occlusion), 此处不重复绘制
    if occluded:
        status_color = COLOR_RED
        status_text = "OCCLUDED"
    elif nav_valid:
        status_color = COLOR_GREEN
        status_text = "NAV OK"
    else:
        status_color = COLOR_RED
        status_text = "NAV LOST"

    # 状态圆点
    cv2.circle(out, (22, 20), 6, status_color, -1)
    cv2.circle(out, (22, 20), 6, tuple(c // 2 for c in status_color), 2)
    cv2.putText(out, status_text, (36, 27), FONT, 0.6,
                status_color, 2)

    # 控制模式标签 (作业循行 / 地头转向程序段)
    mode_color = COLOR_GRAY if control_mode == 'WORK' else COLOR_YELLOW
    cv2.putText(out, f"MODE {control_mode}", (150, 27), FONT, 0.55,
                mode_color, 2 if control_mode != 'WORK' else 1)

    # ── 左列：导航数据 ──
    x_left = 16
    y0 = 52
    dy = 24
    fs = 0.5  # font scale
    th = 1    # thickness

    # Yaw
    _draw_param(out, x_left, y0, "Yaw", f"{yaw_deg:+.3f}", "deg",
                COLOR_WHITE, fs, th)
    # Front dist
    front_color = (COLOR_GREEN if abs(front_dist) < 0.05
                   else COLOR_YELLOW if abs(front_dist) < 0.15
                   else COLOR_RED)
    _draw_param(out, x_left, y0 + dy, "Front", f"{front_dist:+.4f}", "m",
                front_color, fs, th)
    # Rear dist
    _draw_param(out, x_left, y0 + dy * 2, "Rear", f"{rear_dist:+.4f}", "m",
                COLOR_WHITE, fs, th)
    # Quality bar
    _draw_quality_bar(out, x_left, y0 + dy * 3, quality)

    # ── 右列：车辆状态 ──
    x_right = w // 2 + 20
    y0_r = 20

    # 速度（大号显示）
    cv2.putText(out, f"{speed_ms:.1f}", (x_right, y0_r + 18),
                FONT, 0.8, COLOR_CYAN, 2)
    cv2.putText(out, "m/s", (x_right + 70, y0_r + 18),
                FONT, 0.4, COLOR_GRAY, 1)

    # 转向角
    steer_color = (COLOR_GREEN if abs(steering_deg) < 2
                   else COLOR_YELLOW if abs(steering_deg) < 10
                   else COLOR_RED)
    _draw_param(out, x_right, y0_r + 50, "Steer",
                f"{steering_deg:+.2f}", "deg", steer_color, fs, th)

    # 车身姿态 (Roll / Pitch)
    cv2.putText(out, "R/P:", (x_right, y0_r + 76), FONT, fs * 0.75,
                COLOR_GRAY, th)
    cv2.putText(out, f"{roll_deg:+.1f}", (x_right + 70, y0_r + 76),
                FONT, fs, COLOR_WHITE, th)
    cv2.putText(out, f"{pitch_deg:+.1f} deg", (x_right + 130, y0_r + 76),
                FONT, fs * 0.8, COLOR_GRAY, th)

    # Frame / Time / Masks / Rows (合并一行; R:d/e = 检出/可见行数)
    row_tag = f"  R:{rows_detected}/{rows_expected}" if rows_expected else ""
    cv2.putText(out,
                f"F {frame_idx}  {sim_time:.1f}s  M:{num_masks}{row_tag}",
                (x_right, y0_r + 102), FONT, 0.38, COLOR_GRAY, 1)

    # 额外信息
    if extra_info:
        for i, (key, val) in enumerate(extra_info.items()):
            cv2.putText(out, f"{key}: {val}",
                        (x_right, y0_r + 124 + i * 20),
                        FONT, 0.35, COLOR_YELLOW, 1)

    # ── 扰动状态行 (P0 注入: 遮挡/缺苗/弱苗) ──
    y_perturb = y0_r + 124 + len(extra_info or {}) * 20
    if perturb_info:
        cv2.putText(out, f"PERTURB: {perturb_info}",
                    (x_right, y_perturb), FONT, 0.38, COLOR_YELLOW, 1)

    # ── 底部转向指示条 ──
    _draw_steering_bar(out, w, h, steering_deg)

    return out


def _draw_param(img, x, y, label, value, unit, color, fs, th):
    """绘制参数标签+值+单位"""
    cv2.putText(img, f"{label}:", (x, y), FONT, fs * 0.75,
                COLOR_GRAY, th)
    cv2.putText(img, value, (x + 70, y), FONT, fs, color, th)
    cv2.putText(img, unit, (x + 70 + int(len(value) * fs * 12) + 5, y),
                FONT, fs * 0.6, COLOR_GRAY, th)


def _draw_quality_bar(img, x, y, quality):
    """绘制质量指示条"""
    bar_w = 100
    bar_h = 10

    # 背景
    cv2.rectangle(img, (x, y - 8), (x + bar_w, y + bar_h - 8),
                  (60, 60, 60), -1)

    # 填充
    fill_w = int(bar_w * quality / 100)
    if quality > 70:
        color = COLOR_GREEN
    elif quality > 40:
        color = COLOR_YELLOW
    else:
        color = COLOR_RED
    if fill_w > 0:
        cv2.rectangle(img, (x, y - 8), (x + fill_w, y + bar_h - 8),
                      color, -1)

    # 边框
    cv2.rectangle(img, (x, y - 8), (x + bar_w, y + bar_h - 8),
                  COLOR_GRAY, 1)

    # 数值
    cv2.putText(img, f"{quality}", (x + bar_w + 8, y),
                FONT, 0.45, color, 1)


def _draw_steering_bar(img, w, h, steering_deg):
    """底部转向角度指示条"""
    bar_w = 200
    bar_h = 6
    cx = w // 2
    by = h - 15

    # 背景条
    overlay = img.copy()
    cv2.rectangle(overlay, (cx - bar_w // 2, by),
                  (cx + bar_w // 2, by + bar_h), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # 中心刻度
    cv2.line(img, (cx, by - 3), (cx, by + bar_h + 3),
             COLOR_GRAY, 1)

    # 转向指示
    max_steer = 30.0  # 最大显示角度
    offset = int(np.clip(steering_deg / max_steer, -1, 1) * bar_w / 2)
    if abs(offset) > 1:
        color = (COLOR_GREEN if abs(steering_deg) < 5
                 else COLOR_YELLOW if abs(steering_deg) < 15
                 else COLOR_RED)
        cv2.rectangle(img,
                      (min(cx, cx + offset), by),
                      (max(cx, cx + offset), by + bar_h),
                      color, -1)
