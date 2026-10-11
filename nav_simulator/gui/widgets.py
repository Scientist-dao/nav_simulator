"""
widgets.py - GUI 基础控件库

在 OpenCV 图像上原位绘制的仪表盘控件集合, 配色/字号统一取自 theme.py。
所有绘制函数第一个参数为目标图像 (原地绘制, 避免高频拷贝)。

控件清单:
  rounded_rect / panel      圆角面板 (半透明底 + 标题 + 强调线)
  led                       状态指示灯 (光晕)
  text / key_value          文字 / 数据行 (标签+值+单位)
  hbar / centered_bar       水平条 / 中心对称条 (转向角)
  gauge                     半圆仪表 (速度表)
  attitude_gauge            姿态仪 (人工地平线, roll/pitch)
  line_chart                滚动曲线图 (多系列 + 自适应量程)
  separator                 分隔线
"""

import cv2
import numpy as np

from . import theme as T


# ══════════════════════════════════════════════════════════════
#  基础图形
# ══════════════════════════════════════════════════════════════

def rounded_rect(img, x, y, w, h, color, r=T.RADIUS):
    """填充圆角矩形 (矩形 + 四角圆)"""
    r = max(1, min(r, w // 2, h // 2))
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r),
                   (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, color, -1)


def rounded_outline(img, x, y, w, h, color, r=T.RADIUS, thick=1):
    """圆角矩形描边"""
    r = max(1, min(r, w // 2, h // 2))
    cv2.line(img, (x + r, y), (x + w - r, y), color, thick, cv2.LINE_AA)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thick, cv2.LINE_AA)
    cv2.line(img, (x, y + r), (x, y + h - r), color, thick, cv2.LINE_AA)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thick, cv2.LINE_AA)
    for cx, cy, a0, a1 in ((x + r, y + r, 180, 270),
                           (x + w - r, y + r, 270, 360),
                           (x + w - r, y + h - r, 0, 90),
                           (x + r, y + h - r, 90, 180)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a1, color, thick, cv2.LINE_AA)


def separator(img, x, y, w, color=T.EDGE):
    """水平分隔线"""
    cv2.line(img, (x, y), (x + w, y), color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
#  面板
# ══════════════════════════════════════════════════════════════

# 面板标题条高度
TITLE_H = 30


def panel(img, x, y, w, h, title=None, accent=None, alpha=0.94):
    """半透明圆角面板; 返回内容区左上角 (cx, cy)

    title:   左上标题 (ASCII); None 时无标题条
    accent:  标题左侧强调竖线颜色 (默认青色; None 不画)
    """
    # 半透明底 (ROI 加权, 不影响面板外像素)
    roi = img[y:y + h, x:x + w]
    if roi.size == 0:
        return (x + T.PAD, y + T.PAD)
    ov = roi.copy()
    rounded_rect(ov, 0, 0, w, h, T.PANEL)
    cv2.addWeighted(ov, alpha, roi, 1.0 - alpha, 0, roi)
    rounded_outline(img, x, y, w, h, T.EDGE)

    if title is not None:
        ty = y + 20
        if accent is None:
            accent = T.ACCENT
        cv2.line(img, (x + 12, y + 10), (x + 12, y + TITLE_H - 10),
                 accent, 2, cv2.LINE_AA)
        cv2.putText(img, title, (x + 22, ty), T.FONT, T.SCALE_TITLE * 0.75,
                    T.TEXT_MID, 1, cv2.LINE_AA)
        return (x + T.PAD, y + TITLE_H + 4)
    return (x + T.PAD, y + T.PAD)


# ══════════════════════════════════════════════════════════════
#  文字
# ══════════════════════════════════════════════════════════════

def text(img, xy, s, color=T.TEXT_HI, scale=T.SCALE_VAL, thick=1,
         shadow=True, align_right=False):
    """带阴影文字 (深背景上更清晰); align_right 时 xy 为右下角"""
    x, y = xy
    if align_right:
        (tw, _), _ = cv2.getTextSize(s, T.FONT, scale, thick)
        x -= tw
    if shadow:
        cv2.putText(img, s, (x + 1, y + 1), T.FONT, scale,
                    (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), T.FONT, scale, color, thick, cv2.LINE_AA)


def text_right(img, xy, s, **kw):
    kw.setdefault('align_right', True)
    text(img, xy, s, **kw)


def key_value(img, x, y, label, value, unit="", value_color=None,
              label_w=88, value_w=110):
    """数据行: 标签(弱) + 值(强, 右对齐) + 单位(弱); 返回下一行 y"""
    cv2.putText(img, label, (x, y), T.FONT, T.SCALE_LABEL, T.TEXT_MID,
                1, cv2.LINE_AA)
    text_right(img, (x + label_w + value_w, y), value,
               color=value_color or T.TEXT_HI, scale=T.SCALE_VAL, thick=1)
    if unit:
        cv2.putText(img, unit, (x + label_w + value_w + 6, y), T.FONT,
                    T.SCALE_SMALL, T.TEXT_LOW, 1, cv2.LINE_AA)
    return y + T.LINE_H


# ══════════════════════════════════════════════════════════════
#  状态灯 / 条
# ══════════════════════════════════════════════════════════════

def led(img, cx, cy, on, color=T.OK, r=5, label=None, label_color=None):
    """状态指示灯: on=点亮(实心+外圈+高光) / off=暗圈; label 画在右侧"""
    if on:
        cv2.circle(img, (cx, cy), r + 3, color, 1, cv2.LINE_AA)   # 光晕圈
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)      # 实心
        cv2.circle(img, (cx - r // 3, cy - r // 3), max(1, r // 3),
                   (255, 255, 255), -1, cv2.LINE_AA)              # 高光点
    else:
        cv2.circle(img, (cx, cy), r, T.TEXT_LOW, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), max(1, r - 3), T.TRACK, -1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (cx + r + 8, cy + 5), T.FONT,
                    T.SCALE_LABEL, label_color or T.TEXT_MID, 1, cv2.LINE_AA)


def hbar(img, x, y, w, h, ratio, color=T.OK, bg_color=T.TRACK):
    """水平填充条 (ratio 0~1)"""
    ratio = float(np.clip(ratio, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    if ratio > 0:
        cv2.rectangle(img, (x, y), (x + int(w * ratio), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), T.EDGE, 1)


def centered_bar(img, cx, y, w, h, ratio_signed, color_pos=T.OK,
                 color_neg=T.WARN):
    """中心对称条 (ratio_signed -1~1, 左负右正); 中心刻度线"""
    ratio_signed = float(np.clip(ratio_signed, -1.0, 1.0))
    cv2.rectangle(img, (cx - w // 2, y), (cx + w // 2, y + h), T.TRACK, -1)
    if ratio_signed >= 0:
        x0 = cx
        x1 = cx + int(w / 2 * ratio_signed)
        color = color_pos
    else:
        x0 = cx + int(w / 2 * ratio_signed)
        x1 = cx
        color = color_neg
    if x1 > x0:
        cv2.rectangle(img, (x0, y), (x1, y + h), color, -1)
    cv2.line(img, (cx, y - 2), (cx, y + h + 2), T.TEXT_MID, 1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - w // 2, y), (cx + w // 2, y + h), T.EDGE, 1)


# ══════════════════════════════════════════════════════════════
#  仪表
# ══════════════════════════════════════════════════════════════

# 仪表弧角 (OpenCV 角度制, 顺时针, 0=x 正轴): 210°→330° 开口朝下
_GAUGE_A0, _GAUGE_A1 = 210.0, 330.0


def gauge(img, cx, cy, r, value, vmax, label="", unit="",
          color=T.ACCENT, minor_ticks=5):
    """半圆仪表 (速度表样式): 轨道 + 值弧 + 刻度 + 中心大数字"""
    vmax = max(abs(float(vmax)), 1e-6)
    ratio = float(np.clip(value / vmax, 0.0, 1.0))

    # 轨道 + 值弧
    thick = max(6, r // 8)
    cv2.ellipse(img, (cx, cy), (r, r), 0, _GAUGE_A0, _GAUGE_A1,
                T.TRACK, thick, cv2.LINE_AA)
    if ratio > 0.003:
        cv2.ellipse(img, (cx, cy), (r, r), 0, _GAUGE_A0,
                    _GAUGE_A0 + (_GAUGE_A1 - _GAUGE_A0) * ratio,
                    color, thick, cv2.LINE_AA)

    # 刻度 (主刻度带数字)
    rt = r - thick // 2 - 4
    for i in range(minor_ticks + 1):
        a = np.radians(_GAUGE_A0 + (_GAUGE_A1 - _GAUGE_A0) * i / minor_ticks)
        c, s = np.cos(a), np.sin(a)
        p1 = (int(cx + c * (rt - 4)), int(cy + s * (rt - 4)))
        p2 = (int(cx + c * rt), int(cy + s * rt))
        cv2.line(img, p1, p2, T.TEXT_LOW, 1, cv2.LINE_AA)
        if vmax >= 10 or i % 2 == 0:
            lbl = f"{vmax * i / minor_ticks:.0f}"
            pl = (int(cx + c * (rt - 16)), int(cy + s * (rt - 16)))
            cv2.putText(img, lbl, pl, T.FONT, T.SCALE_SMALL,
                        T.TEXT_LOW, 1, cv2.LINE_AA)

    # 中心数值 + 单位 + 底部标签 (居中)
    s_val = f"{value:.1f}" if value < 10 else f"{value:.0f}"
    (tw, _), _ = cv2.getTextSize(s_val, T.FONT, T.SCALE_BIG, 2)
    text(img, (cx - tw // 2, cy - 4), s_val,
         color=T.TEXT_HI, scale=T.SCALE_BIG, thick=2)
    if unit:
        (uw, _), _ = cv2.getTextSize(unit, T.FONT, T.SCALE_SMALL, 1)
        cv2.putText(img, unit, (cx - uw // 2, cy + 14), T.FONT,
                    T.SCALE_SMALL, T.TEXT_MID, 1, cv2.LINE_AA)
    if label:
        (lw, _), _ = cv2.getTextSize(label, T.FONT, T.SCALE_LABEL, 1)
        cv2.putText(img, label, (cx - lw // 2, cy + r - 4), T.FONT,
                    T.SCALE_LABEL, T.TEXT_MID, 1, cv2.LINE_AA)


def attitude_gauge(img, cx, cy, r, roll_deg, pitch_deg, gain=4.0,
                   labels=True):
    """姿态仪 (人工地平线): roll 旋转 / pitch 平移 (视觉增益 gain)

    物理姿态通常 <1°, gain 放大后仍可感知; 圆形窗口内:
    上半天空 / 下半地面, 白色地平线随姿态旋转平移, 黄色翼标记固定。
    """
    if r < 8 or cx - r < 0 or cy - r < 0 \
            or cy + r > img.shape[0] or cx + r > img.shape[1]:
        return
    ang = np.radians(roll_deg * gain)
    dy = float(np.clip(pitch_deg * gain * 2.0, -r * 0.6, r * 0.6))
    c, s = np.cos(ang), np.sin(ang)
    L = r * 1.7

    # ROI 内构造天地画面, 圆掩膜裁剪后回贴
    x0, y0 = cx - r, cy - r
    roi = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
    if roi.size == 0:
        return
    tmp = np.full_like(roi, (60, 80, 45))          # 地面棕 (BGR)
    # 天空半平面: 地平线过 P=(r, r-dy), 方向 D=(c,s), 上侧 U=(s,-c)
    P = np.array([r, r - dy], dtype=float)
    D = np.array([c, s], dtype=float)
    U = np.array([s, -c], dtype=float)
    sky_poly = np.array([P + D * L, P - D * L,
                         P - D * L + U * L, P + D * L + U * L],
                        dtype=np.int32)
    cv2.fillPoly(tmp, [sky_poly], (130, 100, 48))  # 天空蓝 (BGR)
    mask = np.zeros((2 * r, 2 * r), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)
    tmp = cv2.bitwise_and(tmp, tmp, mask=mask)
    cv2.addWeighted(tmp, 0.85, roi, 0.15, 0, roi)

    # 地平线 (裁剪到圆内)
    p1 = (int(cx - L * c), int(cy - dy + L * s))
    p2 = (int(cx + L * c), int(cy - dy - L * s))
    clipped, q1, q2 = cv2.clipLine(
        (cx - r, cy - r, 2 * r, 2 * r), p1, p2)
    if clipped:
        cv2.line(img, q1, q2, T.TEXT_HI, 1, cv2.LINE_AA)

    # 中心固定翼标记 (不随姿态动)
    cv2.line(img, (cx - r + 8, cy), (cx - 10, cy), T.WARN, 2, cv2.LINE_AA)
    cv2.line(img, (cx + 10, cy), (cx + r - 8, cy), T.WARN, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, T.WARN, -1, cv2.LINE_AA)
    # 外圈 + roll 中心刻度
    cv2.circle(img, (cx, cy), r, T.EDGE, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - r - 2), (cx, cy - r + 4), T.TEXT_MID, 1, cv2.LINE_AA)
    if labels:
        cv2.putText(img, f"R{roll_deg:+.1f}", (cx - r + 4, cy + r + 14),
                    T.FONT, T.SCALE_SMALL, T.TEXT_MID, 1, cv2.LINE_AA)
        cv2.putText(img, f"P{pitch_deg:+.1f}", (cx + 6, cy + r + 14),
                    T.FONT, T.SCALE_SMALL, T.TEXT_MID, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
#  曲线图
# ══════════════════════════════════════════════════════════════

def line_chart(img, x, y, w, h, series, y_range=None, show_zero=True,
               max_points=360, label_h=16):
    """滚动曲线图 (多系列)

    series:    [(label, data_list, color), ...]
    y_range:   (ymin, ymax) 固定; None 自适应 (含零对称)
    max_points: 仅渲染每个序列尾部这么多点
    """
    # 绘图区 (顶部留一行图例行)
    plot_y = y + label_h
    plot_h = h - label_h
    if plot_h < 10 or w < 10:
        return
    cv2.rectangle(img, (x, y), (x + w, y + h), T.PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), T.EDGE, 1)

    # 数据尾部裁剪 + 量程 (先转 list: 调用方可能传 deque, 不支持切片)
    data_all = []
    for lbl, data, color in series:
        d = list(data)[-max_points:] if data else []
        data_all.append(d)
    if y_range is not None:
        ymin, ymax = float(y_range[0]), float(y_range[1])
    else:
        vmax = 1e-6
        for d in data_all:
            for v in d:
                vmax = max(vmax, abs(v))
        if show_zero:
            ymin, ymax = -vmax * 1.15, vmax * 1.15
        else:
            lo = min((min(d) for d in data_all if d), default=0.0)
            hi = max((max(d) for d in data_all if d), default=1.0)
            ymin = lo - (hi - lo) * 0.15 - 1e-6
            ymax = hi + (hi - lo) * 0.15 + 1e-6
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0

    def vy(v):
        return int(plot_y + plot_h * (1.0 - (v - ymin) / (ymax - ymin)))

    # 网格 (3 横线) + 量程标注
    for k in (1, 2):
        gy = plot_y + plot_h * k / 3.0
        cv2.line(img, (x, int(gy)), (x + w, int(gy)), T.GRID, 1)
    cv2.putText(img, f"{ymax:.2f}", (x + 3, plot_y + 10), T.FONT,
                T.SCALE_SMALL * 0.9, T.TEXT_LOW, 1, cv2.LINE_AA)
    cv2.putText(img, f"{ymin:.2f}", (x + 3, y + h - 4), T.FONT,
                T.SCALE_SMALL * 0.9, T.TEXT_LOW, 1, cv2.LINE_AA)
    # 零线
    if show_zero and ymin < 0 < ymax:
        zy = vy(0.0)
        cv2.line(img, (x, zy), (x + w, zy), T.TEXT_LOW, 1, cv2.LINE_AA)

    # 系列
    lx = x + 6
    for i, (lbl, data, color) in enumerate(series):
        d = data_all[i]
        if len(d) >= 2:
            n = len(d)
            pts = np.empty((n, 2), dtype=np.int32)
            for j, v in enumerate(d):
                pts[j, 0] = x + int(w * j / (n - 1))
                pts[j, 1] = vy(v)
            cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
        # 图例色块 + 名称 + 最新值
        cv2.line(img, (lx, y + 9), (lx + 14, y + 9), color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{lbl}", (lx + 18, y + 13), T.FONT,
                    T.SCALE_SMALL, T.TEXT_MID, 1, cv2.LINE_AA)
        if d:
            last = f"{d[-1]:+.3f}"
            text_right(img, (lx + 96, y + 13), last, color=color,
                       scale=T.SCALE_SMALL, thick=1)
        lx += 108
