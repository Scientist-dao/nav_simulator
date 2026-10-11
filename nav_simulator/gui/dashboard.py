"""
dashboard.py - 单窗口综合仪表盘

把相机视图 / 俯视图 / 遥测数据 / 传感器状态 / 历史曲线组装到一张
画布, 单窗口呈现, 布局:

  ┌────────────────────────────────────────────────────────────┐
  │ 顶栏: 标题 | NAV 状态灯 | 控制模式 | 传感器模式 | 时间/帧    │
  ├─────────────┬─────────────┬────────────────────────────────┤
  │             │             │  遥测: CTE 大数字 / yaw / 转向  │
  │  CAMERA     │  TOP VIEW   │        速度表 / 姿态仪          │
  │  (缩放)     │  (深色重绘) │────────────────────────────────│
  │             │             │  传感器: RTK/IMU 灯 + 诊断      │
  ├─────────────┴─────────────┴────────────────────────────────┤
  │  曲线: CTE | YAW | QUALITY | SPEED (四联滚动图)             │
  └────────────────────────────────────────────────────────────┘

Dashboard 只负责渲染 (返回整幅画布), 窗口与键盘交互由 Visualizer
管理 (分层: 渲染归渲染, 交互归交互)。
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional

from ..core.simulator import FrameResult, NavSimulator, SENSOR_MODE_NAMES
from . import theme as T
from . import widgets as W
from .visualizer import TopViewRenderer


# ── 画布布局 (总尺寸 1520 x 1046) ──
_CANVAS_W, _CANVAS_H = 1520, 1046

_TOP_H = 46                       # 顶栏高
_MAIN_Y = 54                      # 主区起始 y
_MAIN_H = 734                     # 主区高 (相机/俯视/右栏)

_CAM_X, _CAM_W = 12, 524          # 相机面板
_TOPV_X, _TOPV_W = 544, 480       # 俯视面板
_SIDE_X, _SIDE_W = 1032, 476      # 右栏

_TELEM_H = 470                    # 遥测面板高 (v1.4: MAP/SIGMA 两行后
                                  # 376 装不下, QUALITY 条压进速度表;
                                  # 470 = 内容 372 + 仪表区 152)
_SENS_Y = _MAIN_Y + _TELEM_H + 8  # 传感器面板 y
_SENS_H = _MAIN_H - _TELEM_H - 8  # 传感器面板高 (不变 256)

_CHART_Y = _MAIN_Y + _MAIN_H + 8  # 曲线区 y
_CHART_H = 238
_CHART1_X, _CHART1_W = 12, 748
_CHART2_X, _CHART2_W = 768, 740

_HIST_MAX = 360                   # 曲线历史长度 (15fps 下 ~24s)

# RTK 定位状态文案
_POS_STATUS_TEXT = {0: "INVALID", 1: "SINGLE", 2: "SBAS", 4: "FIX", 5: "FLOAT"}


class Dashboard:
    """单窗口综合仪表盘渲染器"""

    def __init__(self, simulator: NavSimulator, scenario_name: str = ""):
        self._sim = simulator
        self._scenario = scenario_name
        # 画布尺寸 (Visualizer 创建窗口用)
        self.size = (_CANVAS_W, _CANVAS_H)

        # 俯视图渲染器 (适配面板内容区 464x692; 关闭内部 HUD,
        # 面板标题/姿态信息由本仪表盘统一提供)
        # scale=25 + 车后方留 4.3m → 车前方可见 ~21m, 覆盖 RowMap
        # 锚点红/黄区 (z=10~20m), σ 三色分级完整可见
        self._top = TopViewRenderer(width=464, height=692, scale=25.0,
                                    center_back_m=4.3)

        # 相机图像缩放 (656x800 -> 面板内容区)
        area_w = _CAM_W - 16
        area_h = _MAIN_H - W.TITLE_H - 10
        s = min(area_w / simulator.crop_w, area_h / simulator.crop_h)
        self._cam_size = (max(1, int(simulator.crop_w * s)),
                          max(1, int(simulator.crop_h * s)))

        # 曲线历史
        self._h_front = deque(maxlen=_HIST_MAX)
        self._h_rear = deque(maxlen=_HIST_MAX)
        self._h_yaw = deque(maxlen=_HIST_MAX)
        self._h_quality = deque(maxlen=_HIST_MAX)
        self._h_speed = deque(maxlen=_HIST_MAX)

        # 预建静态底 (背景 + 面板壳), 每帧只画动态内容
        self._base = self._build_base()

    # ════════════════════════════════════════════════════════
    #  静态骨架
    # ════════════════════════════════════════════════════════
    def _build_base(self) -> np.ndarray:
        img = np.full((_CANVAS_H, _CANVAS_W, 3), T.BG, dtype=np.uint8)
        # 顶栏底
        roi = img[0:_TOP_H, :]
        ov = roi.copy()
        ov[:] = T.PANEL
        cv2.addWeighted(ov, 0.85, roi, 0.15, 0, roi)
        cv2.line(img, (0, _TOP_H), (_CANVAS_W, _TOP_H), T.EDGE, 1, cv2.LINE_AA)

        W.panel(img, _CAM_X, _MAIN_Y, _CAM_W, _MAIN_H, title="CAMERA VIEW")
        W.panel(img, _TOPV_X, _MAIN_Y, _TOPV_W, _MAIN_H, title="TOP VIEW")
        W.panel(img, _SIDE_X, _MAIN_Y, _SIDE_W, _TELEM_H, title="TELEMETRY")
        W.panel(img, _SIDE_X, _SENS_Y, _SIDE_W, _SENS_H, title="SENSORS",
                accent=T.INFO)
        W.panel(img, _CHART1_X, _CHART_Y, _CHART1_W, _CHART_H,
                title="TRACKING", accent=T.INFO)
        W.panel(img, _CHART2_X, _CHART_Y, _CHART2_W, _CHART_H,
                title="PERFORMANCE", accent=T.INFO)
        return img

    # ════════════════════════════════════════════════════════
    #  主渲染
    # ════════════════════════════════════════════════════════
    def update(self, result: FrameResult) -> np.ndarray:
        """组装一帧仪表盘 (基于静态底拷贝, 返回整幅画布)"""
        img = self._base.copy()

        # 历史更新
        self._h_front.append(result.front_dist)
        self._h_rear.append(result.rear_dist)
        self._h_yaw.append(result.yaw_deg)
        self._h_quality.append(result.image_quality)
        self._h_speed.append(result.veh_speed)

        # 俯视图轨迹 (自车历史位置)
        self._top.add_trail_point(result.veh_x, result.veh_y)

        self._draw_topbar(img, result)
        self._draw_camera(img, result)
        self._draw_topview(img, result)
        self._draw_telemetry(img, result)
        self._draw_sensors(img, result)
        self._draw_charts(img)
        return img

    # ── 顶栏 ──
    def _draw_topbar(self, img, result):
        y_mid = _TOP_H // 2 + 5
        # 标题
        W.text(img, (14, y_mid), "AGV NAV SIMULATOR",
               color=T.ACCENT, scale=0.55, thick=2)
        if self._scenario:
            W.text(img, (250, y_mid), f"/ {self._scenario}",
                   color=T.TEXT_MID, scale=0.45, thick=1)

        # 键提示 (居中)
        W.text(img, (_CANVAS_W // 2 - 210, y_mid),
               "SPC pause | R reset | +/- speed | M mode | Q quit",
               color=T.TEXT_LOW, scale=0.38, thick=1)

        x = _CANVAS_W - 12
        # 帧号/时间 (最右)
        s_t = f"F{result.frame_idx}  {result.sim_time:7.1f}s"
        W.text_right(img, (x, y_mid), s_t, color=T.TEXT_MID,
                     scale=0.45, thick=1)
        x -= 150
        # 传感器模式徽章
        mode = result.sensor_mode
        mode_name = SENSOR_MODE_NAMES.get(mode, f"MODE {mode}")
        self._badge(img, x, y_mid, mode_name,
                    T.ACCENT if mode == 2 else T.WARN if mode == 1 else T.TEXT_MID,
                    right=True)
        x -= 150
        # 控制模式徽章
        cm = result.control_mode
        self._badge(img, x, y_mid, cm,
                    T.OK if cm == "WORK" else T.WARN, right=True)
        x -= 100
        # 扰动徽章 (P0 注入激活: 遮挡剩余/断垄/弱苗, 红色醒目)
        if result.perturb_tags:
            self._badge(img, x, y_mid, result.perturb_tags, T.ERR,
                        right=True)
            x -= 150
        # NAV 状态灯
        W.led(img, x, _TOP_H // 2, result.nav_valid,
              T.OK if result.nav_valid else T.ERR, r=6)
        W.text(img, (x - 78, y_mid), "NAV", color=T.TEXT_MID,
               scale=0.45, thick=1)

    @staticmethod
    def _badge(img, x_right, y_mid, label, color, right=False):
        """小徽章: 描边圆角块 + 文字; right=True 时 x_right 为右边界"""
        (tw, th), _ = cv2.getTextSize(label, T.FONT, 0.42, 1)
        pad = 8
        bw, bh = tw + pad * 2, th + 8
        if right:
            x0 = x_right - bw
        else:
            x0 = x_right
        y0 = y_mid - bh // 2 - 2
        # 底色 (半透明)
        roi = img[y0:y0 + bh, x0:x0 + bw]
        if roi.size:
            ov = roi.copy()
            ov[:] = color
            cv2.addWeighted(ov, 0.18, roi, 0.82, 0, roi)
        W.rounded_outline(img, x0, y0, bw, bh, color, r=6)
        cv2.putText(img, label, (x0 + pad, y_mid + th // 2 - 1), T.FONT,
                    0.42, color, 1, cv2.LINE_AA)

    # ── 相机视图 ──
    def _draw_camera(self, img, result):
        cam = result.mask_image
        if cam is None:
            return
        cw, ch = self._cam_size
        cam = cv2.resize(cam, (cw, ch), interpolation=cv2.INTER_AREA)
        x0 = _CAM_X + (_CAM_W - cw) // 2
        y0 = _MAIN_Y + W.TITLE_H + 5
        img[y0:y0 + ch, x0:x0 + cw] = cam

    # ── 俯视图 ──
    def _draw_topview(self, img, result):
        # RowMap 后端: 行地图可视化数据 (基线后端无此接口, 不画)
        vn = getattr(self._sim, '_vn', None)
        map_overlay = (vn.get_map_overlay()
                       if hasattr(vn, 'get_map_overlay') else None)
        top_img = self._top.render(
            self._sim.field_model,
            result.veh_x, result.veh_y,
            np.radians(result.veh_heading_deg),
            np.radians(result.veh_steering_deg),
            result.veh_speed, result.nav_line_world,
            np.radians(result.body_roll_deg),
            np.radians(result.body_pitch_deg),
            show_hud=False,
            perturbation=getattr(self._sim, 'perturbation', None),
            occluded=result.occluded,
            map_overlay=map_overlay)
        h, w = top_img.shape[:2]
        x0 = _TOPV_X + (_TOPV_W - w) // 2
        y0 = _MAIN_Y + W.TITLE_H + 5
        img[y0:y0 + h, x0:x0 + w] = top_img

    # ── 遥测面板 ──
    def _draw_telemetry(self, img, result):
        x = _SIDE_X + T.PAD
        w = _SIDE_W - T.PAD * 2
        right = x + w
        y = _MAIN_Y + W.TITLE_H + 10

        # ── CTE 大数字 (核心指标) ──
        fd = result.front_dist
        fd_color = (T.OK if abs(fd) < 0.05 else
                    T.WARN if abs(fd) < 0.15 else T.ERR)
        cv2.putText(img, "FRONT DIST", (x, y + 4), T.FONT,
                    T.SCALE_LABEL, T.TEXT_MID, 1, cv2.LINE_AA)
        W.text_right(img, (right, y + 26), f"{fd:+.4f}",
                     color=fd_color, scale=T.SCALE_BIG, thick=2)
        cv2.putText(img, "m", (x, y + 26), T.FONT, T.SCALE_LABEL,
                    T.TEXT_LOW, 1, cv2.LINE_AA)
        y += 46

        # ── 数据行 ──
        yaw_col = (T.OK if abs(result.yaw_deg) < 3 else
                   T.WARN if abs(result.yaw_deg) < 8 else T.ERR)
        y = W.key_value(img, x, y, "YAW", f"{result.yaw_deg:+.3f}",
                        "deg", yaw_col)
        y = W.key_value(img, x, y, "REAR", f"{result.rear_dist:+.4f}", "m")
        y = W.key_value(img, x, y, "MASKS", f"{result.num_masks}", "")
        # 行检出统计: 检出/画面可见 (差值即漏检行数)
        det, exp = result.rows_detected, result.rows_expected
        y = W.key_value(img, x, y, "ROWS", f"{det}/{exp}", "det/view",
                        value_color=(T.OK if det >= exp else T.WARN))
        # 行地图状态 (RowMap 后端): 行数/任务行锚点数/最小 σ (置信度)
        vn = getattr(self._sim, '_vn', None)
        if hasattr(vn, 'map_stats'):
            ms = vn.map_stats()
            y = W.key_value(img, x, y, "MAP",
                            f"{ms['n_rows']}/{ms['task_anchors']}",
                            "rows/anc",
                            value_color=(T.OK if ms['task_row_id'] is not None
                                         else T.WARN))
            sig_min = ms.get('task_var_min', float('nan'))
            if sig_min == sig_min:
                y = W.key_value(img, x, y, "SIGMA",
                                f"{sig_min ** 0.5 * 100:.2f}", "cm min")
        y += 4

        # ── 转向角 (中心对称条) ──
        cv2.putText(img, "STEERING", (x, y + 4), T.FONT, T.SCALE_LABEL,
                    T.TEXT_MID, 1, cv2.LINE_AA)
        sd = result.veh_steering_deg
        W.text_right(img, (right, y + 4), f"{sd:+.2f} deg",
                     color=(T.OK if abs(sd) < 5 else
                            T.WARN if abs(sd) < 15 else T.ERR),
                     scale=T.SCALE_VAL, thick=1)
        W.centered_bar(img, x + w // 2, y + 10, w - 8, 8, sd / 30.0)
        y += 30

        # ── 图像质量条 ──
        q = result.image_quality
        cv2.putText(img, "QUALITY", (x, y + 4), T.FONT, T.SCALE_LABEL,
                    T.TEXT_MID, 1, cv2.LINE_AA)
        W.text_right(img, (right, y + 4), f"{q}",
                     color=(T.OK if q > 70 else T.WARN if q > 40 else T.ERR),
                     scale=T.SCALE_VAL, thick=1)
        W.hbar(img, x, y + 10, w, 8, q / 100.0,
               color=T.OK if q > 70 else T.WARN if q > 40 else T.ERR)
        y += 26

        # ── 速度表 + 姿态仪 ──
        gauge_r = 58
        gx = x + gauge_r + 10
        gy = _MAIN_Y + _TELEM_H - gauge_r - 26
        W.gauge(img, gx, gy, gauge_r, result.veh_speed, 3.0,
                label="SPEED", unit="m/s", color=T.ACCENT)

        ax = x + w - 52
        ay = gy + 2
        W.attitude_gauge(img, ax, ay, 46,
                         result.body_roll_deg, result.body_pitch_deg)

    # ── 传感器面板 ──
    def _draw_sensors(self, img, result):
        x = _SIDE_X + T.PAD
        w = _SIDE_W - T.PAD * 2
        right = x + w
        y = _SENS_Y + W.TITLE_H + 6

        imu_on = result.imu_enabled
        rtk_on = result.rtk_enabled
        any_sensor = imu_on or rtk_on

        # ── 模式行: 当前 mode 徽章 + M 键提示 ──
        mode = result.sensor_mode
        mode_name = SENSOR_MODE_NAMES.get(mode, f"MODE {mode}")
        self._badge(img, x + 118, y - 4, mode_name,
                    T.ACCENT if mode == 2 else T.WARN if mode == 1 else T.TEXT_MID)
        cv2.putText(img, "SENSOR MODE", (x, y - 4), T.FONT, T.SCALE_LABEL,
                    T.TEXT_MID, 1, cv2.LINE_AA)
        cv2.putText(img, "[M]", (right - 22, y - 4), T.FONT,
                    T.SCALE_SMALL, T.TEXT_LOW, 1, cv2.LINE_AA)
        y += 24

        if not any_sensor:
            # 未启用: 简洁空态提示
            cv2.putText(img, "RTK / IMU simulation off", (x, y + 4),
                        T.FONT, T.SCALE_LABEL, T.TEXT_LOW, 1, cv2.LINE_AA)
            cv2.putText(img, "run sensor scenarios (see --list)",
                        (x, y + 24), T.FONT, T.SCALE_SMALL, T.TEXT_LOW,
                        1, cv2.LINE_AA)
            return

        # ── RTK 行 ──
        st_txt = _POS_STATUS_TEXT.get(result.rtk_pos_status,
                                      f"ST{result.rtk_pos_status}")
        rtk_ok = rtk_on and result.rtk_pos_status == 4
        W.led(img, x + 6, y, rtk_ok, T.OK if rtk_on else T.TEXT_LOW)
        cv2.putText(img, "RTK", (x + 20, y + 5), T.FONT, T.SCALE_LABEL,
                    T.TEXT_MID, 1, cv2.LINE_AA)
        W.text_right(img, (right, y + 5), st_txt,
                     color=(T.OK if rtk_ok else
                            T.WARN if rtk_on else T.TEXT_LOW),
                     scale=T.SCALE_LABEL, thick=1)
        # ENU hint 灯 (mode 2 生效标志)
        W.led(img, x + 90, y, result.rtk_hint_valid, T.ACCENT, r=4)
        cv2.putText(img, "ENU", (x + 100, y + 5), T.FONT, T.SCALE_SMALL,
                    T.TEXT_MID, 1, cv2.LINE_AA)
        y += 22

        # ── IMU 行 ──
        W.led(img, x + 6, y, result.imu_applied,
              T.OK if imu_on else T.TEXT_LOW)
        cv2.putText(img, "IMU", (x + 20, y + 5), T.FONT, T.SCALE_LABEL,
                    T.TEXT_MID, 1, cv2.LINE_AA)
        if result.imu_applied:
            ap = (f"applied  R{result.imu_roll_deg:+.2f} "
                  f"P{result.imu_pitch_deg:+.2f} deg")
        elif imu_on:
            ap = "no compensation (stale/timeout)"
        else:
            ap = "off"
        W.text_right(img, (right, y + 5), ap,
                     color=T.OK if result.imu_applied else
                     (T.WARN if imu_on else T.TEXT_LOW),
                     scale=T.SCALE_LABEL, thick=1)
        y += 22

        # ── 真值姿态 vs 测量 ──
        tr = f"body  R{result.body_roll_deg:+.2f} P{result.body_pitch_deg:+.2f}"
        W.text_right(img, (right, y + 5), tr,
                     color=T.TEXT_MID, scale=T.SCALE_LABEL, thick=1)
        cv2.putText(img, "TRUTH", (x, y + 5), T.FONT, T.SCALE_SMALL,
                    T.TEXT_LOW, 1, cv2.LINE_AA)
        y += 24
        W.separator(img, x, y, w)
        y += 16

        # ── 链路统计 ──
        sent = result.rtk_packets_total
        drop = self._sim.sensor_sim.packets_dropped
        rate = self._sim.sensor_sim.rtk_cfg.rate_hz if rtk_on else 0.0
        cv2.putText(img, "LINK", (x, y + 5), T.FONT, T.SCALE_SMALL,
                    T.TEXT_LOW, 1, cv2.LINE_AA)
        link = (f"{rate:.0f} Hz  |  rx {result.rtk_packets_frame}/f"
                f"  |  tot {sent}  |  drop {drop}")
        W.text_right(img, (right, y + 5), link,
                     color=T.TEXT_MID, scale=T.SCALE_LABEL, thick=1)
        y += 20

        # ── 位置 (经纬度) ──
        rtk_cfg = self._sim.sensor_sim.rtk_cfg
        lat = rtk_cfg.origin_lat_deg + result.veh_y / 111320.0
        lon = (rtk_cfg.origin_lon_deg
               + result.veh_x / (111320.0 * np.cos(np.radians(rtk_cfg.origin_lat_deg))))
        cv2.putText(img, "POS", (x, y + 5), T.FONT, T.SCALE_SMALL,
                    T.TEXT_LOW, 1, cv2.LINE_AA)
        W.text_right(img, (right, y + 5),
                     f"{lat:.7f} N  {lon:.7f} E",
                     color=T.TEXT_MID, scale=T.SCALE_LABEL, thick=1)
        y += 20

        # ── 扰动源状态 (姿态进投影 / 颠簸强度) ──
        cfg = self._sim.sim_config
        proj = "ON" if getattr(cfg, 'attitude_in_projection', False) else "OFF"
        bump = getattr(cfg, 'bump_intensity', 1.0)
        cv2.putText(img, "DIST", (x, y + 5), T.FONT, T.SCALE_SMALL,
                    T.TEXT_LOW, 1, cv2.LINE_AA)
        W.text_right(img, (right, y + 5),
                     f"cam-attitude {proj}  |  bump x{bump:.1f}",
                     color=(T.WARN if proj == "ON" else T.TEXT_MID),
                     scale=T.SCALE_LABEL, thick=1)

    # ── 曲线区 ──
    def _draw_charts(self, img):
        y = _CHART_Y + W.TITLE_H + 4
        h = _CHART_H - W.TITLE_H - 12
        # 面板1: CTE (左) + YAW (右)
        gap = 12
        cw1 = (_CHART1_W - 24 - gap) // 2
        W.line_chart(img, _CHART1_X + 12, y, cw1, h,
                     [("front", self._h_front, T.ACCENT),
                      ("rear", self._h_rear, T.INFO)])
        W.line_chart(img, _CHART1_X + 12 + cw1 + gap, y, cw1, h,
                     [("yaw", self._h_yaw, T.OK)],
                     show_zero=True)
        # 面板2: QUALITY (左) + SPEED (右)
        cw2 = (_CHART2_W - 24 - gap) // 2
        W.line_chart(img, _CHART2_X + 12, y, cw2, h,
                     [("quality", self._h_quality, T.WARN)],
                     y_range=(0, 100), show_zero=False)
        W.line_chart(img, _CHART2_X + 12 + cw2 + gap, y, cw2, h,
                     [("speed", self._h_speed, T.VIOLET)],
                     show_zero=False)

    # ════════════════════════════════════════════════════════
    #  交互辅助
    # ════════════════════════════════════════════════════════
    def clear_history(self):
        """仿真重置时清空曲线历史与轨迹"""
        self._h_front.clear()
        self._h_rear.clear()
        self._h_yaw.clear()
        self._h_quality.clear()
        self._h_speed.clear()
        self._top.clear_trail()
