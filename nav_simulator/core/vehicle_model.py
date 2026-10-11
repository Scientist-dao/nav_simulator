"""
vehicle_model.py - 车辆运动学模型

模拟农机在田间的运动。使用运动学自行车模型，根据导航算法输出的
yaw 角和横向偏差 (CTE) 计算转向指令，更新车辆位置和朝向。

坐标系约定:
  - 全局坐标系: X=东, Y=北
  - 车辆朝向: heading = 0 表示朝北 (Y+), 顺时针为正 (罗盘角;
    右转 steering>0 → heading 增大, 与 C++ nav_geo.hpp 一致)
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


class ControlMode(Enum):
    """整机控制状态机

    WORK:      作业循行 (闭环跟行), 内含直/弯子状态
    HD_*:      地头转向程序 (梨形掉头, 开环时序/航向累积驱动)
    """
    WORK = 'WORK'
    HD_RUNOUT = 'RUNOUT'     # 直行越出行尾
    HD_ARC1 = 'TURN_ARC1'    # 满舵弧 (转 225°)
    HD_JOG = 'TURN_JOG'      # 斜行消横向偏移
    HD_ARC2 = 'TURN_ARC2'    # 回正弧 (45°)
    HD_CAPTURE = 'CAPTURE'   # 直行重新捕获行


@dataclass
class VehicleState:
    """车辆状态"""
    x: float = 0.0          # 全局 X (东, m)
    y: float = 0.0          # 全局 Y (北, m)
    heading: float = 0.0    # 朝向角 (rad, 0=北, 顺时针正, 罗盘角)
    speed: float = 1.0      # 速度 (m/s)
    steering: float = 0.0   # 前轮转角 (rad, 右正)
    roll: float = 0.0       # 车身侧倾角 (rad, 右倾正)
    pitch: float = 0.0      # 车身俯仰角 (rad, 抬头正)


@dataclass
class VehicleParams:
    """车辆参数"""
    wheelbase_length: float = 2.7     # 轴距 (m)
    wheelbase_width: float = 1.72     # 轮距 (m)
    max_steering_angle: float = 0.6   # 最大前轮转角 (rad, ~35°)
    max_steering_rate: float = 0.3    # 最大转向速率 (rad/s)
    default_speed: float = 1.0        # 默认速度 (m/s)


class VehicleModel:
    """
    运动学自行车模型

    根据导航算法输出的 yaw (航向角偏差) 和 CTE (横向偏差)，
    使用简化的 PD 控制器计算转向指令，驱动车辆跟踪作物行。
    """

    def __init__(self, params: VehicleParams = None):
        self.params = params or VehicleParams()
        self.state = VehicleState(speed=self.params.default_speed)
        # 控制器增益 (导航输出 → 转向指令)
        # δ = -atan(k_cte·front/v) - kd_cte·d(front)/dt - w·k_yaw·yaw + ...
        # 主项只用 front_dist (前轴到导航线的垂距, 右偏为正 → 负转向纠偏):
        #   - front/rear 在弯道上可能异号 (各锁不同参考), 取平均会稀释偏差
        # yaw 航向阻尼经软门控 w = 1-|yaw|/yaw_gate 启用:
        #   - C++ yaw 语义 = 车头相对导航线的偏角 (罗盘系, 偏左为负,
        #     与 nav_geo.hpp/标准投影一致): 偏左 → yaw<0 → 需右转(正舵),
        #     故取负号 -k_yaw·yaw, 提供姿态负反馈阻尼;
        #   - 弯道上 yaw 经内部 EMA/holdover 滤波漂移(|yaw| 达几十度),
        #     门控自动退出, 避免与 CTE 项对消
        # k_ff: 弯道曲率前馈 (导航线弦方向角 φ_line, 指示前方曲率,
        #   补偿纯 CTE 反馈在弯道上的转向不足; 低通滤波抑制像素噪声)
        # k_cte 扫参实测 (2026-08, sensor_full_stack 0.25m 捕获):
        #   2.0: 峰值 yaw 10.2°/过冲 0.18m/敛定 15.3s (mode2 EMA+限速器
        #        交互下欠阻尼, 长时间小摆);
        #   1.4: 峰值 yaw 7.5°/过冲 0.10m/敛定 6.4s, 弯道跟踪无回退
        #        (积分前馈兑底稳态误差, 尾段 CTE 1.6mm vs 1.9mm)
        self.k_cte = 1.4        # 横向偏差增益 (归一化到 atan)
        self.kd_cte = 0.5       # 横向偏差微分阻尼
        self.k_yaw = 1.5        # 航向阻尼增益 (软门控内)
        self.yaw_gate_deg = 15.0  # yaw 阻尼退出门限 (度)
        # 低速增益调度 (见 _work_control 注释)
        self.vp_floor = 0.3       # P/D 项速度调度的 v 下限 (m/s)
        self.d_speed_boost = False  # (已废弃) 旧 D 项低速补偿开关,
                                    # 由 D 项 ÷max(v, vp_floor) 取代
        self.k_ff = 0.0         # 曲率前馈增益 (0=关闭; 归一化形式
                                #   atan(L·tan φ/z_mid), 未归一化形式已证伪)
        self._line_phi_f = 0.0  # 弦方向角低通状态 (度)
        self._prev_yaw = 0.0    # 上一帧 yaw
        self._prev_cte = 0.0    # 上一帧前轴 CTE

        # ── 弯道积分前馈 (WORK_CURVE 子状态启用) ──
        # 感知输出的弦参考特性使显式曲率前馈不可用 (实验证伪),
        # 积分器作为等效的自适应前馈: 弯道上累积消除转向不足,
        # 直线子状态下衰减清零, 避免直道漂移
        self.k_i = 0.1              # 积分增益
        self.i_limit = 0.15         # 积分项限幅 (rad)
        self._i_front = 0.0         # ∫front·dt 状态
        # 直/弯子状态滞回门限 (按滤波后的导航线弦向角 |φ_f|)
        self.curve_enter_deg = 10.0
        self.curve_exit_deg = 5.0
        self._in_curve = False
        # 弯道子状态: front 反馈项限幅 (rad, 0=关闭)。实验结论:
        # 限幅 3° 使 curved_sharp valid 74%→61% (收敛变慢反而更早
        # 丢行), 默认关闭; 保留接口供后续研究
        self.cte_limit_curve = 0.0


        # ── 地头转向状态机 ──
        self.mode = ControlMode.WORK
        self.headland_trigger_frames = 25    # 连续丢行触发帧数
        self.headland_steer_gate = np.radians(8.0)  # 丢行起点舵角门限
        self.turn_steering = 0.55            # 掉头弧舵角 (rad)
        self.capture_valid_frames = 5        # 捕获确认帧数
        # 掉头方向判定门限 (m): 丢行前 |cte| 超过此值才改向, 否则
        # 保持传统左转 (行尾对称区左右等价; 门限防 cte 噪声抖向)
        self.headland_dir_gate = 0.15
        # 地头限速 (农机地头减速惯例): 掉头/捕获段实际速度上限。
        # 高速作业时不限速则转弯半径×帧位移巨大, ARC 拖尾、CAPTURE
        # 穿越行带仅 2~3 帧 —— 交回判定窗口不足, 永远捕获失败
        self.hd_speed_cap = 2.0              # m/s
        self._invalid_frames = 0
        self._invalid_dist = 0.0            # 丢行期间累积路程 (m;
                                            # 高速下帧数触发太慢, 跑飞
                                            # 27m 才掉头, 掉头原点已
                                            # 出捕获带 —— 加路程维度)
        self._invalid_steer0 = 0.0           # 丢行起点舵角
        self._hd_dh = 0.0                    # 掉头段航向累积 (rad)
        self._hd_dist = 0.0                  # 掉头段路程累积 (m)
        self._hd_turn_sign = -1.0            # 掉头转向符号 (-1 左/+1 右,
                                             # _enter_headland 按 cte 重判)
        self._hd_aborted = False             # 掉头中止标志 (row_alive 触发
                                             # 过一次, 防中止-再掉头死循环)
        # 掉头中止机制总开关 (True=启用; GUI --no-abort / 消融对比用)
        self.row_alive_abort = True
        self._capture_frames = 0
        # C5 行尾概率 hint (可开关, None = 未接入保持基线阈值);
        # 高→缩短丢行触发窗口 (行尾主动预判), 低→拉长 (断垄/遮挡
        # 误掉头防护) —— 决策仍是丢行计数触发门, 网络只调阈值
        self._headland_prob = None

        # 车身姿态扰动状态
        self._disturb_phase = 0.0   # 扰动相位
        self._roll_noise = 0.0      # 侧倾低通噪声
        self._pitch_noise = 0.0     # 俯仰低通噪声
        # 颠簸强度倍率 (1.0=标称): 缩放正弦颠簸与随机路面激励,
        # 不影响转向离心耦合侧倾 (那是动力学必然, 与路面无关)。
        # 传感器场景可调大 (如 3.0) 使 IMU 姿态补偿效果可见
        self.bump_intensity = 1.0

    def reset(self, x: float = 0.0, y: float = 0.0,
              heading: float = 0.0, speed: float = None):
        """重置车辆状态"""
        self.state = VehicleState(
            x=x, y=y, heading=heading,
            speed=speed if speed is not None else self.params.default_speed
        )
        self._prev_yaw = 0.0
        self._prev_cte = 0.0
        self._i_front = 0.0
        self._line_phi_f = 0.0
        self._in_curve = False
        self.mode = ControlMode.WORK
        self._invalid_frames = 0
        self._invalid_dist = 0.0
        self._hd_dh = 0.0
        self._hd_dist = 0.0
        self._hd_turn_sign = -1.0
        self._hd_aborted = False
        self._capture_frames = 0
        self._headland_prob = None
        self._disturb_phase = 0.0
        self._roll_noise = 0.0
        self._pitch_noise = 0.0

    def set_headland_prob(self, p):
        """C5 行尾概率 hint (仿真器每帧喂; None = 未接入保持基线)

        P(end) 调制丢行触发门: 决策仍是丢行计数, 网络只调阈值
        (学习不接管控制 —— 掉头方向/时机程序仍是几何控制器)。
        """
        self._headland_prob = None if p is None else float(p)

    def update(self, yaw_deg: float, cte_front: float, cte_rear: float,
               dt: float, nav_valid: bool = True,
               line_angle_deg: float = 0.0,
               line_z_mid: float = 4.0,
               quality: int = -1,
               row_alive: bool = False) -> VehicleState:
        """
        根据导航输出更新车辆状态 (前轴 CTE atan 反馈 + 软门控航向阻尼
        + 弦方向角曲率前馈)。

        Args:
            yaw_deg: 导航输出的航向角 (度; 经软门控用作航向阻尼——
                     |yaw|<yaw_gate 时启用, 弯道上滤波漂移自动退出)
            cte_front: 前轴横向偏差 (m, 右偏正)
            cte_rear: 后轴横向偏差 (m, 右偏正; 仅记录, 弯道上与
                      front 可能异号, 不参与闭环)
            dt: 时间步长 (s)
            nav_valid: 导航输出是否有效 (quality > 0)
            line_angle_deg: 导航线弦方向角 (度, 右正), 指示前方曲率;
                            经 k_ff 加权作前馈, 补偿弯道转向不足
            quality: 导航质量档 (RowMap 后端提供 95/70/50/0;
                     50=重捕获引导 —— CAPTURE 段用引导 yaw 对准
                     行向, 掉头目标朝向已失去意义)
            row_alive: 掉头中止信号 (rowmap 后端): 掉头期间任务行
                       y 跨度增长 ≥2m = 对岸延续锚点落图 = 行未
                       结束, 开环段 (ARC1/JOG/ARC2) 立即中止跳
                       捕获段; baseline 后端恒 False

        Returns:
            更新后的车辆状态
        """
        # ── 控制状态机分发 ──
        if self.mode != ControlMode.WORK:
            # 地头转向程序 (开环 + 捕获段半闭环)
            steering_target = self._update_headland(
                dt, nav_valid, cte_front, quality, yaw_deg, row_alive)
        elif not nav_valid:
            # WORK 中丢行: 计数, 持续丢行且非大转向状态 → 判定行尾, 进掉头
            if self._invalid_frames == 0:
                self._invalid_steer0 = self.state.steering
            self._invalid_frames += 1
            self._invalid_dist += self.state.speed * dt
            # 常规触发: 持续丢行且丢行时近似直行 (区分弯道甩出视野)
            # 高速触发: 丢行路程超限 (>=4 帧确认) —— 帧数条件在
            #   v=16m/s 下等拍 27m, 掉头原点已出捕获带
            # 死锁突破: 丢行超 250 帧无条件掉头 (防捕获失败卡死)
            # C5 调制 (可开关): P(end) 高 → 提前掉头 (行尾主动预判,
            #   P=1 时 6 帧); 低 → 保守 (断垄/遮挡误掉头防护,
            #   P=0 时 54 帧/10.5m); None → 固定阈值 (基线不变)
            if self._headland_prob is None:
                trig_frames = self.headland_trigger_frames
                trig_dist = 2.5
            else:
                p = self._headland_prob
                trig_frames = int(round(6.0 + (1.0 - p) * 48.0))
                trig_dist = 2.5 + (1.0 - p) * 8.0
            small_steer = abs(self._invalid_steer0) < self.headland_steer_gate
            if (self._headland_prob is not None
                    and self._headland_prob >= 0.65):
                # C5 高置信行尾 (>=训练工作点 0.65): 学习判据替代
                # 转向角代理判据 —— 弯道甩出时行仍延续, 前方锚点
                # 密度特征使 P 低, 不会误旁路; 断垄/遮挡同理
                small_steer = True
            if ((self._invalid_frames >= trig_frames
                    and small_steer)
                    or (self._invalid_dist >= trig_dist
                        and self._invalid_frames >= 4)
                    or self._invalid_frames >= 250):
                self._enter_headland()
            steering_target = self.state.steering * 0.95  # 缓慢回正
        else:
            self._invalid_frames = 0
            self._invalid_dist = 0.0
            steering_target = self._work_control(
                yaw_deg, cte_front, dt, line_angle_deg, line_z_mid)
            self._prev_yaw = np.radians(yaw_deg)
            self._prev_cte = cte_front

        # 限制转向角
        max_s = self.params.max_steering_angle
        steering_target = np.clip(steering_target, -max_s, max_s)

        # 转向速率限制 (一阶惯性)
        max_ds = self.params.max_steering_rate * dt
        ds = np.clip(steering_target - self.state.steering, -max_ds, max_ds)
        self.state.steering += ds

        # 地头/捕获段限速 (实际运动学速度; WORK 段不受影响)
        v_eff = self.actual_speed

        # 运动学更新
        s = self.state
        s.heading += (v_eff / self.params.wheelbase_length) * np.tan(s.steering) * dt

        # 位置更新 (heading=0 朝北/Y+)
        s.x += v_eff * np.sin(s.heading) * dt
        s.y += v_eff * np.cos(s.heading) * dt

        # 车身姿态扰动 (颠簸 + 转向侧倾)
        self._update_body_attitude(dt)

        return self.state

    # ════════════════════════════════════════════════════════════
    #  作业循行 (WORK): 前轴 CTE 反馈 + 软门控航向阻尼 + 弯道积分前馈
    # ════════════════════════════════════════════════════════════
    def _work_control(self, yaw_deg: float, cte_front: float,
                      dt: float, line_angle_deg: float,
                      line_z_mid: float = 4.0) -> float:
        """WORK 状态闭环控制律, 返回转向目标 (rad)"""
        # atan 压缩大偏差响应, 小偏差时等效增益 k_cte/v;
        # 弯道稳态误差 ≈ v·tan(δ_ss)/k_cte, 厘米级 (原 PD 为 0.59m)
        #
        # 低速增益调度 (0.2 m/s 级实测: 横向时间常数 1/k≈0.5s 远快于低速
        # 航向响应 v/L·tanδ, 两环失配→振荡→行在图像内摆动→算法跟踪断裂):
        #   增益速度调度: 过程增益 ∝ v²/L (ḋ≈v·ψ, ψ̇=v·δ/L), 增益固定
        #   则高速开环增益×v² → 阻尼比坍塌 → 欠阻尼发散。P/D/i/ff ÷v²、
        #   yaw ÷v (与 ḋ=v·ψ 耦合少除一次) 后闭环极点与 v=1 基准一致:
        #   ω=√(k_cte/L) 恒定, ζ≈0.5 全速不变 (v16 反向蛇形根治)
        v = max(self.state.speed, self.vp_floor)
        v2 = v * v
        d_cte = (cte_front - self._prev_cte) / max(dt, 1e-3)
        yaw_rad = np.radians(yaw_deg)
        w_yaw = max(0.0, 1.0 - abs(yaw_deg) / self.yaw_gate_deg)
        # 弦方向角低通 (抑制像素噪声/单帧跳变)
        self._line_phi_f += 0.3 * (line_angle_deg - self._line_phi_f)

        # 直/弯子状态 (滞回): 弯道启用积分前馈, 直线衰减清零
        if not self._in_curve and abs(self._line_phi_f) > self.curve_enter_deg:
            self._in_curve = True
        elif self._in_curve and abs(self._line_phi_f) < self.curve_exit_deg:
            self._in_curve = False
        if self._in_curve:
            if abs(cte_front) < 0.02 or cte_front * self._i_front < 0:
                self._i_front *= 0.95  # anti-windup 衰减
            else:
                self._i_front += cte_front * dt
        else:
            self._i_front *= 0.90
        i_term = np.clip(-self.k_i * self._i_front / v2,
                         -self.i_limit, self.i_limit)

        # 几何曲率前馈: 圆的性质——弦向角 φ = 弧中点切线转角,
        # 故 κ ≈ tan(φ)/z_mid; 同时含航向偏差项 -ψ_e/z_mid
        # (线偏左 φ 小 → 前馈减舵, 等效航向负反馈, 纯追踪结构)
        phi_f_rad = np.radians(self._line_phi_f)
        ff_term = (self.k_ff / v2) * np.arctan(
            self.params.wheelbase_length * np.tan(phi_f_rad)
            / max(line_z_mid, 2.0))

        # 弯道子状态: front 反馈限幅 (默认关闭, 实测收敛变慢反而更差)
        cte_p_term = -np.arctan(self.k_cte * cte_front / v2)
        if self._in_curve and self.cte_limit_curve > 0:
            cte_p_term = float(np.clip(
                cte_p_term, -self.cte_limit_curve, self.cte_limit_curve))

        return (cte_p_term
                - self.kd_cte * d_cte / v2
                - w_yaw * (self.k_yaw / v) * yaw_rad
                + ff_term
                + i_term)

    # ════════════════════════════════════════════════════════════
    #  地头转向 (HEADLAND): 梨形掉头程序
    # ════════════════════════════════════════════════════════════
    def _enter_headland(self):
        """判定行尾, 进入掉头程序 (RUNOUT 段起步)"""
        self.mode = ControlMode.HD_RUNOUT
        self._hd_dh = 0.0
        self._hd_dist = 0.0
        self._hd_aborted = False
        self._invalid_frames = 0
        self._invalid_dist = 0.0
        self._capture_frames = 0
        self._i_front = 0.0
        # 掉头方向自适应: ARC1 鼓包峰值横移 2R≈8.8m, 朝向由转向
        # 决定——鼓包朝线侧会横穿任务行带压苗, 故应朝"背线"侧。
        # 丢行前最后有效 cte 的符号指示行在哪侧 (cte>0 = 车在线右
        # → 线在左 → 右转使鼓包朝右背线); |cte| 门限内 (正常行尾,
        # 左右对称) 保持传统左转, 与旧行为零差异
        if self._prev_cte > self.headland_dir_gate:
            self._hd_turn_sign = +1.0   # 线在左侧 → 右转 (鼓包背线)
        else:
            self._hd_turn_sign = -1.0   # 线在右/行尾对称区: 左转 (背线)
        # 掉头原点 (JOG 段位置闭环的基准)
        self._hd_origin = (self.state.x, self.state.y, self.state.heading)

    def _update_headland(self, dt: float, nav_valid: bool,
                         cte_front: float = 0.0,
                         quality: int = -1,
                         yaw_deg: float = 0.0,
                         row_alive: bool = False) -> float:
        """地头掉头程序状态推进, 返回转向目标 (rad)

        梨形掉头 (方向自适应): 丢行前行在车哪侧决定 ARC1 转向
        (_hd_turn_sign; 行尾对称区默认左转)。RUNOUT 直行越出 →
        ARC1 满舵转 225° (净朝向后转 180° 且多摆 45° 横向回位,
        车身落向行的一侧) → JOG 直行 2.414R 消掉头横向偏移 →
        ARC2 反向 45° 回到行向 → CAPTURE 直行重新捕获行。
        航向/路程累积驱动, 适应任意车速。

        掉头中止 (row_alive): 掉头理由 (行尾) 消失 —— 对岸延续
        锚点落图使任务行 y 跨度增长。仅在航向偏离行向 <90° 时
        中止 (小回正代价); 已转深 (ARC1 后半/JOG/ARC2, 航向接近
        反向) 完成掉头落邻行更划算 —— CAPTURE 对准行向需再转
        近 180°, ±0.14~0.30 rad 限幅修不回来反而失控。
        """
        s = self.state
        p = self.params
        v_eff = self.actual_speed
        dh = (v_eff / p.wheelbase_length) * np.tan(s.steering) * dt
        self._hd_dh += dh
        self._hd_dist += v_eff * dt

        r_turn = p.wheelbase_length / np.tan(self.turn_steering)

        # ── 掉头中止: 行未结束 (对岸延续锚点落图), 回行向捕获 ──
        # 每次掉头只中止一次 (_hd_aborted 防 CAPTURE 失败再掉头→
        # row_alive 仍 True→再中止的死循环); RUNOUT/ARC1 前半航向
        # 尚未离行向远, 中止后 CAPTURE 修 ≤90° 回正代价可接受
        if (row_alive and self.row_alive_abort
                and not self._hd_aborted
                and self.mode != ControlMode.HD_CAPTURE):
            h0 = self._hd_origin[2]
            dh_now = (s.heading - h0 + np.pi) % (2 * np.pi) - np.pi
            if abs(dh_now) < np.pi / 2:
                self._hd_aborted = True
                self._advance(ControlMode.HD_CAPTURE)
                return 0.0

        if self.mode == ControlMode.HD_RUNOUT:
            # 直行越出行尾 1.0s 路程 ~1m (按路程 1.2m 截止)
            if self._hd_dist > 1.2:
                self._advance(ControlMode.HD_ARC1)
            return 0.0

        if self.mode == ControlMode.HD_ARC1:
            # 满舵转向 225°, 方向自适应 (_hd_turn_sign, 朝行侧)
            if abs(self._hd_dh) >= np.radians(225.0):
                self._advance(ControlMode.HD_JOG)
            return self._hd_turn_sign * self.turn_steering

        if self.mode == ControlMode.HD_JOG:
            # 斜行回位 (位置闭环): 横向位置回到原点线上 ARC2 预估
            # 横移量处即切 ARC2——自适应吸收 ARC1/JOG 的转向爬升
            # 段累积误差 (固定距离方案误差达 1.4m, 会错过行带)
            x0, y0, h0 = self._hd_origin
            lat = ((self.state.x - x0) * np.cos(h0)
                   - (self.state.y - y0) * np.sin(h0))
            t_ramp = self.turn_steering / p.max_steering_rate
            # 路程预估用 v_eff: 地头限速后设定速度不代表实际位移速度
            arc2_shift = (r_turn * (1.0 - np.cos(np.pi / 4))
                          + 0.75 * t_ramp * v_eff)
            # 统一双方向条件: sign·lat 自 1.707R 单调降到 arc2_shift
            # (左转版 lat 自 -1.707R 回升 / 右转版自 +1.707R 回落, 镜像)
            if self._hd_turn_sign * lat <= arc2_shift:
                self._advance(ControlMode.HD_ARC2)
            return 0.0

        if self.mode == ControlMode.HD_ARC2:
            # 反向 45° 修回朝向 (与 ARC1 转向相反)。本段不提前接管:
            # 朝向未修足时接管会因 front 信号未稳定 + 航向差过大
            # 导致捕获失败
            if abs(self._hd_dh) >= np.radians(45.0):
                self._advance(ControlMode.HD_CAPTURE)
            return -self._hd_turn_sign * self.turn_steering

        # HD_CAPTURE: 半闭环缓修 (±8° 限幅, 横向 front + 航向对准
        # 掉头起点反向), 两者都收敛且连续有效才交回 WORK——只修
        # front 会以航向偏离为代价, 接管后欠阻尼振荡反复穿线丢行。
        # 无信号直走超时 (随速度放大) → 重新掉头 (行带内梨形, JOG 会再见行)
        if nav_valid:
            self._capture_frames += 1
            # 重捕获引导帧 (quality=50): 掉头目标朝向 (跑飞航向的反向)
            # 已无意义, 与引导 homing 方向打架会绕锚点区画弧交不回 ——
            # 改用引导 yaw 对准行向, 交回条件同样换成行向对准
            guided = (quality == 50)
            if guided:
                ok_hd = abs(yaw_deg) < 8.0
                steer_cmd = (-np.arctan(cte_front)
                             - 0.3 * np.radians(yaw_deg))
            elif self._hd_aborted:
                # 中止掉头: 航向目标 = 行向 (掉头起点航向) 而非
                # 邻行向 (起点反向) —— 行还活着, 回到原行工作
                h_err = (self.state.heading - self._hd_origin[2]
                         + np.pi) % (2 * np.pi) - np.pi
                ok_hd = abs(h_err) < np.radians(8.0)
                steer_cmd = (-np.arctan(cte_front) - 0.3 * h_err)
            else:
                h_err = self._heading_to_target()
                ok_hd = abs(h_err) < np.radians(8.0)
                steer_cmd = (-np.arctan(cte_front) - 0.3 * h_err)
            if (abs(cte_front) < 0.15 and ok_hd
                    and self._capture_frames >= self.capture_valid_frames):
                self.mode = ControlMode.WORK
                self._hd_aborted = False
                self._prev_cte = cte_front
                return 0.0
            # 半增益缓修: 横向收敛 + 航向对准, 限幅防振荡丢行。
            # 大偏差 (掉头回位偏出捕获带, >0.8m) 时放宽限幅 ——
            # ±0.14rad 在 7m 级偏差下弧长不够, 永远修不回去
            cap_lim = 0.30 if abs(cte_front) > 0.8 else 0.14
            return float(np.clip(steer_cmd, -cap_lim, cap_lim))
        self._capture_frames = 0
        # 无信号直走超时 → 重新掉头 (行带内梨形, JOG 会再见行)。
        # 距离随速度放大 (低速 10m 不变; 高速给足观察窗, 否则 0.6s
        # 就超时, 车还没看见行就再次掉头 → 原地转圈死循环)
        if self._hd_dist > max(10.0, 2.5 * v_eff):
            self._enter_headland()
            self.mode = ControlMode.HD_ARC1  # 跳过 RUNOUT (已在行尾区)
        return 0.0

    def _heading_to_target(self) -> float:
        """当前朝向与掉头目标朝向 (起点反向) 的误差, 归一化到 ±π"""
        h_target = self._hd_origin[2] + np.pi
        err = self.state.heading - h_target
        return (err + np.pi) % (2 * np.pi) - np.pi

    def _advance(self, next_mode: ControlMode):
        """掉头段切换: 清零段内累积量"""
        self.mode = next_mode
        self._hd_dh = 0.0
        self._hd_dist = 0.0

    def _update_body_attitude(self, dt: float):
        """
        演化车身姿态扰动 (roll/pitch)。

        组成:
          - 速度相关的正弦颠簸 (模拟不平地面)
          - 转向离心耦合 (转弯时车身外倾, 不受 bump_intensity 缩放)
          - 低通随机噪声 (模拟随机路面激励)
        静止时扰动衰减到 0。扰动为 IMU 仿真的真值姿态来源
        (sensor_simulator 在此基础上叠加测量噪声)。
        """
        s = self.state
        # 速度因子: 静止时无扰动, 0.5 m/s 以上满幅
        speed_factor = min(1.0, s.speed / 0.5)
        bump = self.bump_intensity

        # 扰动相位随速度推进 (速度越快颠簸频率越高)
        self._disturb_phase += dt * (2.0 + 4.0 * s.speed)

        # 侧倾: 正弦颠簸 + 转向耦合 + 低通噪声
        roll_base = 0.007 * bump * np.sin(self._disturb_phase * 2.3)
        roll_steer = -0.10 * s.steering * min(1.0, s.speed / 2.0)
        self._roll_noise = (0.92 * self._roll_noise
                            + 0.08 * bump * np.random.normal(0, 0.004))
        s.roll = (roll_base + roll_steer + self._roll_noise) * speed_factor
    
        # 俯仰: 不同频率正弦 + 低通噪声
        pitch_base = 0.005 * bump * np.sin(self._disturb_phase * 3.1 + 1.3)
        self._pitch_noise = (0.92 * self._pitch_noise
                             + 0.08 * bump * np.random.normal(0, 0.003))
        s.pitch = (pitch_base + self._pitch_noise) * speed_factor

    def set_speed(self, speed: float):
        """设置车速"""
        self.state.speed = max(0.0, speed)

    @property
    def actual_speed(self) -> float:
        """实际运动学速度: WORK 段 = 设定速度; 地头/捕获段受
        hd_speed_cap 限速 (农机地头减速惯例)"""
        if self.mode != ControlMode.WORK:
            return min(self.state.speed, self.hd_speed_cap)
        return self.state.speed

    @property
    def position(self) -> tuple:
        """当前全局位置 (x, y)"""
        return (self.state.x, self.state.y)

    @property
    def heading_deg(self) -> float:
        """当前朝向角 (度)"""
        return np.degrees(self.state.heading)
