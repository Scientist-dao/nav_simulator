"""
row_map.py - 行结构地图核心数据结构 (RowMap 参考实现)

设计要点 (开发文档 §3.1 C1/C2):
  - 锚点折线表达: 每条作物行为世界系下沿行方向 (y) 排序的锚点串,
    每个锚点持有横向位置 x 及其方差 var。行天然支持直线/弯道任意形状。
  - 1/var 信息加权融合: 新观测与既有锚点按信息权重合成,
    var ~ 1/n 收敛 —— 同一锚点被观测次数越多越准 (C4 时序聚合的
    数学基础: 20m 滑窗 × 15fps × 1m/s ≈ 每锚点 290 次观测)。
  - 行级关联: 每帧每条 mask 的观测点串整体与已有行按平均横向偏差
    匹配 (取最近, 门限 GATE_ABS_M), 未匹配则建新行。行距 0.6m 而
    门限 0.35m, 邻行不可能误配。
  - 任务行 ID 锁定: 锁定一次终身不变 —— 导航解算只依赖锁定行的
    地图位置, 不受"车正下方对称中心"干扰, 从机理上免疫 D1 死锁。

坐标系: 世界系 (ENU) 与 simulator 一致; 行沿 y 延伸, 行间沿 x 排列
(初始直行工况, 弯道行 x 随 y 缓变由锚点折线自动表达)。
"""

import bisect
from typing import Dict, List, Optional, Tuple

# ── 参考实现参数 (集中常量, 调参入口) ──
ANCHOR_STEP_M = 0.5     # 新建行时的锚点间距 (m, 沿行方向)
ANCHOR_FUSE_M = 0.30    # 观测 y 距最近锚点小于此值则融合, 否则插入新锚点
GATE_ABS_M = 0.35       # 行级关联门限: 平均横向偏差 (m)。
                        # 必须满足: < 行距(0.60) 防邻行误配,
                        #            > 常规观测噪声+位姿误差 允许初始捕获
TASK_LOCK_MIN_ANCHORS = 2   # 行成为任务行锁定候选的最少锚点数
TASK_LOCK_TIE_TOL_M = 0.05 # 锁定平局容差: 距离差小于此值视为平局,
                            # 取 id 小 (= 先建/左侧行) —— 死锁位形下
                            # 车在两行正中, 左行即任务行 (骑中心行语义)
STALE_ROW_FRAMES = 30       # 垃圾行回收: 连续无观测且锚点过少的行删除阈值
MIN_DISPLAY_ANCHORS = 3     # overlay 可视化的最少锚点数 (滤噪声 mask 碎行)
PSEUDO_INFO_BUDGET = 16.0   # 单锚点伪观测信息预算 = 1/nav_core.BEV_PSEUDO_VAR
                            # (一笔 σ=0.25m 记忆的信息量)。记忆证据同源
                            # 高度相关, 复读不产生新信息 —— 累计伪信息
                            # 超过一笔记忆后 var 不再虚假坍缩 (2026-09-16
                            # 修复: 修复前 150 帧同位置伪观测把 σ 25cm
                            # 压到 2.4cm, 下游按 var 给信任度会高估漂移
                            # 的记忆)。真观测不受预算约束。


class RowAnchor:
    """行锚点: 行方向位置 y 处的横向位置估计

    y 固定取首次观测值 (y 误差只影响插值位置, 不引入横向误差, 不融合);
    x 按 1/var 信息权重融合, 融合后 var = 1/(w_old+w_new) 单调下降。

    pseudo=True 的锚点来自 C4 伪观测 (网络记忆而非本帧测量): 只支撑
    导航线 (低权融合), 不参与行关联打分 —— 伪锚点随失明时长漂移,
    若用它给对岸观测打分, 漂移线会抢走真延续段的行身份。真观测
    融合进来后锚点转正 (1/var 权重下 x 立即被真值拉回)。
    """
    __slots__ = ("y", "x", "var", "n", "pseudo", "pseudo_w")

    def __init__(self, y: float, x: float, var: float,
                 pseudo: bool = False):
        self.y = float(y)
        self.x = float(x)
        self.var = max(float(var), 1e-8)
        self.n = 1
        self.pseudo = bool(pseudo)
        # 已入账的伪信息量: 新建自伪观测的锚点视为已记满一笔
        # (其 var 即这笔记忆的 var), 之后同锚点复读只能花剩余预算
        self.pseudo_w = (1.0 / self.var) if pseudo else 0.0

    def fuse(self, x_obs: float, var_obs: float, pseudo: bool = False):
        """信息加权融合一个观测 (σ 融合 ~ σ_obs/√n 收敛)

        pseudo=True 走信息预算: 同源记忆证据高度相关, 不是独立
        测量 —— 单锚点累计伪信息封顶 PSEUDO_INFO_BUDGET (一笔
        σ=0.25m 记忆)。预算耗尽后复读不再记账 (位置与 var 均不
        动); 真观测照常融合且清零伪账 (锚点转正, 下一段失明
        重新获得全额预算)。
        """
        w_new = 1.0 / max(float(var_obs), 1e-8)
        if pseudo:
            budget_left = PSEUDO_INFO_BUDGET - self.pseudo_w
            if budget_left <= 0.0:
                return                  # 预算耗尽: 记忆复读不记账
            w_new = min(w_new, budget_left)
            self.pseudo_w += w_new
        w_old = 1.0 / self.var
        self.x = (self.x * w_old + x_obs * w_new) / (w_old + w_new)
        self.var = 1.0 / (w_old + w_new)
        self.n += 1


class RowModel:
    """单条作物行: 世界系锚点折线 (按 y 升序, bisect 维护插入)"""

    def __init__(self, row_id: int):
        self.row_id = row_id
        self.anchors: List[RowAnchor] = []
        self._ys: List[float] = []        # 与 anchors 同步的 y 键 (bisect 用)
        self.last_seen: int = -1          # 最近观测到的帧号

    # ── 查询 ──
    def x_at(self, y: float) -> Tuple[float, float]:
        """行在 y 处的 (x, var): 段内线性插值, 端外用端段斜率外推"""
        n = len(self.anchors)
        if n == 0:
            return 0.0, 1e6
        if n == 1:
            a = self.anchors[0]
            return a.x, a.var
        i = bisect.bisect_left(self._ys, y)
        i = min(max(i, 1), n - 1)         # 所在/最近段的右锚点索引
        a0, a1 = self.anchors[i - 1], self.anchors[i]
        dy = a1.y - a0.y
        t = (y - a0.y) / dy if abs(dy) > 1e-9 else 1.0
        x = a0.x + t * (a1.x - a0.x)
        var = a0.var + max(min(t, 1.0), 0.0) * (a1.var - a0.var)
        return x, max(var, 1e-8)

    def mean_abs_dx(self, pts: List[Tuple[float, float, float]]) -> float:
        """观测点串 [(wx, wy, var), ...] 对本行的平均横向偏差 (关联打分)

        只用真锚点评估: 伪锚点是 C4 记忆, 随失明时长漂移, 若参与
        打分, 漂移线会劫持对岸真观测的行身份。端外查询取端锚点 x
        (见 _x_at_seq), 不与 x_at 的斜率外推混同。行内无真锚点
        → inf (不可关联)。
        """
        real = self._real_anchors()
        if not real:
            return float("inf")
        total = 0.0
        for wx, wy, _ in pts:
            total += abs(wx - self._x_at_seq(real, wy))
        return total / len(pts)

    def x_at_real(self, y: float) -> Optional[float]:
        """仅真锚点在 y 处的 x (段内插值/端外取端锚点 x)

        供 C4 伪观测的预测线使用: 锚定在真锚点几何上, 切断"伪锚点
        漂移 → 预测跟着漂 → 伪观测跟着漂"的自反馈。无真锚点 → None。
        """
        real = self._real_anchors()
        if not real:
            return None
        return self._x_at_seq(real, y)

    def _real_anchors(self) -> List["RowAnchor"]:
        if not any(a.pseudo for a in self.anchors):
            return self.anchors
        return [a for a in self.anchors if not a.pseudo]

    @staticmethod
    def _x_at_seq(anchors: List["RowAnchor"], y: float) -> float:
        """锚点序列 (按 y 升序) 在 y 处的 x: 段内线性插值, 端外取端锚点 x

        端外不用端段斜率外推: 端段基线短 (~0.7m), 斜率噪声 ~0.01 m/m
        外推 24m 放大成 ~0.3m 横向位移 —— 恰为关联门限量级, 会把行
        预测线拽到邻行位置劫持关联 (gap 对岸首帧跟错行根因)。
        农行直行先验下端锚点 x 是更稳的估计。段内插值不受影响。
        """
        n = len(anchors)
        if n == 1:
            return anchors[0].x
        ys = [a.y for a in anchors]
        i = bisect.bisect_left(ys, y)
        i = min(max(i, 1), n - 1)         # 所在/最近段的右锚点索引
        a0, a1 = anchors[i - 1], anchors[i]
        dy = a1.y - a0.y
        t = (y - a0.y) / dy if abs(dy) > 1e-9 else 1.0
        t = min(max(t, 0.0), 1.0)          # 端外钳到端锚点 (直行先验)
        return a0.x + t * (a1.x - a0.x)

    def segment(self, y0: float, y1: float) -> List[Tuple[float, float, float]]:
        """[y0, y1] 内的锚点段 [(y, x, var), ...] (供解算/可视化)"""
        i0 = bisect.bisect_left(self._ys, y0)
        i1 = bisect.bisect_right(self._ys, y1)
        return [(a.y, a.x, a.var) for a in self.anchors[i0:i1]]

    def nearest_segment(self, y: float, n: int = 8
                        ) -> List[Tuple[float, float, float]]:
        """离纵向位置 y 最近的 n 个锚点 [(y, x, var), ...]

        重捕获引导用: 车可能远离行带 (冲出行尾/掉头回位偏差),
        以车为中心的窗会与锚点区完全错开 —— 取最近锚点邻域保证
        查表总有解 (引导 = 朝最近已建段 homing)。
        """
        if not self.anchors:
            return []
        i = bisect.bisect_left(self._ys, y)
        lo = max(0, i - n // 2)
        hi = min(len(self.anchors), lo + n)
        lo = max(0, hi - n)
        return [(a.y, a.x, a.var) for a in self.anchors[lo:hi]]

    @property
    def y_min(self) -> float:
        return self._ys[0] if self._ys else float("nan")

    @property
    def y_max(self) -> float:
        return self._ys[-1] if self._ys else float("nan")

    # ── 更新 ──
    def _append_anchor(self, anchor: RowAnchor):
        self.anchors.append(anchor)
        self._ys.append(anchor.y)

    def absorb(self, pts: List[Tuple[float, float, float]], frame: int,
               pseudo: bool = False):
        """观测点串 (按 y 升序) 融合进锚点串

        每点找 y 最近锚点: 距离 < ANCHOR_FUSE_M 融合, 否则按 y 排序插入
        新锚点 —— 锚点自然间隔 >= ANCHOR_FUSE_M, 密度有界。
        pseudo=True 时新锚点带伪标记; 真观测融合进伪锚点将其转正
        (真观测 var 小, 1/var 权重下 x 立即被真值主导)。
        """
        self.last_seen = frame
        ys = self._ys
        for wx, wy, var in pts:
            j = bisect.bisect_left(ys, wy)
            best_j, best_d = -1, ANCHOR_FUSE_M
            for jj in (j - 1, j):
                if 0 <= jj < len(ys):
                    d = abs(ys[jj] - wy)
                    if d < best_d:
                        best_d, best_j = d, jj
            if best_j >= 0:
                a = self.anchors[best_j]
                a.fuse(wx, var, pseudo)
                if not pseudo:
                    a.pseudo = False    # 真观测转正: 清伪标记与伪账
                    a.pseudo_w = 0.0    # (下段失明重新获全额预算)
            else:
                self.anchors.insert(j, RowAnchor(wy, wx, var, pseudo))
                ys.insert(j, wy)

    @staticmethod
    def new_from_points(row_id: int,
                        pts: List[Tuple[float, float, float]],
                        frame: int) -> "RowModel":
        """从观测点串 (按 y 升序) 新建行, 按 ANCHOR_STEP_M 重采样建锚点"""
        row = RowModel(row_id)
        row.last_seen = frame
        next_y = float("-inf")
        for wx, wy, var in pts:
            if wy >= next_y:
                row._append_anchor(RowAnchor(wy, wx, var))
                next_y = wy + ANCHOR_STEP_M
        return row


class RowMapCore:
    """行结构地图: 多行管理 + 行级关联 + 任务行 ID 锁定"""

    def __init__(self):
        self.rows: Dict[int, RowModel] = {}
        self._next_id: int = 0
        self.task_row_id: Optional[int] = None

    def reset(self):
        self.rows.clear()
        self._next_id = 0
        self.task_row_id = None

    # ── 每帧更新 ──
    def update(self, obs_rows: List[List[Tuple[float, float, float]]],
               frame: int, pseudo: bool = False) -> List[int]:
        """观测点串列表 (每条 mask 一个) 关联/融合进地图

        pseudo=True 为 C4 伪观测通道: 绕过行关联直接融合进任务行 ——
        伪观测本就提取自任务行预测邻域, 送进关联是循环论证, 且漂移
        超门限时还会孵化垃圾行; 返回空 matched (quality 语义保持
        纯视觉, 不被伪观测抬高)。

        Returns: 本帧关联成功的行 id 列表 (供 quality 判定:
        任务行是否被本帧观测到)。
        """
        if pseudo:
            task = self.task_row()
            if task is None:
                return []
            for pts in obs_rows:
                pts_sorted = sorted(pts, key=lambda p: p[1])
                if len(pts_sorted) >= 2:
                    task.absorb(pts_sorted, frame, pseudo=True)
            return []
        matched: List[int] = []
        for pts in obs_rows:
            pts_sorted = sorted(pts, key=lambda p: p[1])
            if len(pts_sorted) < 2:
                continue
            best_id, best_cost = None, GATE_ABS_M
            for rid, row in self.rows.items():
                cost = row.mean_abs_dx(pts_sorted)
                if cost < best_cost:
                    best_cost, best_id = cost, rid
            if best_id is not None:
                self.rows[best_id].absorb(pts_sorted, frame)
                matched.append(best_id)
            else:
                row = RowModel.new_from_points(
                    self._next_id, pts_sorted, frame)
                self.rows[self._next_id] = row
                matched.append(self._next_id)
                self._next_id += 1
        self._gc_stale_rows(frame)
        return matched

    def _gc_stale_rows(self, frame: int):
        """回收垃圾行: 长期无观测且锚点过少的非任务行 (噪声 mask 残留)"""
        stale = [rid for rid, r in self.rows.items()
                 if (rid != self.task_row_id
                     and len(r.anchors) < 4
                     and frame - r.last_seen > STALE_ROW_FRAMES)]
        for rid in stale:
            del self.rows[rid]

    # ── 任务行锁定 (C1 核心: 锁定后不变, 死锁免疫的来源) ──
    def lock_task_row(self, pose_x: float, pose_y: float) -> Optional[int]:
        """锁定任务行: 距当前车位最近的成熟行; 平局取 id 小 (先建=左)

        平局语义: 死锁位形 (车在两行正中) 下两候选距离相等 (首帧锚点
        噪声 ~2cm 会破坏严格相等, 故用 TASK_LOCK_TIE_TOL_M 容差),
        取 id 小 = 先检出 = 画面左侧行 = 任务中心行。
        锁定后 ID 身份固定, 对称性不再影响解算, CTE 直接给出真实
        偏差并驱动纠偏 (基线此位形恒 0 永不纠偏)。
        """
        if self.task_row_id is not None:
            return self.task_row_id
        cands = [(abs(self.rows[rid].x_at(pose_y)[0] - pose_x), rid)
                 for rid, r in self.rows.items()
                 if len(r.anchors) >= TASK_LOCK_MIN_ANCHORS]
        if not cands:
            return None
        d_min = min(d for d, _ in cands)
        self.task_row_id = min(rid for d, rid in cands
                               if d - d_min < TASK_LOCK_TIE_TOL_M)
        return self.task_row_id

    def task_row(self) -> Optional[RowModel]:
        if self.task_row_id is None:
            return None
        return self.rows.get(self.task_row_id)

    # ── 诊断 ──
    def stats(self) -> Dict:
        """地图统计 (诊断/GUI)"""
        task = self.task_row()
        return {
            "n_rows": len(self.rows),
            "task_row_id": self.task_row_id,
            "task_anchors": len(task.anchors) if task else 0,
            "task_var_min": (min(a.var for a in task.anchors)
                             if task and task.anchors else float("nan")),
            "total_anchors": sum(len(r.anchors) for r in self.rows.values()),
        }
