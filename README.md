# nav_simulator — 农机视觉导航闭环仿真器

农机沿作物行作业的视觉导航算法闭环仿真平台: 渲染合成作物行图像 →
分割 mask → 导航解算 → 车辆运动闭环, 用于算法迭代与回归基准。

## 架构

- **四层投影渲染链**: 世界 → 车辆 → 相机 → 裁剪图像, 掩膜由地面
  几何点经相机内外参投影生成, 与背景渲染同源;
- **双层扰动**: 车身姿态 (roll/pitch, 相机随动) + 传感器噪声
  (RTK 降级/丢包、IMU、误检 mask), 为姿态补偿与降级兜底提供
  可复现的扰动源;
- **确定性时钟**: 固定步长 dt=1/15 s, 同种子逐帧完全可复现;
- **行结构地图 Python 参考实现** (`rowmap_proto/`): 行结构地图
  导航算法的纯 Python 版本, 供快速迭代与数值交叉验证。

## 导航后端 (`--backend`)

| 后端 | 说明 |
|---|---|
| `rowmap` | 内置 Python 参考实现 (默认) |
| `rowmap_cpp` | C++ 工程编译产物 (`vision_navigation`); 需将 rowmap_navigation 仓库与本仓库克隆到同一父目录并编译, 产物在其 `build-pc/Release/` |
| `baseline` | 既有板端基线库 (可选, 不随本仓库分发)。缺失时程序给出指引: 设置环境变量 `NAV_BASELINE_LIB` 指向含基线库的目录 |

## 目录结构

```
├── nav_simulator/
│   ├── core/            # 仿真核心: 车辆模型/传感器/扰动/指标/后端装配
│   ├── rendering/       # 作物行渲染与掩膜生成 (投影链)
│   ├── scenarios/       # 29 个预设场景
│   ├── rowmap_proto/    # 行结构地图导航 Python 参考实现
│   ├── dl/              # 深度学习模块 (数据工厂/训练/运行时/部署导出)
│   ├── gui/             # pygame 仪表盘
│   ├── main.py          # 仿真入口
│   └── run_benchmark.py # 多种子闭环基准
├── models/              # 已训练权重 (质量评估/BEV 预测/行尾感知)
├── navsim.py/.cmd       # GUI 快捷入口
└── pyproject.toml
```

## 快速开始

```powershell
uv sync

# 列出全部场景
.\.venv\Scripts\python.exe -m nav_simulator.main -l

# 无头闭环仿真 (300 帧)
.\.venv\Scripts\python.exe -m nav_simulator.main -s straight_3row `
    --backend rowmap --headless --max-frames 300

# GUI 模式
.\.venv\Scripts\python.exe -m nav_simulator.main -s straight_3row
# 或快捷入口: navsim straight_3row

# 多种子闭环基准 (压行/丢行/CTE/掉头触发等指标)
.\.venv\Scripts\python.exe -m nav_simulator.run_benchmark --backend rowmap --seeds 1
```

数据集 (`datasets/`, 由 dl 数据工厂生成) 与实验输出 (`results/`)
不入仓库, 可用 `nav_simulator/dl/` 下脚本重新生成。

## 场景预览 (共 29 个)

直线 (标准/偏移/密行/偏角) · 曲线 (缓/急/极弯/正弦/回旋) ·
行尾终止 · 缺行 (侧/中) · 断垄 (10/15/25m 极限) · 时变遮挡
(3 段/10s 持续) · 弱苗 · 噪声误检 · 覆膜行 · 行距 (宽/窄) ·
高速 · 传感器组合 (全融合/纯 IMU/RTK 浮点解/RTK 丢包) ·
对称死锁点。完整列表见 `python -m nav_simulator.main -l`。

## 深度学习模块

`nav_simulator/dl/` 提供三个网络的全流程 (数据工厂 dump →
训练 train → 运行时 runtime → 部署权重导出):

| 模块 | 网络 | 用途 |
|---|---|---|
| `c3_*` | 异方差观测质量 MLP | 从带内特征预测观测方差, 替代手写方差 |
| `c4_*` | ConvGRU BEV 语义预测 | 断垄/遮挡期间的记忆续推证据 |
| `c5_*` | 行尾感知 MLP | 行尾概率 P(end), 支撑掉头触发时机 |

训练好的权重在 `models/` (npz 为 Python 运行时格式, wts 为
C++ 部署格式), 闭环注入开关见 `main.py --help` (`--qnet/--hnet`)。
