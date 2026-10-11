"""navsim - 快捷入口: 调起 nav_simulator GUI, 自动补默认参数

用法:
    navsim <场景名> [--speed X] [--backend baseline|rowmap] [其余 main.py 参数...]
    navsim -l                  列出全部场景

自动补参:
    - 第一个裸 token (不以 - 开头) 视作场景名, 等价 --scenario <名>
    - 未显式指定 --backend 时默认 rowmap (新 RowMap 算法)
"""
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
# 第一个裸 token -> 场景名 (已显式给 --scenario/-s 则不动)
if args and not args[0].startswith("-") \
        and "--scenario" not in args and "-s" not in args:
    args = ["--scenario", args[0]] + args[1:]
# 默认后端 rowmap (显式传 --backend baseline 可对比基线)
if "--backend" not in args:
    args += ["--backend", "rowmap"]
root = Path(__file__).resolve().parent
sys.exit(subprocess.call(
    [sys.executable, "-m", "nav_simulator.main", *args], cwd=root))
