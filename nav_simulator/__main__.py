"""__main__.py - 支持 python -m nav_simulator 直接运行

用法等价于 python nav_simulator/main.py:
  python -m nav_simulator --scenario sensor_full_stack
  python -m nav_simulator --list
"""

from nav_simulator.main import main

if __name__ == "__main__":
    main()
