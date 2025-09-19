# test_viewer.py
from __future__ import annotations

import math

import numpy as np
# ========= 选择 VisPy 后端（尽量自动；若失败再尝试 PyQt5） =========
from vispy import app

try:
    app.use_app()  # 使用默认已安装后端
except Exception:
    try:
        app.use_app('pyqt5')  # 需要: pip/conda 安装 pyqt5
    except Exception as e:
        print("No GUI backend found for VisPy. Please install one, e.g. `pip install pyqt5`.")
        raise

from env.viewer import Viewer


def main():
    # 两个体：追击者和逃避者
    agents = ["p_0", "e_0"]

    # 创建 Viewer：窗口 900x700，轨迹历史长度 300
    viewer = Viewer(width=900, height=700, agents=agents, max_history=30)

    # 运动参数（单位：米；这里只是测试数据，不是高保真力学）
    R_p = 4.21663e7  # ~ GEO 轨道半径（仅做示意）
    R_e = 4.21663e7 * 1.01  # 稍大一点
    w_p = 2 * math.pi / (200.0)  # 24h 一圈
    w_e = 2 * math.pi / (100.0)  # 稍快一点
    z_amp_p, z_amp_e = 1.2e6, 0.8e6  # z 方向轻微起伏
    noise = 1.0e5  # 叠加少量噪声（米）
    dt = 2.0  # 仿真步长（秒）
    state_t = 0.0

    # 初相位
    theta0_p = np.random.uniform(0, 2 * np.pi)
    theta0_e = np.random.uniform(0, 2 * np.pi)

    # 限制总步数，自动退出
    max_steps = 4000
    step_count = 0

    # 每帧更新函数
    def on_timer(event):
        nonlocal state_t, step_count

        # 计算“准轨道”位置（x-y 圆，z 起伏）
        theta_p = theta0_p + w_p * state_t
        theta_e = theta0_e + w_e * state_t

        p_pos = np.array([
            R_p * math.cos(theta_p),
            R_p * math.sin(theta_p),
            z_amp_p * math.sin(0.3 * theta_p)
        ], dtype=np.float64)

        e_pos = np.array([
            R_e * math.cos(theta_e),
            R_e * math.sin(theta_e),
            z_amp_e * math.cos(0.25 * theta_e)
        ], dtype=np.float64)

        # 叠加一点随机扰动，避免完全光滑（更容易看出轨迹刷新）
        p_pos += np.random.normal(scale=noise, size=3)
        e_pos += np.random.normal(scale=noise, size=3)

        # 构造与环境一致的状态字典（这里假装速度为 0）
        states = {
            "p_0": np.concatenate([p_pos, np.zeros(3)]).astype(np.float64),
            "e_0": np.concatenate([e_pos, np.zeros(3)]).astype(np.float64),
        }

        # 驱动你的 Viewer 刷新
        viewer.update(states)

        # 推进时间 & 计数
        state_t += dt
        step_count += 1

        # 达到步数后自动关闭
        if step_count >= max_steps:
            timer.stop()
            viewer.close()
            app.quit()

    # 用 VisPy 定时器实现 ~30 FPS 动画
    timer = app.Timer(interval=1.0 / 30.0, connect=on_timer, start=True)

    # 进入事件循环
    app.run()


if __name__ == "__main__":
    main()
