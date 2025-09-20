from collections import deque
from typing import List

import numpy as np
from vispy import app, scene


class Viewer:
    def __init__(
            self,
            width: int,
            height: int,
            agents: List[str],
            max_history: int = 20,
    ):
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            size=(width, height),
            show=True
        )
        self.view = self.canvas.central_widget.add_view()
        self.max_history = max_history

        self.agents = agents
        self.positions = {a: np.zeros(3, ) for a in self.agents}
        self.history_positions = {a: deque(maxlen=self.max_history) for a in self.agents}

        self.markers = {a: scene.visuals.Markers() for a in self.agents}
        self.trajs = {a: scene.visuals.Line() for a in self.agents}
        self.colors = {a: np.random.rand(3, ) for a in self.agents}

        # 绘图参数
        self._marker_size: int = 20
        self._line_width: int = 5

        for a in self.agents:
            self.markers[a].set_data(
                np.expand_dims(self.positions[a], 0),
                face_color=self.colors[a],
                size=self._marker_size
            )
            self.view.add(self.markers[a])
            self.view.add(self.trajs[a])

        self.view.camera = scene.cameras.TurntableCamera(fov=60, elevation=20, azimuth=30)

        # 相机追踪参数
        self._autoframe_margin = 0.10  # 10% 边距
        self._autoframe_every = 2  # 每帧更新相机位置
        self._frame_count = 0
        self._smooth_center = None
        self._smooth_alpha = 0.5  # 中心点平滑（0-1），越大追随越快

        # 图例相关
        self._legend_margin_px = (12, 12)  # 图例左上角偏移（像素）
        self._legend_row_spacing = 30  # 每行间距（像素）
        self._legend_swatch_size = 20  # 色块大小（像素）
        self._legend_items = {}  # 存储图例的字典
        self._init_legend_fixed()  # 调用图例初始化方法

    def _init_legend_fixed(self):
        """一次性创建图例，位置固定在左上角像素坐标。"""
        x0, y0 = self._legend_margin_px  # 图例起始位置（像素）
        for idx, a in enumerate(self.agents):
            y = y0 + idx * self._legend_row_spacing

            # 色块（放在 canvas.scene 下，位置通过 STTransform 设置）
            swatch = scene.visuals.Markers(parent=self.canvas.scene)
            swatch.set_data(np.array([[0.0, 0.0]]), face_color=self.colors[a], size=self._legend_swatch_size)
            swatch.transform = scene.transforms.STTransform(translate=(x0, y, 0))  # 图例色块位置

            # 文本：文字放在色块右侧 8px
            label = scene.visuals.Text(
                text=str(a),
                color='white',
                anchor_x='left', anchor_y='center',
                parent=self.canvas.scene
            )
            label.transform = scene.transforms.STTransform(translate=(x0 + self._legend_swatch_size + 8, y, 0))

            self._legend_items[a] = {'swatch': swatch, 'text': label}  # 存储每个图例

    def update(self, states):
        for a in self.agents:
            # 更新当前位置
            self.positions[a] = states[a][:3]
            # 更新历史轨迹
            self.history_positions[a].append(self.positions[a])
            # 更新可视化标记位置
            self.markers[a].set_data(np.expand_dims(self.positions[a], 0), face_color=self.colors[a],
                                     size=self._marker_size)

            # 绘制历史轨迹
            if len(self.history_positions[a]) > 1:
                self.trajs[a].set_data(
                    np.array(self.history_positions[a]),
                    color=self.colors[a],
                    width=5,
                )

        self._frame_count += 1
        if (self._frame_count % self._autoframe_every) == 0:
            self._auto_frame()
            app.process_events()

    def reset(self):
        # 重置
        self.positions = {a: np.zeros(3, ) for a in self.agents}
        for a in self.agents:
            self.history_positions[a].clear()  # 清空历史轨迹
            self.markers[a].set_data(np.expand_dims(self.positions[a], 0), face_color=self.colors[a])  # 重置标记位置
            # self.trajs[a].set_data([], color=self.colors[a])  # 清空轨迹
            self.trajs[a].set_data([], [])

        # 重置相机平滑
        self._smooth_center = None
        self._frame_count = 0

    def run(self):
        app.run()

    def close(self):
        self.canvas.close()

    def _auto_frame(self):
        pts = []
        for a in self.agents:
            if len(self.history_positions[a]) > 0:
                pts.append(self.history_positions[a])
            else:
                pts.append(self.positions[a])

        if len(pts) == 0:
            return

        P = np.vstack(pts)  # (N, 3)

        # 计算 bounding box
        pmin = P.min(axis=0)
        pmax = P.max(axis=0)

        # 如果所有点共点，给一个小范围避免 set_range 出问题
        if np.allclose(pmin, pmax):
            pmin = pmin - 1.0
            pmax = pmax + 1.0

        # 加边距
        size = (pmax - pmin)
        pad = size * self._autoframe_margin
        bb_min = pmin - pad
        bb_max = pmax + pad

        # 平滑中心，让视角不抖
        center = 0.5 * (bb_min + bb_max)
        if self._smooth_center is None:
            self._smooth_center = center
        else:
            self._smooth_center = (1 - self._smooth_alpha) * self._smooth_center + self._smooth_alpha * center
        bb_half = 0.5 * (bb_max - bb_min)
        bb_min_s = self._smooth_center - bb_half
        bb_max_s = self._smooth_center + bb_half

        # 设置相机范围
        self.view.camera.center = tuple(self._smooth_center.tolist())
        # TurntableCamera 支持 set_range
        self.view.camera.set_range(x=(bb_min_s[0], bb_max_s[0]),
                                   y=(bb_min_s[1], bb_max_s[1]),
                                   z=(bb_min_s[2], bb_max_s[2]))
