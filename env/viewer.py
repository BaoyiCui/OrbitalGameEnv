from collections import deque
import numpy as np
from vispy import app, scene


class Viewer:
    def __init__(
            self,
            width,
            height,
            agents,
            max_history=20
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

        # for marker in self.markers.values():
        #     marker.set_data([self.positions], face_color=self.colors)
        #     self.view.add(marker)
        for a in self.agents:
            self.markers[a].set_data(np.expand_dims(self.positions[a], 0), face_color=self.colors[a], size=10)
            self.view.add(self.markers[a])
            self.view.add(self.trajs[a])

        self.view.camera = scene.cameras.TurntableCamera(fov=60, elevation=20, azimuth=30)

    def update(self, states):
        for a in self.agents:
            # 更新当前位置
            self.positions[a] = states[a][:3]
            # 更新历史轨迹
            self.history_positions[a].append(self.positions[a])
            # 更新可视化标记位置
            self.markers[a].set_data(np.expand_dims(self.positions[a], 0), face_color=self.colors[a], size=10)
            # 绘制历史轨迹
            if len(self.history_positions[a]) > 1:
                self.trajs[a].set_data(
                    self.history_positions[a],
                    color=self.colors[a],
                    width=2,
                )

    def reset(self):
        # TODO: 重置
        self.positions = {a: np.zeros(3, ) for a in self.agents}
        for a in self.agents:
            self.history_positions[a].clear()  # 清空历史轨迹
            self.markers[a].set_data(np.expand_dims(self.positions[a], 0), face_color=self.colors[a])  # 重置标记位置
            self.trajs[a].set_data([], color=self.colors[a])  # 清空轨迹

    def run(self):
        app.run()

    def close(self):
        self.canvas.close()
