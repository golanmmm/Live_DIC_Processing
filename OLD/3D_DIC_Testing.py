# UPGRADE: Add Z-Displacement display, RGB overlay, and seed point density control
import sys
import numpy as np
import pyrealsense2 as rs
import cv2
from PyQt5 import QtWidgets, QtCore
import pyqtgraph.opengl as gl

class RealSense3DDIC(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Intel RealSense 3D DIC Viewer')
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('k')
        self.gl_widget.opts['distance'] = 2
        self.gl_widget.orbit(0, 0)
        self.scatter = gl.GLScatterPlotItem(size=1)
        self.seeds_visual = gl.GLScatterPlotItem(size=6, color=(1,0.5,0,1))
        self.gl_widget.addItem(self.scatter)
        self.gl_widget.addItem(self.seeds_visual)

        layout = QtWidgets.QVBoxLayout()
        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.addItems(['Equivalent Strain', 'X-Strain', 'Y-Strain', 'Z-Displacement', 'X-Displacement', 'Y-Displacement', 'RGB Live Overlay'])
        self.dropdown.currentIndexChanged.connect(self.change_mode)
        layout.addWidget(self.dropdown)

        self.seed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.seed_slider.setMinimum(5)
        self.seed_slider.setMaximum(100)
        self.seed_slider.setValue(30)
        self.seed_slider.valueChanged.connect(self.change_seed_density)
        layout.addWidget(QtWidgets.QLabel('Seed Point Density'))
        layout.addWidget(self.seed_slider)

        layout.addWidget(self.gl_widget, stretch=1)
        self.setLayout(layout)

        self.gl_widget.setMouseTracking(True)
        self.gl_widget.mousePressEvent = self.mouse_press
        self.gl_widget.mouseMoveEvent = self.mouse_move
        self.gl_widget.mouseReleaseEvent = self.mouse_release
        self._mouse_pressed = False
        self._last_pos = None

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(cfg)

        sensor = self.profile.get_device().query_sensors()[0]
        if sensor.supports(rs.option.power_line_frequency):
            sensor.set_option(rs.option.power_line_frequency, rs.power_line_frequency.fifty_hz)

        self.pc = rs.pointcloud()
        self.align = rs.align(rs.stream.color)
        self.prev_frame = None
        self.mode = 'Equivalent Strain'
        self.seed_step = 30
        self.seed_history = []
        self.smoothing_window = 5

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(30)

    def mouse_press(self, event):
        self._mouse_pressed = True
        self._last_pos = event.pos()

    def mouse_move(self, event):
        if self._mouse_pressed:
            dx = event.x() - self._last_pos.x()
            dy = event.y() - self._last_pos.y()
            self.gl_widget.orbit(-dx, dy)
            self._last_pos = event.pos()

    def mouse_release(self, event):
        self._mouse_pressed = False

    def change_mode(self, index):
        self.mode = self.dropdown.currentText()

    def change_seed_density(self, value):
        self.seed_step = value

    def auto_seed_points(self, shape):
        h, w = shape
        pts = np.array([[x, y] for y in range(0, h, self.seed_step) for x in range(0, w, self.seed_step)], np.float32)
        return pts

    def update_data(self):
        try:
            frames = self.pipeline.wait_for_frames()
        except Exception as e:
            print(f'[ERROR] Frame grab failed: {e}')
            return

        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        self.points = self.pc.calculate(depth_frame)
        verts = np.asanyarray(self.points.get_vertices()).view(np.float32).reshape(-1, 3)

        seed_pts = self.auto_seed_points(gray_image.shape)
        seed_3d = []
        for x, y in seed_pts:
            idx = int(y) * gray_image.shape[1] + int(x)
            if idx < verts.shape[0]:
                seed_3d.append(verts[idx])
        seed_3d = np.array(seed_3d)

        self.seed_history.append(seed_3d)
        if len(self.seed_history) > self.smoothing_window:
            self.seed_history.pop(0)
        smoothed_seeds = np.mean(np.array(self.seed_history), axis=0)

        if self.prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev_frame, gray_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx = cv2.GaussianBlur(flow[..., 0], (3, 3), 0)
            dy = cv2.GaussianBlur(flow[..., 1], (3, 3), 0)
            dz = verts[:, 2].reshape(gray_image.shape)
            exx = np.gradient(dx, axis=1)
            eyy = np.gradient(dy, axis=0)
            equiv_strain = np.sqrt(exx ** 2 + eyy ** 2)

            if self.mode == 'Equivalent Strain':
                selected_map = np.abs(equiv_strain)
            elif self.mode == 'X-Strain':
                selected_map = np.abs(exx)
            elif self.mode == 'Y-Strain':
                selected_map = np.abs(eyy)
            elif self.mode == 'X-Displacement':
                selected_map = np.abs(dx)
            elif self.mode == 'Y-Displacement':
                selected_map = np.abs(dy)
            elif self.mode == 'Z-Displacement':
                selected_map = np.abs(dz)
            elif self.mode == 'RGB Live Overlay':
                cv2.imshow('RGB Live Feed', color_image)
                cv2.waitKey(1)
                selected_map = np.zeros_like(dx)
            else:
                selected_map = np.sqrt(dx**2 + dy**2)

            selected_map = cv2.GaussianBlur(selected_map, (3, 3), 0)
            selected_flat = cv2.resize(selected_map, (depth_image.shape[1], depth_image.shape[0])).flatten()
            selected_norm = cv2.normalize(selected_flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colormap = cv2.applyColorMap(selected_norm, cv2.COLORMAP_JET)
            colormap_rgb = colormap.reshape(-1, 3) / 255.0
        else:
            colormap_rgb = np.full((verts.shape[0], 3), [0.3, 0.3, 0.3])

        mask = np.isfinite(verts).all(axis=1)
        verts = verts[mask]
        colormap_rgb = colormap_rgb[mask]

        if verts.shape[0] > 80000:
            idx = np.random.choice(verts.shape[0], 80000, replace=False)
            verts = verts[idx]
            colormap_rgb = colormap_rgb[idx]

        self.scatter.setData(pos=verts, color=colormap_rgb, size=1)
        if smoothed_seeds.size > 0:
            self.seeds_visual.setData(pos=smoothed_seeds, size=6, color=(1,0.5,0,1))
        self.prev_frame = gray_image.copy()

    def closeEvent(self, event):
        try:
            self.pipeline.stop()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f'[WARNING] Failed to stop pipeline cleanly: {e}')
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    viewer = RealSense3DDIC()
    viewer.resize(1280, 800)
    viewer.show()
    sys.exit(app.exec_())
