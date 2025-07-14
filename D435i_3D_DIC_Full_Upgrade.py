# ADD JET COLORMAP FOR STRAIN/DISPLACEMENT VISUALIZATION
import sys
import numpy as np
import pyrealsense2 as rs
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph.opengl as gl

class DIC3DApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Intel D435i 3D DIC System')
        self.setGeometry(100, 100, 1600, 900)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('k')
        self.gl_widget.opts['distance'] = 2
        self.scatter = gl.GLScatterPlotItem(size=1)
        self.gl_widget.addItem(self.scatter)

        control_layout = QtWidgets.QHBoxLayout()
        self.mode_dropdown = QtWidgets.QComboBox()
        self.mode_dropdown.addItems(['Equivalent Strain', 'X-Displacement', 'Y-Displacement', 'Z-Displacement'])
        control_layout.addWidget(QtWidgets.QLabel('Mode:'))
        control_layout.addWidget(self.mode_dropdown)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.gl_widget, stretch=1)
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(main_layout)

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(cfg)
        self.pc = rs.pointcloud()
        self.align = rs.align(rs.stream.color)
        self.prev_gray = None

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                return
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            self.pc.map_to(color_frame)
            points = self.pc.calculate(depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            mask = np.isfinite(verts).all(axis=1)
            verts = verts[mask]

            if self.prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                dx = flow[..., 0]
                dy = flow[..., 1]
                dz = verts[:, 2] if verts.shape[0] > 0 else np.zeros_like(dx)

                mode = self.mode_dropdown.currentText()
                if mode == 'X-Displacement':
                    disp = np.abs(dx)
                elif mode == 'Y-Displacement':
                    disp = np.abs(dy)
                elif mode == 'Z-Displacement':
                    disp = np.abs(dz)
                else:
                    disp = np.sqrt(dx**2 + dy**2)

                disp_resized = cv2.resize(disp, (gray.shape[1], gray.shape[0])).flatten()
                disp_norm = cv2.normalize(disp_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cmap = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                colors = cmap.reshape(-1, 3) / 255.0
                colors = colors[mask]
            else:
                colors = np.full((verts.shape[0], 3), [0.3, 0.3, 0.3])

            if verts.shape[0] > 50000:
                idx = np.random.choice(verts.shape[0], 50000, replace=False)
                verts = verts[idx]
                colors = colors[idx]

            self.scatter.setData(pos=verts, color=colors, size=1)
            self.prev_gray = gray.copy()
        except Exception as e:
            print(f'[ERROR] {e}')

    def closeEvent(self, event):
        self.pipeline.stop()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = DIC3DApp()
    win.showMaximized()
    sys.exit(app.exec_())
