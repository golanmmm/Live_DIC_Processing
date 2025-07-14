import sys
import numpy as np
import pyrealsense2 as rs
import cv2
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class CameraWorker(QtCore.QThread):
    data_ready = QtCore.pyqtSignal(object, object, object)  # verts, colors, seed points

    def __init__(self):
        super().__init__()
        self.running = True
        self.roi = None
        self.seed_density = 20

    def run(self):
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                QtWidgets.QMessageBox.critical(None, "Error", "❗ No Intel RealSense device detected.")
                sys.exit(1)
            dev = devices[0]
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
            print(f'[INFO] Connected over {usb_type}')
            if '2.0' in usb_type:
                QtWidgets.QMessageBox.critical(None, "Error", "❗ Intel RealSense D435i must be connected over USB 3.0")
                sys.exit(1)

            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(cfg)
            pc = rs.pointcloud()
            align = rs.align(rs.stream.color)
            prev_gray = None

            while self.running:
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                pc.map_to(color_frame)
                points = pc.calculate(depth_frame)
                verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
                mask = np.isfinite(verts).all(axis=1)
                verts = verts[mask]

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    dx = flow[..., 0]
                    dy = flow[..., 1]
                    dz = verts[:, 2] if verts.shape[0] > 0 else np.zeros_like(dx)
                    disp = np.sqrt(dx ** 2 + dy ** 2)
                    disp_resized = cv2.resize(disp, (gray.shape[1], gray.shape[0])).flatten()
                    disp_norm = cv2.normalize(disp_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cmap = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                    colors = cmap.reshape(-1, 3) / 255.0
                    colors = colors[mask]
                else:
                    colors = np.full((verts.shape[0], 3), [0.3, 0.3, 0.3])

                # Reshape to image grid
                try:
                    verts_img = verts.reshape((480, 640, 3))
                    colors_img = colors.reshape((480, 640, 3))
                except:
                    verts_img = np.zeros((480, 640, 3))
                    colors_img = np.zeros((480, 640, 3))

                # Apply ROI crop if set
                if self.roi:
                    x, y, w, h = self.roi
                    verts_crop = verts_img[y:y + h, x:x + w, :].reshape(-1, 3)
                    colors_crop = colors_img[y:y + h, x:x + w, :].reshape(-1, 3)
                else:
                    verts_crop = verts
                    colors_crop = colors

                # Seed points generation
                seeds = []
                step_y = 480 // self.seed_density
                step_x = 640 // self.seed_density
                for iy in range(0, 480, step_y):
                    for ix in range(0, 640, step_x):
                        if self.roi:
                            if not (x <= ix < x + w and y <= iy < y + h):
                                continue
                        pt = verts_img[iy, ix, :]
                        if np.isfinite(pt).all():
                            seeds.append(pt)
                seeds = np.array(seeds)

                self.data_ready.emit(verts_crop, colors_crop, seeds)
                prev_gray = gray.copy()

            pipeline.stop()

        except Exception as e:
            print(f'[ERROR in CameraWorker] {e}')

    def stop(self):
        self.running = False
        self.wait()

    def get_last_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None
            color_image = np.asanyarray(color_frame.get_data())
            return True, color_image
        except:
            return False, None


class DIC3DApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Intel D435i 3D DIC System')
        self.setGeometry(100, 100, 1600, 900)

        # 3D view
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('k')
        self.gl_widget.opts['distance'] = 2
        self.scatter = gl.GLScatterPlotItem(size=1)
        self.seeds_scatter = gl.GLScatterPlotItem(size=5, color=(1, 1, 1, 1))  # white seeds
        self.gl_widget.addItem(self.scatter)
        self.gl_widget.addItem(self.seeds_scatter)

        # Grid at origin
        grid = gl.GLGridItem()
        grid.setSize(0.5, 0.5)
        grid.setSpacing(0.05, 0.05)
        grid.setDepthValue(-10)
        self.gl_widget.addItem(grid)

        # X, Y, Z axes
        axis_length = 0.1
        x_line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]),
                                   color=pg.glColor('r'), width=3, antialias=True)
        y_line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]),
                                   color=pg.glColor('g'), width=3, antialias=True)
        z_line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]),
                                   color=pg.glColor('b'), width=3, antialias=True)
        self.gl_widget.addItem(x_line)
        self.gl_widget.addItem(y_line)
        self.gl_widget.addItem(z_line)

        # Labels overlay
        self.x_label = QtWidgets.QLabel('X', self)
        self.x_label.setStyleSheet("color: red; font-size: 14px; background-color: rgba(0,0,0,0%)")
        self.x_label.move(50, 50)
        self.y_label = QtWidgets.QLabel('Y', self)
        self.y_label.setStyleSheet("color: green; font-size: 14px; background-color: rgba(0,0,0,0%)")
        self.y_label.move(80, 50)
        self.z_label = QtWidgets.QLabel('Z', self)
        self.z_label.setStyleSheet("color: blue; font-size: 14px; background-color: rgba(0,0,0,0%)")
        self.z_label.move(110, 50)

        # Controls
        control_layout = QtWidgets.QHBoxLayout()
        self.mode_dropdown = QtWidgets.QComboBox()
        self.mode_dropdown.addItems(['Equivalent Strain', 'X-Displacement', 'Y-Displacement', 'Z-Displacement'])
        control_layout.addWidget(QtWidgets.QLabel('Mode:'))
        control_layout.addWidget(self.mode_dropdown)

        self.roi_button = QtWidgets.QPushButton('Select ROI')
        control_layout.addWidget(self.roi_button)
        self.roi_button.clicked.connect(self.select_roi)

        self.density_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.density_slider.setMinimum(5)
        self.density_slider.setMaximum(100)
        self.density_slider.setValue(20)
        self.density_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.density_slider.setTickInterval(5)
        control_layout.addWidget(QtWidgets.QLabel('Seed Density'))
        control_layout.addWidget(self.density_slider)
        self.density_slider.valueChanged.connect(self.update_seed_density)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.gl_widget, stretch=1)
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(main_layout)

        # Worker thread
        self.worker = CameraWorker()
        self.worker.data_ready.connect(self.update_view)
        self.worker.start()

    def update_view(self, verts, colors, seeds):
        self.scatter.setData(pos=verts, color=colors, size=1)
        if seeds.size > 0:
            self.seeds_scatter.setData(pos=seeds, size=5)
        else:
            self.seeds_scatter.setData(pos=np.zeros((1, 3)), size=0)

    def select_roi(self):
        ret, frame = self.worker.get_last_frame()
        if not ret:
            QtWidgets.QMessageBox.warning(self, "Warning", "No frame available for ROI selection.")
            return
        roi = cv2.selectROI("Select ROI", frame, False, False)
        cv2.destroyWindow("Select ROI")
        if roi != (0, 0, 0, 0):
            self.worker.roi = roi
            print(f'[INFO] ROI set to: {roi}')
        else:
            self.worker.roi = None

    def update_seed_density(self, value):
        self.worker.seed_density = value
        print(f'[INFO] Seed point density set to: {value}')

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = DIC3DApp()
    win.showMaximized()
    sys.exit(app.exec_())
