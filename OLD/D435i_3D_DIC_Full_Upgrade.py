import sys
import numpy as np
import pyrealsense2 as rs
import cv2
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class CameraWorker(QtCore.QThread):
    data_ready = QtCore.pyqtSignal(object, object, object, object, dict)  # verts, colors, seeds, facet_mesh_data, stats

    def __init__(self):
        super().__init__()
        self.running = True
        self.roi = None
        self.seed_density = 20
        self.latest_frame = None
        self.mode = 'Total Displacement'
        self.resolution = (640, 480)
        self.fps = 30

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
            cfg.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps)
            cfg.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, self.fps)
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
                self.latest_frame = color_image.copy()
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                pc.map_to(color_frame)
                points = pc.calculate(depth_frame)
                verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
                mask = np.isfinite(verts).all(axis=1)
                verts_masked = verts[mask]

                disp_stats = {'min': 0, 'max': 0, 'mean': 0}

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    dx = flow[..., 0]
                    dy = flow[..., 1]
                    dz_full = verts[:, 2]
                    dz = dz_full[mask] if verts_masked.shape[0] > 0 else np.zeros_like(dx.flatten())

                    if self.mode == 'X-Displacement':
                        disp = np.abs(dx)
                    elif self.mode == 'Y-Displacement':
                        disp = np.abs(dy)
                    elif self.mode == 'Z-Displacement':
                        disp = np.abs(dz).reshape(-1)
                    elif self.mode == 'Equivalent Strain':
                        disp = np.sqrt(dx ** 2 + dy ** 2)
                    elif self.mode == 'Total Displacement':
                        disp_xy = np.sqrt(dx ** 2 + dy ** 2)
                        disp = np.sqrt(disp_xy.flatten()[mask] ** 2 + dz.flatten() ** 2)
                    else:
                        disp = np.sqrt(dx ** 2 + dy ** 2)

                    disp_stats['min'] = float(np.min(disp))
                    disp_stats['max'] = float(np.max(disp))
                    disp_stats['mean'] = float(np.mean(disp))

                    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cmap = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                    colors = cmap.reshape(-1, 3) / 255.0
                else:
                    colors = np.full((verts_masked.shape[0], 3), [0.3, 0.3, 0.3])

                try:
                    verts_img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.float32)
                    verts_img.reshape(-1, 3)[mask] = verts_masked
                except:
                    verts_img = np.zeros((self.resolution[1], self.resolution[0], 3))

                if self.roi:
                    x, y, w, h = self.roi
                    verts_crop = verts_img[y:y + h, x:x + w, :].reshape(-1, 3)
                    colors_crop = colors.reshape(-1, 3)
                else:
                    verts_crop = verts_masked
                    colors_crop = colors

                seeds = []
                quad_verts = []
                quad_faces = []
                step_y = max(1, self.resolution[1] // self.seed_density)
                step_x = max(1, self.resolution[0] // self.seed_density)
                count = 0
                patch_size = 0.002  # ~2mm patch
                for iy in range(0, self.resolution[1], step_y):
                    for ix in range(0, self.resolution[0], step_x):
                        if self.roi:
                            if not (x <= ix < x + w and y <= iy < y + h):
                                continue
                        pt = verts_img[iy, ix, :]
                        if np.isfinite(pt).all():
                            seeds.append(pt)
                            # Create quad vertices around point
                            v0 = pt + [-patch_size, -patch_size, 0]
                            v1 = pt + [patch_size, -patch_size, 0]
                            v2 = pt + [patch_size, patch_size, 0]
                            v3 = pt + [-patch_size, patch_size, 0]
                            base_idx = count * 4
                            quad_verts.extend([v0, v1, v2, v3])
                            quad_faces.extend([[base_idx, base_idx + 1, base_idx + 2],
                                               [base_idx, base_idx + 2, base_idx + 3]])
                            count += 1
                seeds = np.array(seeds)
                quad_verts = np.array(quad_verts)
                quad_faces = np.array(quad_faces)

                self.data_ready.emit(verts_crop, colors_crop, seeds, (quad_verts, quad_faces), disp_stats)
                prev_gray = gray.copy()

            pipeline.stop()

        except Exception as e:
            print(f'[ERROR in CameraWorker] {e}')

    def stop(self):
        self.running = False
        self.wait()

    def get_last_frame(self):
        if self.latest_frame is not None:
            return True, self.latest_frame.copy()
        else:
            return False, None


class DIC3DApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Intel D435i 3D DIC System')
        self.setGeometry(100, 100, 1600, 900)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('k')
        self.gl_widget.opts['distance'] = 2
        self.scatter = gl.GLScatterPlotItem(size=1)
        self.seeds_scatter = gl.GLScatterPlotItem(size=5, color=(1, 1, 1, 1))
        self.facets_mesh = gl.GLMeshItem(vertexes=np.array([[0, 0, 0]]),
                                         faces=np.array([[0, 0, 0]]),
                                         color=(1, 1, 1, 1),
                                         smooth=False,
                                         drawEdges=False,
                                         drawFaces=True)
        self.gl_widget.addItem(self.scatter)
        self.gl_widget.addItem(self.seeds_scatter)
        self.gl_widget.addItem(self.facets_mesh)

        grid = gl.GLGridItem()
        grid.setSize(0.5, 0.5)
        grid.setSpacing(0.05, 0.05)
        grid.setDepthValue(-10)
        self.gl_widget.addItem(grid)

        self.stats_label = QtWidgets.QLabel(self)
        self.stats_label.setStyleSheet("color: white; font-size: 14px; background-color: rgba(0,0,0,0%)")
        self.stats_label.move(20, 20)

        control_layout = QtWidgets.QHBoxLayout()
        self.mode_dropdown = QtWidgets.QComboBox()
        self.mode_dropdown.addItems(['Equivalent Strain', 'X-Displacement', 'Y-Displacement', 'Z-Displacement', 'Total Displacement'])
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

        self.res_dropdown = QtWidgets.QComboBox()
        self.res_dropdown.addItems(['640x480 @30fps', '848x480 @60fps', '1280x720 @30fps', '1920x1080 @15fps'])
        control_layout.addWidget(QtWidgets.QLabel('Resolution'))
        control_layout.addWidget(self.res_dropdown)
        self.res_dropdown.currentTextChanged.connect(self.update_resolution)

        self.show_seeds_checkbox = QtWidgets.QCheckBox('Show Seed Points')
        self.show_seeds_checkbox.setChecked(True)
        self.show_facets_checkbox = QtWidgets.QCheckBox('Show Facets')
        self.show_facets_checkbox.setChecked(True)
        control_layout.addWidget(self.show_seeds_checkbox)
        control_layout.addWidget(self.show_facets_checkbox)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.gl_widget, stretch=1)
        self.setLayout(main_layout)

        self.worker = CameraWorker()
        self.worker.data_ready.connect(self.update_view)
        self.worker.start()

        self.mode_dropdown.currentTextChanged.connect(self.update_mode)
        self.update_mode(self.mode_dropdown.currentText())

    def update_view(self, verts, colors, seeds, facet_mesh_data, stats):
        self.scatter.setData(pos=verts, color=colors, size=1)

        if self.show_seeds_checkbox.isChecked() and seeds.size > 0:
            self.seeds_scatter.setData(pos=seeds, size=5)
        else:
            self.seeds_scatter.setData(pos=np.zeros((1, 3)), size=0)

        if self.show_facets_checkbox.isChecked() and facet_mesh_data[0].shape[0] > 0:
            verts_mesh, faces_mesh = facet_mesh_data
            self.facets_mesh.setMeshData(vertexes=verts_mesh, faces=faces_mesh, color=(1, 1, 1, 1))
        else:
            self.facets_mesh.setMeshData(vertexes=np.array([[0, 0, 0]]), faces=np.array([[0, 0, 0]]))

        stats_text = f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Mean: {stats['mean']:.4f}"
        self.stats_label.setText(stats_text)

    def select_roi(self):
        ret, frame = self.worker.get_last_frame()
        if not ret:
            QtWidgets.QMessageBox.warning(self, "Warning", "No frame available for ROI selection.")
            return
        roi = cv2.selectROI("Select ROI", frame, False, False)
        cv2.destroyWindow("Select ROI")
        if roi != (0, 0, 0, 0):
            self.worker.roi = roi
        else:
            self.worker.roi = None

    def update_seed_density(self, value):
        self.worker.seed_density = value

    def update_mode(self, mode):
        self.worker.mode = mode

    def update_resolution(self, text):
        res_map = {
            '640x480 @30fps': (640, 480, 30),
            '848x480 @60fps': (848, 480, 60),
            '1280x720 @30fps': (1280, 720, 30),
            '1920x1080 @15fps': (1920, 1080, 15),
        }
        res, h, fps = res_map[text]
        self.worker.resolution = (res, h)
        self.worker.fps = fps
        self.worker.stop()
        self.worker = CameraWorker()
        self.worker.resolution = (res, h)
        self.worker.fps = fps
        self.worker.data_ready.connect(self.update_view)
        self.worker.start()

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = DIC3DApp()
    win.showMaximized()
    sys.exit(app.exec_())
