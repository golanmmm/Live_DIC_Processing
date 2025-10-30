
import sys
import numpy as np
import pyrealsense2 as rs
import cv2
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

def processing_worker(pipe, config):
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, config['resolution'][0], config['resolution'][1], rs.format.z16, config['fps'])
        cfg.enable_stream(rs.stream.color, config['resolution'][0], config['resolution'][1], rs.format.bgr8, config['fps'])
        pipeline.start(cfg)

        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        decimation = rs.decimation_filter()
        hole_filling = rs.hole_filling_filter()

        pc = rs.pointcloud()
        align = rs.align(rs.stream.color)
        prev_gray = None
        frame_counter = 0

        while True:
            if pipe.poll():
                msg = pipe.recv()
                if msg == 'STOP':
                    break
                else:
                    config.update(msg)

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            if config['filters_enabled']:
                spatial.set_option(rs.option.filter_magnitude, config['spatial_strength'])
                temporal.set_option(rs.option.filter_smooth_alpha, config['temporal_alpha'])
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                depth_frame = decimation.process(depth_frame)
                depth_frame = hole_filling.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            mask = np.isfinite(verts).all(axis=1)
            verts_masked = verts[mask]

            if verts_masked.shape[0] < 100:
                continue

            disp_stats = {'min': 0, 'max': 0, 'mean': 0}
            H, W = gray.shape
            if prev_gray is not None and verts.shape[0] == H * W:
                try:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    dx = flow[..., 0].flatten()
                    dy = flow[..., 1].flatten()
                    dz = verts[:, 2]

                    disp_xy = np.sqrt(dx ** 2 + dy ** 2)
                    if config['mode'] == 'X-Displacement':
                        disp = np.abs(dx)
                    elif config['mode'] == 'Y-Displacement':
                        disp = np.abs(dy)
                    elif config['mode'] == 'Z-Displacement':
                        disp = np.abs(dz)
                    elif config['mode'] == 'Equivalent Strain':
                        disp = np.sqrt(dx ** 2 + dy ** 2)
                    elif config['mode'] == 'Total Displacement':
                        disp = np.sqrt(disp_xy ** 2 + dz ** 2)
                    else:
                        disp = disp_xy

                    disp = disp[mask]
                    disp_stats['min'] = float(np.min(disp))
                    disp_stats['max'] = float(np.max(disp))
                    disp_stats['mean'] = float(np.mean(disp))

                    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cmap = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                    colors = cmap.reshape(-1, 3) / 255.0

                except Exception as e:
                    print(f'[Worker Warning] {str(e)}')
                    colors = np.full((verts_masked.shape[0], 3), [0.3, 0.3, 0.3])
            else:
                colors = np.full((verts_masked.shape[0], 3), [0.3, 0.3, 0.3])

            prev_gray = gray.copy()
            frame_counter += 1

            if frame_counter % config['mesh_update_rate'] == 0:
                reduced_verts = verts_masked[::config['downsample_factor']]
                reduced_colors = colors[::config['downsample_factor']]
                pipe.send(('DATA', reduced_verts, reduced_colors, disp_stats))

        pipeline.stop()
    except Exception as e:
        pipe.send(('ERROR', str(e)))


class DIC3DApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Intel D435i 3D DIC System (Multiprocessing)')
        self.setGeometry(100, 100, 1600, 900)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('k')
        self.gl_widget.opts['distance'] = 2
        self.scatter = gl.GLScatterPlotItem(size=1)
        self.gl_widget.addItem(self.scatter)

        grid = gl.GLGridItem()
        grid.setSize(0.5, 0.5)
        grid.setSpacing(0.05, 0.05)
        self.gl_widget.addItem(grid)

        self.stats_label = QtWidgets.QLabel(self)
        self.stats_label.setStyleSheet("color: white; font-size: 14px; background-color: rgba(0,0,0,0%)")
        self.stats_label.move(20, 20)

        control_layout = QtWidgets.QHBoxLayout()
        self.mode_dropdown = QtWidgets.QComboBox()
        self.mode_dropdown.addItems(['Equivalent Strain', 'X-Displacement', 'Y-Displacement', 'Z-Displacement', 'Total Displacement'])
        control_layout.addWidget(QtWidgets.QLabel('Mode:'))
        control_layout.addWidget(self.mode_dropdown)

        self.filter_checkbox = QtWidgets.QCheckBox('Enable Filters')
        self.filter_checkbox.setChecked(True)
        control_layout.addWidget(self.filter_checkbox)

        self.spatial_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.spatial_slider.setMinimum(1)
        self.spatial_slider.setMaximum(5)
        self.spatial_slider.setValue(3)
        control_layout.addWidget(QtWidgets.QLabel('Spatial Strength'))
        control_layout.addWidget(self.spatial_slider)

        self.temporal_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.temporal_slider.setMinimum(0)
        self.temporal_slider.setMaximum(100)
        self.temporal_slider.setValue(40)
        control_layout.addWidget(QtWidgets.QLabel('Temporal Alpha'))
        control_layout.addWidget(self.temporal_slider)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.gl_widget)
        self.setLayout(main_layout)

        self.parent_conn, child_conn = mp.Pipe()
        self.config = mp.Manager().dict({
            'resolution': (640, 480),
            'fps': 30,
            'filters_enabled': True,
            'spatial_strength': 3,
            'temporal_alpha': 0.4,
            'mode': 'Total Displacement',
            'mesh_update_rate': 3,
            'downsample_factor': 10
        })

        self.proc = mp.Process(target=processing_worker, args=(child_conn, self.config))
        self.proc.start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.pull_data)
        self.timer.start(30)

        self.mode_dropdown.currentTextChanged.connect(self.update_mode)
        self.filter_checkbox.stateChanged.connect(self.update_filter_toggle)
        self.spatial_slider.valueChanged.connect(self.update_spatial_strength)
        self.temporal_slider.valueChanged.connect(self.update_temporal_alpha)

    def pull_data(self):
        while self.parent_conn.poll():
            data = self.parent_conn.recv()
            if isinstance(data, tuple) and isinstance(data[0], str) and data[0] == 'ERROR':
                QtWidgets.QMessageBox.critical(self, "Error", f"Worker error: {data[1]}")
                continue
            if isinstance(data, tuple) and data[0] == 'DATA':
                verts, colors, stats = data[1], data[2], data[3]
                self.scatter.setData(pos=verts, color=colors, size=1)
                self.stats_label.setText(f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Mean: {stats['mean']:.4f}")

    def update_mode(self, mode):
        self.config['mode'] = mode
        self.parent_conn.send({'mode': mode})

    def update_filter_toggle(self, state):
        self.config['filters_enabled'] = bool(state)
        self.parent_conn.send({'filters_enabled': bool(state)})

    def update_spatial_strength(self, value):
        self.config['spatial_strength'] = value
        self.parent_conn.send({'spatial_strength': value})

    def update_temporal_alpha(self, value):
        alpha = value / 100.0
        self.config['temporal_alpha'] = alpha
        self.parent_conn.send({'temporal_alpha': alpha})

    def closeEvent(self, event):
        try:
            self.parent_conn.send('STOP')
            self.proc.join(timeout=5)
        except:
            pass
        event.accept()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    app = QtWidgets.QApplication(sys.argv)
    win = DIC3DApp()
    win.showMaximized()
    sys.exit(app.exec_())
