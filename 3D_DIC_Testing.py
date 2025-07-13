import sys
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5 import QtCore, QtGui, QtWidgets

DEPTH_SCALE = 1000.0
FACET_SIZE = 21
STEP_SIZE = 30
REF_ACCUM_FRAMES = 5  # number of frames to accumulate reference

class RealSenseThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    def __init__(self, fps):
        super().__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.fps = fps
        self.started = False
        self.running = True

    def run(self):
        try:
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.y8, self.fps)
            align = rs.align(rs.stream.color)
            self.pipeline.start(self.config)
            self.started = True
            sensor = self.pipeline.get_active_profile().get_device().first_color_sensor()
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            sensor.set_option(rs.option.power_line_frequency, 1)
            while self.running:
                frames = self.pipeline.wait_for_frames()
                aligned = align.process(frames)
                depth = np.asanyarray(aligned.get_depth_frame().get_data()) / DEPTH_SCALE
                color = np.asanyarray(aligned.get_color_frame().get_data())
                self.frame_ready.emit(color, depth)
        except Exception as e:
            print(f"[ERROR] RealSense thread error: {e}")
        finally:
            if self.started:
                try:
                    self.pipeline.stop()
                except Exception as e:
                    print(f"[ERROR] Failed to stop RealSense pipeline: {e}")

    def stop(self):
        self.running = False
        self.wait()

class DIC3DApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Live 3D DIC - Intel D435i')
        self.img_label = QtWidgets.QLabel()
        self.img_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.img_label.setAlignment(QtCore.Qt.AlignCenter)

        self.display_selector = QtWidgets.QComboBox()
        self.display_selector.addItems(['Displacement magnitude', 'Strain magnitude'])

        self.fps_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fps_slider.setMinimum(6)
        self.fps_slider.setMaximum(60)
        self.fps_slider.setValue(30)
        self.allowed_fps = [6, 15, 30, 60]

        self.fps_timer = QtCore.QTimer()
        self.fps_timer.setSingleShot(True)
        self.fps_slider.valueChanged.connect(self.schedule_fps_update)
        self.fps_timer.timeout.connect(self.update_fps)

        self.roi_button = QtWidgets.QPushButton('Select ROI')
        self.roi_button.clicked.connect(self.select_roi)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.img_label)
        layout.addWidget(QtWidgets.QLabel('Display Mode:'))
        layout.addWidget(self.display_selector)
        layout.addWidget(QtWidgets.QLabel('FPS (allowed: 6,15,30,60):'))
        layout.addWidget(self.fps_slider)
        layout.addWidget(self.roi_button)
        self.setLayout(layout)

        self.ref_gray = None
        self.ref_accum = None
        self.ref_count = 0
        self.ref_pts = None
        self.roi = None
        self.thread = RealSenseThread(fps=self.fps_slider.value())
        self.thread.frame_ready.connect(self.process_frame)
        self.thread.start()

    def schedule_fps_update(self):
        self.fps_timer.start(1000)

    def update_fps(self):
        fps_value = min(self.allowed_fps, key=lambda x: abs(x - self.fps_slider.value()))
        self.fps_slider.setEnabled(False)
        try:
            self.thread.stop()
            time.sleep(1)
            self.thread = RealSenseThread(fps=fps_value)
            self.thread.frame_ready.connect(self.process_frame)
            self.thread.start()
        except Exception as e:
            print(f"[ERROR] Failed to restart RealSense: {e}")
        self.fps_slider.setEnabled(True)

    def select_roi(self):
        if hasattr(self, 'latest_color'):
            r = cv2.selectROI('Select ROI', self.latest_color, False, False)
            cv2.destroyAllWindows()
            if r[2] > 0 and r[3] > 0:
                self.roi = r
                self.ref_gray = None
                self.ref_accum = None
                self.ref_count = 0
                self.ref_pts = None

    def auto_seed_points(self, gray_img):
        h, w = gray_img.shape
        pts = []
        for yy in range(FACET_SIZE, h - FACET_SIZE, STEP_SIZE):
            for xx in range(FACET_SIZE, w - FACET_SIZE, STEP_SIZE):
                pts.append([xx, yy])
        return np.array(pts, np.float32).reshape(-1, 1, 2)

    def process_frame(self, color_img, depth_img):
        self.latest_color = color_img.copy()
        if self.roi:
            x, y, rw, rh = [int(v) for v in self.roi]
            color_img = color_img[y:y+rh, x:x+rw]
            depth_img = depth_img[y:y+rh, x:x+rw]
        gray = color_img
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if self.ref_count < REF_ACCUM_FRAMES:
            if self.ref_accum is None:
                self.ref_accum = gray.astype(np.float32)
            else:
                self.ref_accum += gray.astype(np.float32)
            self.ref_count += 1
            if self.ref_count == REF_ACCUM_FRAMES:
                self.ref_gray = (self.ref_accum / REF_ACCUM_FRAMES).astype(np.uint8)
                self.ref_pts = self.auto_seed_points(self.ref_gray)
            return

        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        new_pts, st, err = cv2.calcOpticalFlowPyrLK(self.ref_gray, gray, self.ref_pts, None, **lk_params)

        magnitudes = np.zeros(len(self.ref_pts))
        for i, (old, new, status) in enumerate(zip(self.ref_pts.reshape(-1, 2), new_pts.reshape(-1, 2), st.reshape(-1))):
            if status:
                dx, dy = new[0] - old[0], new[1] - old[1]
                if self.display_selector.currentText() == 'Displacement magnitude':
                    magnitudes[i] = np.sqrt(dx ** 2 + dy ** 2)
                else:
                    magnitudes[i] = (abs(dx) + abs(dy)) / FACET_SIZE
                cv2.arrowedLine(overlay, (int(old[0]), int(old[1])), (int(new[0]), int(new[1])), (255, 0, 0), 1, tipLength=0.3)

        heatmap = np.zeros(gray.shape, np.float32)
        for pt, mag in zip(self.ref_pts.reshape(-1, 2), magnitudes):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
                heatmap[y, x] = mag
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=5, sigmaY=5)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap_color, 0.5, overlay, 0.5, 0)

        qimg = QtGui.QImage(overlay.data, overlay.shape[1], overlay.shape[0], overlay.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def closeEvent(self, e):
        self.thread.stop()
        e.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = DIC3DApp()
    window.resize(1280, 800)
    window.show()
    sys.exit(app.exec_())
