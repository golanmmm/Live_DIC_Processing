import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


class DICLive(QtWidgets.QMainWindow):
    def __init__(self, rtsp_url):
        super().__init__()
        self.setWindowTitle("Live DIC Strain & Poisson GUI")
        self.rtsp = cv2.VideoCapture(rtsp_url)
        if not self.rtsp.isOpened():
            raise RuntimeError("Cannot open RTSP stream")

        # UI elements
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.info_label = QtWidgets.QLabel("press 'Set Reference' to begin", alignment=QtCore.Qt.AlignCenter)
        self.btn_ref = QtWidgets.QPushButton("Set Reference")
        self.scale_input = QtWidgets.QLineEdit("0.1")  # mm per pixel
        self.L0_input = QtWidgets.QLineEdit("50.0")  # gauge length in mm
        self.W0_input = QtWidgets.QLineEdit("10.0")  # gauge width in mm

        form = QtWidgets.QFormLayout()
        form.addRow("scale (mm/pixel):", self.scale_input)
        form.addRow("gauge length L₀ (mm):", self.L0_input)
        form.addRow("gauge width  W₀ (mm):", self.W0_input)

        ctrl = QtWidgets.QVBoxLayout()
        ctrl.addLayout(form)
        ctrl.addWidget(self.btn_ref)
        ctrl.addWidget(self.info_label)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.video_label, 3)
        layout.addLayout(ctrl, 1)

        central = QtWidgets.QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # state
        self.ref_gray = None
        self.ref_pts = None

        # signals
        self.btn_ref.clicked.connect(self.set_reference)

        # timer for live update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 fps

    def set_reference(self):
        ret, frame = self.rtsp.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect good features to track
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if pts is None:
            self.info_label.setText("no features found – try better contrast")
            return
        self.ref_gray = gray
        self.ref_pts = pts
        self.info_label.setText(f"reference set: {len(pts)} points")

    def update_frame(self):
        ret, frame = self.rtsp.read()
        if not ret:
            return
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.ref_gray is not None and self.ref_pts is not None:
            # track points from reference → current
            new_pts, st, err = cv2.calcOpticalFlowPyrLK(
                self.ref_gray, gray, self.ref_pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            good = st.flatten() == 1
            ref = self.ref_pts[good].reshape(-1, 2)
            cur = new_pts[good].reshape(-1, 2)
            disp = cur - ref

            if len(disp):
                # parse user inputs
                try:
                    scale = float(self.scale_input.text())
                    L0 = float(self.L0_input.text())
                    W0 = float(self.W0_input.text())
                except ValueError:
                    scale, L0, W0 = 1.0, 1.0, 1.0

                # mean vertical (y) and horizontal (x) displacement in pixels
                dy = np.mean(disp[:, 1])
                dx = np.mean(disp[:, 0])

                # strains
                axial_strain = (dy * scale) / L0
                transverse_strain = (dx * scale) / W0
                poisson_ratio = - transverse_strain / axial_strain if axial_strain != 0 else 0

                self.info_label.setText(
                    f"ε_axial: {axial_strain:.4e}  ε_trans: {transverse_strain:.4e}  ν: {poisson_ratio:.4f}"
                )

                # draw tracking
                for p in cur.astype(int):
                    cv2.circle(display, tuple(p), 3, (0, 255, 0), -1)

        # convert to Qt image
        h, w, ch = display.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(display.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        self.rtsp.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = DICLive("rtsp://10.5.0.2:8554/mystream")
    win.resize(1200, 600)
    win.show()
    sys.exit(app.exec_())
