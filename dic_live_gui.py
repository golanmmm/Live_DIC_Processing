import sys
import traceback
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

def safe_normalize_uint8(field: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Normalize `field` to 0–255 uint8 using [vmin, vmax], guarding against zero range
    and NaNs/Infs.
    """
    diff = vmax - vmin
    if abs(diff) < 1e-8 or np.isnan(vmin) or np.isnan(vmax):
        return np.zeros_like(field, dtype=np.uint8)
    norm = (field - vmin) / diff
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)

class DICLive(QtWidgets.QMainWindow):
    def __init__(self, rtsp_url):
        super().__init__()
        self.setWindowTitle("Live Subset-based DIC GUI")

        # Video capture
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open RTSP stream")

        # State
        self.orig_roi  = None    # (x,y,w,h)
        self.ref_gray  = None    # reference ROI image
        self.ref_pts   = None    # seed points in ROI coords

        # DIC parameters
        self.scale      = 1.0    # mm/pixel
        self.L0         = 50.0   # mm
        self.W0         = 10.0   # mm
        self.facet_size = 21     # px
        self.step       = 15     # px

        # UI Elements
        self.video_label      = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.info_label       = QtWidgets.QLabel("select ROI → set reference", alignment=QtCore.Qt.AlignCenter)
        self.disp_label       = QtWidgets.QLabel("Disp X: 0.000 mm | Disp Y: 0.000 mm")

        # Buttons & inputs
        self.roi_btn          = QtWidgets.QPushButton("Select ROI")
        self.ref_btn          = QtWidgets.QPushButton("Set Reference")
        self.calib_btn        = QtWidgets.QPushButton("Calibrate Scale")
        self.auto_seeds_btn   = QtWidgets.QPushButton("Auto Seeds")
        self.manual_seeds_btn = QtWidgets.QPushButton("Manual Seeds")

        self.scale_edit       = QtWidgets.QLineEdit(f"{self.scale:.6f}")
        self.scale_edit.setReadOnly(True)
        self.L0_edit          = QtWidgets.QLineEdit(f"{self.L0:.1f}")
        self.W0_edit          = QtWidgets.QLineEdit(f"{self.W0:.1f}")

        self.facet_spin       = QtWidgets.QSpinBox()
        self.facet_spin.setRange(3, 500)
        self.facet_spin.setValue(self.facet_size)

        self.step_spin        = QtWidgets.QSpinBox()
        self.step_spin.setRange(1, 500)
        self.step_spin.setValue(self.step)

        self.metric_combo     = QtWidgets.QComboBox()
        self.metric_combo.addItems([
            "Axial Strain", "Transverse Strain", "Poisson",
            "Disp X (mm)", "Disp Y (mm)"
        ])

        self.auto_scale_chk   = QtWidgets.QCheckBox("Auto Scale")
        self.auto_scale_chk.setChecked(True)

        self.vmin_spin        = QtWidgets.QDoubleSpinBox()
        self.vmin_spin.setRange(-1e6, 1e6)
        self.vmin_spin.setDecimals(6)
        self.vmax_spin        = QtWidgets.QDoubleSpinBox()
        self.vmax_spin.setRange(-1e6, 1e6)
        self.vmax_spin.setDecimals(6)

        self.opacity_spin     = QtWidgets.QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(0.5)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Scale (mm/pix):", self.scale_edit)
        form.addRow(self.calib_btn)
        form.addRow("Gauge L₀ (mm):", self.L0_edit)
        form.addRow("Gauge W₀ (mm):", self.W0_edit)
        form.addRow("Facet size (px):", self.facet_spin)
        form.addRow("Step (px):", self.step_spin)
        form.addRow(self.auto_seeds_btn)
        form.addRow(self.manual_seeds_btn)
        form.addRow("Metric:", self.metric_combo)
        form.addRow(self.auto_scale_chk)
        form.addRow("vmin:", self.vmin_spin)
        form.addRow("vmax:", self.vmax_spin)
        form.addRow("Opacity:", self.opacity_spin)
        form.addRow(self.disp_label)
        form.addRow(self.info_label)
        form.addRow(self.roi_btn)
        form.addRow(self.ref_btn)

        ctrl = QtWidgets.QVBoxLayout()
        ctrl.addLayout(form)
        ctrl.addStretch()

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.video_label, 3)
        main_layout.addLayout(ctrl, 1)

        container = QtWidgets.QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Signals
        self.roi_btn.clicked.connect(self.select_roi)
        self.ref_btn.clicked.connect(self.set_reference)
        self.calib_btn.clicked.connect(self.calibrate_scale)
        self.auto_seeds_btn.clicked.connect(self.auto_seeds)
        self.manual_seeds_btn.clicked.connect(self.manual_seeds)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def select_roi(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        r = cv2.selectROI("Select ROI", frame, False, False)
        cv2.destroyWindow("Select ROI")
        x,y,w,h = map(int, r)
        if w>0 and h>0:
            self.orig_roi = (x,y,w,h)
            self.info_label.setText(f"ROI set: {self.orig_roi}")

    def calibrate_scale(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Calibrate", "Enter scale (mm/pixel):",
            self.scale, 1e-6, 1000.0, 6
        )
        if ok:
            self.scale = val
            self.scale_edit.setText(f"{self.scale:.6f}")

    def auto_seeds(self):
        if not self.orig_roi:
            self.info_label.setText("Define ROI first")
            return
        self.facet_size = self.facet_spin.value()
        self.step       = self.step_spin.value()
        x,y,w,h = self.orig_roi
        half = self.facet_size // 2
        pts = []
        for yy in range(y+half, y+h-half+1, self.step):
            for xx in range(x+half, x+w-half+1, self.step):
                pts.append((xx,yy))
        if not pts:
            self.info_label.setText("Adjust facet/step")
            return
        arr = np.array(pts, dtype=np.float32).reshape(-1,1,2)
        self.seed_pts = arr
        self.info_label.setText(f"{len(pts)} seeds placed")

    def manual_seeds(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        temp = frame.copy()
        pts = []
        def on_mouse(evt, xx, yy, flags, arg):
            if evt == cv2.EVENT_LBUTTONDOWN:
                if self.orig_roi:
                    x0,y0,w,h = self.orig_roi
                    if not (x0<=xx<x0+w and y0<=yy<y0+h):
                        return
                pts.append((xx,yy))
                cv2.circle(temp, (xx,yy), 3, (0,255,0), -1)
        cv2.namedWindow("Manual Seeds")
        cv2.setMouseCallback("Manual Seeds", on_mouse)
        while True:
            cv2.imshow("Manual Seeds", temp)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
        cv2.destroyWindow("Manual Seeds")
        if pts:
            self.seed_pts = np.array(pts, dtype=np.float32).reshape(-1,1,2)
            self.info_label.setText(f"{len(pts)} manual seeds")
        else:
            self.info_label.setText("No seeds selected")

    def set_reference(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Cannot grab frame")
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not self.orig_roi:
                self.orig_roi = (0,0, gray_full.shape[1], gray_full.shape[0])
            x,y,w,h = self.orig_roi
            self.ref_gray = gray_full[y:y+h, x:x+w].copy()
            if self.seed_pts is not None:
                rel = self.seed_pts.reshape(-1,2) - np.array([x,y])
                self.ref_pts = rel.reshape(-1,1,2).astype(np.float32)
            else:
                self.auto_seeds()
                rel = self.seed_pts.reshape(-1,2) - np.array([x,y])
                self.ref_pts = rel.reshape(-1,1,2).astype(np.float32)
            self.info_label.setText(f"Reference set ({len(self.ref_pts)} points)")
        except Exception:
            tb = traceback.format_exc()
            QtWidgets.QMessageBox.critical(self, "Reference Error", tb)

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                return
            display = frame.copy()

            # if no reference yet, just show ROI
            if self.ref_gray is None or self.ref_pts is None:
                if self.orig_roi:
                    x,y,w,h = self.orig_roi
                    cv2.rectangle(display, (x,y), (x+w, y+h), (255,255,255), 1)
                h_,w_,ch = display.shape
                qimg = QtGui.QImage(display.data, w_, h_, ch*w_,
                                    QtGui.QImage.Format_BGR888)
                self.video_label.setPixmap(
                    QtGui.QPixmap.fromImage(qimg).scaled(
                        self.video_label.size(),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation
                    )
                )
                return

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x,y,w,h = self.orig_roi
            cur_gray = gray_full[y:y+h, x:x+w]

            half = self.facet_size // 2
            dx_list, dy_list = [], []
            for pt in self.ref_pts.reshape(-1,2):
                j0,i0 = int(pt[0]), int(pt[1])
                tpl = self.ref_gray[i0-half:i0+half, j0-half:j0+half]
                if tpl.size == 0:
                    dx_list.append(0.0); dy_list.append(0.0); continue

                sr = self.step
                y0 = max(i0-half-sr, 0); y1 = min(i0+half+sr, h)
                x0 = max(j0-half-sr, 0); x1 = min(j0+half+sr, w)
                search = cur_gray[y0:y1, x0:x1]
                if (search.shape[0] < tpl.shape[0] or
                    search.shape[1] < tpl.shape[1]):
                    dx_list.append(0.0); dy_list.append(0.0); continue

                res = cv2.matchTemplate(search, tpl, cv2.TM_CCORR_NORMED)
                _,_,_,ml = cv2.minMaxLoc(res)
                dy = (ml[1] + half + y0) - i0
                dx = (ml[0] + half + x0) - j0
                dx_list.append(dx); dy_list.append(dy)

                # draw facet area
                cx, cy = x+j0, y+i0
                cv2.rectangle(display,
                              (cx-half, cy-half),
                              (cx+half, cy+half), (0,255,0), 1)

            dx_arr = np.array(dx_list) * self.scale
            dy_arr = np.array(dy_list) * self.scale

            # remove rigid-body shift
            rigid_x = np.nanmean(dx_arr)
            rigid_y = np.nanmean(dy_arr)
            dx_arr -= rigid_x
            dy_arr -= rigid_y

            # pick metric
            metric = self.metric_combo.currentText()
            if metric == "Disp X (mm)":
                field = dx_arr
            elif metric == "Disp Y (mm)":
                field = dy_arr
            else:
                if metric == "Axial Strain":
                    field = dy_arr / float(self.L0_edit.text())
                elif metric == "Transverse Strain":
                    field = dx_arr / float(self.W0_edit.text())
                else:  # Poisson
                    eps_ax = dy_arr / float(self.L0_edit.text())
                    eps_tr = dx_arr / float(self.W0_edit.text())
                    with np.errstate(divide='ignore', invalid='ignore'):
                        p = -eps_tr/eps_ax
                        p[np.isnan(p)] = 0; p[np.isinf(p)] = 0
                    field = p

            # color mapping
            vmin = (self.vmin_spin.value()
                    if not self.auto_scale_chk.isChecked()
                    else float(np.nanmin(field)))
            vmax = (self.vmax_spin.value()
                    if not self.auto_scale_chk.isChecked()
                    else float(np.nanmax(field)))
            if self.auto_scale_chk.isChecked():
                self.vmin_spin.setValue(vmin)
                self.vmax_spin.setValue(vmax)

            norm = safe_normalize_uint8(field, vmin, vmax)
            for val, pt in zip(norm, self.ref_pts.reshape(-1,2)):
                j0,i0 = int(pt[0]), int(pt[1])
                cx, cy = x+j0, y+i0
                color = tuple(int(c) for c in cv2.applyColorMap(
                    np.array([[val]],dtype=np.uint8), cv2.COLORMAP_JET
                )[0,0])
                cv2.circle(display, (cx,cy), max(1, half//2), color, -1)

            # show displacement
            self.disp_label.setText(f"Disp X: {rigid_x:.4f} mm | Disp Y: {rigid_y:.4f} mm")

            # render
            h_,w_,ch = display.shape
            qimg = QtGui.QImage(display.data, w_, h_, ch*w_,
                                QtGui.QImage.Format_BGR888)
            self.video_label.setPixmap(
                QtGui.QPixmap.fromImage(qimg).scaled(
                    self.video_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
            )

        except Exception:
            tb = traceback.format_exc()
            QtWidgets.QMessageBox.critical(self, "Update Error", tb)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = DICLive("rtsp://10.5.0.2:8554/mystream")
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec_())
