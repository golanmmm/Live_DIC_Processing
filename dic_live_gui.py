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
        self.orig_roi   = None      # (x,y,w,h)
        self.ref_gray   = None      # reference ROI image
        self.ref_pts    = None      # seed points in ROI coords
        self.grid_rows  = 0         # structured‐grid dims
        self.grid_cols  = 0

        # DIC parameters
        self.scale      = 1.0       # mm/pixel
        self.L0         = 50.0      # mm
        self.W0         = 10.0      # mm
        self.facet_size = 21        # px
        self.step       = 15        # px

        # UI Elements
        self.video_label      = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.info_label       = QtWidgets.QLabel("select ROI → set reference", alignment=QtCore.Qt.AlignCenter)
        self.disp_label       = QtWidgets.QLabel("Disp X: 0.000 mm | Disp Y: 0.000 mm")

        self.roi_btn          = QtWidgets.QPushButton("Select ROI")
        self.ref_btn          = QtWidgets.QPushButton("Set Reference")
        self.calib_btn        = QtWidgets.QPushButton("Calibrate Scale")
        self.auto_seeds_btn   = QtWidgets.QPushButton("Auto Seeds")
        self.manual_seeds_btn = QtWidgets.QPushButton("Manual Seeds")

        self.scale_edit       = QtWidgets.QLineEdit(f"{self.scale:.6f}"); self.scale_edit.setReadOnly(True)
        self.L0_edit          = QtWidgets.QLineEdit(f"{self.L0:.1f}")
        self.W0_edit          = QtWidgets.QLineEdit(f"{self.W0:.1f}")

        self.facet_spin       = QtWidgets.QSpinBox(); self.facet_spin.setRange(3,500); self.facet_spin.setValue(self.facet_size)
        self.step_spin        = QtWidgets.QSpinBox(); self.step_spin.setRange(1,500); self.step_spin.setValue(self.step)

        self.metric_combo     = QtWidgets.QComboBox()
        self.metric_combo.addItems([
            "Axial Strain","Transverse Strain","Poisson",
            "Disp X (mm)","Disp Y (mm)"
        ])
        self.auto_scale_chk   = QtWidgets.QCheckBox("Auto Scale"); self.auto_scale_chk.setChecked(True)

        self.vmin_spin        = QtWidgets.QDoubleSpinBox(); self.vmin_spin.setRange(-1e6,1e6); self.vmin_spin.setDecimals(6)
        self.vmax_spin        = QtWidgets.QDoubleSpinBox(); self.vmax_spin.setRange(-1e6,1e6); self.vmax_spin.setDecimals(6)

        self.opacity_spin     = QtWidgets.QDoubleSpinBox(); self.opacity_spin.setRange(0.0,1.0)
        self.opacity_spin.setSingleStep(0.05); self.opacity_spin.setValue(0.5)

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

        ctrl = QtWidgets.QVBoxLayout(); ctrl.addLayout(form); ctrl.addStretch()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.video_label,3); layout.addLayout(ctrl,1)
        container = QtWidgets.QWidget(); container.setLayout(layout)
        self.setCentralWidget(container)

        # Signals
        self.roi_btn.clicked.connect(self.select_roi)
        self.ref_btn.clicked.connect(self.set_reference)
        self.calib_btn.clicked.connect(self.calibrate_scale)
        self.auto_seeds_btn.clicked.connect(self.auto_seeds)
        self.manual_seeds_btn.clicked.connect(self.manual_seeds)

        # Timer
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.update_frame); self.timer.start(30)

    def select_roi(self):
        ret, frame = self.cap.read()
        if not ret: return
        r = cv2.selectROI("Select ROI", frame, False, False)
        cv2.destroyWindow("Select ROI")
        x,y,w,h = map(int, r)
        if w>0 and h>0:
            self.orig_roi = (x,y,w,h)
            self.info_label.setText(f"ROI set: {self.orig_roi}")

    def calibrate_scale(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Calibrate", "Enter scale (mm/pixel):",
            self.scale, 1e-6, 1e3, 6
        )
        if ok:
            self.scale = val
            self.scale_edit.setText(f"{self.scale:.6f}")

    def auto_seeds(self):
        if not self.orig_roi:
            self.info_label.setText("Define ROI first"); return
        self.facet_size = self.facet_spin.value()
        self.step       = self.step_spin.value()
        x,y,w,h = self.orig_roi
        half = self.facet_size // 2

        ys = list(range(y+half, y+h-half+1, self.step))
        if ys[-1] != y+h-half:  ys.append(y+h-half)
        xs = list(range(x+half, x+w-half+1, self.step))
        if xs[-1] != x+w-half:  xs.append(x+w-half)

        pts = [(xx,yy) for yy in ys for xx in xs]
        self.grid_rows = len(ys)
        self.grid_cols = len(xs)

        rel = [(px - x, py - y) for px,py in pts]
        self.ref_pts = np.array(rel, dtype=np.float32).reshape(-1,1,2)
        self.info_label.setText(f"{len(pts)} seeds ({self.grid_rows}×{self.grid_cols})")

    def manual_seeds(self):
        ret, frame = self.cap.read()
        if not ret: return
        temp, pts = frame.copy(), []
        def on_mouse(evt, xx, yy, flags, arg):
            if evt == cv2.EVENT_LBUTTONDOWN and self.orig_roi:
                x0,y0,w,h = self.orig_roi
                if x0<=xx<x0+w and y0<=yy<y0+h:
                    pts.append((xx,yy))
                    cv2.circle(temp, (xx,yy), 3, (0,255,0), -1)
                    cv2.imshow("Manual Seeds", temp)
        cv2.namedWindow("Manual Seeds"); cv2.setMouseCallback("Manual Seeds", on_mouse)
        while True:
            cv2.imshow("Manual Seeds", temp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('c'): break
            if k == 27: pts=[]; break
        cv2.destroyWindow("Manual Seeds")
        if pts:
            x,y,_,_ = self.orig_roi
            rel = [(px-x, py-y) for px,py in pts]
            self.ref_pts = np.array(rel, dtype=np.float32).reshape(-1,1,2)
            self.grid_rows = self.grid_cols = 0
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
                h_full, w_full = gray_full.shape
                self.orig_roi = (0,0,w_full,h_full)
            x,y,w,h = self.orig_roi
            self.ref_gray = gray_full[y:y+h, x:x+w].copy()
            if self.ref_pts is None:
                self.auto_seeds()
            self.info_label.setText(f"Reference set ({len(self.ref_pts)} points)")
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Reference Error", traceback.format_exc())

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret: return
            disp = frame.copy()

            if self.ref_gray is None or self.ref_pts is None:
                if self.orig_roi:
                    x,y,w,h = self.orig_roi
                    cv2.rectangle(disp,(x,y),(x+w,y+h),(255,255,255),1)
                H,W,_ = disp.shape
                img = QtGui.QImage(disp.data, W, H, 3*W, QtGui.QImage.Format_BGR888)
                self.video_label.setPixmap(
                    QtGui.QPixmap.fromImage(img).scaled(
                        self.video_label.size(),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation))
                return

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x,y,w,h = self.orig_roi
            cur = gray_full[y:y+h, x:x+w]

            half, step = self.facet_size//2, self.step
            dx_list, dy_list = [], []
            for pt in self.ref_pts.reshape(-1,2):
                j0, i0 = int(pt[0]), int(pt[1])
                tpl = self.ref_gray[i0-half:i0+half, j0-half:j0+half]
                if tpl.size == 0:
                    dx_list.append(0); dy_list.append(0); continue
                y0, y1 = max(i0-half-step,0), min(i0+half+step,h)
                x0, x1 = max(j0-half-step,0), min(j0+half+step,w)
                wnd = cur[y0:y1, x0:x1]
                if wnd.shape[0]<tpl.shape[0] or wnd.shape[1]<tpl.shape[1]:
                    dx_list.append(0); dy_list.append(0); continue
                _,_,_,ml = cv2.minMaxLoc(cv2.matchTemplate(wnd,tpl,cv2.TM_CCORR_NORMED))
                dy_list.append((ml[1]+half+y0)-i0)
                dx_list.append((ml[0]+half+x0)-j0)

            dx_arr = np.array(dx_list,dtype=np.float64)*self.scale
            dy_arr = np.array(dy_list,dtype=np.float64)*self.scale
            rx, ry = float(np.nanmean(dx_arr)), float(np.nanmean(dy_arr))
            dx_arr -= rx; dy_arr -= ry

            met = self.metric_combo.currentText()
            if met == "Disp X (mm)":    field = dx_arr
            elif met == "Disp Y (mm)":  field = dy_arr
            else:
                L0, W0 = float(self.L0_edit.text()), float(self.W0_edit.text())
                if met == "Axial Strain":      field = dy_arr / L0
                elif met == "Transverse Strain": field = dx_arr / W0
                else:
                    ea, et = dy_arr/L0, dx_arr/W0
                    with np.errstate(divide='ignore',invalid='ignore'):
                        p = -et/ea
                    field = np.nan_to_num(p,0,0,0)

            # structured‐grid interpolation
            if self.grid_rows>0 and self.grid_cols>0 and len(field)==self.grid_rows*self.grid_cols:
                grid = field.reshape(self.grid_rows,self.grid_cols)
                full_field = cv2.resize(grid,(w,h),interpolation=cv2.INTER_CUBIC)
            else:
                fmap = np.zeros((h,w),dtype=np.float32)
                for val,pt in zip(field,self.ref_pts.reshape(-1,2)):
                    j0,i0 = int(pt[0]),int(pt[1])
                    if 0<=i0<h and 0<=j0<w:
                        fmap[i0,j0] = val
                full_field = cv2.resize(fmap,(w,h),interpolation=cv2.INTER_CUBIC)

            # normalize & colormap
            vmin = self.vmin_spin.value() if not self.auto_scale_chk.isChecked() else float(np.nanmin(full_field))
            vmax = self.vmax_spin.value() if not self.auto_scale_chk.isChecked() else float(np.nanmax(full_field))
            if self.auto_scale_chk.isChecked():
                self.vmin_spin.setValue(vmin); self.vmax_spin.setValue(vmax)

            norm_map = safe_normalize_uint8(full_field, vmin, vmax)
            cmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            alpha = float(self.opacity_spin.value())
            roi_disp = disp[y:y+h, x:x+w]
            cv2.addWeighted(cmap, alpha, roi_disp, 1-alpha, 0, roi_disp)

            # seed preview
            for pt in self.ref_pts.reshape(-1,2):
                cx, cy = x+int(pt[0]), y+int(pt[1])
                cv2.drawMarker(disp, (cx,cy), (255,255,255),
                               cv2.MARKER_CROSS, markerSize=8, thickness=1)

            self.disp_label.setText(f"Disp X: {rx:.4f} mm | Disp Y: {ry:.4f} mm")

            # colorbar legend
            H_disp, W_disp, _ = disp.shape
            legend_w = 80
            bar = np.linspace(vmax, vmin, H_disp, dtype=np.float32)
            bar_norm = safe_normalize_uint8(bar, vmin, vmax)
            bar_col = cv2.applyColorMap(bar_norm.reshape(H_disp,1), cv2.COLORMAP_JET)
            legend = np.repeat(bar_col, legend_w, axis=1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(legend, met,          (5,25),   font, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(legend, f"{vmax:.2e}", (5,45),   font, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(legend, f"{vmin:.2e}", (5,H_disp-10), font,0.5,(255,255,255),1,cv2.LINE_AA)

            combined = np.hstack((disp, legend))
            Hc, Wc, _ = combined.shape
            img = QtGui.QImage(combined.data, Wc, Hc, 3*Wc, QtGui.QImage.Format_BGR888)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
                self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        except Exception:
            QtWidgets.QMessageBox.critical(self, "Update Error", traceback.format_exc())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = DICLive("rtsp://10.5.0.2:8554/mystream")
    win.resize(1200,700); win.show(); sys.exit(app.exec_())
