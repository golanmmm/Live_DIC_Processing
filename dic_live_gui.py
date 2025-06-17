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

        # Video
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open RTSP stream")

        # State
        self.orig_roi  = None      # x,y,w,h
        self.ref_gray  = None      # reference ROI gray image
        self.ref_pts   = None      # seed points in ROI coords
        self.grid_rows = 0
        self.grid_cols = 0
        self.mask      = None      # boolean mask for full frame
        self.L0_pts    = None      # two screen points for gauge length
        self.W0_pts    = None      # two screen points for gauge width

        # DIC params
        self.scale      = 1.0      # mm/pix
        self.facet_size = 21       # px
        self.step       = 15       # px

        # Differential-update
        self.diff_chk      = QtWidgets.QCheckBox("Differential reference update")
        self.diff_interval = QtWidgets.QSpinBox()
        self.diff_interval.setRange(1,1000)
        self.diff_interval.setValue(30)
        self.frame_count   = 0
        self.cum_dx_px     = None
        self.cum_dy_px     = None

        # UI
        self.video_label      = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.info_label       = QtWidgets.QLabel("select ROI → set reference", alignment=QtCore.Qt.AlignCenter)
        self.disp_label       = QtWidgets.QLabel("Disp X: 0.000 mm | Disp Y: 0.000 mm")

        self.roi_btn          = QtWidgets.QPushButton("Select ROI")
        self.ref_btn          = QtWidgets.QPushButton("Set Reference")
        self.calib_btn        = QtWidgets.QPushButton("Calibrate Scale")
        self.measure_L0_btn   = QtWidgets.QPushButton("Measure L₀")
        self.measure_W0_btn   = QtWidgets.QPushButton("Measure W₀")
        self.mask_rect_btn    = QtWidgets.QPushButton("Mask Rect")
        self.mask_circle_btn  = QtWidgets.QPushButton("Mask Circle")
        self.mask_poly_btn    = QtWidgets.QPushButton("Mask Polygon")
        self.clear_mask_btn   = QtWidgets.QPushButton("Clear Mask")
        self.auto_seeds_btn   = QtWidgets.QPushButton("Auto Seeds")
        self.manual_seeds_btn = QtWidgets.QPushButton("Manual Seeds")

        self.scale_edit = QtWidgets.QLineEdit(f"{self.scale:.6f}")
        self.scale_edit.setReadOnly(True)
        self.L0_edit    = QtWidgets.QLineEdit("0.000")
        self.W0_edit    = QtWidgets.QLineEdit("0.000")

        self.facet_spin = QtWidgets.QSpinBox()
        self.facet_spin.setRange(3,500)
        self.facet_spin.setValue(self.facet_size)

        self.step_spin  = QtWidgets.QSpinBox()
        self.step_spin.setRange(1,500)
        self.step_spin.setValue(self.step)

        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems([
            "Axial Strain","Transverse Strain","Poisson",
            "Disp X (mm)","Disp Y (mm)"
        ])
        self.auto_scale_chk = QtWidgets.QCheckBox("Auto Scale")
        self.auto_scale_chk.setChecked(True)

        self.vmin_spin = QtWidgets.QDoubleSpinBox()
        self.vmin_spin.setRange(-1e6,1e6); self.vmin_spin.setDecimals(6)
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        self.vmax_spin.setRange(-1e6,1e6); self.vmax_spin.setDecimals(6)

        self.opacity_spin = QtWidgets.QDoubleSpinBox()
        self.opacity_spin.setRange(0.0,1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(0.5)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Scale (mm/pix):", self.scale_edit)
        form.addRow("Gauge L₀ (mm):", self.L0_edit)
        form.addRow("Gauge W₀ (mm):", self.W0_edit)
        form.addRow(self.calib_btn)
        form.addRow(self.measure_L0_btn)
        form.addRow(self.measure_W0_btn)
        form.addRow(self.mask_rect_btn)
        form.addRow(self.mask_circle_btn)
        form.addRow(self.mask_poly_btn)
        form.addRow(self.clear_mask_btn)
        form.addRow(self.diff_chk)
        form.addRow("Ref interval (frames):", self.diff_interval)
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

        main = QtWidgets.QHBoxLayout()
        main.addWidget(self.video_label, 3)
        main.addLayout(ctrl, 1)

        container = QtWidgets.QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)

        # Signals
        self.roi_btn.clicked.connect(self.select_roi)
        self.ref_btn.clicked.connect(self.set_reference)
        self.calib_btn.clicked.connect(self.calibrate_scale)
        self.measure_L0_btn.clicked.connect(self.measure_L0)
        self.measure_W0_btn.clicked.connect(self.measure_W0)
        self.mask_rect_btn.clicked.connect(self.mask_rectangle)
        self.mask_circle_btn.clicked.connect(self.mask_circle)
        self.mask_poly_btn.clicked.connect(self.mask_polygon)
        self.clear_mask_btn.clicked.connect(self.clear_mask)
        self.auto_seeds_btn.clicked.connect(self.auto_seeds)
        self.manual_seeds_btn.clicked.connect(self.manual_seeds)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # --- ROI & Calibration ---

    def select_roi(self):
        ret,frame = self.cap.read()
        if not ret: return
        r = cv2.selectROI("Select ROI", frame, False, False)
        cv2.destroyWindow("Select ROI")
        x,y,w,h = map(int, r)
        if w>0 and h>0:
            self.orig_roi = (x,y,w,h)
            self.info_label.setText(f"ROI set: {self.orig_roi}")

    def calibrate_scale(self):
        ret,frame = self.cap.read()
        if not ret:
            QtWidgets.QMessageBox.warning(self,"Calibration","Failed to grab frame")
            return
        temp,pts = frame.copy(), []
        def on_mouse(e,x,y,f,p):
            if e==cv2.EVENT_LBUTTONDOWN and len(pts)<2:
                pts.append((x,y))
                cv2.circle(temp,(x,y),5,(0,0,255),-1)
                cv2.imshow("Calib: click 2 pts then 'c'", temp)
        cv2.namedWindow("Calib: click 2 pts then 'c'")
        cv2.setMouseCallback("Calib: click 2 pts then 'c'", on_mouse)
        while True:
            cv2.imshow("Calib: click 2 pts then 'c'", temp)
            k = cv2.waitKey(1)&0xFF
            if k==ord('c') and len(pts)==2: break
            if k==27: pts=[]; break
        cv2.destroyAllWindows()
        if len(pts)!=2:
            return
        (x1,y1),(x2,y2) = pts
        pd = np.hypot(x2-x1, y2-y1)
        if pd<1e-6:
            QtWidgets.QMessageBox.warning(self,"Calibration","Points too close")
            return
        mm,ok = QtWidgets.QInputDialog.getDouble(
            self,"Calibrate","Real dist (mm):", 1.0,1e-6,1e6,3
        )
        if not ok: return
        self.scale = mm/pd
        self.scale_edit.setText(f"{self.scale:.6f}")
        self.info_label.setText(f"Calibrated: {self.scale:.6f} mm/pix")

    # --- Gauge measurement ---

    def measure_L0(self):
        pts = self._pick_two("Measure L₀: click 2 points then 'c'")
        if not pts: return
        self.L0_pts = pts
        pd = np.hypot(pts[1][0]-pts[0][0], pts[1][1]-pts[0][1])
        self.L0_edit.setText(f"{pd*self.scale:.3f}")
        self.info_label.setText("L₀ measured")

    def measure_W0(self):
        pts = self._pick_two("Measure W₀: click 2 points then 'c'")
        if not pts: return
        self.W0_pts = pts
        pd = np.hypot(pts[1][0]-pts[0][0], pts[1][1]-pts[0][1])
        self.W0_edit.setText(f"{pd*self.scale:.3f}")
        self.info_label.setText("W₀ measured")

    def _pick_two(self, title):
        ret,frame = self.cap.read()
        if not ret:
            QtWidgets.QMessageBox.warning(self,title,"Failed to grab frame")
            return None
        temp,pts = frame.copy(), []
        def on_mouse(e,x,y,f,p):
            if e==cv2.EVENT_LBUTTONDOWN and len(pts)<2:
                pts.append((x,y))
                cv2.circle(temp,(x,y),5,(0,255,0),-1)
                cv2.imshow(title,temp)
        cv2.namedWindow(title)
        cv2.setMouseCallback(title,on_mouse)
        while True:
            cv2.imshow(title,temp)
            k = cv2.waitKey(1)&0xFF
            if k==ord('c') and len(pts)==2: break
            if k==27: pts=[]; break
        cv2.destroyAllWindows()
        return pts if len(pts)==2 else None

    # --- Masking ---

    def mask_rectangle(self):
        ret,frame = self.cap.read()
        if not ret: return
        r = cv2.selectROI("Mask Rect", frame, False, False)
        cv2.destroyWindow("Mask Rect")
        x,y,w,h = map(int, r)
        if w>0 and h>0:
            H,W = frame.shape[:2]
            if self.mask is None:
                self.mask = np.zeros((H,W),dtype=bool)
            self.mask[y:y+h, x:x+w] = True
            self.filter_seeds()
            self.info_label.setText("Rectangular mask applied")

    def mask_circle(self):
        ret,frame = self.cap.read()
        if not ret: return
        temp,pts = frame.copy(), []
        def on_mouse(e,x,y,f,p):
            if e==cv2.EVENT_LBUTTONDOWN and len(pts)<2:
                pts.append((x,y))
                cv2.circle(temp,(x,y),5,(0,0,255),-1)
                cv2.imshow("Mask Circle",temp)
        cv2.namedWindow("Mask Circle")
        cv2.setMouseCallback("Mask Circle",on_mouse)
        while True:
            cv2.imshow("Mask Circle",temp)
            k = cv2.waitKey(1)&0xFF
            if k==ord('c') and len(pts)==2: break
            if k==27: pts=[]; break
        cv2.destroyAllWindows()
        if len(pts)==2:
            (cx,cy),(px,py)=pts
            r = int(round(np.hypot(px-cx,py-cy)))
            H,W = frame.shape[:2]
            if self.mask is None:
                self.mask = np.zeros((H,W),dtype=bool)
            Y,X = np.ogrid[:H,:W]
            circle = (X-cx)**2 + (Y-cy)**2 <= r*r
            self.mask |= circle
            self.filter_seeds()
            self.info_label.setText("Circular mask applied")

    def mask_polygon(self):
        ret,frame = self.cap.read()
        if not ret: return
        temp,pts = frame.copy(), []
        def on_mouse(e,x,y,f,p):
            if e==cv2.EVENT_LBUTTONDOWN:
                pts.append((x,y))
                cv2.circle(temp,(x,y),3,(0,0,255),-1)
                if len(pts)>1:
                    cv2.polylines(temp,[np.array(pts)],False,(0,0,255),1)
                cv2.imshow("Mask Polygon",temp)
        cv2.namedWindow("Mask Polygon")
        cv2.setMouseCallback("Mask Polygon",on_mouse)
        while True:
            cv2.imshow("Mask Polygon",temp)
            k = cv2.waitKey(1)&0xFF
            if k==ord('c'): break
            if k==27: pts=[]; break
        cv2.destroyAllWindows()
        if len(pts)>=3:
            H,W = frame.shape[:2]
            if self.mask is None:
                self.mask = np.zeros((H,W),dtype=bool)
            poly = np.array(pts, dtype=np.int32)
            tmp = np.zeros((H,W),dtype=np.uint8)
            cv2.fillPoly(tmp, [poly], 1)
            self.mask |= (tmp.astype(bool))
            self.filter_seeds()
            self.info_label.setText("Polygon mask applied")

    def clear_mask(self):
        self.mask = None
        self.info_label.setText("Mask cleared")
        # optionally re-generate seeds:
        if self.ref_pts is not None:
            self.auto_seeds()

    def filter_seeds(self):
        """Drop any seeds whose absolute positions fall inside the mask."""
        if self.orig_roi is None or self.ref_pts is None or self.mask is None:
            return
        x0,y0,_,_ = self.orig_roi
        pts = self.ref_pts.reshape(-1,2)
        kept = []
        for px,py in pts:
            ax = int(round(x0 + px))
            ay = int(round(y0 + py))
            if 0 <= ay < self.mask.shape[0] and 0 <= ax < self.mask.shape[1]:
                if not self.mask[ay,ax]:
                    kept.append((px,py))
        if not kept:
            self.info_label.setText("All seeds masked!")
            return
        arr = np.array(kept, dtype=np.float32).reshape(-1,1,2)
        self.ref_pts = arr
        self.info_label.setText(f"{len(kept)} seeds remain after masking")

    # --- Seeding & reference ---

    def auto_seeds(self):
        if not self.orig_roi:
            self.info_label.setText("Define ROI first"); return
        self.facet_size = self.facet_spin.value()
        self.step       = self.step_spin.value()
        x,y,w,h = self.orig_roi
        half = self.facet_size//2

        ys = list(range(y+half, y+h-half+1, self.step))
        if ys[-1] != y+h-half: ys.append(y+h-half)
        xs = list(range(x+half, x+w-half+1, self.step))
        if xs[-1] != x+w-half: xs.append(x+w-half)

        pts = [(xx,yy) for yy in ys for xx in xs]
        self.grid_rows, self.grid_cols = len(ys), len(xs)
        rel = [(px-x, py-y) for px,py in pts]
        self.ref_pts = np.array(rel, dtype=np.float32).reshape(-1,1,2)
        self.filter_seeds()
        self.info_label.setText(f"{len(self.ref_pts)} seeds placed")

    def manual_seeds(self):
        ret,frame = self.cap.read()
        if not ret: return
        temp,pts = frame.copy(), []
        def on_mouse(e,x,y,f,p):
            if e==cv2.EVENT_LBUTTONDOWN and self.orig_roi:
                x0,y0,w,h = self.orig_roi
                if x0<=x<x0+w and y0<=y<y0+h:
                    pts.append((x,y))
                    cv2.circle(temp,(x,y),4,(0,255,0),-1)
                    cv2.imshow("Manual Seeds",temp)
        cv2.namedWindow("Manual Seeds")
        cv2.setMouseCallback("Manual Seeds",on_mouse)
        while True:
            cv2.imshow("Manual Seeds",temp)
            k = cv2.waitKey(1)&0xFF
            if k==ord('c'): break
            if k==27: pts=[]; break
        cv2.destroyAllWindows()
        if pts:
            x,y,_,_ = self.orig_roi
            rel = [(px-x,py-y) for px,py in pts]
            self.ref_pts = np.array(rel,dtype=np.float32).reshape(-1,1,2)
            self.grid_rows=self.grid_cols=0
            self.filter_seeds()
            self.info_label.setText(f"{len(self.ref_pts)} manual seeds")
        else:
            self.info_label.setText("No seeds selected")

    def set_reference(self):
        try:
            ret,frame = self.cap.read()
            if not ret: raise RuntimeError("Cannot grab frame")
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if not self.orig_roi:
                Hf,Wf = gray.shape; self.orig_roi=(0,0,Wf,Hf)
            x,y,w,h = self.orig_roi
            self.ref_gray = gray[y:y+h, x:x+w].copy()
            if self.ref_pts is None:
                self.auto_seeds()
            self.frame_count = 0
            self.cum_dx_px = None
            self.cum_dy_px = None
            self.info_label.setText(f"Reference set ({len(self.ref_pts)} seeds)")
        except Exception:
            QtWidgets.QMessageBox.critical(self,"Reference Error",traceback.format_exc())

    # --- Main update loop ---

    def update_frame(self):
        try:
            ret,frame = self.cap.read()
            if not ret: return
            disp = frame.copy()

            # overlay mask region (greyed out)
            if self.mask is not None:
                ys,xs = np.where(self.mask)
                disp[ys, xs] = (disp[ys, xs]//2 + 80)

            if self.ref_gray is None or self.ref_pts is None:
                if self.orig_roi:
                    x,y,w,h = self.orig_roi
                    cv2.rectangle(disp,(x,y),(x+w,y+h),(255,255,255),1)
                H0,W0,_=disp.shape
                img = QtGui.QImage(disp.data,W0,H0,3*W0,QtGui.QImage.Format_BGR888)
                self.video_label.setPixmap(
                    QtGui.QPixmap.fromImage(img).scaled(
                        self.video_label.size(),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation))
                return

            # differential counter
            self.frame_count += 1

            gray_full = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            x,y,w,h = self.orig_roi
            cur = gray_full[y:y+h, x:x+w]

            half, step = self.facet_size//2, self.step
            raw_dx, raw_dy = [], []
            for pt in self.ref_pts.reshape(-1,2):
                j0,i0 = int(pt[0]), int(pt[1])
                tpl = self.ref_gray[i0-half:i0+half, j0-half:j0+half]
                if tpl.size==0:
                    raw_dx.append(0); raw_dy.append(0); continue
                y0,y1 = max(i0-half-step,0), min(i0+half+step,h)
                x0,x1 = max(j0-half-step,0), min(j0+half+step,w)
                wnd = cur[y0:y1, x0:x1]
                if wnd.shape[0]<tpl.shape[0] or wnd.shape[1]<tpl.shape[1]:
                    raw_dx.append(0); raw_dy.append(0); continue
                _,_,_,ml = cv2.minMaxLoc(cv2.matchTemplate(wnd,tpl,cv2.TM_CCORR_NORMED))
                raw_dx.append((ml[0]+half+x0)-j0)
                raw_dy.append((ml[1]+half+y0)-i0)

            raw_dx = np.array(raw_dx, dtype=np.float64)
            raw_dy = np.array(raw_dy, dtype=np.float64)

            # remove rigid body
            rigid_x_px = float(np.nanmean(raw_dx))
            rigid_y_px = float(np.nanmean(raw_dy))
            dx_px = raw_dx - rigid_x_px
            dy_px = raw_dy - rigid_y_px

            # init cumulative arrays
            if self.diff_chk.isChecked() and self.cum_dx_px is None:
                self.cum_dx_px = np.zeros_like(dx_px)
                self.cum_dy_px = np.zeros_like(dy_px)
            # accumulate & refresh
            if self.diff_chk.isChecked() and self.frame_count >= self.diff_interval.value():
                self.cum_dx_px += dx_px
                self.cum_dy_px += dy_px
                self.ref_gray = cur.copy()
                self.ref_pts[:,0,0] += raw_dx
                self.ref_pts[:,0,1] += raw_dy
                self.frame_count = 0
                rigid_x_px = rigid_y_px = 0.0
                dx_px[:] = 0.0; dy_px[:] = 0.0

            # displayed total shift
            if self.diff_chk.isChecked():
                disp_dx_px = self.cum_dx_px + dx_px
                disp_dy_px = self.cum_dy_px + dy_px
            else:
                disp_dx_px = dx_px
                disp_dy_px = dy_px

            # to mm
            dx_mm = disp_dx_px * self.scale
            dy_mm = disp_dy_px * self.scale
            rx_mm = rigid_x_px * self.scale
            ry_mm = rigid_y_px * self.scale

            # pick metric
            met = self.metric_combo.currentText()
            if met=="Disp X (mm)":
                field = dx_mm
            elif met=="Disp Y (mm)":
                field = dy_mm
            else:
                L0 = float(self.L0_edit.text())
                W0 = float(self.W0_edit.text())
                if met=="Axial Strain":
                    field = dy_mm / L0
                elif met=="Transverse Strain":
                    field = dx_mm / W0
                else:  # Poisson
                    ea,et = dy_mm/L0, dx_mm/W0
                    with np.errstate(divide='ignore',invalid='ignore'):
                        p = -et/ea
                    field = np.nan_to_num(p,0,0,0)

            # build full_field (structured or sparse)
            if self.grid_rows>0 and self.grid_cols>0 and len(field)==self.grid_rows*self.grid_cols:
                grid = field.reshape(self.grid_rows,self.grid_cols)
                full_field = cv2.resize(grid,(w,h),interpolation=cv2.INTER_CUBIC)
            else:
                fmap = np.zeros((h,w),dtype=np.float32)
                for val,(j0f,i0f) in zip(field,self.ref_pts.reshape(-1,2)):
                    j0 = int(round(j0f)); i0 = int(round(i0f))
                    if 0<=i0<h and 0<=j0<w:
                        fmap[i0,j0] = val
                full_field = cv2.resize(fmap,(w,h),interpolation=cv2.INTER_CUBIC)

            # normalize & overlay heatmap
            vmin = self.vmin_spin.value() if not self.auto_scale_chk.isChecked() else float(np.nanmin(full_field))
            vmax = self.vmax_spin.value() if not self.auto_scale_chk.isChecked() else float(np.nanmax(full_field))
            if self.auto_scale_chk.isChecked():
                self.vmin_spin.setValue(vmin); self.vmax_spin.setValue(vmax)
            norm_map = safe_normalize_uint8(full_field,vmin,vmax)
            cmap     = cv2.applyColorMap(norm_map,cv2.COLORMAP_JET)
            alpha    = float(self.opacity_spin.value())
            disp[y:y+h, x:x+w] = cv2.addWeighted(cmap,alpha,disp[y:y+h, x:x+w],1-alpha,0)

            # draw moving facets
            half = self.facet_size//2
            for k,(j0,i0) in enumerate(self.ref_pts.reshape(-1,2)):
                cx = int(x + j0 + disp_dx_px[k])
                cy = int(y + i0 + disp_dy_px[k])
                cv2.rectangle(disp,(cx-half,cy-half),(cx+half,cy+half),(255,255,255),1)
                cv2.drawMarker(disp,(cx,cy),(255,255,255),cv2.MARKER_CROSS,6,1)

            # draw L0/W0 lines & annotate strains
            if self.L0_pts:
                (x1,y1),(x2,y2) = self.L0_pts
                cv2.line(disp,(x1,y1),(x2,y2),(0,255,255),2)
                avg_ax = float(np.nanmean(dy_mm)) / float(self.L0_edit.text())
                mx,my = ( (x1+x2)//2, (y1+y2)//2 )
                cv2.putText(disp,f"ε_ax={avg_ax:.3e}",(mx,my-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,cv2.LINE_AA)
            if self.W0_pts:
                (x1,y1),(x2,y2) = self.W0_pts
                cv2.line(disp,(x1,y1),(x2,y2),(0,255,0),2)
                avg_tr = float(np.nanmean(dx_mm)) / float(self.W0_edit.text())
                mx,my = ( (x1+x2)//2, (y1+y2)//2 )
                cv2.putText(disp,f"ε_tr={avg_tr:.3e}",(mx,my-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)

            # displacement label
            self.disp_label.setText(f"DX: {rx_mm:.4f} mm | DY: {ry_mm:.4f} mm")

            # colorbar legend
            H_disp,W_disp,_ = disp.shape; legend_w=80
            bar = np.linspace(vmax,vmin,H_disp,dtype=np.float32)
            bn  = safe_normalize_uint8(bar,vmin,vmax)
            bc  = cv2.applyColorMap(bn.reshape(H_disp,1),cv2.COLORMAP_JET)
            legend = np.repeat(bc, legend_w, axis=1)
            cv2.putText(legend,self.metric_combo.currentText(),(5,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            cv2.putText(legend,f"{vmax:.2e}",(5,45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(legend,f"{vmin:.2e}",(5,H_disp-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

            combined = np.hstack((disp,legend))
            Hc,Wc,_ = combined.shape
            img = QtGui.QImage(combined.data,Wc,Hc,3*Wc,QtGui.QImage.Format_BGR888)
            self.video_label.setPixmap( 
                QtGui.QPixmap.fromImage(img).scaled(
                    self.video_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation))
        except Exception:
            QtWidgets.QMessageBox.critical(self,"Update Error",traceback.format_exc())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = DICLive("rtsp://10.5.0.2:8554/mystream")
    win.resize(1200,700)
    win.show()
    sys.exit(app.exec_())
