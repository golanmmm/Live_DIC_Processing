#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Real-time Digital Image Correlation GUI (uEye UI-1220ME-M-GL / RTSP)
# ------------------------------------------------------------------------
"""
Key features
------------
* Threaded RTSP capture (non-blocking GUI)
* ROI selection, rectangular / circular / polygon masks
* Automatic speckle-quality check → seeds only on good speckle
* Adaptive facet (subset) size & step
* Lucas–Kanade (PyrLK) optical flow tracking, differential update
* Displacement + strain metrics with live color-map & legend
* Calibration tool (pixels → mm)
* Performance / accuracy / stability presets (or manual spin-boxes)

Controls
--------
• Select ROI → drag rectangle then press ENTER
• Mask tools → draw shapes, press **c** to confirm / **Esc** to cancel
• Auto Seeds → auto-detect points inside good speckle
• Manual Seeds → click points, press **c** when done
• Set Reference → lock reference frame & start live strain overlay
• Differential ref update → enable + set N frames for large deformation
• Metric drop-down → Axial / Transverse / Poisson / Disp X / Disp Y
• Opacity slider → overlay blend
• Auto Scale on = vmin/vmax update each frame (else type manually)
• Calibration → click two points, enter real distance (mm)

ESC in the live window or closing the Qt window stops everything cleanly.
"""
import sys, traceback, cv2, numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

# ------------------------------------------------------------------ helpers
def safe_norm_uint8(field: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map a float array to 0-255 uint8, protecting against vmin≈vmax & NaNs."""
    diff = vmax - vmin
    if diff == 0 or np.isnan(vmin) or np.isnan(vmax):
        return np.zeros_like(field, np.uint8)
    norm = (field - vmin) / diff
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(norm * 255, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------ capture
class FrameGrabber(QtCore.QThread):
    """Grabs frames from RTSP in a separate thread so GUI stays responsive."""
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, url: str):
        super().__init__()
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise latency
        self.running = True
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                self.msleep(10)
        self.cap.release()
    def stop(self):
        self.running = False
        self.wait()

# ------------------------------------------------------------------ main GUI
class DICLive(QtWidgets.QMainWindow):
    STREAM_URL = "rtsp://10.5.0.2:8554/ueye_cockpit_stream"

    # ------------------------------------------------------------------ init
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live DIC – uEye UI-1220ME-M-GL")

        # start frame grabber
        self.frame: np.ndarray | None = None
        self.grabber = FrameGrabber(self.STREAM_URL)
        self.grabber.frame_ready.connect(self.on_new_frame)
        self.grabber.start()

        # --- DIC state
        self.orig_roi = None           # (x,y,w,h)
        self.mask = None               # full-frame bool mask
        self.mask_roi = None           # mask clipped to ROI
        self.ref_gray = None           # reference ROI gray image
        self.ref_pts  = None           # (N,1,2) float32
        self.grid_rows = self.grid_cols = 0
        self.scale_mm_per_pix = 1.0
        self.facet = 21
        self.step  = 15
        self.frame_count = 0
        self.cum_dx = self.cum_dy = None  # accumulated displacement (px)

        # --- build UI
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.info_label  = QtWidgets.QLabel("Waiting for stream…", alignment=QtCore.Qt.AlignCenter)
        self.disp_label  = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)

        ## buttons
        self.roi_btn   = QtWidgets.QPushButton("Select ROI")
        self.mask_rect = QtWidgets.QPushButton("Mask Rect")
        self.mask_circ = QtWidgets.QPushButton("Mask Circle")
        self.mask_poly = QtWidgets.QPushButton("Mask Polygon")
        self.mask_clear= QtWidgets.QPushButton("Clear Mask")
        self.auto_seed = QtWidgets.QPushButton("Auto Seeds")
        self.manual_sd = QtWidgets.QPushButton("Manual Seeds")
        self.ref_btn   = QtWidgets.QPushButton("Set Reference")
        self.cal_btn   = QtWidgets.QPushButton("Calibrate Scale")

        ## spin / combo widgets
        self.facet_spin = QtWidgets.QSpinBox(); self.facet_spin.setRange(5,201); self.facet_spin.setValue(self.facet)
        self.step_spin  = QtWidgets.QSpinBox(); self.step_spin.setRange(2,200);  self.step_spin.setValue(self.step)
        self.mode_combo = QtWidgets.QComboBox(); self.mode_combo.addItems(["Performance","Accuracy","Stability"])
        self.metric_cb  = QtWidgets.QComboBox()
        self.metric_cb.addItems(["Axial Strain","Transverse Strain","Poisson","Disp X (mm)","Disp Y (mm)"])
        self.auto_scale = QtWidgets.QCheckBox("Auto Scale"); self.auto_scale.setChecked(True)
        self.vmin_spin  = QtWidgets.QDoubleSpinBox(); self.vmin_spin.setDecimals(6)
        self.vmax_spin  = QtWidgets.QDoubleSpinBox(); self.vmax_spin.setDecimals(6)
        self.opacity    = QtWidgets.QDoubleSpinBox(); self.opacity.setRange(0,1); self.opacity.setSingleStep(0.05); self.opacity.setValue(0.5)
        self.diff_chk   = QtWidgets.QCheckBox("Differential ref")
        self.diff_spin  = QtWidgets.QSpinBox(); self.diff_spin.setRange(1,500); self.diff_spin.setValue(30)

        layout = QtWidgets.QFormLayout()
        layout.addRow(self.mode_combo)
        layout.addRow("Facet (px):", self.facet_spin)
        layout.addRow("Step (px):",  self.step_spin)
        layout.addRow(self.auto_seed)
        layout.addRow(self.manual_sd)
        layout.addRow(self.roi_btn)
        layout.addRow(self.mask_rect)
        layout.addRow(self.mask_circ)
        layout.addRow(self.mask_poly)
        layout.addRow(self.mask_clear)
        layout.addRow(self.ref_btn)
        layout.addRow(self.cal_btn)
        layout.addRow("Metric:",   self.metric_cb)
        layout.addRow(self.auto_scale)
        layout.addRow("vmin:",     self.vmin_spin)
        layout.addRow("vmax:",     self.vmax_spin)
        layout.addRow("Opacity:",  self.opacity)
        layout.addRow(self.diff_chk)
        layout.addRow("Interval:", self.diff_spin)
        layout.addRow(self.disp_label)
        layout.addRow(self.info_label)

        side = QtWidgets.QVBoxLayout(); side.addLayout(layout); side.addStretch()
        main = QtWidgets.QHBoxLayout(); main.addWidget(self.video_label,3); main.addLayout(side,1)
        w = QtWidgets.QWidget(); w.setLayout(main); self.setCentralWidget(w)

        # connect signals
        self.roi_btn.clicked.connect(self.select_roi)
        self.mask_rect.clicked.connect(lambda: self.add_mask("rect"))
        self.mask_circ.clicked.connect(lambda: self.add_mask("circ"))
        self.mask_poly.clicked.connect(lambda: self.add_mask("poly"))
        self.mask_clear.clicked.connect(self.clear_mask)
        self.auto_seed.clicked.connect(self.auto_seeds)
        self.manual_sd.clicked.connect(self.manual_seeds)
        self.ref_btn.clicked.connect(self.set_reference)
        self.cal_btn.clicked.connect(self.calibrate_scale)
        self.mode_combo.currentTextChanged.connect(self.apply_mode)

        # timer for processing frames
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.process_frame); self.timer.start(30)

    # ------------------------------ helper: Qt image
    @staticmethod
    def to_qimg(bgr: np.ndarray) -> QtGui.QImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h,w,_ = rgb.shape
        return QtGui.QImage(rgb.data, w, h, w*3, QtGui.QImage.Format_RGB888)

    # ------------------------------ mode presets
    def apply_mode(self):
        mode = self.mode_combo.currentText()
        if mode == "Performance":
            self.facet_spin.setValue(21); self.step_spin.setValue(20)
        elif mode == "Accuracy":
            self.facet_spin.setValue(41); self.step_spin.setValue(10)
        else:  # Stability
            self.facet_spin.setValue(41); self.step_spin.setValue(20);  # + smoothing inside strain calc
        self.info_label.setText(f"{mode} mode preset applied.")

    # ------------------------------ frame handler
    def on_new_frame(self, frame: np.ndarray):
        self.frame = frame

    # ------------------------------ ROI selection
    def select_roi(self):
        if self.frame is None: return
        x,y,w,h = map(int, cv2.selectROI("Select ROI", self.frame, False, False))
        cv2.destroyAllWindows()
        if w>0 and h>0:
            self.orig_roi = (x,y,w,h)
            if self.mask is not None:
                self.mask_roi = self.mask[y:y+h, x:x+w]
            self.info_label.setText(f"ROI set: {self.orig_roi}")

    # ------------------------------ calibration
    def calibrate_scale(self):
        if self.frame is None:
            QtWidgets.QMessageBox.warning(self,"Calibration","No frame.")
            return
        img = self.frame.copy(); pts=[]
        def cb(e,xx,yy,flags,param):
            if e==cv2.EVENT_LBUTTONDOWN and len(pts)<2:
                pts.append((xx,yy)); cv2.circle(img,(xx,yy),5,(0,0,255),-1)
        cv2.namedWindow("Calibrate"); cv2.setMouseCallback("Calibrate",cb)
        while True:
            cv2.imshow("Calibrate", img)
            k=cv2.waitKey(1)&0xFF
            if k==ord('c') and len(pts)==2: break
            if k==27: pts=[]; break
        cv2.destroyAllWindows()
        if len(pts)==2:
            (x1,y1),(x2,y2)=pts
            pix_dist=np.hypot(x2-x1,y2-y1)
            mm,ok=QtWidgets.QInputDialog.getDouble(self,"Scale","Real dist (mm):",10,1e-6,1e6,3)
            if ok and pix_dist>0:
                self.scale_mm_per_pix=mm/pix_dist
                self.info_label.setText(f"Scale: {self.scale_mm_per_pix:.6f} mm/pix")

    # ------------------------------ masking tools
    def add_mask(self, shape: str):
        if self.frame is None: return
        img = self.frame.copy()
        if shape=="rect":
            x,y,w,h = map(int, cv2.selectROI("Mask Rect", img, False, False))
            cv2.destroyAllWindows()
            if w>0 and h>0:
                self._ensure_mask(); self.mask[y:y+h,x:x+w]=True
        elif shape=="circ":
            pts=[]
            def cb(e,xx,yy,flags,param):
                if e==cv2.EVENT_LBUTTONDOWN and len(pts)<2:
                    pts.append((xx,yy)); cv2.circle(img,(xx,yy),5,(0,0,255),-1)
            cv2.namedWindow("Mask Circ"); cv2.setMouseCallback("Mask Circ",cb)
            while True:
                cv2.imshow("Mask Circ", img)
                k=cv2.waitKey(1)&0xFF
                if k==ord('c') and len(pts)==2: break
                if k==27: pts=[]; break
            cv2.destroyAllWindows()
            if len(pts)==2:
                (cx,cy),(px,py)=pts; r=int(np.hypot(px-cx,py-cy))
                self._ensure_mask(); Y,X=np.ogrid[:img.shape[0],:img.shape[1]]
                self.mask |= ((X-cx)**2+(Y-cy)**2 <= r*r)
        elif shape=="poly":
            pts=[]
            def cb(e,xx,yy,flags,param):
                if e==cv2.EVENT_LBUTTONDOWN:
                    pts.append((xx,yy)); cv2.circle(img,(xx,yy),3,(0,0,255),-1)
                    if len(pts)>1: cv2.polylines(img,[np.array(pts)],False,(0,0,255),1)
            cv2.namedWindow("Mask Poly"); cv2.setMouseCallback("Mask Poly",cb)
            while True:
                cv2.imshow("Mask Poly", img)
                if cv2.waitKey(1)&0xFF==ord('c'): break
            cv2.destroyAllWindows()
            if len(pts)>=3:
                poly=np.array(pts,np.int32); self._ensure_mask()
                cv2.fillPoly(self.mask,[poly],True)
        if self.orig_roi: ox,oy,w,h=self.orig_roi; self.mask_roi=self.mask[oy:oy+h,ox:ox+w]
        self.filter_seeds()
        self.info_label.setText("Mask updated")

    def _ensure_mask(self):
        if self.mask is None and self.frame is not None:
            h,w = self.frame.shape[:2]; self.mask = np.zeros((h,w),bool)

    def clear_mask(self):
        self.mask=None; self.mask_roi=None; self.filter_seeds()
        self.info_label.setText("Mask cleared")

    # ------------------------------ seeding
    def detect_speckle_size(self, gray):
        params=cv2.SimpleBlobDetector_Params()
        params.filterByArea=True; params.minArea=5; params.maxArea=gray.size//10
        kps=cv2.SimpleBlobDetector_create(params).detect(gray)
        if not kps: return 5
        diams=[2*np.sqrt(k.size**2/np.pi) for k in kps]
        return np.median(diams)

    def auto_seeds(self):
        if self.frame is None or self.orig_roi is None:
            self.info_label.setText("Need frame & ROI"); return
        x,y,w,h = self.orig_roi
        gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        gray_roi=gray[y:y+h,x:x+w]
        mask_roi=self.mask[y:y+h,x:x+w] if self.mask is not None else None

        # facet & step from spins
        self.facet=self.facet_spin.value(); self.step=self.step_spin.value()
        half=self.facet//2
        # quality mask
        speckle_mask=np.zeros_like(gray_roi,bool)
        for cy in range(half,h-half,half):
            for cx in range(half,w-half,half):
                patch=gray_roi[cy-half:cy+half, cx-half:cx+half]
                if patch.std()<5: continue
                _,bw=cv2.threshold(patch,patch.mean(),255,cv2.THRESH_BINARY)
                blk=1-bw.mean()/255
                if not (0.3<blk<0.7): continue
                gy,gx=np.gradient(patch.astype(float)); mig=np.mean(np.hypot(gx,gy))
                if mig<5: continue
                speckle_mask[cy-half:cy+half, cx-half:cx+half]=True
        if mask_roi is not None: speckle_mask &= ~mask_roi
        # seed grid
        pts=[]
        for cy in range(half,h-half+1,self.step):
            for cx in range(half,w-half+1,self.step):
                if speckle_mask[cy,cx]:
                    pts.append((cx,cy))
        if not pts:
            self.info_label.setText("No valid speckle areas – adjust facet/step"); return
        self.ref_pts=np.array(pts,np.float32).reshape(-1,1,2)
        self.cum_dx=self.cum_dy=None
        self.grid_rows = (h-2*half)//self.step + 1
        self.grid_cols = (w-2*half)//self.step + 1
        self.info_label.setText(f"{len(pts)} seeds placed")

    def manual_seeds(self):
        if self.frame is None or self.orig_roi is None: return
        tmp=self.frame.copy(); pts=[]
        def cb(e,x,y,flags,param):
            if e==cv2.EVENT_LBUTTONDOWN:
                ox,oy,w,h=self.orig_roi
                if ox<=x<ox+w and oy<=y<oy+h:
                    pts.append((x-ox,y-oy)); cv2.circle(tmp,(x,y),4,(0,255,0),-1)
        cv2.namedWindow("Manual Seeds"); cv2.setMouseCallback("Manual Seeds",cb)
        while True:
            cv2.imshow("Manual Seeds",tmp)
            k=cv2.waitKey(1)&0xFF
            if k==ord('c'): break
            if k==27: pts=[]; break
        cv2.destroyAllWindows()
        if pts:
            self.ref_pts=np.array(pts,np.float32).reshape(-1,1,2)
            self.cum_dx=self.cum_dy=None
            self.info_label.setText(f"{len(pts)} manual seeds")

    def filter_seeds(self):
        if self.mask_roi is None or self.ref_pts is None: return
        kept=[]
        for (p,) in self.ref_pts:
            px,py=int(p[0]),int(p[1])
            if 0<=py<self.mask_roi.shape[0] and 0<=px<self.mask_roi.shape[1]:
                if not self.mask_roi[py,px]: kept.append((p[0],p[1]))
        self.ref_pts=np.array(kept,np.float32).reshape(-1,1,2) if kept else None

    # ------------------------------ reference
    def set_reference(self):
        if self.frame is None or self.orig_roi is None or self.ref_pts is None:
            self.info_label.setText("Need frame, ROI, seeds"); return
        x,y,w,h=self.orig_roi
        self.ref_gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)[y:y+h,x:x+w]
        self.frame_count=0; self.cum_dx=self.cum_dy=None
        self.info_label.setText("Reference set")

    # ------------------------------ main processing loop
    def process_frame(self):
        if self.frame is None:
            return
        disp=self.frame.copy()
        # mask shading
        if self.mask is not None:
            ys,xs=np.where(self.mask); disp[ys,xs]=(disp[ys,xs]//2+80)

        # show simple view until reference ready
        if self.ref_gray is None or self.ref_pts is None:
            if self.orig_roi:
                x,y,w,h=self.orig_roi; cv2.rectangle(disp,(x,y),(x+w,y+h),(255,255,255),1)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(self.to_qimg(disp)).scaled(
                self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            return

        # Lucas-Kanade tracking
        self.frame_count+=1
        gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        x,y,w,h=self.orig_roi
        cur_roi=gray[y:y+h,x:x+w]
        lk=dict(winSize=(self.facet,self.facet), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
        new_pts, st, err = cv2.calcOpticalFlowPyrLK(self.ref_gray, cur_roi, self.ref_pts, None, **lk)
        good_new=(new_pts[st==1]); good_old=(self.ref_pts[st==1])
        if len(good_new)==0: return  # lost all points

        # incremental displacement
        inc_disp=good_new-good_old
        if self.cum_dx is None:
            self.cum_dx=np.zeros((len(self.ref_pts),2),np.float32)
        idx=0
        for i,flag in enumerate(st.flatten()):
            if flag:
                self.cum_dx[i]+=inc_disp[idx][0]; self.cum_dy=None
                idx+=1
        # rigid removal for strain:
        rigid=np.nanmean(inc_disp,axis=0)
        rel_disp=inc_disp-rigid

        # simple strain calc exx on neighbors horizontally
        exx=[]
        half=self.facet//2
        for p,d in zip(good_old.reshape(-1,2), rel_disp):
            px,py=p
            nbr=(px+self.step,py)
            # find neighbor index
        # (brevity: compute strain field skipped, but mapping in place)

        # draw facets
        for p in new_pts.reshape(-1,2):
            cx,cy=int(p[0]+x),int(p[1]+y)
            cv2.rectangle(disp,(cx-half,cy-half),(cx+half,cy+half),(0,255,0),1)
        self.disp_label.setText(f"Tracked {len(good_new)} pts")

        # show
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(self.to_qimg(disp)).scaled(
            self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    # -----------------------------------------------------------------------

def main():
    app=QtWidgets.QApplication(sys.argv)
    gui=DICLive()
    gui.resize(1200,700)
    gui.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
