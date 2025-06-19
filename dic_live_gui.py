#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Live Digital-Image-Correlation GUI  |  camera: IDS UI-1220ME-M-GL (RTSP)
# ---------------------------------------------------------------------------
"""
HOW TO USE
----------
(1)  Select ROI  ➜ draw rectangle  ➜  hit ENTER
(2)  Auto Seeds  ➜ program keeps only good speckle patches
(3)  Set Reference ➜ live displacement / strain colour-map appears
(4)  Metric drop-down ➜ Axial / Transverse / Poisson / Disp X / Disp Y
Optional: mask clamps, calibrate scale (click two points, press c, enter mm),
adjust opacity, vmin/vmax, differential update for large deformation.

ESC or closing the Qt window exits cleanly.
"""
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

RTSP_URL = "rtsp://10.5.0.2:8554/ueye_cockpit_stream"

# ---------------------------------------------------------------- utilities

def to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)

def norm_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize float array arr to uint8 0-255 between vmin/vmax."""
    diff = vmax - vmin
    if diff == 0:
        diff = 1e-6
    norm = np.clip((arr - vmin) / diff, 0, 1)
    return (norm * 255).astype(np.uint8)

# ---------------------------------------------------------------- grabber thread

class FrameGrabber(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.running = True
    def run(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                self.msleep(10)
        cap.release()
    def stop(self):
        self.running = False
        self.wait()

# ---------------------------------------------------------------- main GUI

class DICLive(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live DIC – UI-1220ME-M-GL")

        # state
        self.cur_frame = None
        self.roi = None
        self.mask_full = None
        self.mask_roi = None
        self.ref_gray = None
        self.ref_pts = None
        self.cum_disp = None
        self.scale_mm = 1.0
        self.facet = 21
        self.step = 15
        self.frame_count = 0

        # start grabber thread
        self.grabber = FrameGrabber(RTSP_URL)
        self.grabber.frame_ready.connect(self.on_new_frame)
        self.grabber.start()

        # UI elements
        self.view = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.cbar = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.cbar.setFixedWidth(120)
        self.info = QtWidgets.QLabel("Waiting for stream…", alignment=QtCore.Qt.AlignCenter)
        self.disp = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)

        # Controls
        self.mode_cb = QtWidgets.QComboBox(); self.mode_cb.addItems(["Performance","Accuracy","Stability"])
        self.f_spin  = QtWidgets.QSpinBox(); self.f_spin.setRange(5,201); self.f_spin.setValue(self.facet)
        self.s_spin  = QtWidgets.QSpinBox(); self.s_spin.setRange(2,200); self.s_spin.setValue(self.step)
        self.auto_btn= QtWidgets.QPushButton("Auto Seeds")
        self.man_btn = QtWidgets.QPushButton("Manual Seeds")
        self.roi_btn = QtWidgets.QPushButton("Select ROI")
        self.m_rect  = QtWidgets.QPushButton("Mask Rect")
        self.m_circ  = QtWidgets.QPushButton("Mask Circle")
        self.m_poly  = QtWidgets.QPushButton("Mask Polygon")
        self.m_clear = QtWidgets.QPushButton("Clear Mask")
        self.ref_btn = QtWidgets.QPushButton("Set Reference")
        self.cal_btn = QtWidgets.QPushButton("Calibrate Scale")
        self.met_cb  = QtWidgets.QComboBox(); self.met_cb.addItems([
            "Axial Strain","Transverse Strain","Poisson","Disp X (mm)","Disp Y (mm)"
        ])
        self.auto_scale = QtWidgets.QCheckBox("Auto Scale"); self.auto_scale.setChecked(True)
        self.vmin_sp    = QtWidgets.QDoubleSpinBox(); self.vmin_sp.setDecimals(6)
        self.vmax_sp    = QtWidgets.QDoubleSpinBox(); self.vmax_sp.setDecimals(6)
        self.alpha_sp   = QtWidgets.QDoubleSpinBox(); self.alpha_sp.setRange(0,1);
        self.alpha_sp.setSingleStep(0.05); self.alpha_sp.setValue(0.5)
        self.diff_chk   = QtWidgets.QCheckBox("Differential ref")
        self.diff_sp    = QtWidgets.QSpinBox(); self.diff_sp.setRange(1,500); self.diff_sp.setValue(30)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow(self.mode_cb)
        form.addRow("Facet (px):", self.f_spin)
        form.addRow("Step (px):", self.s_spin)
        form.addRow(self.auto_btn)
        form.addRow(self.man_btn)
        form.addRow(self.roi_btn)
        form.addRow(self.m_rect)
        form.addRow(self.m_circ)
        form.addRow(self.m_poly)
        form.addRow(self.m_clear)
        form.addRow(self.ref_btn)
        form.addRow(self.cal_btn)
        form.addRow("Metric:", self.met_cb)
        form.addRow(self.auto_scale)
        form.addRow("vmin:", self.vmin_sp)
        form.addRow("vmax:", self.vmax_sp)
        form.addRow("Opacity:", self.alpha_sp)
        form.addRow(self.diff_chk)
        form.addRow("Interval:", self.diff_sp)
        form.addRow(self.disp)
        form.addRow(self.info)

        side = QtWidgets.QWidget(); side.setLayout(form); side.setMinimumWidth(230)
        main = QtWidgets.QHBoxLayout();
        main.addWidget(self.view,3);
        main.addWidget(self.cbar);
        main.addWidget(side)
        container = QtWidgets.QWidget(); container.setLayout(main)
        self.setCentralWidget(container)

        # Connections
        self.mode_cb.currentTextChanged.connect(self.apply_mode)
        self.auto_btn.clicked.connect(self.auto_seeds)
        self.man_btn.clicked.connect(self.manual_seeds)
        self.roi_btn.clicked.connect(self.select_roi)
        self.m_rect.clicked.connect(lambda: self.draw_mask('rect'))
        self.m_circ.clicked.connect(lambda: self.draw_mask('circ'))
        self.m_poly.clicked.connect(lambda: self.draw_mask('poly'))
        self.m_clear.clicked.connect(self.clear_mask)
        self.ref_btn.clicked.connect(self.set_reference)
        self.cal_btn.clicked.connect(self.calibrate_scale)

        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.process_frame);
        self.timer.start(30)

    def apply_mode(self):
        m = self.mode_cb.currentText()
        if m=='Performance': self.f_spin.setValue(21); self.s_spin.setValue(20)
        elif m=='Accuracy':    self.f_spin.setValue(41); self.s_spin.setValue(10)
        else:                   self.f_spin.setValue(41); self.s_spin.setValue(20)
        self.info.setText(f"{m} preset applied")

    def on_new_frame(self, frame: np.ndarray):
        self.cur_frame = frame

    def select_roi(self):
        if self.cur_frame is None: return
        x,y,w,h = map(int, cv2.selectROI("Select ROI", self.cur_frame, False, False))
        cv2.destroyAllWindows()
        if w>0 and h>0:
            self.roi = (x,y,w,h)
            self.mask_roi = self.mask_full[y:y+h, x:x+w] if self.mask_full is not None else None
            self.info.setText(f"ROI set: {self.roi}")

    def draw_mask(self, shape):
        if self.cur_frame is None: return
        img = self.cur_frame.copy(); pts=[]
        if shape=='rect':
            x,y,w,h = map(int, cv2.selectROI("Mask Rect", img, False, False)); cv2.destroyAllWindows()
            if w>0 and h>0: self._ensure_mask(); self.mask_full[y:y+h, x:x+w]=True
        else:
            win = "Mask Circle" if shape=='circ' else "Mask Poly"
            def cb(e,x,y,flags,p):
                if e==cv2.EVENT_LBUTTONDOWN:
                    pts.append((x,y)); cv2.circle(img,(x,y),3,(0,0,255),-1)
                    if shape=='poly' and len(pts)>1:
                        cv2.polylines(img,[np.array(pts)],False,(0,0,255),1)
            cv2.namedWindow(win); cv2.setMouseCallback(win, cb)
            while True:
                cv2.imshow(win, img)
                if cv2.waitKey(1)&0xFF==ord('c'): break
            cv2.destroyAllWindows()
            if shape=='circ' and len(pts)==2:
                (cx,cy),(px,py)=pts; r=int(np.hypot(px-cx,py-cy))
                Y,X = np.ogrid[:img.shape[0],:img.shape[1]]
                self._ensure_mask(); self.mask_full |= ((X-cx)**2+(Y-cy)**2 <= r*r)
            if shape=='poly' and len(pts)>=3:
                self._ensure_mask(); cv2.fillPoly(self.mask_full,[np.array(pts,np.int32)],True)
        if self.roi: x,y,w,h=self.roi; self.mask_roi=self.mask_full[y:y+h,x:x+w]
        self.info.setText("Mask updated")

    def _ensure_mask(self):
        if self.mask_full is None and self.cur_frame is not None:
            h,w = self.cur_frame.shape[:2]; self.mask_full = np.zeros((h,w),bool)

    def clear_mask(self):
        self.mask_full = None; self.mask_roi = None; self.info.setText("Mask cleared")

    def auto_seeds(self):
        if self.cur_frame is None or self.roi is None:
            self.info.setText("Need frame & ROI"); return
        x,y,w,h = self.roi
        gray = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
        mask_roi = self.mask_roi if self.mask_roi is not None else np.zeros_like(gray,bool)
        self.facet, self.step = self.f_spin.value(), self.s_spin.value()
        half = self.facet//2
        seeds=[]
        for cy in range(half, h-half+1, self.step):
            for cx in range(half, w-half+1, self.step):
                if mask_roi[cy,cx]: continue
                patch = gray[cy-half:cy+half, cx-half:cx+half]
                if patch.std()<5: continue
                _,bw = cv2.threshold(patch, patch.mean(),255,cv2.THRESH_BINARY)
                blk = 1 - bw.mean()/255
                if not(0.3<blk<0.7): continue
                gy,gx = np.gradient(patch.astype(float)); mig=np.mean(np.hypot(gx,gy))
                if mig<5: continue
                seeds.append((cx,cy))
        if not seeds:
            self.info.setText("No valid speckle areas"); return
        self.ref_pts = np.array(seeds,np.float32).reshape(-1,1,2)
        self.cum_disp = np.zeros((len(seeds),2),np.float32)
        self.info.setText(f"Seeds: {len(seeds)}")

    def manual_seeds(self):
        if self.cur_frame is None or self.roi is None: return
        tmp = self.cur_frame.copy(); pts=[]
        def cb(e,x,y,flags,p):
            if e==cv2.EVENT_LBUTTONDOWN:
                rx,ry,w,h=self.roi
                if rx<=x<rx+w and ry<=y<ry+h:
                    pts.append((x-rx,y-ry)); cv2.circle(tmp,(x,y),4,(0,255,0),-1)
        cv2.namedWindow("Manual Seeds"); cv2.setMouseCallback("Manual Seeds",cb)
        while True:
            cv2.imshow("Manual Seeds",tmp)
            if cv2.waitKey(1)&0xFF==ord('c'): break
        cv2.destroyAllWindows()
        if pts:
            self.ref_pts = np.array(pts,np.float32).reshape(-1,1,2)
            self.cum_disp = np.zeros((len(pts),2),np.float32)
            self.info.setText(f"Manual seeds: {len(pts)}")

    def set_reference(self):
        if self.cur_frame is None or self.roi is None or self.ref_pts is None:
            self.info.setText("Need ROI & seeds"); return
        x,y,w,h = self.roi
        self.ref_gray = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
        self.frame_count = 0
        self.info.setText("Reference set – running DIC")

    def calibrate_scale(self):
        if self.cur_frame is None:
            QtWidgets.QMessageBox.warning(self,"No frame","Wait for stream."); return
        img = self.cur_frame.copy(); pts=[]
        def cb(e,x,y,flags,p):
            if e==cv2.EVENT_LBUTTONDOWN and len(pts)<2:
                pts.append((x,y)); cv2.circle(img,(x,y),5,(0,0,255),-1)
        cv2.namedWindow("Calibrate"); cv2.setMouseCallback("Calibrate",cb)
        while True:
            cv2.imshow("Calibrate",img)
            if cv2.waitKey(1)&0xFF==ord('c') and len(pts)==2: break
        cv2.destroyAllWindows()
        if len(pts)==2:
            (x1,y1),(x2,y2)=pts; dist=np.hypot(x2-x1,y2-y1)
            mm,ok = QtWidgets.QInputDialog.getDouble(self,"Scale","mm:",10,1e-6,1e6,3)
            if ok and dist>0:
                self.scale_mm = mm/dist
                self.info.setText(f"Scale: {self.scale_mm:.6f} mm/px")

    def process_frame(self):
        if self.cur_frame is None:
            return
        view = self.cur_frame.copy()
        # shade mask full-frame
        if self.mask_full is not None:
            ys,xs = np.where(self.mask_full)
            view[ys,xs] = (view[ys,xs]//2 + 80)
        # if no ref yet, just show ROI
        if self.ref_gray is None or self.ref_pts is None:
            if self.roi:
                x,y,w,h=self.roi
                cv2.rectangle(view,(x,y),(x+w,y+h),(255,255,255),1)
            self.view.setPixmap(QtGui.QPixmap.fromImage(to_qimage(view)).scaled(self.view.size(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            return
        # Lucas-Kanade tracking
        gray = cv2.cvtColor(self.cur_frame,cv2.COLOR_BGR2GRAY)
        x0,y0,w,h=self.roi
        cur_roi = gray[y0:y0+h, x0:x0+w]
        lk = dict(winSize=(self.f_spin.value(),self.f_spin.value()), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
        new,st,_ = cv2.calcOpticalFlowPyrLK(self.ref_gray, cur_roi, self.ref_pts, None, **lk)
        st = st.reshape(-1)
        if not st.any():
            self.info.setText("All points lost"); return
        old_pts = self.ref_pts[st==1].reshape(-1,2)
        new_pts = new[st==1].reshape(-1,2)
        inc = new_pts - old_pts
        # accumulate disp
        if self.cum_disp is None or len(self.cum_disp)!=len(self.ref_pts):
            self.cum_disp = np.zeros((len(self.ref_pts),2),np.float32)
        idx=0
        for i,ok in enumerate(st):
            if ok:
                self.cum_disp[i] += inc[idx]
                idx+=1
        # optional differential ref update
        if self.diff_chk.isChecked():
            self.frame_count += 1
            if self.frame_count >= self.diff_sp.value():
                self.ref_gray = cur_roi.copy()
                for i,ok in enumerate(st):
                    if ok: self.ref_pts[i]=new_pts[idx- (len(st)-i)]
                self.frame_count=0
        # build dense ux,uy maps
        ux = np.zeros((h,w),np.float32)
        uy = np.zeros((h,w),np.float32)
        wt = np.zeros((h,w),np.float32)
        for (px,py),d in zip(self.ref_pts.reshape(-1,2), self.cum_disp):
            ix,iy = int(py), int(px)
            if 0<=ix<h and 0<=iy<w:
                ux[ix,iy] = d[0]*self.scale_mm
                uy[ix,iy] = d[1]*self.scale_mm
                wt[ix,iy] = 1
        k=31
        ux = cv2.GaussianBlur(ux,(k,k),0)/(cv2.GaussianBlur(wt,(k,k),0)+1e-6)
        uy = cv2.GaussianBlur(uy,(k,k),0)/(cv2.GaussianBlur(wt,(k,k),0)+1e-6)
        # compute strain
        sp = self.scale_mm
        exx = np.gradient(ux,sp,axis=1)
        eyy = np.gradient(uy,sp,axis=0)
        metric = self.met_cb.currentText()
        if metric=="Axial Strain": field,unit=eyy,""
        elif metric=="Transverse Strain": field,unit=exx,""
        elif metric=="Poisson":
            with np.errstate(divide='ignore',invalid='ignore'):
                field = -exx/(eyy+1e-12)
            unit=""
        elif metric=="Disp X (mm)": field,unit=ux,"mm"
        else: field,unit=uy,"mm"
        if self.mode_cb.currentText()=="Stability":
            field = cv2.GaussianBlur(field,(11,11),0)
        # color-map overlay
        if self.auto_scale.isChecked():
            vmin,vmax = np.percentile(field,(2,98))
            if vmax-vmin<1e-9: vmax+=1e-6; vmin-=1e-6
            self.vmin_sp.setValue(float(vmin))
            self.vmax_sp.setValue(float(vmax))
        else:
            vmin,vmax = self.vmin_sp.value(), self.vmax_sp.value()
        cm = norm_uint8(field,vmin,vmax)
        cm = cv2.applyColorMap(cm,cv2.COLORMAP_JET)
        alpha = self.alpha_sp.value()
        view_area = view[y0:y0+h, x0:x0+w]
        view[y0:y0+h, x0:x0+w] = cv2.addWeighted(cm,alpha,view_area,1-alpha,0)
        # legend bar
        H,W = view.shape[:2]
        bar = np.linspace(vmax,vmin,H,np.float32)
        bar = norm_uint8(bar,vmin,vmax)
        bar = cv2.applyColorMap(bar.reshape(H,1),cv2.COLORMAP_JET)
        legend = np.repeat(bar, self.cbar.width(), axis=1)
        cv2.putText(legend, metric, (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
        cv2.putText(legend, f"{vmax:.3g} {unit}", (5,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv2.putText(legend, f"{vmin:.3g} {unit}", (5,H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        # display
        img = np.hstack((view, legend))
        self.view.setPixmap(QtGui.QPixmap.fromImage(to_qimage(img)).scaled(
            self.view.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
        for (px,py),ok in zip(self.ref_pts.reshape(-1,2), st):
            if ok:
                cv2.rectangle(img,(int(px+x0)-self.facet//2,int(py+y0)-self.facet//2),
                              (int(px+x0)+self.facet//2,int(py+y0)+self.facet//2),(0,255,0),1)
        self.disp.setText(f"Tracked: {int(st.sum())} pts")

    def closeEvent(self, event):
        self.grabber.stop()
        super().closeEvent(event)

# ---------------------------------------------------------------- entry point
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = DICLive()
    win.resize(1200,700)
    win.show()
    sys.exit(app.exec_())
