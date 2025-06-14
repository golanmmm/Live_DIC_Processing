import sys
import cv2
import numpy as np
import traceback
from PyQt5 import QtCore, QtGui, QtWidgets

class DICLive(QtWidgets.QMainWindow):
    def __init__(self, rtsp_url):
        super().__init__()
        self.setWindowTitle("Live DIC Strain & Poisson GUI")
        # Video source
        self.rtsp = cv2.VideoCapture(rtsp_url)
        if not self.rtsp.isOpened():
            QtWidgets.QMessageBox.critical(self, "Stream Error", f"Cannot open stream: {rtsp_url}")
            sys.exit(1)

        # Video display
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)

        # Controls
        self.info_label = QtWidgets.QLabel("press 'Set Reference' to begin", alignment=QtCore.Qt.AlignCenter)
        self.btn_ref = QtWidgets.QPushButton("Set Reference")
        self.btn_roi = QtWidgets.QPushButton("Select ROI")
        self.btn_clear = QtWidgets.QPushButton("Clear ROI")

        # Inputs
        self.scale_input = QtWidgets.QLineEdit("0.1")
        self.L0_input = QtWidgets.QLineEdit("50.0")
        self.W0_input = QtWidgets.QLineEdit("10.0")

        # Tracking params
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["reference", "incremental"])
        self.quality_spin = QtWidgets.QDoubleSpinBox(); self.quality_spin.setRange(0.001,0.1); self.quality_spin.setValue(0.01)
        self.min_dist_spin = QtWidgets.QSpinBox(); self.min_dist_spin.setRange(5,50); self.min_dist_spin.setValue(10)
        self.outlier_spin = QtWidgets.QDoubleSpinBox(); self.outlier_spin.setRange(0.0,10.0); self.outlier_spin.setValue(2.0)
        self.smooth_spin = QtWidgets.QDoubleSpinBox(); self.smooth_spin.setRange(0.0,1.0); self.smooth_spin.setValue(0.2)

        # Visualization params
        self.metric_combo = QtWidgets.QComboBox(); self.metric_combo.addItems(["axial","transverse","poisson"])
        self.cmin_spin = QtWidgets.QDoubleSpinBox(); self.cmin_spin.setRange(-1.0,1.0); self.cmin_spin.setValue(0.0)
        self.cmax_spin = QtWidgets.QDoubleSpinBox(); self.cmax_spin.setRange(-1.0,1.0); self.cmax_spin.setValue(0.001)
        self.opacity_spin = QtWidgets.QDoubleSpinBox(); self.opacity_spin.setRange(0.0,1.0); self.opacity_spin.setValue(0.4)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("scale mm/px:", self.scale_input)
        form.addRow("gauge L0 mm:", self.L0_input)
        form.addRow("width W0 mm:", self.W0_input)
        form.addRow("mode:", self.mode_combo)
        form.addRow("qualityLevel:", self.quality_spin)
        form.addRow("minDist px:", self.min_dist_spin)
        form.addRow("outlier STD:", self.outlier_spin)
        form.addRow("smooth α:", self.smooth_spin)
        form.addRow("metric:", self.metric_combo)
        form.addRow("cmap min:", self.cmin_spin)
        form.addRow("cmap max:", self.cmax_spin)
        form.addRow("opacity:", self.opacity_spin)

        ctrl = QtWidgets.QVBoxLayout()
        ctrl.addLayout(form)
        for w in (self.btn_ref, self.btn_roi, self.btn_clear, self.info_label): ctrl.addWidget(w)

        main = QtWidgets.QHBoxLayout()
        main.addWidget(self.video_label,3)
        main.addLayout(ctrl,1)

        self.setCentralWidget(QtWidgets.QWidget())
        self.centralWidget().setLayout(main)

        # State
        self.ref_gray = None
        self.ref_pts = None
        self.prev_gray = None
        self.disp_sum = None
        self.roi = None
        self.prev_disp = None

        # Connections
        self.btn_ref.clicked.connect(self.set_reference)
        self.btn_roi.clicked.connect(self.select_roi)
        self.btn_clear.clicked.connect(self.clear_roi)

        # Timer
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.update_frame); self.timer.start(30)

    def set_reference(self):
        try:
            ret, frame = self.rtsp.read()
            if not ret: raise RuntimeError("Frame read failed")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ql, md = self.quality_spin.value(), self.min_dist_spin.value()
            pts = self._detect_points(gray, ql, md)
            self.ref_gray = gray.copy()
            self.ref_pts = pts.astype(np.float32).reshape(-1,1,2)
            self.prev_gray = gray.copy()
            self.disp_sum = None
            self.prev_disp = None
            self.info_label.setText(f"Reference: {pts.shape[0]} pts")
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Reference Error", traceback.format_exc())

    def select_roi(self):
        self.timer.stop()
        ret, frame = self.rtsp.read()
        if ret:
            roi = cv2.selectROI("ROI", frame, showCrosshair=True)
            cv2.destroyWindow("ROI")
            if roi[2] and roi[3]: self.roi = tuple(map(int,roi)); self.info_label.setText(f"ROI {self.roi}")
        self.timer.start(30)

    def clear_roi(self):
        self.roi = None; self.info_label.setText("ROI cleared")

    def update_frame(self):
        try:
            ret, frame = self.rtsp.read()
            if not ret: return
            disp = frame.copy(); gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.ref_gray is None:
                self._show(disp); return
            # choose reference
            if self.mode_combo.currentText()=="incremental":
                base_gray = self.prev_gray
            else: base_gray = self.ref_gray
            # track
            new, st, _ = cv2.calcOpticalFlowPyrLK(
                base_gray, gray, self.ref_pts, None,
                winSize=(21,21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01)
            )
            good = st.flatten()==1; ref = self.ref_pts[good].reshape(-1,2); cur=new[good].reshape(-1,2)
            disp_vec = cur - ref
            # filter/outliers
            mag=np.linalg.norm(disp_vec,axis=1); m,s=mag.mean(),mag.std(); t=self.outlier_spin.value()
            mask=(mag>=m-t*s)&(mag<=m+t*s); disp_vec=disp_vec[mask]; cur=cur[mask]
            # smooth
            a=self.smooth_spin.value()
            if self.prev_disp is not None and disp_vec.shape==self.prev_disp.shape:
                disp_vec=disp_vec*a + self.prev_disp*(1-a)
            self.prev_disp=disp_vec.copy()
            # draw
            for p in cur.astype(int): cv2.circle(disp,tuple(p),3,(0,255,0),-1)
            # accumulate differential
            metric=self.metric_combo.currentText()
            scale=float(self.scale_input.text()); L0=float(self.L0_input.text()); W0=float(self.W0_input.text())
            if self.mode_combo.currentText()=="incremental":
                # compute small map
                small_map = self._compute_strain_map(self.prev_gray, gray, scale, L0, W0, metric)
                if self.disp_sum is None: self.disp_sum=np.zeros_like(small_map)
                self.disp_sum += small_map
                smap = self.disp_sum
            else:
                smap = self._compute_strain_map(self.ref_gray, gray, scale, L0, W0, metric)
            # update prev_gray
            self.prev_gray = gray.copy()
            # overlay
            cmap = self._make_cmap(smap); o=self.opacity_spin.value()
            if self.roi:
                x,y,rw,rh=self.roi;disp[y:y+rh,x:x+rw]=cv2.addWeighted(disp[y:y+rh,x:x+rw],1-o,cmap,o,0)
                cv2.rectangle(disp,(x,y),(x+rw,y+rh),(255,255,255),2)
            else: disp=cv2.addWeighted(disp,1-o,cmap,o,0)
            self._show(disp)
        except Exception:
            QtWidgets.QMessageBox.critical(self,"Update Error",traceback.format_exc())

    def _detect_points(self, gray, ql, md):
        if self.roi:
            h,w=gray.shape; x,y,rw,rh=self.roi
            x,y=max(0,x),max(0,y); rw=min(rw,w-x); rh=min(rh,h-y)
            crop=gray[y:y+rh,x:x+rw]
            pts=cv2.goodFeaturesToTrack(crop,200,ql,md)
            if pts is None: raise RuntimeError
            pts=pts.reshape(-1,2); pts[:,0]+=x; pts[:,1]+=y
        else:
            pts=cv2.goodFeaturesToTrack(gray,200,ql,md)
            if pts is None: raise RuntimeError
            pts=pts.reshape(-1,2)
        return pts

    def _compute_strain_map(self, ref_gray, cur_gray, scale, L0, W0, metric):
        if self.roi:
            x,y,rw,rh=self.roi
            ref_c=ref_gray[y:y+rh,x:x+rw].astype(np.float32)
            cur_c=cur_gray[y:y+rh,x:x+rw].astype(np.float32)
        else:
            ref_c=ref_gray.astype(np.float32); cur_c=cur_gray.astype(np.float32)
        dy=(cur_c-ref_c)*scale/L0; dx=(cur_c-ref_c)*scale/W0
        if metric=="axial": return dy
        if metric=="transverse": return dx
        with np.errstate(divide='ignore',invalid='ignore'):
            pr=-dx/dy; pr[~np.isfinite(pr)]=0; return pr

    def _make_cmap(self, smap):
        vmin,vmax=self.cmin_spin.value(),self.cmax_spin.value();d=vmax-vmin
        if abs(d)<1e-6: norm=np.zeros_like(smap,dtype=np.uint8)
        else: norm=np.clip((smap-vmin)/d*255,0,255).astype(np.uint8)
        return cv2.applyColorMap(norm,cv2.COLORMAP_JET)

    def _show(self, frame):
        h,w,ch=frame.shape; bpl=ch*w
        img=QtGui.QImage(frame.data,w,h,bpl,QtGui.QImage.Format_BGR888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.video_label.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))

    def closeEvent(self,event):
        self.rtsp.release(); super().closeEvent(event)

if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    win=DICLive("rtsp://10.5.0.2:8554/mystream")
    win.resize(1200,600); win.show(); sys.exit(app.exec_())
