#!/usr/bin/env python3
# ------------------------------------------------------------------
# Live-DIC GUI  –  with dual‐format recording + gauge CSV export
# ------------------------------------------------------------------
import sys, os, time, traceback, csv  # <-- CSV added
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

STREAM_URL = "rtsp://10.5.0.2:8554/ueye_cockpit_stream"

# ------------------------- helpers
def to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, w*3, QtGui.QImage.Format_RGB888)

def norm_uint8(arr, vmin, vmax):
    if vmax - vmin == 0: vmax += 1e-6
    out = np.clip((arr - vmin)/(vmax - vmin), 0, 1)
    out[np.isnan(out)] = 0
    return (out * 255).astype(np.uint8)

# ------------------------- RTSP grabber
class Grabber(QtCore.QThread):
    frame = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.keep_running = True
    def run(self):
        while self.keep_running:
            #uEye streamer COM "2" no need for DSHOW
            cap = cv2.VideoCapture(1)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                time.sleep(1); continue
            while self.keep_running and cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    self.frame.emit(frame)
                else:
                    time.sleep(0.02)
            cap.release()
    def stop(self):
        self.keep_running = False
        self.wait()

# ------------------------- main GUI
class CSVWriterThread(QtCore.QThread):
    def __init__(self, filepath, header, q):
        super().__init__()
        self.filepath = filepath
        self.header = header
        self.q = q
        self._running = True
        self._last_flush = time.time()
    def run(self):
        try:
            # open once, buffered
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.header)
                writer.writeheader()
                while self._running:
                    try:
                        row = self.q.get(timeout=0.2)
                    except Exception:
                        row = None
                    if row is None:
                        # periodic flush
                        if time.time()-self._last_flush > 0.5:
                            try:
                                f.flush()
                            except Exception:
                                pass
                            self._last_flush = time.time()
                        continue
                    writer.writerow(row)
                    try:
                        f.flush()
                    except Exception:
                        pass
                    self._last_flush = time.time()
        except Exception as e:
            pass
    def stop(self):
        self._running = False
        self.wait()

class DICLive(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live-DIC GUI – UI-1220ME-M-GL")

        # DIC state
        self.cur = None
        self.roi = None
        self.mask_full = None
        self.mask_roi  = None
        self.ref_gray  = None
        self.ref_pts   = None
        self.cum_disp  = None
        self.facet, self.step = 21, 15
        self.scale_mm = 1.0
        self.frame_cnt = 0
        self.frozen = False
        self.vmin = self.vmax = 0.0

        # gauge
        self.gauge_pts = None
        self.gauge_L0_mm = None

        # --- NEW: export tracking for gauge line averages
        self.export_rows = []  # in-memory cache (optional)
        self._export_start_time = None
        # --- NEW: live-logging infra
        import queue
        self.log_queue = queue.Queue(maxsize=10000)
        self.logger_thread = None
        self.live_logging = False
        self.log_every_n = 1  # frames
        self._last_logged_frame = 0

        # recording & FPS
        self.recording = False
        self.recorder  = None
        self._last_fps_time = time.time()
        self._fps_counter  = 0
        self.fps = 0.0

        # --- UI widgets
        self.view      = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.info      = QtWidgets.QLabel("Waiting for stream…", alignment=QtCore.Qt.AlignCenter)
        self.disp      = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.gauge_lbl = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.lbl_fps   = QtWidgets.QLabel("FPS: 0.0", alignment=QtCore.Qt.AlignCenter)

        # Buttons
        self.btn_roi    = QtWidgets.QPushButton("Select ROI")
        self.btn_mrect  = QtWidgets.QPushButton("Mask Rect")
        self.btn_mcirc  = QtWidgets.QPushButton("Mask Circle")
        self.btn_mpoly  = QtWidgets.QPushButton("Mask Polygon")
        self.btn_clear  = QtWidgets.QPushButton("Clear Mask")
        self.btn_auto   = QtWidgets.QPushButton("Auto Seeds")
        self.btn_manual = QtWidgets.QPushButton("Manual Seeds")
        self.btn_ref    = QtWidgets.QPushButton("Set Reference")
        self.btn_cal    = QtWidgets.QPushButton("Calibrate")
        self.btn_gauge  = QtWidgets.QPushButton("Select Gauge Points")
        self.btn_rec    = QtWidgets.QPushButton("Start Recording")
        # --- NEW: export button (writes CSV readable by Excel)
        self.btn_export = QtWidgets.QPushButton("Export Gauge Averages (CSV)")
        # --- NEW: live logging controls
        self.btn_log_toggle = QtWidgets.QPushButton("Start Live Logging")
        self.spin_logN = QtWidgets.QSpinBox(); self.spin_logN.setRange(1, 1000); self.spin_logN.setValue(1)

        # Format selector
        self.combo_rec_format = QtWidgets.QComboBox()
        self.combo_rec_format.addItems(["Uncompressed AVI", "MP4 H.264"])

        # Controls
        self.spin_facet  = QtWidgets.QSpinBox(); self.spin_facet.setRange(5,201); self.spin_facet.setValue(self.facet)
        self.spin_step   = QtWidgets.QSpinBox(); self.spin_step.setRange(2,200); self.spin_step.setValue(self.step)
        self.combo_mode  = QtWidgets.QComboBox(); self.combo_mode.addItems(["Performance","Accuracy","Stability"])
        self.combo_met   = QtWidgets.QComboBox(); self.combo_met.addItems([
            "Axial Strain","Transverse Strain","Poisson",
            "Equivalent Strain","Principal Strain",
            "Disp X (mm)","Disp Y (mm)",
            "Total Displacement (mm)"
        ])
        self.chk_auto    = QtWidgets.QCheckBox("Auto Scale"); self.chk_auto.setChecked(True)
        self.chk_freeze  = QtWidgets.QCheckBox("Freeze Scale")
        self.spin_vmin   = QtWidgets.QDoubleSpinBox(); self.spin_vmin.setDecimals(6)
        self.spin_vmax   = QtWidgets.QDoubleSpinBox(); self.spin_vmax.setDecimals(6)
        self.spin_alpha  = QtWidgets.QDoubleSpinBox(); self.spin_alpha.setRange(0,1); self.spin_alpha.setSingleStep(0.05); self.spin_alpha.setValue(0.5)
        self.spin_ks     = QtWidgets.QSpinBox(); self.spin_ks.setRange(1,101); self.spin_ks.setSingleStep(2); self.spin_ks.setValue(100)
        self.spin_cmblur = QtWidgets.QSpinBox(); self.spin_cmblur.setRange(1,101); self.spin_cmblur.setSingleStep(2); self.spin_cmblur.setValue(1)
        self.chk_facets  = QtWidgets.QCheckBox("Show Facets"); self.chk_facets.setChecked(True)
        self.chk_diff    = QtWidgets.QCheckBox("Differential ref"); self.chk_diff.setChecked(True)
        self.spin_int    = QtWidgets.QSpinBox(); self.spin_int.setRange(1,500); self.spin_int.setValue(1)

        # Layout
        form = QtWidgets.QFormLayout()
        for row in [
            (self.combo_mode,),
            ("Facet px:",  self.spin_facet),
            ("Step px:",   self.spin_step),
            (self.btn_auto,),
            (self.btn_manual,),
            (self.btn_roi,),
            (self.btn_mrect,),
            (self.btn_mcirc,),
            (self.btn_mpoly,),
            (self.btn_clear,),
            (self.btn_ref,),
            (self.btn_cal,),
            (self.btn_gauge,),
            (self.btn_rec,),
            ("Rec Format:", self.combo_rec_format),
            # --- NEW: export + live logging controls
            (self.btn_export,),
            (self.btn_log_toggle,),
            ("Log every N frames:", self.spin_logN),
            ("FPS:",       self.lbl_fps),
            ("Metric:",    self.combo_met),
            (self.chk_auto,),
            (self.chk_freeze,),
            ("vmin:",      self.spin_vmin),
            ("vmax:",      self.spin_vmax),
            ("Opacity:",   self.spin_alpha),
            ("Disp-smooth ksize:", self.spin_ks),
            ("Colormap-blur ksize:", self.spin_cmblur),
            (self.chk_facets,),
            (self.chk_diff,),
            ("Diff interval:", self.spin_int),
            (self.disp,),
            (self.gauge_lbl,),
            (self.info,)
        ]:
            if len(row)==2:
                form.addRow(row[0], row[1])
            else:
                form.addRow(row[0])
        side = QtWidgets.QVBoxLayout(); side.addLayout(form); side.addStretch()
        main = QtWidgets.QHBoxLayout(); main.addWidget(self.view,3); main.addLayout(side,1)
        container = QtWidgets.QWidget(); container.setLayout(main)
        self.setCentralWidget(container)
        self.setMinimumWidth(1100)

        # Signals
        self.btn_roi.   clicked.connect(self.select_roi)
        self.btn_mrect. clicked.connect(lambda: self.draw_mask("rect"))
        self.btn_mcirc. clicked.connect(lambda: self.draw_mask("circ"))
        self.btn_mpoly. clicked.connect(lambda: self.draw_mask("poly"))
        self.btn_clear.clicked.connect(self.clear_mask)
        self.btn_auto.  clicked.connect(self.auto_seed)
        self.btn_manual.clicked.connect(self.manual_seed)
        self.btn_ref.   clicked.connect(self.set_reference)
        self.btn_cal.   clicked.connect(self.calibrate)
        self.btn_gauge. clicked.connect(self.select_gauge)
        self.btn_rec.   clicked.connect(self.toggle_record)
        self.combo_mode.currentTextChanged.connect(self.apply_mode)
        # --- NEW: export handler
        self.btn_export.clicked.connect(self.export_csv)
        # --- NEW: live logging handlers
        self.btn_log_toggle.clicked.connect(self.toggle_live_logging)
        self.spin_logN.valueChanged.connect(lambda v: setattr(self, 'log_every_n', int(v)))

        # RTSP grabber
        self.grabber = Grabber(STREAM_URL)
        self.grabber.frame.connect(lambda f: setattr(self, "cur", f))
        self.grabber.start()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process)
        self.timer.start(30)

    # -------- recording toggle
    def toggle_record(self):
        if not self.recording:
            fmt = self.combo_rec_format.currentText()
            if fmt == "Uncompressed AVI":
                filters = "AVI (*.avi)"
            else:
                filters = "MP4 H.264 (*.mp4)"
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Recording", "", filters)
            if not fname:
                return
            ext = os.path.splitext(fname)[1].lower()
            if fmt == "Uncompressed AVI":
                fourcc = 0
            else:
                fourcc = cv2.VideoWriter_fourcc(*"H264")
            # estimate size from current frame + legend
            h, w = self.cur.shape[:2]
            vw = w + 100
            vh = h
            fps = max(1, int(self.fps))
            self.recorder = cv2.VideoWriter(fname, fourcc, fps, (vw, vh))
            if not self.recorder.isOpened():
                self.info.setText("Failed to open file for recording")
            else:
                self.recording = True
                self.btn_rec.setText("Stop Recording")
                self.info.setText(f"Recording → {fname}")
        else:
            self.recording = False
            if self.recorder:
                self.recorder.release()
            self.recorder = None
            self.btn_rec.setText("Start Recording")
            self.info.setText("Recording stopped")

    # -------- UI callbacks
    def apply_mode(self):
        m = self.combo_mode.currentText()
        if m == "Performance":
            self.spin_facet.setValue(21); self.spin_step.setValue(20)
        elif m == "Accuracy":
            self.spin_facet.setValue(41); self.spin_step.setValue(10)
        else:
            self.spin_facet.setValue(41); self.spin_step.setValue(20)
        self.info.setText(f"{m} preset")

    def select_roi(self):
        if self.cur is None: return
        x,y,w,h = map(int, cv2.selectROI("ROI", self.cur, False, False))
        cv2.destroyAllWindows()
        if w>0 and h>0:
            self.roi = (x,y,w,h)
            self.mask_roi = (self.mask_full[y:y+h, x:x+w] if self.mask_full is not None else None)
            self.ref_gray = None
            self.info.setText(f"ROI set: {self.roi}")

    def _ensure_mask(self):
        if self.cur is not None and self.mask_full is None:
            H,W = self.cur.shape[:2]
            self.mask_full = np.zeros((H,W), bool)

    def draw_mask(self, shape):
        if self.cur is None or self.roi is None: return
        img = self.cur.copy(); pts=[]
        if shape=="rect":
            x,y,w,h = map(int, cv2.selectROI("Mask Rect", img, False, False))
            cv2.destroyAllWindows()
            if w>0 and h>0:
                self._ensure_mask()
                self.mask_full[y:y+h, x:x+w] = True
        elif shape=="circ":
            def cb(evt, xx, yy, flags, prm):
                if evt==cv2.EVENT_LBUTTONDOWN and len(pts)<2:
                    pts.append((xx,yy)); cv2.circle(img,(xx,yy),4,(0,0,255),-1)
            cv2.namedWindow("Mask Circle"); cv2.setMouseCallback("Mask Circle", cb)
            while True:
                cv2.imshow("Mask Circle", img)
                k = cv2.waitKey(1)&0xFF
                if k==ord('c') and len(pts)==2: break
                if k==27: pts=[]; break
            cv2.destroyAllWindows()
            if len(pts)==2:
                (cx,cy),(px,py) = pts
                r = int(np.hypot(px-cx, py-cy))
                self._ensure_mask()
                Y,X = np.ogrid[:img.shape[0],:img.shape[1]]
                self.mask_full |= ((X-cx)**2 + (Y-cy)**2 <= r*r)
        else:  # poly
            def cb(evt, xx, yy, flags, prm):
                if evt==cv2.EVENT_LBUTTONDOWN:
                    pts.append((xx,yy)); cv2.circle(img,(xx,yy),3,(0,0,255),-1)
                    if len(pts)>1:
                        cv2.polylines(img,[np.array(pts)],False,(0,0,255),1)
            cv2.namedWindow("Mask Poly"); cv2.setMouseCallback("Mask Poly", cb)
            while True:
                cv2.imshow("Mask Poly", img)
                if cv2.waitKey(1)&0xFF == ord('c'): break
            cv2.destroyAllWindows()
            if len(pts)>=3:
                self._ensure_mask()
                cv2.fillPoly(self.mask_full,[np.array(pts,np.int32)],True)
        x,y,w,h = self.roi
        self.mask_roi = self.mask_full[y:y+h, x:x+w]
        self.info.setText("Mask updated")

    def clear_mask(self):
        self.mask_full = None
        self.mask_roi  = None
        self.info.setText("Mask cleared")

    def auto_seed(self):
        if self.cur is None or self.roi is None:
            self.info.setText("Define ROI first"); return
        x,y,w,h = self.roi
        gray = cv2.cvtColor(self.cur, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
        mask = (self.mask_full[y:y+h, x:x+w] if self.mask_full is not None else None)
        self.facet,self.step = self.spin_facet.value(), self.spin_step.value()
        half = self.facet//2
        qual = np.zeros_like(gray,bool)
        for cy in range(half, h-half, half):
            for cx in range(half, w-half, half):
                patch = gray[cy-half:cy+half, cx-half:cx+half]
                if patch.size==0 or patch.std()<5: continue
                _,bw = cv2.threshold(patch,patch.mean(),255,cv2.THRESH_BINARY)
                blk = 1 - bw.mean()/255
                if not (0.3<blk<0.7): continue
                gy,gx = np.gradient(patch.astype(float))
                if np.mean(np.hypot(gx,gy))<5: continue
                qual[cy-half:cy+half, cx-half:cx+half] = True
        if mask is not None:
            qual &= ~mask
        pts = [(cx,cy) for cy in range(half,h-half+1,self.step)
                      for cx in range(half,w-half+1,self.step)
                      if qual[cy,cx]]
        if not pts:
            self.info.setText("No speckle seeds found"); return
        self.ref_pts  = np.array(pts,np.float32).reshape(-1,1,2)
        self.cum_disp = np.zeros((len(self.ref_pts),2), np.float32)
        self.info.setText(f"Auto seeds: {len(pts)}")

    def manual_seed(self):
        if self.cur is None or self.roi is None: return
        img = self.cur.copy(); pts=[]
        def cb(evt, xx, yy, flags, prm):
            if evt==cv2.EVENT_LBUTTONDOWN:
                rx,ry,w,h = self.roi
                if rx<=xx<rx+w and ry<=yy<ry+h:
                    pts.append((xx-rx, yy-ry))
                    cv2.circle(img,(xx,yy),4,(0,255,0),-1)
        cv2.namedWindow("Manual Seeds"); cv2.setMouseCallback("Manual Seeds", cb)
        while True:
            cv2.imshow("Manual Seeds", img)
            k = cv2.waitKey(1)&0xFF
            if k==ord('c'): break
            if k==27: pts=[]; break
        cv2.destroyAllWindows()
        if pts:
            self.ref_pts  = np.array(pts,np.float32).reshape(-1,1,2)
            self.cum_disp = np.zeros((len(self.ref_pts),2), np.float32)
            self.info.setText(f"Manual seeds: {len(pts)}")

    def select_gauge(self):
        if self.cur is None or self.roi is None:
            self.info.setText("Define ROI first"); return
        img = self.cur.copy(); pts=[]
        def cb(evt, xx, yy, flags, prm):
            if evt==cv2.EVENT_LBUTTONDOWN and len(pts)<2:
                pts.append((xx,yy)); cv2.circle(img,(xx,yy),5,(255,255,255),-1)
        cv2.namedWindow("Gauge"); cv2.setMouseCallback("Gauge", cb)
        while True:
            cv2.imshow("Gauge", img)
            k = cv2.waitKey(1)&0xFF
            if k==ord('c') and len(pts)==2: break
            if k==27: pts=[]; break
        cv2.destroyAllWindows()
        if len(pts)==2:
            self.gauge_pts = pts
            dx,dy = pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]
            L0 = np.hypot(dx,dy)
            self.gauge_L0_mm = L0 * self.scale_mm
            self.gauge_lbl.setText(f"Gauge L₀ = {self.gauge_L0_mm:.3f} mm")

    def set_reference(self):
        try:
            if self.cur is None or self.roi is None or self.ref_pts is None:
                self.info.setText("Need ROI & seeds"); return
            x,y,w,h = self.roi
            gray_full = cv2.cvtColor(self.cur, cv2.COLOR_BGR2GRAY)
            self.ref_gray = gray_full[y:y+h, x:x+w].copy()
            self.cum_disp = np.zeros((len(self.ref_pts),2), np.float32)
            self.frame_cnt = 0
            self.frozen = False
            # --- NEW: reset export time series
            self._export_start_time = time.time()
            self.export_rows = []
            self.info.setText("Reference set")
        except Exception as e:
            self.info.setText(f"Ref error: {e}")

    def calibrate(self):
        if self.cur is None:
            return
        img = self.cur.copy(); pts=[]
        def cb(evt, xx, yy, flags, prm):
            if evt==cv2.EVENT_LBUTTONDOWN and len(pts)<2:
                pts.append((xx,yy)); cv2.circle(img,(xx,yy),5,(0,0,255),-1)
        cv2.namedWindow("Calibrate"); cv2.setMouseCallback("Calibrate", cb)
        while True:
            cv2.imshow("Calibrate", img)
            k = cv2.waitKey(1)&0xFF
            if k==ord('c') and len(pts)==2: break
            if k==27: pts=[]; break
        cv2.destroyAllWindows()
        if len(pts)==2:
            (x1,y1),(x2,y2)=pts
            pix = np.hypot(x2-x1, y2-y1)
            mm,ok = QtWidgets.QInputDialog.getDouble(self,"Real dist (mm)","mm:",10,1e-6,1e6,3)
            if ok and pix>0:
                self.scale_mm = mm/pix
                self.info.setText(f"Scale = {self.scale_mm:.6f} mm/pix")

    # --- NEW: helper — average values along a line in ROI coords
    def _avg_along_roi_line(self, field, rx1, ry1, rx2, ry2):
        h, w = field.shape[:2]
        n = max(2, int(np.hypot(rx2-rx1, ry2-ry1)) + 1)
        xs = np.clip(np.round(np.linspace(rx1, rx2, n)).astype(int), 0, w-1)
        ys = np.clip(np.round(np.linspace(ry1, ry2, n)).astype(int), 0, h-1)
        vals = field[ys, xs]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float('nan')
        return float(np.mean(vals))

    # --- NEW: export handler
    def export_csv(self):
        if not self.export_rows:
            QtWidgets.QMessageBox.information(self, "Export", "No data captured yet. Make sure a gauge is set and a reference is active.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Gauge Averages", "gauge_averages.csv", "CSV Files (*.csv)")
        if not fname:
            return
        header = list(self.export_rows[0].keys())
        try:
            with open(fname, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for row in self.export_rows:
                    writer.writerow(row)
            self.info.setText(f"CSV saved → {fname}")
        except Exception as e:
            self.info.setText(f"CSV error: {e}")

    # --- NEW: live logging toggle
    def toggle_live_logging(self):
        if not self.live_logging:
            # preconditions
            if self._export_start_time is None:
                QtWidgets.QMessageBox.information(self, "Live Logging", "Set a reference first (and preferably a gauge line)." )
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Live CSV Log", "gauge_live_log.csv", "CSV Files (*.csv)")
            if not path:
                return
            header = [
                "time_s","frame",
                "exx_avg","eyy_avg","exy_avg","poisson_avg",
                "equiv_strain_avg","principal_strain_avg",
                "ux_avg_mm","uy_avg_mm","disp_total_avg_mm"
            ]
            try:
                self.logger_thread = CSVWriterThread(path, header, self.log_queue)
                self.logger_thread.start()
                self.live_logging = True
                self.btn_log_toggle.setText("Stop Live Logging")
                self.info.setText(f"Live logging → {path}")
            except Exception as e:
                self.logger_thread = None
                self.live_logging = False
                self.info.setText(f"Log start error: {e}")
        else:
            try:
                if self.logger_thread is not None:
                    self.logger_thread.stop()
                self.logger_thread = None
            finally:
                self.live_logging = False
                self.btn_log_toggle.setText("Start Live Logging")
                self.info.setText("Live logging stopped")

    def process(self):
        if self.cur is None:
            return

        # FPS
        self._fps_counter += 1
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self.fps = self._fps_counter/(now - self._last_fps_time)
            self.lbl_fps.setText(f"FPS: {self.fps:.1f}")
            self._fps_counter = 0
            self._last_fps_time = now

        vis = self.cur.copy()

        # shade mask
        if self.mask_full is not None:
            ys,xs = np.where(self.mask_full)
            vis[ys,xs] = (vis[ys,xs]//2 + 80)

        # before reference
        if self.ref_gray is None or self.ref_pts is None:
            if self.roi:
                x,y,w,h = self.roi
                cv2.rectangle(vis,(x,y),(x+w,y+h),(255,255,255),1)
            self.view.setPixmap(QtGui.QPixmap.fromImage(
                to_qimage(vis)).scaled(self.view.size(),
                QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            return

        try:
            # optical flow & cumulative update
            self.frame_cnt += 1
            gray = cv2.cvtColor(self.cur, cv2.COLOR_BGR2GRAY)
            x0,y0,w,h = self.roi
            if w < self.facet*2 or h < self.facet*2:
                self.info.setText("ROI too small"); return
            cur_roi = gray[y0:y0+h, x0:x0+w]
            lk = dict(winSize=(self.facet,self.facet), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
            new,st,_ = cv2.calcOpticalFlowPyrLK(self.ref_gray, cur_roi,
                                               self.ref_pts, None, **lk)
            st = st.reshape(-1)
            if not st.any():
                self.info.setText("All points lost"); return

            old_pts = self.ref_pts.reshape(-1,2)
            new_pts = new.reshape(-1,2)
            delta   = (new_pts - old_pts).astype(np.float32)
            rigid   = np.nanmean(delta, axis=0)
            delta  -= rigid
            delta *= st.reshape(-1,1)
            self.cum_disp += delta

            if self.chk_diff.isChecked() and self.frame_cnt >= self.spin_int.value():
                self.ref_gray = cur_roi.copy()
                self.ref_pts[st==1] = new[st==1].reshape(-1,1,2)
                self.frame_cnt = 0
                self.frozen = False

            # dense disp maps
            ux = np.zeros((h,w), np.float32)
            uy = np.zeros((h,w), np.float32)
            wt = np.zeros((h,w), np.float32)
            for (p,), d in zip(self.ref_pts, self.cum_disp):
                iy,ix = int(p[1]), int(p[0])
                if 0<=iy<h and 0<=ix<w:
                    ux[iy,ix], uy[iy,ix], wt[iy,ix] = d[0]*self.scale_mm, d[1]*self.scale_mm, 1

            # smooth disp
            k = self.spin_ks.value()|1
            k = min(k, (min(h,w)-1)|1)
            ux = cv2.GaussianBlur(ux,(k,k),0)/(cv2.GaussianBlur(wt,(k,k),0)+1e-6)
            uy = cv2.GaussianBlur(uy,(k,k),0)/(cv2.GaussianBlur(wt,(k,k),0)+1e-6)

            # compute strains + total disp
            sp = self.scale_mm
            exx = np.gradient(ux,sp,axis=1)
            eyy = np.gradient(uy,sp,axis=0)
            dux_dy = np.gradient(ux,sp,axis=0)
            duy_dx = np.gradient(uy,sp,axis=1)
            exy = 0.5*(dux_dy+duy_dx)

            # also compute derived fields for export
            with np.errstate(divide='ignore',invalid='ignore'):
                poisson = -exx/(eyy+1e-12)
                poisson[np.isnan(poisson)] = 0
            eqv = np.sqrt(0.5*((exx-eyy)**2 + exx**2 + eyy**2) + 3*exy**2)
            princ = 0.5*((exx+eyy) + np.sqrt((exx-eyy)**2+4*exy**2))
            disp_tot = np.hypot(ux, uy)
            metric = self.combo_met.currentText()
            if metric=="Axial Strain":
                field,unit = eyy, ""
            elif metric=="Transverse Strain":
                field,unit = exx, ""
            elif metric=="Poisson":
                field,unit = poisson, ""
            elif metric=="Equivalent Strain":
                field,unit = eqv, ""
            elif metric=="Principal Strain":
                field,unit = princ, ""
            elif metric=="Disp X (mm)":
                field,unit = ux, "mm"
            elif metric=="Disp Y (mm)":
                field,unit = uy, "mm"
            else:
                field = disp_tot
                unit = "mm"

            # gauge line deformation
            if self.gauge_pts and self.gauge_L0_mm:
                (gx1,gy1),(gx2,gy2) = self.gauge_pts
                rx,ry,_w,_h = self.roi
                ry1,rx1 = gy1-ry, gx1-rx
                ry2,rx2 = gy2-ry, gx2-rx
                if 0<=ry1<h and 0<=rx1<w and 0<=ry2<h and 0<=rx2<w:
                    dp1x,dp1y = ux[ry1,rx1]/self.scale_mm, uy[ry1,rx1]/self.scale_mm
                    dp2x,dp2y = ux[ry2,rx2]/self.scale_mm, uy[ry2,rx2]/self.scale_mm
                    p1 = (int(gx1+dp1x), int(gy1+dp1y))
                    p2 = (int(gx2+dp2x), int(gy2+dp2y))
                    cv2.line(vis, p1, p2, (0,255,255), 2)

                    # --- NEW: capture gauge-line averages each frame
                    if self._export_start_time is not None:
                        # averages along the line in ROI coords
                        ax = self._avg_along_roi_line(exx, rx1, ry1, rx2, ry2)
                        ay = self._avg_along_roi_line(eyy, rx1, ry1, rx2, ry2)
                        axy = self._avg_along_roi_line(exy, rx1, ry1, rx2, ry2)
                        apois = self._avg_along_roi_line(poisson, rx1, ry1, rx2, ry2)
                        aeqv = self._avg_along_roi_line(eqv, rx1, ry1, rx2, ry2)
                        aprinc = self._avg_along_roi_line(princ, rx1, ry1, rx2, ry2)
                        aux = self._avg_along_roi_line(ux, rx1, ry1, rx2, ry2)
                        auy = self._avg_along_roi_line(uy, rx1, ry1, rx2, ry2)
                        adisp = self._avg_along_roi_line(disp_tot, rx1, ry1, rx2, ry2)
                        t_rel = now - self._export_start_time
                        row = {
                            "time_s": round(t_rel, 6),
                            "frame": int(self.frame_cnt),
                            "exx_avg": ax,
                            "eyy_avg": ay,
                            "exy_avg": axy,
                            "poisson_avg": apois,
                            "equiv_strain_avg": aeqv,
                            "principal_strain_avg": aprinc,
                            "ux_avg_mm": aux,
                            "uy_avg_mm": auy,
                            "disp_total_avg_mm": adisp
                        }
                        # store in-memory list (optional for post-export)
                        self.export_rows.append(row)
                        # push to logger queue if enabled, throttled by N frames
                        if self.live_logging and (self.frame_cnt - self._last_logged_frame) >= self.log_every_n:
                            try:
                                self.log_queue.put_nowait(row)
                                self._last_logged_frame = self.frame_cnt
                            except Exception:
                                # queue full → drop to avoid blocking DIC pipeline
                                pass

            # auto/freeze scale
            if self.chk_auto.isChecked():
                if self.chk_freeze.isChecked():
                    if not self.frozen:
                        self.vmin,self.vmax = np.percentile(field,(2,98))
                        if self.vmax-self.vmin<1e-9:
                            self.vmax+=1e-6; self.vmin-=1e-6
                        self.spin_vmin.setValue(self.vmin)
                        self.spin_vmax.setValue(self.vmax)
                        self.frozen = True
                else:
                    self.vmin,self.vmax = np.percentile(field,(2,98))
                    if self.vmax-self.vmin<1e-9:
                        self.vmax+=1e-6; self.vmin-=1e-6
                    self.spin_vmin.setValue(self.vmin)
                    self.spin_vmax.setValue(self.vmax)
                    self.frozen = False
            if not self.chk_auto.isChecked():
                vmin,vmax = self.spin_vmin.value(), self.spin_vmax.value()
            else:
                vmin,vmax = self.vmin, self.vmax

            # color map overlay
            cm = cv2.applyColorMap(norm_uint8(field, vmin, vmax), cv2.COLORMAP_JET)
            k2 = self.spin_cmblur.value()|1
            if k2>1:
                cm = cv2.GaussianBlur(cm,(k2,k2),0)
            alpha = self.spin_alpha.value()
            vis[y0:y0+h, x0:x0+w] = cv2.addWeighted(cm, alpha,
                                                   vis[y0:y0+h, x0:x0+w], 1-alpha,0)

            # legend
            H = vis.shape[0]
            bar = norm_uint8(np.linspace(vmax,vmin,H,np.float32), vmin,vmax)
            bar = cv2.applyColorMap(bar.reshape(H,1), cv2.COLORMAP_JET)
            leg = np.repeat(bar, 100, axis=1)
            cv2.putText(leg, metric, (5,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            cv2.putText(leg, f"{vmax:.3g} {unit}", (5,55), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(leg, f"{vmin:.3g} {unit}", (5,H-10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            vis = np.hstack((vis, leg))

            # facets
            if self.chk_facets.isChecked():
                half = self.facet//2
                for (p,), ok in zip(self.ref_pts, st):
                    if ok:
                        cx,cy = int(p[0]+x0), int(p[1]+y0)
                        cv2.rectangle(vis,(cx-half,cy-half),(cx+half,cy+half),(0,255,0),1)

            # record
            if self.recording and self.recorder:
                try:
                    self.recorder.write(vis)
                except Exception as e:
                    self.info.setText(f"Record error: {e}")

            # display
            self.view.setPixmap(QtGui.QPixmap.fromImage(
                to_qimage(vis)).scaled(self.view.size(),
                QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            self.disp.setText(f"Tracked pts: {int(st.sum())}")

        except cv2.error as e:
            self.info.setText(f"CV error: {e}")
        except Exception:
            self.info.setText(traceback.format_exc())

    def closeEvent(self, ev):
        # stop threads first
        if self.logger_thread is not None:
            try:
                self.logger_thread.stop()
            except Exception:
                pass
        self.grabber.stop()
        if self.recorder:
            self.recorder.release()
        super().closeEvent(ev)

# ------------------------- entry point
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DICLive()
    gui.resize(1200,700)
    gui.show()
    sys.exit(app.exec_())
