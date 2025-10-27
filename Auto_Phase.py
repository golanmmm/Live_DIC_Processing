import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QInputDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class CalibrationPopup(QWidget):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Calibration: Select 2 Points')
        self.original_image = image.copy()
        self.scaled_img, self.scale = self.fit_image_to_window(image, 800, 600)
        self.point1 = None
        self.point2 = None
        self.label = QLabel(self)
        self.update_display()
        self.setFixedSize(self.scaled_img.shape[1], self.scaled_img.shape[0])

    def fit_image_to_window(self, img, maxw, maxh):
        h, w = img.shape[:2]
        scale = min(maxw / w, maxh / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h)), scale

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.point1 is None:
                self.point1 = (event.x(), event.y())
            elif self.point2 is None:
                self.point2 = (event.x(), event.y())
                self.update_display()
                self.calibrate_distance()
            self.update_display()

    def update_display(self):
        disp_img = cv2.cvtColor(self.scaled_img, cv2.COLOR_BGR2RGB)
        disp_img = disp_img.copy()
        if self.point1:
            cv2.circle(disp_img, self.point1, 5, (255, 0, 0), -1)
        if self.point2:
            cv2.circle(disp_img, self.point2, 5, (0, 255, 0), -1)
            cv2.line(disp_img, self.point1, self.point2, (0, 255, 0), 2)
        h, w, ch = disp_img.shape
        qt_img = QImage(disp_img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.label.setPixmap(pixmap)
        self.label.setFixedSize(w, h)

    def calibrate_distance(self):
        px_dist = np.sqrt((self.point1[0] - self.point2[0])**2 + (self.point1[1] - self.point2[1])**2)
        px_dist_original = px_dist / self.scale
        micron, ok = QInputDialog.getDouble(self, 'Calibration', 'Enter real distance (microns):', min=0.1)
        if ok:
            self.parent().set_px_per_micron(px_dist_original / micron)
            self.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ASTM A247 Graphite Analysis')
        self.px_per_micron = None
        self.image = None
        self.cv_img = None
        self.calib_popup = None  # Hold popup reference!
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.img_label = QLabel('Load an image to start.', self)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(800, 600)
        layout.addWidget(self.img_label)

        btn_load = QPushButton('Load Image', self)
        btn_load.clicked.connect(self.load_image)
        layout.addWidget(btn_load)

        btn_calib = QPushButton('Calibrate px/mm', self)
        btn_calib.clicked.connect(self.calibrate)
        layout.addWidget(btn_calib)

        btn_detect = QPushButton('Detect Phases', self)
        btn_detect.clicked.connect(self.analyze_graphite)
        layout.addWidget(btn_detect)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if fname:
            self.cv_img = cv2.imread(fname)
            if self.cv_img is None:
                QMessageBox.warning(self, 'Error', 'Failed to load image!')
                return
            self.show_image(self.cv_img)
            self.image = self.cv_img.copy()
            self.px_per_micron = None

    def show_image(self, img, overlay=None):
        disp_img, scale = self.fit_image_to_window(img, 800, 600)
        if overlay is not None:
            overlay_resized = cv2.resize(overlay, (disp_img.shape[1], disp_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            disp_img = cv2.addWeighted(disp_img, 0.8, overlay_resized, 0.5, 0)
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
        h, w = disp_img.shape[:2]
        qimg = QImage(disp_img.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.img_label.setPixmap(pix)

    def fit_image_to_window(self, img, maxw, maxh):
        h, w = img.shape[:2]
        scale = min(maxw / w, maxh / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h)), scale

    def calibrate(self):
        if self.cv_img is None:
            QMessageBox.warning(self, 'No Image', 'Please load an image first.')
            return
        self.calib_popup = CalibrationPopup(self.cv_img, parent=self)
        self.calib_popup.show()

    def set_px_per_micron(self, val):
        self.px_per_micron = val
        QMessageBox.information(self, 'Calibration', f'Calibration set: {val:.4f} px/micron')

    def analyze_graphite(self):
        if self.cv_img is None or self.px_per_micron is None:
            QMessageBox.warning(self, 'Missing Data', 'Load and calibrate image first.')
            return
        gray = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.medianBlur(binary, 5)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Color palette for up to 8 size classes
        colors = [
            (255, 0, 0),      # Class 1 - Blue
            (0, 128, 255),    # Class 2 - Orange
            (0, 255, 255),    # Class 3 - Yellow
            (0, 255, 0),      # Class 4 - Green
            (255, 255, 0),    # Class 5 - Cyan
            (255, 0, 255),    # Class 6 - Magenta
            (128, 0, 255),    # Class 7 - Violet
            (0, 0, 255),      # Class 8 - Red
        ]
        size_bins = [(640, float('inf')), (320, 640), (160, 320), (80, 160), (40, 80), (20, 40), (10, 20), (0, 10)]
        size_classes = [0] * 8
        overlay = np.zeros_like(self.cv_img)
        diameters = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 5:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            diameter = radius * 2 / self.px_per_micron
            diameters.append(diameter)
            class_idx = None
            for i, (low, high) in enumerate(size_bins):
                if low <= diameter < high:
                    size_classes[i] += 1
                    class_idx = i
                    break
            if class_idx is not None:
                cv2.drawContours(overlay, [c], -1, colors[class_idx], thickness=cv2.FILLED)
        self.show_image(self.cv_img, overlay)
        total = sum(size_classes)
        class_report = []
        for i, count in enumerate(size_classes):
            if count > 0:
                pct = 100 * count / total
                class_report.append(f'Class {i + 1}: {pct:.1f}%')
        msg = '\n'.join(class_report) if class_report else 'No graphite particles detected!'
        QMessageBox.information(self, 'Graphite Size Classes', f'Graphite size distribution:\n{msg}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
