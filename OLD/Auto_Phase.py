import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QInputDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtCore import Qt, QRect

class CalibrationPopup(QWidget):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Calibration: Select 2 Points')
        self.original_image = image.copy()
        self.scaled_img, self.scale, self.offset = self.fit_image_to_window(image, 800, 600)
        self.point1 = None
        self.point2 = None
        self.label = QLabel(self)
        self.update_display()
        self.setFixedSize(self.scaled_img.shape[1], self.scaled_img.shape[0])

    def fit_image_to_window(self, img, maxw, maxh):
        h, w = img.shape[:2]
        scale = min(maxw / w, maxh / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        out_img = np.zeros((maxh, maxw, 3), dtype=np.uint8)
        y_offset = (maxh - new_h) // 2
        x_offset = (maxw - new_w) // 2
        out_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        return out_img, scale, (x_offset, y_offset)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x, y = event.x(), event.y()
            if self.point1 is None:
                self.point1 = (x, y)
            elif self.point2 is None:
                self.point2 = (x, y)
                self.update_display()
                self.calibrate_distance()
            self.update_display()

    def update_display(self):
        disp_img = cv2.cvtColor(self.scaled_img, cv2.COLOR_BGR2RGB).copy()
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
        (ox, oy) = self.offset
        # Map screen points back to image coordinates
        pt1 = (self.point1[0] - ox, self.point1[1] - oy)
        pt2 = (self.point2[0] - ox, self.point2[1] - oy)
        px_dist = np.sqrt((pt1[0])**2 + (pt1[1])**2 + (pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        px_dist_original = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2) / self.scale
        micron, ok = QInputDialog.getDouble(self, 'Calibration', 'Enter real distance (microns):', min=0.1)
        if ok:
            self.parent().set_px_per_micron(px_dist_original / micron)
            self.close()

class MaskWidget(QLabel):
    def __init__(self, img, scale, offset, maxw, maxh, parent=None):
        super().__init__(parent)
        self.base_img, self.scale, self.offset = img, scale, offset
        self.maxw, self.maxh = maxw, maxh
        self.drawing = False
        self.rect_start = None
        self.rect_end = None
        self.mask_rects = []
        self.update_pixmap()
        self.setMouseTracking(True)
        self.setFixedSize(self.maxw, self.maxh)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x, y = event.x(), event.y()
            self.drawing = True
            self.rect_start = (x, y)
            self.rect_end = (x, y)
            self.update_pixmap()

    def mouseMoveEvent(self, event):
        if self.drawing:
            x, y = event.x(), event.y()
            self.rect_end = (x, y)
            self.update_pixmap()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            x, y = event.x(), event.y()
            self.rect_end = (x, y)
            self.mask_rects.append((self.rect_start, self.rect_end))
            self.rect_start = None
            self.rect_end = None
            self.update_pixmap()

    def update_pixmap(self):
        disp = cv2.cvtColor(self.base_img, cv2.COLOR_BGR2RGB).copy()
        # Letterbox
        out_img = np.zeros((self.maxh, self.maxw, 3), dtype=np.uint8)
        y_offset, x_offset = self.offset[1], self.offset[0]
        h, w = disp.shape[:2]
        out_img[y_offset:y_offset + h, x_offset:x_offset + w] = disp
        # Draw mask rectangles
        pixmap = QPixmap.fromImage(QImage(out_img.data, self.maxw, self.maxh, self.maxw * 3, QImage.Format_RGB888))
        qp = QPainter(pixmap)
        qp.setPen(QColor(255, 0, 0, 120))
        for r1, r2 in self.mask_rects:
            rect = QRect(r1[0], r1[1], r2[0] - r1[0], r2[1] - r1[1])
            qp.fillRect(rect.normalized(), QColor(255, 0, 0, 100))
        if self.drawing and self.rect_start and self.rect_end:
            rect = QRect(self.rect_start[0], self.rect_start[1], self.rect_end[0] - self.rect_start[0], self.rect_end[1] - self.rect_start[1])
            qp.fillRect(rect.normalized(), QColor(255, 0, 0, 100))
        qp.end()
        self.setPixmap(pixmap)
        self.setFixedSize(self.maxw, self.maxh)

    def get_mask(self, orig_shape):
        mask = np.zeros((self.maxh, self.maxw), np.uint8)
        for r1, r2 in self.mask_rects:
            x1, y1 = r1
            x2, y2 = r2
            rect = QRect(x1, y1, x2 - x1, y2 - y1).normalized()
            mask[rect.top():rect.bottom(), rect.left():rect.right()] = 1
        # Remove letterbox and scale mask to original image
        y_offset, x_offset = self.offset[1], self.offset[0]
        img_mask = mask[y_offset:y_offset + int(self.base_img.shape[0]), x_offset:x_offset + int(self.base_img.shape[1])]
        img_mask_resized = cv2.resize(img_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        return img_mask_resized

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ASTM A247 Graphite Analysis')
        self.px_per_micron = None
        self.image = None
        self.cv_img = None
        self.calib_popup = None
        self.mask_widget = None
        self.user_mask = None
        self.init_ui()
        self.display_w = 800
        self.display_h = 600

    def init_ui(self):
        layout = QVBoxLayout()
        self.img_label = QLabel('Load an image to start.', self)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(self.display_w, self.display_h)
        layout.addWidget(self.img_label)

        btn_load = QPushButton('Load Image', self)
        btn_load.clicked.connect(self.load_image)
        layout.addWidget(btn_load)

        btn_calib = QPushButton('Calibrate px/mm', self)
        btn_calib.clicked.connect(self.calibrate)
        layout.addWidget(btn_calib)

        btn_mask = QPushButton('Add Mask Area', self)
        btn_mask.clicked.connect(self.add_mask)
        layout.addWidget(btn_mask)

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
            self.user_mask = None
            self.mask_widget = None

    def show_image(self, img, overlay=None, mask=None):
        disp_img, scale, offset = self.fit_image_to_window(img, self.display_w, self.display_h)
        if overlay is not None:
            overlay_resized = cv2.resize(overlay, (disp_img.shape[1], disp_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            disp_img = cv2.addWeighted(disp_img, 0.8, overlay_resized, 0.5, 0)
        if mask is not None:
            mask_disp = cv2.resize(mask.astype(np.uint8) * 255, (disp_img.shape[1], disp_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            disp_img[mask_disp > 0] = (disp_img[mask_disp > 0] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
        h, w = disp_img.shape[:2]
        qimg = QImage(disp_img.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.img_label.setPixmap(pix)

    def fit_image_to_window(self, img, maxw, maxh):
        h, w = img.shape[:2]
        scale = min(maxw / w, maxh / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        out_img = np.zeros((maxh, maxw, 3), dtype=np.uint8)
        y_offset = (maxh - new_h) // 2
        x_offset = (maxw - new_w) // 2
        out_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        return out_img, scale, (x_offset, y_offset)

    def calibrate(self):
        if self.cv_img is None:
            QMessageBox.warning(self, 'No Image', 'Please load an image first.')
            return
        self.calib_popup = CalibrationPopup(self.cv_img, parent=self)
        self.calib_popup.show()

    def set_px_per_micron(self, val):
        self.px_per_micron = val
        QMessageBox.information(self, 'Calibration', f'Calibration set: {val:.4f} px/micron')

    def add_mask(self):
        if self.cv_img is None:
            QMessageBox.warning(self, 'No Image', 'Please load an image first.')
            return
        disp_img, scale, offset = self.fit_image_to_window(self.cv_img, self.display_w, self.display_h)
        self.mask_widget = MaskWidget(self.cv_img, scale, offset, self.display_w, self.display_h, parent=self)
        self.mask_widget.setWindowTitle('Draw mask rectangles. Close when done.')
        self.mask_widget.show()
        self.mask_widget.destroyed.connect(self.get_mask_from_widget)

    def get_mask_from_widget(self):
        if self.mask_widget:
            self.user_mask = self.mask_widget.get_mask(self.cv_img.shape)
        self.show_image(self.cv_img, mask=self.user_mask)

    def analyze_graphite(self):
        if self.cv_img is None or self.px_per_micron is None:
            QMessageBox.warning(self, 'Missing Data', 'Load and calibrate image first.')
            return
        mask = self.user_mask
        gray = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.medianBlur(binary, 5)
        if mask is not None:
            if mask.shape != binary.shape:
                mask = cv2.resize(mask.astype(np.uint8), (binary.shape[1], binary.shape[0]), interpolation=cv2.INTER_NEAREST)
            binary[mask > 0] = 0
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colors = [
            (255, 0, 0), (0, 128, 255), (0, 255, 255), (0, 255, 0),
            (255, 255, 0), (255, 0, 255), (128, 0, 255), (0, 0, 255),
        ]
        size_bins = [(640, float('inf')), (320, 640), (160, 320), (80, 160), (40, 80), (20, 40), (10, 20), (0, 10)]
        size_classes = [0] * 8
        overlay = np.zeros_like(self.cv_img)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 5:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            diameter = radius * 2 / self.px_per_micron
            class_idx = None
            for i, (low, high) in enumerate(size_bins):
                if low <= diameter < high:
                    size_classes[i] += 1
                    class_idx = i
                    break
            if class_idx is not None:
                cv2.drawContours(overlay, [c], -1, colors[class_idx], thickness=cv2.FILLED)
        self.show_image(self.cv_img, overlay, mask=self.user_mask)
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
