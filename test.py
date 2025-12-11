import sys
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QTextEdit, QFileDialog, QHBoxLayout, QVBoxLayout,
    QWidget, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer, QRect


# -----------------------------------
#  Custom QLabel for Drawing ROI
# -----------------------------------
class DrawArea(QLabel):
    """QLabel that allows drawing rectangular regions on an image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self._pixmap_ref = None
        self._start_point = None
        self._end_point = None
        self._is_drawing = False
        self.selected_roi = None  # (x, y, w, h)

    def setPixmap(self, pixmap: QPixmap):
        super().setPixmap(pixmap)
        self._pixmap_ref = pixmap

    # MOUSE EVENTS
    def mousePressEvent(self, event):
        if not self._pixmap_ref:
            return
        if event.button() == Qt.LeftButton:
            self._is_drawing = True
            self._start_point = event.pos()
            self._end_point = self._start_point
            self.update()

    def mouseMoveEvent(self, event):
        if self._is_drawing:
            self._end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_drawing:
            self._is_drawing = False
            self._end_point = event.pos()
            self._calculate_roi()
            self.update()

    # DRAW RECTANGLES
    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 255, 0), 2))

        # Live drawing
        if self._is_drawing and self._start_point and self._end_point:
            rect = QRect(self._start_point, self._end_point).normalized()
            painter.drawRect(rect)

        # Saved ROI
        if self.selected_roi:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            x, y, w, h = self._convert_image_to_widget(self.selected_roi)
            painter.drawRect(QRect(int(x), int(y), int(w), int(h)))

    # ROI CALCULATION + COORDINATE CONVERSIONS
    def _calculate_roi(self):
        if not self._pixmap_ref:
            return
        x1, y1 = self._start_point.x(), self._start_point.y()
        x2, y2 = self._end_point.x(), self._end_point.y()

        rx, ry, rw, rh = QRect(x1, y1, x2 - x1, y2 - y1).normalized().getRect()
        self.selected_roi = self._convert_widget_to_image(rx, ry, rw, rh)

    def _convert_widget_to_image(self, x, y, w, h):
        label_w, label_h = self.width(), self.height()
        img_w, img_h = self._pixmap_ref.width(), self._pixmap_ref.height()

        scale = max(img_w / label_w, img_h / label_h)

        ix, iy = int(x * scale), int(y * scale)
        iw, ih = int(w * scale), int(h * scale)

        return (
            max(0, ix),
            max(0, iy),
            max(0, min(iw, img_w - ix)),
            max(0, min(ih, img_h - iy)),
        )

    def _convert_image_to_widget(self, rect):
        x, y, w, h = rect
        img_w, img_h = self._pixmap_ref.width(), self._pixmap_ref.height()
        label_w, label_h = self.width(), self.height()

        scale = 1.0 / max(img_w / label_w, img_h / label_h)
        return x * scale, y * scale, w * scale, h * scale


# -----------------------------------
#          MAIN OCR WINDOW
# -----------------------------------
class TextScanner(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Extractor (PyQt5 + OCR)")

        self.raw_image = None
        self.rendered_image = None
        self.camera_stream = None

        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._refresh_camera_frame)

        # IMAGE AREA
        self.viewer = DrawArea()
        self.viewer.setFixedSize(640, 480)

        # BUTTONS
        btn_load = QPushButton("Open Image")
        btn_load.clicked.connect(self.load_image_file)

        btn_cam_on = QPushButton("Start Camera")
        btn_cam_on.clicked.connect(self.activate_camera)

        btn_cam_off = QPushButton("Stop Camera")
        btn_cam_off.clicked.connect(self.deactivate_camera)

        btn_snap = QPushButton("Capture")
        btn_snap.clicked.connect(self.capture_still_frame)

        btn_ocr = QPushButton("Extract Text")
        btn_ocr.clicked.connect(self.perform_ocr)

        btn_reset = QPushButton("Reset ROI")
        btn_reset.clicked.connect(self.clear_selection)

        btn_export = QPushButton("Save Result")
        btn_export.clicked.connect(self.save_output_frame)

        # TEXT OUTPUT
        self.console = QTextEdit()
        self.console.setPlaceholderText("Recognized text will appear here...")

        # *** RENAMED LAYOUT VARIABLES ***
        button_bar = QHBoxLayout()
        for item in [btn_load, btn_cam_on, btn_cam_off, btn_snap, btn_ocr, btn_reset, btn_export]:
            button_bar.addWidget(item)

        left_panel = QVBoxLayout()
        left_panel.addWidget(self.viewer)
        left_panel.addLayout(button_bar)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel)
        main_layout.addWidget(self.console)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # IMAGE LOADING
    def load_image_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp)")
        if not path:
            return

        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Error", "Unable to load image.")
            return

        self.raw_image = image
        self._display_frame(image)

    # CAMERA METHODS
    def activate_camera(self):
        self.camera_stream = cv2.VideoCapture(0)
        if not self.camera_stream.isOpened():
            QMessageBox.warning(self, "Camera", "Camera unavailable.")
            return
        self.frame_timer.start(30)

    def deactivate_camera(self):
        if self.frame_timer.isActive():
            self.frame_timer.stop()
        if self.camera_stream:
            self.camera_stream.release()
            self.camera_stream = None

    def _refresh_camera_frame(self):
        if not self.camera_stream:
            return
        ok, frame = self.camera_stream.read()
        if ok:
            self.raw_image = frame.copy()
            self._display_frame(self.raw_image)

    def capture_still_frame(self):
        if self.raw_image is None:
            QMessageBox.information(self, "Info", "No frame available.")
            return
        QMessageBox.information(self, "Captured", "Image captured.")

    # IMAGE DISPLAY
    def _display_frame(self, frame, boxes=None, words=None):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = rgb.copy()

        if boxes:
            for (x, y, w, h), text in zip(boxes, words):
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, text, (x, max(10, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.rendered_image = output
        h, w, ch = output.shape

        qt_img = QImage(output.data, w, h, ch * w, QImage.Format_RGB888)

        scaled_pix = QPixmap.fromImage(qt_img).scaled(
            self.viewer.width(), self.viewer.height(), Qt.KeepAspectRatio
        )
        self.viewer.setPixmap(scaled_pix)

    # OCR ENGINE
    def perform_ocr(self):
        if self.raw_image is None:
            QMessageBox.information(self, "Info", "Provide an image first.")
            return

        roi = self.viewer.selected_roi
        target = (
            self.raw_image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
            if roi
            else self.raw_image
        )

        if target.size == 0:
            QMessageBox.warning(self, "Error", "Invalid ROI.")
            return

        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
        )

        pil_img = Image.fromarray(thresh)
        config = "--oem 3 --psm 6"

        try:
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=config)
            full_text = pytesseract.image_to_string(pil_img, config=config)
        except Exception as e:
            QMessageBox.critical(self, "OCR Error", str(e))
            return

        boxes, word_list = [], []
        for i in range(len(data["text"])):
            try:
                conf = int(data["conf"][i])
            except:
                conf = -1

            if conf > 40:
                word = data["text"][i].strip()
                if word:
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]
                    boxes.append((x, y, w, h))
                    word_list.append(word)

        if roi:
            ox, oy, _, _ = roi
            boxes = [(x + ox, y + oy, w, h) for (x, y, w, h) in boxes]

        display_copy = self.raw_image.copy()
        self._display_frame(display_copy, boxes, word_list)

        self.console.setPlainText(full_text)

    # UTILITY BUTTONS
    def clear_selection(self):
        self.viewer.selected_roi = None
        self.viewer.update()

    def save_output_frame(self):
        if self.rendered_image is None:
            QMessageBox.information(self, "Info", "Nothing to save.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png)")
        if not path:
            return

        bgr = cv2.cvtColor(self.rendered_image, cv2.COLOR_RGB2BGR)
        _, buff = cv2.imencode(".png", bgr)
        buff.tofile(path)

        QMessageBox.information(self, "Saved", f"Saved to:\n{path}")


# -----------------------------------
#  RUN APPLICATION
# -----------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TextScanner()
    win.show()
    sys.exit(app.exec_())
