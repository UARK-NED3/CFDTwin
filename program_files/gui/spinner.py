"""
Spinner Widget
==============
Animated loading spinner for blocking operations.
"""

import math
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QConicalGradient

from . import theme


class SpinnerWidget(QWidget):
    """Animated circular loading spinner with optional status text."""

    def __init__(self, size=40, parent=None):
        super().__init__(parent)
        self._size = size
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self.setFixedSize(size, size)
        self.hide()

    def start(self):
        self.show()
        self._timer.start(16)  # ~60fps

    def stop(self):
        self._timer.stop()
        self.hide()

    def _rotate(self):
        self._angle = (self._angle + 5) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw arc
        pen_width = max(3, self._size // 10)
        margin = pen_width
        rect = self.rect().adjusted(margin, margin, -margin, -margin)

        # Background track
        track_pen = QPen(QColor(theme.BORDER), pen_width)
        track_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(rect, 0, 360 * 16)

        # Spinning arc
        arc_pen = QPen(QColor(theme.ORANGE_LIGHT), pen_width)
        arc_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(arc_pen)
        start = int(self._angle * 16)
        span = 90 * 16  # quarter circle
        painter.drawArc(rect, start, span)

        painter.end()


class LoadingOverlay(QWidget):
    """Spinner + status label, centered. Show over a parent widget during loading."""

    def __init__(self, text="Loading...", size=48, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self._spinner = SpinnerWidget(size, self)
        layout.addWidget(self._spinner, 0, Qt.AlignCenter)

        self._label = QLabel(text)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: 13px; background: transparent;"
        )
        layout.addWidget(self._label)

        self.setStyleSheet(f"background-color: rgba(26, 26, 26, 200); border-radius: {theme.RADIUS};")
        self.hide()

    def set_text(self, text):
        self._label.setText(text)

    def start(self, text=None):
        if text:
            self._label.setText(text)
        self._spinner.start()
        self.show()
        self.raise_()

    def stop(self):
        self._spinner.stop()
        self.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def showEvent(self, event):
        """Resize to fill parent."""
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().showEvent(event)
