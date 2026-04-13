"""
Error Handling Helpers
======================
Reusable error presentation: modal dialogs for critical errors,
inline banner widgets for minor/recoverable issues.
"""

from PySide6.QtWidgets import QMessageBox, QLabel, QHBoxLayout, QFrame, QPushButton
from PySide6.QtCore import Qt

from . import theme


def show_error_dialog(parent, title, message, details=None):
    """Show a modal error dialog for critical errors."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    if details:
        box.setDetailedText(details)
    box.exec()


def show_warning_dialog(parent, title, message):
    """Show a modal warning dialog."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(title)
    box.setText(message)
    box.exec()


class InlineErrorBanner(QFrame):
    """
    Dismissible red banner for minor/recoverable errors.
    Place at the top of a page layout.
    """

    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background-color: #3d1a1a; border: 1px solid {theme.RED_ERROR}; "
            f"border-radius: {theme.RADIUS_SMALL}; padding: 4px 8px;"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        label = QLabel(message)
        label.setStyleSheet(f"color: {theme.RED_ERROR}; background: transparent;")
        label.setWordWrap(True)
        layout.addWidget(label, 1)

        dismiss = QPushButton("x")
        dismiss.setFixedSize(24, 24)
        dismiss.setStyleSheet(
            f"background: transparent; color: {theme.RED_ERROR}; "
            f"border: none; font-weight: bold; font-size: 14px;"
        )
        dismiss.clicked.connect(self._dismiss)
        layout.addWidget(dismiss)

    def _dismiss(self):
        self.setVisible(False)
        self.deleteLater()
