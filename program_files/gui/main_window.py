"""
Main Window Module
==================
Top-level QMainWindow: header bar, sidebar wizard steps,
stacked content area, collapsible log panel.
"""

import logging
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QStackedWidget, QPlainTextEdit,
    QDockWidget, QPushButton, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QFont

from . import theme

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log handler that routes Python logging -> QPlainTextEdit
# ---------------------------------------------------------------------------

class QtLogHandler(logging.Handler):
    """Logging handler that emits records to a signal for thread-safe GUI updates."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record):
        try:
            msg = self.format(record)
            self._callback(msg)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Wizard step definitions
# ---------------------------------------------------------------------------

STEPS = [
    {"name": "Setup",    "index": 0},
    {"name": "DOE",      "index": 1},
    {"name": "Simulate", "index": 2},
    {"name": "Train",    "index": 3},
    {"name": "Validate", "index": 4},
]


# ---------------------------------------------------------------------------
# Fluent status indicator
# ---------------------------------------------------------------------------

class FluentStatusWidget(QLabel):
    """Small label in the header showing Fluent connection state."""

    _COLORS = {
        "Disconnected": theme.TEXT_DISABLED,
        "Launching":    theme.YELLOW_WARNING,
        "Connected":    theme.GREEN_SUCCESS,
        "Busy":         theme.ORANGE_LIGHT,
    }

    def __init__(self, parent=None):
        super().__init__("Disconnected", parent)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.set_status("Disconnected")

    def set_status(self, status):
        color = self._COLORS.get(status, theme.TEXT_DISABLED)
        self.setText(f"Fluent: {status}")
        self.setStyleSheet(f"color: {color}; font-weight: bold; background: transparent;")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Application main window."""

    step_changed = Signal(int)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.project = None
        self._unlocked = {0}  # Setup always unlocked

        self.setWindowTitle("Fluent PODNN Surrogate Builder")
        self.setMinimumSize(1200, 800)

        self._build_header()
        self._build_sidebar()
        self._build_content_area()
        self._build_log_panel()
        self._assemble_layout()

        self._install_log_handler()

    # --- Header ---

    def _build_header(self):
        self._header = QFrame()
        self._header.setFixedHeight(52)
        self._header.setProperty("panel", True)

        layout = QHBoxLayout(self._header)
        layout.setContentsMargins(16, 0, 16, 0)

        self._project_label = QLabel("No Project")
        font = self._project_label.font()
        font.setPointSize(14)
        font.setBold(True)
        self._project_label.setFont(font)

        self._settings_btn = QPushButton("Settings")
        self._settings_btn.setProperty("flat", True)
        self._settings_btn.setFixedSize(100, 32)

        self._fluent_status = FluentStatusWidget()

        layout.addWidget(self._project_label)
        layout.addStretch()
        layout.addWidget(self._fluent_status)
        layout.addSpacing(16)
        layout.addWidget(self._settings_btn)

    # --- Sidebar ---

    def _build_sidebar(self):
        self._sidebar = QListWidget()
        self._sidebar.setObjectName("sidebar")
        self._sidebar.setFixedWidth(180)
        self._current_step = 0

        for step in STEPS:
            item = QListWidgetItem(f"  {step['name']}")
            item.setData(Qt.UserRole, step["index"])
            item.setSizeHint(QSize(180, 48))
            self._sidebar.addItem(item)

        self._sidebar.setCurrentRow(0)
        self._sidebar.itemClicked.connect(self._on_sidebar_click)
        self._update_sidebar_styles()

    # --- Content area ---

    def _build_content_area(self):
        self._stack = QStackedWidget()

    # --- Log panel ---

    def _build_log_panel(self):
        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(5000)

        self._log_dock = QDockWidget("Log")
        self._log_dock.setWidget(self._log_text)
        self._log_dock.setFeatures(QDockWidget.DockWidgetClosable)
        self._log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)

    # --- Assemble ---

    def _assemble_layout(self):
        # Central widget: sidebar | content
        central = QWidget()
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)
        h_layout.addWidget(self._sidebar)
        h_layout.addWidget(self._stack, 1)

        # Wrap header + body
        wrapper = QWidget()
        v_layout = QVBoxLayout(wrapper)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(0)
        v_layout.addWidget(self._header)
        body = QWidget()
        body.setLayout(h_layout)
        v_layout.addWidget(body, 1)

        self.setCentralWidget(wrapper)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)
        # Collapsed by default
        self._log_dock.hide()

    # --- Logging ---

    def _install_log_handler(self):
        self._log_handler = QtLogHandler(self._append_log)
        self._log_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self._log_handler)

    def _append_log(self, msg):
        self._log_text.appendPlainText(msg)

    # --- Public API ---

    def set_project(self, project):
        """Set the active project and update header."""
        self.project = project
        name = project.info.get('project_name', 'Untitled') if project else 'No Project'
        self._project_label.setText(name)

        if project:
            self._refresh_unlocked_steps()

    def add_page(self, widget):
        """Append a page to the stack. Pages must be added in order (0, 1, 2, ...)."""
        self._stack.addWidget(widget)

    def unlock_step(self, index):
        """Unlock a wizard step in the sidebar."""
        self._unlocked.add(index)
        self._update_sidebar_styles()

    def lock_step(self, index):
        """Lock a wizard step."""
        self._unlocked.discard(index)
        self._update_sidebar_styles()

    def go_to_step(self, index):
        """Programmatically switch to a step (if unlocked)."""
        if index in self._unlocked:
            self._current_step = index
            self._sidebar.setCurrentRow(index)
            self._stack.setCurrentIndex(index)
            self._update_sidebar_styles()
            self.step_changed.emit(index)

    def toggle_log_panel(self):
        """Show/hide the log panel."""
        self._log_dock.setVisible(not self._log_dock.isVisible())

    def get_settings_button(self):
        """Return the settings button so the app can connect it."""
        return self._settings_btn

    def get_fluent_status_widget(self):
        """Return the Fluent status indicator."""
        return self._fluent_status

    # --- Refresh step lock state from project ---

    def _refresh_unlocked_steps(self):
        """Scan project state and unlock steps accordingly."""
        if not self.project:
            return

        state = self.project.get_project_state()

        # Setup always unlocked
        self.unlock_step(0)

        # DOE: unlocked when inputs + outputs configured
        if state['has_inputs'] and state['has_outputs']:
            self.unlock_step(1)
        else:
            self.lock_step(1)

        # Simulate: unlocked when DOE has points
        if state['has_doe']:
            self.unlock_step(2)
        else:
            self.lock_step(2)

        # Train: unlocked when sims exist
        if state['has_simulations']:
            self.unlock_step(3)
        else:
            self.lock_step(3)

        # Validate: unlocked when models exist
        if state['has_models']:
            self.unlock_step(4)
        else:
            self.lock_step(4)

    # --- Sidebar visual state ---

    def _update_sidebar_styles(self):
        """Update sidebar item text and colors based on locked/unlocked/active state."""
        from PySide6.QtGui import QColor, QBrush
        step_names = [s['name'] for s in STEPS]
        for i in range(self._sidebar.count()):
            item = self._sidebar.item(i)
            name = step_names[i]
            if i == self._current_step:
                item.setText(f"  {name}")
                item.setForeground(QBrush(QColor(theme.TEXT_PRIMARY)))
            elif i in self._unlocked:
                item.setText(f"  {name}")
                item.setForeground(QBrush(QColor(theme.ORANGE_LIGHT)))
            else:
                item.setText(f"  {name}")
                item.setForeground(QBrush(QColor(theme.TEXT_DISABLED)))

    # --- Slots ---

    def _on_sidebar_click(self, item):
        row = self._sidebar.row(item)
        if row < 0 or row not in self._unlocked:
            # Revert selection to current step
            self._sidebar.setCurrentRow(self._current_step)
            return
        self._current_step = row
        self._sidebar.setCurrentRow(row)
        self._stack.setCurrentIndex(row)
        self._update_sidebar_styles()
        self.step_changed.emit(row)
