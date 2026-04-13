"""
Project Dialog Module
=====================
Modal dialog shown on launch. Supports Create New, Open Existing,
and Recent Projects. Returns a WorkflowProject or None.
"""

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QListWidget, QListWidgetItem,
    QFrame, QMessageBox, QWidget,
)
from PySide6.QtCore import Qt, QSize

from . import theme
from ..modules.project_system import WorkflowProject, create_project, open_project

logger = logging.getLogger(__name__)


class ProjectDialog(QDialog):
    """
    Project selection dialog shown on app launch.

    Returns a WorkflowProject via self.project after accept(),
    or None if the dialog is rejected.
    """

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.project = None

        self.setWindowTitle("Select Project")
        self.setFixedSize(520, 420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._build_ui()
        self._populate_recent()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel("Fluent PODNN Surrogate Builder")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._new_btn = QPushButton("Create New")
        self._open_btn = QPushButton("Open Existing")
        self._new_btn.setFixedHeight(38)
        self._open_btn.setFixedHeight(38)

        btn_row.addWidget(self._new_btn)
        btn_row.addWidget(self._open_btn)
        layout.addLayout(btn_row)

        # --- Create new panel (hidden by default) ---
        self._create_panel = QFrame()
        self._create_panel.setProperty("panel", True)
        cp_layout = QVBoxLayout(self._create_panel)
        cp_layout.setContentsMargins(12, 12, 12, 12)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Project name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("My Surrogate Study")
        name_row.addWidget(self._name_edit)
        cp_layout.addLayout(name_row)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Location:"))
        self._folder_edit = QLineEdit()
        self._folder_edit.setReadOnly(True)
        self._folder_edit.setPlaceholderText("Choose folder...")
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.setProperty("flat", True)
        self._browse_btn.setFixedWidth(70)
        folder_row.addWidget(self._folder_edit, 1)
        folder_row.addWidget(self._browse_btn)
        cp_layout.addLayout(folder_row)

        self._create_confirm_btn = QPushButton("Create")
        self._create_confirm_btn.setEnabled(False)
        cp_layout.addWidget(self._create_confirm_btn)

        self._create_panel.hide()
        layout.addWidget(self._create_panel)

        # --- Recent projects ---
        recent_label = QLabel("Recent Projects")
        recent_label.setProperty("secondary", True)
        layout.addWidget(recent_label)

        self._recent_list = QListWidget()
        self._recent_list.setSpacing(0)
        self._recent_list.setStyleSheet(f"""
            QListWidget {{
                font-size: 14px;
            }}
            QListWidget::item {{
                color: {theme.TEXT_PRIMARY};
                padding: 10px 12px;
                border-bottom: 1px solid {theme.BORDER};
            }}
            QListWidget::item:hover {{
                background-color: {theme.BG_INPUT};
            }}
            QListWidget::item:selected {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {theme.ORANGE_DARK}, stop:0.6 transparent);
            }}
        """)
        layout.addWidget(self._recent_list, 1)

        # --- Connections ---
        self._new_btn.clicked.connect(self._toggle_create_panel)
        self._open_btn.clicked.connect(self._open_existing)
        self._browse_btn.clicked.connect(self._browse_folder)
        self._create_confirm_btn.clicked.connect(self._create_new)
        self._name_edit.textChanged.connect(self._validate_create)
        self._recent_list.itemDoubleClicked.connect(self._open_recent)

    # --- Recent projects ---

    def _populate_recent(self):
        self._recent_list.clear()
        recent = self.settings.get_recent_project_folders()
        if not recent:
            item = QListWidgetItem("No recent projects")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self._recent_list.addItem(item)
            return

        for path in recent:
            # Try to read project name from project_info.json
            info_file = Path(path) / "project_info.json"
            display = Path(path).name
            if info_file.exists():
                try:
                    import json
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    display = info.get('project_name', display)
                except Exception:
                    pass

            item = QListWidgetItem(f"{display}\n{path}")
            item.setData(Qt.UserRole, path)
            item.setSizeHint(QSize(0, 54))
            self._recent_list.addItem(item)

    # --- Create new ---

    def _toggle_create_panel(self):
        visible = self._create_panel.isVisible()
        self._create_panel.setVisible(not visible)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose Project Location")
        if folder:
            self._folder_edit.setText(folder)
            self._validate_create()

    def _validate_create(self):
        name_ok = len(self._name_edit.text().strip()) > 0
        folder_ok = len(self._folder_edit.text().strip()) > 0
        self._create_confirm_btn.setEnabled(name_ok and folder_ok)

    def _create_new(self):
        name = self._name_edit.text().strip()
        folder = Path(self._folder_edit.text().strip()) / name

        if folder.exists() and any(folder.iterdir()):
            reply = QMessageBox.question(
                self, "Folder Exists",
                f"{folder} already exists and is not empty. Use it anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        project = create_project(folder, name)
        if project is None:
            QMessageBox.critical(self, "Error", "Failed to create project.")
            return

        self.settings.add_recent_project_folder(str(folder))
        self.project = project
        logger.info(f"Created project: {name}")
        self.accept()

    # --- Open existing ---

    def _open_existing(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Project Folder")
        if not folder:
            return
        self._try_open(folder)

    def _open_recent(self, item):
        path = item.data(Qt.UserRole)
        if path:
            self._try_open(path)

    def _try_open(self, folder):
        project = open_project(folder)
        if project is None:
            QMessageBox.critical(
                self, "Error",
                f"Could not open project at:\n{folder}\n\nNo project_info.json found.",
            )
            return

        self.settings.add_recent_project_folder(str(folder))
        self.project = project
        logger.info(f"Opened project: {project.info.get('project_name', 'Unknown')}")
        self.accept()
