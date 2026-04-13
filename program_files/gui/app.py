"""
Application Entry Point
=======================
Launches the PySide6 application, shows project dialog, opens main window.
Wires up all pages, settings dialog, fluent manager, step unlocking.
"""

import sys
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .theme import get_stylesheet
from .main_window import MainWindow
from .project_dialog import ProjectDialog
from .settings_dialog import SettingsDialog
from .fluent_manager import FluentManager
from .pages.setup_page import SetupPage
from .pages.doe_page import DOEPage
from .pages.simulate_page import SimulatePage
from .pages.train_page import TrainPage
from .pages.validate_page import ValidatePage
from ..modules.user_settings import UserSettings

logger = logging.getLogger(__name__)

# Settings file lives next to the Python package (portable)
_SETTINGS_FILE = Path(__file__).resolve().parent.parent / "user_settings.json"


def run():
    """Launch the application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    app = QApplication(sys.argv)
    app.setStyleSheet(get_stylesheet())

    settings = UserSettings(_SETTINGS_FILE)

    # Show project dialog -- app exits if user closes without selecting
    dialog = ProjectDialog(settings)
    if dialog.exec() != ProjectDialog.Accepted:
        sys.exit(0)

    project = dialog.project

    # Launch main window
    window = MainWindow(settings)
    window.set_project(project)

    # --- Fluent manager -> header status ---
    fm = FluentManager.instance()
    fm.status_changed.connect(window.get_fluent_status_widget().set_status)

    # --- Pages ---
    setup_page = SetupPage(project, settings)
    doe_page = DOEPage(project)
    sim_page = SimulatePage(project)
    train_page = TrainPage(project, settings)
    validate_page = ValidatePage(project, settings)

    # Add pages in order (0=Setup, 1=DOE, 2=Simulate, 3=Train, 4=Validate)
    window.add_page(setup_page)
    window.add_page(doe_page)
    window.add_page(sim_page)
    window.add_page(train_page)
    window.add_page(validate_page)

    # --- Step unlocking signals ---
    def refresh_steps():
        window._refresh_unlocked_steps()

    setup_page.setup_complete.connect(refresh_steps)
    doe_page.doe_changed.connect(refresh_steps)
    sim_page.simulations_changed.connect(refresh_steps)
    training_complete_connected = train_page.training_complete.connect(refresh_steps)

    # Refresh data-dependent pages when switching to them
    def on_step_changed(idx):
        if idx == 1:  # DOE
            doe_page._load_state()
        elif idx == 2:  # Simulate
            sim_page._refresh_status()
        elif idx == 3:  # Train
            train_page.refresh()
        elif idx == 4:  # Validate
            validate_page.refresh()

    window.step_changed.connect(on_step_changed)

    # --- Settings dialog ---
    def open_settings():
        dlg = SettingsDialog(settings, window)
        dlg.exec()

    window.get_settings_button().clicked.connect(open_settings)

    # --- Cleanup on exit ---
    def on_close():
        fm.disconnect()

    app.aboutToQuit.connect(on_close)

    window.show()
    sys.exit(app.exec())
