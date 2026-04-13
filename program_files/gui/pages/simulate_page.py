"""
Simulate Page Module
====================
Run/Continue button, reinitialization checkbox, progress bar,
per-sim status table, stop button. Uses SimulationWorker on QThread.
"""

import json
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QFrame,
)
from PySide6.QtCore import Qt, Signal

from .. import theme
from ..fluent_manager import FluentManager
from ..dataset_manager import DatasetManager
from ..workers import SimulationWorker
from ..spinner import SpinnerWidget

logger = logging.getLogger(__name__)


class SimulatePage(QWidget):
    """Batch simulation page with progress tracking."""

    simulations_changed = Signal()  # emitted when sims complete

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.dm = DatasetManager(project.dataset_dir)
        self._worker = None

        self._build_ui()
        self._refresh_status()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Simulate")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        layout.addSpacing(8)

        # Controls row
        ctrl = QHBoxLayout()

        self._run_btn = QPushButton("Run")
        self._run_btn.setFixedWidth(120)
        self._run_btn.clicked.connect(self._start_simulations)
        ctrl.addWidget(self._run_btn)

        self._reinit_cb = QCheckBox("Reinitialize between runs")
        self._reinit_cb.setChecked(True)
        ctrl.addWidget(self._reinit_cb)

        ctrl.addStretch()

        self._stop_btn = QPushButton("Stop after current")
        self._stop_btn.setProperty("flat", True)
        self._stop_btn.setFixedWidth(160)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._request_stop)
        ctrl.addWidget(self._stop_btn)

        layout.addLayout(ctrl)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        layout.addWidget(self._progress)

        # Status summary with inline spinner
        status_row = QHBoxLayout()
        self._spinner = SpinnerWidget(20)
        status_row.addWidget(self._spinner)
        self._status_label = QLabel("")
        self._status_label.setProperty("secondary", True)
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        layout.addLayout(status_row)

        # Per-sim status table
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Sample ID", "Status", "Details"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

    # --- Status ---

    def _refresh_status(self):
        completed = self.dm.get_completed_ids()
        total = self._get_total_doe_points()

        if total == 0:
            self._status_label.setText("No DOE points configured.")
            self._run_btn.setEnabled(False)
            self._progress.setValue(0)
            return

        self._progress.setMaximum(total)
        self._progress.setValue(len(completed))
        self._progress.setFormat(f"{len(completed)}/{total}")

        remaining = total - len(completed)
        if remaining == 0:
            self._status_label.setText("All samples complete.")
            self._run_btn.setText("Run")
            self._run_btn.setEnabled(True)
        elif len(completed) > 0:
            self._status_label.setText(f"{len(completed)}/{total} complete, {remaining} remaining.")
            self._run_btn.setText("Continue")
            self._run_btn.setEnabled(True)
        else:
            self._status_label.setText(f"0/{total} complete. Ready to run.")
            self._run_btn.setText("Run")
            self._run_btn.setEnabled(True)

        # Populate table with current state
        self._table.setRowCount(total)
        for i in range(total):
            sim_id = i + 1
            self._table.setItem(i, 0, QTableWidgetItem(f"Sample {sim_id}"))
            if sim_id in completed:
                status_item = QTableWidgetItem("Done")
                status_item.setForeground(Qt.green)
            else:
                status_item = QTableWidgetItem("Queued")
                status_item.setForeground(Qt.gray)
            self._table.setItem(i, 1, status_item)
            self._table.setItem(i, 2, QTableWidgetItem(""))

    def _get_total_doe_points(self):
        from ...modules.doe_setup import load_doe_samples
        samples, _ = load_doe_samples(self.project.doe_samples_file)
        return len(samples)

    # --- Run ---

    def _start_simulations(self):
        fm = FluentManager.instance()
        if not fm.is_available():
            QMessageBox.warning(self, "Not Connected", "Fluent is not connected. Go to Setup and launch Fluent.")
            return

        if not self.project.model_setup_file.exists():
            QMessageBox.warning(self, "No Setup", "Complete Setup before running simulations.")
            return

        try:
            with open(self.project.model_setup_file, 'r') as f:
                setup_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read model_setup.json: {e}")
            return

        # Load DOE samples
        from ...modules.doe_setup import load_doe_samples
        doe_samples, _ = load_doe_samples(self.project.doe_samples_file)
        if not doe_samples:
            QMessageBox.warning(self, "No DOE", "No DOE samples found. Generate samples in the DOE step first.")
            return

        fm.set_busy()
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._spinner.start()

        self._worker = SimulationWorker(
            solver=fm.solver,
            setup_data=setup_data,
            dataset_dir=self.project.dataset_dir,
            reinitialize=self._reinit_cb.isChecked(),
            doe_samples=doe_samples,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _request_stop(self):
        if self._worker:
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._stop_btn.setText("Stopping...")

    # --- Callbacks ---

    def _on_progress(self, idx, total, sim_id, status):
        self._progress.setMaximum(total)
        self._progress.setValue(idx)
        self._progress.setFormat(f"{idx}/{total}")

        row = sim_id - 1
        if 0 <= row < self._table.rowCount():
            status_item = QTableWidgetItem(status.capitalize())
            if status == 'done':
                status_item.setForeground(Qt.green)
                # Unlock Train step as soon as first sim completes
                self.simulations_changed.emit()
            elif status == 'failed':
                status_item.setForeground(Qt.red)
            elif status == 'running':
                status_item.setForeground(Qt.yellow)
            self._table.setItem(row, 1, status_item)

        self._status_label.setText(f"Running sample {sim_id} ({idx}/{total})")

    def _on_finished(self, summary):
        fm = FluentManager.instance()
        fm.set_idle()
        self._spinner.stop()
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setText("Stop after current")
        self._worker = None

        ok = summary.get('successful', 0)
        fail = summary.get('failed', 0)

        # Only bump dataset version if new sims were completed
        if ok > 0:
            self.dm.bump_version()
            self.simulations_changed.emit()

        self._refresh_status()
        logger.info(f"Batch complete: {ok} successful, {fail} failed")

    def _on_error(self, msg):
        fm = FluentManager.instance()
        fm.set_idle()
        self._spinner.stop()
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setText("Stop after current")
        self._worker = None

        QMessageBox.critical(self, "Simulation Error", msg)
        self._refresh_status()
