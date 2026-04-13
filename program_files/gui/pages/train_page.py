"""
Train Page Module
=================
Two-column transfer list for sample selection, training config,
and loss curve display after training.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox,
    QListWidget, QListWidgetItem, QTabWidget,
    QMessageBox, QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal

from .. import theme
from ..dataset_manager import DatasetManager
from ..workers import TrainingWorker

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TrainPage(QWidget):
    """Model training page with transfer list filter, config, and loss curves."""

    training_complete = Signal()

    def __init__(self, project, settings, parent=None):
        super().__init__(parent)
        self.project = project
        self.settings = settings
        self.dm = DatasetManager(project.dataset_dir)
        self._worker = None
        self._train_queue = []
        self._train_results = []
        self._nn_settings = {}
        self._current_model_name = ""
        self._live_canvases = {}  # {model_name: (canvas, ax, train_line, val_line, epochs, train_losses, val_losses)}

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Train")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        layout.addSpacing(8)

        # --- Transfer list: All Samples <-> Training Samples ---
        transfer_row = QHBoxLayout()

        # Left: all samples
        left_frame = QFrame()
        left_frame.setProperty("panel", True)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(12, 12, 12, 12)
        self._all_label = QLabel("All Samples")
        self._all_label.setProperty("secondary", True)
        left_layout.addWidget(self._all_label)
        self._all_list = QListWidget()
        self._all_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left_layout.addWidget(self._all_list, 1)
        transfer_row.addWidget(left_frame, 1)

        # Middle: transfer buttons
        btn_col = QVBoxLayout()
        btn_col.addStretch()

        self._add_all_btn = QPushButton(">>")
        self._add_all_btn.setFixedSize(48, 32)
        self._add_all_btn.setToolTip("Move all to training")
        self._add_all_btn.clicked.connect(self._move_all_right)
        btn_col.addWidget(self._add_all_btn)

        self._add_sel_btn = QPushButton(">")
        self._add_sel_btn.setFixedSize(48, 32)
        self._add_sel_btn.setToolTip("Move selected to training")
        self._add_sel_btn.clicked.connect(self._move_selected_right)
        btn_col.addWidget(self._add_sel_btn)

        btn_col.addSpacing(8)

        self._rem_sel_btn = QPushButton("<")
        self._rem_sel_btn.setFixedSize(48, 32)
        self._rem_sel_btn.setToolTip("Remove selected from training")
        self._rem_sel_btn.clicked.connect(self._move_selected_left)
        btn_col.addWidget(self._rem_sel_btn)

        self._rem_all_btn = QPushButton("<<")
        self._rem_all_btn.setFixedSize(48, 32)
        self._rem_all_btn.setToolTip("Remove all from training")
        self._rem_all_btn.clicked.connect(self._move_all_left)
        btn_col.addWidget(self._rem_all_btn)

        btn_col.addStretch()
        transfer_row.addLayout(btn_col)

        # Right: training samples
        right_frame = QFrame()
        right_frame.setProperty("panel", True)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(12, 12, 12, 12)
        self._train_label = QLabel("Training Samples")
        self._train_label.setProperty("secondary", True)
        right_layout.addWidget(self._train_label)
        self._train_list = QListWidget()
        self._train_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        right_layout.addWidget(self._train_list, 1)
        transfer_row.addWidget(right_frame, 1)

        # Far right: output checkboxes
        outputs_frame = QFrame()
        outputs_frame.setProperty("panel", True)
        outputs_frame.setFixedWidth(220)
        outputs_layout = QVBoxLayout(outputs_frame)
        outputs_layout.setContentsMargins(12, 12, 12, 12)
        outputs_header = QLabel("Outputs to Train")
        outputs_header.setProperty("secondary", True)
        outputs_layout.addWidget(outputs_header)
        self._output_checks_layout = QVBoxLayout()
        self._output_checks_layout.setAlignment(Qt.AlignTop)
        outputs_layout.addLayout(self._output_checks_layout, 1)
        transfer_row.addWidget(outputs_frame)

        layout.addLayout(transfer_row, 1)
        layout.addSpacing(12)

        # --- Config row ---
        config_row = QHBoxLayout()

        config_row.addWidget(QLabel("Epochs:"))
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(10, 10000)
        self._epochs_spin.setValue(500)
        self._epochs_spin.setButtonSymbols(QSpinBox.NoButtons)
        config_row.addWidget(self._epochs_spin)

        config_row.addSpacing(16)
        config_row.addWidget(QLabel("Test split:"))
        self._split_spin = QDoubleSpinBox()
        self._split_spin.setRange(0.0, 0.5)
        self._split_spin.setSingleStep(0.05)
        self._split_spin.setValue(0.2)
        self._split_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        config_row.addWidget(self._split_spin)

        config_row.addSpacing(16)
        self._train_btn = QPushButton("Train")
        self._train_btn.setFixedWidth(100)
        self._train_btn.clicked.connect(self._start_training)
        config_row.addWidget(self._train_btn)

        layout.addLayout(config_row)
        layout.addSpacing(8)

        # --- Loss curve tabs ---
        self._loss_tabs = QTabWidget()
        self._loss_tabs.setMinimumHeight(200)
        layout.addWidget(self._loss_tabs, 1)

    # --- Transfer list operations ---

    def _move_all_right(self):
        while self._all_list.count():
            item = self._all_list.takeItem(0)
            self._train_list.addItem(item)
        self._update_counts()

    def _move_selected_right(self):
        for item in self._all_list.selectedItems():
            row = self._all_list.row(item)
            taken = self._all_list.takeItem(row)
            self._train_list.addItem(taken)
        self._update_counts()

    def _move_selected_left(self):
        for item in self._train_list.selectedItems():
            row = self._train_list.row(item)
            taken = self._train_list.takeItem(row)
            self._all_list.addItem(taken)
        self._sort_list(self._all_list)
        self._update_counts()

    def _move_all_left(self):
        while self._train_list.count():
            item = self._train_list.takeItem(0)
            self._all_list.addItem(item)
        self._sort_list(self._all_list)
        self._update_counts()

    def _sort_list(self, list_widget):
        """Sort list items by sample ID."""
        items = []
        while list_widget.count():
            items.append(list_widget.takeItem(0))
        items.sort(key=lambda it: it.data(Qt.UserRole))
        for item in items:
            list_widget.addItem(item)

    def _update_counts(self):
        total = self._all_list.count() + self._train_list.count()
        train = self._train_list.count()
        self._all_label.setText(f"All Samples ({self._all_list.count()})")
        self._train_label.setText(f"Training Samples ({train}/{total})")

    def _get_training_ids(self):
        """Return list of sim IDs in the training list."""
        ids = []
        for i in range(self._train_list.count()):
            sid = self._train_list.item(i).data(Qt.UserRole)
            if sid is not None:
                ids.append(sid)
        return ids

    # --- Refresh ---

    def refresh(self):
        """Reload sample lists from current dataset."""
        completed = self.dm.get_completed_ids()

        # Remember what was in the training list
        existing_train = set(self._get_training_ids())

        self._all_list.clear()
        self._train_list.clear()

        for sid in completed:
            item = QListWidgetItem(f"Sample {sid}")
            item.setData(Qt.UserRole, sid)
            if existing_train and sid in existing_train:
                self._train_list.addItem(item)
            elif not existing_train:
                # First load: all samples go to training by default
                self._train_list.addItem(item)
            else:
                self._all_list.addItem(item)

        self._update_counts()

        # Populate output checkboxes and auto-named textboxes
        self._refresh_output_checks()

    # --- Output checkboxes ---

    def _refresh_output_checks(self):
        """Populate output checkboxes and auto-named textboxes from output_parameters.json."""
        # Clear existing
        while self._output_checks_layout.count():
            child = self._output_checks_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                # Recursively clear sub-layouts
                sub = child.layout()
                while sub.count():
                    sc = sub.takeAt(0)
                    if sc.widget():
                        sc.widget().deleteLater()

        self._output_cbs = {}   # name -> checkbox
        self._output_names = {}  # name -> line edit

        if not self.project.output_parameters_file.exists():
            return

        version = self.dm.get_version()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        try:
            with open(self.project.output_parameters_file, 'r') as f:
                out_data = json.load(f)

            for out in out_data.get('outputs', []):
                name = out['name']
                category = out.get('category', '')
                fields = out.get('field_variables', [])

                # Determine dimension label
                if category == 'Report Definition':
                    dim = '1D'
                elif category == 'Surface':
                    dim = '2D'
                elif category == 'Cell Zone':
                    dim = '3D'
                else:
                    dim = '2D'

                # Sanitize name for filesystem
                safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
                auto_name = f"{dim}_{safe_name}_v{version}_{timestamp}"

                cb = QCheckBox(name)
                cb.setChecked(True)
                self._output_checks_layout.addWidget(cb)

                name_edit = QLineEdit(auto_name)
                name_edit.setStyleSheet("font-size: 11px;")
                self._output_checks_layout.addWidget(name_edit)

                # Small spacer between rows
                self._output_checks_layout.addSpacing(6)

                self._output_cbs[name] = cb
                self._output_names[name] = name_edit
        except Exception as e:
            logger.warning(f"Error loading outputs for training: {e}")

    def _get_selected_outputs(self):
        """Return dict of {output_name: model_name} for checked outputs."""
        result = {}
        for name, cb in self._output_cbs.items():
            if cb.isChecked():
                model_name = self._output_names[name].text().strip()
                if model_name:
                    result[name] = model_name
        return result

    # --- Train ---

    def _start_training(self):
        training_ids = self._get_training_ids()
        if not training_ids:
            QMessageBox.warning(self, "No Data", "Move samples to the Training list first.")
            return

        if not self.project.model_setup_file.exists():
            QMessageBox.warning(self, "No Setup", "Complete Setup first.")
            return

        # Get {output_name: model_name} for checked outputs
        selected_outputs = self._get_selected_outputs()
        if not selected_outputs:
            QMessageBox.warning(self, "No Outputs", "Check at least one output to train.")
            return

        # Validate no duplicate folder names
        for out_name, model_name in selected_outputs.items():
            model_dir = self.project.models_dir / model_name
            if model_dir.exists():
                QMessageBox.warning(self, "Name Exists",
                                    f"Model '{model_name}' already exists. Choose a different name.")
                return

        nn_settings = self.settings.get_nn_settings()

        self._train_btn.setEnabled(False)

        # Clear any existing live canvases from a previous run
        self._live_canvases.clear()
        self._loss_tabs.clear()

        # Queue: each output becomes its own training run
        self._train_queue = list(selected_outputs.items())
        self._train_results = []
        self._run_next_training(nn_settings)

    def _run_next_training(self, nn_settings):
        """Start training the next output in the queue."""
        if not self._train_queue:
            self._on_all_training_done()
            return

        self._nn_settings = nn_settings
        output_name, model_name = self._train_queue.pop(0)
        self._current_model_name = model_name

        # Create a live canvas tab for this model
        self._create_live_canvas(model_name)

        self._worker = TrainingWorker(
            project_dir=self.project.project_path,
            model_name=model_name,
            model_selection=nn_settings,
            test_size=self._split_spin.value(),
            epochs=self._epochs_spin.value(),
            output_filter=[output_name],
        )
        self._worker.model_started.connect(self._on_model_started)
        self._worker.epoch_update.connect(self._on_epoch_update)
        self._worker.finished.connect(self._on_one_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_one_finished(self, summary):
        """Called when a single output's training completes — run the next."""
        if summary:
            self._train_results.append(summary)
        self._worker = None
        self._run_next_training(self._nn_settings)

    def _on_all_training_done(self):
        """All queued trainings have completed."""
        self._train_btn.setEnabled(True)
        self.training_complete.emit()
        total_models = sum(r.get('n_models', 0) for r in self._train_results)
        logger.info(f"Training complete: {total_models} model(s) across {len(self._train_results)} output(s)")
        # Regenerate auto-names for next session
        self._refresh_output_checks()

    def _on_model_started(self, name):
        logger.info(f"Training started: {name}")

    def _on_error(self, msg):
        self._train_btn.setEnabled(True)
        self._worker = None
        self._train_queue = []
        QMessageBox.critical(self, "Training Error", msg)

    # --- Live loss curves ---

    def _create_live_canvas(self, model_name):
        """Create a matplotlib canvas tab for live loss updates."""
        if not HAS_MATPLOTLIB:
            label = QLabel(f"Training {model_name}... (matplotlib not available)")
            label.setAlignment(Qt.AlignCenter)
            self._loss_tabs.addTab(label, model_name)
            self._loss_tabs.setCurrentIndex(self._loss_tabs.count() - 1)
            return

        fig = Figure(figsize=(8, 4), facecolor=theme.BG_PANEL)
        ax = fig.add_subplot(111, facecolor=theme.BG_DARK)
        ax.set_xlabel("Epoch", color=theme.TEXT_SECONDARY)
        ax.set_ylabel("Loss", color=theme.TEXT_SECONDARY)
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.3, color=theme.BORDER)
        ax.tick_params(colors=theme.TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(theme.BORDER)

        train_line, = ax.plot([], [], color=theme.ORANGE_LIGHT, label='Train', linewidth=2)
        val_line, = ax.plot([], [], color=theme.BLUE_INFO, label='Validation', linewidth=2, linestyle='--')
        ax.legend(loc='upper right', facecolor=theme.BG_PANEL,
                  edgecolor=theme.BORDER, labelcolor=theme.TEXT_PRIMARY)

        fig.tight_layout()
        canvas = FigureCanvasQTAgg(fig)

        self._loss_tabs.addTab(canvas, model_name)
        self._loss_tabs.setCurrentIndex(self._loss_tabs.count() - 1)

        self._live_canvases[model_name] = {
            'canvas': canvas,
            'fig': fig,
            'ax': ax,
            'train_line': train_line,
            'val_line': val_line,
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
        }

    def _on_epoch_update(self, name, epoch, train_loss, val_loss):
        """Update the live loss curve for a model."""
        data = self._live_canvases.get(name)
        if data is None:
            return

        data['epochs'].append(epoch)
        data['train_losses'].append(train_loss)
        data['train_line'].set_data(data['epochs'], data['train_losses'])

        # -1 sentinel means "no validation data" (small dataset mode)
        if val_loss < 0:
            data['val_line'].set_visible(False)
            if data.get('_legend_updated') is not True:
                data['ax'].legend(
                    [data['train_line']], ['Train'],
                    loc='upper right', facecolor=theme.BG_PANEL,
                    edgecolor=theme.BORDER, labelcolor=theme.TEXT_PRIMARY,
                )
                data['_legend_updated'] = True
        else:
            data['val_losses'].append(val_loss)
            data['val_line'].set_data(data['epochs'][:len(data['val_losses'])], data['val_losses'])

        ax = data['ax']
        ax.relim()
        ax.autoscale_view()
        data['canvas'].draw_idle()
