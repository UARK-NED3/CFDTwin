"""
DOE Page Module
===============
Two-panel layout: config (left) + preview table & scatter matrix (right).
Supports LHS, factorial, and manual point entry with redundancy checking.
"""

import json
import logging
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QFrame, QFormLayout, QDoubleSpinBox, QSpinBox,
    QComboBox, QTableView, QHeaderView, QMessageBox, QAbstractItemView,
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal

from .. import theme
from ...modules.doe_setup import (
    generate_lhs_samples, generate_factorial_samples,
    save_doe_samples, load_doe_samples,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Table model for DOE samples
# -----------------------------------------------------------------------

class DOESampleModel(QAbstractTableModel):
    """Editable table model backed by a list of sample dicts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = []
        self._keys = []

    def set_data(self, samples, keys):
        self.beginResetModel()
        self._samples = list(samples)
        self._keys = list(keys)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._samples)

    def columnCount(self, parent=QModelIndex()):
        return len(self._keys)

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def data(self, index, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.EditRole):
            val = self._samples[index.row()].get(self._keys[index.column()], 0)
            return f"{val:.6g}"
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            try:
                self._samples[index.row()][self._keys[index.column()]] = float(value)
                self.dataChanged.emit(index, index)
                return True
            except (ValueError, TypeError):
                return False
        return False

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal and section < len(self._keys):
                return self._keys[section].replace('|', '\n')
            if orientation == Qt.Vertical:
                return str(section + 1)
        return None

    def get_samples(self):
        return list(self._samples)

    def remove_rows(self, row_indices):
        """Remove rows by index list (descending order safe)."""
        for idx in sorted(row_indices, reverse=True):
            if 0 <= idx < len(self._samples):
                self.beginRemoveRows(QModelIndex(), idx, idx)
                self._samples.pop(idx)
                self.endRemoveRows()


# -----------------------------------------------------------------------
# DOE Page
# -----------------------------------------------------------------------

class DOEPage(QWidget):
    """Design of Experiments page."""

    doe_changed = Signal()  # emitted when DOE points are added/removed

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._ranges = {}
        self._param_keys = []

        self._build_ui()
        self._load_state()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Design of Experiments")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        layout.addSpacing(8)

        splitter = QSplitter(Qt.Horizontal)

        # --- Left panel: config ---
        left = QFrame()
        left.setProperty("panel", True)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)

        left_layout.addWidget(QLabel("Parameter Ranges"))

        # Range inputs (populated from model_setup.json)
        self._ranges_form = QFormLayout()
        self._range_widgets = {}
        left_layout.addLayout(self._ranges_form)

        left_layout.addSpacing(12)

        # Sampling method
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["LHS", "Factorial"])
        method_row.addWidget(self._method_combo)
        left_layout.addLayout(method_row)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Count:"))
        self._count_spin = QSpinBox()
        self._count_spin.setRange(1, 10000)
        self._count_spin.setValue(20)
        count_row.addWidget(self._count_spin)
        left_layout.addLayout(count_row)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.clicked.connect(self._generate_samples)
        left_layout.addWidget(self._generate_btn)

        self._add_row_btn = QPushButton("Add Manual Point")
        self._add_row_btn.setProperty("flat", True)
        self._add_row_btn.clicked.connect(self._add_manual_point)
        left_layout.addWidget(self._add_row_btn)

        self._delete_btn = QPushButton("Delete Selected")
        self._delete_btn.setProperty("flat", True)
        self._delete_btn.clicked.connect(self._delete_selected)
        left_layout.addWidget(self._delete_btn)

        left_layout.addStretch()

        self._count_label = QLabel("0 points")
        self._count_label.setProperty("secondary", True)
        left_layout.addWidget(self._count_label)

        splitter.addWidget(left)

        # --- Right panel: table + scatter ---
        right = QFrame()
        right.setProperty("panel", True)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 16, 16, 16)

        self._table_model = DOESampleModel()
        self._table_model.dataChanged.connect(self._on_cell_edited)
        self._table_view = QTableView()
        self._table_view.setModel(self._table_model)
        self._table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self._table_view, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, 1)

    # --- Load / Save ---

    def _load_state(self):
        """Load parameter ranges from model_setup and DOE samples from doe_samples.json."""
        self._load_ranges_from_setup()
        self._load_samples()

    def _load_ranges_from_setup(self):
        """Read model_setup.json to build param keys and range widgets."""
        # Clear existing
        while self._ranges_form.count():
            item = self._ranges_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._range_widgets.clear()
        self._param_keys.clear()

        if not self.project.model_setup_file.exists():
            return

        try:
            with open(self.project.model_setup_file, 'r') as f:
                setup = json.load(f)
        except Exception:
            return

        for inp in setup.get('model_inputs', []):
            key = f"{inp['name']}|{inp.get('parameter', 'value')}"
            self._param_keys.append(key)

            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            min_spin = QDoubleSpinBox()
            min_spin.setDecimals(4)
            min_spin.setRange(-1e12, 1e12)
            max_spin = QDoubleSpinBox()
            max_spin.setDecimals(4)
            max_spin.setRange(-1e12, 1e12)
            max_spin.setValue(1.0)
            row_l.addWidget(QLabel("Min:"))
            row_l.addWidget(min_spin)
            row_l.addWidget(QLabel("Max:"))
            row_l.addWidget(max_spin)

            label = key.replace('|', ' : ')
            self._ranges_form.addRow(label, row_w)
            self._range_widgets[key] = (min_spin, max_spin)

        # Pre-fill from saved ranges
        if self.project.doe_samples_file.exists():
            _, saved_ranges = load_doe_samples(self.project.doe_samples_file)
            for key, rng in saved_ranges.items():
                if key in self._range_widgets:
                    self._range_widgets[key][0].setValue(rng['min'])
                    self._range_widgets[key][1].setValue(rng['max'])

    def _load_samples(self):
        samples, _ = load_doe_samples(self.project.doe_samples_file)
        keys = self._param_keys if self._param_keys else (sorted(samples[0].keys()) if samples else [])
        self._table_model.set_data(samples, keys)
        self._update_count()

    def _save_samples(self):
        samples = self._table_model.get_samples()
        ranges = self._current_ranges()
        save_doe_samples(self.project.doe_samples_file, samples, ranges)
        self._update_count()
        self.doe_changed.emit()

    def _current_ranges(self):
        ranges = {}
        for key, (min_s, max_s) in self._range_widgets.items():
            ranges[key] = {'min': min_s.value(), 'max': max_s.value()}
        return ranges

    def _on_cell_edited(self):
        """Auto-save when a cell value is edited."""
        self._save_samples()

    def _update_count(self):
        n = self._table_model.rowCount()
        self._count_label.setText(f"{n} point{'s' if n != 1 else ''}")

    # --- Generate ---

    def _generate_samples(self):
        ranges = self._current_ranges()
        if not ranges:
            QMessageBox.warning(self, "No Parameters", "No input parameters configured. Complete Setup first.")
            return

        # Validate min < max
        for key, rng in ranges.items():
            if rng['min'] >= rng['max']:
                QMessageBox.warning(self, "Invalid Range", f"Min must be less than Max for {key}")
                return

        existing = self._table_model.get_samples()
        method = self._method_combo.currentText()
        count = self._count_spin.value()

        if method == "LHS":
            new_samples = generate_lhs_samples(ranges, count, existing_samples=existing)
        else:
            new_samples = generate_factorial_samples(ranges, count, existing_samples=existing)

        if not new_samples:
            QMessageBox.information(self, "No New Points", "All generated points were redundant with existing samples.")
            return

        all_samples = existing + new_samples
        self._table_model.set_data(all_samples, self._param_keys)
        self._save_samples()
        logger.info(f"Added {len(new_samples)} {method} points ({len(all_samples)} total)")

    # --- Manual point ---

    def _add_manual_point(self):
        ranges = self._current_ranges()
        if not ranges:
            QMessageBox.warning(self, "No Parameters", "No input parameters configured.")
            return

        # Use midpoint of each range as default
        point = {}
        for key in self._param_keys:
            rng = ranges.get(key, {'min': 0, 'max': 1})
            point[key] = (rng['min'] + rng['max']) / 2.0

        samples = self._table_model.get_samples()
        samples.append(point)
        self._table_model.set_data(samples, self._param_keys)
        self._save_samples()

        # Select the new row for easy editing visibility
        last_row = self._table_model.rowCount() - 1
        self._table_view.selectRow(last_row)

    # --- Delete ---

    def _delete_selected(self):
        selected = self._table_view.selectionModel().selectedRows()
        if not selected:
            return

        row_indices = sorted([idx.row() for idx in selected], reverse=True)

        reply = QMessageBox.question(
            self, "Delete Points",
            f"Delete {len(row_indices)} selected point(s)?\n"
            "Corresponding sample files will also be deleted.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Delete corresponding sim files (1-indexed: sim_0001.npz)
        for row_idx in row_indices:
            sim_id = row_idx + 1
            sim_file = self.project.dataset_dir / f"sim_{sim_id:04d}.npz"
            if sim_file.exists():
                sim_file.unlink()
                logger.info(f"Deleted {sim_file.name}")

        self._table_model.remove_rows(row_indices)
        self._save_samples()
        logger.info(f"Deleted {len(row_indices)} DOE point(s)")
