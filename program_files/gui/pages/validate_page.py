"""
Validate Page Module
====================
Model list with checkboxes, metrics dashboard per model,
prediction panel (dataset point or custom params), Fluent comparison.

Design:
  - Click a model = focus it (metrics, prediction view) AND check it
  - Check a model = include in predict/compare batch
  - Predict runs all checked models
  - Fluent comparison runs once, caches per-params to disk
"""

import json
import logging
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFrame, QComboBox, QLineEdit,
    QFormLayout, QStackedWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QDoubleSpinBox, QScrollArea, QTabWidget,
)
from PySide6.QtCore import Qt, Signal, QSize

from .. import theme
from ..fluent_manager import FluentManager
from ..dataset_manager import DatasetManager
from ..fluent_cache import FluentCache
from ..workers import ValidationWorker

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    import mpl_toolkits.mplot3d  # noqa: F401  registers 3d projection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# Plotting helpers
# ============================================================

def _make_scalar_bar_figure(nn_val, fluent_val, truth_val, label):
    """Bar chart for 1D scalar comparison. Missing values skipped."""
    fig = Figure(figsize=(6, 3.5), facecolor=theme.BG_PANEL)
    ax = fig.add_subplot(111)
    _style_axis(ax)

    categories = []
    values = []
    colors = []
    if nn_val is not None:
        categories.append('NN')
        values.append(nn_val)
        colors.append(theme.ORANGE_LIGHT)
    if fluent_val is not None:
        categories.append('Fluent')
        values.append(fluent_val)
        colors.append(theme.BLUE_INFO)
    if truth_val is not None:
        categories.append('Dataset')
        values.append(truth_val)
        colors.append(theme.GREEN_SUCCESS)

    if not values:
        ax.text(0.5, 0.5, "No values", ha='center', va='center',
                color=theme.TEXT_SECONDARY, transform=ax.transAxes)
    else:
        bars = ax.bar(categories, values, color=colors, edgecolor=theme.BORDER)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4g}", ha='center', va='bottom',
                    color=theme.TEXT_PRIMARY)
    ax.set_ylabel(label, color=theme.TEXT_SECONDARY)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color=theme.BORDER)
    fig.tight_layout()
    return fig


def _is_3d_data(coordinates, eps_rel=1e-3):
    """True if all three coordinate axes have non-trivial span."""
    if coordinates is None:
        return False
    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] < 3:
        return False
    spans = coords.max(axis=0) - coords.min(axis=0)
    max_span = spans.max()
    if max_span == 0:
        return False
    return np.sum(spans > eps_rel * max_span) >= 3


def _downsample(n_points, max_points):
    """Return an array of indices to plot, randomly chosen if n_points > max_points."""
    if max_points <= 0 or n_points <= max_points:
        return np.arange(n_points)
    rng = np.random.default_rng(seed=42)  # deterministic per run
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def _make_field_figure(nn_values, truth_values, coordinates=None, title="Field",
                        max_points=5000):
    """
    Horizontal 1x3 figure for 2D/3D field comparison: NN | Truth | Abs Error.
    NN/Truth share a colorbar; Error has its own.
    If only NN is available, shows just that panel.

    Large fields are randomly downsampled to max_points for faster rendering
    and interactivity.
    """
    # Apply random downsampling to all arrays together
    nn_values = np.asarray(nn_values).flatten()
    coordinates = np.asarray(coordinates) if coordinates is not None else None
    truth_values = np.asarray(truth_values).flatten() if truth_values is not None else None

    n_full = len(nn_values)
    if n_full > max_points:
        idx = _downsample(n_full, max_points)
        nn_values = nn_values[idx]
        if coordinates is not None and len(coordinates) == n_full:
            coordinates = coordinates[idx]
        if truth_values is not None and len(truth_values) == n_full:
            truth_values = truth_values[idx]

    is_3d = _is_3d_data(coordinates)
    # Use constrained_layout so colorbars get their own space automatically
    fig = Figure(figsize=(13, 4.5), facecolor=theme.BG_PANEL, layout='constrained')

    has_truth = truth_values is not None

    if not has_truth:
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        _style_axis(ax, is_3d=is_3d)
        im = _plot_field(ax, nn_values, coordinates, title="NN Prediction", is_3d=is_3d)
        if im is not None:
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
            _style_colorbar(cb)
        return fig

    # 1x3 layout
    if is_3d:
        ax_nn = fig.add_subplot(1, 3, 1, projection='3d')
        ax_truth = fig.add_subplot(1, 3, 2, projection='3d')
        ax_err = fig.add_subplot(1, 3, 3, projection='3d')
    else:
        ax_nn = fig.add_subplot(1, 3, 1)
        ax_truth = fig.add_subplot(1, 3, 2)
        ax_err = fig.add_subplot(1, 3, 3)

    for ax in [ax_nn, ax_truth, ax_err]:
        _style_axis(ax, is_3d=is_3d)

    # Shared color scale for NN/Truth
    vmin = float(min(np.nanmin(nn_values), np.nanmin(truth_values)))
    vmax = float(max(np.nanmax(nn_values), np.nanmax(truth_values)))

    im1 = _plot_field(ax_nn, nn_values, coordinates, title="NN",
                      vmin=vmin, vmax=vmax, is_3d=is_3d)
    im2 = _plot_field(ax_truth, truth_values, coordinates, title="Truth",
                      vmin=vmin, vmax=vmax, is_3d=is_3d)
    # Each of NN and Truth gets its own colorbar so constrained_layout
    # can allocate space without overlapping the next axis.
    if im1 is not None:
        cb1 = fig.colorbar(im1, ax=ax_nn, fraction=0.05, pad=0.04)
        _style_colorbar(cb1)
    if im2 is not None:
        cb2 = fig.colorbar(im2, ax=ax_truth, fraction=0.05, pad=0.04)
        _style_colorbar(cb2)

    err = np.abs(np.asarray(nn_values).flatten() - np.asarray(truth_values).flatten())
    im3 = _plot_field(ax_err, err, coordinates, title="Absolute Error",
                      cmap='Reds', is_3d=is_3d)
    if im3 is not None:
        cb3 = fig.colorbar(im3, ax=ax_err, fraction=0.05, pad=0.04)
        _style_colorbar(cb3)

    return fig


def _style_colorbar(cb):
    cb.ax.tick_params(colors=theme.TEXT_SECONDARY)
    for spine in cb.ax.spines.values():
        spine.set_color(theme.BORDER)


def _make_single_field_figure(values, coordinates=None, title="Field",
                               cmap='viridis', max_points=300):
    """Single-panel figure for one field (NN, Truth, or Error) in a pop-out window."""
    values = np.asarray(values).flatten()
    coordinates = np.asarray(coordinates) if coordinates is not None else None
    n_full = len(values)
    if n_full > max_points and coordinates is not None and len(coordinates) == n_full:
        idx = _downsample(n_full, max_points)
        values = values[idx]
        coordinates = coordinates[idx]

    is_3d = _is_3d_data(coordinates)
    fig = Figure(figsize=(8, 6), facecolor=theme.BG_PANEL, layout='constrained')

    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    _style_axis(ax, is_3d=is_3d)

    im = _plot_field(ax, values, coordinates, title=title, cmap=cmap, is_3d=is_3d)
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        _style_colorbar(cb)
    return fig


def _style_axis(ax, is_3d=False):
    if is_3d:
        # 3D axes have different APIs
        ax.set_facecolor(theme.BG_DARK)
        ax.tick_params(colors=theme.TEXT_SECONDARY)
        try:
            ax.xaxis.pane.set_facecolor(theme.BG_DARK)
            ax.yaxis.pane.set_facecolor(theme.BG_DARK)
            ax.zaxis.pane.set_facecolor(theme.BG_DARK)
            ax.xaxis.pane.set_edgecolor(theme.BORDER)
            ax.yaxis.pane.set_edgecolor(theme.BORDER)
            ax.zaxis.pane.set_edgecolor(theme.BORDER)
        except Exception:
            pass
        ax.xaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.yaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.zaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.title.set_color(theme.TEXT_PRIMARY)
    else:
        ax.set_facecolor(theme.BG_DARK)
        ax.tick_params(colors=theme.TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(theme.BORDER)
        ax.xaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.yaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.title.set_color(theme.TEXT_PRIMARY)


def _plot_field(ax, values, coordinates, title, cmap='viridis',
                vmin=None, vmax=None, is_3d=False):
    """
    Scatter plot a field.
    Returns the ScalarMappable for colorbar, or None.
    """
    ax.set_title(title)
    values = np.asarray(values).flatten()
    if coordinates is None or len(coordinates) != len(values):
        ax.plot(values, color=theme.ORANGE_LIGHT)
        ax.set_xlabel("Point index")
        ax.set_ylabel("Value")
        return None

    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] < 2:
        ax.plot(values, color=theme.ORANGE_LIGHT)
        return None

    if is_3d and coords.shape[1] >= 3:
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=values, s=4, cmap=cmap, vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # Enforce 1:1:1 aspect ratio in data space
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        x_mid = (coords[:, 0].max() + coords[:, 0].min()) / 2
        y_mid = (coords[:, 1].max() + coords[:, 1].min()) / 2
        z_mid = (coords[:, 2].max() + coords[:, 2].min()) / 2
        spans = coords.max(axis=0) - coords.min(axis=0)
        half = spans.max() / 2
        if half > 0:
            ax.set_xlim(x_mid - half, x_mid + half)
            ax.set_ylim(y_mid - half, y_mid + half)
            ax.set_zlim(z_mid - half, z_mid + half)
        return scatter

    # 2D: pick the two widest dimensions
    spans = coords.max(axis=0) - coords.min(axis=0)
    dims = np.argsort(-spans)[:2]
    x = coords[:, dims[0]]
    y = coords[:, dims[1]]
    scatter = ax.scatter(x, y, c=values, s=6, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel(f"axis {dims[0]}")
    ax.set_ylabel(f"axis {dims[1]}")
    return scatter


# ============================================================
# Model list item widget (checkbox + label)
# ============================================================

class ModelListItem(QWidget):
    """Row widget with a checkbox + a multi-line label for the model list."""

    def __init__(self, display_text, checked=True, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        from PySide6.QtWidgets import QCheckBox
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        layout.addWidget(self.checkbox)

        self.label = QLabel(display_text)
        self.label.setWordWrap(False)
        layout.addWidget(self.label, 1)


# ============================================================
# Validate page
# ============================================================

class ValidatePage(QWidget):
    """Model validation and prediction page."""

    def __init__(self, project, settings, parent=None):
        super().__init__(parent)
        self.project = project
        self.settings = settings
        self.dm = DatasetManager(project.dataset_dir)
        self.fluent_cache = FluentCache(
            project.fluent_cache_dir, project.fluent_cache_index_file)
        self._worker = None
        self._models_meta = {}      # name -> metadata dict
        self._model_widgets = {}    # name -> ModelListItem
        self._current_model = None  # focused model name
        self._last_fluent_data = None  # dict of NPZ keys -> arrays from last comparison run

        self._build_ui()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Validate")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        layout.addSpacing(8)

        # --- Model list header ---
        list_row = QHBoxLayout()
        list_label = QLabel("Trained Models")
        list_label.setProperty("secondary", True)
        list_row.addWidget(list_label)
        list_row.addStretch()

        self._delete_btn = QPushButton("Delete Model")
        self._delete_btn.setProperty("flat", True)
        self._delete_btn.setFixedWidth(120)
        self._delete_btn.clicked.connect(self._delete_model)
        list_row.addWidget(self._delete_btn)
        layout.addLayout(list_row)

        self._model_list = QListWidget()
        self._model_list.setSpacing(2)
        self._model_list.itemClicked.connect(self._on_model_clicked)
        layout.addWidget(self._model_list, 1)

        # --- Dashboard (shown when a model is focused) ---
        self._dashboard = QFrame()
        self._dashboard.setProperty("panel", True)
        dash_layout = QVBoxLayout(self._dashboard)
        dash_layout.setContentsMargins(16, 16, 16, 16)

        # Metrics table
        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(2)
        self._metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._metrics_table.setMaximumHeight(160)
        dash_layout.addWidget(self._metrics_table)

        # Prediction mode row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Prediction mode:"))
        self._pred_mode = QComboBox()
        self._pred_mode.addItems(["Dataset Point", "Custom Parameters"])
        self._pred_mode.currentIndexChanged.connect(self._on_pred_mode_changed)
        mode_row.addWidget(self._pred_mode)
        mode_row.addSpacing(16)

        self._dataset_combo = QComboBox()
        self._dataset_combo.setMinimumWidth(180)
        mode_row.addWidget(self._dataset_combo)
        mode_row.addStretch()
        dash_layout.addLayout(mode_row)

        # Custom params form (hidden in dataset-point mode)
        self._custom_scroll = QScrollArea()
        self._custom_scroll.setWidgetResizable(True)
        self._custom_scroll.setMaximumHeight(120)
        self._custom_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._custom_container = QWidget()
        self._custom_form = QFormLayout(self._custom_container)
        self._custom_scroll.setWidget(self._custom_container)
        self._custom_scroll.hide()
        dash_layout.addWidget(self._custom_scroll)

        self._extrap_warning = QLabel("")
        self._extrap_warning.setStyleSheet(f"color: {theme.YELLOW_WARNING}; background: transparent;")
        self._extrap_warning.hide()
        dash_layout.addWidget(self._extrap_warning)

        # Action buttons
        btn_row = QHBoxLayout()
        self._predict_btn = QPushButton("Predict")
        self._predict_btn.setFixedWidth(120)
        self._predict_btn.clicked.connect(self._run_predict)
        btn_row.addWidget(self._predict_btn)

        self._compare_btn = QPushButton("Run Fluent Comparison")
        self._compare_btn.setFixedWidth(180)
        self._compare_btn.clicked.connect(self._run_fluent_comparison)
        btn_row.addWidget(self._compare_btn)

        self._clear_cache_btn = QPushButton("Clear Cache")
        self._clear_cache_btn.setProperty("flat", True)
        self._clear_cache_btn.setFixedWidth(120)
        self._clear_cache_btn.clicked.connect(self._clear_cache)
        btn_row.addWidget(self._clear_cache_btn)

        self._cache_label = QLabel("")
        self._cache_label.setProperty("secondary", True)
        btn_row.addWidget(self._cache_label)

        btn_row.addSpacing(16)
        btn_row.addWidget(QLabel("Plot points:"))
        from PySide6.QtWidgets import QSpinBox
        self._downsample_spin = QSpinBox()
        self._downsample_spin.setRange(100, 1000000)
        self._downsample_spin.setValue(300)
        self._downsample_spin.setButtonSymbols(QSpinBox.NoButtons)
        self._downsample_spin.setFixedWidth(100)
        self._downsample_spin.setToolTip(
            "Maximum number of points to plot. Larger fields are randomly downsampled."
        )
        btn_row.addWidget(self._downsample_spin)
        btn_row.addStretch()
        dash_layout.addLayout(btn_row)

        # Results tab widget (one tab per model after predict)
        self._results_tabs = QTabWidget()
        self._results_tabs.setMinimumHeight(360)
        dash_layout.addWidget(self._results_tabs, 1)

        self._dashboard.hide()
        layout.addWidget(self._dashboard, 2)

    # ------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------

    def refresh(self):
        """Reload model list from disk."""
        self._model_list.clear()
        self._models_meta.clear()
        self._model_widgets.clear()

        if not self.project.models_dir.exists():
            self._update_cache_label()
            return

        for model_dir in sorted(self.project.models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            meta_files = list(model_dir.glob("*_metadata.json"))
            if not meta_files:
                continue
            try:
                with open(meta_files[0], 'r') as f:
                    meta = json.load(f)
            except Exception:
                continue

            name = model_dir.name
            test_metrics = meta.get('test_metrics', {})
            r2 = test_metrics.get('r2', 'N/A')
            if isinstance(r2, float):
                r2 = f"{r2:.4f}"
            dv = meta.get('dataset_version', '?')
            date = meta.get('trained_date', '?')
            if isinstance(date, str) and 'T' in date:
                date = date.split('T')[0]
            stale = self.dm.is_model_stale(meta)

            display = f"{name}    R² {r2}    v{dv}    {date}"
            if stale:
                display += "    [STALE]"

            row_widget = ModelListItem(display, checked=True)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, name)
            item.setSizeHint(row_widget.sizeHint())
            self._model_list.addItem(item)
            self._model_list.setItemWidget(item, row_widget)

            self._models_meta[name] = meta
            self._model_widgets[name] = row_widget

        # Populate dataset point combo
        self._dataset_combo.clear()
        for sid in self.dm.get_completed_ids():
            self._dataset_combo.addItem(f"Sample {sid}", sid)

        self._update_cache_label()

    def _update_cache_label(self):
        n = self.fluent_cache.count()
        self._cache_label.setText(f"{n} cached run{'s' if n != 1 else ''}")

    # ------------------------------------------------------------
    # Model list interaction
    # ------------------------------------------------------------

    def _on_model_clicked(self, item):
        """Click focuses a model AND checks its checkbox."""
        name = item.data(Qt.UserRole)
        if not name:
            return

        # Auto-check when clicked
        widget = self._model_widgets.get(name)
        if widget:
            widget.checkbox.setChecked(True)

        self._focus_model(name)

    def _focus_model(self, name):
        """Show metrics dashboard for this model."""
        self._current_model = name
        meta = self._models_meta.get(name, {})
        self._dashboard.show()
        self._model_list.setMaximumHeight(200)

        # Metrics
        test_metrics = meta.get('test_metrics', {})
        train_metrics = meta.get('train_metrics', {})
        rows = []
        for key in ['r2', 'rmse', 'mae']:
            test_val = test_metrics.get(key, 'N/A')
            train_val = train_metrics.get(key, 'N/A')
            if isinstance(test_val, float):
                test_val = f"{test_val:.6f}"
            if isinstance(train_val, float):
                train_val = f"{train_val:.6f}"
            rows.append((f"Test {key.upper()}", str(test_val)))
            rows.append((f"Train {key.upper()}", str(train_val)))

        rows.append(("Output", meta.get('output_key', '?')))
        rows.append(("Type", meta.get('output_type', '?')))
        rows.append(("Dataset Version", str(meta.get('dataset_version', '?'))))
        date = meta.get('trained_date', '?')
        if isinstance(date, str) and 'T' in date:
            date = date.split('T')[0]
        rows.append(("Date Trained", date))

        self._metrics_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self._metrics_table.setItem(i, 0, QTableWidgetItem(k))
            self._metrics_table.setItem(i, 1, QTableWidgetItem(v))

        self._build_custom_param_inputs()

    def _get_checked_models(self):
        """Return list of model names that are checked."""
        checked = []
        for i in range(self._model_list.count()):
            item = self._model_list.item(i)
            name = item.data(Qt.UserRole)
            widget = self._model_widgets.get(name)
            if widget and widget.checkbox.isChecked():
                checked.append(name)
        return checked

    # ------------------------------------------------------------
    # Prediction mode
    # ------------------------------------------------------------

    def _on_pred_mode_changed(self, idx):
        self._dataset_combo.setVisible(idx == 0)
        self._custom_scroll.setVisible(idx == 1)
        self._extrap_warning.hide()

    def _build_custom_param_inputs(self):
        """Build parameter input fields from model_setup.json."""
        while self._custom_form.count():
            item = self._custom_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._param_spins = {}
        if not self.project.model_setup_file.exists():
            return
        try:
            with open(self.project.model_setup_file, 'r') as f:
                setup = json.load(f)
        except Exception:
            return

        from ...modules.doe_setup import load_doe_samples
        _, ranges = load_doe_samples(self.project.doe_samples_file)

        for inp in setup.get('model_inputs', []):
            key = f"{inp['name']}|{inp.get('parameter', 'value')}"
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-1e12, 1e12)
            spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
            rng = ranges.get(key, {})
            if rng:
                mid = (rng['min'] + rng['max']) / 2
                spin.setValue(mid)
            label = key.replace('|', ' : ')
            self._custom_form.addRow(label, spin)
            self._param_spins[key] = (spin, rng)

    def _get_current_params(self):
        """Get the current input params as {bc|param: value}, plus the mode."""
        if self._pred_mode.currentIndex() == 0:
            # Dataset point: look up from doe_samples
            sim_id = self._dataset_combo.currentData()
            if sim_id is None:
                return None, None
            from ...modules.doe_setup import load_doe_samples
            samples, _ = load_doe_samples(self.project.doe_samples_file)
            if sim_id - 1 >= len(samples):
                return None, None
            return dict(samples[sim_id - 1]), sim_id
        else:
            params = {k: spin.value() for k, (spin, _) in self._param_spins.items()}
            return params, None

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------

    def _run_predict(self):
        checked = self._get_checked_models()
        if not checked:
            QMessageBox.warning(self, "No Models", "Check at least one model.")
            return

        params, sim_id = self._get_current_params()
        if params is None:
            QMessageBox.warning(self, "No Params", "Select a dataset point or fill custom parameters.")
            return

        # Extrapolation check for custom mode
        if self._pred_mode.currentIndex() == 1:
            outside = []
            for key, (spin, rng) in self._param_spins.items():
                val = spin.value()
                if rng and (val < rng.get('min', val) or val > rng.get('max', val)):
                    outside.append(key)
            if outside:
                self._extrap_warning.setText(
                    f"Warning: extrapolating outside DOE range for: "
                    f"{', '.join(k.replace('|', ':') for k in outside)}"
                )
                self._extrap_warning.show()
            else:
                self._extrap_warning.hide()

        # Build X vector in the same sorted-key order the trainer uses
        param_names = sorted(params.keys())
        X = np.array([[params[k] for k in param_names]])

        # Check for cached Fluent data for these params
        cached_fluent = self.fluent_cache.lookup(params)
        if cached_fluent is not None:
            logger.info("Found cached Fluent data for these params")
        self._last_fluent_data = cached_fluent

        # Clear existing tabs
        self._results_tabs.clear()

        from ...modules.multi_model_visualizer import predict_single_model, predict_dataset_point_single
        from ...modules.doe_setup import load_doe_samples

        doe_samples, _ = load_doe_samples(self.project.doe_samples_file)

        for name in checked:
            model_dir = self.project.models_dir / name
            try:
                if sim_id is not None:
                    result = predict_dataset_point_single(
                        model_dir, self.project.dataset_dir, doe_samples, sim_id)
                else:
                    result = predict_single_model(model_dir, X)
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
                self._add_error_tab(name, str(e))
                continue

            if result is None:
                self._add_error_tab(name, "Prediction returned None")
                continue

            self._add_result_tab(name, result, cached_fluent)

        if self._results_tabs.count() > 0:
            self._results_tabs.setCurrentIndex(0)

    def _add_error_tab(self, name, msg):
        label = QLabel(f"Error: {msg}")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"color: {theme.RED_ERROR}; background: transparent;")
        self._results_tabs.addTab(label, name)

    def _add_result_tab(self, name, result, fluent_data):
        """Add a tab showing prediction results for a single model."""
        meta = result['metadata']
        output_type = meta.get('output_type', '1D')
        output_key = meta.get('output_key', '')
        npz_key = meta.get('npz_key', '')

        nn_pred = np.asarray(result['prediction'])
        truth = result.get('ground_truth')
        if truth is not None:
            truth = np.asarray(truth)

        fluent_vals = None
        if fluent_data and npz_key in fluent_data:
            fluent_vals = np.asarray(fluent_data[npz_key])

        if output_type == '1D':
            widget = self._build_scalar_result_widget(name, result, nn_pred, truth, fluent_vals)
        else:
            widget = self._build_field_result_widget(name, result, nn_pred, truth, fluent_vals)

        self._results_tabs.addTab(widget, name)

    def _build_scalar_result_widget(self, name, result, nn_pred, truth, fluent_vals):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        # Summary row
        meta = result['metadata']
        field_label = meta.get('field_name', 'value')
        nn_val = float(np.asarray(nn_pred).flatten()[0])
        truth_val = float(truth.flatten()[0]) if truth is not None else None
        fluent_val = float(fluent_vals.flatten()[0]) if fluent_vals is not None else None

        text = f"<b>NN prediction:</b> {nn_val:.6g}"
        if truth_val is not None:
            err = abs(nn_val - truth_val)
            pct = 100 * err / abs(truth_val) if truth_val != 0 else 0
            text += f"<br><b>Dataset truth:</b> {truth_val:.6g}  (abs err {err:.4g}, {pct:.2f}%)"
        if fluent_val is not None:
            err = abs(nn_val - fluent_val)
            pct = 100 * err / abs(fluent_val) if fluent_val != 0 else 0
            text += f"<br><b>Fluent:</b> {fluent_val:.6g}  (abs err {err:.4g}, {pct:.2f}%)"

        summary_row = QHBoxLayout()
        summary = QLabel(text)
        summary.setTextFormat(Qt.RichText)
        summary.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; background: transparent; padding: 8px;")
        summary_row.addWidget(summary, 1)

        if HAS_MATPLOTLIB:
            def build_fig():
                return _make_scalar_bar_figure(nn_val, fluent_val, truth_val, field_label)

            popout_btn = QPushButton("Open in new window")
            popout_btn.setProperty("flat", True)
            popout_btn.setFixedHeight(28)
            popout_btn.clicked.connect(
                lambda: self._open_plot_window(name, build_fig)
            )
            summary_row.addWidget(popout_btn, 0, Qt.AlignTop)
        layout.addLayout(summary_row)

        if HAS_MATPLOTLIB:
            fig = build_fig()
            canvas = FigureCanvasQTAgg(fig)
            layout.addWidget(canvas, 1)

        return container

    def _build_field_result_widget(self, name, result, nn_pred, truth, fluent_vals):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        meta = result['metadata']
        output_key = meta.get('output_key', '')

        # Metrics summary
        info_parts = [f"<b>{meta.get('field_name', 'value')}</b> @ {meta.get('location', '?')}"]
        metrics = result.get('metrics')
        if metrics:
            info_parts.append(
                f"R² {metrics['r2']:.4f}  RMSE {metrics['rmse']:.4g}  MAE {metrics['mae']:.4g}"
            )
        if fluent_vals is not None:
            err = np.abs(nn_pred.flatten() - fluent_vals.flatten())
            info_parts.append(f"Fluent MAE {np.mean(err):.4g}  Max err {np.max(err):.4g}")

        # Plot — prefer Fluent as truth if we have both
        compare_vals = fluent_vals if fluent_vals is not None else truth
        coords = self._load_coordinates_for(meta)

        summary = QLabel("<br>".join(info_parts))
        summary.setTextFormat(Qt.RichText)
        summary.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; background: transparent; padding: 8px;")
        layout.addWidget(summary)

        if HAS_MATPLOTLIB:
            # Combined tri-plot in the tab
            def build_tri():
                return _make_field_figure(
                    nn_pred, compare_vals, coords, title=output_key,
                    max_points=self._downsample_spin.value(),
                )

            fig = build_tri()
            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, container)
            toolbar.setStyleSheet(f"background: {theme.BG_PANEL}; color: {theme.TEXT_PRIMARY};")
            layout.addWidget(toolbar)
            layout.addWidget(canvas, 1)

            # Individual pop-out buttons
            btn_row = QHBoxLayout()
            btn_row.addStretch()

            def _make_single_fig(values, title_str, cmap='viridis'):
                return _make_single_field_figure(
                    values, coords, title=title_str, cmap=cmap,
                    max_points=self._downsample_spin.value(),
                )

            nn_btn = QPushButton("NN")
            nn_btn.setProperty("flat", True)
            nn_btn.setFixedHeight(26)
            nn_btn.clicked.connect(
                lambda: self._open_plot_window(
                    f"{name} — NN",
                    lambda: _make_single_fig(nn_pred, "NN Prediction")
                )
            )
            btn_row.addWidget(nn_btn)

            if compare_vals is not None:
                truth_btn = QPushButton("Truth")
                truth_btn.setProperty("flat", True)
                truth_btn.setFixedHeight(26)
                truth_btn.clicked.connect(
                    lambda: self._open_plot_window(
                        f"{name} — Truth",
                        lambda: _make_single_fig(compare_vals, "Truth")
                    )
                )
                btn_row.addWidget(truth_btn)

                err_btn = QPushButton("Error")
                err_btn.setProperty("flat", True)
                err_btn.setFixedHeight(26)
                err_vals = np.abs(nn_pred.flatten() - np.asarray(compare_vals).flatten())
                err_btn.clicked.connect(
                    lambda: self._open_plot_window(
                        f"{name} — Error",
                        lambda: _make_single_fig(err_vals, "Absolute Error", cmap='Reds')
                    )
                )
                btn_row.addWidget(err_btn)

            layout.addLayout(btn_row)
        else:
            layout.addWidget(QLabel("matplotlib not available"))

        return container

    def _open_plot_window(self, title, fig_builder):
        """Open a resizable dialog containing a fresh canvas with the same plot."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout as _VLayout

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(1100, 700)

        layout = _VLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)

        try:
            fig = fig_builder()
        except Exception as e:
            layout.addWidget(QLabel(f"Failed to build plot: {e}"))
            dlg.exec()
            return

        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, dlg)
        toolbar.setStyleSheet(f"background: {theme.BG_PANEL}; color: {theme.TEXT_PRIMARY};")
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)

        # Modeless so you can keep interacting with the main window
        dlg.setModal(False)
        dlg.show()
        # Hold a reference to prevent garbage collection
        if not hasattr(self, '_plot_windows'):
            self._plot_windows = []
        self._plot_windows.append(dlg)
        dlg.finished.connect(lambda _: self._plot_windows.remove(dlg) if dlg in self._plot_windows else None)

    def _load_coordinates_for(self, meta):
        """Load coordinate array for a field output from coordinates.npz."""
        coords_file = self.project.dataset_dir / "coordinates.npz"
        if not coords_file.exists():
            return None
        try:
            data = np.load(coords_file, allow_pickle=True)
            location = meta.get('location', '')
            coord_key = f"{location}|coordinates"
            if coord_key in data.files:
                return data[coord_key]
        except Exception as e:
            logger.warning(f"Could not load coordinates: {e}")
        return None

    # ------------------------------------------------------------
    # Fluent comparison
    # ------------------------------------------------------------

    def _run_fluent_comparison(self):
        fm = FluentManager.instance()
        if not fm.is_available():
            QMessageBox.warning(self, "Not Connected",
                                "Fluent is not connected. Go to Setup and launch Fluent.")
            return

        params, sim_id = self._get_current_params()
        if params is None:
            QMessageBox.warning(self, "No Params", "Select a dataset point or fill custom parameters.")
            return

        # Cache hit?
        cached = self.fluent_cache.lookup(params)
        if cached is not None:
            self._last_fluent_data = cached
            logger.info("Using cached Fluent run for comparison")
            QMessageBox.information(self, "Cached",
                                    "Found a cached Fluent run for these parameters. "
                                    "Click Predict to see the comparison.")
            return

        # Read model_setup
        try:
            with open(self.project.model_setup_file, 'r') as f:
                setup_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot read model_setup.json: {e}")
            return

        fm.set_busy()
        self._compare_btn.setEnabled(False)
        self._compare_btn.setText("Running Fluent...")

        self._pending_params = dict(params)
        self._worker = ValidationWorker(
            solver=fm.solver,
            setup_data=setup_data,
            dataset_dir=self.project.dataset_dir,
            parameters=params,
        )
        self._worker.finished.connect(self._on_comparison_done)
        self._worker.error.connect(self._on_comparison_error)
        self._worker.start()

    def _on_comparison_done(self, results):
        fm = FluentManager.instance()
        fm.set_idle()
        self._compare_btn.setEnabled(True)
        self._compare_btn.setText("Run Fluent Comparison")

        if results:
            self.fluent_cache.store(self._pending_params, results)
            self._last_fluent_data = results
            self._update_cache_label()
            logger.info("Fluent comparison complete and cached")
            QMessageBox.information(self, "Done",
                                    "Fluent comparison complete. Click Predict to view results.")
        self._worker = None

    def _on_comparison_error(self, msg):
        fm = FluentManager.instance()
        fm.set_idle()
        self._compare_btn.setEnabled(True)
        self._compare_btn.setText("Run Fluent Comparison")
        self._worker = None
        QMessageBox.critical(self, "Validation Error", msg)

    def _clear_cache(self):
        reply = QMessageBox.question(
            self, "Clear Cache",
            f"Delete {self.fluent_cache.count()} cached Fluent run(s)?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.fluent_cache.clear()
        self._last_fluent_data = None
        self._update_cache_label()

    # ------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------

    def _delete_model(self):
        current = self._model_list.currentItem()
        if not current:
            return
        name = current.data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Delete model '{name}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if self.project.delete_model(name):
            self.refresh()
            self._dashboard.hide()
            self._model_list.setMaximumHeight(16777215)
