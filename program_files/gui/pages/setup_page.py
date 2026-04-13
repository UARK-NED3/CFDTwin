"""
Setup Page Module
=================
Mini-wizard with 3 sub-steps:
  1. Case file selection + Fluent launch
  2. Input BC selection + parameter per BC
  3. Output location selection + field variable multi-select
Selections save to model_setup.json and output_parameters.json.
"""

import json
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QStackedWidget, QFrame, QMessageBox,
    QCheckBox, QComboBox, QScrollArea, QGridLayout, QGroupBox,
    QSpinBox, QFormLayout, QTabWidget,
)
from PySide6.QtCore import Qt, Signal

from .. import theme
from ..fluent_manager import FluentManager
from ..workers import FluentLaunchWorker
from ..error_helpers import InlineErrorBanner
from ..spinner import LoadingOverlay
from ...modules.project_manager import get_available_inputs, get_available_outputs
from ...modules.doe_setup import get_bc_parameters
from ...modules.output_parameters import get_available_field_variables

logger = logging.getLogger(__name__)


class SetupPage(QWidget):
    """Setup mini-wizard: Case File > Inputs > Outputs."""

    setup_complete = Signal()  # emitted when inputs + outputs are configured

    def __init__(self, project, settings, parent=None):
        super().__init__(parent)
        self.project = project
        self.settings = settings
        self._worker = None
        self._input_rows = []   # list of (row_widget, bc_combo, param_combo)
        self._available_inputs = []  # cached from Fluent query
        self._output_rows = []  # list of (checkbox, name, category, field_checks)

        self._build_ui()
        self._loading = LoadingOverlay("Launching Fluent...", parent=self)
        self._load_state()

        # Listen for Fluent disconnect → return to case file step
        FluentManager.instance().status_changed.connect(self._on_fluent_status_changed)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel("Setup")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        # Sub-step indicator
        self._step_label = QLabel("Step 1 of 3: Case File")
        self._step_label.setProperty("secondary", True)
        layout.addWidget(self._step_label)

        # Dataset lock banner (shown when dataset exists)
        self._lock_banner = QLabel("Dataset exists. Delete dataset to modify inputs/outputs.")
        self._lock_banner.setStyleSheet(
            f"background-color: #3d2a1a; border: 1px solid {theme.ORANGE_LIGHT}; "
            f"border-radius: {theme.RADIUS_SMALL}; padding: 8px; color: {theme.ORANGE_LIGHT};"
        )
        self._lock_banner.hide()
        layout.addWidget(self._lock_banner)
        layout.addSpacing(8)

        # Sub-step stack
        self._sub_stack = QStackedWidget()
        self._sub_stack.addWidget(self._build_case_file_step())
        self._sub_stack.addWidget(self._build_inputs_step())
        self._sub_stack.addWidget(self._build_outputs_step())
        layout.addWidget(self._sub_stack, 1)

        # Navigation
        nav_row = QHBoxLayout()
        nav_row.addStretch()
        self._back_btn = QPushButton("Back")
        self._back_btn.setProperty("flat", True)
        self._back_btn.setFixedWidth(80)
        self._back_btn.clicked.connect(self._go_back)

        self._next_btn = QPushButton("Next")
        self._next_btn.setFixedWidth(80)
        self._next_btn.clicked.connect(self._go_next)

        nav_row.addWidget(self._back_btn)
        nav_row.addWidget(self._next_btn)
        layout.addLayout(nav_row)

        self._update_nav()

    # ===================================================================
    # Sub-step 1: Case File
    # ===================================================================

    def _build_case_file_step(self):
        page = QFrame()
        page.setProperty("panel", True)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        info = QLabel("Select a Fluent case file and configure solver settings.")
        info.setWordWrap(True)
        layout.addWidget(info)

        # File picker
        file_row = QHBoxLayout()
        self._case_edit = QLineEdit()
        self._case_edit.setReadOnly(True)
        self._case_edit.setPlaceholderText("No case file selected")
        self._browse_cas_btn = QPushButton("Browse")
        self._browse_cas_btn.setFixedWidth(80)
        self._browse_cas_btn.clicked.connect(self._browse_case_file)
        file_row.addWidget(self._case_edit, 1)
        file_row.addWidget(self._browse_cas_btn)
        layout.addLayout(file_row)

        # Fluent settings
        form = QFormLayout()
        form.setSpacing(8)

        self._precision_combo = QComboBox()
        self._precision_combo.addItems(["single", "double"])
        form.addRow("Precision:", self._precision_combo)

        self._processors_spin = QSpinBox()
        self._processors_spin.setRange(1, 128)
        self._processors_spin.setValue(1)
        self._processors_spin.setButtonSymbols(QSpinBox.NoButtons)
        form.addRow("Processors:", self._processors_spin)

        self._dimension_combo = QComboBox()
        self._dimension_combo.addItems(["2", "3"])
        self._dimension_combo.setCurrentText("3")
        form.addRow("Dimension:", self._dimension_combo)

        self._fluent_gui_cb = QCheckBox("Show Fluent GUI window")
        self._fluent_gui_cb.setChecked(True)
        form.addRow("", self._fluent_gui_cb)

        layout.addLayout(form)

        # Launch
        self._launch_status = QLabel("")
        self._launch_status.setWordWrap(True)
        layout.addWidget(self._launch_status)

        self._launch_btn = QPushButton("Launch Fluent")
        self._launch_btn.setFixedWidth(140)
        self._launch_btn.clicked.connect(self._launch_fluent)
        self._launch_btn.setEnabled(False)
        layout.addWidget(self._launch_btn)

        layout.addStretch()
        return page

    # ===================================================================
    # Sub-step 2: Inputs (plus-button row pattern)
    # ===================================================================

    def _build_inputs_step(self):
        page = QFrame()
        page.setProperty("panel", True)
        outer = QVBoxLayout(page)
        outer.setContentsMargins(20, 20, 20, 20)

        info = QLabel("Add input parameters. Select a boundary condition, then choose "
                       "which sub-component to vary. You can add multiple parameters from the same BC.")
        info.setWordWrap(True)
        outer.addWidget(info)

        # Scrollable rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._inputs_container = QWidget()
        self._inputs_layout = QVBoxLayout(self._inputs_container)
        self._inputs_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self._inputs_container)
        outer.addWidget(scroll, 1)

        # Plus button
        self._add_input_btn = QPushButton("+  Add Input")
        self._add_input_btn.setFixedWidth(140)
        self._add_input_btn.clicked.connect(self._add_input_row)
        outer.addWidget(self._add_input_btn)

        return page

    def _add_input_row(self, bc_name=None, param_name=None):
        """Add a new input row with BC dropdown + parameter dropdown."""
        fm = FluentManager.instance()

        # Query Fluent for BCs on first add (cache the results)
        if not self._available_inputs and fm.is_available():
            self._available_inputs = get_available_inputs(fm.solver)
        if not self._available_inputs:
            QMessageBox.warning(self, "Not Connected",
                                "Connect to Fluent first so available BCs can be loaded.")
            return

        locked = self._is_dataset_locked()

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 4, 0, 4)

        # Left dropdown: BCs
        bc_combo = QComboBox()
        bc_combo.setMinimumWidth(200)
        bc_combo.addItem("-- Select BC --")
        for item in self._available_inputs:
            bc_combo.addItem(f"{item['name']}  ({item['type']})", item)
        bc_combo.setEnabled(not locked)

        # Right dropdown: sub-components (disabled until BC selected)
        param_combo = QComboBox()
        param_combo.setMinimumWidth(220)
        param_combo.addItem("-- Select parameter --")
        param_combo.setEnabled(False)

        # Remove button
        remove_btn = QPushButton("x")
        remove_btn.setFixedSize(28, 28)
        remove_btn.setStyleSheet(
            f"background: transparent; color: {theme.RED_ERROR}; "
            f"border: 1px solid {theme.RED_ERROR}; border-radius: 4px; "
            f"font-weight: bold; font-size: 14px;"
        )
        remove_btn.setEnabled(not locked)

        row_layout.addWidget(bc_combo, 1)
        row_layout.addWidget(param_combo, 1)
        row_layout.addWidget(remove_btn)

        # Wire BC selection -> populate params
        def on_bc_changed(idx, pc=param_combo):
            pc.clear()
            pc.addItem("-- Select parameter --")
            if idx <= 0:
                pc.setEnabled(False)
                return
            bc_data = bc_combo.itemData(idx)
            if bc_data and fm.is_available():
                params = get_bc_parameters(fm.solver, bc_data['name'], bc_data['type'])
                for p in params:
                    pc.addItem(p['name'], p)
                pc.setEnabled(not self._is_dataset_locked())
            else:
                pc.addItem("value")
                pc.setEnabled(not self._is_dataset_locked())

        bc_combo.currentIndexChanged.connect(on_bc_changed)

        # Wire remove
        def on_remove(_, rw=row_widget, bc=bc_combo, pc=param_combo):
            self._input_rows = [(w, b, p) for w, b, p in self._input_rows if w is not rw]
            self._inputs_layout.removeWidget(rw)
            rw.deleteLater()

        remove_btn.clicked.connect(on_remove)

        self._inputs_layout.addWidget(row_widget)
        self._input_rows.append((row_widget, bc_combo, param_combo))

        # Restore values if provided (for loading saved state)
        if bc_name:
            for i in range(1, bc_combo.count()):
                data = bc_combo.itemData(i)
                if data and data['name'] == bc_name:
                    bc_combo.setCurrentIndex(i)
                    break
            if param_name:
                # Param combo gets populated by on_bc_changed signal; set after
                from PySide6.QtCore import QTimer
                def set_param():
                    idx = param_combo.findText(param_name)
                    if idx >= 0:
                        param_combo.setCurrentIndex(idx)
                QTimer.singleShot(0, set_param)

    def _restore_input_rows(self):
        """Restore input rows from model_setup.json (requires Fluent for dropdowns)."""
        if not self.project.model_setup_file.exists():
            return
        # Clear read-only labels first
        self._clear_readonly_inputs()
        try:
            with open(self.project.model_setup_file, 'r') as f:
                setup = json.load(f)
            for inp in setup.get('model_inputs', []):
                self._add_input_row(bc_name=inp['name'], param_name=inp.get('parameter', ''))
        except Exception as e:
            logger.warning(f"Error restoring input rows: {e}")

    def _restore_saved_inputs_readonly(self):
        """Show saved inputs as read-only labels (no Fluent needed)."""
        if not self.project.model_setup_file.exists():
            return
        try:
            with open(self.project.model_setup_file, 'r') as f:
                setup = json.load(f)
            inputs = setup.get('model_inputs', [])
            if not inputs:
                return
            for inp in inputs:
                label = QLabel(f"  {inp['name']}  :  {inp.get('parameter', 'value')}")
                label.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; background: transparent; padding: 6px 0;")
                label.setProperty("readonly_input", True)
                self._inputs_layout.addWidget(label)
        except Exception as e:
            logger.warning(f"Error showing saved inputs: {e}")

    def _clear_readonly_inputs(self):
        """Remove read-only input labels (before replacing with interactive rows)."""
        to_remove = []
        for i in range(self._inputs_layout.count()):
            widget = self._inputs_layout.itemAt(i).widget()
            if widget and widget.property("readonly_input"):
                to_remove.append(widget)
        for w in to_remove:
            self._inputs_layout.removeWidget(w)
            w.deleteLater()

    def _restore_saved_outputs_readonly(self):
        """Show saved outputs as read-only labels (no Fluent needed)."""
        if not self.project.output_parameters_file.exists():
            return
        try:
            with open(self.project.output_parameters_file, 'r') as f:
                out_data = json.load(f)
            outputs = out_data.get('outputs', [])
            if not outputs:
                return
            for out in outputs:
                name = out['name']
                category = out.get('category', '')
                fields = out.get('field_variables', [])

                if category == 'Report Definition':
                    label = QLabel(f"  {name}  ({category})")
                    label.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; background: transparent; padding: 6px 0;")
                    label.setProperty("readonly_output", True)
                    self._report_layout.addWidget(label)
                else:
                    fields_str = ', '.join(fields) if fields else 'none'
                    label = QLabel(f"  {name}  ({category})  :  {fields_str}")
                    label.setWordWrap(True)
                    label.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; background: transparent; padding: 6px 0;")
                    label.setProperty("readonly_output", True)
                    self._field_layout.addWidget(label)
        except Exception as e:
            logger.warning(f"Error showing saved outputs: {e}")

    def _clear_readonly_outputs(self):
        """Remove read-only output labels."""
        for container in [self._field_layout, self._report_layout]:
            to_remove = []
            for i in range(container.count()):
                widget = container.itemAt(i).widget()
                if widget and widget.property("readonly_output"):
                    to_remove.append(widget)
            for w in to_remove:
                container.removeWidget(w)
                w.deleteLater()

    # ===================================================================
    # Sub-step 3: Outputs (tabbed: Field Data | Report Definitions)
    # ===================================================================

    def _build_outputs_step(self):
        page = QFrame()
        page.setProperty("panel", True)
        outer = QVBoxLayout(page)
        outer.setContentsMargins(20, 20, 20, 20)

        self._query_outputs_btn = QPushButton("Query Fluent for Outputs")
        self._query_outputs_btn.setFixedWidth(200)
        self._query_outputs_btn.clicked.connect(self._query_outputs)
        outer.addWidget(self._query_outputs_btn)

        # Tabbed output types
        self._output_tabs = QTabWidget()

        # --- Field Data tab (surfaces + volumes) ---
        field_tab = QWidget()
        field_layout = QVBoxLayout(field_tab)
        field_layout.setContentsMargins(12, 12, 12, 12)

        field_note = QLabel("Surfaces and volumes output fields of data (e.g. temperature at every point "
                            "on a surface). Select a location and the field variables to capture.")
        field_note.setWordWrap(True)
        field_note.setProperty("secondary", True)
        field_layout.addWidget(field_note)

        field_scroll = QScrollArea()
        field_scroll.setWidgetResizable(True)
        field_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._field_container = QWidget()
        self._field_layout = QVBoxLayout(self._field_container)
        self._field_layout.setAlignment(Qt.AlignTop)
        field_scroll.setWidget(self._field_container)
        field_layout.addWidget(field_scroll, 1)

        self._output_tabs.addTab(field_tab, "Field Data")

        # --- Report Definitions tab ---
        report_tab = QWidget()
        report_layout = QVBoxLayout(report_tab)
        report_layout.setContentsMargins(12, 12, 12, 12)

        report_note = QLabel("Report definitions are scalar outputs configured in Fluent "
                             "(e.g. average temperature, mass flow rate). Add them in Fluent's "
                             "Solution > Report Definitions, then query here to include them.")
        report_note.setWordWrap(True)
        report_note.setProperty("secondary", True)
        report_layout.addWidget(report_note)

        report_scroll = QScrollArea()
        report_scroll.setWidgetResizable(True)
        report_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._report_container = QWidget()
        self._report_layout = QVBoxLayout(self._report_container)
        self._report_layout.setAlignment(Qt.AlignTop)
        report_scroll.setWidget(self._report_container)
        report_layout.addWidget(report_scroll, 1)

        self._output_tabs.addTab(report_tab, "Report Definitions")

        outer.addWidget(self._output_tabs, 1)

        # Save button
        self._save_btn = QPushButton("Save Configuration")
        self._save_btn.setFixedWidth(160)
        self._save_btn.clicked.connect(self._save_configuration)
        outer.addWidget(self._save_btn)

        return page

    def _query_outputs(self):
        fm = FluentManager.instance()
        if not fm.is_available():
            QMessageBox.warning(self, "Not Connected", "Fluent is not connected.")
            return

        items = get_available_outputs(fm.solver)
        if not items:
            QMessageBox.information(self, "No Outputs Found", "No output locations found in the Fluent case.")
            return

        # Clear existing
        self._output_rows.clear()
        self._report_rows = []
        for container in [self._field_layout, self._report_layout]:
            while container.count():
                child = container.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

        dataset_locked = self._is_dataset_locked()

        for item in items:
            if item['category'] == 'Report Definition':
                self._add_report_row(item, dataset_locked)
            else:
                self._add_field_row(item, dataset_locked)

        self._restore_output_selections()

    def _add_field_row(self, item, locked):
        """Add a Surface/Cell Zone output with field variable checkboxes."""
        group = QGroupBox(f"{item['name']}  ({item['category']})")
        group_layout = QVBoxLayout(group)

        cb = QCheckBox("Include this output")
        cb.setEnabled(not locked)
        group_layout.addWidget(cb)

        field_vars = get_available_field_variables(item['category'])
        field_checks = {}
        fields_widget = QWidget()
        fields_grid = QGridLayout(fields_widget)
        fields_grid.setContentsMargins(16, 4, 0, 4)

        col = 0
        row_idx = 0
        for category, variables in field_vars.items():
            for var in variables:
                fcb = QCheckBox(var)
                fcb.setEnabled(False)
                fields_grid.addWidget(fcb, row_idx, col)
                field_checks[var] = fcb
                col += 1
                if col >= 4:
                    col = 0
                    row_idx += 1

        fields_widget.hide()
        group_layout.addWidget(fields_widget)

        def on_output_check(checked, fw=fields_widget, fcs=field_checks):
            fw.setVisible(checked)
            if not self._is_dataset_locked():
                for fc in fcs.values():
                    fc.setEnabled(checked)

        cb.toggled.connect(on_output_check)

        self._field_layout.addWidget(group)
        self._output_rows.append((cb, item['name'], item['category'], field_checks))

    def _add_report_row(self, item, locked):
        """Add a Report Definition output (simple checkbox, no field selection)."""
        cb = QCheckBox(f"{item['name']}  ({item['type']})")
        cb.setEnabled(not locked)
        self._report_layout.addWidget(cb)
        # Store with empty field_checks — report defs have no field variable selection
        self._output_rows.append((cb, item['name'], item['category'], {}))

    def _restore_output_selections(self):
        """Restore output selections from output_parameters.json."""
        if not self.project.output_parameters_file.exists():
            return
        try:
            with open(self.project.output_parameters_file, 'r') as f:
                out_data = json.load(f)
            saved_outputs = out_data.get('outputs', [])
            saved_map = {o['name']: o for o in saved_outputs}

            for cb, name, category, field_checks in self._output_rows:
                if name in saved_map:
                    cb.setChecked(True)
                    for var in saved_map[name].get('field_variables', []):
                        if var in field_checks:
                            field_checks[var].setChecked(True)
        except Exception as e:
            logger.warning(f"Error restoring output selections: {e}")

    # ===================================================================
    # Save
    # ===================================================================

    def _save_configuration(self):
        """Save inputs to model_setup.json and outputs to output_parameters.json."""
        # Collect inputs from row dropdowns
        selected_inputs = []
        for row_widget, bc_combo, param_combo in self._input_rows:
            bc_data = bc_combo.currentData()
            param_data = param_combo.currentData()
            param_text = param_combo.currentText()
            if not bc_data or bc_combo.currentIndex() <= 0:
                continue
            if param_combo.currentIndex() <= 0:
                QMessageBox.warning(self, "Incomplete",
                                    f"Select a parameter for BC '{bc_data['name']}'.")
                return
            # Store both display name and dot-path for the runner
            param_path = param_data.get('path', param_text) if isinstance(param_data, dict) else param_text
            selected_inputs.append({
                'name': bc_data['name'],
                'type': bc_data['type'],
                'parameter': param_text,
                'parameter_path': param_path,
            })

        if not selected_inputs:
            QMessageBox.warning(self, "No Inputs", "Add at least one input parameter.")
            return

        # Collect outputs
        selected_outputs = []
        for cb, name, category, field_checks in self._output_rows:
            if cb.isChecked():
                if category == 'Report Definition':
                    # Report defs have no field variable selection
                    selected_outputs.append({
                        'name': name,
                        'category': category,
                        'field_variables': [],
                    })
                else:
                    fields = [var for var, fcb in field_checks.items() if fcb.isChecked()]
                    if not fields:
                        QMessageBox.warning(self, "No Fields",
                                            f"Select at least one field variable for output '{name}'.")
                        return
                    selected_outputs.append({
                        'name': name,
                        'category': category,
                        'field_variables': fields,
                    })

        if not selected_outputs:
            QMessageBox.warning(self, "No Outputs", "Select at least one output.")
            return

        # Save model_setup.json
        setup_data = {'model_inputs': selected_inputs}
        # Preserve existing DOE config if present
        if self.project.model_setup_file.exists():
            try:
                with open(self.project.model_setup_file, 'r') as f:
                    existing = json.load(f)
                existing['model_inputs'] = selected_inputs
                setup_data = existing
            except Exception:
                pass

        with open(self.project.model_setup_file, 'w') as f:
            json.dump(setup_data, f, indent=2)

        # Save output_parameters.json
        out_data = {'outputs': selected_outputs}
        with open(self.project.output_parameters_file, 'w') as f:
            json.dump(out_data, f, indent=2)

        logger.info(f"Setup saved: {len(selected_inputs)} inputs, {len(selected_outputs)} outputs")
        self.setup_complete.emit()

    # ===================================================================
    # Helpers
    # ===================================================================

    def _is_dataset_locked(self):
        """True when dataset directory contains sim files."""
        return self.project.dataset_dir.exists() and any(self.project.dataset_dir.glob("sim_*.npz"))

    def _load_state(self):
        case_file = self.project.get_case_file()
        if case_file:
            self._case_edit.setText(case_file)
            valid, _ = self.project.validate_case_file()
            if valid:
                self._launch_status.setText("Case file found.")
                self._launch_btn.setEnabled(True)
            else:
                self._launch_status.setText(
                    f"<span style='color:{theme.RED_ERROR}'>Case file not found. Browse to relocate.</span>"
                )

        # Restore Fluent settings from project
        solver = self.project.info.get('solver_settings', {})
        if solver:
            self._precision_combo.setCurrentText(solver.get('precision', 'single'))
            self._processors_spin.setValue(solver.get('processor_count', 2))
            self._dimension_combo.setCurrentText(str(solver.get('dimension', 3)))
            self._fluent_gui_cb.setChecked(solver.get('use_gui', True))

        if self._is_dataset_locked():
            self._lock_banner.show()

        # Restore saved inputs/outputs as read-only (no Fluent needed)
        self._restore_saved_inputs_readonly()
        self._restore_saved_outputs_readonly()

    # ===================================================================
    # Case file actions
    # ===================================================================

    def _browse_case_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Fluent Case File", "",
            "Fluent Case Files (*.cas *.cas.h5 *.cas.gz);;All Files (*)"
        )
        if not path:
            return
        self._case_edit.setText(path)
        self.project.set_case_file(path)
        self._launch_status.setText("Case file selected.")
        self._launch_btn.setEnabled(True)

    def _get_solver_settings(self):
        """Read Fluent settings from the widgets."""
        return {
            'precision': self._precision_combo.currentText(),
            'processor_count': self._processors_spin.value(),
            'dimension': int(self._dimension_combo.currentText()),
            'use_gui': self._fluent_gui_cb.isChecked(),
        }

    def _save_solver_settings(self):
        """Persist Fluent settings into project_info.json."""
        self.project.info['solver_settings'] = self._get_solver_settings()
        self.project._save_info()

    def _set_ui_locked(self, locked):
        """Disable/enable all interactive elements during Fluent launch."""
        self._launch_btn.setEnabled(not locked)
        self._browse_cas_btn.setEnabled(not locked)
        self._next_btn.setEnabled(not locked)
        self._back_btn.setEnabled(not locked)
        self._precision_combo.setEnabled(not locked)
        self._processors_spin.setEnabled(not locked)
        self._dimension_combo.setEnabled(not locked)
        self._fluent_gui_cb.setEnabled(not locked)

    def _launch_fluent(self):
        fm = FluentManager.instance()
        if fm.is_available():
            return
        if self._worker is not None:
            return  # already launching

        case_file = self.project.get_case_file()
        if not case_file or not Path(case_file).exists():
            QMessageBox.critical(self, "Error", "Case file path is invalid.")
            return

        # Save settings to project for next time
        self._save_solver_settings()
        solver_settings = self._get_solver_settings()
        log_dir = self.project.logs_dir

        fm.set_launching()
        self._set_ui_locked(True)
        self._loading.start("Launching Fluent... please wait.")

        self._worker = FluentLaunchWorker(case_file, solver_settings, log_dir)
        self._worker.finished.connect(self._on_launch_success)
        self._worker.error.connect(self._on_launch_error)
        self._worker.start()

    def _on_launch_success(self, solver):
        FluentManager.instance().set_connected(solver)
        self._loading.stop()
        self._launch_status.setText(
            f"<span style='color:{theme.GREEN_SUCCESS}'>Fluent connected.</span>"
        )
        self._launch_btn.setEnabled(False)
        self._browse_cas_btn.setEnabled(True)
        self._next_btn.setEnabled(True)
        self._back_btn.setEnabled(self._sub_stack.currentIndex() > 0)
        self._worker = None

        # Cache BCs and restore saved input rows as interactive
        self._available_inputs = get_available_inputs(FluentManager.instance().solver)
        if not self._input_rows:
            self._restore_input_rows()

        # Auto-query outputs and restore selections if previously saved
        if not self._output_rows and self.project.output_parameters_file.exists():
            self._clear_readonly_outputs()
            self._query_outputs()

        self._update_edit_state()

    def _on_launch_error(self, msg):
        FluentManager.instance().set_failed()
        self._loading.stop()
        self._set_ui_locked(False)
        self._launch_status.setText(
            f"<span style='color:{theme.RED_ERROR}'>Launch failed: {msg}</span>"
        )
        self._worker = None
        QMessageBox.critical(self, "Fluent Launch Error", msg)

    def _on_fluent_status_changed(self, status):
        """Update edit state when Fluent connects/disconnects."""
        if status == "Disconnected" and self._worker is None:
            self._launch_btn.setEnabled(bool(self.project.get_case_file()))
            self._launch_status.setText("Fluent disconnected.")
        self._update_edit_state()

    # ===================================================================
    # Navigation
    # ===================================================================

    def _go_back(self):
        idx = self._sub_stack.currentIndex()
        if idx > 0:
            self._sub_stack.setCurrentIndex(idx - 1)
        self._update_nav()

    def _go_next(self):
        idx = self._sub_stack.currentIndex()
        if idx < self._sub_stack.count() - 1:
            self._sub_stack.setCurrentIndex(idx + 1)
        self._update_nav()

    def _update_nav(self):
        idx = self._sub_stack.currentIndex()
        self._back_btn.setEnabled(idx > 0)
        self._next_btn.setVisible(idx < self._sub_stack.count() - 1)
        step_names = ["Case File", "Inputs", "Outputs"]
        self._step_label.setText(f"Step {idx + 1} of 3: {step_names[idx]}")
        self._update_edit_state()

    def _update_edit_state(self):
        """Enable/disable editing widgets based on Fluent connection."""
        connected = FluentManager.instance().is_available()
        locked = self._is_dataset_locked()
        can_edit = connected and not locked

        # Inputs step
        self._add_input_btn.setEnabled(can_edit)
        for row_widget, bc_combo, param_combo in self._input_rows:
            bc_combo.setEnabled(can_edit)
            param_combo.setEnabled(can_edit and bc_combo.currentIndex() > 0)
            # Find remove button in the row
            for child in row_widget.children():
                if isinstance(child, QPushButton):
                    child.setEnabled(can_edit)

        # Outputs step
        self._query_outputs_btn.setEnabled(connected)
        self._save_btn.setEnabled(connected)
