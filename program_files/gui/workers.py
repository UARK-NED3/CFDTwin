"""
Worker Threads Module
=====================
QThread subclasses for blocking operations.
All emit Qt signals for progress, completion, and errors.
"""

import logging
from pathlib import Path
from PySide6.QtCore import QThread, Signal

from ..modules.fluent_interface import launch_fluent
from ..modules.simulation_runner import run_remaining_simulations

logger = logging.getLogger(__name__)


class FluentLaunchWorker(QThread):
    """
    Launches Fluent in a background thread.

    Signals
    -------
    finished(object)
        Emitted with the solver object on success, or None on failure.
    error(str)
        Emitted with error message on failure.
    """

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, case_file_path, solver_settings, log_dir=None, parent=None):
        super().__init__(parent)
        self.case_file_path = case_file_path
        self.solver_settings = solver_settings
        self.log_dir = log_dir

    def run(self):
        try:
            solver = launch_fluent(
                self.case_file_path,
                self.solver_settings,
                self.log_dir,
            )
            if solver is not None:
                self.finished.emit(solver)
            else:
                self.error.emit("Fluent launch returned None. Check logs for details.")
        except Exception as e:
            logger.error(f"FluentLaunchWorker error: {e}")
            self.error.emit(str(e))


class SimulationWorker(QThread):
    """
    Runs remaining simulations in a background thread.

    Signals
    -------
    progress(int, int, int, str)
        (current_index, total, sim_id, status)
    finished(dict)
        Summary dict from run_remaining_simulations.
    error(str)
        Error message on failure.
    """

    progress = Signal(int, int, int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, solver, setup_data, dataset_dir, reinitialize=True,
                 doe_samples=None, parent=None):
        super().__init__(parent)
        self.solver = solver
        self.setup_data = setup_data
        self.dataset_dir = Path(dataset_dir)
        self.reinitialize = reinitialize
        self.doe_samples = doe_samples
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def run(self):
        try:
            def on_progress(idx, total, sim_id, status):
                self.progress.emit(idx, total, sim_id, status)

            summary = run_remaining_simulations(
                solver=self.solver,
                setup_data=self.setup_data,
                dataset_dir=self.dataset_dir,
                on_progress=on_progress,
                stop_flag=lambda: self.stop_requested,
                reinitialize=self.reinitialize,
                doe_samples=self.doe_samples,
            )
            self.finished.emit(summary)
        except Exception as e:
            logger.error(f"SimulationWorker error: {e}")
            self.error.emit(str(e))


class TrainingWorker(QThread):
    """
    Runs model training in a background thread.

    Signals
    -------
    epoch_update(str, int, float, float)
        (model_name, epoch, train_loss, val_loss)
    model_started(str)
        Name of model that just started training.
    finished(dict)
        Training summary dict.
    error(str)
    """

    epoch_update = Signal(str, int, float, float)
    model_started = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, project_dir, model_name, model_selection,
                 test_size=0.2, epochs=500, exclude_range=None,
                 output_filter=None, parent=None):
        super().__init__(parent)
        self.project_dir = Path(project_dir)
        self.model_name = model_name
        self.model_selection = model_selection
        self.test_size = test_size
        self.epochs = epochs
        self.exclude_range = exclude_range
        self.output_filter = output_filter

    def run(self):
        try:
            import time
            import matplotlib
            matplotlib.use('Agg')
            from ..modules.multi_model_trainer import train_all_models

            def on_progress(name, status):
                if status == 'training':
                    self.model_started.emit(name)

            # Throttle epoch updates to ~1 Hz to keep the UI responsive.
            # Always emit using the worker's user-chosen model_name so the train page
            # can route updates to the correct live canvas.
            # First emit fires immediately (last_emit starts far in the past).
            last_emit = [-1e9]
            last_sent_epoch = [0]

            def on_epoch(name, epoch, train_loss, val_loss):
                now = time.monotonic()
                # Always emit the final epoch, plus any epoch at least 1s after the last emit
                if now - last_emit[0] >= 1.0:
                    # -1 sentinel means "no validation data for this run"
                    val_to_send = float(val_loss) if val_loss is not None else -1.0
                    self.epoch_update.emit(
                        self.model_name, epoch, float(train_loss), val_to_send,
                    )
                    last_emit[0] = now
                    last_sent_epoch[0] = epoch

            summary = train_all_models(
                dataset_dir=self.project_dir,
                model_name=self.model_name,
                model_selection=self.model_selection,
                test_size=self.test_size,
                epochs=self.epochs,
                exclude_range=self.exclude_range,
                on_progress=on_progress,
                output_filter=self.output_filter,
                on_epoch=on_epoch,
            )
            self.finished.emit(summary)
        except Exception as e:
            logger.error(f"TrainingWorker error: {e}")
            self.error.emit(str(e))


class ValidationWorker(QThread):
    """
    Runs a single Fluent validation simulation.

    Signals
    -------
    finished(dict)
        Results dict with Fluent output data.
    error(str)
    """

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, solver, setup_data, dataset_dir, parameters, parent=None):
        super().__init__(parent)
        self.solver = solver
        self.setup_data = setup_data
        self.dataset_dir = Path(dataset_dir)
        self.parameters = parameters

    def run(self):
        try:
            from ..modules.multi_model_visualizer import run_fluent_comparison

            results = run_fluent_comparison(
                solver=self.solver,
                setup_data=self.setup_data,
                dataset_dir=self.dataset_dir,
                params=self.parameters,
            )
            if results is None:
                self.error.emit("Fluent validation returned no data.")
                return
            self.finished.emit(results)
        except Exception as e:
            logger.error(f"ValidationWorker error: {e}")
            self.error.emit(str(e))
