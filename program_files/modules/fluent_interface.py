"""
Fluent Interface Module
=======================
Handles launching PyFluent and loading case files.
All functions are GUI-agnostic.
"""

import logging
import sys
from pathlib import Path
from contextlib import contextmanager
from io import StringIO

logger = logging.getLogger(__name__)


@contextmanager
def _redirect_to_file(log_file):
    """Context manager to redirect stdout/stderr to a log file."""
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = log_file, log_file
    try:
        yield
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


def launch_fluent(case_file_path, solver_settings, log_dir=None):
    """
    Launch Fluent and load a case file.

    Parameters
    ----------
    case_file_path : str or Path
        Path to Fluent case file (.cas, .cas.h5, .cas.gz)
    solver_settings : dict
        Keys: precision, processor_count, dimension, use_gui
    log_dir : Path, optional
        Directory for Fluent log files. If None, logs are discarded.

    Returns
    -------
    solver
        PyFluent solver object, or None on failure
    """
    case_file_path = Path(case_file_path)
    if not case_file_path.exists():
        logger.error(f"Case file not found: {case_file_path}")
        return None

    log_file = None
    try:
        import ansys.fluent.core as pyfluent
        from ansys.fluent.core.launcher.launcher import UIMode

        # Set up logging
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True)
            log_file_path = log_dir / f"fluent_launch_{case_file_path.stem}.log"
            log_file = open(log_file_path, 'w', buffering=1)
            logger.info(f"Fluent output redirected to: {log_file_path}")
        else:
            log_file = StringIO()

        ui_mode = UIMode.GUI if solver_settings.get('use_gui', False) else UIMode.NO_GUI_OR_GRAPHICS

        logger.info(f"Launching Fluent (precision={solver_settings['precision']}, "
                     f"processors={solver_settings['processor_count']}, "
                     f"dim={solver_settings['dimension']}D)")

        with _redirect_to_file(log_file):
            solver = pyfluent.launch_fluent(
                precision=solver_settings['precision'],
                processor_count=solver_settings['processor_count'],
                dimension=solver_settings['dimension'],
                mode="solver",
                ui_mode=ui_mode
            )

        logger.info(f"Fluent launched (version {solver.get_fluent_version()})")
        logger.info(f"Loading case: {case_file_path.name}")

        with _redirect_to_file(log_file):
            solver.settings.file.read_case(file_name=str(case_file_path))

        logger.info("Case file loaded successfully")

        if hasattr(log_file, 'name'):
            log_file.close()

        solver._case_file_path = str(case_file_path)
        return solver

    except Exception as e:
        error_str = str(e).lower()
        if 'no module named' in error_str and 'ansys' in error_str:
            logger.error("PyFluent is not installed. Run: pip install ansys-fluent-core")
            raise RuntimeError("PyFluent is not installed. Run: pip install ansys-fluent-core") from e
        elif any(kw in error_str for kw in ['connection refused', 'connect', 'unavailable', '10061']):
            logger.error("CONNECTION ERROR: Check that VPN is enabled and license server is reachable")
        else:
            logger.error(f"Error launching Fluent: {e}")

        try:
            if log_file is not None and hasattr(log_file, 'close'):
                log_file.close()
        except Exception:
            pass

        return None
