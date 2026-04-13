"""
Project System Module
=====================
Handles project creation, opening, scanning, and management.
All functions are GUI-agnostic.
"""

import logging
import json
import shutil
import time
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class WorkflowProject:
    """
    Represents a Workflow Surrogate project.

    Flat project structure (no cases/ nesting):
    --------------------------------------------
    project_folder/
    +-- project_info.json
    +-- model_setup.json
    +-- output_parameters.json
    +-- doe_samples.json
    +-- dataset/
    |   +-- coordinates.npz
    |   +-- dataset_version.json
    |   +-- sim_0001.npz
    +-- models/
    |   +-- outlet_temp_v3/
    |       +-- *_nn.h5, *_nn.npz, *_pod.npz, *_metadata.json
    +-- loss_curves/
    +-- logs/
        +-- residuals/
    """

    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.project_info_file = self.project_path / "project_info.json"
        self.info = None

    # --- Paths ---

    @property
    def model_setup_file(self):
        return self.project_path / "model_setup.json"

    @property
    def output_parameters_file(self):
        return self.project_path / "output_parameters.json"

    @property
    def doe_samples_file(self):
        return self.project_path / "doe_samples.json"

    @property
    def dataset_dir(self):
        return self.project_path / "dataset"

    @property
    def models_dir(self):
        return self.project_path / "models"

    @property
    def loss_curves_dir(self):
        return self.project_path / "loss_curves"

    @property
    def logs_dir(self):
        return self.project_path / "logs"

    @property
    def residuals_dir(self):
        return self.logs_dir / "residuals"

    @property
    def fluent_cache_dir(self):
        return self.project_path / ".cache" / "fluent_validation"

    @property
    def fluent_cache_index_file(self):
        return self.fluent_cache_dir / "index.json"

    # --- Lifecycle ---

    def create(self, project_name):
        """Create a new project. Returns True on success."""
        try:
            self.project_path.mkdir(parents=True, exist_ok=True)
            self.dataset_dir.mkdir(exist_ok=True)
            self.models_dir.mkdir(exist_ok=True)
            self.loss_curves_dir.mkdir(exist_ok=True)
            self.logs_dir.mkdir(exist_ok=True)
            self.residuals_dir.mkdir(exist_ok=True)

            self.info = {
                'project_name': project_name,
                'case_file': None,
                'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'last_opened': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': '2.0'
            }

            with open(self.project_info_file, 'w') as f:
                json.dump(self.info, f, indent=2)

            logger.info(f"Project created: {project_name} at {self.project_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return False

    def load(self):
        """Load an existing project. Returns True on success."""
        try:
            if not self.project_info_file.exists():
                return False

            with open(self.project_info_file, 'r') as f:
                self.info = json.load(f)

            self.info['last_opened'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.project_info_file, 'w') as f:
                json.dump(self.info, f, indent=2)

            # Ensure directories exist (in case of manual deletion)
            self.dataset_dir.mkdir(exist_ok=True)
            self.models_dir.mkdir(exist_ok=True)
            self.loss_curves_dir.mkdir(exist_ok=True)
            self.logs_dir.mkdir(exist_ok=True)
            self.residuals_dir.mkdir(exist_ok=True)

            logger.info(f"Project loaded: {self.info['project_name']}")
            return True
        except Exception as e:
            logger.error(f"Error loading project: {e}")
            return False

    # --- State scanning ---

    def get_project_state(self):
        """
        Scan project files to determine workflow state.

        Returns dict with booleans for each milestone:
            has_case_file, has_inputs, has_outputs, has_doe,
            has_simulations, has_models, sim_count, model_count
        """
        state = {
            'has_case_file': False,
            'has_inputs': False,
            'has_outputs': False,
            'has_doe': False,
            'has_simulations': False,
            'has_models': False,
            'sim_count': 0,
            'model_count': 0,
        }

        # Case file
        case_file = self.info.get('case_file') if self.info else None
        state['has_case_file'] = case_file is not None and Path(case_file).exists()

        # Inputs / outputs
        if self.model_setup_file.exists():
            try:
                with open(self.model_setup_file, 'r') as f:
                    setup = json.load(f)
                state['has_inputs'] = len(setup.get('model_inputs', [])) > 0
            except Exception:
                pass

        if self.output_parameters_file.exists():
            try:
                with open(self.output_parameters_file, 'r') as f:
                    out_params = json.load(f)
                state['has_outputs'] = len(out_params.get('outputs', [])) > 0
            except Exception:
                pass

        # DOE
        if self.doe_samples_file.exists():
            try:
                with open(self.doe_samples_file, 'r') as f:
                    doe = json.load(f)
                state['has_doe'] = len(doe.get('samples', [])) > 0
            except Exception:
                pass

        # Simulations
        if self.dataset_dir.exists():
            sim_files = list(self.dataset_dir.glob("sim_*.npz"))
            state['sim_count'] = len(sim_files)
            state['has_simulations'] = state['sim_count'] > 0

        # Models
        if self.models_dir.exists():
            model_dirs = [
                d for d in self.models_dir.iterdir()
                if d.is_dir() and list(d.glob("*_metadata.json"))
            ]
            state['model_count'] = len(model_dirs)
            state['has_models'] = state['model_count'] > 0

        return state

    # --- Case file ---

    def set_case_file(self, case_file_path):
        """Store the .cas file reference path."""
        self.info['case_file'] = str(case_file_path)
        self._save_info()

    def get_case_file(self):
        """Get stored .cas file path, or None."""
        path = self.info.get('case_file') if self.info else None
        return path

    def validate_case_file(self):
        """Check if stored .cas path still exists. Returns (valid, path)."""
        path = self.get_case_file()
        if path is None:
            return False, None
        return Path(path).exists(), path

    # --- Delete operations ---

    def delete_dataset(self):
        """Delete all simulation files and reset dataset. Returns True on success."""
        try:
            if self.dataset_dir.exists():
                for f in self.dataset_dir.glob("sim_*.npz"):
                    f.unlink()
                coords = self.dataset_dir / "coordinates.npz"
                if coords.exists():
                    coords.unlink()
                version_file = self.dataset_dir / "dataset_version.json"
                if version_file.exists():
                    version_file.unlink()

            # Remove DOE samples too since they're tied to the dataset
            if self.doe_samples_file.exists():
                self.doe_samples_file.unlink()

            # Remove residuals
            if self.residuals_dir.exists():
                for f in self.residuals_dir.glob("*.npz"):
                    f.unlink()

            logger.info("Dataset deleted")
            return True
        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            return False

    def delete_model(self, model_dir_name):
        """Delete a single trained model directory. Returns True on success."""
        model_path = self.models_dir / model_dir_name
        if not model_path.exists():
            return False

        try:
            def handle_remove_readonly(func, path, exc_info):
                import os, stat
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass

            shutil.rmtree(model_path, onerror=handle_remove_readonly)
            logger.info(f"Deleted model: {model_dir_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model {model_dir_name}: {e}")
            return False

    def delete_all_models(self):
        """Delete all trained models. Returns True on success."""
        try:
            if self.models_dir.exists():
                for d in self.models_dir.iterdir():
                    if d.is_dir():
                        shutil.rmtree(d)
            logger.info("All models deleted")
            return True
        except Exception as e:
            logger.error(f"Error deleting all models: {e}")
            return False

    # --- Internal ---

    def _save_info(self):
        """Persist project_info.json."""
        with open(self.project_info_file, 'w') as f:
            json.dump(self.info, f, indent=2)


def create_project(project_path, project_name):
    """
    Create a new project.

    Parameters
    ----------
    project_path : Path or str
        Full path to project folder
    project_name : str
        Display name for the project

    Returns
    -------
    WorkflowProject or None
    """
    project = WorkflowProject(project_path)
    if project.create(project_name):
        return project
    return None


def open_project(project_path):
    """
    Open an existing project.

    Parameters
    ----------
    project_path : Path or str
        Path to project folder

    Returns
    -------
    WorkflowProject or None
    """
    project = WorkflowProject(project_path)
    if project.load():
        return project
    return None
