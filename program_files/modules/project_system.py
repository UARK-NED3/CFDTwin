"""
Project System Module
=====================
Handles project creation, opening, scanning, and management.
"""

import json
from pathlib import Path
from datetime import datetime


class WorkflowProject:
    """
    Represents a Workflow Surrogate project.

    Project Structure:
    ------------------
    project_folder/
    ├── project_info.json          # Project metadata
    └── cases/                     # Simulation cases (DOE configurations + models)
        ├── case_name_1/
        │   ├── model_setup.json
        │   ├── output_parameters.json
        │   ├── dataset/           # Simulation output data
        │   │   └── sim_*.npz
        │   └── models/            # Trained models for this case
        │       ├── 3D_velocity_1.h5
        │       ├── 2D_pressure_1.h5
        │       └── ...
        └── case_name_2/
    """

    def __init__(self, project_path):
        """
        Initialize project.

        Parameters
        ----------
        project_path : Path
            Path to project folder
        """
        self.project_path = Path(project_path)
        self.project_info_file = self.project_path / "project_info.json"
        self.cases_dir = self.project_path / "cases"

        self.info = None
        self.cases = []

    def create(self, project_name):
        """
        Create a new project.

        Parameters
        ----------
        project_name : str
            Name of the project

        Returns
        -------
        bool
            True if successful
        """
        try:
            # Create directories
            self.project_path.mkdir(parents=True, exist_ok=True)
            self.cases_dir.mkdir(exist_ok=True)

            # Create project info
            self.info = {
                'project_name': project_name,
                'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'last_opened': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': '1.0'
            }

            # Save project info
            with open(self.project_info_file, 'w') as f:
                json.dump(self.info, f, indent=2)

            return True

        except Exception as e:
            print(f"Error creating project: {e}")
            return False

    def load(self):
        """
        Load an existing project.

        Returns
        -------
        bool
            True if successful
        """
        try:
            if not self.project_info_file.exists():
                return False

            # Load project info
            with open(self.project_info_file, 'r') as f:
                self.info = json.load(f)

            # Update last opened
            self.info['last_opened'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.project_info_file, 'w') as f:
                json.dump(self.info, f, indent=2)

            # Scan for datasets and models
            self.scan()

            return True

        except Exception as e:
            print(f"Error loading project: {e}")
            return False

    def scan(self):
        """
        Scan project for cases.
        """
        self.cases = []

        # Scan cases
        if self.cases_dir.exists():
            for case_dir in self.cases_dir.iterdir():
                if case_dir.is_dir():
                    case_info = self._scan_case(case_dir)
                    if case_info:
                        self.cases.append(case_info)

    def _scan_case(self, case_dir):
        """
        Scan a case directory.

        Parameters
        ----------
        case_dir : Path
            Case directory

        Returns
        -------
        dict or None
            Case information
        """
        setup_file = case_dir / "model_setup.json"

        if not setup_file.exists():
            return None

        try:
            with open(setup_file, 'r') as f:
                setup_data = json.load(f)

            # Count simulation files
            dataset_dir = case_dir / "dataset"
            num_sims = 0
            if dataset_dir.exists():
                num_sims = len(list(dataset_dir.glob("sim_*.npz")))

            # Count total required
            from modules import doe_setup as doe
            analysis = doe.analyze_setup_dimensions(setup_data)
            total_required = analysis['total_input_combinations']

            completeness = (num_sims / total_required * 100) if total_required > 0 else 0

            # Count trained models
            models_dir = case_dir / "models"
            num_models = 0
            if models_dir.exists():
                num_models = len(list(models_dir.glob("*.h5"))) + len(list(models_dir.glob("*.npz")))

            return {
                'name': case_dir.name,
                'path': case_dir,
                'setup_file': setup_file,
                'num_inputs': len(setup_data.get('model_inputs', [])),
                'num_outputs': len(setup_data.get('model_outputs', [])),
                'num_simulations': num_sims,
                'total_required': total_required,
                'completeness': completeness,
                'num_models': num_models,
                'created': setup_data.get('timestamp', 'Unknown')
            }

        except Exception as e:
            print(f"Warning: Error scanning case {case_dir.name}: {e}")
            return None

    def get_case(self, case_name):
        """
        Get case by name.

        Parameters
        ----------
        case_name : str
            Case name

        Returns
        -------
        dict or None
            Case information
        """
        for case in self.cases:
            if case['name'] == case_name:
                return case
        return None

    def delete_case(self, case_name):
        """
        Delete a case.

        Parameters
        ----------
        case_name : str
            Case name

        Returns
        -------
        bool
            True if successful
        """
        case = self.get_case(case_name)
        if not case:
            return False

        try:
            import shutil
            import time

            # On Windows, sometimes files are locked. Try with onerror handler
            def handle_remove_readonly(func, path, exc_info):
                """Error handler for Windows readonly files."""
                import os
                import stat
                # Try to remove readonly flag and retry
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except:
                    pass

            # First attempt with error handler
            try:
                shutil.rmtree(case['path'], onerror=handle_remove_readonly)
            except PermissionError:
                # If still fails, try manual deletion with retry
                print("\n⚠ Permission error. Attempting to close any open files...")
                time.sleep(1)  # Give OS time to release handles

                # Try again
                shutil.rmtree(case['path'], onerror=handle_remove_readonly)

            self.scan()  # Refresh
            return True
        except Exception as e:
            print(f"Error deleting case: {e}")
            print("\nTroubleshooting tips:")
            print("  1. Close any programs that might have files open (Excel, editors, etc.)")
            print("  2. Close the Fluent case if it's still running")
            print("  3. Try deleting the folder manually in File Explorer")
            return False


def create_new_project(ui_helpers):
    """
    Create a new project interactively.

    Parameters
    ----------
    ui_helpers : module
        UI helpers module

    Returns
    -------
    WorkflowProject or None
        Created project
    """
    from tkinter import Tk, filedialog

    ui_helpers.clear_screen()
    ui_helpers.print_header("CREATE NEW PROJECT")

    # Get project name
    project_name = input("\nEnter project name: ").strip()
    if not project_name:
        print("\n[X] Project name cannot be empty")
        ui_helpers.pause()
        return None

    # Select parent directory
    print("\nSelect parent directory for project...")
    Tk().withdraw()

    parent_dir = filedialog.askdirectory(
        title="Select Parent Directory for Project"
    )

    if not parent_dir:
        print("\n[X] No directory selected")
        ui_helpers.pause()
        return None

    # Create project folder
    project_folder = Path(parent_dir) / project_name

    if project_folder.exists():
        overwrite = input(f"\n[WARNING] Folder '{project_name}' already exists. Overwrite? [y/N]: ").strip().lower()
        if overwrite != 'y':
            print("\n[X] Project creation cancelled")
            ui_helpers.pause()
            return None

    # Create project
    project = WorkflowProject(project_folder)

    if project.create(project_name):
        print(f"\n✓ Project created successfully!")
        print(f"  Location: {project_folder}")
        ui_helpers.pause()
        return project
    else:
        print(f"\n✗ Failed to create project")
        ui_helpers.pause()
        return None


def open_existing_project(ui_helpers):
    """
    Open an existing project from file explorer.

    Parameters
    ----------
    ui_helpers : module
        UI helpers module

    Returns
    -------
    WorkflowProject or None
        Opened project
    """
    from tkinter import Tk, filedialog

    ui_helpers.clear_screen()
    ui_helpers.print_header("OPEN EXISTING PROJECT")

    print("\nSelect project folder...")
    Tk().withdraw()

    project_folder = filedialog.askdirectory(
        title="Select Project Folder"
    )

    if not project_folder:
        print("\n✗ No folder selected")
        ui_helpers.pause()
        return None

    project_folder = Path(project_folder)

    # Check if it's a valid project
    project = WorkflowProject(project_folder)

    if project.load():
        print(f"\n✓ Project opened successfully!")
        print(f"  Name: {project.info['project_name']}")
        print(f"  Location: {project_folder}")
        print(f"  Created: {project.info['created']}")
        ui_helpers.pause()
        return project
    else:
        print(f"\n✗ Invalid project folder (project_info.json not found)")
        ui_helpers.pause()
        return None


def open_recent_project(project_path, ui_helpers):
    """
    Open a recent project.

    Parameters
    ----------
    project_path : Path or str
        Project path
    ui_helpers : module
        UI helpers module

    Returns
    -------
    WorkflowProject or None
        Opened project
    """
    project = WorkflowProject(project_path)

    if project.load():
        return project
    else:
        print(f"\n✗ Could not open project: {project_path}")
        ui_helpers.pause()
        return None
