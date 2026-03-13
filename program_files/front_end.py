#!/usr/bin/env python
"""
Workflow Surrogate - Interactive Fluent Integration
====================================================
Complete workflow for neural network surrogate model creation from CFD data.
"""

import sys
import json
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

# Import modules
from modules import ui_helpers
from modules import user_settings as us
from modules import fluent_interface as fi
from modules import project_manager as pm
from modules import output_parameters as op
from modules import simulation_runner as sr
from modules import multi_model_trainer as mt
from modules import multi_model_visualizer as dv
from modules import project_system as ps

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).parent
CONFIG_FILE = PROJECT_DIR / "user_settings.json"

# Initialize user settings
user_settings = us.UserSettings(CONFIG_FILE)


# ============================================================
# PROJECT OPENING MENU
# ============================================================

def project_opening_menu():
    """
    Menu for opening or creating projects.
    Returns the opened/created project.
    """
    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("WORKFLOW SURROGATE - PROJECT SELECTION")

        # Show recent project folders only
        recent_project_folders = user_settings.get_recent_project_folders()

        print("\n  [1] Create New Project")
        print("  [2] Open Existing Project (Browse)")

        if recent_project_folders:
            print("\n  Recent Projects:")
            for i, project_path in enumerate(recent_project_folders[:5], 3):  # Show max 5
                project_path = Path(project_path)
                print(f"  [{i}] {project_path.name}")

        print(f"\n  [0] Exit")
        print("="*70)

        max_choice = 2 + min(len(recent_project_folders), 5)
        choice = ui_helpers.get_choice(max_choice)

        if choice == 0:
            print("\nGoodbye!")
            sys.exit(0)

        elif choice == 1:
            # Create new project
            project = ps.create_new_project(ui_helpers)
            if project:
                # Add to recent project folders
                user_settings.add_recent_project_folder(project.project_path)
                return project

        elif choice == 2:
            # Open existing project
            project = ps.open_existing_project(ui_helpers)
            if project:
                # Add to recent project folders
                user_settings.add_recent_project_folder(project.project_path)
                return project

        elif 3 <= choice <= max_choice:
            # Open recent project
            project_idx = choice - 3
            project_path = recent_project_folders[project_idx]
            project = ps.open_recent_project(project_path, ui_helpers)
            if project:
                # Move to top of recent project folders
                user_settings.add_recent_project_folder(project.project_path)
                return project


# ============================================================
# PROJECT MAIN MENU
# ============================================================

def project_main_menu(project):
    """
    Main menu for an opened project.
    Displays project status and provides access to all workflows.
    """
    while True:
        # Refresh project scan
        project.scan()

        ui_helpers.clear_screen()
        ui_helpers.print_header(f"PROJECT: {project.info['project_name']}")

        # Display project status bar
        print("\n" + "="*70)
        print("PROJECT STATUS:")
        print("="*70)
        print(f"  Location: {project.project_path}")
        print(f"  Created: {project.info['created']}")
        print(f"\n  Cases: {len(project.cases)}")
        if project.cases:
            for case in project.cases:
                status = f"{case['completeness']:.0f}% complete"
                num_models = case.get('num_models', 0)
                models_str = f", {num_models} models" if num_models > 0 else ""
                print(f"    - {case['name']:30s} [{status}{models_str}]")
        else:
            print("    (none)")

        print("\n" + "="*70)
        print("  [1] Test I/O & Simulations")
        print("      (Configure Fluent I/O, run simulation setups)")
        print("\n  [2] Model Setup & Training")
        print("      (Select dataset, configure, train model)")
        print("\n  [3] Data Visualization")
        print("      (View training curves, comparisons, metrics)")
        print("\n  [4] Manage Project Data")
        print("      (Delete datasets or models)")
        print(f"\n  [0] Close Project & Return to Project Selection")
        print("="*70)

        choice = ui_helpers.get_choice(4)

        if choice == 0:
            return

        elif choice == 1:
            test_io_and_simulations_menu(project)

        elif choice == 2:
            model_setup_and_training_menu(project)

        elif choice == 3:
            data_visualization_model_select_menu(project, ui_helpers)

        elif choice == 4:
            manage_project_data_menu(project)


# ============================================================
# TEST I/O & SIMULATIONS MENU
# ============================================================

def test_io_and_simulations_menu(project):
    """
    Menu for configuring I/O and running simulations.
    Saves datasets as named folders within the project.
    """
    solver = None

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("TEST I/O & SIMULATIONS")

        # Show status
        if solver:
            print("\n✓ Fluent session active")
        else:
            print("\n○ No Fluent session")

        print(f"\nProject: {project.info['project_name']}")
        print(f"Cases: {len(project.cases)}")

        print(f"\n{'='*70}")
        print("  [1] Open Fluent Case File")
        print("  [2] Configure New Simulation Setup")
        print("  [3] Edit Existing Dataset I/O")
        print("  [4] Run Simulations for Dataset")
        print("  [0] Back to Main Menu")
        print("="*70)

        choice = ui_helpers.get_choice(4)

        if choice == 0:
            # Close Fluent if open
            if solver:
                try:
                    print("\nClosing Fluent session...")
                    solver.exit()
                    print("✓ Fluent closed")
                except:
                    pass
            return

        elif choice == 1:
            # Open Fluent
            new_solver = open_fluent_case()
            if new_solver:
                if solver:  # Close previous
                    try:
                        solver.exit()
                    except:
                        pass
                solver = new_solver

        elif choice == 2:
            # Configure new simulation setup
            if not solver:
                print("\n✗ Please open Fluent first (Option 1)")
                ui_helpers.pause()
            else:
                configure_new_dataset(project, solver)

        elif choice == 3:
            # Edit existing dataset I/O
            if not solver:
                print("\n✗ Please open Fluent first (Option 1)")
                ui_helpers.pause()
            else:
                edit_existing_dataset(project, solver)

        elif choice == 4:
            # Run simulations for dataset
            run_simulations_for_dataset(project, solver)


def open_fluent_case():
    """Open Fluent case file."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("OPEN FLUENT CASE FILE")

    # Show recent case files only
    recent_case_files = user_settings.get_recent_case_files()

    print("\n  [1] Browse for Case File")

    if recent_case_files:
        print("\n  Recent Case Files:")
        for i, case_path in enumerate(recent_case_files[:5], 2):
            case_path = Path(case_path)
            parent_dir = case_path.parent.name
            print(f"  [{i}] {case_path.name} ({parent_dir})")

    print(f"\n  [0] Cancel")
    print("="*70)

    max_choice = 1 + min(len(recent_case_files), 5)
    choice = ui_helpers.get_choice(max_choice)

    if choice == 0:
        return None
    elif choice == 1:
        return fi.open_case_file(user_settings, PROJECT_DIR, ui_helpers)
    elif 2 <= choice <= max_choice:
        case_idx = choice - 2
        case_path = recent_case_files[case_idx]
        return fi.open_recent_project(case_path, user_settings, PROJECT_DIR, ui_helpers)


def configure_new_dataset(project, solver):
    """Configure a new simulation setup with I/O setup."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("CONFIGURE NEW SIMULATION SETUP")

    # Get dataset name
    dataset_name = input("\nEnter dataset name: ").strip()
    if not dataset_name:
        print("\n✗ Dataset name cannot be empty")
        ui_helpers.pause()
        return

    # Check if case already exists
    dataset_dir = project.cases_dir / dataset_name
    if dataset_dir.exists():
        overwrite = input(f"\n⚠ Dataset '{dataset_name}' already exists. Overwrite? [y/N]: ").strip().lower()
        if overwrite != 'y':
            print("\n✗ Dataset configuration cancelled")
            ui_helpers.pause()
            return

    # Run I/O setup
    selected_inputs = []
    selected_outputs = []
    output_params = {}

    result = input_output_setup_menu(solver, selected_inputs, selected_outputs, output_params)

    if not result:
        print("\n✗ I/O setup not completed")
        ui_helpers.pause()
        return

    solver, selected_inputs, selected_outputs, output_params, setup_data, analysis, _ = result

    if not setup_data:
        print("\n✗ Setup data not saved")
        ui_helpers.pause()
        return

    # Create dataset directory
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save setup file
    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'w') as f:
        json.dump(setup_data, f, indent=2)

    # Create dataset structure
    from modules import doe_setup as doe
    doe.create_dataset_structure(dataset_dir, analysis, setup_data, ui_helpers)

    # Save output parameters
    output_params_file = dataset_dir / "output_parameters.json"
    with open(output_params_file, 'w') as f:
        json.dump(output_params, f, indent=2)

    print(f"\n✓ Dataset '{dataset_name}' configured successfully!")
    ui_helpers.pause()

    # Refresh project
    project.scan()


def edit_existing_dataset(project, solver):
    """Edit I/O configuration of an existing dataset."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("EDIT EXISTING DATASET I/O")

    if not project.cases:
        print("\n✗ No datasets found in project")
        ui_helpers.pause()
        return

    # List datasets
    print("\nAvailable Datasets:")
    for i, dataset in enumerate(project.cases, 1):
        print(f"  [{i}] {dataset['name']}")

    print(f"\n  [0] Cancel")
    print("="*70)

    choice = ui_helpers.get_choice(len(project.cases))

    if choice == 0:
        return

    dataset = project.cases[choice - 1]
    dataset_dir = dataset['path']
    setup_file = dataset['setup_file']

    # Load existing setup
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    selected_inputs = setup_data['model_inputs']
    selected_outputs = setup_data['model_outputs']

    # Load DOE parameters and attach to inputs
    doe_config = setup_data.get('doe_configuration', {})
    for input_item in selected_inputs:
        input_item['doe_parameters'] = doe_config.get(input_item['name'], {})

    # Load output parameters
    output_params = {}
    output_params_file = dataset_dir / "output_parameters.json"
    if output_params_file.exists():
        with open(output_params_file, 'r') as f:
            output_params = json.load(f)

    # Run I/O setup menu
    result = input_output_setup_menu(solver, selected_inputs, selected_outputs, output_params)

    if result:
        solver, selected_inputs, selected_outputs, output_params, setup_data, analysis, _ = result

        if setup_data:
            # Save updated setup
            with open(setup_file, 'w') as f:
                json.dump(setup_data, f, indent=2)

            # Save output parameters
            with open(output_params_file, 'w') as f:
                json.dump(output_params, f, indent=2)

            print(f"\n✓ Dataset '{dataset['name']}' updated successfully!")
            ui_helpers.pause()

            # Refresh project
            project.scan()


def run_simulations_for_dataset(project, solver):
    """Run simulations for a selected dataset."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("RUN SIMULATIONS FOR DATASET")

    if not project.cases:
        print("\n✗ No datasets found in project")
        ui_helpers.pause()
        return

    # List datasets
    print("\nAvailable Datasets:")
    for i, dataset in enumerate(project.cases, 1):
        status = f"{dataset['completeness']:.0f}% complete"
        print(f"  [{i}] {dataset['name']:30s} [{status}] ({dataset['num_simulations']}/{dataset['total_required']})")

    print(f"\n  [0] Cancel")
    print("="*70)

    choice = ui_helpers.get_choice(len(project.cases))

    if choice == 0:
        return

    dataset = project.cases[choice - 1]
    dataset_dir = dataset['path']
    setup_file = dataset['setup_file']

    # Load setup
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    from modules import doe_setup as doe
    analysis = doe.analyze_setup_dimensions(setup_data)

    # Check if Fluent is connected
    if not solver:
        print("\n✗ Please open Fluent first")
        ui_helpers.pause()
        return

    # Run simulations menu
    sr.run_simulations_menu(solver, setup_data, analysis, dataset_dir, ui_helpers)

    # Refresh project
    project.scan()


def input_output_setup_menu(solver, selected_inputs, selected_outputs, output_params):
    """
    Submenu for input/output configuration.
    """
    setup_data = None
    analysis = None

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("INPUT/OUTPUT SETUP")

        if solver:
            print("\n✓ Fluent session active")
        else:
            print("\n○ No Fluent session - please open case file")

        print(f"\nInputs Selected: {len(selected_inputs)}")
        print(f"Outputs Selected: {len(selected_outputs)}")

        # Count configured output parameters
        num_configured = sum(1 for params in output_params.values() if params)
        print(f"Output Parameters Configured: {num_configured}")

        # Count DOE configured inputs
        num_doe_configured = sum(
            1 for input_item in selected_inputs
            if 'doe_parameters' in input_item and any(
                values for values in input_item.get('doe_parameters', {}).values()
            )
        )
        print(f"DOE Configured: {num_doe_configured}/{len(selected_inputs)} inputs")

        print(f"\n{'='*70}")
        print("  [1] Configure Model Inputs (BCs & Zones)")
        print("  [2] Configure Model Outputs (Surfaces & Zones)")
        print("  [3] Configure Output Parameters (Temp, Pressure, etc.)")
        print("  [4] Design of Experiment Setup")
        print("  [5] Add Latin Hypercube Sample Points")
        print("  [6] Save Setup & Finish")
        print("  [0] Back (Discard Changes)")
        print("="*70)

        choice = ui_helpers.get_choice(6)

        if choice == 0:
            return None

        elif choice == 1:
            selected_inputs = pm.setup_model_inputs(solver, selected_inputs, ui_helpers)

        elif choice == 2:
            selected_outputs = pm.setup_model_outputs(solver, selected_outputs, ui_helpers)

        elif choice == 3:
            if not selected_outputs:
                print("\n✗ Please configure outputs first (Option 2)")
                ui_helpers.pause()
            else:
                output_params = op.setup_output_parameters(selected_outputs, output_params, ui_helpers)

        elif choice == 4:
            if not selected_inputs:
                print("\n✗ Please configure inputs first (Option 1)")
                ui_helpers.pause()
            else:
                from modules.doe_setup import setup_doe

                # Initialize doe_parameters from existing data in selected_inputs
                doe_parameters = {}
                for input_item in selected_inputs:
                    if 'doe_parameters' in input_item and input_item['doe_parameters']:
                        doe_parameters[input_item['name']] = input_item['doe_parameters']

                # Run DOE setup
                doe_parameters = setup_doe(solver, selected_inputs, doe_parameters, ui_helpers)

                # Save DOE back to selected_inputs
                for input_item in selected_inputs:
                    if 'doe_parameters' not in input_item:
                        input_item['doe_parameters'] = {}
                    input_item['doe_parameters'] = doe_parameters.get(input_item['name'], {})

        elif choice == 5:
            # Add Latin Hypercube Sample Points
            if not selected_inputs:
                print("\n✗ Please configure inputs first (Option 1)")
                ui_helpers.pause()
            else:
                from modules.doe_setup import add_lhs_points_simple

                # Get existing DOE parameters
                doe_parameters = {}
                for input_item in selected_inputs:
                    if 'doe_parameters' in input_item and input_item['doe_parameters']:
                        doe_parameters[input_item['name']] = input_item['doe_parameters']

                # Add LHS points
                doe_parameters = add_lhs_points_simple(doe_parameters, ui_helpers)

                # Save back to selected_inputs
                for input_item in selected_inputs:
                    if 'doe_parameters' not in input_item:
                        input_item['doe_parameters'] = {}
                    input_item['doe_parameters'] = doe_parameters.get(input_item['name'], {})

        elif choice == 6:
            if not solver or not selected_inputs or not selected_outputs:
                print("\n✗ Please complete all configuration steps first")
                ui_helpers.pause()
            else:
                # Collect DOE parameters from selected inputs
                doe_parameters = {}
                for input_item in selected_inputs:
                    if 'doe_parameters' in input_item and input_item['doe_parameters']:
                        doe_parameters[input_item['name']] = input_item['doe_parameters']

                # Create setup data
                from datetime import datetime
                setup_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model_inputs': [
                        {
                            'name': item['name'],
                            'type': item['type'],
                            'category': item.get('category', 'Unknown'),
                            'doe_parameters': doe_parameters.get(item['name'], {})
                        }
                        for item in selected_inputs
                    ],
                    'model_outputs': [
                        {
                            'name': item['name'],
                            'type': item['type'],
                            'category': item.get('category', 'Unknown')
                        }
                        for item in selected_outputs
                    ],
                    'doe_configuration': doe_parameters,
                    'case_file': getattr(solver, '_case_file_path', '')  # Store case file path
                }

                from modules import doe_setup as doe
                analysis = doe.analyze_setup_dimensions(setup_data)

                print(f"\n✓ Setup complete: {analysis['total_input_combinations']} simulations required")
                ui_helpers.pause()

                return (solver, selected_inputs, selected_outputs, output_params,
                       setup_data, analysis, None)


# ============================================================
# MODEL SETUP & TRAINING MENU
# ============================================================

def model_setup_and_training_menu(project):
    """
    Menu for model configuration and training.
    Lists available simulation setups.
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("MODEL SETUP & TRAINING")

    if not project.cases:
        print("\n✗ No simulation datasets found in project")
        print("  Please create a setup and simulate first (Test I/O & Simulations menu)")
        ui_helpers.pause()
        return

    # List datasets
    print("\nAvailable Simulation Datasets:")
    for i, dataset in enumerate(project.cases, 1):
        status = f"{dataset['completeness']:.0f}% complete"
        print(f"  [{i}] {dataset['name']:30s} [{status}]")

    print(f"\n  [0] Back")
    print("="*70)

    choice = ui_helpers.get_choice(len(project.cases))

    if choice == 0:
        return

    dataset = project.cases[choice - 1]
    dataset_dir = dataset['path']

    # Check completeness
    if dataset['completeness'] < 100:
        print(f"\n⚠ Warning: Dataset is only {dataset['completeness']:.0f}% complete")
        proceed = input("  Proceed anyway? [y/N]: ").strip().lower()
        if proceed != 'y':
            return

    # Enter training menu
    mt.train_model_menu(dataset_dir, ui_helpers)

    # Refresh project
    project.scan()


# ============================================================
# Data Visualization Menu
# ============================================================

def data_visualization_model_select_menu(project, ui_helpers):
    """
    Menu for visualizing data.
    Lists available trained models.
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("DATA VISUALIZATION")

    if not project.cases:
        print("\n✗ No trained models found in project")
        print("  Please train a model first")
        ui_helpers.pause()
        return

    # List datasets
    print("\nAvailable Models:")
    for i, dataset in enumerate(project.cases, 1):
        print(f"  [{i}] {dataset['name']:30s}")

    print(f"\n  [0] Back")
    print("="*70)

    choice = ui_helpers.get_choice(len(project.cases))

    if choice == 0:
        return

    case = project.cases[choice - 1]
    case_dir = case['path']

    # Enter Visualize menu
    dv.visualization_menu(case_dir, ui_helpers)

    # Refresh project
    project.scan()


# ============================================================
# MANAGE PROJECT DATA MENU
# ============================================================

def manage_project_data_menu(project):
    """
    Menu for managing project data (deleting datasets and models).
    """
    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("MANAGE PROJECT DATA")

        print(f"\nProject: {project.info['project_name']}")
        print(f"Cases: {len(project.cases)}")
        print(f"Trained Models: {len(project.cases)}")

        print(f"\n{'='*70}")
        print("  [1] Delete Simulation Setup")
        print("  [2] Delete Trained Model")
        print("  [0] Back")
        print("="*70)

        choice = ui_helpers.get_choice(2)

        if choice == 0:
            return

        elif choice == 1:
            delete_dataset_menu(project)

        elif choice == 2:
            delete_model_menu(project)


def delete_dataset_menu(project):
    """Delete a simulation setup."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("DELETE SIMULATION SETUP")

    if not project.cases:
        print("\n✗ No datasets found in project")
        ui_helpers.pause()
        return

    # List datasets
    print("\nAvailable Datasets:")
    for i, dataset in enumerate(project.cases, 1):
        print(f"  [{i}] {dataset['name']}")

    print(f"\n  [0] Cancel")
    print("="*70)

    choice = ui_helpers.get_choice(len(project.cases))

    if choice == 0:
        return

    dataset = project.cases[choice - 1]

    # Confirm deletion
    confirm = input(f"\n⚠ Delete dataset '{dataset['name']}'? This cannot be undone. [y/N]: ").strip().lower()

    if confirm == 'y':
        if project.delete_case(dataset['name']):
            print(f"\n✓ Dataset '{dataset['name']}' deleted successfully")
        else:
            print(f"\n✗ Failed to delete dataset '{dataset['name']}'")

    ui_helpers.pause()


def delete_model_menu(project):
    """Delete trained models from a case."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("DELETE TRAINED MODELS")

    if not project.cases:
        print("\n✗ No cases found in project")
        ui_helpers.pause()
        return

    # First, select a case
    print("\nSelect Case:")
    for i, case in enumerate(project.cases, 1):
        num_models = case.get('num_models', 0)
        print(f"  [{i}] {case['name']} ({num_models} models)")

    print(f"\n  [0] Cancel")
    print("="*70)

    choice = ui_helpers.get_choice(len(project.cases))

    if choice == 0:
        return

    case = project.cases[choice - 1]
    models_dir = case['path'] / "models"

    if not models_dir.exists():
        print(f"\n✗ No models directory found in case '{case['name']}'")
        ui_helpers.pause()
        return

    # List models in this case
    model_files = list(models_dir.glob("*.h5"))

    if not model_files:
        print(f"\n✗ No models found in case '{case['name']}'")
        ui_helpers.pause()
        return

    print(f"\nModels in '{case['name']}':")
    for i, model_file in enumerate(model_files, 1):
        print(f"  [{i}] {model_file.stem}")

    print(f"  [0] Cancel")

    model_choice = ui_helpers.get_choice(len(model_files))

    if model_choice == 0:
        return

    selected_model = model_files[model_choice - 1]

    # Confirm deletion
    confirm = input(f"\n⚠ Delete model '{selected_model.stem}'? This cannot be undone. [y/N]: ").strip().lower()

    if confirm == 'y':
        try:
            # Delete both .h5 and .npz files
            selected_model.unlink()
            npz_file = selected_model.with_suffix('.npz')
            if npz_file.exists():
                npz_file.unlink()

            # Delete metadata if exists
            metadata_file = models_dir / f"{selected_model.stem}_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()

            print(f"\n✓ Model '{selected_model.stem}' deleted successfully")
            project.scan()  # Refresh
        except Exception as e:
            print(f"\n✗ Failed to delete model: {e}")

    ui_helpers.pause()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point with project selection."""
    # Open or create project
    project = project_opening_menu()

    # Enter project main menu
    project_main_menu(project)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
