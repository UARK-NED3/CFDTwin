"""
Multi-Model Visualizer Module
===============================
Visualizes predictions from multiple trained models (1D, 2D, 3D).
Supports comparison plots, error analysis, and custom parameter prediction.
Includes optional Fluent validation for predictions.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D

from .scalar_nn_model import ScalarNNModel
from .field_nn_model import FieldNNModel
from .volume_nn_model import VolumeNNModel


def visualization_menu(dataset_dir, ui_helpers):
    """
    Main visualization menu for trained models.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    ui_helpers : module
        UI helpers module
    """
    # First, select which model to visualize
    ui_helpers.clear_screen()
    ui_helpers.print_header("SELECT MODEL TO VISUALIZE")

    print(f"\nCase: {dataset_dir.name}\n")

    # Find all model folders (directories that contain model files)
    model_folders = []
    for item in dataset_dir.iterdir():
        if item.is_dir() and list(item.glob("*_metadata.json")):
            model_folders.append(item.name)

    if not model_folders:
        print("[X] No trained models found. Train models first.")
        ui_helpers.pause()
        return

    print("Available models:")
    print("="*70)
    for i, folder_name in enumerate(sorted(model_folders), 1):
        summary_file = dataset_dir / folder_name / "training_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            timestamp = summary.get('training_info', {}).get('timestamp', 'Unknown')
            num_models = len(summary.get('models', []))
            print(f"  [{i}] {folder_name:20s} ({num_models} models, trained {timestamp})")
        else:
            print(f"  [{i}] {folder_name:20s}")
    print("="*70)

    choice = input("\nSelect model number (or 'B' to go back): ").strip().upper()

    if choice == 'B':
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(model_folders):
            selected_model = sorted(model_folders)[idx]
            models_dir = dataset_dir / selected_model
        else:
            print("\n[X] Invalid selection")
            ui_helpers.pause()
            return
    except ValueError:
        print("\n[X] Invalid input")
        ui_helpers.pause()
        return

    # Load training summary
    summary_file = models_dir / "training_summary.json"
    if not summary_file.exists():
        print("\n[X] Training summary not found.")
        ui_helpers.pause()
        return

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header(f"DATA VISUALIZATION: {selected_model}")

        print(f"\nCase: {dataset_dir.name}")

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print(f"Trained models: {summary['n_models']}")
        print(f"Trained: {summary['trained_date']}")

        print(f"\n{'='*70}")
        print("  [1] View Model Performance Summary")
        print("  [2] Compare with Fluent Simulation")
        print("  [0] Back")
        print("="*70)

        choice = ui_helpers.get_choice(4)

        if choice == 0:
            return
        elif choice == 1:
            view_model_summary(dataset_dir, summary, ui_helpers)
        elif choice == 2:
            predict_only(dataset_dir, summary, models_dir, ui_helpers)

def view_model_summary(dataset_dir, summary, ui_helpers):
    """
    Display summary of all trained models.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    summary : dict
        Training summary data
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("MODEL PERFORMANCE SUMMARY")

    print(f"\nCase: {summary['case_name']}")
    print(f"Training Date: {summary['trained_date']}")
    print(f"Total Models: {summary['n_models']}")
    print(f"Train Samples: {summary['n_train_samples']}")
    print(f"Test Samples: {summary['n_test_samples']}")

    print(f"\n{'='*80}")
    print(f"{'Model Name':<35s} {'Type':<6s} {'R² (Test)':<12s} {'MAE':<12s} {'RMSE':<12s}")
    print(f"{'='*80}")

    for model_meta in summary['models']:
        name = model_meta['model_name']
        mtype = model_meta['output_type']
        r2 = model_meta['test_metrics']['r2']
        mae = model_meta['test_metrics']['mae']
        rmse = model_meta['test_metrics']['rmse']

        print(f"{name:<35s} {mtype:<6s} {r2:>11.4f} {mae:>11.4e} {rmse:>11.4e}")

    print(f"{'='*80}")

    ui_helpers.pause()

def predict_only(dataset_dir, summary, models_dir, ui_helpers):
    """
    Prediction submenu with optional Fluent validation.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    summary : dict
        Training summary data
    models_dir : Path
        Directory containing the trained models
    ui_helpers : module
        UI helpers module
    """
    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("PREDICTION MENU")

        print(f"\nCase: {dataset_dir.name}")
        print(f"Trained models: {summary['n_models']}")

        print(f"\n{'='*70}")
        print("  [1] Run Prediction (NN Only)")
        print("  [2] Run Prediction with Fluent Validation")
        print("  [3] Validate Dataset (Random Point vs Fluent)")
        print("  [4] Compare Dataset Point (NN vs Dataset Ground Truth)")
        print("  [0] Back")
        print("="*70)

        choice = ui_helpers.get_choice(4)

        if choice == 0:
            return
        elif choice == 1:
            run_prediction_workflow(dataset_dir, summary, models_dir, ui_helpers, run_fluent=False)
        elif choice == 2:
            run_prediction_workflow(dataset_dir, summary, models_dir, ui_helpers, run_fluent=True)
        elif choice == 3:
            validate_dataset_point(dataset_dir, summary, models_dir, ui_helpers)
        elif choice == 4:
            compare_dataset_point_with_nn(dataset_dir, summary, models_dir, ui_helpers)


def run_prediction_workflow(dataset_dir, summary, models_dir, ui_helpers, run_fluent=False):
    """
    Run prediction workflow with optional Fluent validation.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    summary : dict
        Training summary data
    models_dir : Path
        Directory containing the trained models
    ui_helpers : module
        UI helpers module
    run_fluent : bool
        Whether to run Fluent validation simulation
    """
    ui_helpers.clear_screen()
    header = "MODEL PREDICTION WITH FLUENT VALIDATION" if run_fluent else "MODEL PREDICTION (NN ONLY)"
    ui_helpers.print_header(header)

    # Load model setup
    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    doe_config = setup_data.get('doe_configuration', {})

    # Build parameter list - MUST match training order (sorted keys)
    param_info = []
    for bc_name in sorted(doe_config.keys()):
        params = doe_config[bc_name]
        for param_name in sorted(params.keys()):
            values = params[param_name]
            param_info.append({
                'bc_name': bc_name,
                'param_name': param_name,
                'full_name': f"{bc_name}.{param_name}",
                'min': min(values),
                'max': max(values)
            })

    # Display parameter info
    print(f"\nInput Parameters ({len(param_info)}):")
    for i, info in enumerate(param_info, 1):
        print(f"  {i}. {info['full_name']}: [{info['min']:.3f}, {info['max']:.3f}]")

    # Get user input
    print(f"\nEnter parameter values (or press Enter for random):")
    custom_params = []

    for info in param_info:
        user_input = input(f"  {info['full_name']}: ").strip()

        if user_input:
            try:
                value = float(user_input)
            except ValueError:
                print(f"    Invalid input, using random value")
                value = np.random.uniform(info['min'], info['max'])
        else:
            value = np.random.uniform(info['min'], info['max'])
            print(f"    Using: {value:.3f}")

        custom_params.append(value)

    X_custom = np.array([custom_params])

    # Display parameters being used
    print(f"\n{'='*70}")
    print("PARAMETERS FOR PREDICTION")
    print(f"{'='*70}")
    for i, info in enumerate(param_info):
        print(f"  {info['full_name']:<45} = {X_custom[0][i]:.6f}")
    print(f"{'='*70}")

    # Run predictions
    print(f"\n{'='*70}")
    print("RUNNING MODEL PREDICTIONS")
    print(f"{'='*70}")

    predictions = {}

    for model_meta in summary['models']:
        model_name = model_meta['model_name']
        output_key = model_meta['output_key']
        output_type = model_meta['output_type']

        try:
            # Load model
            model_path = models_dir / model_name

            if output_type == '1D':
                model = ScalarNNModel.load(model_path)
            elif output_type == '2D':
                model = FieldNNModel.load(model_path)
            else:  # 3D
                model = VolumeNNModel.load(model_path)

            # Predict
            Y_pred = model.predict(X_custom)[0]

            predictions[output_key] = {
                'prediction': Y_pred,
                'model_name': model_name,
                'output_type': output_type,
                'model_meta': model_meta
            }

            print(f"  {model_name} ({output_type}): [OK]")

        except Exception as e:
            print(f"  {model_name}: [X] Error: {e}")

    # Run Fluent validation if requested
    fluent_data = None
    if run_fluent:
        print(f"\n{'='*70}")
        print("RUNNING FLUENT VALIDATION")
        print(f"{'='*70}")
        fluent_data = run_fluent_validation(dataset_dir, setup_data, custom_params, param_info, ui_helpers)

        if fluent_data is None:
            print("\n[X] Fluent validation failed. Continuing with NN predictions only.")
            input("\nPress Enter to continue...")

    # Store parameters and results for visualization menu
    viz_data = {
        'predictions': predictions,
        'fluent_data': fluent_data,
        'custom_params': X_custom[0],
        'param_info': param_info,
        'dataset_dir': dataset_dir,
        'summary': summary
    }

    # Show visualization menu
    visualization_selection_menu(viz_data, ui_helpers)


def run_fluent_validation(dataset_dir, setup_data, custom_params, param_info, ui_helpers):
    """
    Run a Fluent simulation with the given parameters for validation.
    Uses existing simulation_runner functions to avoid code duplication.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    setup_data : dict
        Model setup configuration
    custom_params : list
        Parameter values to use
    param_info : list
        Parameter information
    ui_helpers : module
        UI helpers module

    Returns
    -------
    dict or None
        Dictionary of Fluent results, or None if failed
    """
    try:
        # Import simulation runner module
        from . import simulation_runner as sr
        import ansys.fluent.core as pyfluent
        from ansys.fluent.core.launcher.launcher import UIMode

        # Get case file path from setup
        case_file = setup_data.get('case_file')
        if not case_file or not Path(case_file).exists():
            print(f"\n[X] Case file not found: {case_file}")
            print("    Please check model_setup.json")
            return None

        print(f"\nCase file: {case_file}")

        # Ask for number of iterations
        iterations_str = input("\nEnter number of iterations [100]: ").strip() or "100"
        try:
            iterations = int(iterations_str)
        except ValueError:
            print("  Invalid input, using 100 iterations")
            iterations = 100

        # Ask for number of processors
        processors_str = input("\nEnter number of processors [2]: ").strip() or "2"
        try:
            processors = int(processors_str)
        except ValueError:
            print("  Invalid input, using 2 processors")
            processors = 2

        print(f"\n[1/4] Launching Fluent...")
        #print("  Note: Fluent console window will open showing iteration progress")

        # Launch Fluent using same pattern as fluent_interface.py
        solver = pyfluent.launch_fluent(
            precision='double',
            processor_count=processors,
            dimension=3,
            mode='solver',
            #ui_mode=UIMode.NO_GUI_OR_GRAPHICS
        )

        print(f"  [OK] Fluent launched")

        # Read case file
        print(f"\n[2/4] Reading case file...")
        solver.settings.file.read_case(file_name=case_file)
        print(f"  [OK] Case file loaded")

        # Build BC values dict (same format as simulation_runner)
        bc_values = {}
        for i, info in enumerate(param_info):
            bc_key = f"{info['bc_name']}|{info['param_name']}"

            # Get BC type from model_inputs
            bc_type = 'Unknown'
            for input_item in setup_data['model_inputs']:
                if input_item['name'] == info['bc_name']:
                    bc_type = input_item['type']
                    break

            bc_values[bc_key] = {
                'bc_name': info['bc_name'],
                'bc_type': bc_type,
                'param_name': info['param_name'],
                'param_path': info['param_name'],
                'value': custom_params[i]
            }

        # Apply boundary conditions using simulation_runner function
        print(f"\n[3/4] Applying boundary conditions...")
        if not sr.apply_boundary_conditions(solver, bc_values):
            print("\n[X] Failed to apply boundary conditions")
            print("\n  Note: The Fluent console window is still open for debugging.")
            print("        Close the Fluent window manually when you're done reviewing.")
            input("\nPress Enter to continue...")
            return None

        # Initialize and solve
        print(f"\n[4/4] Running simulation ({iterations} iterations)...")
        print("  Watch the Fluent console window for iteration progress...")

        try:
            # Initialize
            solver.settings.solution.initialization.initialization_type = "standard"
            solver.settings.solution.initialization.standard_initialize()

            # Run iterations - output goes to Fluent console
            solver.settings.solution.run_calculation.iterate(iter_count=iterations)

            print(f"  [OK] Simulation complete")

        except Exception as e:
            print(f"  [X] Simulation error: {e}")
            print("\n  Note: The Fluent console window is still open for debugging.")
            print("        Close the Fluent window manually when you're done reviewing.")
            input("\nPress Enter to continue...")
            return None

        # Extract results using simulation_runner function
        print(f"\n[5/5] Extracting results...")
        fluent_results = sr.extract_field_data(solver, setup_data, dataset_dir)

        # Keep Fluent session open (don't call solver.exit())
        print(f"\n  Note: Fluent session is kept open for review.")
        print("        Close the Fluent console window manually when finished.")

        if fluent_results:
            print(f"\n  [OK] Fluent validation complete")
            print(f"  Extracted {len(fluent_results)} output fields")
            return fluent_results
        else:
            print(f"  [X] Failed to extract results")
            input("\nPress Enter to continue...")
            return None

    except ImportError as e:
        print(f"\n[X] Import error: {e}")
        print("    Make sure PyFluent is installed: pip install ansys-fluent-core")
        return None
    except Exception as e:
        print(f"\n[X] Error running Fluent validation: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualization_selection_menu(viz_data, ui_helpers):
    """
    Menu for selecting which plots to display.

    Parameters
    ----------
    viz_data : dict
        Visualization data containing predictions, fluent_data, etc.
    ui_helpers : module
        UI helpers module
    """
    predictions = viz_data['predictions']
    fluent_data = viz_data['fluent_data']
    has_fluent = fluent_data is not None

    while True:
       # ui_helpers.clear_screen()
        ui_helpers.print_header("VISUALIZATION MENU")

        print(f"\nPrediction Results Available:")
        print(f"  Neural Network Models: {len(predictions)}")
        print(f"  Fluent Validation: {'Yes' if has_fluent else 'No'}")

        # Categorize predictions
        scalar_results = {k: v for k, v in predictions.items() if v['output_type'] == '1D'}
        field_2d_results = {k: v for k, v in predictions.items() if v['output_type'] == '2D'}
        field_3d_results = {k: v for k, v in predictions.items() if v['output_type'] == '3D'}

        print(f"\n{'='*70}")
        print("VISUALIZATION OPTIONS:")
        print(f"{'='*70}")

        menu_options = []
        option_num = 1

        # Scalar results summary
        if scalar_results:
            print(f"  [{option_num}] View Scalar Results Summary")
            menu_options.append(('scalar_summary', None))
            option_num += 1

        # Individual 2D field plots
        if field_2d_results:
            print(f"\n  2D Field Plots:")
            for output_key, data in field_2d_results.items():
                # Format name nicely: "yz-mid_temperature" -> "yz-mid Temperature"
                display_name = data['model_name'].replace('_', ' ').title()
                print(f"  [{option_num}] {display_name}")
                menu_options.append(('2d_plot', output_key))
                option_num += 1

        # Individual 3D field plots
        if field_3d_results:
            print(f"\n  3D Field Plots:")
            for output_key, data in field_3d_results.items():
                # Format name nicely: "volume_temperature" -> "Volume Temperature"
                display_name = data['model_name'].replace('_', ' ').title()
                print(f"  [{option_num}] {display_name}")
                menu_options.append(('3d_plot', output_key))
                option_num += 1

        # Show all plots
        if len(predictions) > 1:
            print(f"\n  [{option_num}] Show All Plots")
            menu_options.append(('show_all', None))
            option_num += 1

        print(f"\n  [0] Back")
        print(f"{'='*70}")

        choice = ui_helpers.get_choice(len(menu_options))

        if choice == 0:
            return

        # Execute selected option
        action, output_key = menu_options[choice - 1]

        if action == 'scalar_summary':
            display_scalar_summary(scalar_results, fluent_data, viz_data, ui_helpers)
        elif action == '2d_plot':
            display_2d_plot(output_key, predictions[output_key], fluent_data, viz_data, ui_helpers)
        elif action == '3d_plot':
            display_3d_plot(output_key, predictions[output_key], fluent_data, viz_data, ui_helpers)
        elif action == 'show_all':
            display_all_plots(predictions, fluent_data, viz_data, ui_helpers)


def display_scalar_summary(scalar_results, fluent_data, _viz_data, ui_helpers):
    """Display scalar results as a summary table/plot."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("SCALAR RESULTS SUMMARY")

    print(f"\n{'='*70}")
    print(f"{'Output':<40} {'NN Prediction':<15} {'Fluent':<15} {'Error'}")
    print(f"{'='*70}")

    for output_key, data in scalar_results.items():
        nn_value = data['prediction'][0]

        # Use npz_key (pipe format) for Fluent lookup instead of output_key (underscore format)
        fluent_key = data['model_meta']['npz_key']

        if fluent_data and fluent_key in fluent_data:
            fluent_value = fluent_data[fluent_key][0]
            error = abs(nn_value - fluent_value)
            error_pct = (error / fluent_value * 100) if fluent_value != 0 else 0
            print(f"{output_key:<40} {nn_value:>14.6e} {fluent_value:>14.6e} {error_pct:>6.2f}%")
        else:
            print(f"{output_key:<40} {nn_value:>14.6e} {'N/A':<15} {'N/A'}")

    print(f"{'='*70}")
    ui_helpers.pause()


def format_field_label(field_name):
    """Add units to field name if applicable."""
    if 'temperature' in field_name.lower():
        return f'{field_name} (K)'
    return field_name


def display_2d_plot(output_key, pred_data, fluent_data, viz_data, ui_helpers):
    """Display 2D field plot with optional Fluent comparison."""
    dataset_dir = viz_data['dataset_dir']

    # Load coordinates from a file with matching shape
    dataset_output_dir = dataset_dir / "dataset"
    output_files = sorted(dataset_output_dir.glob("sim_*.npz"))

    if not output_files:
        print("\n[X] No training data found for coordinates")
        ui_helpers.pause()
        return

    location = pred_data['model_meta']['location']
    field_name = pred_data['model_meta']['field_name']
    coord_key = f"{location}|coordinates"
    npz_key = pred_data['model_meta']['npz_key']
    nn_pred = pred_data['prediction']
    expected_size = len(nn_pred)

    # Find a file with matching field size for coordinates
    # IMPORTANT: Use a file from the middle of the dataset to ensure it was included in training.
    # First and last files may have been excluded due to data quality issues.
    coordinates = None

    # Collect all valid files with matching size
    valid_coord_files = []
    for sample_file_path in output_files:
        sample_data = np.load(sample_file_path, allow_pickle=True)
        if coord_key in sample_data.files and npz_key in sample_data.files:
            if len(sample_data[npz_key]) == expected_size:
                valid_coord_files.append((sample_file_path, sample_data))

    if not valid_coord_files:
        print(f"\n[X] Could not find coordinates with matching size ({expected_size} points)")
        ui_helpers.pause()
        return

    # Use a file from the middle of the valid files (most likely to be in training set)
    middle_idx = len(valid_coord_files) // 2
    sample_file_path, sample_data = valid_coord_files[middle_idx]
    coordinates = sample_data[coord_key]
    print(f"  Loading coordinates from {sample_file_path.name} ({len(coordinates)} points)")

    # Detect which 2 dimensions vary (for 2D surface plots)
    # Calculate variance for each dimension to find the varying axes
    variances = [np.var(coordinates[:, i]) for i in range(3)]
    # Get indices of 2 dimensions with highest variance
    varying_dims = sorted(range(3), key=lambda i: variances[i], reverse=True)[:2]
    varying_dims.sort()  # Sort to maintain X, Y, Z order preference

    # Get axis labels
    axis_names = ['X', 'Y', 'Z']
    xlabel = f'{axis_names[varying_dims[0]]} (m)'
    ylabel = f'{axis_names[varying_dims[1]]} (m)'

    # Check if we have Fluent data
    # Use npz_key (pipe format) for Fluent lookup instead of output_key (underscore format)
    fluent_key = pred_data['model_meta']['npz_key']
    has_fluent = fluent_data is not None and fluent_key in fluent_data

    if has_fluent:
        fluent_values = fluent_data[fluent_key]
        fluent_coords = fluent_data[f"{location}|coordinates"]

        # Always interpolate to ensure coordinate alignment
        # Even if sizes match, coordinate ordering may differ between Fluent and dataset
        print(f"\n  Aligning Fluent data with dataset coordinates...")
        print(f"  Fluent: {len(fluent_values)} points, NN: {len(nn_pred)} points")

        try:
            from scipy.interpolate import griddata
            # Interpolate Fluent values onto NN coordinates
            fluent_values_interp = griddata(
                fluent_coords, fluent_values, coordinates,
                method='nearest'
            )
            fluent_values = fluent_values_interp
            fluent_coords = coordinates
            print(f"  Alignment successful!")
        except Exception as e:
            print(f"  Alignment failed: {e}")
            print(f"  Showing NN prediction only (no Fluent comparison)")
            has_fluent = False

    # Downsample by half for performance
    n_points = len(coordinates)
    target_points = n_points // 2
    if n_points > target_points:
        indices = np.random.choice(n_points, target_points, replace=False)
        coordinates = coordinates[indices]
        nn_pred = nn_pred[indices]
        if has_fluent:
            fluent_values = fluent_values[indices]
            fluent_coords = fluent_coords[indices]
        print(f"  Downsampled from {n_points} to {target_points} points for visualization")

    if has_fluent:
        # Create 3-panel plot: NN, Fluent, Error
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Determine if we need to rotate the plot (yz-mid should be rotated 90 degrees CCW)
        rotate_plot = 'yz' in location.lower()

        # NN Prediction
        if rotate_plot:
            # Swap X and Y for 90-degree CCW rotation
            scatter1 = axes[0].scatter(
                coordinates[:, varying_dims[1]], coordinates[:, varying_dims[0]],
                c=nn_pred, cmap='viridis', s=15, alpha=0.8
            )
            axes[0].set_xlabel(ylabel)
            axes[0].set_ylabel(xlabel)
        else:
            scatter1 = axes[0].scatter(
                coordinates[:, varying_dims[0]], coordinates[:, varying_dims[1]],
                c=nn_pred, cmap='viridis', s=15, alpha=0.8
            )
            axes[0].set_xlabel(xlabel)
            axes[0].set_ylabel(ylabel)
        axes[0].set_title(f'Neural Network\n{field_name}')
        axes[0].set_aspect('equal')
        plt.colorbar(scatter1, ax=axes[0], label=format_field_label(field_name))
        axes[0].grid(True, alpha=0.3)

        # Fluent Result
        if rotate_plot:
            # Swap X and Y for 90-degree CCW rotation
            scatter2 = axes[1].scatter(
                fluent_coords[:, varying_dims[1]], fluent_coords[:, varying_dims[0]],
                c=fluent_values, cmap='viridis', s=15, alpha=0.8
            )
            axes[1].set_xlabel(ylabel)
            axes[1].set_ylabel(xlabel)
        else:
            scatter2 = axes[1].scatter(
                fluent_coords[:, varying_dims[0]], fluent_coords[:, varying_dims[1]],
                c=fluent_values, cmap='viridis', s=15, alpha=0.8
            )
            axes[1].set_xlabel(xlabel)
            axes[1].set_ylabel(ylabel)
        axes[1].set_title(f'Fluent CFD\n{field_name}')
        axes[1].set_aspect('equal')
        plt.colorbar(scatter2, ax=axes[1], label=format_field_label(field_name))
        axes[1].grid(True, alpha=0.3)

        # Match color scales
        vmin = min(nn_pred.min(), fluent_values.min())
        vmax = max(nn_pred.max(), fluent_values.max())
        scatter1.set_clim(vmin, vmax)
        scatter2.set_clim(vmin, vmax)

        # Error plot
        error = np.abs(nn_pred - fluent_values)
        if rotate_plot:
            # Swap X and Y for 90-degree CCW rotation
            scatter3 = axes[2].scatter(
                coordinates[:, varying_dims[1]], coordinates[:, varying_dims[0]],
                c=error, cmap='Reds', s=15, alpha=0.8
            )
            axes[2].set_xlabel(ylabel)
            axes[2].set_ylabel(xlabel)
        else:
            scatter3 = axes[2].scatter(
                coordinates[:, varying_dims[0]], coordinates[:, varying_dims[1]],
                c=error, cmap='Reds', s=15, alpha=0.8
            )
            axes[2].set_xlabel(xlabel)
            axes[2].set_ylabel(ylabel)
        axes[2].set_title(f'Absolute Error\nMAE: {error.mean():.4e}')
        axes[2].set_aspect('equal')
        plt.colorbar(scatter3, ax=axes[2], label='Error')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'{pred_data["model_name"]}', fontsize=14, fontweight='bold')

    else:
        # Single plot: NN only
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Determine if we need to rotate the plot (yz-mid should be rotated 90 degrees CCW)
        rotate_plot = 'yz' in location.lower()

        if rotate_plot:
            # Swap X and Y for 90-degree CCW rotation
            scatter = ax.scatter(
                coordinates[:, varying_dims[1]], coordinates[:, varying_dims[0]],
                c=nn_pred, cmap='viridis', s=15, alpha=0.8
            )
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)
        else:
            scatter = ax.scatter(
                coordinates[:, varying_dims[0]], coordinates[:, varying_dims[1]],
                c=nn_pred, cmap='viridis', s=15, alpha=0.8
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        ax.set_title(f'{pred_data["model_name"]}\n{field_name}')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label=format_field_label(field_name))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)  # Don't block so menu stays open

    ui_helpers.pause()


def display_3d_plot(output_key, pred_data, fluent_data, viz_data, ui_helpers):
    """Display 3D field plot with optional Fluent comparison."""
    dataset_dir = viz_data['dataset_dir']

    # Load coordinates from a file with matching shape
    dataset_output_dir = dataset_dir / "dataset"
    output_files = sorted(dataset_output_dir.glob("sim_*.npz"))

    if not output_files:
        print("\n[X] No training data found for coordinates")
        ui_helpers.pause()
        return

    location = pred_data['model_meta']['location']
    field_name = pred_data['model_meta']['field_name']
    coord_key = f"{location}|coordinates"
    npz_key = pred_data['model_meta']['npz_key']
    nn_pred = pred_data['prediction']
    expected_size = len(nn_pred)

    # Find a file with matching field size for coordinates
    # IMPORTANT: Use a file from the middle of the dataset to ensure it was included in training.
    # First and last files may have been excluded due to data quality issues.
    coordinates = None

    # Collect all valid files with matching size
    valid_coord_files = []
    for sample_file_path in output_files:
        sample_data = np.load(sample_file_path, allow_pickle=True)
        if coord_key in sample_data.files and npz_key in sample_data.files:
            if len(sample_data[npz_key]) == expected_size:
                valid_coord_files.append((sample_file_path, sample_data))

    if not valid_coord_files:
        print(f"\n[X] Could not find coordinates with matching size ({expected_size} points)")
        ui_helpers.pause()
        return

    # Use a file from the middle of the valid files (most likely to be in training set)
    middle_idx = len(valid_coord_files) // 2
    sample_file_path, sample_data = valid_coord_files[middle_idx]
    coordinates = sample_data[coord_key]
    print(f"  Loading coordinates from {sample_file_path.name} ({len(coordinates)} points)")

    # Downsample by half for performance
    n_points = len(coordinates)
    target_points = n_points // 2
    if n_points > target_points:
        indices = np.random.choice(n_points, target_points, replace=False)
        coordinates = coordinates[indices]
        nn_pred = nn_pred[indices]
        print(f"\n  Note: Downsampled from {n_points} to {target_points} points for visualization")
    else:
        indices = None

    # Check if we have Fluent data
    # Use npz_key (pipe format) for Fluent lookup instead of output_key (underscore format)
    fluent_key = pred_data['model_meta']['npz_key']
    has_fluent = fluent_data is not None and fluent_key in fluent_data

    if has_fluent:
        fluent_values = fluent_data[fluent_key]
        fluent_coords = fluent_data[f"{location}|coordinates"]

        # Always interpolate to ensure coordinate alignment
        # Even if sizes match, coordinate ordering may differ between Fluent and dataset
        print(f"\n  Aligning Fluent data with dataset coordinates...")
        print(f"  Fluent: {len(fluent_values)} points, NN: {expected_size} points")

        try:
            from scipy.interpolate import griddata
            # Get full coordinates before downsampling for interpolation
            full_coords = sample_data[coord_key]

            # Interpolate Fluent values onto full NN coordinates
            fluent_values_interp = griddata(
                fluent_coords, fluent_values, full_coords,
                method='nearest'
            )
            fluent_values = fluent_values_interp
            fluent_coords = full_coords

            # Now apply downsampling if needed
            if indices is not None:
                fluent_values = fluent_values[indices]
                fluent_coords = fluent_coords[indices]

            print(f"  Alignment successful!")
        except Exception as e:
            print(f"  Alignment failed: {e}")
            print(f"  Showing NN prediction only (no Fluent comparison)")
            has_fluent = False

    if has_fluent:
        # Calculate data ranges for aspect ratio
        x_range = coordinates[:, 0].max() - coordinates[:, 0].min()
        y_range = coordinates[:, 1].max() - coordinates[:, 1].min()
        z_range = coordinates[:, 2].max() - coordinates[:, 2].min()
        max_range = max(x_range, y_range, z_range)

        # Normalized aspect ratio
        aspect = [x_range/max_range, y_range/max_range, z_range/max_range]

        # Create 3-panel plot: NN, Fluent, Error
        fig = plt.figure(figsize=(18, 5))

        # NN Prediction
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
            c=nn_pred, cmap='viridis', s=5, alpha=0.6
        )
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'Neural Network\n{field_name}')
        plt.colorbar(scatter1, ax=ax1, label=format_field_label(field_name), shrink=0.7)
        ax1.set_box_aspect(aspect)

        # Fluent Result
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(
            fluent_coords[:, 0], fluent_coords[:, 1], fluent_coords[:, 2],
            c=fluent_values, cmap='viridis', s=5, alpha=0.6
        )
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title(f'Fluent CFD\n{field_name}')
        plt.colorbar(scatter2, ax=ax2, label=format_field_label(field_name), shrink=0.7)
        ax2.set_box_aspect(aspect)

        # Match color scales
        vmin = min(nn_pred.min(), fluent_values.min())
        vmax = max(nn_pred.max(), fluent_values.max())
        scatter1.set_clim(vmin, vmax)
        scatter2.set_clim(vmin, vmax)

        # Error plot
        error = np.abs(nn_pred - fluent_values)
        ax3 = fig.add_subplot(133, projection='3d')
        scatter3 = ax3.scatter(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
            c=error, cmap='Reds', s=5, alpha=0.6
        )
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title(f'Absolute Error\nMAE: {error.mean():.4e}')
        plt.colorbar(scatter3, ax=ax3, label='Error', shrink=0.7)
        ax3.set_box_aspect(aspect)

        fig.suptitle(f'{pred_data["model_name"]}', fontsize=14, fontweight='bold')

    else:
        # Single plot: NN only
        # Calculate data ranges for aspect ratio
        x_range = coordinates[:, 0].max() - coordinates[:, 0].min()
        y_range = coordinates[:, 1].max() - coordinates[:, 1].min()
        z_range = coordinates[:, 2].max() - coordinates[:, 2].min()
        max_range = max(x_range, y_range, z_range)

        # Normalized aspect ratio
        aspect = [x_range/max_range, y_range/max_range, z_range/max_range]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
            c=nn_pred, cmap='viridis', s=5, alpha=0.6
        )
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{pred_data["model_name"]}\n{field_name}')
        plt.colorbar(scatter, ax=ax, label=format_field_label(field_name), shrink=0.7)
        ax.set_box_aspect(aspect)

    plt.tight_layout()
    plt.show(block=False)  # Don't block so menu stays open

    ui_helpers.pause()


def display_all_plots(predictions, fluent_data, viz_data, ui_helpers):
    """Display all plots at once."""
    print("\nDisplaying all plots...")

    for output_key, pred_data in predictions.items():
        if pred_data['output_type'] == '2D':
            display_2d_plot(output_key, pred_data, fluent_data, viz_data, ui_helpers)
        elif pred_data['output_type'] == '3D':
            display_3d_plot(output_key, pred_data, fluent_data, viz_data, ui_helpers)


def compare_test_data_point(dataset_dir, summary, models_dir, ui_helpers):
    """
    Compare neural network predictions against ground truth from test dataset.
    This allows visualization of NN vs actual data without running Fluent.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    summary : dict
        Training summary data
    models_dir : Path
        Directory containing the trained models
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("TEST DATA COMPARISON: NN Prediction vs Ground Truth")

    print("\nThis tool:")
    print("  1. Selects a point from the test dataset")
    print("  2. Runs NN prediction on that point")
    print("  3. Compares prediction vs ground truth (no Fluent needed)")

    # Load model setup
    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    doe_config = setup_data.get('doe_configuration', {})

    # Get list of dataset files
    dataset_output_dir = dataset_dir / "dataset"
    output_files = sorted(dataset_output_dir.glob("sim_*.npz"))

    if not output_files:
        print("\n[X] No dataset files found")
        ui_helpers.pause()
        return

    # Extract available indices from filenames
    available_indices = []
    for f in output_files:
        try:
            # Extract index from filename like "sim_0001.npz"
            idx = int(f.stem.split('_')[1])
            available_indices.append(idx)
        except:
            pass

    if not available_indices:
        print("\n[X] Could not extract indices from dataset files")
        ui_helpers.pause()
        return

    # Get test indices from summary (if available)
    test_indices = summary.get('test_indices', [])

    # Detect dataset offset based on model directory name
    # FF_exclude uses files 2501-4500, so indices need +2500 offset
    # model_corrupt uses files 2501-4500, so indices need +2500 offset
    model_dir_name = models_dir.name.lower()
    dataset_offset = 0
    if 'ff_exclude' in model_dir_name or 'ff-exclude' in model_dir_name:
        dataset_offset = 2500
        print(f"\n  Detected FF_exclude model - using dataset offset +{dataset_offset}")
    elif 'corrupt' in model_dir_name:
        dataset_offset = 2500
        print(f"\n  Detected corrupt exclusion model - using dataset offset +{dataset_offset}")

    if test_indices:
        # Apply offset to test indices to map to actual file numbers
        test_indices_with_offset = [idx + dataset_offset for idx in test_indices]

        print(f"\n{len(test_indices)} test samples available")
        print(f"  Test indices in subset: {sorted(test_indices)[:5]}...{sorted(test_indices)[-5:]}")
        print(f"  Mapped to file indices: {sorted(test_indices_with_offset)[:5]}...{sorted(test_indices_with_offset)[-5:]}")
        available_for_selection = test_indices_with_offset
        index_type = "test"
    else:
        # Fall back to all available indices
        print(f"\n{len(available_indices)} dataset samples available (indices {min(available_indices)}-{max(available_indices)})")
        print("Note: Test indices not found in training summary - using all available data points")
        available_for_selection = available_indices
        index_type = "dataset"

    # Let user choose random or specific test sample
    choice = input(f"\nSelect random {index_type} point? (y/n, default=y): ").strip().lower()

    if choice == 'n':
        print(f"\nYou can select the Nth {index_type} sample (e.g., 1 for first, 2 for second, etc.)")
        print(f"Or enter a specific file index from the {index_type} set")
        print(f"Available file indices: {sorted(available_for_selection)[:5]}...{sorted(available_for_selection)[-5:]}")
        try:
            user_input = int(input(f"\nEnter sample number (1-{len(available_for_selection)}) or file index: ").strip())

            # Check if it's a sample number (1-based count)
            if 1 <= user_input <= len(available_for_selection):
                # Treat as Nth sample
                sorted_indices = sorted(available_for_selection)
                selected_idx = sorted_indices[user_input - 1]
                print(f"  → Selected {index_type} sample #{user_input} (file index: {selected_idx})")
            elif user_input in available_for_selection:
                # Treat as actual file index
                selected_idx = user_input
                sample_num = sorted(available_for_selection).index(user_input) + 1
                print(f"  → Selected file index {selected_idx} ({index_type} sample #{sample_num})")
            else:
                print(f"\n[X] Invalid selection. Using random {index_type} sample.")
                selected_idx = np.random.choice(available_for_selection)
        except:
            print(f"\nInvalid input. Using random {index_type} point.")
            selected_idx = np.random.choice(available_for_selection)
    else:
        selected_idx = np.random.choice(available_for_selection)

    print(f"\nUsing test sample: sim_{selected_idx:04d}.npz")

    # Load the selected test file
    test_file = dataset_output_dir / f"sim_{selected_idx:04d}.npz"
    test_data = np.load(test_file, allow_pickle=True)

    # Reconstruct input parameters from DOE configuration
    # IMPORTANT: Must use SORTED keys to match training order
    param_names = []
    param_values = []

    for bc_name in sorted(doe_config.keys()):
        params = doe_config[bc_name]
        for param_name in sorted(params.keys()):
            values = params[param_name]
            param_names.append(f"{bc_name}.{param_name}")
            param_values.append(values)

    # Check if all arrays have the same length (LHS) or different (full factorial)
    array_lengths = [len(arr) for arr in param_values]

    if len(set(array_lengths)) == 1:
        # All same length - LHS or parallel samples (zip together)
        param_combinations = list(zip(*param_values))
    else:
        # Different lengths - full factorial (all combinations)
        import itertools
        param_combinations = list(itertools.product(*param_values))

    # Get the parameters for the selected index (1-based in filename, 0-based in list)
    if selected_idx < 1 or selected_idx > len(param_combinations):
        print(f"\n[X] Selected index {selected_idx} out of range (1-{len(param_combinations)})")
        ui_helpers.pause()
        return

    X_test = np.array([param_combinations[selected_idx - 1]])

    # Build parameter info for display
    param_info = []
    for bc_name in sorted(doe_config.keys()):
        params = doe_config[bc_name]
        for param_name in sorted(params.keys()):
            values = params[param_name]
            param_info.append({
                'bc_name': bc_name,
                'param_name': param_name,
                'full_name': f"{bc_name}.{param_name}",
                'min': min(values),
                'max': max(values)
            })

    # Display parameters
    print(f"\n{'='*70}")
    print("INPUT PARAMETERS - DEBUG")
    print(f"{'='*70}")
    print(f"  Selected index: {selected_idx} (file: sim_{selected_idx:04d}.npz)")
    print(f"  Total param combinations available: {len(param_combinations)}")
    print(f"  Parameter vector shape: {X_test.shape}")
    print(f"  BC names order: {sorted(doe_config.keys())}")
    print(f"  Parameter names (sorted order): {param_names}")
    print(f"\n  Reconstructed parameters for this index:")
    for i, info in enumerate(param_info):
        print(f"    [{i}] {info['full_name']:<45} = {X_test[0][i]:.6f}")

    # Also load and display what's actually in the NPZ file for comparison
    print(f"\n  Checking NPZ file contents:")
    scalar_keys = [k for k in test_data.files if '|' in k and 'coordinates' not in k.lower() and 'mid' not in k.lower() and 'bottom' not in k.lower()]
    print(f"    Scalar output keys: {scalar_keys[:5]}")
    if scalar_keys:
        first_scalar_key = scalar_keys[0]
        val = test_data[first_scalar_key]
        print(f"    Ground truth {first_scalar_key}: {float(val) if val.size == 1 else val[0]:.6e}")

    print(f"{'='*70}")

    # Run predictions
    print(f"\n{'='*70}")
    print("RUNNING MODEL PREDICTIONS")
    print(f"{'='*70}")

    predictions = {}

    for model_meta in summary['models']:
        model_name = model_meta['model_name']
        output_key = model_meta['output_key']
        output_type = model_meta['output_type']
        npz_key = model_meta['npz_key']

        try:
            # Load model
            model_path = models_dir / model_name

            if output_type == '1D':
                model = ScalarNNModel.load(model_path)
            elif output_type == '2D':
                model = FieldNNModel.load(model_path)
            else:  # 3D
                model = VolumeNNModel.load(model_path)

            # Predict
            Y_pred = model.predict(X_test)[0]

            # Get ground truth from test file
            if npz_key not in test_data.files:
                print(f"  {model_name}: [X] Ground truth not found in test file")
                continue

            Y_true = test_data[npz_key]

            # Calculate error metrics
            mae = np.mean(np.abs(Y_pred - Y_true))
            rmse = np.sqrt(np.mean((Y_pred - Y_true)**2))

            # R² score
            ss_res = np.sum((Y_true - Y_pred)**2)
            ss_tot = np.sum((Y_true - np.mean(Y_true))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            predictions[output_key] = {
                'prediction': Y_pred,
                'ground_truth': Y_true,
                'model_name': model_name,
                'output_type': output_type,
                'model_meta': model_meta,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
            }

            # Debug output for ALL models to check values
            pred_val = float(Y_pred) if np.isscalar(Y_pred) or Y_pred.size == 1 else Y_pred.mean()
            true_val = float(Y_true) if np.isscalar(Y_true) or Y_true.size == 1 else Y_true.mean()

            print(f"  {model_name} ({output_type}): R²={r2:.4f}, MAE={mae:.4e}")
            if output_type == '1D':
                print(f"    → Pred={pred_val:.6e}, Truth={true_val:.6e}, Diff={abs(pred_val-true_val):.6e}")

        except Exception as e:
            print(f"  {model_name}: [X] Error: {e}")

    if not predictions:
        print("\n[X] No successful predictions")
        ui_helpers.pause()
        return

    # Pause to review debug output
    print("\n" + "="*70)
    print("DEBUG OUTPUT COMPLETE - Review the predictions above")
    print("="*70)
    input("\nPress Enter to continue to visualization menu...")

    # Store parameters and results for visualization menu
    viz_data = {
        'dataset_dir': dataset_dir,
        'custom_params': X_test[0],
        'param_info': param_info
    }

    # Display results
    display_test_comparison_menu(predictions, viz_data, ui_helpers)


def display_test_comparison_menu(predictions, viz_data, ui_helpers):
    """
    Display menu for visualizing test data comparisons.

    Parameters
    ----------
    predictions : dict
        Dictionary of predictions with ground truth
    viz_data : dict
        Visualization data including dataset_dir and parameters
    ui_helpers : module
        UI helpers module
    """
    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("TEST DATA COMPARISON - VISUALIZATION")

        # Separate by output type
        scalar_outputs = {k: v for k, v in predictions.items() if v['output_type'] == '1D'}
        field_2d_outputs = {k: v for k, v in predictions.items() if v['output_type'] == '2D'}
        field_3d_outputs = {k: v for k, v in predictions.items() if v['output_type'] == '3D'}

        print(f"\nAvailable Outputs:")
        print("="*70)

        menu_items = []
        counter = 1

        if scalar_outputs:
            print(f"\n1D Outputs (Scalars):")
            for output_key, pred_data in scalar_outputs.items():
                print(f"  [{counter}] {pred_data['model_name']} (R²={pred_data['metrics']['r2']:.4f})")
                menu_items.append((counter, output_key, pred_data))
                counter += 1

        if field_2d_outputs:
            print(f"\n2D Outputs (Fields):")
            for output_key, pred_data in field_2d_outputs.items():
                print(f"  [{counter}] {pred_data['model_name']} (R²={pred_data['metrics']['r2']:.4f})")
                menu_items.append((counter, output_key, pred_data))
                counter += 1

        if field_3d_outputs:
            print(f"\n3D Outputs (Volumes):")
            for output_key, pred_data in field_3d_outputs.items():
                print(f"  [{counter}] {pred_data['model_name']} (R²={pred_data['metrics']['r2']:.4f})")
                menu_items.append((counter, output_key, pred_data))
                counter += 1

        print(f"\n  [{counter}] View All Plots")
        print(f"  [0] Back")
        print("="*70)

        max_choice = counter
        choice = ui_helpers.get_choice(max_choice)

        if choice == 0:
            return
        elif choice == max_choice:
            # Display all plots
            for _, output_key, pred_data in menu_items:
                if pred_data['output_type'] == '1D':
                    display_1d_test_comparison(output_key, pred_data, ui_helpers)
                elif pred_data['output_type'] == '2D':
                    display_2d_test_comparison(output_key, pred_data, viz_data, ui_helpers)
                elif pred_data['output_type'] == '3D':
                    display_3d_test_comparison(output_key, pred_data, viz_data, ui_helpers)
        else:
            # Find selected item
            selected = None
            for num, output_key, pred_data in menu_items:
                if num == choice:
                    selected = (output_key, pred_data)
                    break

            if selected:
                output_key, pred_data = selected
                if pred_data['output_type'] == '1D':
                    display_1d_test_comparison(output_key, pred_data, ui_helpers)
                elif pred_data['output_type'] == '2D':
                    display_2d_test_comparison(output_key, pred_data, viz_data, ui_helpers)
                elif pred_data['output_type'] == '3D':
                    display_3d_test_comparison(output_key, pred_data, viz_data, ui_helpers)


def display_1d_test_comparison(output_key, pred_data, ui_helpers):
    """Display bar chart comparison for scalar outputs."""
    ui_helpers.clear_screen()
    ui_helpers.print_header(f"1D COMPARISON: {pred_data['model_name']}")

    prediction = pred_data['prediction']
    ground_truth = pred_data['ground_truth']
    metrics = pred_data['metrics']

    print(f"\nModel: {pred_data['model_name']}")
    print(f"R² Score: {metrics['r2']:.6f}")
    print(f"MAE: {metrics['mae']:.6e}")
    print(f"RMSE: {metrics['rmse']:.6e}")

    print(f"\nPrediction:    {prediction:.6e}")
    print(f"Ground Truth:  {ground_truth:.6e}")
    print(f"Error:         {abs(prediction - ground_truth):.6e}")

    ui_helpers.pause()


def display_2d_test_comparison(output_key, pred_data, viz_data, ui_helpers):
    """Display 2D field comparison plot (Prediction, Ground Truth, Error)."""
    dataset_dir = viz_data['dataset_dir']

    # Load coordinates from a file with matching shape
    dataset_output_dir = dataset_dir / "dataset"
    output_files = sorted(dataset_output_dir.glob("sim_*.npz"))

    if not output_files:
        print("\n[X] No training data found for coordinates")
        ui_helpers.pause()
        return

    location = pred_data['model_meta']['location']
    field_name = pred_data['model_meta']['field_name']
    coord_key = f"{location}|coordinates"
    npz_key = pred_data['model_meta']['npz_key']
    nn_pred = pred_data['prediction']
    ground_truth = pred_data['ground_truth']
    expected_size = len(nn_pred)

    # Find a file with matching field size for coordinates
    coordinates = None
    valid_coord_files = []
    for sample_file_path in output_files:
        sample_data = np.load(sample_file_path, allow_pickle=True)
        if coord_key in sample_data.files and npz_key in sample_data.files:
            if len(sample_data[npz_key]) == expected_size:
                valid_coord_files.append((sample_file_path, sample_data))

    if not valid_coord_files:
        print(f"\n[X] Could not find coordinates with matching size ({expected_size} points)")
        ui_helpers.pause()
        return

    # Use a file from the middle of the valid files
    middle_idx = len(valid_coord_files) // 2
    sample_file_path, sample_data = valid_coord_files[middle_idx]
    coordinates = sample_data[coord_key]

    # Detect which 2 dimensions vary
    variances = [np.var(coordinates[:, i]) for i in range(3)]
    varying_dims = sorted(range(3), key=lambda i: variances[i], reverse=True)[:2]
    varying_dims.sort()

    # Get axis labels
    axis_names = ['X', 'Y', 'Z']
    xlabel = f'{axis_names[varying_dims[0]]} (m)'
    ylabel = f'{axis_names[varying_dims[1]]} (m)'

    # Downsample for performance
    n_points = len(coordinates)
    target_points = n_points // 2
    if n_points > target_points:
        indices = np.random.choice(n_points, target_points, replace=False)
        coordinates = coordinates[indices]
        nn_pred = nn_pred[indices]
        ground_truth = ground_truth[indices]

    # Determine if we need to rotate the plot
    rotate_plot = 'yz' in location.lower()

    # Create 3-panel plot: NN, Ground Truth, Error
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Set consistent color limits
    vmin = min(nn_pred.min(), ground_truth.min())
    vmax = max(nn_pred.max(), ground_truth.max())

    # NN Prediction
    if rotate_plot:
        scatter1 = axes[0].scatter(
            coordinates[:, varying_dims[1]], coordinates[:, varying_dims[0]],
            c=nn_pred, cmap='viridis', s=15, alpha=0.8
        )
        axes[0].set_xlabel(ylabel)
        axes[0].set_ylabel(xlabel)
    else:
        scatter1 = axes[0].scatter(
            coordinates[:, varying_dims[0]], coordinates[:, varying_dims[1]],
            c=nn_pred, cmap='viridis', s=15, alpha=0.8
        )
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
    axes[0].set_title(f'Neural Network\n{field_name}')
    axes[0].set_aspect('equal')
    scatter1.set_clim(vmin, vmax)
    plt.colorbar(scatter1, ax=axes[0], label=f'{field_name}')
    axes[0].grid(True, alpha=0.3)

    # Ground Truth
    if rotate_plot:
        scatter2 = axes[1].scatter(
            coordinates[:, varying_dims[1]], coordinates[:, varying_dims[0]],
            c=ground_truth, cmap='viridis', s=15, alpha=0.8
        )
        axes[1].set_xlabel(ylabel)
        axes[1].set_ylabel(xlabel)
    else:
        scatter2 = axes[1].scatter(
            coordinates[:, varying_dims[0]], coordinates[:, varying_dims[1]],
            c=ground_truth, cmap='viridis', s=15, alpha=0.8
        )
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
    axes[1].set_title(f'Ground Truth\n{field_name}')
    axes[1].set_aspect('equal')
    scatter2.set_clim(vmin, vmax)
    plt.colorbar(scatter2, ax=axes[1], label=f'{field_name}')
    axes[1].grid(True, alpha=0.3)

    # Error plot
    error = np.abs(nn_pred - ground_truth)
    if rotate_plot:
        scatter3 = axes[2].scatter(
            coordinates[:, varying_dims[1]], coordinates[:, varying_dims[0]],
            c=error, cmap='Reds', s=15, alpha=0.8
        )
        axes[2].set_xlabel(ylabel)
        axes[2].set_ylabel(xlabel)
    else:
        scatter3 = axes[2].scatter(
            coordinates[:, varying_dims[0]], coordinates[:, varying_dims[1]],
            c=error, cmap='Reds', s=15, alpha=0.8
        )
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'Absolute Error\nMAE: {error.mean():.4e}')
    axes[2].set_aspect('equal')
    plt.colorbar(scatter3, ax=axes[2], label='Error')
    axes[2].grid(True, alpha=0.3)

    metrics = pred_data['metrics']
    fig.suptitle(f'{pred_data["model_name"]} (R²={metrics["r2"]:.4f}, MAE={metrics["mae"]:.4e})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show(block=False)
    ui_helpers.pause()


def display_3d_test_comparison(output_key, pred_data, viz_data, ui_helpers):
    """Display 3D field comparison (simplified for now)."""
    ui_helpers.clear_screen()
    ui_helpers.print_header(f"3D COMPARISON: {pred_data['model_name']}")

    metrics = pred_data['metrics']
    print(f"\nModel: {pred_data['model_name']}")
    print(f"R² Score: {metrics['r2']:.6f}")
    print(f"MAE: {metrics['mae']:.6e}")
    print(f"RMSE: {metrics['rmse']:.6e}")

    print(f"\n[Note: 3D visualization similar to 2D - full 3D plot support can be added]")

    ui_helpers.pause()


def validate_dataset_point(dataset_dir, summary, models_dir, ui_helpers):
    """
    Validate dataset by comparing a random dataset point with a fresh Fluent simulation.
    This diagnostic tool checks if the stored dataset values match what Fluent produces
    with the same input parameters, helping identify data alignment issues.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    summary : dict
        Training summary data
    models_dir : Path
        Directory containing the trained models
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("DATASET VALIDATION: Compare Dataset vs Fluent")

    print("\nThis tool validates your dataset by:")
    print("  1. Randomly selecting a point from your dataset")
    print("  2. Running a fresh Fluent simulation with the SAME input parameters")
    print("  3. Comparing the stored dataset values vs fresh Fluent results")
    print("\nIf values don't match, it indicates a data alignment issue.")

    # Load model setup
    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    doe_config = setup_data.get('doe_configuration', {})

    # Get list of dataset files
    dataset_output_dir = dataset_dir / "dataset"
    output_files = sorted(dataset_output_dir.glob("sim_*.npz"))

    if not output_files:
        print("\n[X] No dataset files found!")
        ui_helpers.pause()
        return

    print(f"\n  Found {len(output_files)} simulations in dataset")

    # Build parameter info (must match training order)
    param_info = []
    for bc_name in sorted(doe_config.keys()):
        params = doe_config[bc_name]
        for param_name in sorted(params.keys()):
            values = params[param_name]
            param_info.append({
                'bc_name': bc_name,
                'param_name': param_name,
                'full_name': f"{bc_name}.{param_name}",
                'values': values
            })

    # User can choose specific index or random
    print("\nSelect dataset point:")
    choice = input(f"  Enter simulation number (1-{len(output_files)}) or press Enter for random: ").strip()

    if choice:
        try:
            sim_index = int(choice) - 1
            if sim_index < 0 or sim_index >= len(output_files):
                print(f"  Invalid index, using random")
                sim_index = np.random.randint(0, len(output_files))
        except ValueError:
            print(f"  Invalid input, using random")
            sim_index = np.random.randint(0, len(output_files))
    else:
        sim_index = np.random.randint(0, len(output_files))

    selected_file = output_files[sim_index]
    sim_number = sim_index + 1

    print(f"\n{'='*70}")
    print(f"  Selected: {selected_file.name} (Simulation #{sim_number})")
    print(f"{'='*70}")

    # Load the dataset point
    print(f"\nLoading dataset point...")
    dataset_data = np.load(selected_file, allow_pickle=True)

    # Get the input parameters for this simulation
    # Parameters are stored in DOE arrays - extract the values at this index
    custom_params = []
    print(f"\nInput parameters for this simulation:")
    for i, info in enumerate(param_info):
        param_value = info['values'][sim_index]
        custom_params.append(param_value)
        print(f"  {info['full_name']}: {param_value:.6f}")

    # Show dataset outputs
    print(f"\nDataset outputs (from {selected_file.name}):")
    print(f"{'='*70}")
    for key in sorted(dataset_data.files):
        if '|' in key and 'coordinates' not in key.lower():
            data_values = dataset_data[key]
            if len(data_values) == 1:
                print(f"  {key:40s}: {data_values[0]:.6f}")
            else:
                print(f"  {key:40s}: Array with {len(data_values)} points (mean={np.mean(data_values):.4f})")
    print(f"{'='*70}")

    # Confirm to run Fluent
    confirm = input("\nRun Fluent simulation with these same parameters? [y/N]: ").strip().lower()

    if confirm != 'y':
        print("\n  Validation cancelled")
        ui_helpers.pause()
        return

    # Run Fluent validation using the same function as option 2
    fluent_results = run_fluent_validation(dataset_dir, setup_data, custom_params, param_info, ui_helpers)

    if not fluent_results:
        print("\n[X] Fluent validation failed")
        ui_helpers.pause()
        return

    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON: Dataset vs Fresh Fluent Simulation")
    print(f"{'='*70}")
    print(f"{'Output':40s} {'Dataset':>15s} {'Fluent':>15s} {'Diff':>15s} {'% Error':>10s}")
    print(f"{'='*70}")

    mismatches = []

    for key in sorted(dataset_data.files):
        if '|' in key and 'coordinates' not in key.lower():
            dataset_values = dataset_data[key]

            # Find corresponding Fluent data
            if key in fluent_results:
                fluent_values = fluent_results[key]

                # Compare
                if len(dataset_values) == 1:
                    # Scalar value
                    dataset_val = dataset_values[0]
                    fluent_val = fluent_values[0] if len(fluent_values) == 1 else np.mean(fluent_values)
                    diff = fluent_val - dataset_val
                    pct_error = abs(diff / dataset_val * 100) if dataset_val != 0 else float('inf')

                    match_symbol = "✓" if abs(pct_error) < 1.0 else "✗"
                    print(f"{key:40s} {dataset_val:>15.6f} {fluent_val:>15.6f} {diff:>15.6f} {pct_error:>9.2f}% {match_symbol}")

                    if abs(pct_error) > 5.0:
                        mismatches.append((key, pct_error))

                else:
                    # Field data - compare means
                    dataset_mean = np.mean(dataset_values)
                    fluent_mean = np.mean(fluent_values)
                    diff = fluent_mean - dataset_mean
                    pct_error = abs(diff / dataset_mean * 100) if dataset_mean != 0 else float('inf')

                    match_symbol = "✓" if abs(pct_error) < 1.0 else "✗"
                    print(f"{key:40s} {dataset_mean:>15.6f} {fluent_mean:>15.6f} {diff:>15.6f} {pct_error:>9.2f}% {match_symbol}")

                    if abs(pct_error) > 5.0:
                        mismatches.append((key, pct_error))
            else:
                print(f"{key:40s} (not found in Fluent results)")

    print(f"{'='*70}")

    # Summary
    if mismatches:
        print(f"\n⚠ WARNING: {len(mismatches)} significant mismatches found (>5% error):")
        for key, error in mismatches:
            print(f"  - {key}: {error:.2f}% error")
        print(f"\n  This suggests a DATA ALIGNMENT ISSUE in your dataset!")
        print(f"  The stored dataset values don't match what Fluent produces.")
    else:
        print(f"\n✓ All values match within 5% tolerance")
        print(f"  Dataset appears to be correctly aligned with input parameters")

    ui_helpers.pause()


def compare_dataset_point_with_nn(dataset_dir, summary, models_dir, ui_helpers):
    """
    Compare NN predictions against ground truth from dataset files.
    This tool lets you select any dataset point and compare NN predictions vs stored values
    without running Fluent simulations.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    summary : dict
        Training summary data
    models_dir : Path
        Directory containing the trained models
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("DATASET COMPARISON: NN Predictions vs Ground Truth")

    print("\nThis tool:")
    print("  1. Selects a point from your dataset")
    print("  2. Runs NN predictions on that point")
    print("  3. Compares predictions vs ground truth (no Fluent needed)")

    # Load model setup
    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    doe_config = setup_data.get('doe_configuration', {})

    # Get list of dataset files
    dataset_output_dir = dataset_dir / "dataset"
    output_files = sorted(dataset_output_dir.glob("sim_*.npz"))

    if not output_files:
        print("\n[X] No dataset files found!")
        ui_helpers.pause()
        return

    print(f"\n  Found {len(output_files)} simulations in dataset")

    # Build parameter info (must match training order)
    param_info = []
    for bc_name in sorted(doe_config.keys()):
        params = doe_config[bc_name]
        for param_name in sorted(params.keys()):
            values = params[param_name]
            param_info.append({
                'bc_name': bc_name,
                'param_name': param_name,
                'full_name': f"{bc_name}.{param_name}",
                'values': values
            })

    # User can choose specific index or random
    print("\nSelect dataset point:")
    choice = input(f"  Enter simulation number (1-{len(output_files)}) or press Enter for random: ").strip()

    if choice:
        try:
            sim_index = int(choice) - 1
            if sim_index < 0 or sim_index >= len(output_files):
                print(f"  Invalid index, using random")
                sim_index = np.random.randint(0, len(output_files))
        except ValueError:
            print(f"  Invalid input, using random")
            sim_index = np.random.randint(0, len(output_files))
    else:
        sim_index = np.random.randint(0, len(output_files))

    selected_file = output_files[sim_index]
    sim_number = sim_index + 1

    print(f"\n{'='*70}")
    print(f"  Selected: {selected_file.name} (Simulation #{sim_number})")
    print(f"{'='*70}")

    # Load the dataset point
    print(f"\nLoading dataset point...")
    dataset_data = np.load(selected_file, allow_pickle=True)

    # Get the input parameters for this simulation
    # Parameters are stored in DOE arrays - extract the values at this index
    custom_params = []
    print(f"\nInput parameters for this simulation:")
    for i, info in enumerate(param_info):
        param_value = info['values'][sim_index]
        custom_params.append(param_value)
        print(f"  {info['full_name']}: {param_value:.6f}")

    # Show dataset outputs
    print(f"\nDataset outputs (ground truth from {selected_file.name}):")
    print(f"{'='*70}")
    for key in sorted(dataset_data.files):
        if '|' in key and 'coordinates' not in key.lower():
            data_values = dataset_data[key]
            if len(data_values) == 1:
                print(f"  {key:40s}: {data_values[0]:.6f}")
            else:
                print(f"  {key:40s}: Array with {len(data_values)} points (mean={np.mean(data_values):.4f})")
    print(f"{'='*70}")

    # Run NN predictions
    print("\n\nRunning NN predictions...")
    print(f"{'='*70}")

    # Convert custom params to numpy array
    input_params = np.array([custom_params])

    # Load all models and run predictions
    nn_results = {}

    # Load metadata files to identify models
    metadata_files = sorted(models_dir.glob("*_metadata.json"))

    if not metadata_files:
        print("\n[X] No trained models found!")
        ui_helpers.pause()
        return

    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        model_name = metadata_file.stem.replace('_metadata', '')
        output_type = metadata.get('output_type', 'unknown')
        output_key = metadata.get('output_key', '')

        # Load appropriate model using the .load() class method
        model_path = models_dir / model_name

        try:
            if output_type == '1D':
                model = ScalarNNModel.load(model_path)
            elif output_type == '2D':
                model = FieldNNModel.load(model_path)
            elif output_type == '3D':
                model = VolumeNNModel.load(model_path)
            else:
                print(f"  [!] Unknown output type for {model_name}: {output_type}")
                continue

            # Run prediction
            prediction = model.predict(input_params)

            # Store result
            nn_results[output_key] = prediction[0]

            print(f"  {model_name} ({output_type}): [OK]")

        except Exception as e:
            print(f"  {model_name}: [X] Error: {e}")
            continue

    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON: NN Predictions vs Dataset Ground Truth")
    print(f"{'='*70}")
    print(f"{'Output':40s} {'NN Pred':>15s} {'Ground Truth':>15s} {'MAE':>15s} {'R²':>10s}")
    print(f"{'='*70}")

    from sklearn.metrics import r2_score, mean_absolute_error

    comparison_data = {}

    for key in sorted(dataset_data.files):
        if '|' in key and 'coordinates' not in key.lower():
            ground_truth = dataset_data[key]

            # Convert dataset key format (pipe) to NN key format (underscore)
            # e.g., "bottom|temperature" -> "bottom_temperature"
            nn_key = key.replace('|', '_')

            if nn_key in nn_results:
                nn_pred = nn_results[nn_key]

                # Calculate metrics
                if len(ground_truth) == 1:
                    # Scalar value
                    gt_val = ground_truth[0]
                    nn_val = nn_pred[0] if len(nn_pred) == 1 else nn_pred
                    mae = abs(nn_val - gt_val)
                    # R² for single point is not meaningful, use relative error instead
                    rel_error = abs(nn_val - gt_val) / abs(gt_val) * 100 if gt_val != 0 else float('inf')
                    print(f"{key:40s} {nn_val:15.6f} {gt_val:15.6f} {mae:15.6e} {rel_error:9.2f}%")
                else:
                    # Field data
                    mae = mean_absolute_error(ground_truth, nn_pred)
                    r2 = r2_score(ground_truth, nn_pred)
                    nn_mean = np.mean(nn_pred)
                    gt_mean = np.mean(ground_truth)
                    print(f"{key:40s} {nn_mean:15.6f} {gt_mean:15.6f} {mae:15.6e} {r2:10.4f}")

                    # Store for visualization
                    comparison_data[key] = {
                        'nn_pred': nn_pred,
                        'ground_truth': ground_truth,
                        'r2': r2,
                        'mae': mae
                    }

    print(f"{'='*70}")

    # Offer to visualize 2D/3D fields
    if comparison_data:
        print(f"\n{len(comparison_data)} 2D/3D field outputs available for visualization")
        viz_choice = input("Visualize field comparisons? [Y/n]: ").strip().lower()

        if viz_choice != 'n':
            # Show each field comparison
            for key, data in comparison_data.items():
                print(f"\nVisualizing: {key}")
                visualize_dataset_comparison(key, data, dataset_data, ui_helpers)

    ui_helpers.pause()


def visualize_dataset_comparison(output_key, comparison_data, dataset_data, ui_helpers):
    """
    Visualize comparison between NN prediction and dataset ground truth for a 2D/3D field.

    Parameters
    ----------
    output_key : str
        Output key in pipe format (e.g., "yz-mid|temperature")
    comparison_data : dict
        Dictionary with 'nn_pred', 'ground_truth', 'r2', 'mae'
    dataset_data : numpy archive
        NPZ file data containing coordinates
    ui_helpers : module
        UI helpers module
    """
    nn_pred = comparison_data['nn_pred']
    ground_truth = comparison_data['ground_truth']
    r2 = comparison_data['r2']
    mae = comparison_data['mae']

    # Get coordinates
    coord_key = output_key.replace('temperature', 'coordinates').replace('pressure', 'coordinates')
    if coord_key not in dataset_data.files:
        # Try with location only
        location = output_key.split('|')[0]
        coord_key = f"{location}|coordinates"

    if coord_key not in dataset_data.files:
        print(f"  [!] Could not find coordinates for {output_key}")
        return

    coordinates = dataset_data[coord_key]

    # Parse field name
    field_name = output_key.split('|')[1] if '|' in output_key else output_key
    location = output_key.split('|')[0] if '|' in output_key else 'field'

    # Determine dimensionality
    is_3d = len(coordinates) > 15000  # Heuristic: 3D fields have more points

    if is_3d:
        # 3D visualization
        visualize_3d_comparison(output_key, nn_pred, ground_truth, coordinates, field_name, r2, mae)
    else:
        # 2D visualization
        visualize_2d_comparison(output_key, nn_pred, ground_truth, coordinates, field_name, location, r2, mae)

    ui_helpers.pause()


def visualize_2d_comparison(output_key, nn_pred, ground_truth, coordinates, field_name, location, r2, mae):
    """Visualize 2D field comparison."""
    # Downsample for performance
    n_points = len(coordinates)
    target_points = n_points // 2
    if n_points > target_points:
        indices = np.random.choice(n_points, target_points, replace=False)
        coordinates = coordinates[indices]
        nn_pred = nn_pred[indices]
        ground_truth = ground_truth[indices]

    # Detect which 2 dimensions vary
    variances = [np.var(coordinates[:, i]) for i in range(3)]
    varying_dims = sorted(range(3), key=lambda i: variances[i], reverse=True)[:2]
    varying_dims.sort()

    # Get axis labels
    axis_names = ['X', 'Y', 'Z']
    xlabel = f'{axis_names[varying_dims[0]]} (m)'
    ylabel = f'{axis_names[varying_dims[1]]} (m)'

    # Determine if we need to rotate the plot
    rotate_plot = 'yz' in location.lower()

    # Create 2x2 grid with 3 plots (2 on top, 1 centered on bottom)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[0, 1])  # Top right
    ax3 = fig.add_subplot(gs[1, :])  # Bottom center (spans both columns)

    # Match color scales
    vmin = min(nn_pred.min(), ground_truth.min())
    vmax = max(nn_pred.max(), ground_truth.max())

    # NN Prediction (top left)
    if rotate_plot:
        scatter1 = ax1.scatter(
            coordinates[:, varying_dims[1]], coordinates[:, varying_dims[0]],
            c=nn_pred, cmap='viridis', s=15, alpha=0.8, vmin=vmin, vmax=vmax
        )
        ax1.set_xlabel(ylabel)
        ax1.set_ylabel(xlabel)
    else:
        scatter1 = ax1.scatter(
            coordinates[:, varying_dims[0]], coordinates[:, varying_dims[1]],
            c=nn_pred, cmap='viridis', s=15, alpha=0.8, vmin=vmin, vmax=vmax
        )
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)

    ax1.set_title(f'NN Prediction\nMean: {np.mean(nn_pred):.2f}')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label=format_field_label(field_name))

    # Ground Truth (top right)
    if rotate_plot:
        scatter2 = ax2.scatter(
            coordinates[:, varying_dims[1]], coordinates[:, varying_dims[0]],
            c=ground_truth, cmap='viridis', s=15, alpha=0.8, vmin=vmin, vmax=vmax
        )
        ax2.set_xlabel(ylabel)
        ax2.set_ylabel(xlabel)
    else:
        scatter2 = ax2.scatter(
            coordinates[:, varying_dims[0]], coordinates[:, varying_dims[1]],
            c=ground_truth, cmap='viridis', s=15, alpha=0.8, vmin=vmin, vmax=vmax
        )
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)

    ax2.set_title(f'Dataset Ground Truth\nMean: {np.mean(ground_truth):.2f}')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label=format_field_label(field_name))

    # Error plot (bottom center)
    error = np.abs(nn_pred - ground_truth)
    if rotate_plot:
        scatter3 = ax3.scatter(
            coordinates[:, varying_dims[1]], coordinates[:, varying_dims[0]],
            c=error, cmap='Reds', s=15, alpha=0.8
        )
        ax3.set_xlabel(ylabel)
        ax3.set_ylabel(xlabel)
    else:
        scatter3 = ax3.scatter(
            coordinates[:, varying_dims[0]], coordinates[:, varying_dims[1]],
            c=error, cmap='Reds', s=15, alpha=0.8
        )
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)

    ax3.set_title(f'Absolute Error\nMAE: {mae:.4e}, R²: {r2:.4f}')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Error')

    fig.suptitle(f'{output_key}', fontsize=14, fontweight='bold')
    plt.show()


def visualize_3d_comparison(output_key, nn_pred, ground_truth, coordinates, field_name, r2, mae):
    """Visualize 3D field comparison."""
    # Downsample significantly for 3D performance
    n_points = len(coordinates)
    target_points = min(5000, n_points // 4)
    if n_points > target_points:
        indices = np.random.choice(n_points, target_points, replace=False)
        coordinates = coordinates[indices]
        nn_pred = nn_pred[indices]
        ground_truth = ground_truth[indices]

    # Calculate aspect ratio
    x_range = coordinates[:, 0].max() - coordinates[:, 0].min()
    y_range = coordinates[:, 1].max() - coordinates[:, 1].min()
    z_range = coordinates[:, 2].max() - coordinates[:, 2].min()
    max_range = max(x_range, y_range, z_range)
    aspect = [x_range/max_range, y_range/max_range, z_range/max_range]

    # Create 3-panel plot
    fig = plt.figure(figsize=(18, 5))

    # Match color scales
    vmin = min(nn_pred.min(), ground_truth.min())
    vmax = max(nn_pred.max(), ground_truth.max())

    # NN Prediction
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
        c=nn_pred, cmap='viridis', s=5, alpha=0.6, vmin=vmin, vmax=vmax
    )
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'NN Prediction\nMean: {np.mean(nn_pred):.2f}')
    plt.colorbar(scatter1, ax=ax1, label=format_field_label(field_name), shrink=0.7)
    ax1.set_box_aspect(aspect)

    # Ground Truth
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
        c=ground_truth, cmap='viridis', s=5, alpha=0.6, vmin=vmin, vmax=vmax
    )
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'Dataset Ground Truth\nMean: {np.mean(ground_truth):.2f}')
    plt.colorbar(scatter2, ax=ax2, label=format_field_label(field_name), shrink=0.7)
    ax2.set_box_aspect(aspect)

    # Error plot
    error = np.abs(nn_pred - ground_truth)
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
        c=error, cmap='Reds', s=5, alpha=0.6
    )
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title(f'Absolute Error\nMAE: {mae:.4e}, R²: {r2:.4f}')
    plt.colorbar(scatter3, ax=ax3, label='Error', shrink=0.7)
    ax3.set_box_aspect(aspect)

    fig.suptitle(f'{output_key}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
