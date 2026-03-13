"""
Multi-Model Trainer Module
===========================
Trains specialized neural network models (1D, 2D, 3D) for each output parameter.
Automatically detects output dimensionality and uses appropriate model type.
"""

import json
from pathlib import Path
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO and WARNING messages
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*tf.function retracing.*')

from .scalar_nn_model import ScalarNNModel
from .field_nn_model import FieldNNModel
from .volume_nn_model import VolumeNNModel


def plot_loss_curves(history, model_name, save_dir, output_type):
    """
    Plot and save training/validation loss curves for a single model.

    Parameters
    ----------
    history : keras.callbacks.History
        Training history object containing loss values
    model_name : str
        Name of the model
    save_dir : Path
        Directory to save the plot
    output_type : str
        Type of model ('1D', '2D', '3D')
    """
    plt.figure(figsize=(10, 6))

    # Plot training loss
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Training Loss')

    # Plot validation loss if available
    if 'val_loss' in history.history:
        plt.plot(epochs, history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'Training History: {model_name}\n({output_type} Model)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = save_dir / f"{model_name}_loss_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Loss curve saved: {plot_path.name}")


def plot_combined_loss_curves(all_histories, save_dir):
    """
    Plot combined training and validation loss curves for all models.
    Creates separate plots for 1D predictions and 2D predictions.

    Parameters
    ----------
    all_histories : list of tuples
        List of (model_name, history, output_type) tuples
    save_dir : Path
        Directory to save the plot
    """
    # Separate histories by output type
    histories_1d = [(name, hist, otype) for name, hist, otype in all_histories if otype == '1D']
    histories_2d = [(name, hist, otype) for name, hist, otype in all_histories if otype == '2D']
    histories_3d = [(name, hist, otype) for name, hist, otype in all_histories if otype == '3D']

    # Combine 3D with 2D if any exist
    histories_2d.extend(histories_3d)

    # Plot 1D models
    if histories_1d:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for model_name, history, output_type in histories_1d:
            epochs = range(1, len(history.history['loss']) + 1)

            # Plot training loss (solid line)
            ax.plot(epochs, history.history['loss'], '-', linewidth=2,
                   alpha=0.8, label=f"{model_name} (train)")

            # Plot validation loss if available (dashed line, same color)
            if 'val_loss' in history.history:
                ax.plot(epochs, history.history['val_loss'], '--', linewidth=2,
                       alpha=0.8, label=f"{model_name} (val)")

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Training & Validation Loss - 1D Predictions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plot_path = save_dir / "combined_loss_curves_1D.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Combined 1D loss curve saved: {plot_path.name}")

    # Plot 2D models
    if histories_2d:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for model_name, history, output_type in histories_2d:
            epochs = range(1, len(history.history['loss']) + 1)

            # Plot training loss (solid line)
            ax.plot(epochs, history.history['loss'], '-', linewidth=2,
                   alpha=0.8, label=f"{model_name} (train)")

            # Plot validation loss if available (dashed line, same color)
            if 'val_loss' in history.history:
                ax.plot(epochs, history.history['val_loss'], '--', linewidth=2,
                       alpha=0.8, label=f"{model_name} (val)")

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Training & Validation Loss - 2D Predictions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plot_path = save_dir / "combined_loss_curves_2D.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Combined 2D loss curve saved: {plot_path.name}")


def detect_output_type(data_shape):
    """
    Detect if output is 1D (scalar), 2D (field), or 3D (volume).

    Parameters
    ----------
    data_shape : tuple
        Shape of the output data for a single sample

    Returns
    -------
    str
        Output type: '1D', '2D', or '3D'
    int
        Suggested number of POD modes (if applicable)
    """
    n_points = data_shape[0] if len(data_shape) > 0 else 1

    if n_points <= 100:
        # Small number of points - scalar/point data
        return '1D', 0
    elif n_points <= 100000:
        # Medium size - likely 2D cut plane (surfaces, slices, etc.)
        # Increased threshold to accommodate larger 2D surfaces
        # Use fewer modes - will be auto-adjusted based on n_samples
        return '2D', min(10, n_points // 100)
    else:
        # Very large size - likely 3D volume
        # Use fewer modes - will be auto-adjusted based on n_samples
        return '3D', min(15, n_points // 1000)


def load_training_data(dataset_dir):
    """
    Load all simulation data from case directory.

    Parameters
    ----------
    dataset_dir : Path
        Case directory

    Returns
    -------
    dict
        Dictionary with:
        - 'parameters': Input parameters array (n_samples, n_params)
        - 'outputs': Dict of output arrays by location name
        - 'output_info': Dict with metadata for each output
    """
    print(f"\nLoading training data from: {dataset_dir.name}")

    # Load model setup to get DOE configuration
    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    # Load output parameters configuration
    output_params_file = dataset_dir / "output_parameters.json"
    with open(output_params_file, 'r') as f:
        output_params = json.load(f)

    # Get simulation output files
    dataset_output_dir = dataset_dir / "dataset"
    output_files = sorted(dataset_output_dir.glob("sim_*.npz"))

    if not output_files:
        raise ValueError(f"No simulation files found in {dataset_output_dir}")

    print(f"  Found {len(output_files)} simulation files")

    # Reconstruct input parameters from DOE
    doe_config = setup_data.get('doe_configuration', {})
    param_names = []
    param_values = []

    # Iterate in original insertion order to match the existing dataset
    for bc_name in doe_config.keys():
        params = doe_config[bc_name]
        for param_name in params.keys():
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

    # Load outputs
    output_data = {}
    output_info = {}

    # First pass: scan all files to detect shape variations
    from collections import Counter, defaultdict

    print(f"  Scanning simulation files for shape consistency...")

    # Dictionary to track shapes for each field: {model_key: {shape: [file_indices]}}
    field_shapes = defaultdict(lambda: defaultdict(list))

    # Scan ALL files to build complete shape map
    for i, output_file_path in enumerate(output_files):
        data = np.load(output_file_path, allow_pickle=True)

        for output_location, field_list in output_params.items():
            for field_name in field_list:
                # Skip coordinates
                if field_name.lower() == 'coordinates':
                    continue

                npz_key = f"{output_location}|{field_name}"
                model_key = f"{output_location}_{field_name}"

                if npz_key in data.files:
                    shape = len(data[npz_key])
                    field_shapes[model_key][shape].append(i)

    # Check for shape inconsistencies and let user choose
    shape_decisions = {}

    for model_key, shapes_dict in field_shapes.items():
        if len(shapes_dict) > 1:
            # Multiple shapes found - present menu
            print(f"\n  ⚠ Shape inconsistency detected for '{model_key}':")
            print(f"  {'='*60}")

            options = []
            for idx, (shape, file_indices) in enumerate(sorted(shapes_dict.items()), 1):
                print(f"  [{idx}] Shape ({shape},) - {len(file_indices)} files")
                print(f"      Files: sim_{file_indices[0]+1:04d} to sim_{file_indices[-1]+1:04d}")
                options.append(shape)

            print(f"  {'='*60}")

            while True:
                try:
                    choice = input(f"  Select which shape to use [1-{len(options)}]: ")
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(options):
                        selected_shape = options[choice_idx]
                        shape_decisions[model_key] = selected_shape
                        print(f"  ✓ Using shape ({selected_shape},) for '{model_key}'")
                        break
                    else:
                        print(f"  Invalid choice. Please enter 1-{len(options)}")
                except ValueError:
                    print(f"  Invalid input. Please enter 1-{len(options)}")
                except KeyboardInterrupt:
                    print("\n  Operation cancelled by user")
                    raise
        else:
            # Only one shape - use it
            shape_decisions[model_key] = list(shapes_dict.keys())[0]

    # Build output_info with user-selected shapes
    for output_location, field_list in output_params.items():
        for field_name in field_list:
            if field_name.lower() == 'coordinates':
                continue

            npz_key = f"{output_location}|{field_name}"
            model_key = f"{output_location}_{field_name}"

            if model_key in shape_decisions:
                expected_n_points = shape_decisions[model_key]

                # Load a sample for type detection
                sample_file = np.load(output_files[0], allow_pickle=True)
                sample_values = sample_file[npz_key]

                output_type, n_modes = detect_output_type(sample_values.shape)

                output_info[model_key] = {
                    'location': output_location,
                    'field': field_name,
                    'npz_key': npz_key,
                    'type': output_type,
                    'n_modes': n_modes,
                    'n_points': expected_n_points
                }
            else:
                print(f"  ⚠ Warning: Key '{npz_key}' not found in simulation data")

    # Optional: Exclude specific simulation ranges (for corrupted data)
    exclude_start = None
    exclude_end = None

    print(f"\n{'='*70}")
    print("DATA EXCLUSION (Optional)")
    print(f"{'='*70}")
    print("If you know certain simulations are corrupted, you can exclude them.")
    print("Example: Exclude simulations 1-2500 to remove bad data")
    print("\nLeave blank to include all data.")

    exclude_input = input("\nExclude simulation range? (format: start-end, e.g., '1-2500'): ").strip()

    if exclude_input:
        try:
            parts = exclude_input.split('-')
            if len(parts) == 2:
                exclude_start = int(parts[0])
                exclude_end = int(parts[1])
                print(f"  Will exclude simulations {exclude_start} to {exclude_end}")
            else:
                print(f"  Invalid format, not excluding any data")
        except ValueError:
            print(f"  Invalid format, not excluding any data")

    # Validate all files for shape consistency across ALL fields
    print(f"\n  Validating {len(output_files)} simulation files...")
    valid_file_indices = []
    invalid_files = []

    for i, output_file in enumerate(output_files):
        # Check if this simulation index should be excluded
        sim_number = i + 1  # Simulation numbers are 1-indexed
        if exclude_start is not None and exclude_end is not None:
            if exclude_start <= sim_number <= exclude_end:
                # Skip this file - it's in the exclusion range
                invalid_files.append((i, output_file.name, 'excluded', 'user_excluded', 'N/A'))
                continue

        data = np.load(output_file, allow_pickle=True)
        is_valid = True

        for model_key, info in output_info.items():
            npz_key = info['npz_key']
            expected_shape = info['n_points']

            if npz_key in data.files:
                actual_shape = len(data[npz_key])
                if actual_shape != expected_shape:
                    is_valid = False
                    invalid_files.append((i, output_file.name, model_key, expected_shape, actual_shape))
                    break
            else:
                is_valid = False
                invalid_files.append((i, output_file.name, model_key, 'missing', 'N/A'))
                break

        if is_valid:
            valid_file_indices.append(i)

    # Report validation results
    if invalid_files:
        excluded_count = sum(1 for f in invalid_files if f[2] == 'excluded')
        shape_mismatch_count = len(invalid_files) - excluded_count

        if excluded_count > 0:
            print(f"  ⚠ Excluded {excluded_count} files based on user-specified range")

        if shape_mismatch_count > 0:
            print(f"  ⚠ Warning: {shape_mismatch_count} files have inconsistent shapes and will be excluded:")
            shown = 0
            for idx, fname, model_key, expected, actual in invalid_files:
                if model_key != 'excluded' and shown < 5:
                    print(f"    - {fname} ({model_key}): expected {expected}, got {actual}")
                    shown += 1
            if shape_mismatch_count > 5:
                print(f"    ... and {shape_mismatch_count - 5} more")

    print(f"  Using {len(valid_file_indices)} valid files out of {len(output_files)}")

    # Filter parameters to match valid files
    X_params = np.array(param_combinations[:len(output_files)])
    X_params = X_params[valid_file_indices]

    print(f"  Input parameters: {X_params.shape}")

    # Second pass: load all data from valid files only
    for model_key, info in output_info.items():
        output_arrays = []
        npz_key = info['npz_key']

        for i in valid_file_indices:
            output_file = output_files[i]
            data = np.load(output_file, allow_pickle=True)

            # Load data directly using NPZ key
            values = data[npz_key]
            output_arrays.append(values)

        # Convert to numpy array (all have same shape now)
        output_data[model_key] = np.array(output_arrays)

    return {
        'parameters': X_params,
        'outputs': output_data,
        'output_info': output_info,
        'param_names': param_names
    }


def select_model_architecture(ui_helpers):
    """
    Menu for selecting model architecture for each output type.

    Parameters
    ----------
    ui_helpers : module
        UI helpers module

    Returns
    -------
    dict
        Model architecture selection: {'1D': str, '2D': str, '3D': str}
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("MODEL ARCHITECTURE SELECTION")

    print("\nSelect model architecture for each output type:")
    print("This determines which neural network architecture to use for training.")

    model_selection = {}

    # 1D (Scalar) model selection
    print(f"\n{'='*70}")
    print("1D OUTPUTS (Scalars - single values)")
    print(f"{'='*70}")
    print("  [1] Feedforward Neural Network (Default)")
    print("      - Simple dense layers")
    print("      - Fast training, good for scalar predictions")
    print(f"{'='*70}")

    choice = input("Select 1D model architecture [1]: ").strip() or "1"
    model_selection['1D'] = 'ScalarNN'  # Only one option for now
    print(f"  Selected: Feedforward Neural Network")

    # 2D (Field) model selection
    print(f"\n{'='*70}")
    print("2D OUTPUTS (Fields - surfaces, cut planes)")
    print(f"{'='*70}")
    print("  [1] POD + Neural Network (Default)")
    print("      - Dimensionality reduction using POD/PCA")
    print("      - Predicts POD modes, reconstructs full field")
    print("      - Efficient for smooth fields")
    print("  [2] Convolutional Neural Network (CNN)")
    print("      - Direct image-to-image prediction")
    print("      - Better for sharp gradients/discontinuities")
    print("      - Requires structured grid (coming soon)")
    print(f"{'='*70}")

    choice = input("Select 2D model architecture [1]: ").strip() or "1"
    if choice == "1":
        model_selection['2D'] = 'FieldNN'
        print(f"  Selected: POD + Neural Network")
    elif choice == "2":
        model_selection['2D'] = 'CNN2D'
        print(f"  Selected: Convolutional Neural Network (2D)")
        print(f"  Note: CNN implementation coming soon - falling back to POD+NN")
        model_selection['2D'] = 'FieldNN'
    else:
        model_selection['2D'] = 'FieldNN'
        print(f"  Invalid choice - using default: POD + Neural Network")

    # 3D (Volume) model selection
    print(f"\n{'='*70}")
    print("3D OUTPUTS (Volumes - full 3D fields)")
    print(f"{'='*70}")
    print("  [1] POD + Neural Network (Default)")
    print("      - Dimensionality reduction using POD/PCA")
    print("      - Predicts POD modes, reconstructs full volume")
    print("      - Efficient for large volumes")
    print("  [2] 3D Convolutional Neural Network (3D-CNN)")
    print("      - Direct volume-to-volume prediction")
    print("      - Better for complex 3D structures")
    print("      - Requires structured grid (coming soon)")
    print(f"{'='*70}")

    choice = input("Select 3D model architecture [1]: ").strip() or "1"
    if choice == "1":
        model_selection['3D'] = 'VolumeNN'
        print(f"  Selected: POD + Neural Network")
    elif choice == "2":
        model_selection['3D'] = 'CNN3D'
        print(f"  Selected: 3D Convolutional Neural Network")
        print(f"  Note: 3D-CNN implementation coming soon - falling back to POD+NN")
        model_selection['3D'] = 'VolumeNN'
    else:
        model_selection['3D'] = 'VolumeNN'
        print(f"  Invalid choice - using default: POD + Neural Network")

    print(f"\n{'='*70}")
    ui_helpers.pause()

    return model_selection


def train_all_models(dataset_dir, ui_helpers, test_size=0.2, epochs=500):
    """
    Train models for all outputs in a case.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    ui_helpers : module
        UI helpers module
    test_size : float
        Fraction of data for testing
    epochs : int
        Training epochs
    """
    # Ask for model name
    ui_helpers.clear_screen()
    ui_helpers.print_header("TRAIN SURROGATE MODELS")

    print("\nEnter a name for this model set (e.g., 'baseline', 'high_fidelity', 'test1')")
    print("This will create a folder to store all trained models.")
    model_name = input("\nModel name: ").strip()

    if not model_name:
        print("\n✗ Model name cannot be empty")
        ui_helpers.pause()
        return

    # Validate model name (no special characters that would break file paths)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
        print("\n✗ Model name can only contain letters, numbers, underscores, and hyphens")
        ui_helpers.pause()
        return

    # Select model architectures
    model_selection = select_model_architecture(ui_helpers)

    ui_helpers.clear_screen()
    ui_helpers.print_header(f"TRAIN SURROGATE MODELS: {model_name}")

    try:
        # Load data
        data = load_training_data(dataset_dir)
        X_params = data['parameters']
        outputs = data['outputs']
        output_info = data['output_info']

        print(f"\n{'='*70}")
        print(f"Training Configuration")
        print(f"{'='*70}")
        print(f"  Input samples: {len(X_params)}")
        print(f"  Input parameters: {X_params.shape[1]}")
        print(f"  Outputs to train: {len(outputs)}")
        print(f"  Test split: {test_size*100:.0f}%")
        print(f"  Epochs: {epochs}")
        print(f"\n  Model Architectures:")
        print(f"    1D (Scalars): {model_selection['1D']}")
        print(f"    2D (Fields):  {model_selection['2D']}")
        print(f"    3D (Volumes): {model_selection['3D']}")

        # Split data
        train_idx, test_idx = train_test_split(
            np.arange(len(X_params)),
            test_size=test_size,
            random_state=42
        )

        print(f"  Train samples: {len(train_idx)}")
        print(f"  Test samples: {len(test_idx)}")

        # Create models directory with model name
        models_dir = dataset_dir / model_name
        models_dir.mkdir(exist_ok=True)

        print(f"\n  Model storage: {models_dir}")
        print(f"{'='*70}\n")

        # Track model counts for naming (in case of duplicates)
        model_counts = {}
        trained_models = []
        all_histories = []  # Store all training histories for combined plot

        # Train each output
        for output_key, output_data in outputs.items():
            info = output_info[output_key]
            output_type = info['type']
            field_name = info['field']
            location = info['location']

            # Use descriptive name: location_field (e.g., "yz-mid_temperature")
            # Only add index if there are duplicates
            base_name = f"{location}_{field_name}"

            if base_name not in model_counts:
                model_counts[base_name] = 0
            model_counts[base_name] += 1

            if model_counts[base_name] > 1:
                model_name = f"{base_name}_{model_counts[base_name]}"
            else:
                model_name = base_name

            print(f"\n{'='*70}")
            print(f"Training model: {model_name}")
            print(f"  Architecture: {model_selection[output_type]}")
            print(f"{'='*70}")

            # Create appropriate model based on selection
            if output_type == '1D':
                if model_selection['1D'] == 'ScalarNN':
                    model = ScalarNNModel(field_name=output_key)
                else:
                    raise ValueError(f"Unknown 1D model type: {model_selection['1D']}")

            elif output_type == '2D':
                if model_selection['2D'] == 'FieldNN':
                    model = FieldNNModel(n_modes=info['n_modes'], field_name=output_key)
                elif model_selection['2D'] == 'CNN2D':
                    # CNN implementation placeholder
                    raise NotImplementedError("CNN2D not yet implemented - use FieldNN")
                else:
                    raise ValueError(f"Unknown 2D model type: {model_selection['2D']}")

            else:  # 3D
                if model_selection['3D'] == 'VolumeNN':
                    model = VolumeNNModel(n_modes=info['n_modes'], field_name=output_key)
                elif model_selection['3D'] == 'CNN3D':
                    # 3D-CNN implementation placeholder
                    raise NotImplementedError("CNN3D not yet implemented - use VolumeNN")
                else:
                    raise ValueError(f"Unknown 3D model type: {model_selection['3D']}")

            # Train with validation split for monitoring convergence
            # Use 20% of training data for validation during training
            model.fit(
                X_params[train_idx],
                output_data[train_idx],
                validation_split=0.2,  # Use 20% of train data for validation monitoring
                epochs=epochs,
                verbose=0
            )

            # Evaluate
            print(f"\nEvaluating on train set...")
            train_metrics = model.evaluate(X_params[train_idx], output_data[train_idx], 'train')

            print(f"Evaluating on test set...")
            test_metrics = model.evaluate(X_params[test_idx], output_data[test_idx], 'test')

            print(f"\n  Train R²: {train_metrics['r2']:.4f}")
            print(f"  Test R²:  {test_metrics['r2']:.4f}")
            print(f"  Test MAE: {test_metrics['mae']:.4f}")

            # Save model
            model_path = models_dir / model_name
            model.save(model_path)

            # Plot and save training/validation loss curves
            if hasattr(model, 'history') and model.history is not None:
                plot_loss_curves(model.history, model_name, models_dir, output_type)
                all_histories.append((model_name, model.history, output_type))

            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_architecture': model_selection[output_type],  # Store which architecture was used
                'output_key': output_key,
                'output_type': output_type,
                'field_name': field_name,
                'location': info['location'],
                'npz_key': info['npz_key'],  # CRITICAL: Store NPZ key for Fluent comparison
                'n_points': info['n_points'],
                'train_metrics': {k: float(v) if not isinstance(v, list) else v
                                  for k, v in train_metrics.items()},
                'test_metrics': {k: float(v) if not isinstance(v, list) else v
                                 for k, v in test_metrics.items()},
                'trained_date': datetime.now().isoformat(),
                'n_train_samples': len(train_idx),
                'n_test_samples': len(test_idx)
            }

            if output_type in ['2D', '3D']:
                metadata['n_modes'] = info['n_modes']
                metadata['variance_explained'] = float(model.variance_explained.sum())

            with open(models_dir / f"{model_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            trained_models.append(metadata)

        # Save training summary
        summary = {
            'case_name': dataset_dir.name,
            'trained_date': datetime.now().isoformat(),
            'n_models': len(trained_models),
            'n_train_samples': len(train_idx),
            'n_test_samples': len(test_idx),
            'test_split': test_size,
            'epochs': epochs,
            'test_indices': test_idx.tolist(),  # Save test indices for visualization
            'train_indices': train_idx.tolist(),  # Save train indices for reference
            'models': trained_models
        }

        with open(models_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Create combined loss curves plot for all models
        if all_histories:
            plot_combined_loss_curves(all_histories, models_dir)

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"\nTrained {len(trained_models)} models:")
        for model_meta in trained_models:
            print(f"  - {model_meta['model_name']:30s} (Test R²: {model_meta['test_metrics']['r2']:.4f})")
        print(f"\nModels saved to: {models_dir}")
        print(f"Loss curves saved: {len(all_histories)} individual plots + 1 combined plot")

    except Exception as e:
        print(f"\n[X] Error during training: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()


def train_model_menu(dataset_dir, ui_helpers):
    """
    Interactive menu for model training.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    ui_helpers : module
        UI helpers module
    """
    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("MODEL TRAINING")

        print(f"\nCase: {dataset_dir.name}")

        # Check if data exists
        dataset_output_dir = dataset_dir / "dataset"
        if not dataset_output_dir.exists():
            print("\n[X] No simulation data found. Run simulations first.")
            ui_helpers.pause()
            return

        output_files = list(dataset_output_dir.glob("sim_*.npz"))
        if not output_files:
            print("\n[X] No simulation files found. Run simulations first.")
            ui_helpers.pause()
            return

        print(f"Simulation files: {len(output_files)}")

        # Check for existing models
        models_dir = dataset_dir / "models"
        existing_models = []
        if models_dir.exists():
            existing_models = list(models_dir.glob("*_metadata.json"))

        if existing_models:
            print(f"Existing models: {len(existing_models)}")

        print(f"\n{'='*70}")
        print("  [1] Train New Models (All Outputs)")
        print("  [2] View Existing Models")
        print("  [0] Back")
        print("="*70)

        choice = ui_helpers.get_choice(2)

        if choice == 0:
            return
        elif choice == 1:
            # Get training parameters
            print("\nTraining Parameters:")

            try:
                test_size_input = input("  Test split (0-1, default 0.2): ").strip()
                test_size = float(test_size_input) if test_size_input else 0.2
                test_size = max(0.1, min(0.5, test_size))

                epochs_input = input("  Epochs (default 500): ").strip()
                epochs = int(epochs_input) if epochs_input else 500
                epochs = max(10, min(2000, epochs))

            except ValueError:
                print("\n[X] Invalid input. Using defaults.")
                test_size = 0.2
                epochs = 500
                ui_helpers.pause()

            # Confirm
            confirm = input(f"\nTrain models with test_size={test_size}, epochs={epochs}? [y/N]: ").strip().lower()
            if confirm == 'y':
                train_all_models(dataset_dir, ui_helpers, test_size=test_size, epochs=epochs)

        elif choice == 2:
            view_existing_models(dataset_dir, ui_helpers)


def view_existing_models(dataset_dir, ui_helpers):
    """
    View information about existing trained models.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("EXISTING MODELS")

    models_dir = dataset_dir / "models"

    if not models_dir.exists():
        print("\n[X] No models directory found.")
        ui_helpers.pause()
        return

    # Look for training summary
    summary_file = models_dir / "training_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print(f"\nCase: {summary['case_name']}")
        print(f"Trained: {summary['trained_date']}")
        print(f"Models: {summary['n_models']}")
        print(f"Training samples: {summary['n_train_samples']}")
        print(f"Test samples: {summary['n_test_samples']}")

        print(f"\n{'='*70}")
        print(f"{'Model Name':<35s} {'Type':<6s} {'Test R²':<10s} {'Test MAE':<10s}")
        print(f"{'='*70}")

        for model_meta in summary['models']:
            model_name = model_meta['model_name']
            model_type = model_meta['output_type']
            test_r2 = model_meta['test_metrics']['r2']
            test_mae = model_meta['test_metrics']['mae']

            print(f"{model_name:<35s} {model_type:<6s} {test_r2:>9.4f} {test_mae:>9.4f}")
    else:
        print("\n[X] No training summary found.")
        print("Looking for individual model metadata files...")

        metadata_files = list(models_dir.glob("*_metadata.json"))
        if metadata_files:
            print(f"\nFound {len(metadata_files)} models:")
            for meta_file in metadata_files:
                print(f"  - {meta_file.stem}")
        else:
            print("\nNo models found.")

    ui_helpers.pause()
