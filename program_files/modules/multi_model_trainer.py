"""
Multi-Model Trainer Module
===========================
Trains surrogate neural network models for each output parameter.
Automatically detects output dimensionality and uses appropriate preset.
All functions are GUI-agnostic.
"""

import logging
import json
import itertools
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .surrogate_nn import SurrogateNN
from .pod_reducer import PODReducer
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


def plot_loss_curves(history, model_name, save_dir, output_type):
    """Plot and save training/validation loss curves for a single model."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(epochs, history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'Training History: {model_name}\n({output_type} Model)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = save_dir / f"{model_name}_loss_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss curve saved: {plot_path.name}")


def plot_combined_loss_curves(all_histories, save_dir):
    """Plot combined loss curves for all models, separated by 1D and 2D/3D."""
    histories_1d = [(n, h, t) for n, h, t in all_histories if t == '1D']
    histories_2d = [(n, h, t) for n, h, t in all_histories if t in ('2D', '3D')]

    for label, histories in [('1D', histories_1d), ('2D', histories_2d)]:
        if not histories:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for model_name, history, output_type in histories:
            epochs = range(1, len(history.history['loss']) + 1)
            ax.plot(epochs, history.history['loss'], '-', linewidth=2, alpha=0.8, label=f"{model_name} (train)")
            if 'val_loss' in history.history:
                ax.plot(epochs, history.history['val_loss'], '--', linewidth=2, alpha=0.8, label=f"{model_name} (val)")
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title(f'Training & Validation Loss - {label} Predictions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        plot_path = save_dir / f"combined_loss_curves_{label}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Combined {label} loss curve saved: {plot_path.name}")


def detect_output_type(data_shape):
    """
    Detect if output is 1D (scalar), 2D (field), or 3D (volume).

    Returns
    -------
    (str, int)
        Output type ('1D', '2D', '3D') and suggested number of POD modes
    """
    n_points = data_shape[0] if len(data_shape) > 0 else 1
    if n_points <= 100:
        return '1D', 0
    elif n_points <= 100000:
        return '2D', min(10, n_points // 100)
    else:
        return '3D', min(15, n_points // 1000)


def load_training_data(dataset_dir, exclude_range=None):
    """
    Load all simulation data from case directory.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    exclude_range : tuple of (int, int), optional
        Range of simulation numbers to exclude (start, end), 1-indexed

    Returns
    -------
    dict
        'parameters': np.ndarray (n_samples, n_params)
        'outputs': dict of np.ndarray by model key
        'output_info': dict of metadata per output
        'param_names': list of str
    """
    logger.info(f"Loading training data from: {dataset_dir.name}")

    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    output_params_file = dataset_dir / "output_parameters.json"
    with open(output_params_file, 'r') as f:
        output_params_raw = json.load(f)

    # Support both new format {'outputs': [...]} and legacy {location: [fields]}
    if 'outputs' in output_params_raw:
        output_params = {}
        for out in output_params_raw['outputs']:
            fields = out.get('field_variables', [])
            if not fields and out.get('category') == 'Report Definition':
                # Report defs store a single scalar as 'value'
                fields = ['value']
            output_params[out['name']] = fields
    else:
        output_params = output_params_raw

    dataset_output_dir = dataset_dir / "dataset"
    output_files = sorted(dataset_output_dir.glob("sim_*.npz"))

    if not output_files:
        raise ValueError(f"No simulation files found in {dataset_output_dir}")

    logger.info(f"Found {len(output_files)} simulation files")

    # Reconstruct input parameters from DOE
    # Try new format (doe_samples.json) first, fall back to legacy doe_configuration
    doe_samples_file = dataset_dir / "doe_samples.json"
    if doe_samples_file.exists():
        with open(doe_samples_file, 'r') as f:
            doe_data = json.load(f)
        samples = doe_data.get('samples', [])
        if samples:
            param_names = sorted(samples[0].keys())
            param_combinations = [tuple(s[k] for k in param_names) for s in samples]
        else:
            param_names = []
            param_combinations = []
    else:
        doe_config = setup_data.get('doe_configuration', {})
        param_names = []
        param_values = []
        for bc_name in sorted(doe_config.keys()):
            params = doe_config[bc_name]
            for param_name in sorted(params.keys()):
                values = params[param_name]
                param_names.append(f"{bc_name}.{param_name}")
                param_values.append(values)
        array_lengths = [len(arr) for arr in param_values]
        if len(set(array_lengths)) == 1:
            param_combinations = list(zip(*param_values))
        else:
            param_combinations = list(itertools.product(*param_values))

    # First pass: scan all files to detect shape variations
    logger.info("Scanning simulation files for shape consistency...")
    field_shapes = defaultdict(lambda: defaultdict(list))

    for i, output_file_path in enumerate(output_files):
        data = np.load(output_file_path, allow_pickle=True)
        for output_location, field_list in output_params.items():
            for field_name in field_list:
                npz_key = f"{output_location}|{field_name}"
                model_key = f"{output_location}_{field_name}"
                if npz_key in data.files:
                    shape = len(data[npz_key])
                    field_shapes[model_key][shape].append(i)

    # Auto-select most common shape for each field (no interactive prompt)
    shape_decisions = {}
    for model_key, shapes_dict in field_shapes.items():
        if len(shapes_dict) > 1:
            # Pick the shape with the most files
            best_shape = max(shapes_dict.keys(), key=lambda s: len(shapes_dict[s]))
            excluded_count = sum(len(v) for k, v in shapes_dict.items() if k != best_shape)
            logger.warning(f"Shape inconsistency for '{model_key}': "
                           f"using shape ({best_shape},), excluding {excluded_count} files with different shapes")
            shape_decisions[model_key] = best_shape
        else:
            shape_decisions[model_key] = list(shapes_dict.keys())[0]

    # Build output_info
    output_info = {}
    for output_location, field_list in output_params.items():
        for field_name in field_list:
            npz_key = f"{output_location}|{field_name}"
            model_key = f"{output_location}_{field_name}"

            if model_key in shape_decisions:
                expected_n_points = shape_decisions[model_key]
                sample_file = np.load(output_files[0], allow_pickle=True)
                if npz_key in sample_file.files:
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
                logger.warning(f"Key '{npz_key}' not found in simulation data")

    # Parse sim_id from each filename (1-indexed → 0-indexed into param_combinations).
    # This handles gaps (failed sims) correctly.
    file_sim_ids = []
    for f in output_files:
        try:
            sid = int(f.stem.split('_')[1])
            file_sim_ids.append(sid - 1)  # 0-indexed
        except (ValueError, IndexError):
            logger.warning(f"Could not parse sim_id from {f.name}, skipping")
            file_sim_ids.append(None)

    # Validate all files
    logger.info(f"Validating {len(output_files)} simulation files...")
    valid_file_indices = []
    invalid_count = 0
    excluded_count = 0

    for i, output_file in enumerate(output_files):
        if file_sim_ids[i] is None:
            invalid_count += 1
            continue

        sim_number = file_sim_ids[i] + 1  # use actual sim_id, not enumerate index
        if exclude_range and exclude_range[0] <= sim_number <= exclude_range[1]:
            excluded_count += 1
            continue

        data = np.load(output_file, allow_pickle=True)
        is_valid = True
        for model_key, info in output_info.items():
            npz_key = info['npz_key']
            if npz_key in data.files:
                if len(data[npz_key]) != info['n_points']:
                    is_valid = False
                    break
            else:
                is_valid = False
                break

        if is_valid:
            valid_file_indices.append(i)
        else:
            invalid_count += 1

    if excluded_count > 0:
        logger.info(f"Excluded {excluded_count} files from user-specified range")
    if invalid_count > 0:
        logger.warning(f"{invalid_count} files have inconsistent shapes and were excluded")
    logger.info(f"Using {len(valid_file_indices)} valid files out of {len(output_files)}")

    # Build parameter array using sim_id → doe_samples mapping
    # (NOT the order in output_files — handles gaps from failed sims)
    X_params_list = []
    for i in valid_file_indices:
        param_idx = file_sim_ids[i]
        if param_idx >= len(param_combinations):
            logger.error(f"sim_{param_idx + 1:04d}.npz has no matching DOE sample (sample index {param_idx})")
            continue
        X_params_list.append(param_combinations[param_idx])
    X_params = np.array(X_params_list)
    logger.info(f"Input parameters: {X_params.shape}")

    # Log a few samples for sanity-checking alignment
    if len(X_params) > 0:
        logger.info(f"Param names: {param_names}")
        logger.info(f"First sim: sim_{file_sim_ids[valid_file_indices[0]] + 1:04d} "
                    f"params={X_params[0].tolist()}")
        if len(X_params) > 1:
            logger.info(f"Last sim: sim_{file_sim_ids[valid_file_indices[-1]] + 1:04d} "
                        f"params={X_params[-1].tolist()}")

    # Second pass: load output data from valid files (matching X_params order)
    output_data = {}
    for model_key, info in output_info.items():
        output_arrays = []
        npz_key = info['npz_key']
        for i in valid_file_indices:
            param_idx = file_sim_ids[i]
            if param_idx >= len(param_combinations):
                continue
            data = np.load(output_files[i], allow_pickle=True)
            output_arrays.append(data[npz_key])
        output_data[model_key] = np.array(output_arrays)

    return {
        'parameters': X_params,
        'outputs': output_data,
        'output_info': output_info,
        'param_names': param_names
    }


def train_all_models(dataset_dir, model_name, model_selection=None,
                     test_size=0.2, epochs=500, exclude_range=None,
                     on_progress=None, output_filter=None, on_epoch=None):
    """
    Train models for all outputs in a case.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    model_name : str
        Name for the model set (creates a folder)
    model_selection : dict, optional
        Architecture selection per type: {'1D': str, '2D': str, '3D': str}
        Defaults to POD+NN for all types.
    test_size : float
        Fraction of data for testing
    epochs : int
        Training epochs
    exclude_range : tuple of (int, int), optional
        Range of simulation numbers to exclude
    on_progress : callable, optional
        Called as on_progress(model_name, status_str) for each model

    Returns
    -------
    dict
        Training summary (same data written to training_summary.json)
    """
    if not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
        raise ValueError("Model name can only contain letters, numbers, underscores, and hyphens")

    if model_selection is None:
        model_selection = {'1D': 'SurrogateNN', '2D': 'POD+NN', '3D': 'POD+NN'}

    # Load data
    data = load_training_data(dataset_dir, exclude_range=exclude_range)
    X_params = data['parameters']
    outputs = data['outputs']
    output_info = data['output_info']

    logger.info(f"Training config: {len(X_params)} samples, {X_params.shape[1]} params, "
                f"{len(outputs)} outputs, test={test_size}, epochs={epochs}")

    # Split data
    train_idx, test_idx = train_test_split(np.arange(len(X_params)), test_size=test_size, random_state=42)
    logger.info(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Create models directory
    models_dir = dataset_dir / "models" / model_name
    models_dir.mkdir(exist_ok=True)

    model_counts = {}
    trained_models = []
    all_histories = []

    for output_key, output_data in outputs.items():
        info = output_info[output_key]
        output_type = info['type']
        field_name = info['field']
        location = info['location']

        # Skip outputs not in the filter (if provided)
        if output_filter is not None and location not in output_filter:
            logger.info(f"Skipping {location}_{field_name} (not in output filter)")
            continue

        base_name = f"{location}_{field_name}"
        if base_name not in model_counts:
            model_counts[base_name] = 0
        model_counts[base_name] += 1

        output_model_name = f"{base_name}_{model_counts[base_name]}" if model_counts[base_name] > 1 else base_name
        preset = output_type.lower()

        logger.info(f"Training: {output_model_name} (preset={preset})")
        if on_progress:
            on_progress(output_model_name, 'training')

        # POD reduction for 2D/3D
        pod = None
        if output_type in ['2D', '3D']:
            pod = PODReducer(n_modes=info['n_modes'])
            train_targets = pod.fit_transform(output_data[train_idx])
        else:
            train_targets = output_data[train_idx]

        # Train NN
        nn = SurrogateNN.from_preset(preset, field_name=output_key)

        # Per-epoch callback that tags the model name for GUI routing
        epoch_cb = None
        if on_epoch is not None:
            def epoch_cb(epoch, train_loss, val_loss, _name=output_model_name):
                on_epoch(_name, epoch, train_loss, val_loss)

        # For small training sets, use a larger validation fraction so the val signal
        # isn't dominated by 2-3 noisy samples. For very small sets, disable the
        # inner val split entirely and let early stopping use train loss.
        n_train = len(train_idx)
        if n_train < 15:
            inner_val_split = 0.0
            logger.info(f"Small training set ({n_train}), using validation_split=0 (train-loss early stopping)")
        elif n_train < 40:
            inner_val_split = 0.3
        else:
            inner_val_split = 0.2

        nn.fit(X_params[train_idx], train_targets,
               validation_split=inner_val_split, epochs=epochs, verbose=0,
               on_epoch=epoch_cb)

        # Evaluate
        train_pred = nn.predict(X_params[train_idx])
        if pod:
            train_pred = pod.inverse_transform(train_pred)
        train_metrics = compute_metrics(output_data[train_idx], train_pred)

        test_pred = nn.predict(X_params[test_idx])
        if pod:
            test_pred = pod.inverse_transform(test_pred)
        test_metrics = compute_metrics(output_data[test_idx], test_pred)

        logger.info(f"  Train R²: {train_metrics['r2']:.4f}, Test R²: {test_metrics['r2']:.4f}, "
                     f"Test MAE: {test_metrics['mae']:.4e}")

        # Save
        model_path = models_dir / output_model_name
        nn.save(model_path)
        if pod:
            pod.save(model_path)

        if nn.history is not None:
            plot_loss_curves(nn.history, output_model_name, models_dir, output_type)
            all_histories.append((output_model_name, nn.history, output_type))

        # Read dataset version if available
        version_file = dataset_dir / "dataset" / "dataset_version.json"
        ds_version = 0
        if version_file.exists():
            try:
                with open(version_file, 'r') as vf:
                    ds_version = json.load(vf).get('version', 0)
            except Exception:
                pass

        metadata = {
            'model_name': output_model_name,
            'model_architecture': model_selection.get(output_type, 'SurrogateNN'),
            'preset': preset,
            'has_pod': pod is not None,
            'output_key': output_key,
            'output_type': output_type,
            'field_name': field_name,
            'location': info['location'],
            'npz_key': info['npz_key'],
            'n_points': info['n_points'],
            'train_metrics': {k: float(v) if not isinstance(v, list) else v for k, v in train_metrics.items()},
            'test_metrics': {k: float(v) if not isinstance(v, list) else v for k, v in test_metrics.items()},
            'trained_date': datetime.now().isoformat(),
            'dataset_version': ds_version,
            'n_train_samples': len(train_idx),
            'n_test_samples': len(test_idx)
        }
        if pod:
            metadata['n_modes'] = pod.n_modes
            metadata['variance_explained'] = float(pod.variance_explained.sum())

        with open(models_dir / f"{output_model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        trained_models.append(metadata)

        if on_progress:
            on_progress(output_model_name, 'complete')

    # Save training summary
    summary = {
        'case_name': dataset_dir.name,
        'trained_date': datetime.now().isoformat(),
        'n_models': len(trained_models),
        'n_train_samples': len(train_idx),
        'n_test_samples': len(test_idx),
        'test_split': test_size,
        'epochs': epochs,
        'test_indices': test_idx.tolist(),
        'train_indices': train_idx.tolist(),
        'models': trained_models
    }

    with open(models_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    if all_histories:
        plot_combined_loss_curves(all_histories, models_dir)

    logger.info(f"Training complete: {len(trained_models)} models saved to {models_dir}")
    return summary
