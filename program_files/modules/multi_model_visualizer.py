"""
Multi-Model Visualizer Module
===============================
Prediction, comparison, and visualization functions for trained surrogate models.
All functions are GUI-agnostic: they accept data, return figures or result dicts.
"""

import logging
import json
import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

from .surrogate_nn import SurrogateNN
from .pod_reducer import PODReducer

logger = logging.getLogger(__name__)


# ============================================================
# Helper Functions
# ============================================================

def build_doe_param_info(doe_config):
    """
    Build parameter info list from DOE config using sorted key order.

    Returns
    -------
    (param_info, param_combinations)
        param_info: list of dicts with bc_name, param_name, full_name, min, max
        param_combinations: list of tuples
    """
    param_info = []
    param_values = []
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
            param_values.append(values)

    array_lengths = [len(arr) for arr in param_values]
    if len(set(array_lengths)) == 1:
        param_combinations = list(zip(*param_values))
    else:
        param_combinations = list(itertools.product(*param_values))

    return param_info, param_combinations


def load_matching_coordinates(dataset_dir, coord_key, npz_key, expected_size):
    """
    Load coordinates from the reference coordinates.npz file.

    Returns
    -------
    (np.ndarray or None, None)
    """
    coord_file = dataset_dir / "dataset" / "coordinates.npz"
    if not coord_file.exists():
        logger.error(f"coordinates.npz not found in {dataset_dir / 'dataset'}")
        return None, None

    coord_data = np.load(coord_file, allow_pickle=True)
    if coord_key not in coord_data.files:
        logger.error(f"Key '{coord_key}' not found in coordinates.npz")
        return None, None

    coordinates = coord_data[coord_key]
    if len(coordinates) != expected_size:
        logger.error(f"Coordinate size mismatch: {len(coordinates)} vs expected {expected_size}")
        return None, None

    logger.info(f"Loaded coordinates from coordinates.npz ({len(coordinates)} points)")
    return coordinates, None


def align_fluent_to_coordinates(fluent_data, location, npz_key, target_coords):
    """
    Align Fluent data to target coordinates via nearest-neighbor interpolation.

    Returns
    -------
    (np.ndarray or None, bool)
        (aligned_values, has_fluent)
    """
    if fluent_data is None or npz_key not in fluent_data:
        return None, False

    fluent_values = fluent_data[npz_key]
    fluent_coords = fluent_data[f"{location}|coordinates"]

    logger.info(f"Aligning Fluent data: {len(fluent_values)} -> {len(target_coords)} points")

    try:
        fluent_values = griddata(fluent_coords, fluent_values, target_coords, method='nearest')
        logger.info("Alignment successful")
        return fluent_values, True
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return None, False


def load_and_predict(model_path, model_meta, X_input):
    """
    Load a trained model and run prediction, handling POD reconstruction if needed.

    Returns
    -------
    np.ndarray
    """
    nn = SurrogateNN.load(model_path)
    pred = nn.predict(X_input)
    if model_meta.get('has_pod', False):
        pod = PODReducer.load(model_path)
        pred = pod.inverse_transform(pred)
    return pred


def load_model_folder(model_dir):
    """
    Load a single-model folder and return its metadata dict.

    Returns None if no metadata file is found.
    """
    model_dir = Path(model_dir)
    meta_files = list(model_dir.glob("*_metadata.json"))
    if not meta_files:
        return None
    try:
        with open(meta_files[0], 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {meta_files[0]}: {e}")
        return None


def predict_single_model(model_dir, X_params):
    """
    Run a single trained model on given input parameters.

    Parameters
    ----------
    model_dir : Path
        Directory containing exactly one trained model (nn, optional pod, metadata).
    X_params : np.ndarray
        Shape (1, n_params) or (n_samples, n_params)

    Returns
    -------
    dict or None
        {'prediction': np.ndarray, 'metadata': dict, 'model_dir': Path}
    """
    model_dir = Path(model_dir)
    meta = load_model_folder(model_dir)
    if meta is None:
        return None

    try:
        model_name = meta['model_name']
        model_path = model_dir / model_name
        Y_pred = load_and_predict(model_path, meta, X_params)
        return {
            'prediction': Y_pred[0] if len(X_params) == 1 else Y_pred,
            'metadata': meta,
            'model_dir': model_dir,
        }
    except Exception as e:
        logger.error(f"predict_single_model({model_dir.name}): {e}")
        return None


def run_fluent_comparison(solver, setup_data, dataset_dir, params, iterations=100):
    """
    Run a single Fluent simulation using an already-launched solver.
    Returns a dict of NPZ-style field data.

    Parameters
    ----------
    solver : PyFluent solver (from FluentManager)
    setup_data : dict from model_setup.json
    dataset_dir : Path to project dataset/ folder
    params : dict {bc_name|param_name: value}
    iterations : int
    """
    from . import simulation_runner as sr

    # Build bc_values in the format apply_boundary_conditions expects
    bc_type_map = {i['name']: i['type'] for i in setup_data.get('model_inputs', [])}
    path_map = {}
    for i in setup_data.get('model_inputs', []):
        key = f"{i['name']}|{i.get('parameter', 'value')}"
        path_map[key] = i.get('parameter_path', i.get('parameter', 'value').replace(' > ', '.'))

    bc_values = {}
    for key, value in params.items():
        parts = key.split('|', 1)
        bc_name = parts[0]
        param_name = parts[1] if len(parts) > 1 else 'value'
        bc_values[key] = {
            'bc_name': bc_name,
            'bc_type': bc_type_map.get(bc_name, 'Unknown'),
            'param_name': param_name,
            'param_path': path_map.get(key, param_name.replace(' > ', '.')),
            'value': value,
        }

    logger.info("Applying validation boundary conditions...")
    if not sr.apply_boundary_conditions(solver, bc_values):
        logger.error("Failed to apply BCs for validation run")
        return None

    try:
        from .simulation_runner import suppress_fluent_output
        with suppress_fluent_output():
            solver.settings.solution.initialization.hybrid_initialize()
        logger.info("Initialized (hybrid)")
    except Exception as e:
        logger.error(f"Validation init failed: {e}")
        return None

    try:
        with suppress_fluent_output():
            solver.settings.solution.run_calculation.iterate(iter_count=iterations)
        logger.info("Validation solve complete")
    except Exception as e:
        logger.error(f"Validation solve failed: {e}")
        return None

    fluent_results = sr.extract_field_data(solver, setup_data, dataset_dir)
    if fluent_results is None:
        return None

    return {k: np.asarray(v) for k, v in fluent_results.items()}


def predict_dataset_point_single(model_dir, dataset_dir, doe_samples, sim_id):
    """
    Predict for a specific dataset point and compare with ground truth.

    Parameters
    ----------
    model_dir : Path
        Single-model folder.
    dataset_dir : Path
        Project dataset/ directory containing sim_*.npz files.
    doe_samples : list of dict
        Samples from doe_samples.json. Each dict maps 'bc_name|param_name' to value.
    sim_id : int
        1-based sim id (matches file name).

    Returns
    -------
    dict with prediction, ground_truth, metrics, metadata
    """
    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    meta = load_model_folder(model_dir)
    if meta is None:
        return None

    # Build X in the same sorted-key order the trainer uses
    if not doe_samples or sim_id - 1 >= len(doe_samples):
        raise ValueError(f"sim_id {sim_id} out of range")

    sample = doe_samples[sim_id - 1]
    param_names = sorted(sample.keys())
    X = np.array([[sample[k] for k in param_names]])

    # Predict
    model_name = meta['model_name']
    model_path = model_dir / model_name
    Y_pred = load_and_predict(model_path, meta, X)[0]

    # Load ground truth from sim file
    sim_file = dataset_dir / f"sim_{sim_id:04d}.npz"
    if not sim_file.exists():
        raise FileNotFoundError(f"Sim file not found: {sim_file}")

    sim_data = np.load(sim_file, allow_pickle=True)
    npz_key = meta['npz_key']
    if npz_key not in sim_data.files:
        raise KeyError(f"NPZ key '{npz_key}' not in {sim_file.name}")
    Y_true = sim_data[npz_key]

    # Metrics
    mae = float(np.mean(np.abs(Y_pred - Y_true)))
    rmse = float(np.sqrt(np.mean((Y_pred - Y_true) ** 2)))
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    return {
        'prediction': Y_pred,
        'ground_truth': Y_true,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'metadata': meta,
        'model_dir': model_dir,
    }


def format_field_label(field_name):
    """Add units to field name if applicable."""
    if 'temperature' in field_name.lower():
        return f'{field_name} (K)'
    return field_name


def _scatter_2d(ax, coordinates, values, varying_dims, rotate, xlabel, ylabel, cmap='viridis', s=15, alpha=0.8):
    """Plot a 2D scatter with optional axis rotation for yz-planes."""
    if rotate:
        scatter = ax.scatter(
            coordinates[:, varying_dims[1]], coordinates[:, varying_dims[0]],
            c=values, cmap=cmap, s=s, alpha=alpha)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        scatter = ax.scatter(
            coordinates[:, varying_dims[0]], coordinates[:, varying_dims[1]],
            c=values, cmap=cmap, s=s, alpha=alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return scatter


# ============================================================
# Prediction Functions
# ============================================================

def predict_from_params(models_dir, summary, X_params):
    """
    Run all trained models on given input parameters.

    Parameters
    ----------
    models_dir : Path
        Directory containing trained models
    summary : dict
        Training summary data
    X_params : np.ndarray
        Shape (1, n_params) or (n_samples, n_params)

    Returns
    -------
    dict
        {output_key: {'prediction': np.ndarray, 'model_name': str, 'output_type': str, 'model_meta': dict}}
    """
    predictions = {}
    for model_meta in summary['models']:
        model_name = model_meta['model_name']
        output_key = model_meta['output_key']
        try:
            model_path = models_dir / model_name
            Y_pred = load_and_predict(model_path, model_meta, X_params)
            predictions[output_key] = {
                'prediction': Y_pred[0] if len(X_params) == 1 else Y_pred,
                'model_name': model_name,
                'output_type': model_meta['output_type'],
                'model_meta': model_meta
            }
            logger.info(f"  {model_name}: OK")
        except Exception as e:
            logger.error(f"  {model_name}: {e}")
    return predictions


def predict_dataset_point(dataset_dir, summary, models_dir, sim_index):
    """
    Predict for a specific dataset point and compare with ground truth.

    Parameters
    ----------
    dataset_dir : Path
    summary : dict
    models_dir : Path
    sim_index : int
        0-based index into the simulation files

    Returns
    -------
    dict
        {output_key: {'prediction': np.ndarray, 'ground_truth': np.ndarray,
                       'model_name': str, 'output_type': str, 'model_meta': dict,
                       'metrics': {'mae': float, 'rmse': float, 'r2': float}}}
    """
    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    doe_config = setup_data.get('doe_configuration', {})
    param_info, param_combinations = build_doe_param_info(doe_config)

    dataset_output_dir = dataset_dir / "dataset"
    output_files = sorted(dataset_output_dir.glob("sim_*.npz"))

    if sim_index < 0 or sim_index >= len(output_files):
        raise ValueError(f"sim_index {sim_index} out of range (0-{len(output_files)-1})")

    test_file = output_files[sim_index]
    test_data = np.load(test_file, allow_pickle=True)

    # Get params for this simulation
    if sim_index >= len(param_combinations):
        raise ValueError(f"sim_index {sim_index} exceeds param_combinations length")

    X_test = np.array([param_combinations[sim_index]])

    predictions = {}
    for model_meta in summary['models']:
        model_name = model_meta['model_name']
        output_key = model_meta['output_key']
        npz_key = model_meta['npz_key']

        try:
            model_path = models_dir / model_name
            Y_pred = load_and_predict(model_path, model_meta, X_test)[0]

            if npz_key not in test_data.files:
                logger.warning(f"{model_name}: ground truth key '{npz_key}' not found")
                continue

            Y_true = test_data[npz_key]
            mae = np.mean(np.abs(Y_pred - Y_true))
            rmse = np.sqrt(np.mean((Y_pred - Y_true)**2))
            ss_res = np.sum((Y_true - Y_pred)**2)
            ss_tot = np.sum((Y_true - np.mean(Y_true))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            predictions[output_key] = {
                'prediction': Y_pred,
                'ground_truth': Y_true,
                'model_name': model_name,
                'output_type': model_meta['output_type'],
                'model_meta': model_meta,
                'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2}
            }
            logger.info(f"  {model_name}: R2={r2:.4f}, MAE={mae:.4e}")
        except Exception as e:
            logger.error(f"  {model_name}: {e}")

    return predictions


def run_fluent_validation(dataset_dir, setup_data, custom_params, param_info, iterations=100, processors=2):
    """
    Run a Fluent simulation with given parameters for validation.

    Returns
    -------
    dict or None
        Dictionary of Fluent results (field data + coordinates), or None if failed
    """
    try:
        from . import simulation_runner as sr
        import ansys.fluent.core as pyfluent

        case_file = setup_data.get('case_file')
        if not case_file or not Path(case_file).exists():
            logger.error(f"Case file not found: {case_file}")
            return None

        logger.info(f"Launching Fluent ({processors} processors)...")
        solver = pyfluent.launch_fluent(
            precision='double', processor_count=processors, dimension=3, mode='solver')
        logger.info("Fluent launched")

        logger.info(f"Reading case file: {case_file}")
        solver.settings.file.read_case(file_name=case_file)

        # Build BC values
        bc_values = {}
        for i, info in enumerate(param_info):
            bc_key = f"{info['bc_name']}|{info['param_name']}"
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

        logger.info("Applying boundary conditions...")
        if not sr.apply_boundary_conditions(solver, bc_values):
            logger.error("Failed to apply boundary conditions")
            return None

        logger.info(f"Running simulation ({iterations} iterations)...")
        try:
            solver.settings.solution.initialization.initialization_type = "standard"
            solver.settings.solution.initialization.standard_initialize()
            solver.settings.solution.run_calculation.iterate(iter_count=iterations)
            logger.info("Simulation complete")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return None

        logger.info("Extracting results...")
        fluent_results = sr.extract_field_data(solver, setup_data, dataset_dir)
        fluent_coords = sr.extract_coordinates(solver, setup_data, dataset_dir)
        if fluent_results and fluent_coords:
            fluent_results.update(fluent_coords)

        if fluent_results:
            logger.info(f"Fluent validation complete: {len(fluent_results)} fields")
        return fluent_results

    except ImportError as e:
        logger.error(f"Import error: {e}. Install ansys-fluent-core.")
        return None
    except Exception as e:
        logger.error(f"Fluent validation error: {e}")
        return None


# ============================================================
# Plot Functions (return matplotlib Figure objects)
# ============================================================

def plot_scalar_comparison(scalar_results, fluent_data=None):
    """
    Create a comparison table/bar chart for scalar predictions.

    Returns
    -------
    dict
        {output_key: {'nn_value': float, 'fluent_value': float or None, 'error_pct': float or None}}
    """
    results = {}
    for output_key, data in scalar_results.items():
        nn_value = float(data['prediction'][0]) if hasattr(data['prediction'], '__len__') else float(data['prediction'])
        fluent_key = data['model_meta']['npz_key']
        fluent_value = None
        error_pct = None
        if fluent_data and fluent_key in fluent_data:
            fluent_value = float(fluent_data[fluent_key][0])
            if fluent_value != 0:
                error_pct = abs(nn_value - fluent_value) / abs(fluent_value) * 100
        results[output_key] = {'nn_value': nn_value, 'fluent_value': fluent_value, 'error_pct': error_pct}
    return results


def plot_2d_field(coordinates, nn_pred, location, field_name, fluent_values=None, downsample=0.5):
    """
    Create a 2D field plot with optional Fluent comparison.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Detect varying dimensions
    variances = [np.var(coordinates[:, i]) for i in range(3)]
    varying_dims = sorted(range(3), key=lambda i: variances[i], reverse=True)[:2]
    varying_dims.sort()

    axis_names = ['X', 'Y', 'Z']
    xlabel = f'{axis_names[varying_dims[0]]} (m)'
    ylabel = f'{axis_names[varying_dims[1]]} (m)'
    rotate = 'yz' in location.lower()

    # Downsample
    n_points = len(coordinates)
    if downsample < 1.0:
        target = int(n_points * downsample)
        indices = np.random.choice(n_points, target, replace=False)
        coordinates = coordinates[indices]
        nn_pred = nn_pred[indices]
        if fluent_values is not None:
            fluent_values = fluent_values[indices]

    has_fluent = fluent_values is not None

    if has_fluent:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        scatter1 = _scatter_2d(axes[0], coordinates, nn_pred, varying_dims, rotate, xlabel, ylabel)
        axes[0].set_title(f'Neural Network\n{field_name}')
        plt.colorbar(scatter1, ax=axes[0], label=format_field_label(field_name))

        scatter2 = _scatter_2d(axes[1], coordinates, fluent_values, varying_dims, rotate, xlabel, ylabel)
        axes[1].set_title(f'Fluent CFD\n{field_name}')
        plt.colorbar(scatter2, ax=axes[1], label=format_field_label(field_name))

        vmin = min(nn_pred.min(), fluent_values.min())
        vmax = max(nn_pred.max(), fluent_values.max())
        scatter1.set_clim(vmin, vmax)
        scatter2.set_clim(vmin, vmax)

        error = np.abs(nn_pred - fluent_values)
        scatter3 = _scatter_2d(axes[2], coordinates, error, varying_dims, rotate, xlabel, ylabel, cmap='Reds')
        axes[2].set_title(f'Absolute Error\nMAE: {error.mean():.4e}')
        plt.colorbar(scatter3, ax=axes[2], label='Error')

        fig.suptitle(f'{location} - {field_name}', fontsize=14, fontweight='bold')
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        scatter = _scatter_2d(ax, coordinates, nn_pred, varying_dims, rotate, xlabel, ylabel)
        ax.set_title(f'{location} - {field_name}')
        plt.colorbar(scatter, ax=ax, label=format_field_label(field_name))

    plt.tight_layout()
    return fig


def plot_3d_field(coordinates, nn_pred, field_name, fluent_values=None, downsample=0.5):
    """
    Create a 3D field plot with optional Fluent comparison.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_points = len(coordinates)
    if downsample < 1.0:
        target = int(n_points * downsample)
        indices = np.random.choice(n_points, target, replace=False)
        coordinates = coordinates[indices]
        nn_pred = nn_pred[indices]
        if fluent_values is not None:
            fluent_values = fluent_values[indices]

    # Aspect ratio
    ranges = [coordinates[:, i].max() - coordinates[:, i].min() for i in range(3)]
    max_range = max(ranges)
    aspect = [r / max_range for r in ranges]

    has_fluent = fluent_values is not None

    if has_fluent:
        fig = plt.figure(figsize=(18, 5))

        ax1 = fig.add_subplot(131, projection='3d')
        s1 = ax1.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
                         c=nn_pred, cmap='viridis', s=5, alpha=0.6)
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
        ax1.set_title(f'Neural Network\n{field_name}')
        plt.colorbar(s1, ax=ax1, label=format_field_label(field_name), shrink=0.7)
        ax1.set_box_aspect(aspect)

        ax2 = fig.add_subplot(132, projection='3d')
        s2 = ax2.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
                         c=fluent_values, cmap='viridis', s=5, alpha=0.6)
        ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_zlabel('Z (m)')
        ax2.set_title(f'Fluent CFD\n{field_name}')
        plt.colorbar(s2, ax=ax2, label=format_field_label(field_name), shrink=0.7)
        ax2.set_box_aspect(aspect)

        vmin = min(nn_pred.min(), fluent_values.min())
        vmax = max(nn_pred.max(), fluent_values.max())
        s1.set_clim(vmin, vmax)
        s2.set_clim(vmin, vmax)

        error = np.abs(nn_pred - fluent_values)
        ax3 = fig.add_subplot(133, projection='3d')
        s3 = ax3.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
                         c=error, cmap='Reds', s=5, alpha=0.6)
        ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)'); ax3.set_zlabel('Z (m)')
        ax3.set_title(f'Absolute Error\nMAE: {error.mean():.4e}')
        plt.colorbar(s3, ax=ax3, label='Error', shrink=0.7)
        ax3.set_box_aspect(aspect)

        fig.suptitle(field_name, fontsize=14, fontweight='bold')
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
                       c=nn_pred, cmap='viridis', s=5, alpha=0.6)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title(field_name)
        plt.colorbar(s, ax=ax, label=format_field_label(field_name), shrink=0.7)
        ax.set_box_aspect(aspect)

    plt.tight_layout()
    return fig


def plot_2d_comparison(coordinates, nn_pred, ground_truth, location, field_name, downsample=0.5):
    """
    Create a 2D comparison plot: NN prediction vs ground truth vs error.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return plot_2d_field(coordinates, nn_pred, location, field_name,
                         fluent_values=ground_truth, downsample=downsample)


def plot_3d_comparison(coordinates, nn_pred, ground_truth, field_name, downsample=0.5):
    """
    Create a 3D comparison plot: NN prediction vs ground truth vs error.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return plot_3d_field(coordinates, nn_pred, field_name,
                         fluent_values=ground_truth, downsample=downsample)
