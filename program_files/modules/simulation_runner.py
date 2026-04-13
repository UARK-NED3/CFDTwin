"""
Simulation Runner Module
=========================
Handles running Fluent simulations for all DOE combinations and saving results.
All functions are GUI-agnostic: they accept parameters, return results, and report
progress via Python logging.
"""

import logging
import numpy as np
from pathlib import Path
import json
import itertools
import time
import sys
import warnings
from contextlib import contextmanager
from io import StringIO

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', message="'get_scalar_field_data' is deprecated")

# Map user-facing parameter names to Fluent solution variable names (SV_*)
FLUENT_VAR_MAP = {
    'temperature': 'SV_T',
    'pressure': 'SV_P',
    'density': 'SV_DENSITY',
    'x-velocity': 'SV_U',
    'y-velocity': 'SV_V',
    'z-velocity': 'SV_W',
    'k': 'SV_K',
    'omega': 'SV_O',
    'turbulent-viscosity': 'SV_MU_T',
    'enthalpy': 'SV_H',
    'wall-distance': 'SV_WALL_DIST'
}


@contextmanager
def suppress_fluent_output():
    """Context manager to suppress stdout/stderr during Fluent API calls."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = StringIO(), StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def extract_surface_data(data_dict, key):
    """Extract data from PyFluent surface data dict, handling both formats."""
    surface_data = data_dict[key]
    if isinstance(surface_data, dict) and 'scalar-field' in surface_data:
        return np.array(surface_data['scalar-field'])
    return np.array(surface_data)


def generate_doe_combinations(setup_data, doe_samples=None):
    """
    Generate all DOE combinations from setup data.

    Supports two formats:
    - Legacy: doe_configuration nested in setup_data
    - New: doe_samples list of dicts from doe_samples.json (takes priority)

    Parameters
    ----------
    setup_data : dict
        Contents of model_setup.json
    doe_samples : list of dict, optional
        Samples from doe_samples.json. Each dict maps 'bc_name|param_name' to value.

    Returns
    -------
    list of (sim_id, bc_values_dict)
    """
    # New format: build from doe_samples list
    if doe_samples:
        bc_type_map = {}
        param_path_map = {}  # 'bc_name|param_display' -> dot-path for PyFluent
        for input_item in setup_data.get('model_inputs', []):
            bc_type_map[input_item['name']] = input_item['type']
            # Map the DOE key to the actual PyFluent attribute path
            param_display = input_item.get('parameter', 'value')
            param_path = input_item.get('parameter_path', param_display.replace(' > ', '.'))
            key = f"{input_item['name']}|{param_display}"
            param_path_map[key] = param_path

        doe_list = []
        for sim_id, sample in enumerate(doe_samples, 1):
            bc_values = {}
            for key, value in sample.items():
                parts = key.split('|', 1)
                bc_name = parts[0]
                param_name = parts[1] if len(parts) > 1 else 'value'
                param_path = param_path_map.get(key, param_name.replace(' > ', '.'))
                bc_values[key] = {
                    'bc_name': bc_name,
                    'bc_type': bc_type_map.get(bc_name, 'Unknown'),
                    'param_name': param_name,
                    'param_path': param_path,
                    'value': value
                }
            doe_list.append((sim_id, bc_values))
        return doe_list

    # Legacy format
    doe_config = setup_data.get('doe_configuration', {})

    param_arrays = []
    param_mapping = []

    bc_type_map = {}
    for input_item in setup_data['model_inputs']:
        bc_type_map[input_item['name']] = input_item['type']

    for bc_name in sorted(doe_config.keys()):
        doe_params = doe_config[bc_name]
        bc_type = bc_type_map.get(bc_name, 'Unknown')

        for param_name in sorted(doe_params.keys()):
            param_values = doe_params[param_name]
            if param_values:
                param_arrays.append(param_values)
                param_mapping.append({
                    'bc_name': bc_name,
                    'bc_type': bc_type,
                    'param_name': param_name,
                    'param_path': param_name
                })

    if not param_arrays:
        return []

    array_lengths = [len(arr) for arr in param_arrays]
    if len(set(array_lengths)) == 1:
        combinations = list(zip(*param_arrays))
    else:
        combinations = list(itertools.product(*param_arrays))

    doe_list = []
    for sim_id, combo in enumerate(combinations, 1):
        bc_values = {}
        for param_info, value in zip(param_mapping, combo):
            bc_key = f"{param_info['bc_name']}|{param_info['param_name']}"
            bc_values[bc_key] = {
                'bc_name': param_info['bc_name'],
                'bc_type': param_info['bc_type'],
                'param_name': param_info['param_name'],
                'param_path': param_info['param_path'],
                'value': value
            }
        doe_list.append((sim_id, bc_values))

    return doe_list


def apply_boundary_conditions(solver, bc_values):
    """
    Apply boundary conditions to Fluent solver.

    Returns
    -------
    bool
        True if successful
    """
    try:
        boundary_conditions = solver.settings.setup.boundary_conditions

        for bc_key, bc_info in bc_values.items():
            bc_name = bc_info['bc_name']
            bc_type = bc_info['bc_type'].lower().replace(' ', '_')
            param_path = bc_info['param_path']
            value = bc_info['value']

            if hasattr(boundary_conditions, bc_type):
                bc_container = getattr(boundary_conditions, bc_type)
                if bc_name in bc_container:
                    bc_obj = bc_container[bc_name]
                    try:
                        path_parts = param_path.split('.')
                        target_obj = bc_obj

                        for part in path_parts:
                            if hasattr(target_obj, part):
                                target_obj = getattr(target_obj, part)
                            elif part == 'velocity_magnitude' and hasattr(target_obj, 'velocity'):
                                logger.info(f"Using 'velocity' instead of 'velocity_magnitude' for {bc_name}")
                                target_obj = getattr(target_obj, 'velocity')
                            elif part == 'velocity' and hasattr(target_obj, 'velocity_magnitude'):
                                logger.info(f"Using 'velocity_magnitude' instead of 'velocity' for {bc_name}")
                                target_obj = getattr(target_obj, 'velocity_magnitude')
                            else:
                                logger.error(f"Path part '{part}' not found in {bc_name}.{param_path}")
                                return False

                        if hasattr(target_obj, 'value'):
                            target_obj.value = float(value)
                            logger.info(f"{bc_name}.{param_path} = {value}")
                        else:
                            logger.error(f"{bc_name}.{param_path} has no .value attribute")
                            return False

                    except Exception as e:
                        logger.error(f"Exception setting {bc_name}.{param_path}: {e}")
                        return False
                else:
                    logger.error(f"BC '{bc_name}' not found in {bc_type}")
                    return False
            else:
                logger.error(f"BC type '{bc_type}' not found")
                return False

        return True

    except Exception as e:
        logger.error(f"Error applying boundary conditions: {e}")
        return False


def extract_coordinates(solver, setup_data, dataset_dir):
    """
    Extract coordinates from all configured output locations.

    Returns
    -------
    dict or None
        Keys like "location|coordinates" -> np.array shape (n_points, 3)
    """
    try:
        output_params_file = dataset_dir.parent / "output_parameters.json"
        if not output_params_file.exists():
            logger.error("output_parameters.json not found")
            return None

        with open(output_params_file, 'r') as f:
            output_params = json.load(f)

        coord_data = {}

        for output_item in output_params.get('outputs', []):
            output_name = output_item['name']
            output_category = output_item.get('category', 'Surface')

            if output_category == "Report Definition":
                continue
            if not output_item.get('field_variables', []):
                continue

            coord_key = f"{output_name}|coordinates"

            if output_category == "Cell Zone":
                try:
                    logger.info(f"Extracting centroids for {output_name}...")
                    solution_data = solver.fields.solution_variable_data
                    centroid_dict = solution_data.get_data(
                        zone_names=[output_name], variable_name='SV_CENTROID', domain_name='mixture')
                    centroids_flat = np.array(centroid_dict[output_name])
                    n_cells = len(centroids_flat) // 3
                    coord_data[coord_key] = centroids_flat.reshape((n_cells, 3))
                    logger.info(f"  {output_name}: {n_cells} cells")
                except Exception as e:
                    logger.error(f"Error extracting centroids for {output_name}: {e}")
            else:
                try:
                    logger.info(f"Extracting coordinates for {output_name}...")
                    field_data = solver.fields.field_data
                    x_dict = field_data.get_scalar_field_data(
                        field_name='x-coordinate', surfaces=[output_name], node_value=False)
                    y_dict = field_data.get_scalar_field_data(
                        field_name='y-coordinate', surfaces=[output_name], node_value=False)
                    z_dict = field_data.get_scalar_field_data(
                        field_name='z-coordinate', surfaces=[output_name], node_value=False)
                    x = extract_surface_data(x_dict, output_name)
                    y = extract_surface_data(y_dict, output_name)
                    z = extract_surface_data(z_dict, output_name)
                    coord_data[coord_key] = np.stack([x, y, z], axis=1)
                    logger.info(f"  {output_name}: {len(x)} points")
                except Exception as e:
                    logger.error(f"Error extracting coordinates for {output_name}: {e}")

        return coord_data if coord_data else None

    except Exception as e:
        logger.error(f"Error extracting coordinates: {e}")
        return None


def save_reference_coordinates(coord_data, dataset_dir):
    """Save coordinate reference file to dataset/coordinates.npz."""
    coord_file = dataset_dir / "dataset" / "coordinates.npz"
    np.savez_compressed(coord_file, **coord_data)
    logger.info(f"Reference coordinates saved: {coord_file.name}")


def load_reference_coordinates(dataset_dir):
    """Load coordinate reference file. Returns dict or None."""
    coord_file = dataset_dir / "dataset" / "coordinates.npz"
    if not coord_file.exists():
        return None
    data = np.load(coord_file, allow_pickle=True)
    return {key: data[key] for key in data.files}


def get_reference_point_counts(dataset_dir):
    """Get expected point counts per location from reference coordinates."""
    ref = load_reference_coordinates(dataset_dir)
    if ref is None:
        return None
    return {key.split('|')[0]: len(arr) for key, arr in ref.items()}


def validate_field_point_counts(field_data, dataset_dir):
    """
    Validate that field data point counts match the reference coordinates.

    Returns
    -------
    (bool, str)
        (True, "") if valid, (False, error_message) if mismatch
    """
    ref_counts = get_reference_point_counts(dataset_dir)
    if ref_counts is None:
        return True, ""

    for key, arr in field_data.items():
        location = key.split('|')[0]
        if location in ref_counts:
            expected = ref_counts[location]
            actual = len(arr)
            if actual != expected:
                return False, (
                    f"Mesh inconsistency for '{location}': expected {expected} points, got {actual}. "
                    f"Delete dataset and restart DOE, or use existing dataset as-is."
                )
    return True, ""


def extract_field_data(solver, setup_data, dataset_dir):
    """
    Extract field data (values only, no coordinates) from configured output locations.

    Returns
    -------
    dict or None
    """
    try:
        output_params_file = dataset_dir.parent / "output_parameters.json"
        if not output_params_file.exists():
            logger.error("output_parameters.json not found")
            return None

        with open(output_params_file, 'r') as f:
            output_params = json.load(f)

        output_data = {}

        for output_item in output_params.get('outputs', []):
            output_name = output_item['name']
            output_category = output_item.get('category', 'Surface')
            params_to_extract = output_item.get('field_variables', [])

            if output_category == "Report Definition":
                try:
                    result = solver.settings.solution.report_definitions.compute(report_defs=[output_name])
                    report_value = result[0][output_name][0]
                    param_name = params_to_extract[0] if params_to_extract else 'value'
                    output_data[f"{output_name}|{param_name}"] = np.array([report_value])
                    logger.info(f"{output_name}|{param_name}: {report_value:.6f}")
                except Exception as e:
                    logger.error(f"Error extracting report definition {output_name}: {e}")
                continue

            if not params_to_extract:
                logger.warning(f"No parameters configured for {output_name}, skipping")
                continue

            logger.info(f"Extracting from {output_name} ({output_category}): {', '.join(params_to_extract)}")

            if output_category == "Cell Zone":
                solution_data = solver.fields.solution_variable_data
                for param_name in params_to_extract:
                    try:
                        if param_name == 'velocity-magnitude':
                            u = np.array(solution_data.get_data(
                                zone_names=[output_name], variable_name='SV_U', domain_name='mixture')[output_name])
                            v = np.array(solution_data.get_data(
                                zone_names=[output_name], variable_name='SV_V', domain_name='mixture')[output_name])
                            w = np.array(solution_data.get_data(
                                zone_names=[output_name], variable_name='SV_W', domain_name='mixture')[output_name])
                            output_data[f"{output_name}|{param_name}"] = np.sqrt(u**2 + v**2 + w**2)
                            logger.info(f"  {param_name}: {len(u)} cells (computed from U,V,W)")
                            continue

                        fluent_var = FLUENT_VAR_MAP.get(param_name, param_name.upper())

                        try:
                            zone_info = solver.fields.solution_variable_info.get_variables_info(
                                zone_names=[output_name], domain_name='mixture')
                            if fluent_var not in zone_info.solution_variables:
                                logger.warning(f"{fluent_var} not available for zone {output_name}")
                                continue
                        except Exception:
                            pass  # Continue and let get_data raise if variable doesn't exist

                        data_dict = solution_data.get_data(
                            zone_names=[output_name], variable_name=fluent_var, domain_name='mixture')
                        zone_data = np.array(data_dict[output_name])
                        output_data[f"{output_name}|{param_name}"] = zone_data
                        logger.info(f"  {param_name}: {len(zone_data)} cells")
                    except Exception as e:
                        logger.error(f"Error extracting {param_name} from {output_name}: {e}")
            else:
                field_data = solver.fields.field_data
                for param_name in params_to_extract:
                    try:
                        data_dict = field_data.get_scalar_field_data(
                            field_name=param_name, surfaces=[output_name], node_value=False)
                        surface_data = extract_surface_data(data_dict, output_name)
                        output_data[f"{output_name}|{param_name}"] = surface_data
                        logger.info(f"  {param_name}: {len(surface_data)} points, "
                                    f"range: [{surface_data.min():.6f}, {surface_data.max():.6f}]")
                    except Exception as e:
                        logger.error(f"Error extracting {param_name} from {output_name}: {e}")

        return output_data if output_data else None

    except Exception as e:
        logger.error(f"Error extracting field data: {e}")
        return None


def _ensure_reference_coordinates(solver, setup_data, dataset_dir):
    """Create reference coordinates if they don't exist yet. Returns True on success."""
    if load_reference_coordinates(dataset_dir) is not None:
        return True
    logger.info("Creating reference coordinates...")
    coord_data = extract_coordinates(solver, setup_data, dataset_dir)
    if coord_data:
        save_reference_coordinates(coord_data, dataset_dir)
        return True
    logger.error("Failed to extract reference coordinates")
    return False


def run_single_simulation(solver, setup_data, dataset_dir, sim_id, iterations=100):
    """
    Run a single simulation by DOE index.

    Returns
    -------
    (bool, Path or None)
        (success, output_file_path)
    """
    doe_list = generate_doe_combinations(setup_data)
    if not doe_list:
        logger.error("No DOE combinations found")
        return False, None

    if sim_id < 1 or sim_id > len(doe_list):
        logger.error(f"Simulation ID {sim_id} out of range (1-{len(doe_list)})")
        return False, None

    _, bc_values = doe_list[sim_id - 1]

    logger.info(f"Running simulation {sim_id}")
    for bc_info in bc_values.values():
        logger.info(f"  {bc_info['bc_name']}.{bc_info['param_name']} = {bc_info['value']}")

    if not apply_boundary_conditions(solver, bc_values):
        return False, None

    try:
        with suppress_fluent_output():
            solver.settings.solution.initialization.hybrid_initialize()
        logger.info("Initialized (hybrid)")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return False, None

    try:
        with suppress_fluent_output():
            solver.settings.solution.run_calculation.iterate(iter_count=iterations)
        logger.info("Solution complete")
    except Exception as e:
        logger.error(f"Solution error: {e}")
        return False, None

    output_data = extract_field_data(solver, setup_data, dataset_dir)
    if output_data is None:
        return False, None

    outputs_dir = dataset_dir
    outputs_dir.mkdir(exist_ok=True)

    if not _ensure_reference_coordinates(solver, setup_data, dataset_dir):
        return False, None

    valid, error_msg = validate_field_point_counts(output_data, dataset_dir)
    output_file = outputs_dir / f"sim_{sim_id:04d}.npz"
    np.savez_compressed(output_file, **output_data)

    if not valid:
        logger.error(error_msg)
        return False, output_file

    logger.info(f"Saved {output_file.name} ({len(output_data)} fields)")
    return True, output_file


def run_batch_simulations(solver, setup_data, dataset_dir, iterations=100, on_progress=None):
    """
    Run all DOE simulations in batch mode.

    Returns
    -------
    dict
        Summary: {'successful', 'failed', 'total', 'elapsed', 'stopped_reason'}
    """
    doe_list = generate_doe_combinations(setup_data)
    if not doe_list:
        logger.error("No DOE combinations found")
        return {'successful': 0, 'failed': 0, 'total': 0, 'elapsed': 0, 'stopped_reason': 'No DOE combinations'}

    return _run_simulation_batch(solver, setup_data, dataset_dir, doe_list, iterations, on_progress)


def run_remaining_simulations(solver, setup_data, dataset_dir, iterations=100,
                              on_progress=None, stop_flag=None, reinitialize=True,
                              doe_samples=None):
    """
    Run only uncompleted DOE simulations.

    Parameters
    ----------
    stop_flag : callable, optional
        Returns True when the user requests a stop after the current sim.
    reinitialize : bool
        If True, reinitialize the solver before each simulation. If False,
        continue from the previous solution (faster for small parameter changes).
    doe_samples : list of dict, optional
        Samples from doe_samples.json. If provided, used instead of legacy doe_configuration.

    Returns
    -------
    dict
        Summary: {'successful', 'failed', 'total', 'elapsed', 'stopped_reason'}
    """
    doe_list = generate_doe_combinations(setup_data, doe_samples=doe_samples)
    if not doe_list:
        logger.error("No DOE combinations found")
        return {'successful': 0, 'failed': 0, 'total': 0, 'elapsed': 0, 'stopped_reason': 'No DOE combinations'}

    outputs_dir = dataset_dir
    completed_ids = set()
    if outputs_dir.exists():
        for f in outputs_dir.glob("sim_*.npz"):
            try:
                completed_ids.add(int(f.stem.split('_')[1]))
            except (ValueError, IndexError):
                pass

    remaining = [(sid, bc) for sid, bc in doe_list if sid not in completed_ids]
    if not remaining:
        logger.info("All simulations complete")
        return {'successful': 0, 'failed': 0, 'total': 0, 'elapsed': 0, 'stopped_reason': None}

    logger.info(f"Remaining: {len(remaining)}, Completed: {len(completed_ids)}")
    return _run_simulation_batch(solver, setup_data, dataset_dir, remaining, iterations,
                                 on_progress, stop_flag, reinitialize)


def _run_simulation_batch(solver, setup_data, dataset_dir, doe_list, iterations,
                          on_progress=None, stop_flag=None, reinitialize=True):
    """
    Core batch simulation loop.

    Parameters
    ----------
    on_progress : callable, optional
        Called as on_progress(idx, total, sim_id, status_str) for each simulation
    stop_flag : callable, optional
        Returns True when the user requests a graceful stop.
    reinitialize : bool
        If True, reinitialize before each sim. If False, continue from previous solution.

    Returns
    -------
    dict
        Summary: {'successful', 'failed', 'total', 'elapsed', 'stopped_reason'}
    """
    outputs_dir = dataset_dir
    outputs_dir.mkdir(exist_ok=True)

    total = len(doe_list)
    start_time = time.time()
    successful = 0
    failed = 0
    is_first_sim = True
    has_reference_coords = load_reference_coordinates(dataset_dir) is not None
    stopped_reason = None

    for idx, (sim_id, bc_values) in enumerate(doe_list, 1):
        # Check stop flag between simulations
        if stop_flag and stop_flag():
            logger.info("Stop requested by user")
            stopped_reason = 'User requested stop'
            break

        output_file = outputs_dir / f"sim_{sim_id:04d}.npz"
        if output_file.exists():
            logger.debug(f"Skipping sim {sim_id} (already exists)")
            successful += 1
            if on_progress:
                on_progress(idx, total, sim_id, 'skipped')
            continue

        logger.info(f"[{idx}/{total}] Simulation {sim_id}")
        if on_progress:
            on_progress(idx, total, sim_id, 'running')

        if not apply_boundary_conditions(solver, bc_values):
            failed += 1
            if on_progress:
                on_progress(idx, total, sim_id, 'failed')
            continue

        if is_first_sim or reinitialize:
            try:
                with suppress_fluent_output():
                    solver.settings.solution.initialization.hybrid_initialize()
                logger.info("Initialized (hybrid)")
                is_first_sim = False
            except Exception as e:
                logger.error(f"Initialization failed: {e}")
                failed += 1
                if on_progress:
                    on_progress(idx, total, sim_id, 'failed')
                continue
        else:
            logger.debug("Continuing from previous solution")

        try:
            with suppress_fluent_output():
                solver.settings.solution.run_calculation.iterate(iter_count=iterations)
        except Exception as e:
            logger.error(f"Solution failed: {e}")
            failed += 1
            if on_progress:
                on_progress(idx, total, sim_id, 'failed')
            continue

        output_data = extract_field_data(solver, setup_data, dataset_dir)
        if output_data is None:
            failed += 1
            if on_progress:
                on_progress(idx, total, sim_id, 'failed')
            continue

        if not has_reference_coords:
            coord_data = extract_coordinates(solver, setup_data, dataset_dir)
            if coord_data:
                save_reference_coordinates(coord_data, dataset_dir)
                has_reference_coords = True
            else:
                failed += 1
                if on_progress:
                    on_progress(idx, total, sim_id, 'failed')
                continue

        valid, error_msg = validate_field_point_counts(output_data, dataset_dir)
        if not valid:
            np.savez_compressed(output_file, **output_data)
            logger.error(f"BATCH STOPPED: {error_msg}")
            stopped_reason = error_msg
            if on_progress:
                on_progress(idx, total, sim_id, 'failed')
            break

        try:
            np.savez_compressed(output_file, **output_data)
            logger.info(f"Saved {output_file.name}")
            successful += 1
            if on_progress:
                on_progress(idx, total, sim_id, 'done')
        except Exception as e:
            logger.error(f"Save failed: {e}")
            failed += 1
            if on_progress:
                on_progress(idx, total, sim_id, 'failed')
            continue

        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining_time = (total - idx) * avg_time
        logger.info(f"Progress: {successful}/{total} complete, {failed} failed, "
                     f"~{remaining_time/60:.1f}m remaining")

    elapsed = time.time() - start_time
    summary = {
        'successful': successful,
        'failed': failed,
        'total': total,
        'elapsed': elapsed,
        'stopped_reason': stopped_reason
    }
    logger.info(f"Batch complete: {successful}/{total} successful, {failed} failed, "
                f"{elapsed/60:.1f} minutes")
    return summary


def extract_current_solution(solver, setup_data, dataset_dir, sim_id):
    """
    Extract data from the currently loaded Fluent solution.

    Returns
    -------
    (bool, Path or None)
        (success, output_file_path)
    """
    logger.info(f"Extracting simulation {sim_id}")

    output_data = extract_field_data(solver, setup_data, dataset_dir)
    if output_data is None:
        return False, None

    outputs_dir = dataset_dir
    outputs_dir.mkdir(exist_ok=True)

    if not _ensure_reference_coordinates(solver, setup_data, dataset_dir):
        return False, None

    valid, error_msg = validate_field_point_counts(output_data, dataset_dir)
    output_file = outputs_dir / f"sim_{sim_id:04d}.npz"
    np.savez_compressed(output_file, **output_data)

    if not valid:
        logger.error(error_msg)
        return False, output_file

    logger.info(f"Saved {output_file.name} ({len(output_data)} fields, "
                f"{output_file.stat().st_size / 1024:.1f} KB)")
    return True, output_file
