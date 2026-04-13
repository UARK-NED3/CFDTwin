"""
Design of Experiments Setup Module
===================================
Pure functions for DOE configuration, LHS sampling, and factorial design.
No UI -- all functions accept parameters and return results.
"""

import logging
import itertools
import numpy as np
from scipy.stats import qmc

logger = logging.getLogger(__name__)


def get_bc_parameters(solver, bc_name, bc_type):
    """
    Get available parameters for a boundary condition by querying Fluent.

    Parameters
    ----------
    solver : PyFluent solver
    bc_name : str
        Name of the boundary condition
    bc_type : str
        Type of boundary condition

    Returns
    -------
    list of dict
        Each dict has 'name', 'path', 'type' keys
    """
    parameters = []

    try:
        bc_type_key = bc_type.lower().replace(' ', '_')
        boundary_conditions = solver.settings.setup.boundary_conditions

        if not hasattr(boundary_conditions, bc_type_key):
            return [{'name': 'value', 'path': None, 'type': 'unknown'}]

        bc_container = getattr(boundary_conditions, bc_type_key)
        if bc_name not in bc_container:
            return [{'name': 'value', 'path': None, 'type': 'unknown'}]

        bc_obj = bc_container[bc_name]

        def explore_object(obj, path="", max_depth=3):
            if max_depth <= 0:
                return
            try:
                if hasattr(obj, 'child_names'):
                    for child_name in obj.child_names:
                        if child_name in ['child_names', 'command_names']:
                            continue
                        try:
                            child_obj = getattr(obj, child_name)
                            child_path = f"{path}.{child_name}" if path else child_name

                            if hasattr(child_obj, 'value'):
                                param_info = {
                                    'name': child_path.replace('.', ' > '),
                                    'path': child_path,
                                    'type': type(child_obj).__name__,
                                }
                                if hasattr(child_obj, 'min') and hasattr(child_obj, 'max'):
                                    try:
                                        param_info['min'] = child_obj.min()
                                        param_info['max'] = child_obj.max()
                                    except Exception:
                                        pass
                                if hasattr(child_obj, 'allowed_values'):
                                    try:
                                        param_info['allowed_values'] = child_obj.allowed_values()
                                    except Exception:
                                        pass
                                parameters.append(param_info)
                            else:
                                explore_object(child_obj, child_path, max_depth - 1)
                        except Exception:
                            pass
            except Exception:
                pass

        explore_object(bc_obj)

        if not parameters:
            parameters = [{'name': 'value', 'path': None, 'type': 'unknown'}]

    except Exception:
        parameters = [{'name': 'value', 'path': None, 'type': 'unknown'}]

    return parameters


def generate_lhs_samples(ranges_dict, n_samples, existing_samples=None, tolerance=1e-6):
    """
    Generate Latin Hypercube Samples for all parameters.

    Parameters
    ----------
    ranges_dict : dict
        {'bc_name|param_name': {'min': float, 'max': float}}
    n_samples : int
    existing_samples : list of dict, optional
        Existing samples to check for redundancy
    tolerance : float

    Returns
    -------
    list of dict
        Each dict maps 'bc_name|param_name' to value
    """
    if not ranges_dict:
        return []

    param_keys = sorted(ranges_dict.keys())
    n_dims = len(param_keys)

    sampler = qmc.LatinHypercube(d=n_dims, seed=None)
    samples_unit = sampler.random(n=n_samples)

    samples_list = []
    for sample_unit in samples_unit:
        sample_dict = {}
        for i, param_key in enumerate(param_keys):
            min_val = ranges_dict[param_key]['min']
            max_val = ranges_dict[param_key]['max']
            sample_dict[param_key] = min_val + sample_unit[i] * (max_val - min_val)
        samples_list.append(sample_dict)

    if existing_samples:
        filtered = []
        for new_sample in samples_list:
            is_redundant = any(
                all(abs(new_sample.get(k, 0) - ex.get(k, 0)) < tolerance for k in param_keys)
                for ex in existing_samples
            )
            if not is_redundant:
                filtered.append(new_sample)
        return filtered

    return samples_list


def generate_factorial_samples(ranges_dict, n_points_per_param, existing_samples=None, tolerance=1e-6):
    """
    Generate Full Factorial samples for all parameters.

    Parameters
    ----------
    ranges_dict : dict
        {'bc_name|param_name': {'min': float, 'max': float}}
    n_points_per_param : int
    existing_samples : list of dict, optional
    tolerance : float

    Returns
    -------
    list of dict
    """
    if not ranges_dict or n_points_per_param < 2:
        return []

    param_keys = sorted(ranges_dict.keys())
    param_arrays = []
    for param_key in param_keys:
        min_val = ranges_dict[param_key]['min']
        max_val = ranges_dict[param_key]['max']
        param_arrays.append(np.linspace(min_val, max_val, n_points_per_param))

    combinations = list(itertools.product(*param_arrays))

    samples_list = []
    for combo in combinations:
        sample_dict = {param_keys[i]: combo[i] for i in range(len(param_keys))}
        samples_list.append(sample_dict)

    if existing_samples:
        filtered = []
        for new_sample in samples_list:
            is_redundant = any(
                all(abs(new_sample.get(k, 0) - ex.get(k, 0)) < tolerance for k in param_keys)
                for ex in existing_samples
            )
            if not is_redundant:
                filtered.append(new_sample)
        return filtered

    return samples_list


def samples_to_doe_parameters(samples_list):
    """
    Convert list of sample dicts to doe_parameters format.

    Parameters
    ----------
    samples_list : list of dict
        Each sample maps 'bc_name|param_name' to value

    Returns
    -------
    dict
        {bc_name: {param_name: [values]}}
    """
    if not samples_list:
        return {}

    doe_params = {}
    for param_key in samples_list[0].keys():
        bc_name, param_name = param_key.split('|')
        if bc_name not in doe_params:
            doe_params[bc_name] = {}
        doe_params[bc_name][param_name] = [sample[param_key] for sample in samples_list]

    return doe_params


def doe_parameters_to_ranges(doe_parameters):
    """
    Extract ranges from existing doe_parameters.

    Returns
    -------
    dict
        {'bc_name|param_name': {'min': float, 'max': float}}
    """
    ranges = {}
    for bc_name, params in doe_parameters.items():
        for param_name, values in params.items():
            if values:
                ranges[f"{bc_name}|{param_name}"] = {
                    'min': float(min(values)),
                    'max': float(max(values))
                }
    return ranges


def doe_parameters_to_samples(doe_parameters):
    """
    Convert doe_parameters to list of sample dicts.
    Handles both LHS (parallel arrays) and full factorial formats.

    Returns
    -------
    list of dict
        Each dict maps 'bc_name|param_name' to value
    """
    if not doe_parameters:
        return []

    param_keys = []
    param_arrays = []

    for bc_name, params in sorted(doe_parameters.items()):
        for param_name, values in sorted(params.items()):
            if values:
                param_keys.append(f"{bc_name}|{param_name}")
                param_arrays.append(values)

    if not param_arrays:
        return []

    array_lengths = [len(arr) for arr in param_arrays]
    if len(set(array_lengths)) == 1:
        combinations = list(zip(*param_arrays))
    else:
        combinations = list(itertools.product(*param_arrays))

    return [{param_keys[i]: combo[i] for i in range(len(param_keys))} for combo in combinations]


def save_doe_samples(filepath, samples, ranges):
    """
    Save DOE samples and ranges to a JSON file.

    Parameters
    ----------
    filepath : Path or str
    samples : list of dict
        Each dict maps 'bc_name|param_name' to value
    ranges : dict
        {'bc_name|param_name': {'min': float, 'max': float}}
    """
    import json
    from pathlib import Path
    data = {
        'ranges': {k: {'min': float(v['min']), 'max': float(v['max'])} for k, v in ranges.items()},
        'samples': [{k: float(v) for k, v in s.items()} for s in samples],
    }
    with open(Path(filepath), 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(samples)} DOE samples to {filepath}")


def load_doe_samples(filepath):
    """
    Load DOE samples and ranges from a JSON file.

    Returns
    -------
    tuple (samples, ranges)
        samples: list of dict, ranges: dict
    """
    import json
    from pathlib import Path
    fp = Path(filepath)
    if not fp.exists():
        return [], {}
    with open(fp, 'r') as f:
        data = json.load(f)
    return data.get('samples', []), data.get('ranges', {})


def analyze_setup_dimensions(setup_data):
    """
    Analyze model setup to determine input/output dimensionality and total combinations.

    Returns
    -------
    dict
        Keys: num_inputs, num_outputs, input_details, output_details, total_input_combinations
    """
    analysis = {
        'num_inputs': 0,
        'num_outputs': 0,
        'input_details': [],
        'output_details': [],
        'total_input_combinations': 0
    }

    all_param_lengths = []

    for input_item in setup_data.get('model_inputs', []):
        doe_params = input_item.get('doe_parameters', {})
        for param_name, param_values in doe_params.items():
            if param_values:
                analysis['num_inputs'] += 1
                all_param_lengths.append(len(param_values))
                analysis['input_details'].append({
                    'bc_name': input_item['name'],
                    'bc_type': input_item['type'],
                    'parameter': param_name,
                    'num_values': len(param_values),
                    'range': [min(param_values), max(param_values)] if param_values else None
                })

    if all_param_lengths:
        if len(set(all_param_lengths)) == 1:
            analysis['total_input_combinations'] = all_param_lengths[0]
        else:
            analysis['total_input_combinations'] = 1
            for length in all_param_lengths:
                analysis['total_input_combinations'] *= length

    for output_item in setup_data.get('model_outputs', []):
        analysis['num_outputs'] += 1
        analysis['output_details'].append({
            'name': output_item['name'],
            'type': output_item['type'],
            'category': output_item.get('category', 'Unknown')
        })

    return analysis


def create_dataset_structure(dataset_dir, analysis, setup_data):
    """
    Create the directory structure for dataset storage.

    Parameters
    ----------
    dataset_dir : Path
    analysis : dict
    setup_data : dict
    """
    dataset_dir.mkdir(exist_ok=True)
    (dataset_dir / "dataset").mkdir(exist_ok=True)

    readme_path = dataset_dir / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("Dataset Directory Structure\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Created: {setup_data.get('timestamp', 'Unknown')}\n")
        f.write(f"Required Simulations: {analysis['total_input_combinations']}\n")
        f.write(f"Input Variables: {analysis['num_inputs']}\n")
        f.write(f"Output Locations: {analysis['num_outputs']}\n\n")

    logger.info(f"Dataset structure created at {dataset_dir}")
