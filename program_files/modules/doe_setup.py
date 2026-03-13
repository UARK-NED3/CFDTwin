"""
Design of Experiments Setup Module
===================================
Handles DOE configuration including parameter detection and value setup.
"""

import numpy as np
from scipy.stats import qmc


def get_bc_parameters(solver, bc_name, bc_type):
    """Get available parameters for a given boundary condition by querying Fluent.

    Args:
        solver: PyFluent solver object
        bc_name: Name of the boundary condition (e.g., 'inlet')
        bc_type: Type of boundary condition (e.g., 'velocity_inlet')

    Returns:
        List of parameter dictionaries with 'name', 'path', and 'type' keys
    """
    parameters = []

    try:
        # Normalize the bc_type to match API naming
        bc_type_key = bc_type.lower().replace(' ', '_')

        # Access the boundary condition object
        boundary_conditions = solver.settings.setup.boundary_conditions

        if not hasattr(boundary_conditions, bc_type_key):
            # Fallback to generic parameters if type not found
            return [{'name': 'value', 'path': None, 'type': 'unknown'}]

        bc_container = getattr(boundary_conditions, bc_type_key)

        if bc_name not in bc_container:
            return [{'name': 'value', 'path': None, 'type': 'unknown'}]

        bc_obj = bc_container[bc_name]

        # Recursively explore the BC object to find settable parameters
        def explore_object(obj, path="", max_depth=3):
            """Recursively explore object structure to find parameters."""
            if max_depth <= 0:
                return

            try:
                # Get child names to explore nested structures
                if hasattr(obj, 'child_names'):
                    child_names = obj.child_names

                    for child_name in child_names:
                        if child_name in ['child_names', 'command_names']:
                            continue

                        try:
                            child_obj = getattr(obj, child_name)
                            child_path = f"{path}.{child_name}" if path else child_name

                            # Check if this is a settable value (has 'value' attribute or is numeric)
                            if hasattr(child_obj, 'value'):
                                # This is a parameter we can set
                                param_info = {
                                    'name': child_path.replace('.', ' > '),
                                    'path': child_path,
                                    'type': type(child_obj).__name__,
                                    'object': child_obj
                                }

                                # Try to get min/max if available
                                if hasattr(child_obj, 'min') and hasattr(child_obj, 'max'):
                                    try:
                                        param_info['min'] = child_obj.min()
                                        param_info['max'] = child_obj.max()
                                    except:
                                        pass

                                # Try to get allowed values if available
                                if hasattr(child_obj, 'allowed_values'):
                                    try:
                                        param_info['allowed_values'] = child_obj.allowed_values()
                                    except:
                                        pass

                                parameters.append(param_info)
                            else:
                                # Recurse into this child object
                                explore_object(child_obj, child_path, max_depth - 1)
                        except Exception as e:
                            # Skip parameters that cause errors
                            pass
            except Exception as e:
                pass

        # Start exploration
        explore_object(bc_obj)

        # If no parameters found, return generic fallback
        if not parameters:
            parameters = [{'name': 'value', 'path': None, 'type': 'unknown'}]

    except Exception as e:
        # Fallback to generic parameter on error
        parameters = [{'name': 'value', 'path': None, 'type': 'unknown'}]

    return parameters


def setup_parameter_values(param_name, current_values, ui_helpers):
    """Configure test values for a single parameter."""
    if current_values is None:
        current_values = []

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header(f"CONFIGURE PARAMETER: {param_name.upper()}")

        # Show current values
        if current_values:
            print("\n" + "="*70)
            print("CURRENT VALUES:")
            print("="*70)
            for i, val in enumerate(current_values, 1):
                print(f"  [{i:2d}] {val}")
            print("="*70)
        else:
            print("\nNo values configured yet.")

        print(f"\n{'='*70}")
        print("  [1] Add Value Manually")
        print("  [2] Fill Range (Evenly Spaced)")
        print("  [3] Fill Range (Edge-Biased)")
        print("  [4] Clear All Values")
        print("  [5] Remove Specific Value")
        print("  [D] Done")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'D':
            return current_values
        elif choice == '1':
            # Manual entry
            try:
                val = float(input(f"\nEnter value for {param_name}: ").strip())
                current_values.append(val)
                current_values.sort()
                print(f"✓ Added value: {val}")
                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid number")
                ui_helpers.pause()
        elif choice == '2':
            # Evenly spaced
            try:
                min_val = float(input("\nEnter minimum value: ").strip())
                max_val = float(input("Enter maximum value: ").strip())
                num_points = int(input("Enter number of points: ").strip())

                if num_points < 2:
                    print("✗ Need at least 2 points")
                    ui_helpers.pause()
                    continue

                new_values = np.linspace(min_val, max_val, num_points).tolist()
                current_values.extend(new_values)
                current_values = sorted(list(set(current_values)))  # Remove duplicates and sort
                print(f"✓ Added {len(new_values)} evenly spaced values")
                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid input")
                ui_helpers.pause()
        elif choice == '3':
            # Edge-biased (higher resolution near edges)
            try:
                min_val = float(input("\nEnter minimum value: ").strip())
                max_val = float(input("Enter maximum value: ").strip())
                num_points = int(input("Enter number of points: ").strip())

                if num_points < 2:
                    print("✗ Need at least 2 points")
                    ui_helpers.pause()
                    continue

                # Use cosine spacing for edge bias
                t = np.linspace(0, np.pi, num_points)
                normalized = (1 - np.cos(t)) / 2  # Maps to [0, 1] with edge bias
                new_values = (min_val + normalized * (max_val - min_val)).tolist()
                current_values.extend(new_values)
                current_values = sorted(list(set(current_values)))  # Remove duplicates and sort
                print(f"✓ Added {len(new_values)} edge-biased values")
                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid input")
                ui_helpers.pause()
        elif choice == '4':
            # Clear all
            current_values.clear()
            print("✓ Cleared all values")
            ui_helpers.pause()
        elif choice == '5':
            # Remove specific value
            if not current_values:
                print("✗ No values to remove")
                ui_helpers.pause()
                continue
            try:
                idx = int(input(f"\nEnter index to remove [1-{len(current_values)}]: ").strip())
                if 1 <= idx <= len(current_values):
                    removed = current_values.pop(idx - 1)
                    print(f"✓ Removed value: {removed}")
                else:
                    print("✗ Invalid index")
                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid input")
                ui_helpers.pause()


def setup_parameter_ranges(param_name, current_range, ui_helpers):
    """Configure range (min/max) for a single parameter."""
    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header(f"CONFIGURE RANGE: {param_name.upper()}")

        # Show current range
        if current_range:
            print("\n" + "="*70)
            print("CURRENT RANGE:")
            print("="*70)
            print(f"  Minimum: {current_range['min']}")
            print(f"  Maximum: {current_range['max']}")
            print("="*70)
        else:
            print("\nNo range configured yet.")

        print(f"\n{'='*70}")
        print("  [1] Set Range (Min/Max)")
        print("  [2] Clear Range")
        print("  [D] Done")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'D':
            return current_range
        elif choice == '1':
            try:
                min_val = float(input(f"\nEnter minimum value for {param_name}: ").strip())
                max_val = float(input(f"Enter maximum value for {param_name}: ").strip())

                if min_val >= max_val:
                    print("✗ Minimum must be less than maximum")
                    ui_helpers.pause()
                    continue

                current_range = {'min': min_val, 'max': max_val}
                print(f"✓ Range set: [{min_val}, {max_val}]")
                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid number")
                ui_helpers.pause()
        elif choice == '2':
            current_range = None
            print("✓ Range cleared")
            ui_helpers.pause()


def generate_lhs_samples(ranges_dict, n_samples, existing_samples=None, tolerance=1e-6):
    """
    Generate Latin Hypercube Samples for all parameters.

    Parameters
    ----------
    ranges_dict : dict
        Dictionary mapping parameter keys to {'min': float, 'max': float}
    n_samples : int
        Number of samples to generate
    existing_samples : list of dict, optional
        Existing samples to check for redundancy
    tolerance : float
        Tolerance for considering samples as duplicates

    Returns
    -------
    list of dict
        List of sample dictionaries, each mapping parameter keys to values
    """
    if not ranges_dict:
        return []

    # Build ordered list of parameters
    param_keys = sorted(ranges_dict.keys())
    n_dims = len(param_keys)

    # Create LHS sampler
    sampler = qmc.LatinHypercube(d=n_dims, seed=None)

    # Generate samples in [0, 1] hypercube
    samples_unit = sampler.random(n=n_samples)

    # Scale to actual parameter ranges
    samples_list = []
    for sample_unit in samples_unit:
        sample_dict = {}
        for i, param_key in enumerate(param_keys):
            min_val = ranges_dict[param_key]['min']
            max_val = ranges_dict[param_key]['max']
            sample_dict[param_key] = min_val + sample_unit[i] * (max_val - min_val)
        samples_list.append(sample_dict)

    # Filter out redundant samples if existing_samples provided
    if existing_samples:
        filtered_samples = []
        for new_sample in samples_list:
            is_redundant = False
            for existing_sample in existing_samples:
                # Check if all parameters are within tolerance
                all_close = all(
                    abs(new_sample.get(key, 0) - existing_sample.get(key, 0)) < tolerance
                    for key in param_keys
                )
                if all_close:
                    is_redundant = True
                    break
            if not is_redundant:
                filtered_samples.append(new_sample)
        return filtered_samples

    return samples_list


def generate_factorial_samples(ranges_dict, n_points_per_param, existing_samples=None, tolerance=1e-6):
    """
    Generate Full Factorial samples for all parameters.

    Parameters
    ----------
    ranges_dict : dict
        Dictionary mapping parameter keys to {'min': float, 'max': float}
    n_points_per_param : int
        Number of evenly-spaced points per parameter
    existing_samples : list of dict, optional
        Existing samples to check for redundancy
    tolerance : float
        Tolerance for considering samples as duplicates

    Returns
    -------
    list of dict
        List of sample dictionaries, each mapping parameter keys to values
    """
    if not ranges_dict or n_points_per_param < 2:
        return []

    # Build ordered list of parameters
    param_keys = sorted(ranges_dict.keys())

    # Generate linspace for each parameter
    param_arrays = []
    for param_key in param_keys:
        min_val = ranges_dict[param_key]['min']
        max_val = ranges_dict[param_key]['max']
        param_arrays.append(np.linspace(min_val, max_val, n_points_per_param))

    # Generate all combinations
    import itertools
    combinations = list(itertools.product(*param_arrays))

    # Convert to list of dicts
    samples_list = []
    for combo in combinations:
        sample_dict = {}
        for i, param_key in enumerate(param_keys):
            sample_dict[param_key] = combo[i]
        samples_list.append(sample_dict)

    # Filter out redundant samples if existing_samples provided
    if existing_samples:
        filtered_samples = []
        for new_sample in samples_list:
            is_redundant = False
            for existing_sample in existing_samples:
                # Check if all parameters are within tolerance
                all_close = all(
                    abs(new_sample.get(key, 0) - existing_sample.get(key, 0)) < tolerance
                    for key in param_keys
                )
                if all_close:
                    is_redundant = True
                    break
            if not is_redundant:
                filtered_samples.append(new_sample)
        return filtered_samples

    return samples_list


def samples_to_doe_parameters(samples_list):
    """
    Convert list of sample dicts to doe_parameters format.

    Parameters
    ----------
    samples_list : list of dict
        List of samples where each sample maps 'bc_name|param_name' to value

    Returns
    -------
    dict
        DOE parameters in the format: {bc_name: {param_name: [values]}}
    """
    if not samples_list:
        return {}

    # Initialize structure
    doe_params = {}

    # Get all parameter keys from first sample
    param_keys = samples_list[0].keys()

    # Parse keys and organize values
    for param_key in param_keys:
        bc_name, param_name = param_key.split('|')

        if bc_name not in doe_params:
            doe_params[bc_name] = {}

        # Collect all values for this parameter across samples
        values = [sample[param_key] for sample in samples_list]
        doe_params[bc_name][param_name] = values

    return doe_params


def doe_parameters_to_ranges(doe_parameters):
    """
    Extract ranges from existing doe_parameters.

    Parameters
    ----------
    doe_parameters : dict
        DOE parameters in the format: {bc_name: {param_name: [values]}}

    Returns
    -------
    dict
        Ranges in format: {'bc_name|param_name': {'min': float, 'max': float}}
    """
    ranges = {}

    for bc_name, params in doe_parameters.items():
        for param_name, values in params.items():
            if values:  # Non-empty list
                param_key = f"{bc_name}|{param_name}"
                ranges[param_key] = {
                    'min': float(min(values)),
                    'max': float(max(values))
                }

    return ranges


def doe_parameters_to_samples(doe_parameters):
    """
    Convert doe_parameters to list of sample dicts.

    This function handles both LHS (parallel arrays) and full factorial formats.
    If all parameter arrays have the same length, they are zipped together.
    Otherwise, all combinations are generated using product.

    Parameters
    ----------
    doe_parameters : dict
        DOE parameters in the format: {bc_name: {param_name: [values]}}

    Returns
    -------
    list of dict
        List of samples where each sample maps 'bc_name|param_name' to value
    """
    if not doe_parameters:
        return []

    # Build parameter keys and arrays
    param_keys = []
    param_arrays = []

    for bc_name, params in sorted(doe_parameters.items()):
        for param_name, values in sorted(params.items()):
            if values:
                param_key = f"{bc_name}|{param_name}"
                param_keys.append(param_key)
                param_arrays.append(values)

    if not param_arrays:
        return []

    # Check if all arrays have the same length (indicates LHS or pre-generated samples)
    array_lengths = [len(arr) for arr in param_arrays]

    if len(set(array_lengths)) == 1:
        # All arrays same length - treat as parallel samples (LHS)
        combinations = list(zip(*param_arrays))
    else:
        # Different lengths - use full factorial
        import itertools
        combinations = list(itertools.product(*param_arrays))

    # Convert to list of dicts
    samples_list = []
    for combo in combinations:
        sample_dict = {}
        for i, param_key in enumerate(param_keys):
            sample_dict[param_key] = combo[i]
        samples_list.append(sample_dict)

    return samples_list


def setup_doe(solver, selected_inputs, doe_parameters, ui_helpers):
    """Configure Design of Experiment parameters for selected inputs."""

    if not selected_inputs:
        ui_helpers.print_header("DESIGN OF EXPERIMENT SETUP")
        print("\n✗ No model inputs selected! Please select inputs first (Option 1).")
        ui_helpers.pause()
        return doe_parameters

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("DESIGN OF EXPERIMENT SETUP")

        print("\nSELECTED INPUTS:")
        print("="*70)
        for i, item in enumerate(selected_inputs, 1):
            # Count only parameters that have values configured (non-empty lists)
            params_dict = doe_parameters.get(item['name'], {})
            num_params = sum(1 for values in params_dict.values() if values)
            status = f"({num_params} parameters configured)" if num_params > 0 else "(not configured)"
            print(f"  [{i:2d}] {item['name']:30s} {status}")
        print("="*70)

        print(f"\n{'='*70}")
        print("[Number] Configure DOE for input")
        print("[D] Done")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'D':
            return doe_parameters
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(selected_inputs):
                item = selected_inputs[idx]

                print(f"\nDetecting available parameters for {item['name']}...")

                # Get available parameters for this BC/zone type by querying Fluent
                available_params = get_bc_parameters(solver, item['name'], item['type'])

                if not available_params or (len(available_params) == 1 and available_params[0]['name'] == 'value'):
                    print(f"\n✗ Could not detect parameters for {item['type']}")
                    print("   This BC/zone type may not be supported or has no settable parameters.")
                    ui_helpers.pause()
                    continue

                # Initialize DOE config for this item if not exists
                if item['name'] not in doe_parameters:
                    doe_parameters[item['name']] = {}

                # Configure parameters with new range-based menu
                doe_parameters = setup_doe_for_input(item, available_params, doe_parameters, ui_helpers)

    return doe_parameters


def setup_doe_for_input(input_item, available_params, doe_parameters, ui_helpers):
    """
    Configure DOE for a specific input using range-based approach.

    Parameters
    ----------
    input_item : dict
        Input item with 'name' and 'type'
    available_params : list
        List of available parameters
    doe_parameters : dict
        Current DOE configuration
    ui_helpers : module
        UI helpers

    Returns
    -------
    dict
        Updated DOE parameters
    """
    bc_name = input_item['name']

    # Initialize storage for ranges
    param_ranges = {}  # {'bc_name|param_name': {'min': x, 'max': y}}

    # Try to extract existing ranges from doe_parameters
    if bc_name in doe_parameters:
        for param_name, values in doe_parameters[bc_name].items():
            if values:
                param_key = f"{bc_name}|{param_name}"
                param_ranges[param_key] = {
                    'min': float(min(values)),
                    'max': float(max(values))
                }

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header(f"DOE: {bc_name} ({input_item['type']})")

        # Calculate current sample count
        current_samples = doe_parameters_to_samples({bc_name: doe_parameters.get(bc_name, {})})
        n_samples = len(current_samples)

        print(f"\nCurrent Samples: {n_samples}")

        print("\nAVAILABLE PARAMETERS:")
        print("="*70)
        for i, param in enumerate(available_params, 1):
            param_path = param['path'] if param['path'] else param['name']
            param_key = f"{bc_name}|{param_path}"

            if param_key in param_ranges:
                range_info = f"[{param_ranges[param_key]['min']}, {param_ranges[param_key]['max']}]"
                status = f"Range: {range_info}"
            else:
                status = "(no range set)"

            print(f"  [{i:2d}] {param['name']:40s} {status}")
        print("="*70)

        print(f"\n{'='*70}")
        print("  [1] Define Parameter Ranges")
        print("  [2] Generate LHS Samples")
        print("  [3] Generate Full Factorial Samples")
        print("  [4] Add Manual Values (Legacy)")
        print("  [5] View Current Samples")
        print("  [6] Clear All Samples")
        print("  [B] Back")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'B':
            break
        elif choice == '1':
            # Define parameter ranges
            param_ranges = define_parameter_ranges(bc_name, available_params, param_ranges, ui_helpers)
        elif choice == '2':
            # Generate LHS samples
            if not param_ranges:
                print("\n✗ No parameter ranges defined! Please define ranges first (Option 1).")
                ui_helpers.pause()
                continue

            try:
                n_new_samples = int(input("\nEnter number of LHS samples to generate: ").strip())
                if n_new_samples <= 0:
                    print("✗ Number of samples must be positive")
                    ui_helpers.pause()
                    continue

                # Generate LHS samples
                existing_samples = doe_parameters_to_samples({bc_name: doe_parameters.get(bc_name, {})})
                new_samples = generate_lhs_samples(param_ranges, n_new_samples, existing_samples)

                print(f"\n✓ Generated {len(new_samples)} new LHS samples")
                print(f"  ({n_new_samples - len(new_samples)} were redundant and filtered out)")

                # Merge with existing samples
                all_samples = existing_samples + new_samples
                doe_parameters[bc_name] = samples_to_doe_parameters(all_samples)[bc_name]

                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid input")
                ui_helpers.pause()
        elif choice == '3':
            # Generate Full Factorial samples
            if not param_ranges:
                print("\n✗ No parameter ranges defined! Please define ranges first (Option 1).")
                ui_helpers.pause()
                continue

            try:
                n_points = int(input("\nEnter number of points per parameter: ").strip())
                if n_points < 2:
                    print("✗ Need at least 2 points per parameter")
                    ui_helpers.pause()
                    continue

                # Generate Full Factorial samples
                existing_samples = doe_parameters_to_samples({bc_name: doe_parameters.get(bc_name, {})})
                new_samples = generate_factorial_samples(param_ranges, n_points, existing_samples)

                n_params = len(param_ranges)
                expected_total = n_points ** n_params

                print(f"\n✓ Generated {len(new_samples)} new Full Factorial samples")
                print(f"  (Expected {expected_total}, {expected_total - len(new_samples)} were redundant)")

                # Merge with existing samples
                all_samples = existing_samples + new_samples
                doe_parameters[bc_name] = samples_to_doe_parameters(all_samples)[bc_name]

                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid input")
                ui_helpers.pause()
        elif choice == '4':
            # Legacy manual value entry
            while True:
                ui_helpers.clear_screen()
                ui_helpers.print_header(f"MANUAL VALUES: {bc_name}")

                print("\nAVAILABLE PARAMETERS:")
                print("="*70)
                for i, param in enumerate(available_params, 1):
                    param_key = param['path'] if param['path'] else param['name']
                    num_values = len(doe_parameters.get(bc_name, {}).get(param_key, []))
                    status = f"({num_values} values)" if num_values > 0 else "(not configured)"
                    print(f"  [{i:2d}] {param['name']:40s} {status}")
                print("="*70)

                print(f"\n{'='*70}")
                print("[Number] Configure parameter")
                print("[B] Back")
                print("="*70)

                param_choice = input("\nEnter choice: ").strip().upper()

                if param_choice == 'B':
                    break
                elif param_choice.isdigit():
                    param_idx = int(param_choice) - 1
                    if 0 <= param_idx < len(available_params):
                        param = available_params[param_idx]
                        param_key = param['path'] if param['path'] else param['name']

                        if bc_name not in doe_parameters:
                            doe_parameters[bc_name] = {}

                        current_values = doe_parameters[bc_name].get(param_key, [])
                        new_values = setup_parameter_values(param['name'], current_values.copy(), ui_helpers)

                        if new_values:
                            doe_parameters[bc_name][param_key] = new_values
                        elif param_key in doe_parameters[bc_name]:
                            del doe_parameters[bc_name][param_key]

                        if not doe_parameters[bc_name]:
                            del doe_parameters[bc_name]
        elif choice == '5':
            # View current samples
            view_doe_samples(bc_name, doe_parameters, ui_helpers)
        elif choice == '6':
            # Clear all samples
            confirm = input("\nAre you sure you want to clear all samples? [y/N]: ").strip().lower()
            if confirm == 'y':
                if bc_name in doe_parameters:
                    del doe_parameters[bc_name]
                print("✓ All samples cleared")
                ui_helpers.pause()

    return doe_parameters


def define_parameter_ranges(bc_name, available_params, current_ranges, ui_helpers):
    """
    Define ranges for parameters.

    Parameters
    ----------
    bc_name : str
        Boundary condition name
    available_params : list
        Available parameters
    current_ranges : dict
        Current ranges
    ui_helpers : module
        UI helpers

    Returns
    -------
    dict
        Updated ranges
    """
    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header(f"DEFINE RANGES: {bc_name}")

        print("\nAVAILABLE PARAMETERS:")
        print("="*70)
        for i, param in enumerate(available_params, 1):
            param_path = param['path'] if param['path'] else param['name']
            param_key = f"{bc_name}|{param_path}"

            if param_key in current_ranges:
                range_info = f"[{current_ranges[param_key]['min']}, {current_ranges[param_key]['max']}]"
                status = f"Range: {range_info}"
            else:
                status = "(no range set)"

            print(f"  [{i:2d}] {param['name']:40s} {status}")
        print("="*70)

        print(f"\n{'='*70}")
        print("[Number] Set range for parameter")
        print("[B] Back")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'B':
            break
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_params):
                param = available_params[idx]
                param_path = param['path'] if param['path'] else param['name']
                param_key = f"{bc_name}|{param_path}"

                current_range = current_ranges.get(param_key, None)
                new_range = setup_parameter_ranges(param['name'], current_range, ui_helpers)

                if new_range:
                    current_ranges[param_key] = new_range
                elif param_key in current_ranges:
                    del current_ranges[param_key]

    return current_ranges


def view_doe_samples(bc_name, doe_parameters, ui_helpers):
    """View current DOE samples."""
    ui_helpers.clear_screen()
    ui_helpers.print_header(f"VIEW SAMPLES: {bc_name}")

    if bc_name not in doe_parameters or not doe_parameters[bc_name]:
        print("\nNo samples configured yet.")
        ui_helpers.pause()
        return

    # Get all samples
    samples = doe_parameters_to_samples({bc_name: doe_parameters[bc_name]})

    if not samples:
        print("\nNo samples found.")
        ui_helpers.pause()
        return

    print(f"\nTotal Samples: {len(samples)}")
    print("\nFirst 20 samples:")
    print("="*70)

    # Get parameter names
    param_names = sorted(samples[0].keys())

    # Print header
    header = "  Sample  |  " + "  |  ".join([name.split('|')[1][:15] for name in param_names])
    print(header)
    print("-" * len(header))

    # Print samples (first 20)
    for i, sample in enumerate(samples[:20], 1):
        values = [f"{sample[pname]:12.6g}" for pname in param_names]
        print(f"  {i:5d}   |  " + "  |  ".join(values))

    if len(samples) > 20:
        print(f"\n... and {len(samples) - 20} more samples")

    ui_helpers.pause()


def analyze_setup_dimensions(setup_data):
    """
    Analyze model setup to determine input and output dimensionality.

    Parameters
    ----------
    setup_data : dict
        Model setup dictionary from JSON

    Returns
    -------
    dict
        Dictionary containing dimensional analysis:
        - num_inputs: Number of input variables
        - num_outputs: Number of output variables
        - input_details: List of input configurations
        - output_details: List of output configurations
        - total_input_combinations: Total number of input combinations in DOE
    """
    analysis = {
        'num_inputs': 0,
        'num_outputs': 0,
        'input_details': [],
        'output_details': [],
        'total_input_combinations': 0
    }

    # Collect all parameter value lengths to determine if LHS or full factorial
    all_param_lengths = []

    # Analyze inputs
    for input_item in setup_data.get('model_inputs', []):
        doe_params = input_item.get('doe_parameters', {})

        # Count parameters with configured values
        for param_name, param_values in doe_params.items():
            if param_values:  # Non-empty list
                analysis['num_inputs'] += 1
                all_param_lengths.append(len(param_values))
                analysis['input_details'].append({
                    'bc_name': input_item['name'],
                    'bc_type': input_item['type'],
                    'parameter': param_name,
                    'num_values': len(param_values),
                    'range': [min(param_values), max(param_values)] if param_values else None
                })

    # Calculate total combinations based on whether all lengths are equal (LHS) or not (full factorial)
    if all_param_lengths:
        if len(set(all_param_lengths)) == 1:
            # All parameters have same length - LHS or parallel samples
            analysis['total_input_combinations'] = all_param_lengths[0]
        else:
            # Different lengths - full factorial
            analysis['total_input_combinations'] = 1
            for length in all_param_lengths:
                analysis['total_input_combinations'] *= length

    # Analyze outputs
    for output_item in setup_data.get('model_outputs', []):
        analysis['num_outputs'] += 1
        analysis['output_details'].append({
            'name': output_item['name'],
            'type': output_item['type'],
            'category': output_item.get('category', 'Unknown')
        })

    return analysis


def create_dataset_structure(dataset_dir, analysis, setup_data, ui_helpers):
    """Create the directory structure for dataset storage."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("CREATE DATASET STRUCTURE")

    print(f"\nCreating dataset directory: {dataset_dir}")

    try:
        # Create main dataset directory
        dataset_dir.mkdir(exist_ok=True)

        # Create dataset directory for simulation outputs
        (dataset_dir / "dataset").mkdir(exist_ok=True)


        # Create README
        readme_path = dataset_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write("Dataset Directory Structure\n")
            f.write("="*70 + "\n\n")
            f.write(f"Created: {setup_data['timestamp']}\n")
            f.write(f"Required Simulations: {analysis['total_input_combinations']}\n")
            f.write(f"Input Variables: {analysis['num_inputs']}\n")
            f.write(f"Output Locations: {analysis['num_outputs']}\n\n")


    except Exception as e:
        print(f"\n✗ Error creating dataset structure: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()


def add_lhs_points_simple(doe_parameters, ui_helpers):
    """
    Simple function to add Latin Hypercube Sample points to existing DOE.

    Parameters
    ----------
    doe_parameters : dict
        Existing DOE configuration: {bc_name: {param_name: [values]}}
    ui_helpers : module
        UI helpers module

    Returns
    -------
    dict
        Updated DOE parameters with LHS points added
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("ADD LATIN HYPERCUBE SAMPLE POINTS")

    if not doe_parameters:
        print("\n✗ No DOE parameters configured yet. Please use 'Design of Experiment Setup' first.")
        ui_helpers.pause()
        return doe_parameters

    # Extract ranges from existing DOE parameters
    ranges_dict = doe_parameters_to_ranges(doe_parameters)

    if not ranges_dict:
        print("\n✗ No parameter ranges found. Please configure DOE first.")
        ui_helpers.pause()
        return doe_parameters

    # Display current configuration
    print("\nCurrent Parameter Ranges:")
    print(f"{'='*70}")
    for param_key, range_info in sorted(ranges_dict.items()):
        print(f"  {param_key}:")
        print(f"    Min: {range_info['min']:.6f}")
        print(f"    Max: {range_info['max']:.6f}")
    print(f"{'='*70}")

    # Get existing samples
    existing_samples = doe_parameters_to_samples(doe_parameters)
    print(f"\nCurrent number of samples: {len(existing_samples)}")

    # Ask how many LHS points to add
    print(f"\nHow many Latin Hypercube sample points would you like to add?")
    n_samples_str = input("Number of points to add: ").strip()

    try:
        n_samples = int(n_samples_str)
        if n_samples <= 0:
            print("\n✗ Number of samples must be positive")
            ui_helpers.pause()
            return doe_parameters
    except ValueError:
        print("\n✗ Invalid number")
        ui_helpers.pause()
        return doe_parameters

    print(f"\nGenerating {n_samples} Latin Hypercube samples...")
    print("  (Checking for redundancy with existing samples...)")

    # Generate LHS samples with redundancy check
    new_samples = generate_lhs_samples(ranges_dict, n_samples, existing_samples, tolerance=1e-6)

    if len(new_samples) < n_samples:
        print(f"\n⚠ Warning: {n_samples - len(new_samples)} samples were redundant and excluded")

    if not new_samples:
        print("\n✗ No new unique samples generated. Try increasing the number of samples.")
        ui_helpers.pause()
        return doe_parameters

    print(f"\n✓ Generated {len(new_samples)} new unique samples")

    # Combine with existing samples
    all_samples = existing_samples + new_samples

    # Convert back to doe_parameters format
    updated_doe_parameters = samples_to_doe_parameters(all_samples)

    print(f"\n✓ Total samples: {len(all_samples)} ({len(existing_samples)} existing + {len(new_samples)} new)")
    ui_helpers.pause()

    return updated_doe_parameters
