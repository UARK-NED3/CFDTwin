"""
Simulation Runner Module
=========================
Handles running Fluent simulations for all DOE combinations and saving results.
"""

import numpy as np
from pathlib import Path
import json
import itertools
import time
import sys
import warnings
from io import StringIO

# Suppress PyFluent deprecation warnings for get_scalar_field_data
# (The new get_field_data API requires complex request objects;
# we'll continue using the simpler deprecated method for now)
warnings.filterwarnings('ignore', message="'get_scalar_field_data' is deprecated")


def run_simulations_menu(solver, setup_data, analysis, dataset_dir, ui_helpers):
    """
    Menu for running Fluent simulations for all DOE combinations.

    Parameters
    ----------
    solver : PyFluent solver
        Active Fluent solver instance
    setup_data : dict
        Model setup configuration
    analysis : dict
        Dimensional analysis
    dataset_dir : Path
        Dataset directory path
    ui_helpers : module
        UI helpers module
    """
    if not solver:
        ui_helpers.print_header("RUN SIMULATIONS")
        print("\n✗ No active Fluent session! Please open a case file first.")
        ui_helpers.pause()
        return

    dataset_output_dir = dataset_dir / "dataset"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("RUN SIMULATIONS")

        print(f"\nDataset Directory: {dataset_dir}")
        print(f"Total Simulations Required: {analysis['total_input_combinations']}")

        # Count existing simulation results
        existing_results = list(dataset_output_dir.glob("sim_*.npz"))
        print(f"Completed Simulations: {len(existing_results)}")

        completeness = len(existing_results) / analysis['total_input_combinations'] * 100 if analysis['total_input_combinations'] > 0 else 0
        print(f"Progress: {completeness:.1f}%")

        print(f"\n{'='*70}")
        print("  [1] Run Single Simulation (Manual)")
        print("  [2] Run All Simulations (Batch)")
        print("  [3] Run Remaining Simulations")
        print("  [4] View Simulation Status")
        print("  [5] Extract Data from Current Solution")
        print("  [0] Back")
        print("="*70)

        choice = ui_helpers.get_choice(5)

        if choice == 0:
            return
        elif choice == 1:
            run_single_simulation(solver, setup_data, dataset_dir, ui_helpers)
        elif choice == 2:
            run_batch_simulations(solver, setup_data, analysis, dataset_dir, ui_helpers)
        elif choice == 3:
            run_remaining_simulations(solver, setup_data, analysis, dataset_dir, existing_results, ui_helpers)
        elif choice == 4:
            view_simulation_status(analysis, existing_results, ui_helpers)
        elif choice == 5:
            extract_current_solution(solver, setup_data, dataset_dir, ui_helpers)


def generate_doe_combinations(setup_data):
    """
    Generate all DOE combinations from setup data.

    This function handles both legacy full-factorial DOE and new LHS/range-based DOE.
    For LHS, the samples are already pre-generated and stored as parallel arrays.
    For legacy, we use itertools.product to generate all combinations.

    Parameters
    ----------
    setup_data : dict
        Model setup configuration

    Returns
    -------
    list
        List of tuples, each containing (sim_id, bc_values_dict)
    """
    doe_config = setup_data.get('doe_configuration', {})

    # Build parameter arrays and mapping
    # CRITICAL: Iterate in SAME order as trainer (doe_config.items())
    param_arrays = []
    param_mapping = []  # List of (bc_name, bc_type, param_name, param_path)

    # Build bc_type mapping from model_inputs
    bc_type_map = {}
    for input_item in setup_data['model_inputs']:
        bc_type_map[input_item['name']] = input_item['type']

    # Iterate through doe_config (SAME as training order)
    for bc_name, doe_params in doe_config.items():
        bc_type = bc_type_map.get(bc_name, 'Unknown')

        for param_name, param_values in doe_params.items():
            if param_values:
                param_arrays.append(param_values)
                # Store the parameter path for applying BCs
                param_mapping.append({
                    'bc_name': bc_name,
                    'bc_type': bc_type,
                    'param_name': param_name,
                    'param_path': param_name  # Full path like 'vmag' or 'temperature'
                })

    if not param_arrays:
        return []

    # Check if all arrays have the same length (indicates LHS or pre-generated samples)
    # If they do, zip them together. Otherwise, use product for full factorial.
    array_lengths = [len(arr) for arr in param_arrays]

    if len(set(array_lengths)) == 1:
        # All arrays same length - treat as parallel samples (LHS or manually added)
        # Zip the arrays together instead of using product
        combinations = list(zip(*param_arrays))
    else:
        # Different lengths - use full factorial (legacy behavior)
        combinations = list(itertools.product(*param_arrays))

    # Create list of (sim_id, bc_values_dict)
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

    Parameters
    ----------
    solver : PyFluent solver
        Active Fluent solver instance
    bc_values : dict
        Dictionary of BC values to apply

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        boundary_conditions = solver.settings.setup.boundary_conditions

        for bc_key, bc_info in bc_values.items():
            bc_name = bc_info['bc_name']
            bc_type = bc_info['bc_type'].lower().replace(' ', '_')
            param_path = bc_info['param_path']
            value = bc_info['value']

            # Get the BC object using dictionary-style access (like Field/Volume Surrogate)
            if hasattr(boundary_conditions, bc_type):
                bc_container = getattr(boundary_conditions, bc_type)
                if bc_name in bc_container:
                    bc_obj = bc_container[bc_name]

                    # Navigate to parameter using path - use direct attribute access chain
                    try:
                        # Build attribute chain: bc_obj.momentum.velocity_magnitude.value
                        path_parts = param_path.split('.')
                        target_obj = bc_obj

                        # Navigate through ALL parts to get to the parameter object
                        for i, part in enumerate(path_parts):
                            if hasattr(target_obj, part):
                                target_obj = getattr(target_obj, part)
                            else:
                                # Try alternate names for velocity parameters
                                if part == 'velocity_magnitude' and hasattr(target_obj, 'velocity'):
                                    print(f"  Note: Using 'velocity' instead of 'velocity_magnitude' for {bc_name}")
                                    target_obj = getattr(target_obj, 'velocity')
                                elif part == 'velocity' and hasattr(target_obj, 'velocity_magnitude'):
                                    print(f"  Note: Using 'velocity_magnitude' instead of 'velocity' for {bc_name}")
                                    target_obj = getattr(target_obj, 'velocity_magnitude')
                                else:
                                    print(f"  ✗ Error: Path part '{part}' not found in {bc_name}")
                                    print(f"     Full path attempted: {bc_name}.{param_path}")
                                    if hasattr(target_obj, 'child_names'):
                                        try:
                                            available = target_obj.child_names
                                            print(f"     Available: {available[:15] if len(available) > 15 else available}")
                                        except:
                                            pass
                                    return False

                        # Now target_obj should have a .value attribute we can set
                        if hasattr(target_obj, 'value'):
                            # Set the new value (convert to float to match Field/Volume Surrogate)
                            target_obj.value = float(value)

                            # Simple confirmation - don't try to verify since Fluent returns internal objects
                            print(f"  ✓ {bc_name}.{param_path} = {value}")

                        else:
                            print(f"  ✗ Error: {bc_name}.{param_path} has no .value attribute")
                            print(f"     Object type: {type(target_obj).__name__}")
                            return False

                    except Exception as e:
                        print(f"  ✗ Exception setting {bc_name}.{param_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
                else:
                    print(f"  ✗ Error: BC '{bc_name}' not found in {bc_type}")
                    return False
            else:
                print(f"  ✗ Error: BC type '{bc_type}' not found")
                return False

        return True

    except Exception as e:
        print(f"\n✗ Error applying boundary conditions: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_field_data(solver, setup_data, dataset_dir):
    """
    Extract field data from configured output locations.
    Handles surfaces (2D), cell zones (3D), and report definitions (scalar).

    Parameters
    ----------
    solver : PyFluent solver
        Active Fluent solver instance
    setup_data : dict
        Model setup configuration
    dataset_dir : Path
        Dataset directory path

    Returns
    -------
    dict or None
        Dictionary containing extracted data, or None if error
    """
    try:
        # Load output parameters configuration
        output_params_file = dataset_dir / "output_parameters.json"
        if not output_params_file.exists():
            print("\n✗ output_parameters.json not found!")
            print("  Please configure output parameters in I/O Setup menu.")
            return None

        with open(output_params_file, 'r') as f:
            output_params = json.load(f)

        output_data = {}

        # Extract data from each configured output
        for output_item in setup_data['model_outputs']:
            output_name = output_item['name']
            output_type = output_item['type']
            output_category = output_item['category']

            # Get configured parameters for this output
            params_to_extract = output_params.get(output_name, [])

            # Report Definitions don't need configuration - extract automatically
            if output_category == "Report Definition":
                print(f"  Extracting from {output_name} ({output_category})...")
                try:
                    # Get report definition value using compute method
                    result = solver.settings.solution.report_definitions.compute(report_defs=[output_name])

                    # Extract value from result structure: [{'report-name': [value, ...]}]
                    report_value = result[0][output_name][0]

                    # Use generic 'value' as field name for report definitions
                    # Or use first configured param if available
                    param_name = params_to_extract[0] if params_to_extract else 'value'
                    key = f"{output_name}|{param_name}"
                    output_data[key] = np.array([report_value])
                    print(f"    ✓ {param_name}: {report_value:.6f}")

                except Exception as e:
                    print(f"    ✗ Error extracting report definition: {e}")
                continue

            # For other output types, configuration is required
            if not params_to_extract:
                print(f"  ⚠ Warning: No parameters configured for {output_name}, skipping")
                continue

            print(f"  Extracting from {output_name} ({output_category})...")
            print(f"    Parameters: {', '.join(params_to_extract)}")

            if output_category == "Cell Zone":
                # Extract volume (3D) data using solution_variable_data
                solution_data = solver.fields.solution_variable_data

                # Extract cell centroids first (only once per zone)
                coord_key = f"{output_name}|coordinates"
                if coord_key not in output_data:
                    try:
                        print(f"  Extracting cell centroids...")
                        centroid_dict = solution_data.get_data(
                            zone_names=[output_name],
                            variable_name='SV_CENTROID',
                            domain_name='mixture'
                        )

                        # SV_CENTROID returns flat array [x1,y1,z1,x2,y2,z2,...]
                        centroids_flat = np.array(centroid_dict[output_name])

                        # Reshape to (n_cells, 3)
                        n_cells = len(centroids_flat) // 3
                        coordinates = centroids_flat.reshape((n_cells, 3))

                        output_data[coord_key] = coordinates
                        print(f"    ✓ Centroids: {len(coordinates)} cells")
                    except Exception as e:
                        print(f"    ✗ Error extracting centroids: {e}")

                # Extract field parameters
                for param_name in params_to_extract:
                    try:
                        # Special handling for velocity-magnitude (needs to be computed from components)
                        if param_name == 'velocity-magnitude':
                            # Extract velocity components
                            u_dict = solution_data.get_data(
                                zone_names=[output_name],
                                variable_name='SV_U',
                                domain_name='mixture'
                            )
                            v_dict = solution_data.get_data(
                                zone_names=[output_name],
                                variable_name='SV_V',
                                domain_name='mixture'
                            )
                            w_dict = solution_data.get_data(
                                zone_names=[output_name],
                                variable_name='SV_W',
                                domain_name='mixture'
                            )

                            # Compute magnitude
                            u = np.array(u_dict[output_name])
                            v = np.array(v_dict[output_name])
                            w = np.array(w_dict[output_name])
                            velocity_mag = np.sqrt(u**2 + v**2 + w**2)

                            # Store the data
                            key = f"{output_name}|{param_name}"
                            output_data[key] = velocity_mag
                            print(f"    ✓ {param_name}: {len(velocity_mag)} cells (computed from U,V,W)")
                            continue

                        # Map parameter names to Fluent solution variable names (SV_*)
                        fluent_var_map = {
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

                        fluent_var = fluent_var_map.get(param_name, param_name.upper())

                        # Get available variables for this zone first
                        try:
                            solution_var_info = solver.fields.solution_variable_info
                            zone_info = solution_var_info.get_variables_info(
                                zone_names=[output_name],
                                domain_name='mixture'
                            )
                            available_vars = zone_info.solution_variables

                            if fluent_var not in available_vars:
                                print(f"    ⚠ Variable {fluent_var} not available for zone {output_name}")
                                print(f"      Available variables: {', '.join(available_vars)}")
                                continue
                        except Exception as info_error:
                            print(f"    ⚠ Could not query available variables: {info_error}")
                            # Continue anyway and let get_data raise error if variable doesn't exist

                        # Get volume data
                        data_dict = solution_data.get_data(
                            zone_names=[output_name],
                            variable_name=fluent_var,
                            domain_name='mixture'
                        )

                        # Extract data for the zone
                        zone_data = np.array(data_dict[output_name])

                        # Store the data
                        key = f"{output_name}|{param_name}"
                        output_data[key] = zone_data
                        print(f"    ✓ {param_name}: {len(zone_data)} cells")

                    except Exception as e:
                        print(f"    ✗ Error extracting {param_name}: {e}")

            else:
                # Extract surface (2D) data using field_data
                field_data = solver.fields.field_data

                # Extract coordinates first (only once per surface)
                coord_key = f"{output_name}|coordinates"
                if coord_key not in output_data:
                    try:
                        print(f"  Extracting coordinates...")
                        x_dict = field_data.get_scalar_field_data(
                            field_name='x-coordinate',
                            surfaces=[output_name],
                            node_value=False
                        )
                        y_dict = field_data.get_scalar_field_data(
                            field_name='y-coordinate',
                            surfaces=[output_name],
                            node_value=False
                        )
                        z_dict = field_data.get_scalar_field_data(
                            field_name='z-coordinate',
                            surfaces=[output_name],
                            node_value=False
                        )

                        # Extract data - handle both dict and direct array formats
                        def extract_data(data_dict, key):
                            surface_data = data_dict[key]
                            if isinstance(surface_data, dict) and 'scalar-field' in surface_data:
                                return np.array(surface_data['scalar-field'])
                            else:
                                return np.array(surface_data)

                        x_coords = extract_data(x_dict, output_name)
                        y_coords = extract_data(y_dict, output_name)
                        z_coords = extract_data(z_dict, output_name)

                        # Stack into (n_points, 3) array
                        coordinates = np.stack([x_coords, y_coords, z_coords], axis=1)
                        output_data[coord_key] = coordinates
                        print(f"    ✓ Coordinates: {len(coordinates)} points")
                    except Exception as e:
                        print(f"    ✗ Error extracting coordinates: {e}")

                # Extract field parameters
                for param_name in params_to_extract:
                    try:
                        # Get scalar field data for this surface
                        data_dict = field_data.get_scalar_field_data(
                            field_name=param_name,
                            surfaces=[output_name],
                            node_value=False
                        )

                        # Extract data - handle both dict and direct array formats
                        def extract_data(data_dict, key):
                            surface_data = data_dict[key]
                            if isinstance(surface_data, dict) and 'scalar-field' in surface_data:
                                return np.array(surface_data['scalar-field'])
                            else:
                                return np.array(surface_data)

                        surface_data = extract_data(data_dict, output_name)

                        # Store the data
                        key = f"{output_name}|{param_name}"
                        output_data[key] = surface_data

                        # Show statistics to verify data
                        stats = f"{len(surface_data)} points, range: [{surface_data.min():.6f}, {surface_data.max():.6f}]"
                        print(f"    ✓ {param_name}: {stats}")

                    except Exception as e:
                        print(f"    ✗ Error extracting {param_name}: {e}")

        return output_data if output_data else None

    except Exception as e:
        print(f"\n✗ Error extracting field data: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_single_simulation(solver, setup_data, dataset_dir, ui_helpers):
    """Run a single simulation with manual or CSV input."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("RUN SINGLE SIMULATION")

    # Generate DOE combinations
    doe_list = generate_doe_combinations(setup_data)
    if not doe_list:
        print("\n✗ No DOE combinations found!")
        print("  Please configure DOE parameters first.")
        ui_helpers.pause()
        return

    print(f"\nTotal available simulations: {len(doe_list)}")

    # Ask for simulation ID
    sim_id_str = input("\nEnter simulation ID to run (1-{}): ".format(len(doe_list))).strip()

    if not sim_id_str.isdigit():
        print("✗ Invalid simulation ID")
        ui_helpers.pause()
        return

    sim_id = int(sim_id_str)
    if sim_id < 1 or sim_id > len(doe_list):
        print(f"✗ Simulation ID must be between 1 and {len(doe_list)}")
        ui_helpers.pause()
        return

    # Get the BC values for this simulation
    _, bc_values = doe_list[sim_id - 1]

    print(f"\n{'='*70}")
    print(f"SIMULATION {sim_id}")
    print("="*70)
    print("\nBoundary Conditions to Apply:")
    for bc_key, bc_info in bc_values.items():
        print(f"  {bc_info['bc_name']}.{bc_info['param_name']} = {bc_info['value']}")

    confirm = input("\nProceed with this simulation? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("\n✗ Cancelled")
        ui_helpers.pause()
        return

    # Run the simulation
    print(f"\n{'='*70}")
    print("RUNNING SIMULATION")
    print("="*70)

    # Step 1: Apply BCs
    print("\n[1/4] Applying boundary conditions...")
    if not apply_boundary_conditions(solver, bc_values):
        print("\n✗ Failed to apply boundary conditions")
        ui_helpers.pause()
        return

    # Step 2: Initialize
    print("\n[2/4] Initializing solution...")
    try:
        # Suppress Fluent output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        # Use hybrid initialization
        solver.settings.solution.initialization.hybrid_initialize()

        # Restore output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print("  ✓ Initialized (hybrid)")
    except Exception as e:
        # Restore output on error
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print(f"  ✗ Initialization error: {e}")
        ui_helpers.pause()
        return

    # Step 3: Solve
    print("\n[3/4] Running solution...")
    try:
        # Get number of iterations from user
        iterations_str = input("  Enter number of iterations [100]: ").strip() or "100"
        iterations = int(iterations_str)

        print(f"  Running {iterations} iterations...")

        # Suppress Fluent output during solve
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        solver.settings.solution.run_calculation.iterate(iter_count=iterations)

        # Restore output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print("  ✓ Solution complete")
    except Exception as e:
        # Restore output on error
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print(f"  ✗ Solution error: {e}")
        ui_helpers.pause()
        return

    # Step 4: Extract data
    print("\n[4/4] Extracting field data...")
    output_data = extract_field_data(solver, setup_data, dataset_dir)
    if output_data is None:
        print("\n✗ Failed to extract field data")
        ui_helpers.pause()
        return

    # Save results
    output_file = dataset_dir / "dataset" / f"sim_{sim_id:04d}.npz"
    np.savez_compressed(output_file, **output_data)
    print(f"\n✓ Results saved to: {output_file.name}")
    print(f"  Fields saved: {len(output_data)}")

    ui_helpers.pause()


def run_batch_simulations(solver, setup_data, analysis, dataset_dir, ui_helpers):
    """Run all simulations in batch mode."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("RUN BATCH SIMULATIONS")

    # Generate DOE combinations
    doe_list = generate_doe_combinations(setup_data)
    if not doe_list:
        print("\n✗ No DOE combinations found!")
        ui_helpers.pause()
        return

    total_sims = len(doe_list)
    print(f"\nTotal simulations: {total_sims}")

    # Get simulation parameters
    print("\nSimulation Parameters:")
    iterations_str = input("  Iterations per simulation [100]: ").strip() or "100"
    iterations = int(iterations_str)

    save_interval_str = input("  Save progress every N simulations [10]: ").strip() or "10"
    save_interval = int(save_interval_str)

    confirm = input(f"\nRun {total_sims} simulations? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("\n✗ Cancelled")
        ui_helpers.pause()
        return

    # Run all simulations
    print(f"\n{'='*70}")
    print("BATCH SIMULATION")
    print("="*70)

    outputs_dir = dataset_dir / "dataset"
    outputs_dir.mkdir(exist_ok=True)

    start_time = time.time()
    successful = 0
    failed = 0

    is_first_sim = True
    for sim_id, bc_values in doe_list:
        # Check if simulation already exists
        output_file = outputs_dir / f"sim_{sim_id:04d}.npz"
        if output_file.exists():
            print(f"\n[{sim_id}/{total_sims}] Skipping (already exists)")
            successful += 1
            continue

        print(f"\n{'='*70}")
        print(f"[{sim_id}/{total_sims}] Simulation {sim_id}")
        print("="*70)

        # Apply BCs
        print("Applying BCs...")
        if not apply_boundary_conditions(solver, bc_values):
            print("✗ Failed to apply BCs")
            failed += 1
            continue

        # Initialize only for the first simulation in batch
        if is_first_sim:
            try:
                # Suppress Fluent output
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()

                solver.settings.solution.initialization.hybrid_initialize()

                sys.stdout = old_stdout
                sys.stderr = old_stderr
                print("Initialized (hybrid)")
                is_first_sim = False
            except Exception as e:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                print(f"✗ Initialization failed: {e}")
                failed += 1
                continue
        else:
            print("Continuing from previous solution (no reinitialization)")

        # Solve
        try:
            # Suppress Fluent output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()

            solver.settings.solution.run_calculation.iterate(iter_count=iterations)

            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            print(f"✗ Solution failed: {e}")
            failed += 1
            continue

        # Extract data
        output_data = extract_field_data(solver, setup_data, dataset_dir)
        if output_data is None:
            print("✗ Data extraction failed")
            failed += 1
            continue

        # Save
        try:
            np.savez_compressed(output_file, **output_data)
            print(f"✓ Saved {output_file.name}")
            successful += 1
        except Exception as e:
            print(f"✗ Save failed: {e}")
            failed += 1
            continue

        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / sim_id
        remaining = (total_sims - sim_id) * avg_time
        print(f"\nProgress: {successful}/{total_sims} complete, {failed} failed")
        print(f"Time: {elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining")

    # Final summary
    print(f"\n{'='*70}")
    print("BATCH SIMULATION COMPLETE")
    print("="*70)
    print(f"  Successful: {successful}/{total_sims}")
    print(f"  Failed: {failed}/{total_sims}")
    print(f"  Total time: {(time.time() - start_time)/60:.1f} minutes")

    ui_helpers.pause()


def run_remaining_simulations(solver, setup_data, analysis, dataset_dir, existing_results, ui_helpers):
    """Run only the simulations that haven't been completed yet."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("RUN REMAINING SIMULATIONS")

    # Generate DOE combinations
    doe_list = generate_doe_combinations(setup_data)
    if not doe_list:
        print("\n✗ No DOE combinations found!")
        ui_helpers.pause()
        return

    # Find completed simulation IDs
    completed_ids = set()
    for result_file in existing_results:
        try:
            sim_id = int(result_file.stem.split('_')[1])
            completed_ids.add(sim_id)
        except:
            pass

    # Filter to only remaining
    remaining_doe = [(sim_id, bc_vals) for sim_id, bc_vals in doe_list if sim_id not in completed_ids]

    if not remaining_doe:
        print("\n✓ All simulations complete!")
        ui_helpers.pause()
        return

    total_sims = len(remaining_doe)
    print(f"\nRemaining simulations: {total_sims}")
    print(f"Completed: {len(completed_ids)}")

    # Get simulation parameters
    print("\nSimulation Parameters:")
    iterations_str = input("  Iterations per simulation [100]: ").strip() or "100"
    iterations = int(iterations_str)

    confirm = input(f"\nRun {total_sims} remaining simulations? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("\n✗ Cancelled")
        ui_helpers.pause()
        return

    # Run remaining simulations
    print(f"\n{'='*70}")
    print("RESUMING BATCH SIMULATION")
    print("="*70)

    outputs_dir = dataset_dir / "dataset"
    start_time = time.time()
    successful = 0
    failed = 0

    is_first_sim = True
    for idx, (sim_id, bc_values) in enumerate(remaining_doe, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{total_sims}] Simulation {sim_id}")
        print("="*70)

        # Apply BCs
        print("Applying BCs...")
        if not apply_boundary_conditions(solver, bc_values):
            print("✗ Failed to apply BCs")
            failed += 1
            continue

        # Initialize only for the first simulation in batch
        if is_first_sim:
            try:
                # Suppress Fluent output
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()

                solver.settings.solution.initialization.hybrid_initialize()

                sys.stdout = old_stdout
                sys.stderr = old_stderr
                print("Initialized (hybrid)")
                is_first_sim = False
            except Exception as e:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                print(f"✗ Initialization failed: {e}")
                failed += 1
                continue
        else:
            print("Continuing from previous solution (no reinitialization)")

        # Solve
        try:
            # Suppress Fluent output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()

            solver.settings.solution.run_calculation.iterate(iter_count=iterations)

            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            print(f"✗ Solution failed: {e}")
            failed += 1
            continue

        # Extract data
        output_data = extract_field_data(solver, setup_data, dataset_dir)
        if output_data is None:
            print("✗ Data extraction failed")
            failed += 1
            continue

        # Save
        try:
            output_file = outputs_dir / f"sim_{sim_id:04d}.npz"
            np.savez_compressed(output_file, **output_data)
            print(f"✓ Saved {output_file.name}")
            successful += 1
        except Exception as e:
            print(f"✗ Save failed: {e}")
            failed += 1
            continue

        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining_time = (total_sims - idx) * avg_time
        print(f"\nProgress: {successful}/{total_sims} complete, {failed} failed")
        print(f"Time: {elapsed/60:.1f}m elapsed, ~{remaining_time/60:.1f}m remaining")

    # Final summary
    print(f"\n{'='*70}")
    print("REMAINING SIMULATIONS COMPLETE")
    print("="*70)
    print(f"  Successful: {successful}/{total_sims}")
    print(f"  Failed: {failed}/{total_sims}")
    print(f"  Total time: {(time.time() - start_time)/60:.1f} minutes")

    ui_helpers.pause()


def view_simulation_status(analysis, existing_results, ui_helpers):
    """Display detailed simulation status."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("SIMULATION STATUS")

    print(f"\nTotal Required: {analysis['total_input_combinations']}")
    print(f"Completed: {len(existing_results)}")
    print(f"Remaining: {analysis['total_input_combinations'] - len(existing_results)}")

    completeness = len(existing_results) / analysis['total_input_combinations'] * 100 if analysis['total_input_combinations'] > 0 else 0
    print(f"\nProgress: {completeness:.1f}%")

    # Show completed simulation IDs
    if existing_results:
        print("\nCompleted Simulations:")
        sim_ids = sorted([int(f.stem.split('_')[1]) for f in existing_results])
        print(f"  IDs: {sim_ids[:20]}{'...' if len(sim_ids) > 20 else ''}")

        # Show file sizes
        total_size = sum(f.stat().st_size for f in existing_results)
        avg_size = total_size / len(existing_results)
        print(f"\n  Total data size: {total_size / 1024**2:.1f} MB")
        print(f"  Average per simulation: {avg_size / 1024**2:.1f} MB")

    ui_helpers.pause()


def extract_current_solution(solver, setup_data, dataset_dir, ui_helpers):
    """Extract data from the currently loaded solution."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("EXTRACT CURRENT SOLUTION")

    try:
        sim_id = input("\nEnter simulation ID for this solution: ").strip()

        if not sim_id.isdigit():
            print("✗ Invalid simulation ID")
            ui_helpers.pause()
            return

        sim_id_int = int(sim_id)

        print(f"\n{'='*70}")
        print(f"EXTRACTING SIMULATION {sim_id_int}")
        print("="*70)

        # Extract field data
        print("\nExtracting field data...")
        output_data = extract_field_data(solver, setup_data, dataset_dir)

        if output_data is None:
            print("\n✗ Failed to extract field data")
            ui_helpers.pause()
            return

        # Save results
        outputs_dir = dataset_dir / "dataset"
        outputs_dir.mkdir(exist_ok=True)
        output_file = outputs_dir / f"sim_{sim_id_int:04d}.npz"

        np.savez_compressed(output_file, **output_data)

        print(f"\n✓ Results saved to: {output_file.name}")
        print(f"  Fields saved: {len(output_data)}")
        print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()
