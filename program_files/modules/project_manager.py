"""
Project Manager Module
======================
Handles project configuration including input/output selection and saving.
"""

import json
from pathlib import Path
from datetime import datetime


def setup_model_inputs(solver, selected_inputs, ui_helpers):
    """Configure model inputs (boundary conditions and cell zones)."""

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("CONFIGURE MODEL INPUTS")

        # Show selected items at top
        if selected_inputs:
            print("\n" + "="*70)
            print("SELECTED INPUTS:")
            print("="*70)
            for i, item in enumerate(selected_inputs, 1):
                print(f"  [{i:2d}] {item['name']:30s} (Type: {item['type']})")
            print("="*70)

        print("\nLoading boundary conditions and cell zones...")

        # Get boundary conditions
        try:
            boundary_conditions = solver.settings.setup.boundary_conditions
            surfaces = []

            for bc_type in dir(boundary_conditions):
                if bc_type.startswith('_') or bc_type in ['child_names', 'command_names']:
                    continue

                bc_obj = getattr(boundary_conditions, bc_type)
                if hasattr(bc_obj, '__iter__') and not isinstance(bc_obj, str):
                    try:
                        for name in bc_obj:
                            if name not in ['child_names', 'command_names']:
                                surfaces.append({
                                    'name': name,
                                    'type': bc_type.replace('_', ' ').title(),
                                    'category': 'Boundary Condition'
                                })
                    except Exception as e:
                        pass
        except Exception as e:
            print(f"Warning: Error loading boundary conditions: {e}")
            surfaces = []

        # Get cell zones
        try:
            cell_zones_obj = solver.settings.setup.cell_zone_conditions
            cell_zones = []

            for zone_type in dir(cell_zones_obj):
                if zone_type.startswith('_') or zone_type in ['child_names', 'command_names']:
                    continue

                zone_obj = getattr(cell_zones_obj, zone_type)
                if hasattr(zone_obj, '__iter__') and not isinstance(zone_obj, str):
                    try:
                        for name in zone_obj:
                            if name not in ['child_names', 'command_names']:
                                cell_zones.append({
                                    'name': name,
                                    'type': zone_type.replace('_', ' ').title(),
                                    'category': 'Cell Zone'
                                })
                    except Exception as e:
                        pass
        except Exception as e:
            print(f"Warning: Error loading cell zones: {e}")
            cell_zones = []

        # Combine all available items
        all_items = surfaces + cell_zones

        # Display available items
        print(f"\nAVAILABLE INPUTS ({len(all_items)} total):\n")
        for i, item in enumerate(all_items, 1):
            # Check if item is selected by comparing names
            is_selected = any(s['name'] == item['name'] and s['category'] == item['category']
                            for s in selected_inputs)
            marker = "[X]" if is_selected else "[ ]"
            print(f"  {marker} [{i:2d}] {item['name']:30s} ({item['category']} - {item['type']})")

        print(f"\n{'='*70}")
        print("[Number] Toggle selection")
        print("[R] Refresh list")
        print("[C] Clear all selections")
        print("[D] Done")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'D':
            return selected_inputs
        elif choice == 'R':
            continue  # Refresh - loop will re-fetch data
        elif choice == 'C':
            selected_inputs.clear()
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(all_items):
                item = all_items[idx]
                # Find and remove if already selected (by name comparison)
                found_idx = None
                for i, s in enumerate(selected_inputs):
                    if s['name'] == item['name'] and s['category'] == item['category']:
                        found_idx = i
                        break

                if found_idx is not None:
                    selected_inputs.pop(found_idx)
                else:
                    selected_inputs.append(item)


def setup_model_outputs(solver, selected_outputs, ui_helpers):
    """Configure model outputs (surfaces, cell zones, report definitions)."""

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("CONFIGURE MODEL OUTPUTS")

        # Show selected items at top
        if selected_outputs:
            print("\n" + "="*70)
            print("SELECTED OUTPUTS:")
            print("="*70)
            for i, item in enumerate(selected_outputs, 1):
                print(f"  [{i:2d}] {item['name']:30s} (Type: {item['type']})")
            print("="*70)

        print("\nLoading surfaces, cell zones, and report definitions...")

        # Get surfaces
        try:
            boundary_conditions = solver.settings.setup.boundary_conditions
            surfaces = []

            for bc_type in dir(boundary_conditions):
                if bc_type.startswith('_') or bc_type in ['child_names', 'command_names']:
                    continue

                bc_obj = getattr(boundary_conditions, bc_type)
                if hasattr(bc_obj, '__iter__') and not isinstance(bc_obj, str):
                    try:
                        for name in bc_obj:
                            if name not in ['child_names', 'command_names']:
                                surfaces.append({
                                    'name': name,
                                    'type': bc_type.replace('_', ' ').title(),
                                    'category': 'Surface'
                                })
                    except Exception as e:
                        pass

            # Try to get ALL surfaces (including created surfaces like planes, iso-surfaces, etc.)
            try:
                if hasattr(solver, 'fields') and hasattr(solver.fields, 'field_data'):
                    # Get all accessible surface names using allowed_values()
                    all_surface_names = solver.fields.field_data.surfaces.allowed_values()

                    for surf_name in all_surface_names:
                        # Skip if already in list (avoid duplicates)
                        if not any(s['name'] == surf_name for s in surfaces):
                            surfaces.append({
                                'name': surf_name,
                                'type': 'Created Surface',
                                'category': 'Surface'
                            })
            except:
                pass

        except Exception as e:
            print(f"Warning: Error loading surfaces: {e}")
            surfaces = []

        # Get cell zones
        try:
            cell_zones_obj = solver.settings.setup.cell_zone_conditions
            cell_zones = []

            for zone_type in dir(cell_zones_obj):
                if zone_type.startswith('_') or zone_type in ['child_names', 'command_names']:
                    continue

                zone_obj = getattr(cell_zones_obj, zone_type)
                if hasattr(zone_obj, '__iter__') and not isinstance(zone_obj, str):
                    try:
                        for name in zone_obj:
                            if name not in ['child_names', 'command_names']:
                                cell_zones.append({
                                    'name': name,
                                    'type': zone_type.replace('_', ' ').title(),
                                    'category': 'Cell Zone'
                                })
                    except Exception as e:
                        pass
        except Exception as e:
            print(f"Warning: Error loading cell zones: {e}")
            cell_zones = []

        # Get report definitions
        try:
            report_defs_obj = solver.settings.solution.report_definitions
            report_defs = []

            report_types = ['surface', 'volume', 'flux', 'force', 'lift', 'drag',
                           'moment', 'expression', 'user_defined']

            for report_type in report_types:
                if hasattr(report_defs_obj, report_type):
                    report_obj = getattr(report_defs_obj, report_type)
                    if hasattr(report_obj, '__iter__') and not isinstance(report_obj, str):
                        try:
                            for name in report_obj:
                                if name not in ['child_names', 'command_names']:
                                    report_defs.append({
                                        'name': name,
                                        'type': report_type.replace('_', ' ').title(),
                                        'category': 'Report Definition'
                                    })
                        except Exception as e:
                            pass
        except Exception as e:
            print(f"Warning: Error loading report definitions: {e}")
            report_defs = []

        # Combine all available items
        all_items = surfaces + cell_zones + report_defs

        # Display available items
        print(f"\nAVAILABLE OUTPUTS ({len(all_items)} total):\n")
        for i, item in enumerate(all_items, 1):
            # Check if item is selected by comparing names
            is_selected = any(s['name'] == item['name'] and s['category'] == item['category']
                            for s in selected_outputs)
            marker = "[X]" if is_selected else "[ ]"
            print(f"  {marker} [{i:2d}] {item['name']:30s} ({item['category']} - {item['type']})")

        print(f"\n{'='*70}")
        print("[Number] Toggle selection")
        print("[R] Refresh list")
        print("[C] Clear all selections")
        print("[D] Done")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'D':
            return selected_outputs
        elif choice == 'R':
            continue  # Refresh - loop will re-fetch data
        elif choice == 'C':
            selected_outputs.clear()
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(all_items):
                item = all_items[idx]
                # Find and remove if already selected (by name comparison)
                found_idx = None
                for i, s in enumerate(selected_outputs):
                    if s['name'] == item['name'] and s['category'] == item['category']:
                        found_idx = i
                        break

                if found_idx is not None:
                    selected_outputs.pop(found_idx)
                else:
                    selected_outputs.append(item)


