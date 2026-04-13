"""
Project Manager Module
======================
Queries Fluent for available inputs/outputs. No UI -- returns data for GUI to display.
"""

import logging

logger = logging.getLogger(__name__)


def get_available_inputs(solver):
    """
    Query Fluent for available boundary conditions and cell zones that can be used as inputs.

    Returns
    -------
    list of dict
        Each dict has keys: name, type, category
    """
    items = []

    # Boundary conditions
    try:
        boundary_conditions = solver.settings.setup.boundary_conditions
        for bc_type in dir(boundary_conditions):
            if bc_type.startswith('_') or bc_type in ['child_names', 'command_names']:
                continue
            bc_obj = getattr(boundary_conditions, bc_type)
            if hasattr(bc_obj, '__iter__') and not isinstance(bc_obj, str):
                try:
                    for name in bc_obj:
                        if name not in ['child_names', 'command_names']:
                            items.append({
                                'name': name,
                                'type': bc_type.replace('_', ' ').title(),
                                'category': 'Boundary Condition'
                            })
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Error loading boundary conditions: {e}")

    # Cell zones
    try:
        cell_zones_obj = solver.settings.setup.cell_zone_conditions
        for zone_type in dir(cell_zones_obj):
            if zone_type.startswith('_') or zone_type in ['child_names', 'command_names']:
                continue
            zone_obj = getattr(cell_zones_obj, zone_type)
            if hasattr(zone_obj, '__iter__') and not isinstance(zone_obj, str):
                try:
                    for name in zone_obj:
                        if name not in ['child_names', 'command_names']:
                            items.append({
                                'name': name,
                                'type': zone_type.replace('_', ' ').title(),
                                'category': 'Cell Zone'
                            })
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Error loading cell zones: {e}")

    return items


def get_available_outputs(solver):
    """
    Query Fluent for available surfaces, cell zones, and report definitions that can be outputs.

    Returns
    -------
    list of dict
        Each dict has keys: name, type, category
    """
    items = []

    # Surfaces (from boundary conditions)
    try:
        boundary_conditions = solver.settings.setup.boundary_conditions
        for bc_type in dir(boundary_conditions):
            if bc_type.startswith('_') or bc_type in ['child_names', 'command_names']:
                continue
            bc_obj = getattr(boundary_conditions, bc_type)
            if hasattr(bc_obj, '__iter__') and not isinstance(bc_obj, str):
                try:
                    for name in bc_obj:
                        if name not in ['child_names', 'command_names']:
                            items.append({
                                'name': name,
                                'type': bc_type.replace('_', ' ').title(),
                                'category': 'Surface'
                            })
                except Exception:
                    pass

        # Created surfaces (planes, iso-surfaces, etc.)
        try:
            if hasattr(solver, 'fields') and hasattr(solver.fields, 'field_data'):
                all_surface_names = solver.fields.field_data.surfaces.allowed_values()
                for surf_name in all_surface_names:
                    if not any(s['name'] == surf_name for s in items):
                        items.append({
                            'name': surf_name,
                            'type': 'Created Surface',
                            'category': 'Surface'
                        })
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Error loading surfaces: {e}")

    # Cell zones
    try:
        cell_zones_obj = solver.settings.setup.cell_zone_conditions
        for zone_type in dir(cell_zones_obj):
            if zone_type.startswith('_') or zone_type in ['child_names', 'command_names']:
                continue
            zone_obj = getattr(cell_zones_obj, zone_type)
            if hasattr(zone_obj, '__iter__') and not isinstance(zone_obj, str):
                try:
                    for name in zone_obj:
                        if name not in ['child_names', 'command_names']:
                            items.append({
                                'name': name,
                                'type': zone_type.replace('_', ' ').title(),
                                'category': 'Cell Zone'
                            })
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Error loading cell zones: {e}")

    # Report definitions
    try:
        report_defs_obj = solver.settings.solution.report_definitions
        report_types = ['surface', 'volume', 'flux', 'force', 'lift', 'drag',
                        'moment', 'expression', 'user_defined']
        for report_type in report_types:
            if hasattr(report_defs_obj, report_type):
                report_obj = getattr(report_defs_obj, report_type)
                if hasattr(report_obj, '__iter__') and not isinstance(report_obj, str):
                    try:
                        for name in report_obj:
                            if name not in ['child_names', 'command_names']:
                                items.append({
                                    'name': name,
                                    'type': report_type.replace('_', ' ').title(),
                                    'category': 'Report Definition'
                                })
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"Error loading report definitions: {e}")

    return items
