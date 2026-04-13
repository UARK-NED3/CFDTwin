"""
Output Parameters Module
=========================
Provides available field variables for Fluent outputs.
No UI -- returns data for GUI to display.
"""


def get_available_field_variables(output_category='Surface'):
    """
    Get list of common field variables available in Fluent.

    Parameters
    ----------
    output_category : str
        'Surface' for 2D surface data, 'Cell Zone' for 3D volume data

    Returns
    -------
    dict
        Dictionary of variable categories and their variables
    """
    if output_category == 'Cell Zone':
        return {
            'Pressure': ['pressure'],
            'Velocity': ['x-velocity', 'y-velocity', 'z-velocity'],
            'Temperature': ['temperature'],
            'Density': ['density'],
            'Turbulence': ['k', 'omega', 'turbulent-viscosity'],
            'Other': ['enthalpy', 'wall-distance']
        }
    else:
        return {
            'Pressure': ['absolute-pressure', 'pressure-coefficient', 'dynamic-pressure', 'total-pressure'],
            'Velocity': ['velocity-magnitude', 'x-velocity', 'y-velocity', 'z-velocity', 'radial-velocity', 'axial-velocity'],
            'Temperature': ['temperature', 'total-temperature'],
            'Density': ['density'],
            'Turbulence': ['k', 'epsilon', 'omega', 'turb-kinetic-energy', 'turb-diss-rate', 'turbulent-viscosity'],
            'Wall': ['wall-shear', 'y-plus', 'wall-temperature', 'heat-transfer-coef'],
            'Species': ['mass-fraction', 'mole-fraction'],
            'Vorticity': ['vorticity-magnitude', 'x-vorticity', 'y-vorticity', 'z-vorticity']
        }
