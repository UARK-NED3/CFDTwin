"""Connect to Fluent and list everything cfdtwin can address as inputs/outputs.

Run this once for an unfamiliar case file — copy the bc names, parameter
paths, surface names, etc. into your set_inputs / set_outputs calls.

The last section shows the higher-level ``Project.list_available_inputs()``
helper, which gives you the same data already grouped by category and parsed
into a shape you can feed straight into ``set_inputs``.
"""

import tempfile

import ansys.fluent.core as pyfluent
from ansys.fluent.core import examples

import cfdtwin

CASE_FILE = examples.download_file("mixing_elbow.cas.h5", "pyfluent/mixing_elbow")

print(f"Loading {CASE_FILE}...\n")
solver = pyfluent.launch_fluent(
    precision="single", processor_count=4, dimension=3, mode="solver",
)
solver.settings.file.read_case(file_name=CASE_FILE)

# --- Boundary conditions --------------------------------------------------
print("=== Boundary conditions ===")
bc_settings = solver.settings.setup.boundary_conditions
for bc_type in bc_settings.get_state():
    bc_group = getattr(bc_settings, bc_type)
    for bc_name in bc_group.get_state():
        print(f"  {bc_name:<20}  type={bc_type}")

# --- Velocity-inlet parameter paths ---------------------------------------
print("\n=== Velocity-inlet parameters ===")
vel_inlets = solver.settings.setup.boundary_conditions.velocity_inlet
for bc_name in vel_inlets.get_state():
    inlet = vel_inlets[bc_name]
    state = inlet.get_state()
    print(f"  {bc_name}:")
    for key, val in state.items():
        print(f"    {key:<20}  -> parameter_path candidates")

# --- Surfaces -------------------------------------------------------------
print("\n=== Surfaces ===")
try:
    surfaces = solver.settings.results.surfaces.get_state()
    for s in surfaces:
        print(f"  {s}")
except Exception as e:
    print(f"  (unable to list: {e})")

# --- Report definitions ---------------------------------------------------
print("\n=== Report definitions ===")
try:
    reports = solver.settings.solution.report_definitions.get_state()
    if reports:
        for r in reports:
            print(f"  {r}")
    else:
        print("  (none defined in this case)")
except Exception as e:
    print(f"  (unable to list: {e})")

solver.exit()

# --- Same data via the cfdtwin API ----------------------------------------
# Project.list_available_inputs() wraps the raw PyFluent calls above. It
# returns one list with BCs and Fluent input parameters (named expressions
# flagged input_parameter=True) already merged, each entry tagged with a
# `category`. Feed entries straight back into set_inputs by attaching a range.
print("\n=== cfdtwin.Project.list_available_inputs() ===")
with tempfile.TemporaryDirectory() as tmp:
    project = cfdtwin.Project.create(tmp, name="discovery_demo")
    project.set_case_file(CASE_FILE)
    project.connect_fluent()
    try:
        for item in project.list_available_inputs():
            cat = item['category']
            if cat == 'Input Parameter':
                print(f"  [IP] {item['name']:<20} unit={item.get('unit','')} "
                      f"current={item.get('current_value')}")
            else:
                print(f"  [BC] {item['name']:<20} type={item['type']}")
    finally:
        project.disconnect_fluent()
