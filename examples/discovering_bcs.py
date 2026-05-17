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
        print(f"  {bc_name:<24}  type={bc_type}")


def walk_settable(obj, prefix="", depth=4):
    """Recursively find leaf settings that have a writable .value, yielding
    (dotted-path, type-name) pairs. Used to show what set_inputs's rich-dict
    `parameter_path` can target on a given BC."""
    if depth <= 0:
        return
    if not hasattr(obj, "child_names"):
        return
    for name in obj.child_names:
        if name in ("child_names", "command_names"):
            continue
        try:
            child = getattr(obj, name)
        except Exception:
            continue
        path = f"{prefix}.{name}" if prefix else name
        if hasattr(child, "value"):
            yield path, type(child).__name__
        else:
            yield from walk_settable(child, prefix=path, depth=depth - 1)


# --- Settable parameter paths on each velocity-inlet ---------------------
print("\n=== Velocity-inlet settable parameters ===")
vel_inlets = solver.settings.setup.boundary_conditions.velocity_inlet
for bc_name in vel_inlets.get_state():
    print(f"  {bc_name}:")
    inlet = vel_inlets[bc_name]
    for path, type_name in walk_settable(inlet):
        print(f"    parameter_path={path!r:<48} ({type_name})")

# --- User-defined surfaces -----------------------------------------------
# results.surfaces is grouped by surface type (point_surface, zone_surface,
# plane_surface, ...) — iterate one level deeper to get the actual names.
print("\n=== Surfaces ===")
try:
    surfaces = solver.settings.results.surfaces.get_state() or {}
    any_found = False
    for surf_type, items in surfaces.items():
        if not items:
            continue
        for surf_name in items:
            print(f"  {surf_name:<24}  type={surf_type}")
            any_found = True
    if not any_found:
        print("  (no user-defined surfaces; BC names like 'outlet' still work as outputs)")
except Exception as e:
    print(f"  (unable to list: {e})")

# --- User-defined report definitions -------------------------------------
# Same shape as surfaces: top-level keys are report types, each holding a
# dict of user-created reports.
print("\n=== Report definitions ===")
try:
    reports = solver.settings.solution.report_definitions.get_state() or {}
    any_found = False
    for report_type, items in reports.items():
        if not items:
            continue
        for report_name in items:
            print(f"  {report_name:<24}  type={report_type}")
            any_found = True
    if not any_found:
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
    project.connect_fluent(precision="single")
    try:
        for item in project.list_available_inputs():
            cat = item['category']
            if cat == 'Input Parameter':
                print(f"  [IP] {item['name']:<24} unit={item.get('unit','')} "
                      f"current={item.get('current_value')}")
            else:
                print(f"  [BC] {item['name']:<24} type={item['type']}")
    finally:
        project.disconnect_fluent()
