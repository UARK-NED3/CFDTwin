"""Full mixing_elbow workflow with verbose output and result inspection.

Same pipeline as quickstart.py — extra prints between stages so you can see
what cfdtwin is doing at each step.
"""

from pathlib import Path

import cfdtwin
from ansys.fluent.core import examples

# --- 1. Get the case file -------------------------------------------------
case_file = examples.download_file("mixing_elbow.cas.h5", "pyfluent/mixing_elbow")
print(f"Case file: {case_file}\n")

# --- 2. Project setup -----------------------------------------------------
project_path = Path("elbow_study")
if (project_path / "project_info.json").exists():
    project = cfdtwin.Project.open(project_path)
    print(f"Reopened project: {project.name}")
else:
    project = cfdtwin.Project.create(project_path, name="elbow_v1")
    print(f"Created project: {project.name}")

project.set_case_file(case_file)

# Two velocity inlets, one surface output. set_inputs is declarative — no
# Fluent connection required at this stage.
project.set_inputs({
    "cold-inlet|momentum > velocity_magnitude": (0.2, 0.6),
    "hot-inlet|momentum > velocity_magnitude":  (0.4, 1.2),
})
project.set_outputs([
    {"name": "outlet", "category": "Surface",
     "field_variables": ["temperature"]},
])
print("Inputs and outputs declared.\n")

# --- 3. DOE ---------------------------------------------------------------
n_samples = project.generate_doe(n=20, method="lhs", seed=42)
print(f"DOE: {n_samples} LHS samples\n")

# --- 4. Run sims ----------------------------------------------------------
# mixing_elbow.cas.h5 is a single-precision case file.
project.connect_fluent(precision="single")
sim_result = project.run_simulations(iterations=100, verbose=True)
print()
print(sim_result.summary())
if sim_result.failed:
    print(f"Failed sim IDs: {sim_result.failed_ids}")
print()

# --- 5. Train -------------------------------------------------------------
train_result = project.train(model_name="elbow_v1", epochs=300)
print()
print(train_result.summary())
print()

best = train_result.best_model()
print(f"Best sub-model: {best.model_name}")
print(f"  test R²:    {best.test_metrics.r2:.4f}")
print(f"  test RMSE:  {best.test_metrics.rmse:.4f}")
print(f"  POD modes:  {best.n_modes}")
print(f"  variance:   {best.variance_explained:.4f}")
print()

# --- 6. Predict at a new design point -------------------------------------
pred = project.predict(train_result.model_name, {
    "cold-inlet|momentum > velocity_magnitude": 0.4,
    "hot-inlet|momentum > velocity_magnitude":  0.8,
})
print(f"Predict result:")
print(f"  values shape:      {pred.values.shape}")
print(f"  coordinates shape: {pred.coordinates.shape if pred.coordinates is not None else 'None'}")
print(f"  mean temperature:  {pred.values.mean():.2f} K")
print(f"  range:             [{pred.values.min():.2f}, {pred.values.max():.2f}] K")

# --- 7. Batch predict (a sweep across hot-inlet velocity) -----------------
sweep = [
    {"cold-inlet|momentum > velocity_magnitude": 0.3,
     "hot-inlet|momentum > velocity_magnitude":  v}
    for v in [0.5, 0.7, 0.9, 1.1]
]
sweep_pred = project.predict(train_result.model_name, sweep)
print(f"\nSweep result: {sweep_pred.values.shape}  (4 design points)")
for inp, vals in zip(sweep_pred.inputs, sweep_pred.values):
    v = inp["hot-inlet|momentum > velocity_magnitude"]
    print(f"  hot-inlet vmag {v:.1f} -> mean outlet T = {vals.mean():.2f} K")

project.disconnect_fluent()
