"""Smallest end-to-end cfdtwin pipeline.

Downloads PyFluent's mixing_elbow case, builds a 20-sample DOE, runs the
sims, trains a surrogate, and predicts at one new design point.
"""

from pathlib import Path

import cfdtwin
from ansys.fluent.core import examples

# Get the case file (cached after first download)
case_file = examples.download_file("mixing_elbow.cas.h5", "pyfluent/mixing_elbow")

# Project lives in ./quickstart_study/ — re-run won't clobber if it exists
project_path = Path("quickstart_study")
if (project_path / "project_info.json").exists():
    project = cfdtwin.Project.open(project_path)
else:
    project = cfdtwin.Project.create(project_path, name="quickstart")

project.set_case_file(case_file)

project.set_inputs({
    "cold-inlet|momentum > velocity_magnitude": (0.2, 0.6),
    "hot-inlet|momentum > velocity_magnitude":  (0.4, 1.2),
})
project.set_outputs([
    {"name": "outlet", "category": "Surface",
     "field_variables": ["temperature"]},
])

project.generate_doe(n=20, method="lhs", seed=42)

# mixing_elbow.cas.h5 is a single-precision case file.
project.connect_fluent(precision="single")
project.run_simulations(verbose=True)   # prints "sim X/N: status" lines
result = project.train(model_name="run1")

print()
print(result.summary())

pred = project.predict("run1", {
    "cold-inlet|momentum > velocity_magnitude": 0.4,
    "hot-inlet|momentum > velocity_magnitude":  0.8,
})
print(f"\nPredicted field shape: {pred.values.shape}")
print(f"Mean predicted temperature: {pred.values.mean():.2f} K")
