"""Per-output training configuration: PCA modes, NN overrides.

Assumes you've already run quickstart.py or full_workflow.py to populate
the elbow_study project with simulation data. This script trains three
different model variants on the same data with different configurations
and compares them.
"""

from pathlib import Path

import cfdtwin

project_path = Path("elbow_study")
if not (project_path / "project_info.json").exists():
    raise SystemExit(
        "Run docs/examples/full_workflow.py first to populate elbow_study with sims."
    )
project = cfdtwin.Project.open(project_path)


# Variant 1 — defaults (let cfdtwin pick POD modes by data shape)
print("\n=== Variant 1: defaults ===")
r1 = project.train(model_name="tune_defaults")
print(r1.summary())


# Variant 2 — fixed POD mode count
print("\n=== Variant 2: fixed 15 POD modes ===")
r2 = project.train(
    model_name="tune_modes15",
    outputs={
        # The model_key is "<location>_<field>"
        "outlet_temperature": {"pod": {"modes": 15}},
    },
)
print(r2.summary())


# Variant 3 — variance-driven POD + custom NN learning rate
print("\n=== Variant 3: variance=0.99 POD + LR=5e-4 ===")
r3 = project.train(
    model_name="tune_var99",
    outputs={
        "outlet_temperature": {
            "pod": {"variance": 0.99},
            "nn":  {"learning_rate": 5e-4, "hidden_layers": [128, 64, 32]},
        },
    },
    epochs=500,
)
print(r3.summary())


# --- Compare ---
print("\n=== Comparison ===")
print(f"{'variant':<20} {'modes':<6} {'test R²':<10} {'test RMSE':<12}")
print("-" * 50)
for r in [r1, r2, r3]:
    for m in r.models:
        modes = m.n_modes if m.n_modes is not None else "-"
        print(f"{r.model_name:<20} {str(modes):<6} {m.test_metrics.r2:<10.4f} {m.test_metrics.rmse:<12.4g}")
