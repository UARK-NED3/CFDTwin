# Fluent PODNN Surrogate Builder

Wizard-based desktop app for building neural-network surrogate models from ANSYS Fluent simulations. Load a `.cas` file, sample the parameter space, run the batch sim, train a POD+NN model, and validate — all in one GUI.

## Install

Requires Python 3.10+ and a working ANSYS Fluent installation.

```
pip install -r program_files/requirements.txt
pip install PySide6 scipy pytest pytest-qt
```

## Run

From the repo root:

```
python -m program_files.gui
```

On launch, select or create a project. The sidebar steps unlock as prerequisites are met:

1. **Setup** — pick `.cas` file, set Fluent options, define inputs and outputs
2. **DOE** — generate LHS/factorial samples
3. **Simulate** — batch-run Fluent with live progress
4. **Train** — transfer-list filter, live loss curves, per-output NN
5. **Validate** — metrics dashboard, predictions, Fluent comparison with caching
