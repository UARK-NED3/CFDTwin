# Changelog

## GUI Update (In Progress)

### Bug Fixes

- **Fixed critical parameter ordering mismatch across modules.** The simulation runner and model trainer iterated DOE configuration keys in JSON insertion order, while the visualizer used sorted order. If boundary condition names in `model_setup.json` were not alphabetically ordered, input parameters would be mapped to the wrong BCs during prediction -- silently corrupting all NN predictions and Fluent validation comparisons. All modules now use `sorted()` for deterministic ordering independent of JSON key order.
  - `simulation_runner.py`: `doe_config.items()` / `doe_params.items()` -> `sorted(doe_config.keys())` / `sorted(doe_params.keys())`
  - `multi_model_trainer.py`: `doe_config.keys()` / `params.keys()` -> `sorted(doe_config.keys())` / `sorted(params.keys())`
  - `multi_model_visualizer.py`: Already used `sorted()` -- no change needed.

> **Note:** Existing models trained on datasets where BC names happened to be alphabetical are unaffected. Models trained on datasets with non-alphabetical BC names should be retrained.

- **Fixed shadowed `model_name` variable in `multi_model_trainer.py`.** The per-output model name inside the training loop overwrote the user's model set name, which was used for the output directory. Renamed inner variable to `output_model_name`.

### Code Cleanup (~300 lines removed)

- **Consolidated batch simulation functions** in `simulation_runner.py`. `run_batch_simulations` and `run_remaining_simulations` shared ~100 identical lines for the solve loop. Extracted shared logic into `_run_simulation_batch()`; both functions are now thin wrappers.
- **Extracted `suppress_fluent_output()` context manager** in `simulation_runner.py`. Replaced 6 repetitions of manual `sys.stdout`/`sys.stderr` save/redirect/restore with a single `with` block. Eliminates risk of unrestore on unexpected exceptions.
- **Deduplicated `extract_data` inner function** in `simulation_runner.py`. Two identical closures for parsing PyFluent surface data dicts replaced with module-level `extract_surface_data()`.
- **Moved `fluent_var_map` to module-level constant** `FLUENT_VAR_MAP` in `simulation_runner.py`. Was recreated on every call inside a loop.
- **Extracted `build_doe_param_info()` helper** in `multi_model_visualizer.py`. Five near-identical blocks that rebuilt parameter info + combinations from DOE config replaced with a single call.
- **Extracted `load_matching_coordinates()` helper** in `multi_model_visualizer.py`. Duplicate coordinate-file-scanning logic in `display_2d_plot` and `display_3d_plot` consolidated.
- **Extracted `align_fluent_to_coordinates()` helper** in `multi_model_visualizer.py`. Duplicate Fluent-to-dataset interpolation logic consolidated.
- **Extracted `_scatter_2d()` helper** in `multi_model_visualizer.py`. Four repetitions of the rotate-aware scatter plot pattern (NN, Fluent, Error, NN-only) consolidated.
- **Moved imports to module level.** Removed mid-function `import itertools`, `import re`, `from collections import Counter, defaultdict` in `multi_model_trainer.py`. Removed mid-function `from scipy.interpolate import griddata` in `multi_model_visualizer.py`.

### Coordinate Storage Redesign (Breaking Change)

Coordinates are now stored **once** in a separate reference file (`dataset/coordinates.npz`) instead of being duplicated in every simulation file. This eliminates gigabytes of redundant data for large datasets and guarantees consistent coordinate ordering across all samples.

**New data flow:**
- First simulation in a batch extracts coordinates and saves `dataset/coordinates.npz`
- All subsequent simulations validate their point counts against the reference
- Mesh inconsistency (point count mismatch) triggers a **hard error** that stops the batch -- the mismatched file is still saved to preserve compute time
- Simulation files (`sim_XXXX.npz`) now contain only field values, no coordinate keys
- Visualizer reads coordinates from `coordinates.npz` instead of scanning sim files
- Trainer no longer needs to skip coordinate keys (they don't exist in sim files)

**New functions in `simulation_runner.py`:**
- `extract_coordinates()` -- extracts coordinates from all configured output locations
- `save_reference_coordinates()` / `load_reference_coordinates()` -- persist/load the reference file
- `get_reference_point_counts()` -- returns expected point counts per location
- `validate_field_point_counts()` -- checks field data against reference, returns error on mismatch

**Affected modules:**
- `simulation_runner.py`: `extract_field_data()` no longer returns coordinate keys. Batch loop, single sim, and extract_current_solution all create/validate against reference.
- `multi_model_visualizer.py`: `load_matching_coordinates()` reads from `coordinates.npz`. Fluent validation extracts its own coordinates for interpolation alignment.
- `multi_model_trainer.py`: Removed dead coordinate-skip logic.

> **Breaking:** Old datasets with coordinates baked into sim files are not supported. No migration path -- start fresh datasets.

### Model Architecture Refactor (Breaking Change)

Replaced three near-identical model files (`scalar_nn_model.py`, `field_nn_model.py`, `volume_nn_model.py`) with a modular architecture:

**New files:**
- **`surrogate_nn.py`** -- Single `SurrogateNN` class with named presets (`1d`, `2d`, `3d`) and fully customizable hyperparameters. All config is serialized on save for exact reconstruction. Prepares for future GUI with preset/custom mode toggle.
- **`pod_reducer.py`** -- Standalone `PODReducer` class wrapping sklearn PCA. Handles fit/transform/inverse_transform and save/load. Auto-reduces n_modes if dataset is too small.
- **`metrics.py`** -- Standalone `compute_metrics()` function. Handles both single-output (per-sample MAE) and multi-output (per-sample R2) cases.

**Design:**
- NN and POD are independent -- the NN doesn't know whether its outputs are POD modes or direct values
- The trainer orchestrates: for 2D/3D, POD reduces data first, NN trains on modes; for 1D, NN trains directly
- Evaluation happens on final reconstructed output (after inverse POD if applicable)
- `load_and_predict()` helper in visualizer handles the NN + optional POD reconstruction

**Saved model file convention:**
```
model_dir/
  outlet_temperature_nn.h5        # Keras weights
  outlet_temperature_nn.npz       # NN scalers + config
  outlet_temperature_pod.npz      # POD components (only if 2D/3D)
  outlet_temperature_metadata.json
```

**Deleted files:** `scalar_nn_model.py`, `field_nn_model.py`, `volume_nn_model.py`

> **Breaking:** Old trained models are incompatible with the new save format. Retrain required.

### Backend Refactor for GUI Readiness (Breaking Change)

Stripped all terminal UI from backend modules. Every `input()` call (64 total), `print()` status message, and `ui_helpers` reference has been removed. All module functions now accept parameters and return results, reporting progress via Python `logging`.

**Deleted files:**
- `front_end.py` -- terminal menu system (orchestration paths documented in CHANGELOG for GUI reference)
- `modules/ui_helpers.py` -- terminal-specific helpers (clear_screen, pause, get_choice)
- `modules/scalar_nn_model.py`, `modules/field_nn_model.py`, `modules/volume_nn_model.py` -- replaced by surrogate_nn.py

**Refactored modules:**

- **`simulation_runner.py`**: Deleted `run_simulations_menu()` and `view_simulation_status()`. Remaining functions accept all params directly and return results. `_run_simulation_batch()` accepts `on_progress` callback for GUI progress bars. Returns summary dict.

- **`multi_model_trainer.py`**: Deleted `train_model_menu()`, `select_model_architecture()`, `view_existing_models()`. `train_all_models()` accepts `model_name`, `model_selection` as params. `load_training_data()` auto-selects most common shape on mismatch (no interactive prompt), accepts `exclude_range` param. Returns summary dict.

- **`multi_model_visualizer.py`**: Deleted all menu functions (~15 functions). Kept: `predict_from_params()`, `predict_dataset_point()`, `run_fluent_validation()`, `plot_2d_field()`, `plot_3d_field()`, `plot_scalar_comparison()`, `plot_2d_comparison()`, `plot_3d_comparison()`. Plot functions return `matplotlib.figure.Figure` objects instead of calling `plt.show()`.

- **`doe_setup.py`**: Deleted all interactive functions (~8 functions). Kept pure logic: `get_bc_parameters()`, `generate_lhs_samples()`, `generate_factorial_samples()`, `samples_to_doe_parameters()`, `doe_parameters_to_ranges()`, `doe_parameters_to_samples()`, `analyze_setup_dimensions()`, `create_dataset_structure()`.

- **`project_manager.py`**: Replaced `setup_model_inputs()`/`setup_model_outputs()` with `get_available_inputs(solver)` and `get_available_outputs(solver)` that return data for GUI to display.

- **`output_parameters.py`**: Removed interactive `setup_output_parameters()`. Kept `get_available_field_variables()` (pure data).

- **`project_system.py`**: Removed interactive `create_new_project()`/`open_existing_project()`/`open_recent_project()`. Added `create_project(path, name)` and `open_project(path)` that accept params directly.

- **`fluent_interface.py`**: Consolidated `open_case_file()` and `open_recent_project()` into single `launch_fluent(case_file_path, solver_settings, log_dir)`. No tkinter file dialogs (GUI handles file selection).

**Logging:** All modules use `logging.getLogger(__name__)`. GUI will attach handlers to route to status panels.

**GUI orchestration paths** (from deleted front_end.py, for reference):
1. Project lifecycle: create/open/recent -> WorkflowProject
2. Fluent session: launch_fluent() -> solver (lives for session)
3. Dataset creation: get_available_inputs/outputs -> configure DOE -> generate samples -> save JSONs
4. Simulation: generate_doe_combinations -> run_batch/remaining/single -> save NPZ
5. Training: load_training_data -> train_all_models -> save models + summary
6. Visualization: predict_from_params/predict_dataset_point -> plot_2d/3d_field
7. Data management: project.delete_case()
