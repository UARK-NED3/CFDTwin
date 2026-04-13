"""
Metrics Module
===============
Standalone evaluation metrics for surrogate model predictions.
"""

import warnings
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics between true and predicted values.

    Handles both single-output (scalar) and multi-output (field) data.
    For single-output, per-sample R2 is undefined so per-sample MAE is used instead.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    dict
        Metrics dictionary with keys: r2, rmse, mae, and per-sample statistics
    """
    # Ensure 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())

    metrics = {'r2': r2, 'rmse': rmse, 'mae': mae}

    n_outputs = y_true.shape[1]

    if n_outputs > 1:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            r2_per_sample = [r2_score(y_true[i], y_pred[i]) for i in range(len(y_true))]
        metrics.update({
            'r2_per_sample': r2_per_sample,
            'r2_mean': np.mean(r2_per_sample),
            'r2_std': np.std(r2_per_sample),
            'r2_min': np.min(r2_per_sample),
            'r2_max': np.max(r2_per_sample),
        })
    else:
        abs_errors = np.abs(y_pred.flatten() - y_true.flatten())
        metrics.update({
            'mae_per_sample': abs_errors.tolist(),
            'mae_std': np.std(abs_errors),
            'mae_min': np.min(abs_errors),
            'mae_max': np.max(abs_errors),
        })

    return metrics
