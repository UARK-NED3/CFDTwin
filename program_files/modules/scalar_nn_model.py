"""
Scalar (1D) Neural Network Model
==================================
Simple feedforward neural network for scalar outputs (point data).
Used for outputs like single temperature values, pressure readings, etc.

Architecture: Input parameters -> Dense layers -> Scalar outputs
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras


class ScalarNNModel:
    """
    Neural network model for 1D scalar outputs.

    Simple feedforward architecture for predicting scalar values
    from input parameters.
    """

    def __init__(self, field_name='scalar'):
        """
        Initialize scalar model.

        Parameters
        ----------
        field_name : str
            Name of the scalar field being predicted
        """
        self.field_name = field_name

        # Components
        self.param_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.model = None

        # Metrics storage
        self.train_metrics = {}
        self.test_metrics = {}
        self.history = None

    def build_model(self, input_dim, output_dim):
        """
        Build neural network architecture for scalar prediction.

        Parameters
        ----------
        input_dim : int
            Number of input parameters
        output_dim : int
            Number of scalar outputs
        """
        # Enhanced architecture: Deeper network with 5 hidden layers
        # Wider early layers (128) for better feature extraction
        # Reduced regularization (L2=0.0001) and dropout (0.05) for better fitting
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
            keras.layers.Dropout(0.05),
            keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
            keras.layers.Dropout(0.05),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
            keras.layers.Dropout(0.05),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
            keras.layers.Dropout(0.05),
            keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
            keras.layers.Dense(output_dim)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(self, parameters, outputs, validation_split=0.2, epochs=500, verbose=1):
        """
        Train the scalar model.

        Parameters
        ----------
        parameters : np.ndarray
            Input parameters, shape (n_samples, n_params)
        outputs : np.ndarray
            Output scalars, shape (n_samples, n_outputs)
        validation_split : float
            Fraction of data for validation
        epochs : int
            Number of training epochs
        verbose : int
            Verbosity level
        """
        print(f"\n{'='*70}")
        print(f"Training {self.field_name} scalar model")
        print(f"{'='*70}")

        # Ensure outputs are 2D
        if outputs.ndim == 1:
            outputs = outputs.reshape(-1, 1)

        # Scale data
        print(f"\n[1/3] Scaling data...")
        params_scaled = self.param_scaler.fit_transform(parameters)
        outputs_scaled = self.output_scaler.fit_transform(outputs)

        # Build model
        print(f"\n[2/3] Building neural network...")
        self.model = self.build_model(input_dim=parameters.shape[1], output_dim=outputs.shape[1])
        print(self.model.summary())

        print(f"\n[3/3] Training (epochs={epochs})...")

        # Adaptive batch size for more stable gradient updates
        # Larger batches reduce noise, improve convergence
        batch_size = max(16, len(parameters) // 50)
        print(f"  Using batch size: {batch_size}")

        # Callbacks - relaxed for better convergence
        # Monitor 'loss' when no validation split, otherwise 'val_loss'
        monitor_metric = 'loss' if validation_split == 0.0 else 'val_loss'

        early_stop = keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=50,  # Increased from 30 for better convergence
            restore_best_weights=True,
            min_delta=1e-7  # More sensitive threshold
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=15,  # Increased for better convergence
            min_lr=1e-7
        )

        # Train
        history = self.model.fit(
            params_scaled, outputs_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,  # Adaptive batch size
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        print(f"\n✓ Training complete!")
        if 'val_loss' in history.history:
            print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
        print(f"  Stopped at epoch: {len(history.history['loss'])}")

        # Store history
        self.history = history

        return history

    def predict(self, parameters):
        """
        Predict scalar outputs for given parameters.

        Parameters
        ----------
        parameters : np.ndarray
            Input parameters, shape (n_samples, n_params)

        Returns
        -------
        outputs : np.ndarray
            Predicted outputs, shape (n_samples, n_outputs)
        """
        # Scale parameters
        params_scaled = self.param_scaler.transform(parameters)

        # Predict
        outputs_scaled = self.model.predict(params_scaled, verbose=0)
        outputs = self.output_scaler.inverse_transform(outputs_scaled)

        return outputs

    def evaluate(self, parameters, true_outputs, dataset_name='test'):
        """
        Evaluate model performance.

        Parameters
        ----------
        parameters : np.ndarray
            Input parameters
        true_outputs : np.ndarray
            Ground truth outputs
        dataset_name : str
            Name for reporting (train/test)

        Returns
        -------
        metrics : dict
            Performance metrics
        """
        import warnings

        # Ensure outputs are 2D
        if true_outputs.ndim == 1:
            true_outputs = true_outputs.reshape(-1, 1)

        pred_outputs = self.predict(parameters)

        # Overall metrics
        r2 = r2_score(true_outputs.flatten(), pred_outputs.flatten())
        rmse = np.sqrt(mean_squared_error(true_outputs.flatten(), pred_outputs.flatten()))
        mae = mean_absolute_error(true_outputs.flatten(), pred_outputs.flatten())

        # Per-sample metrics (only if output dim > 1, otherwise R² is undefined)
        n_outputs = true_outputs.shape[1]

        if n_outputs > 1:
            # Suppress R² warnings for near-perfect predictions
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                r2_per_sample = [
                    r2_score(true_outputs[i], pred_outputs[i])
                    for i in range(len(true_outputs))
                ]
        else:
            # For single output, per-sample R² is not meaningful (would need >=2 outputs)
            # Use absolute error as per-sample metric instead
            r2_per_sample = None

        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
        }

        # Add per-sample metrics if available
        if r2_per_sample is not None:
            metrics.update({
                'r2_per_sample': r2_per_sample,
                'r2_mean': np.mean(r2_per_sample),
                'r2_std': np.std(r2_per_sample),
                'r2_min': np.min(r2_per_sample),
                'r2_max': np.max(r2_per_sample)
            })
        else:
            # For single output, provide alternative per-sample metrics
            abs_errors = np.abs(pred_outputs.flatten() - true_outputs.flatten())
            metrics.update({
                'mae_per_sample': abs_errors.tolist(),
                'mae_std': np.std(abs_errors),
                'mae_min': np.min(abs_errors),
                'mae_max': np.max(abs_errors)
            })

        # Store metrics
        if dataset_name == 'train':
            self.train_metrics = metrics
        elif dataset_name == 'test':
            self.test_metrics = metrics

        return metrics

    def save(self, filepath):
        """
        Save model to files.

        Parameters
        ----------
        filepath : Path or str
            Base filepath (without extension)
        """
        from pathlib import Path
        filepath = Path(filepath)

        # Save scalers and metadata
        save_dict = {
            'field_name': self.field_name,
            'param_scaler_mean': self.param_scaler.mean_,
            'param_scaler_scale': self.param_scaler.scale_,
            'output_scaler_mean': self.output_scaler.mean_,
            'output_scaler_scale': self.output_scaler.scale_,
        }
        np.savez_compressed(filepath.with_suffix('.npz'), **save_dict)

        # Save Keras model
        self.model.save(filepath.with_suffix('.h5'))

        print(f"Model saved: {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load model from files.

        Parameters
        ----------
        filepath : Path or str
            Base filepath (without extension)

        Returns
        -------
        model : ScalarNNModel
            Loaded model
        """
        from pathlib import Path
        filepath = Path(filepath)

        # Load metadata
        data = np.load(filepath.with_suffix('.npz'), allow_pickle=True)

        # Create model instance
        model = cls(field_name=str(data['field_name']))

        # Load scalers
        model.param_scaler.mean_ = data['param_scaler_mean']
        model.param_scaler.scale_ = data['param_scaler_scale']
        model.output_scaler.mean_ = data['output_scaler_mean']
        model.output_scaler.scale_ = data['output_scaler_scale']

        # Load Keras model (compile=False to avoid metric deserialization issues)
        model.model = keras.models.load_model(filepath.with_suffix('.h5'), compile=False)

        # Recompile model with current Keras version
        model.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model
