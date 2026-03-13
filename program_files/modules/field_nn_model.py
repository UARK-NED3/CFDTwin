"""
Field (2D) Neural Network Model
=================================
POD-based neural network for 2D field distributions (cut planes).
Uses Proper Orthogonal Decomposition to reduce dimensionality before prediction.

Architecture: Input parameters -> NN -> POD modes -> Reconstruct 2D field
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras


class FieldNNModel:
    """
    POD-based neural network model for 2D field outputs.

    Uses dimensionality reduction via POD before neural network prediction.
    """

    def __init__(self, n_modes=10, field_name='field'):
        """
        Initialize field model.

        Parameters
        ----------
        n_modes : int
            Number of POD modes to retain
        field_name : str
            Name of the field being predicted
        """
        self.n_modes = n_modes
        self.field_name = field_name

        # Components
        self.pca = PCA(n_components=n_modes)
        self.param_scaler = StandardScaler()
        self.mode_scaler = StandardScaler()
        self.model = None

        # Metrics storage
        self.train_metrics = {}
        self.test_metrics = {}
        self.variance_explained = None
        self.history = None

    def build_model(self, input_dim):
        """
        Build neural network architecture for POD mode prediction.

        Parameters
        ----------
        input_dim : int
            Number of input parameters
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dense(self.n_modes)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(self, parameters, fields, validation_split=0.2, epochs=500, verbose=1):
        """
        Train the field model.

        Parameters
        ----------
        parameters : np.ndarray
            Input parameters, shape (n_samples, n_params)
        fields : np.ndarray
            Field data, shape (n_samples, n_points)
        validation_split : float
            Fraction of data for validation
        epochs : int
            Number of training epochs
        verbose : int
            Verbosity level
        """
        print(f"\n{'='*70}")
        print(f"Training {self.field_name} field model (2D)")
        print(f"{'='*70}")

        # Adjust n_modes if dataset is too small
        n_train = int(len(parameters) * (1 - validation_split))
        n_features = fields.shape[1]
        max_modes = min(n_train, n_features)

        if self.n_modes > max_modes:
            print(f"\n⚠️  WARNING: Requested {self.n_modes} POD modes but only {n_train} training samples available.")
            print(f"   Automatically reducing to {max_modes} modes (max possible with this dataset).")
            self.n_modes = max_modes
            self.pca.n_components = max_modes

        # Step 1: Apply POD
        print(f"\n[1/4] Applying POD (reducing {fields.shape[1]} → {self.n_modes} modes)...")
        modes = self.pca.fit_transform(fields)
        self.variance_explained = self.pca.explained_variance_ratio_

        print(f"  Variance explained by {self.n_modes} modes: {self.variance_explained.sum()*100:.2f}%")

        # Step 2: Scale data
        print(f"\n[2/4] Scaling parameters and modes...")
        params_scaled = self.param_scaler.fit_transform(parameters)
        modes_scaled = self.mode_scaler.fit_transform(modes)

        # Step 3: Build neural network
        print(f"\n[3/4] Building neural network...")
        self.model = self.build_model(input_dim=parameters.shape[1])
        print(self.model.summary())

        print(f"\n[4/4] Training (epochs={epochs})...")

        # Callbacks - relaxed for better convergence
        # For small datasets, be very patient
        # Monitor 'loss' when no validation split, otherwise 'val_loss'
        monitor_metric = 'loss' if validation_split == 0.0 else 'val_loss'

        early_stop = keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=80,  # Very patient for small datasets
            restore_best_weights=True,
            min_delta=1e-7,  # More sensitive threshold
            start_from_epoch=30  # Don't start checking until epoch 30
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=15,  # Increased for better convergence
            min_lr=1e-7
        )

        # Train
        history = self.model.fit(
            params_scaled, modes_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=8,  # Fixed batch size for stable training
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
        Predict field for given parameters.

        Parameters
        ----------
        parameters : np.ndarray
            Input parameters, shape (n_samples, n_params)

        Returns
        -------
        fields : np.ndarray
            Predicted fields, shape (n_samples, n_points)
        """
        # Scale parameters
        params_scaled = self.param_scaler.transform(parameters)

        # Predict modes
        modes_scaled = self.model.predict(params_scaled, verbose=0)
        modes = self.mode_scaler.inverse_transform(modes_scaled)

        # Reconstruct field from modes
        fields = self.pca.inverse_transform(modes)

        return fields

    def evaluate(self, parameters, true_fields, dataset_name='test'):
        """
        Evaluate model performance.

        Parameters
        ----------
        parameters : np.ndarray
            Input parameters
        true_fields : np.ndarray
            Ground truth fields
        dataset_name : str
            Name for reporting (train/test)

        Returns
        -------
        metrics : dict
            Performance metrics
        """
        pred_fields = self.predict(parameters)

        # Overall metrics
        r2 = r2_score(true_fields.flatten(), pred_fields.flatten())
        rmse = np.sqrt(mean_squared_error(true_fields.flatten(), pred_fields.flatten()))
        mae = mean_absolute_error(true_fields.flatten(), pred_fields.flatten())

        # Per-sample metrics
        r2_per_sample = [
            r2_score(true_fields[i], pred_fields[i])
            for i in range(len(true_fields))
        ]

        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'r2_per_sample': r2_per_sample,
            'r2_mean': np.mean(r2_per_sample),
            'r2_std': np.std(r2_per_sample),
            'r2_min': np.min(r2_per_sample),
            'r2_max': np.max(r2_per_sample)
        }

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

        # Save PCA, scalers and metadata
        save_dict = {
            'n_modes': self.n_modes,
            'field_name': self.field_name,
            'pca_components': self.pca.components_,
            'pca_mean': self.pca.mean_,
            'pca_variance': self.variance_explained,
            'param_scaler_mean': self.param_scaler.mean_,
            'param_scaler_scale': self.param_scaler.scale_,
            'mode_scaler_mean': self.mode_scaler.mean_,
            'mode_scaler_scale': self.mode_scaler.scale_,
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
        model : FieldNNModel
            Loaded model
        """
        from pathlib import Path
        filepath = Path(filepath)

        # Load metadata
        data = np.load(filepath.with_suffix('.npz'), allow_pickle=True)

        # Create model instance
        n_modes = int(data['n_modes'])
        model = cls(n_modes=n_modes, field_name=str(data['field_name']))

        # Load PCA
        model.pca.components_ = data['pca_components']
        model.pca.mean_ = data['pca_mean']
        model.variance_explained = data['pca_variance']

        # Load scalers
        model.param_scaler.mean_ = data['param_scaler_mean']
        model.param_scaler.scale_ = data['param_scaler_scale']
        model.mode_scaler.mean_ = data['mode_scaler_mean']
        model.mode_scaler.scale_ = data['mode_scaler_scale']

        # Load Keras model (compile=False to avoid metric deserialization issues)
        model.model = keras.models.load_model(filepath.with_suffix('.h5'), compile=False)

        # Recompile model with current Keras version
        model.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model
