"""
Surrogate Neural Network Module
=================================
Generic feedforward neural network for surrogate modeling.
Supports preset configurations (1D, 2D, 3D) and fully custom architectures.
"""

import logging
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
import os

logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*tf.function retracing.*')

import tensorflow as tf
from tensorflow import keras


# Preset configurations matching original model architectures
PRESETS = {
    '1d': {
        'hidden_layers': [128, 128, 64, 32, 16],
        'l2': 0.0001,
        'dropout': [0.05, 0.05, 0.05, 0.05, 0],
        'batch_size': 'adaptive',
        'es_patience': 50,
        'es_start_epoch': 0,
        'es_min_delta': 1e-7,
        'lr_patience': 15,
        'lr_factor': 0.5,
        'lr_min': 1e-7,
        'learning_rate': 0.001,
    },
    '2d': {
        'hidden_layers': [64, 64, 32],
        'l2': 0.001,
        'dropout': [0.1, 0.1, 0],
        'batch_size': 8,
        'es_patience': 80,
        'es_start_epoch': 30,
        'es_min_delta': 1e-7,
        'lr_patience': 15,
        'lr_factor': 0.5,
        'lr_min': 1e-7,
        'learning_rate': 0.001,
    },
    '3d': {
        'hidden_layers': [128, 128, 64, 64],
        'l2': 0.001,
        'dropout': [0.15, 0.15, 0.1, 0],
        'batch_size': 8,
        'es_patience': 100,
        'es_start_epoch': 50,
        'es_min_delta': 1e-7,
        'lr_patience': 20,
        'lr_factor': 0.5,
        'lr_min': 1e-7,
        'learning_rate': 0.001,
    },
}


class SurrogateNN:
    """
    Generic feedforward neural network for surrogate modeling.

    Can be initialized from a named preset or with a fully custom config.
    All config values are serialized on save so the exact architecture can be reconstructed.

    Parameters
    ----------
    config : dict, optional
        Full configuration dict. If preset is also given, config values override preset defaults.
    preset : str, optional
        Preset name ('1d', '2d', '3d'). Loads default config for that type.
    field_name : str
        Name of the output being predicted (for logging/metadata).
    """

    def __init__(self, config=None, preset=None, field_name='output'):
        self.field_name = field_name
        self.preset = preset

        # Build config: start from preset defaults, override with custom config
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
            self.config = dict(PRESETS[preset])
        else:
            self.config = {}

        if config is not None:
            self.config.update(config)

        # Components
        self.param_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.model = None
        self.history = None

    @classmethod
    def from_preset(cls, preset_name, field_name='output', **overrides):
        """Create a SurrogateNN from a named preset with optional overrides."""
        return cls(config=overrides if overrides else None, preset=preset_name, field_name=field_name)

    def _build_model(self, input_dim, output_dim):
        """Build Keras model from config."""
        cfg = self.config
        hidden_layers = cfg['hidden_layers']
        dropout_rates = cfg['dropout']
        l2_val = cfg['l2']
        lr = cfg.get('learning_rate', 0.001)

        layers = [keras.layers.Input(shape=(input_dim,))]

        for i, units in enumerate(hidden_layers):
            layers.append(keras.layers.Dense(
                units, activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_val)
            ))
            if i < len(dropout_rates) and dropout_rates[i] > 0:
                layers.append(keras.layers.Dropout(dropout_rates[i]))

        layers.append(keras.layers.Dense(output_dim))

        model = keras.Sequential(layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae']
        )
        return model

    def _get_batch_size(self, n_samples):
        """Compute batch size from config."""
        bs = self.config.get('batch_size', 8)
        if bs == 'adaptive':
            return max(16, n_samples // 50)
        return bs

    def fit(self, parameters, outputs, validation_split=0.2, epochs=500, verbose=0, on_epoch=None):
        """
        Train the neural network.

        Parameters
        ----------
        parameters : np.ndarray
            Input parameters, shape (n_samples, n_params)
        outputs : np.ndarray
            Output values, shape (n_samples, n_outputs)
        validation_split : float
            Fraction of data for validation
        epochs : int
            Maximum training epochs
        verbose : int
            Keras verbosity level
        on_epoch : callable, optional
            Called as on_epoch(epoch, train_loss, val_loss) after each epoch.
            val_loss may be None if validation_split == 0.
        """
        cfg = self.config

        # Ensure outputs are 2D
        if outputs.ndim == 1:
            outputs = outputs.reshape(-1, 1)

        # Scale
        params_scaled = self.param_scaler.fit_transform(parameters)
        outputs_scaled = self.output_scaler.fit_transform(outputs)

        # Build model
        self.model = self._build_model(
            input_dim=parameters.shape[1],
            output_dim=outputs.shape[1]
        )

        batch_size = self._get_batch_size(len(parameters))
        monitor = 'loss' if validation_split == 0.0 else 'val_loss'

        early_stop = keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=cfg.get('es_patience', 50),
            restore_best_weights=True,
            min_delta=cfg.get('es_min_delta', 1e-7),
            start_from_epoch=cfg.get('es_start_epoch', 0),
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=cfg.get('lr_factor', 0.5),
            patience=cfg.get('lr_patience', 15),
            min_lr=cfg.get('lr_min', 1e-7),
        )

        callbacks_list = [early_stop, reduce_lr]

        if on_epoch is not None:
            class _EpochCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    on_epoch(epoch + 1, float(logs.get('loss', 0.0)),
                             float(logs.get('val_loss', 0.0)) if 'val_loss' in logs else None)
            callbacks_list.append(_EpochCallback())

        self.history = self.model.fit(
            params_scaled, outputs_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose,
        )

        if 'val_loss' in self.history.history:
            logger.info(f"Best val_loss: {min(self.history.history['val_loss']):.6f}")
        logger.info(f"Stopped at epoch: {len(self.history.history['loss'])}")

        return self.history

    def predict(self, parameters):
        """
        Predict outputs for given parameters.

        Parameters
        ----------
        parameters : np.ndarray
            Shape (n_samples, n_params)

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_outputs)
        """
        params_scaled = self.param_scaler.transform(parameters)
        outputs_scaled = self.model.predict(params_scaled, verbose=0)
        return self.output_scaler.inverse_transform(outputs_scaled)

    def save(self, filepath):
        """
        Save model to files.

        Parameters
        ----------
        filepath : Path or str
            Base filepath. Saves {filepath}_nn.h5 and {filepath}_nn.npz
        """
        filepath = Path(filepath)
        base = filepath.parent / f"{filepath.name}_nn"

        # Save scalers and config
        save_dict = {
            'field_name': self.field_name,
            'preset': self.preset if self.preset else '',
            'config_json': json.dumps(self.config),
            'param_scaler_mean': self.param_scaler.mean_,
            'param_scaler_scale': self.param_scaler.scale_,
            'output_scaler_mean': self.output_scaler.mean_,
            'output_scaler_scale': self.output_scaler.scale_,
        }
        np.savez_compressed(base.with_suffix('.npz'), **save_dict)

        # Save Keras model
        self.model.save(base.with_suffix('.h5'))

    @classmethod
    def load(cls, filepath):
        """
        Load model from files.

        Parameters
        ----------
        filepath : Path or str
            Base filepath. Loads from {filepath}_nn.h5 and {filepath}_nn.npz

        Returns
        -------
        SurrogateNN
        """
        filepath = Path(filepath)
        base = filepath.parent / f"{filepath.name}_nn"

        data = np.load(base.with_suffix('.npz'), allow_pickle=True)

        config = json.loads(str(data['config_json']))
        preset = str(data['preset']) if str(data['preset']) else None

        nn = cls(config=config, preset=None, field_name=str(data['field_name']))
        nn.preset = preset
        # Config was already set directly -- don't re-merge with preset

        # Restore scalers
        nn.param_scaler.mean_ = data['param_scaler_mean']
        nn.param_scaler.scale_ = data['param_scaler_scale']
        nn.output_scaler.mean_ = data['output_scaler_mean']
        nn.output_scaler.scale_ = data['output_scaler_scale']

        # Load Keras model
        nn.model = keras.models.load_model(base.with_suffix('.h5'), compile=False)
        nn.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )

        return nn
