"""
POD Reducer Module
===================
Standalone PCA-based Proper Orthogonal Decomposition for dimensionality reduction.
Used to reduce high-dimensional field data to a small number of modes before NN training.
"""

import logging
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class PODReducer:
    """
    PCA-based POD for reducing field data dimensionality.

    Parameters
    ----------
    n_modes : int
        Number of POD modes to retain
    """

    def __init__(self, n_modes=10):
        self.n_modes = n_modes
        self.pca = PCA(n_components=n_modes)
        self.variance_explained = None

    def fit_transform(self, fields):
        """
        Fit POD to field data and return reduced modes.

        Parameters
        ----------
        fields : np.ndarray
            Shape (n_samples, n_points)

        Returns
        -------
        modes : np.ndarray
            Shape (n_samples, n_modes)
        """
        n_samples, n_features = fields.shape
        max_modes = min(n_samples, n_features)

        if self.n_modes > max_modes:
            logger.warning(f"Requested {self.n_modes} POD modes but max is {max_modes}. Reducing to {max_modes}.")
            self.n_modes = max_modes
            self.pca = PCA(n_components=max_modes)

        modes = self.pca.fit_transform(fields)
        self.variance_explained = self.pca.explained_variance_ratio_
        logger.info(f"POD: {n_features} points -> {self.n_modes} modes "
                    f"({self.variance_explained.sum()*100:.2f}% variance explained)")
        return modes

    def transform(self, fields):
        """
        Project field data into mode space (after fitting).

        Parameters
        ----------
        fields : np.ndarray
            Shape (n_samples, n_points)

        Returns
        -------
        modes : np.ndarray
            Shape (n_samples, n_modes)
        """
        return self.pca.transform(fields)

    def inverse_transform(self, modes):
        """
        Reconstruct full field from modes.

        Parameters
        ----------
        modes : np.ndarray
            Shape (n_samples, n_modes)

        Returns
        -------
        fields : np.ndarray
            Shape (n_samples, n_points)
        """
        return self.pca.inverse_transform(modes)

    def save(self, filepath):
        """
        Save POD state to file.

        Parameters
        ----------
        filepath : Path or str
            Base filepath (without suffix). Saves as {filepath}_pod.npz
        """
        filepath = Path(filepath)
        save_dict = {
            'n_modes': self.n_modes,
            'pca_components': self.pca.components_,
            'pca_mean': self.pca.mean_,
            'pca_variance': self.variance_explained,
        }
        np.savez_compressed(filepath.parent / f"{filepath.name}_pod.npz", **save_dict)

    @classmethod
    def load(cls, filepath):
        """
        Load POD state from file.

        Parameters
        ----------
        filepath : Path or str
            Base filepath (without suffix). Loads from {filepath}_pod.npz

        Returns
        -------
        PODReducer
        """
        filepath = Path(filepath)
        data = np.load(filepath.parent / f"{filepath.name}_pod.npz", allow_pickle=True)

        reducer = cls(n_modes=int(data['n_modes']))
        reducer.pca.components_ = data['pca_components']
        reducer.pca.mean_ = data['pca_mean']
        reducer.variance_explained = data['pca_variance']
        return reducer
