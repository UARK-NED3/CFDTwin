"""
Fluent Validation Cache
=======================
Persistent on-disk cache for Fluent comparison runs.
One NPZ file per cached run, indexed by a hash of the input parameters.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


def _hash_params(params):
    """Stable hash of a dict of {key: float}. Sorted key order."""
    payload = json.dumps({k: float(v) for k, v in sorted(params.items())}, sort_keys=True)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]


class FluentCache:
    """Persistent cache for Fluent comparison results."""

    def __init__(self, cache_dir, index_file):
        self.cache_dir = Path(cache_dir)
        self.index_file = Path(index_file)

    def _load_index(self):
        if not self.index_file.exists():
            return {}
        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self, index):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)

    def lookup(self, params):
        """Return cached NPZ data dict for these params, or None."""
        h = _hash_params(params)
        index = self._load_index()
        entry = index.get(h)
        if entry is None:
            return None
        npz_path = self.cache_dir / entry['file']
        if not npz_path.exists():
            # Stale index entry
            del index[h]
            self._save_index(index)
            return None
        try:
            data = np.load(npz_path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        except Exception as e:
            logger.warning(f"Failed to load cached NPZ {npz_path}: {e}")
            return None

    def store(self, params, data):
        """Store NPZ data for these params."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        h = _hash_params(params)
        file_name = f"run_{h}.npz"
        npz_path = self.cache_dir / file_name

        np.savez_compressed(npz_path, **data)

        index = self._load_index()
        index[h] = {
            'file': file_name,
            'timestamp': datetime.now().isoformat(),
            'params': {k: float(v) for k, v in params.items()},
        }
        self._save_index(index)
        logger.info(f"Cached Fluent run to {file_name}")

    def clear(self):
        """Remove all cached runs and the index."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        logger.info("Fluent cache cleared")

    def count(self):
        """Number of cached runs."""
        return len(self._load_index())
