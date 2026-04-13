"""
Dataset Manager Module
======================
Manages dataset versioning and sim file bookkeeping.
Reads/writes dataset_version.json in the dataset directory.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset versioning and sim file tracking."""

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self._version_file = self.dataset_dir / "dataset_version.json"

    def get_version(self):
        """Get current dataset version (0 if no version file)."""
        if not self._version_file.exists():
            return 0
        try:
            with open(self._version_file, 'r') as f:
                data = json.load(f)
            return data.get('version', 0)
        except Exception:
            return 0

    def bump_version(self):
        """Increment dataset version by 1. Creates file if needed."""
        version = self.get_version() + 1
        self.dataset_dir.mkdir(exist_ok=True)
        with open(self._version_file, 'w') as f:
            json.dump({'version': version}, f, indent=2)
        logger.info(f"Dataset version bumped to {version}")
        return version

    def get_completed_ids(self):
        """Return sorted list of completed simulation IDs (ints)."""
        if not self.dataset_dir.exists():
            return []
        ids = []
        for f in self.dataset_dir.glob("sim_*.npz"):
            try:
                sim_id = int(f.stem.split('_')[1])
                ids.append(sim_id)
            except (ValueError, IndexError):
                pass
        return sorted(ids)

    def get_sim_count(self):
        """Return number of completed simulation files."""
        return len(self.get_completed_ids())

    def remove_simulation(self, sim_id):
        """Delete a sim file and bump version. Returns True on success."""
        sim_file = self.dataset_dir / f"sim_{sim_id:04d}.npz"
        if sim_file.exists():
            sim_file.unlink()
            self.bump_version()
            logger.info(f"Removed sim_{sim_id:04d}.npz")
            return True
        return False

    def is_model_stale(self, model_metadata):
        """Check if a model was trained on an older dataset version."""
        current = self.get_version()
        model_version = model_metadata.get('dataset_version', 0)
        return current > model_version
