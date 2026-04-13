"""
User Settings Manager
=====================
Manages user preferences and configuration persistence.
"""

import json
from pathlib import Path


class UserSettings:
    """Manager for user settings and preferences."""

    def __init__(self, config_file):
        self.config_file = Path(config_file)
        self.data = self.load()

    def load(self):
        """Load settings from config file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return self._default_settings()
        return self._default_settings()

    def _default_settings(self):
        """Return default settings structure."""
        return {
            'recent_project_folders': [],
            'solver_settings': {
                'precision': 'single',
                'processor_count': 2,
                'dimension': 3,
                'use_gui': True
            }
        }

    def save(self):
        """Save settings to config file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    # --- Recent projects ---

    def add_recent_project_folder(self, project_path):
        """Add a project folder to recent list (most recent first, max 5)."""
        project_path = str(project_path)

        recent = self.data.get('recent_project_folders', [])
        if project_path in recent:
            recent.remove(project_path)
        recent.insert(0, project_path)
        self.data['recent_project_folders'] = recent[:5]
        self.save()

    def get_recent_project_folders(self):
        """Get list of recent project folders (filters out non-existent)."""
        return [p for p in self.data.get('recent_project_folders', []) if Path(p).exists()]

    # --- Solver settings ---

    def get_solver_settings(self):
        """Get saved solver settings."""
        return self.data.get('solver_settings', self._default_settings()['solver_settings']).copy()

    def save_solver_settings(self, settings):
        """Save solver settings."""
        self.data['solver_settings'] = settings
        self.save()

    # --- NN architecture settings ---

    def get_nn_settings(self):
        """Get NN architecture settings per model type (1d/2d/3d)."""
        return self.data.get('nn_settings', {}).copy()

    def save_nn_settings(self, nn_settings):
        """Save NN architecture settings."""
        self.data['nn_settings'] = nn_settings
        self.save()
