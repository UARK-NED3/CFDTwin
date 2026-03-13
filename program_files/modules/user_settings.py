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
            except:
                return self._default_settings()
        return self._default_settings()

    def _default_settings(self):
        """Return default settings structure."""
        return {
            'recent_project_folders': [],  # Recent project folders
            'recent_case_files': [],       # Recent Fluent case files
            'recent_setups': [],           # Recent model setup JSON files
            'solver_settings': {
                'precision': 'single',
                'processor_count': 2,
                'dimension': 3,
                'use_gui': True
            }
        }

    def save(self):
        """Save settings to config file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def add_recent_project_folder(self, project_path):
        """Add a project folder to recent list (most recent first)."""
        project_path = str(project_path)

        # Initialize if needed
        if 'recent_project_folders' not in self.data:
            self.data['recent_project_folders'] = []

        # Remove if already exists
        if project_path in self.data['recent_project_folders']:
            self.data['recent_project_folders'].remove(project_path)

        # Add to front
        self.data['recent_project_folders'].insert(0, project_path)

        # Keep only 5 most recent
        self.data['recent_project_folders'] = self.data['recent_project_folders'][:5]

        self.save()

    def get_recent_project_folders(self):
        """Get list of recent project folders."""
        if 'recent_project_folders' not in self.data:
            self.data['recent_project_folders'] = []
        return [p for p in self.data['recent_project_folders'] if Path(p).exists()]

    def add_recent_case_file(self, case_path):
        """Add a Fluent case file to recent list (most recent first)."""
        case_path = str(case_path)

        # Initialize if needed
        if 'recent_case_files' not in self.data:
            self.data['recent_case_files'] = []

        # Remove if already exists
        if case_path in self.data['recent_case_files']:
            self.data['recent_case_files'].remove(case_path)

        # Add to front
        self.data['recent_case_files'].insert(0, case_path)

        # Keep only 5 most recent
        self.data['recent_case_files'] = self.data['recent_case_files'][:5]

        self.save()

    def get_recent_case_files(self):
        """Get list of recent Fluent case files."""
        if 'recent_case_files' not in self.data:
            self.data['recent_case_files'] = []
        return [p for p in self.data['recent_case_files'] if Path(p).exists()]

    # Legacy support for old settings
    def add_recent_project(self, project_path):
        """Legacy method - redirects to add_recent_case_file."""
        self.add_recent_case_file(project_path)

    def get_recent_projects(self):
        """Legacy method - redirects to get_recent_case_files."""
        return self.get_recent_case_files()

    def add_recent_setup(self, setup_path):
        """Add a model setup to recent list (most recent first)."""
        setup_path = str(setup_path)

        # Remove if already exists
        if setup_path in self.data.get('recent_setups', []):
            self.data['recent_setups'].remove(setup_path)

        # Add to front
        if 'recent_setups' not in self.data:
            self.data['recent_setups'] = []
        self.data['recent_setups'].insert(0, setup_path)

        # Keep only 3 most recent
        self.data['recent_setups'] = self.data['recent_setups'][:3]

        self.save()

    def get_recent_setups(self):
        """Get list of 3 most recent model setups."""
        return [p for p in self.data.get('recent_setups', []) if Path(p).exists()]

    def get_solver_settings(self):
        """Get saved solver settings."""
        return self.data['solver_settings'].copy()

    def save_solver_settings(self, settings):
        """Save solver settings."""
        self.data['solver_settings'] = settings
        self.save()
