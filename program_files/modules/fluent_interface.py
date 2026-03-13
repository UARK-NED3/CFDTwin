"""
Fluent Interface Module
=======================
Handles all PyFluent API interactions including launching, connecting,
and introspecting Fluent cases.
"""

import sys
from pathlib import Path
from tkinter import Tk, filedialog


def open_case_file(user_settings, project_dir, ui_helpers):
    """Open a Fluent case file with GUI."""
    ui_helpers.print_header("OPEN FLUENT CASE FILE")

    # Use file dialog to select case file
    print("\nOpening file dialog...")
    Tk().withdraw()  # Hide tkinter root window
    case_file = filedialog.askopenfilename(
        title="Select Fluent Case File",
        filetypes=[
            ("Fluent Case Files", "*.cas *.cas.h5 *.cas.gz"),
            ("All Files", "*.*")
        ],
        initialdir=str(project_dir)
    )

    if not case_file:
        print("\n✗ No file selected")
        ui_helpers.pause()
        return None

    case_file = Path(case_file)
    print(f"\n✓ Selected: {case_file.name}")
    print(f"  Full path: {case_file}")

    # Add to recent case files
    user_settings.add_recent_case_file(case_file)

    # Configure solver settings
    settings = ui_helpers.configure_solver_settings(user_settings)
    if settings is None:
        print("\n✗ Launch cancelled by user")
        ui_helpers.pause()
        return None

    # Launch Fluent
    print("\nLaunching Fluent...")
    print(f"  Precision: {settings['precision']}")
    print(f"  Processors: {settings['processor_count']}")
    print(f"  Dimension: {settings['dimension']}D")
    print(f"  Mode: solver")
    print(f"  GUI: {'Enabled' if settings['use_gui'] else 'Disabled'}")

    try:
        import ansys.fluent.core as pyfluent
        from ansys.fluent.core.launcher.launcher import UIMode

        # Create log file for Fluent output
        log_dir = project_dir / "fluent_logs"
        log_dir.mkdir(exist_ok=True)
        log_file_path = log_dir / f"fluent_launch_{case_file.stem}.log"
        fluent_log_file = open(log_file_path, 'w', buffering=1)

        # Redirect stdout/stderr to suppress Fluent TUI output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = fluent_log_file
        sys.stderr = fluent_log_file

        # Determine UI mode
        ui_mode = UIMode.GUI if settings['use_gui'] else UIMode.NO_GUI_OR_GRAPHICS

        solver = pyfluent.launch_fluent(
            precision=settings['precision'],
            processor_count=settings['processor_count'],
            dimension=settings['dimension'],
            mode="solver",
            ui_mode=ui_mode
        )

        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(f"\n✓ Fluent launched (version {solver.get_fluent_version()})")
        print(f"  Loading case file: {case_file.name}")
        print(f"  Fluent output redirected to: {log_file_path.name}")

        # Redirect during case loading
        sys.stdout = fluent_log_file
        sys.stderr = fluent_log_file
        solver.settings.file.read_case(file_name=str(case_file))
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(f"\n✓ Case file loaded successfully")

        fluent_log_file.close()

        # Store case file path as attribute for later reference
        solver._case_file_path = str(case_file)

        return solver

    except Exception as e:
        # Restore stdout/stderr if error occurs
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"\n✗ Error launching Fluent: {e}")

        # Check for connection-related errors (VPN issue)
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['connection refused', 'connect', 'unavailable', '10061']):
            print("\n" + "!"*70)
            print("CONNECTION ERROR DETECTED")
            print("!"*70)
            print("\n⚠️  REMINDER: Make sure your VPN is enabled!")
            print("\nThis error typically occurs when:")
            print("  1. VPN is not connected (MOST COMMON)")
            print("  2. Firewall is blocking the connection")
            print("  3. Fluent license server is unreachable")
            print("\nPlease enable your VPN and try again.")
            print("="*70)

        import traceback
        traceback.print_exc()
        try:
            fluent_log_file.close()
        except:
            pass
        ui_helpers.pause()
        return None


def open_recent_project(project_path, user_settings, project_dir, ui_helpers):
    """Open a recent project."""
    ui_helpers.print_header("OPEN RECENT PROJECT")

    project_path = Path(project_path)
    print(f"\n✓ Selected: {project_path.name}")
    print(f"  Full path: {project_path}")

    if not project_path.exists():
        print(f"\n✗ File not found: {project_path}")
        ui_helpers.pause()
        return None

    # Configure solver settings
    settings = ui_helpers.configure_solver_settings(user_settings)
    if settings is None:
        print("\n✗ Launch cancelled by user")
        ui_helpers.pause()
        return None

    # Launch Fluent
    print("\nLaunching Fluent...")
    print(f"  Precision: {settings['precision']}")
    print(f"  Processors: {settings['processor_count']}")
    print(f"  Dimension: {settings['dimension']}D")
    print(f"  Mode: solver")
    print(f"  GUI: {'Enabled' if settings['use_gui'] else 'Disabled'}")

    try:
        import ansys.fluent.core as pyfluent
        from ansys.fluent.core.launcher.launcher import UIMode

        # Create log file for Fluent output
        log_dir = project_dir / "fluent_logs"
        log_dir.mkdir(exist_ok=True)
        log_file_path = log_dir / f"fluent_launch_{project_path.stem}.log"
        fluent_log_file = open(log_file_path, 'w', buffering=1)

        # Redirect stdout/stderr to suppress Fluent TUI output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = fluent_log_file
        sys.stderr = fluent_log_file

        # Determine UI mode
        ui_mode = UIMode.GUI if settings['use_gui'] else UIMode.NO_GUI_OR_GRAPHICS

        solver = pyfluent.launch_fluent(
            precision=settings['precision'],
            processor_count=settings['processor_count'],
            dimension=settings['dimension'],
            mode="solver",
            ui_mode=ui_mode
        )

        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(f"\n✓ Fluent launched (version {solver.get_fluent_version()})")
        print(f"  Loading case file: {project_path.name}")
        print(f"  Fluent output redirected to: {log_file_path.name}")

        # Redirect during case loading
        sys.stdout = fluent_log_file
        sys.stderr = fluent_log_file
        solver.settings.file.read_case(file_name=str(project_path))
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(f"\n✓ Case file loaded successfully")

        fluent_log_file.close()

        # Store case file path as attribute for later reference
        solver._case_file_path = str(project_path)

        # Update recent case files (move to top)
        user_settings.add_recent_case_file(project_path)

        return solver

    except Exception as e:
        # Restore stdout/stderr if error occurs
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"\n✗ Error opening project: {e}")

        # Check for connection-related errors (VPN issue)
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['connection refused', 'connect', 'unavailable', '10061']):
            print("\n" + "!"*70)
            print("CONNECTION ERROR DETECTED")
            print("!"*70)
            print("\n⚠️  REMINDER: Make sure your VPN is enabled!")
            print("\nThis error typically occurs when:")
            print("  1. VPN is not connected (MOST COMMON)")
            print("  2. Firewall is blocking the connection")
            print("  3. Fluent license server is unreachable")
            print("\nPlease enable your VPN and try again.")
            print("="*70)

        import traceback
        traceback.print_exc()
        try:
            fluent_log_file.close()
        except:
            pass
        ui_helpers.pause()
        return None
