"""
UI Helper Functions
===================
Terminal UI utilities for the Workflow Surrogate application.
"""

import sys
import os


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)


def print_menu(title, options):
    """Print a menu with options."""
    print_header(title)
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    print(f"  [0] {'Back' if title != 'WORKFLOW SURROGATE - MAIN MENU' else 'Exit'}")
    print("="*70)


def get_choice(max_choice):
    """Get user choice with validation."""
    while True:
        try:
            choice = int(input("\nEnter choice: ").strip())
            if 0 <= choice <= max_choice:
                return choice
            print(f"Invalid choice. Please enter 0-{max_choice}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def pause():
    """Pause and wait for user input."""
    input("\nPress Enter to continue...")


def configure_solver_settings(user_settings):
    """
    Configure Fluent solver settings before launch.

    Parameters
    ----------
    user_settings : UserSettings
        User settings manager instance

    Returns
    -------
    dict or None
        Solver settings dict, or None if cancelled
    """
    # Load saved settings or use defaults
    settings = user_settings.get_solver_settings()

    while True:
        print("\n" + "="*70)
        print("FLUENT SOLVER SETTINGS")
        print("="*70)
        print(f"\n  [1] Precision: {settings['precision']}")
        print(f"  [2] Processor Count: {settings['processor_count']}")
        print(f"  [3] Dimension: {settings['dimension']}D")
        print(f"  [4] GUI: {'Enabled' if settings['use_gui'] else 'Disabled'}")
        print("\n  [P] Proceed with these settings")
        print("  [C] Cancel")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'P':
            # Save settings before returning
            user_settings.save_solver_settings(settings)
            return settings
        elif choice == 'C':
            return None
        elif choice == '1':
            print("\n[1] Single precision")
            print("[2] Double precision")
            prec_choice = input("Select precision [1-2]: ").strip()
            if prec_choice == '1':
                settings['precision'] = 'single'
            elif prec_choice == '2':
                settings['precision'] = 'double'
        elif choice == '2':
            try:
                count = int(input("\nEnter processor count: ").strip())
                if count > 0:
                    settings['processor_count'] = count
                else:
                    print("Must be positive")
                    pause()
            except:
                print("Invalid input")
                pause()
        elif choice == '3':
            print("\n[1] 2D")
            print("[2] 3D")
            dim_choice = input("Select dimension [1-2]: ").strip()
            if dim_choice == '1':
                settings['dimension'] = 2
            elif dim_choice == '2':
                settings['dimension'] = 3
        elif choice == '4':
            settings['use_gui'] = not settings['use_gui']
