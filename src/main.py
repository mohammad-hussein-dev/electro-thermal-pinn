#!/usr/bin/env python3
"""
Electro-Thermal PINN: Main entry point.

This script provides a user-friendly menu to choose between:
    1. Streamlit Dashboard (for employers and professional presentation)
    2. Terminal Mode (for development and quick testing)
    3. Exit
"""

import os
import sys
import subprocess


def print_header():
    """Print the application header."""
    print("\n" + "="*70)
    print("  ⚡ ELECTRO-THERMAL PINN - MAIN MENU")
    print("="*70)
    print("\n  A Physics-Informed Neural Network (PINN) for solving")
    print("  coupled Maxwell's equations and heat equation with Joule heating.\n")


def print_menu():
    """Print the main menu options."""
    print("  +---+--------------------------------------------+")
    print("  | # | Mode                                       |")
    print("  +---+--------------------------------------------+")
    print("  | 1 | Dashboard (Streamlit)                      |")
    print("  |   | Interactive UI for employers & clients    |")
    print("  +---+--------------------------------------------+")
    print("  | 2 | Terminal Mode                              |")
    print("  |   | Command-line with architecture selection  |")
    print("  +---+--------------------------------------------+")
    print("  | 0 | Exit                                      |")
    print("  +---+--------------------------------------------+")
    print("\n  Legend:")
    print("    * Mode 1: Professional dashboard with live plots")
    print("    * Mode 2: Terminal-based training & evaluation")
    print("    * Mode 0: Quit the application")
    print("="*70)


def run_streamlit_dashboard():
    """Launch the Streamlit dashboard."""
    print("\n  🚀 Launching Streamlit Dashboard...")
    print("  📊 This will open the interactive UI in your browser.\n")
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("  ❌ Streamlit is not installed.")
        print("  📦 Please install it with: pip install streamlit")
        print("  🔄 Then run this script again.")
        return False
    
    # Check if app.py exists
    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    if not os.path.exists(app_path):
        print(f"  ❌ Dashboard file not found: {app_path}")
        print("  💡 Please ensure app.py exists in the project root.")
        return False
    
    try:
        # Run streamlit
        subprocess.run(["streamlit", "run", app_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error running Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n  👋 Streamlit dashboard closed.")
        return True


def run_terminal_mode():
    """Run the terminal-based training mode."""
    print("\n  📟 Starting Terminal Mode...\n")
    
    # Import and run the main training script
    try:
        # Add the current directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Import the main training function
        from main_terminal import main_terminal
        main_terminal()
        return True
    except ImportError as e:
        print(f"  ❌ Error importing terminal mode: {e}")
        print("  💡 Ensure src/main_terminal.py exists.")
        return False
    except KeyboardInterrupt:
        print("\n  👋 Terminal mode interrupted.")
        return True


def main():
    """Main entry point for the application."""
    while True:
        print_header()
        print_menu()
        
        choice = input("\n  Enter your choice (0-2): ").strip()
        
        if choice == "1":
            run_streamlit_dashboard()
            input("\n  Press Enter to return to main menu...")
        elif choice == "2":
            run_terminal_mode()
            input("\n  Press Enter to return to main menu...")
        elif choice == "0":
            print("\n  👋 Goodbye! Thank you for using Electro-Thermal PINN.\n")
            sys.exit(0)
        else:
            print("\n  ❌ Invalid choice. Please enter 0, 1, or 2.")
            input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()
