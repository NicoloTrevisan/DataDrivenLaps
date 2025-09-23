#!/usr/bin/env python3
"""
Simple launcher for the F1 Data Driven Laps GUI
"""

import subprocess
import sys
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gui_script = os.path.join(script_dir, 'scripts', 'interactive_f1_ghost_racing_gui.py')
    
    if not os.path.exists(gui_script):
        print("❌ GUI script not found!")
        return
    
    print("🏎️ Launching F1 Data Driven Laps GUI...")
    
    try:
        subprocess.run([sys.executable, gui_script], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n👋 GUI closed by user.")
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")

if __name__ == "__main__":
    main() 