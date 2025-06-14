#!/usr/bin/env python3
"""
Fix environment for F1 Data Driven Laps Streamlit App
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🔧 F1 Data Driven Laps - Environment Fixer")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the streamlit_app directory")
        return
    
    print("📦 Fixing package compatibility issues...")
    
    # Uninstall problematic packages
    commands = [
        ("pip uninstall -y numpy pyarrow streamlit", "Removing problematic packages"),
        ("pip install 'numpy<2.0.0'", "Installing compatible NumPy"),
        ("pip install 'pyarrow<15.0.0'", "Installing compatible PyArrow"),
        ("pip install streamlit", "Reinstalling Streamlit"),
        ("pip install -r requirements.txt", "Installing all requirements"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️  Warning: {description} failed, but continuing...")
    
    print("\n🧪 Testing installation...")
    test_imports = [
        "import numpy",
        "import streamlit", 
        "import fastf1",
        "import matplotlib.pyplot",
        "import pandas"
    ]
    
    for test_import in test_imports:
        try:
            exec(test_import)
            print(f"✅ {test_import.split()[-1]} OK")
        except Exception as e:
            print(f"❌ {test_import.split()[-1]} failed: {e}")
    
    print("\n🚀 Environment fixed! Try running:")
    print("   streamlit run app.py --server.port 8502")

if __name__ == "__main__":
    main() 