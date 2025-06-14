#!/usr/bin/env python3
"""
Local launcher for F1 Data Driven Laps Streamlit App
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['streamlit', 'fastf1', 'matplotlib', 'numpy', 'pandas', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found!")
        print("💡 Install FFmpeg:")
        print("   - macOS: brew install ffmpeg")
        print("   - Ubuntu: sudo apt install ffmpeg")
        print("   - Windows: Download from https://ffmpeg.org/")
        return False

def main():
    print("🏎️  F1 Data Driven Laps Streamlit App - Local Launcher")
    print("=" * 50)
    
    # Get current directory
    current_dir = Path(__file__).parent
    app_file = current_dir / "app.py"
    
    if not app_file.exists():
        print("❌ app.py not found in current directory!")
        return
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return
    
    print("✅ Python dependencies OK")
    
    # Check FFmpeg
    print("🔍 Checking FFmpeg...")
    if not check_ffmpeg():
        print("⚠️  FFmpeg not found. Video generation may fail.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    else:
        print("✅ FFmpeg OK")
    
    # Launch Streamlit
    print("\n🚀 Launching Streamlit app...")
    print("📱 App will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Change to app directory and run streamlit
        os.chdir(current_dir)
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.address', 'localhost',
            '--server.port', '8501',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user.")
    except Exception as e:
        print(f"\n❌ Error running app: {e}")

if __name__ == "__main__":
    main() 