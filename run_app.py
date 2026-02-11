#!/usr/bin/env python3
"""
Plant Identifier - Cross-Platform Application Launcher
This script prepares the environment and starts the Flask web application
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 50)
    print(f"  {text}")
    print("=" * 50 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ is required")
        print(f"Your version: {sys.version}")
        sys.exit(1)
    print(f"[OK] Python {sys.version.split()[0]} found")

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'flask', 'tensorflow', 'keras', 'numpy', 'PIL', 'werkzeug'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"[WARNING] Missing packages: {', '.join(missing)}")
        print("[INFO] Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("[OK] Dependencies installed")
    else:
        print("[OK] All dependencies installed")

def check_model():
    """Check if trained model exists"""
    model_path = Path('models/plant_model.h5')
    
    if not model_path.exists():
        print("[WARNING] Trained model not found!")
        print("\nThe app needs a trained model to work.")
        print("Training will take approximately 10-15 minutes.")
        
        response = input("\nDo you want to train the model now? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\n[INFO] Starting model training...")
            result = subprocess.run([sys.executable, 'train_model.py'])
            if result.returncode != 0:
                print("[ERROR] Training failed. Please check the error messages above.")
                sys.exit(1)
            print("[OK] Training completed successfully!")
        else:
            print("[ERROR] Cannot run app without trained model.")
            sys.exit(1)
    else:
        print("[OK] Model found")

def start_app():
    """Start the Flask application"""
    print_header("Starting Flask Web Application")
    print("[INFO] The app will open at: http://localhost:5000")
    print("[INFO] Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, 'app_improved.py'])
    except KeyboardInterrupt:
        print("\n\n[INFO] Application stopped by user")
        sys.exit(0)

def main():
    """Main launcher function"""
    print_header("PLANT IDENTIFIER - Application Launcher")
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Run checks
    print("[INFO] Running pre-flight checks...\n")
    check_python_version()
    check_dependencies()
    check_model()
    
    # Start app
    start_app()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
