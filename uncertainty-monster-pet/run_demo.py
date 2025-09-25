"""
Quick Start Script for Uncertainty Monster Pet Demo
Run this to launch the educational demonstration
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'torch',
        'numpy',
        'scipy',
        'matplotlib',
        'plotly',
        'pandas'
    ]

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
        print("\n📦 Install with: pip install -r requirements_simple.txt")
        return False

    return True

def run_demo():
    """Launch the Streamlit demo"""
    print("🎯 Starting Uncertainty Monster Pet Demo...")
    print("🧠 Loading Bayesian, Spiking, and Statistical models...")

    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "uncertainty_pet.py"])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("💡 Try running manually: streamlit run uncertainty_pet.py")

def main():
    print("🎯 Uncertainty Monster Pet - Educational Demo")
    print("=" * 50)
    print("Demonstrates three approaches to uncertainty in AI:")
    print("🧠 Bayesian Neural Networks - Prediction confidence")
    print("⚡ Spiking Neural Networks - Temporal variability")
    print("📊 Statistical Analysis - Distribution moments")
    print("=" * 50)

    if not check_requirements():
        return

    print("✅ All requirements satisfied!")
    print("🚀 Launching demo...")

    run_demo()

if __name__ == "__main__":
    main()