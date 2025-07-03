#!/usr/bin/env python3
"""
Setup script to verify all components work correctly.
Run this first to test your installation.
"""

import os
import sys
from pathlib import Path

def check_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported")
        
        import numpy as np
        print("✓ numpy imported")
        
        import xarray as xr
        print("✓ xarray imported")
        
        from subsim import SubsidenceSimulator
        print("✓ SubsidenceSimulator imported")
        
        from utils import ConfigManager
        print("✓ ConfigManager imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    dirs = ["config_storage", "results"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Directory created: {dir_name}")

def test_basic_functionality():
    """Test basic functionality without running full simulation."""
    print("\nTesting basic functionality...")
    try:
        from utils import ConfigManager
        from subsim import SubsidenceSimulator
        
        # Test config manager
        config_manager = ConfigManager("config_storage")
        config_manager.create_sample_config("input_config.json")
        print("✓ Config file creation works")
        
        # Test loading config
        params = config_manager.load_config("input_config.json")
        print("✓ Config file loading works")
        
        # Test simulator creation (without generating data)
        simulator = SubsidenceSimulator(**params)
        print("✓ Simulator creation works")
        
        # Cleanup
        os.remove("input_config.json")
        print("✓ Basic functionality test passed")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    print("=== Subsidence Project Setup & Test ===\n")
    
    # Check if we're in the right directory
    required_files = ["subsim.py", "utils.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"✗ Missing required files: {missing_files}")
        print("Make sure you're in the project directory with all Python files.")
        return
    
    # Run tests
    if not check_imports():
        print("\n✗ Import test failed. Install missing packages with:")
        print("  pip install matplotlib numpy xarray")
        return
    
    create_directories()
    
    if not test_basic_functionality():
        print("\n✗ Functionality test failed.")
        return
    
    print("\n🎉 Setup complete! Everything is working.")
    print("\nNext steps:")
    print("1. Edit 'input_config.json' with your simulation parameters")
    print("2. Run 'python main.py' to start your simulation")

if __name__ == "__main__":
    main()