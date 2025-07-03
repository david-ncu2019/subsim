#!/usr/bin/env python3
"""
Main program for subsidence simulation using config files.

This script demonstrates how to use the SubsidenceSimulator class
with configuration files for reproducible experiments.
"""

import os
from datetime import datetime
from pathlib import Path

# Import our custom modules
from subsim import SubsidenceSimulator
from utils import ConfigManager


def main():
    """
    Main function that runs the complete subsidence simulation workflow.
    """
    print("=== Subsidence Simulation Program ABC ===\n")
    
    # Step 1: Setup configuration
    config_manager = ConfigManager("config_storage")
    
    # Check if sample config exists, create if not
    config_file = "sample_config.json"
    if not os.path.exists(config_file):
        print(f"Creating sample config file: {config_file}")
        config_manager.create_sample_config(config_file)
        print("Sample config created! You can edit it with your parameters.\n")
    
    # Step 2: Load configuration
    print("Loading simulation parameters...")
    try:
        params = config_manager.load_config(config_file)
        print("✓ Configuration loaded successfully\n")
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Step 3: Save timestamped copy
    experiment_prefix = "subsidence_exp"  # You can change this
    # print("Saving timestamped config copy...")
    # try:
    #     saved_config = config_manager.save_config_copy(config_file, experiment_prefix)
    #     print(f"✓ Config saved as: {os.path.basename(saved_config)}\n")
    # except Exception as e:
    #     print(f"Error saving config copy: {e}")
    #     return
    
    # Step 4: Create and run simulation
    print("Creating subsidence simulator...")
    simulator = SubsidenceSimulator(**params)
    
    print("Generating simulation data...")
    X, Y, Z = simulator.generate_data()
    print("✓ Data generation complete\n")
    
    # Step 5: Display summary
    # simulator.get_summary()
    # print()
    
    # Step 6: Create visualization
    print("Creating 3D visualization...")
    simulator.plot_interactive()
    
    # Step 7: Save results
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # results_file = f"results/{experiment_prefix}_{timestamp}_results.nc"
    # print(f"Saving results to: {results_file}")
    # simulator.save_to_netcdf(results_file)
    
    # Step 8: Example analysis
    # print("\n=== Quick Analysis ===")
    # row_idx, col_idx = simulator.get_indices_from_coords(
    #     simulator.x_center, simulator.y_center
    # )
    # center_subsidence = simulator.results_array[:, row_idx, col_idx]
    
    # print(f"Subsidence at center coordinates ({simulator.x_center}, {simulator.y_center}):")
    # print(f"  Initial: {center_subsidence[0]:.4f} meters")
    # print(f"  Final: {center_subsidence[-1]:.4f} meters")
    # print(f"  Total change: {center_subsidence[-1] - center_subsidence[0]:.4f} meters")
    
    # print(f"\nSimulation complete! Check the 'results' folder for output files.")


if __name__ == "__main__":
    main()