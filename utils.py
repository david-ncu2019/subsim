import json
import os
import shutil
from datetime import datetime
from pathlib import Path


class ConfigManager:
    """
    Manages configuration files for subsidence simulations.
    
    This class handles loading parameters from config files and saving
    timestamped copies to a storage directory for record-keeping.
    """
    
    def __init__(self, storage_dir="config_storage"):
        """
        Initialize the config manager.
        
        Args:
            storage_dir (str): Directory where config copies will be stored
        """
        self.storage_dir = Path(storage_dir)
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_path):
        """
        Load simulation parameters from a JSON config file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            dict: Dictionary containing simulation parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file contains invalid JSON
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"Successfully loaded config from: {config_path}")
            return config
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in config file: {e}")
    
    def save_config_copy(self, config_path, prefix):
        """
        Save a timestamped copy of the config file to storage directory.
        
        Args:
            config_path (str): Path to the original config file
            prefix (str): Prefix for the saved filename
            
        Returns:
            str: Path to the saved config copy
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Generate timestamp in YYYYMMDD_hhmmss format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the new filename
        new_filename = f"{prefix}_{timestamp}.txt"
        destination_path = self.storage_dir / new_filename
        
        # Copy the file
        shutil.copy2(config_path, destination_path)
        
        print(f"Config copy saved to: {destination_path}")
        return str(destination_path)
    
    def create_sample_config(self, output_path="sample_config.json"):
        """
        Create a sample configuration file with default parameters.
        
        Args:
            output_path (str): Where to save the sample config file
        """
        sample_config = {
            "resolution": 100,
            "time_steps": 120,
            "t_range": [0, 10],
            "x_range": [-10, 4],
            "y_range": [-8, 8],
            "x_center": -3,
            "y_center": 2,
            "A": 1.5,
            "rate": 0.05,
            "sigma": 5,
            "S": 0.2,
            "f": 0.5,
            "sigma_fluc": 1.0
        }
        
        with open(output_path, 'w') as f:
            json.dump(sample_config, f, indent=4)
        
        print(f"Sample config created at: {output_path}")


def run_simulation_from_config(config_path, prefix, storage_dir="config_storage"):
    """
    Complete workflow: load config, save copy, run simulation.
    
    This function demonstrates the entire process of loading parameters
    from a config file, saving a timestamped copy, and running the simulation.
    
    Args:
        config_path (str): Path to the configuration file
        prefix (str): Prefix for the saved config filename
        storage_dir (str): Directory for storing config copies
        
    Returns:
        SubsidenceSimulator: The simulator object with generated data
    """
    # Import here to avoid circular imports
    from subsim import SubsidenceSimulator
    
    # Step 1: Initialize config manager
    config_manager = ConfigManager(storage_dir)
    
    # Step 2: Load simulation parameters
    print("Loading configuration...")
    params = config_manager.load_config(config_path)
    
    # Step 3: Save timestamped copy
    print("Saving config copy...")
    saved_config_path = config_manager.save_config_copy(config_path, prefix)
    
    # Step 4: Create and run simulation
    print("Creating simulator...")
    simulator = SubsidenceSimulator(**params)
    
    print("Generating simulation data...")
    X, Y, Z = simulator.generate_data()
    
    # Step 5: Display results summary
    simulator.get_summary()
    
    return simulator