import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import xarray as xr


class SubsidenceSimulator:
    """
    A class for simulating ground subsidence over time and space.
    
    This class encapsulates all the functionality needed to model
    subsidence bowls with both long-term trends and seasonal variations.
    """
    
    def __init__(self, **params):
        """
        Initialize the subsidence simulator with simulation parameters.
        
        Args:
            **params: Keyword arguments for simulation parameters including:
                resolution (int): Number of grid points per axis (default: 30)
                time_steps (int): Number of time points (default: 100)
                x_range (tuple): X-axis boundaries (default: (-5, 5))
                y_range (tuple): Y-axis boundaries (default: (-5, 5))
                t_range (tuple): Time range (default: (0, 10))
                x_center (float): X-coordinate of subsidence center (default: 0)
                y_center (float): Y-coordinate of subsidence center (default: 0)
                A (float): Maximum depth of subsidence (default: 1)
                sigma (float): Controls spread of subsidence bowl (default: 2)
                rate (float): Long-term linear subsidence rate (default: 0.1)
                S (float): Amplitude of seasonal fluctuations (default: 0.2)
                f (float): Frequency of seasonal fluctuations (default: 0.5)
                sigma_fluc (float): Spatial extent of seasonal fluctuations (default: 1.0)
        """
        # Set default parameters first
        defaults = {
            'resolution': 30,
            'time_steps': 100,
            'x_range': (-5, 5),
            'y_range': (-5, 5),
            't_range': (0, 10),
            'x_center': 0,
            'y_center': 0,
            'A': 1,
            'sigma': 2,
            'rate': 0.1,
            'S': 0.2,
            'f': 0.5,
            'sigma_fluc': 1.0
        }
        
        # Update defaults with user-provided parameters
        defaults.update(params)
        
        # Store all parameters as instance attributes
        for key, value in defaults.items():
            setattr(self, key, value)
        
        # Initialize data containers (will be filled when generate_data is called)
        self.X = None
        self.Y = None
        self.results_array = None
        self.t_values = None
    
    def subsidence_bowl(self, x, y, t):
        """
        Calculate subsidence bowl values for given coordinates and time.
        
        This method uses the instance attributes for all physical parameters,
        making the method call much simpler than the original function.
        
        Args:
            x (np.ndarray): 2D array of x-coordinates
            y (np.ndarray): 2D array of y-coordinates  
            t (float): Current time point
            
        Returns:
            np.ndarray: 2D array of subsidence values
        """
        # Calculate squared distance from the subsidence center
        r_squared = (x - self.x_center)**2 + (y - self.y_center)**2
        
        # Calculate base subsidence (grows over time)
        base_subsidence = -self.A * np.exp(-r_squared / (2 * self.sigma**2)) * (1 + self.rate * t)
        
        # Calculate seasonal fluctuation
        seasonal_fluctuation = (self.S * np.exp(-r_squared / (2 * self.sigma_fluc**2)) * 
                              np.sin(2 * np.pi * self.f * t))
        
        # Return total subsidence
        return base_subsidence + seasonal_fluctuation
    
    def generate_data(self):
        """
        Generate the complete 3D subsidence dataset over time.
        
        This method creates the spatial and temporal grids, then calculates
        subsidence for each time step. Results are stored as instance attributes.
        """
        # Create spatial coordinate arrays
        x_coords = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y_coords = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        
        # Create 2D coordinate grids
        self.X, self.Y = np.meshgrid(x_coords, y_coords)
        
        # Create time array
        self.t_values = np.linspace(self.t_range[0], self.t_range[1], self.time_steps)
        
        # Initialize the results array (time, y, x)
        self.results_array = np.zeros((self.time_steps, self.resolution, self.resolution))
        
        # Calculate subsidence for each time step
        print(f"Generating subsidence data for {self.time_steps} time steps...")
        for i, t in enumerate(self.t_values):
            self.results_array[i, :, :] = self.subsidence_bowl(self.X, self.Y, t)
        
        print("Data generation complete!")
        return self.X, self.Y, self.results_array
    
    def get_indices_from_coords(self, x_target, y_target):
        """
        Find grid indices for given real-world coordinates.
        
        Args:
            x_target (float): Target x-coordinate
            y_target (float): Target y-coordinate
            
        Returns:
            tuple: (row_index, col_index) of closest grid point
            
        Raises:
            ValueError: If grids haven't been generated yet
        """
        if self.X is None or self.Y is None:
            raise ValueError("Grids not generated yet. Call generate_data() first.")
        
        # Extract 1D coordinate arrays from the 2D grids
        x_axis_coords = self.X[0, :]  # First row contains all x-values
        y_axis_coords = self.Y[:, 0]  # First column contains all y-values
        
        # Find closest grid points
        x_diff = np.abs(x_axis_coords - x_target)
        y_diff = np.abs(y_axis_coords - y_target)
        
        col_index = np.argmin(x_diff)
        row_index = np.argmin(y_diff)
        
        return (row_index, col_index)
    
    def save_to_netcdf(self, filepath, data_var_name="subsidence", 
                       data_var_attrs=None, global_attrs=None):
        """
        Save the simulation results to a NetCDF file.
        
        Args:
            filepath (str): Path for the output .nc file
            data_var_name (str): Name for the data variable in the file
            data_var_attrs (dict, optional): Attributes for the data variable
            global_attrs (dict, optional): Global file attributes
            
        Raises:
            ValueError: If no data has been generated yet
        """
        if self.results_array is None:
            raise ValueError("No data to save. Call generate_data() first.")
        
        # Set default attributes if none provided
        if data_var_attrs is None:
            data_var_attrs = {
                'units': 'L',
                'long_name': 'Land subsidence',
                'description': 'Simulated ground subsidence with seasonal variations'
            }
        
        if global_attrs is None:
            global_attrs = {
                'title': 'Subsidence Simulation',
                'description': 'Simulated subsidence over time and space',
                'center_x': self.x_center,
                'center_y': self.y_center,
                'max_amplitude': self.A,
                'subsidence_rate': self.rate,
                'subsidence_size': self.sigma,
                'fluc_ampitude': self.S,
                'fluc_spatialsize': self.sigma_fluc,
                'freq':self.f
            }
        
        # Create coordinate dictionaries
        coords = {
            'time': self.t_values,
            'y': self.Y[:, 0],  # y-coordinates from first column
            'x': self.X[0, :]   # x-coordinates from first row
        }
        
        # Define dimension order
        dims = ('time', 'y', 'x')
        
        try:
            # Create xarray DataArray
            xr_data_array = xr.DataArray(
                data=self.results_array,
                coords=coords,
                dims=dims,
                name=data_var_name,
                attrs=data_var_attrs
            )
            
            # Create Dataset and add global attributes
            dataset = xr_data_array.to_dataset(name=data_var_name)
            dataset.attrs = global_attrs
            
            # Save to file
            print(f"Saving data to '{filepath}'...")
            dataset.to_netcdf(filepath)
            print("Successfully saved file.")
            
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def plot_output(self, figsize=(10, 12), cmap="turbo_r", 
                        elev=30, azim=120, alpha=0.9):
        """
        Create a 3D plot of the final subsidence state.
        
        Args:
            figsize (tuple): Figure size in inches
            cmap (str): Colormap name
            elev (float): Elevation angle for 3D view
            azim (float): Azimuth angle for 3D view
            alpha (float): Surface transparency
        """
        if self.results_array is None:
            raise ValueError("No data to plot. Call generate_data() first.")
        
        # Create the figure and 3D axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        
        # Plot the surface using the final time step
        surface = ax.plot_surface(
            self.X, self.Y, self.results_array[-1], 
            cmap=cmap, edgecolor="none", alpha=alpha
        )
        
        # Set labels and formatting
        ax.set_xlabel("X", fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel("Y", fontsize=16, fontweight='bold', labelpad=10)
        ax.set_zlabel("Z (Subsidence)", fontsize=16, fontweight='bold', labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=12)
        
        # Set axis limits based on simulation parameters
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_zlim(-self.A * 2.5, 0.5)
        
        # Add colorbar
        cbar = fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.05)
        cbar.ax.tick_params(labelsize=14)
        surface.set_clim(np.min(self.results_array), np.max(self.results_array))
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        plt.show()
        return fig, ax

    def plot_interactive(self, figsize=(9, 16)):
        """
        Create an interactive 3D plot with time slider and autoplay.
        
        Args:
            figsize (tuple): Figure size in inches
        """
        if self.results_array is None:
            raise ValueError("No data to plot. Call generate_data() first.")
        
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button, Slider
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Interactive Subsidence Animation", fontsize=20, fontweight="bold")
        
        # Initial surface plot
        Z_initial = self.results_array[0]
        surface = ax.plot_surface(self.X, self.Y, Z_initial, cmap="turbo_r", edgecolor="none", alpha=0.9)
        
        # Set labels and limits
        ax.set_xlabel("X", fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel("Y", fontsize=16, fontweight='bold', labelpad=10)
        ax.set_zlabel("Z (Subsidence)", fontsize=16, fontweight='bold', labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=12)
        
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_zlim(-self.A * 2.5, 0.5)
        ax.view_init(elev=5, azim=120)
        
        # Fixed colorbar
        cbar = fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.05)
        cbar.ax.tick_params(labelsize=14)
        surface.set_clim(np.min(self.results_array), np.max(self.results_array))
        
        # Time slider
        ax_slider = plt.axes([0.4, 0.02, 0.3, 0.03], facecolor="lightgoldenrodyellow")
        time_slider = Slider(ax_slider, "Time", 0, self.time_steps-1, valinit=0, valstep=1)
        time_slider.label.set_fontsize(16)
        
        # Time step slider for autoplay speed
        ax_step_slider = plt.axes([0.4, 0.07, 0.3, 0.03], facecolor="lightblue")
        step_slider = Slider(ax_step_slider, "Speed", 1, 10, valinit=1, valstep=1)
        step_slider.label.set_fontsize(16)
        
        # Surface container for updates
        container = {"surface": surface}
        
        def update_surface(val):
            time_idx = int(time_slider.val)
            Z = self.results_array[time_idx]
            container["surface"].remove()
            container["surface"] = ax.plot_surface(self.X, self.Y, Z, cmap="turbo_r", edgecolor="none", alpha=0.9)
            container["surface"].set_clim(np.min(self.results_array), np.max(self.results_array))
            fig.canvas.draw_idle()
        
        time_slider.on_changed(update_surface)
        
        # Play/Pause button
        ax_play = plt.axes([0.8, 0.02, 0.1, 0.04], facecolor="lightgreen")
        play_button = Button(ax_play, "Play/Pause", color="lightgreen", hovercolor="lime")
        
        # Animation variables
        anim = None
        is_playing = [False]
        time_step = [1]
        
        def autoplay(event):    
            nonlocal anim
            
            def animate(frame):
                if is_playing[0]:
                    current_idx = int(time_slider.val)
                    next_idx = current_idx + time_step[0]
                    if next_idx >= self.time_steps:
                        next_idx = 0
                    time_slider.set_val(next_idx)
            
            if is_playing[0]:
                if anim:
                    anim.event_source.stop()
                is_playing[0] = False
            else:
                if anim is None:
                    anim = FuncAnimation(fig, animate, interval=200)
                else:
                    anim.event_source.start()
                is_playing[0] = True
        
        def update_time_step(val):
            time_step[0] = int(step_slider.val)
        
        play_button.on_clicked(autoplay)
        step_slider.on_changed(update_time_step)
        
        plt.show()
        return fig, ax
    
    def get_summary(self):
        """
        Print a summary of the simulation parameters and results.
        """
        print("=== Subsidence Simulation Summary ===")
        print(f"Spatial domain: X={self.x_range}, Y={self.y_range}")
        print(f"Temporal domain: {self.t_range}")
        print(f"Resolution: {self.resolution} x {self.resolution} grid points")
        print(f"Time steps: {self.time_steps}")
        print(f"Subsidence center: ({self.x_center}, {self.y_center})")
        print(f"Maximum amplitude: {self.A}")
        print(f"Subsidence rate: {self.rate}")
        
        if self.results_array is not None:
            print(f"\n=== Generated Data ===")
            print(f"Data shape: {self.results_array.shape}")
            print(f"Minimum subsidence: {np.min(self.results_array):.4f}")
            print(f"Maximum subsidence: {np.max(self.results_array):.4f}")
        else:
            print("\nNo data generated yet. Call generate_data() first.")


    @staticmethod
    def load_from_netcdf(filepath):
        """
        Load simulation data from a NetCDF file and return arrays.
        
        This function reads a previously saved NetCDF file and extracts
        the coordinate grids and subsidence data.
        
        Args:
            filepath (str): Path to the NetCDF file
            
        Returns:
            tuple: (X_grid, Y_grid, results_array, time_values) where:
                - X_grid: 2D array of x-coordinates
                - Y_grid: 2D array of y-coordinates  
                - results_array: 3D array of subsidence values (time, y, x)
                - time_values: 1D array of time coordinates
                
        Raises:
            FileNotFoundError: If NetCDF file doesn't exist
        """
        try:
            # Load the dataset
            dataset = xr.open_dataset(filepath)
            
            # Extract coordinate arrays
            x_coords = dataset.coords['x'].values
            y_coords = dataset.coords['y'].values
            time_values = dataset.coords['time'].values
            
            # Create coordinate grids
            X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
            
            # Extract the subsidence data (assuming variable name is 'subsidence')
            data_var_name = list(dataset.data_vars.keys())[0]  # Get first data variable
            results_array = dataset[data_var_name].values
            
            print(f"Successfully loaded data from: {filepath}")
            print(f"Data shape: {results_array.shape}")
            print(f"Time range: {time_values[0]:.2f} to {time_values[-1]:.2f}")
            
            # Close the dataset
            dataset.close()
            
            return X_grid, Y_grid, results_array, time_values
            
        except FileNotFoundError:
            raise FileNotFoundError(f"NetCDF file not found: {filepath}")
        except Exception as e:
            print(f"Error loading NetCDF file: {e}")
            return None, None, None, None