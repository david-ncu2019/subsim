import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import xarray as xr  # Import the xarray library

def subsidence_bowl(x, y, t, x_center=0, y_center=0, A=1, sigma=2, rate=0.1, S=0.2, f=1.0, sigma_fluc=1.0):
    """
    Calculates the subsidence bowl value for given coordinates and time.
    The center of the subsidence can be specified.

    Args:
        x (np.ndarray): A 2D array of x-coordinates.
        y (np.ndarray): A 2D array of y-coordinates.
        t (float): The current time point.
        x_center (float): The x-coordinate of the subsidence center.
        y_center (float): The y-coordinate of the subsidence center.
        A (float): Maximum depth of the subsidence.
        sigma (float): Controls the spread of the subsidence bowl.
        rate (float): The long-term linear subsidence rate.
        S (float): Amplitude of the seasonal fluctuations.
        f (float): Frequency of seasonal fluctuations (in cycles per unit time).
        sigma_fluc (float): Controls the spatial extent of seasonal fluctuations.

    Returns:
        np.ndarray: A 2D array of Z-values (subsidence) for the given time `t`.
    """
    # 1. Define r_squared as the squared distance from the specified center.
    r_squared = (x - x_center)**2 + (y - y_center)**2

    # 2. Calculate the base, long-term subsidence.
    base_subsidence = -A * np.exp(-r_squared / (2 * sigma**2)) * (1 + rate * t)

    # 3. Calculate the seasonal fluctuation.
    seasonal_fluctuation = S * np.exp(-r_squared / (2 * sigma_fluc**2)) * np.sin(2 * np.pi * f * t)

    # 4. Return the total subsidence.
    return base_subsidence + seasonal_fluctuation

def generate_subsidence_data(
    x_range=(-5, 5), y_range=(-5, 5), t_range=(0, 10),
    resolution=30, time_steps=100,
    x_center=0, y_center=0,
    A=1, sigma=2, rate=0.1, S=0.2, f=0.5, sigma_fluc=1.0
):
    """
    Generates a 3D NumPy array of the subsidence bowl over time.
    The calculation area is defined by x_range and y_range.
    """
    # 1. Create spatial and temporal grids based on input ranges.
    x_coords = np.linspace(x_range[0], x_range[1], resolution)
    y_coords = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    t_values = np.linspace(t_range[0], t_range[1], time_steps)

    # 2. Initialize results array.
    results_array = np.zeros((time_steps, resolution, resolution))

    # 3. Loop through time and calculate subsidence.
    for i, t in enumerate(t_values):
        z_slice = subsidence_bowl(X, Y, t, x_center, y_center, A, sigma, rate, S, f, sigma_fluc)
        results_array[i, :, :] = z_slice

    # 4. Return the grids and the results.
    return X, Y, results_array

def get_indices_from_coords(x_target, y_target, X_grid, Y_grid):
    """
    Finds the grid indices (row, column) for a given real-world coordinate pair.

    This function "reprojects" a coordinate by finding the point in the grid
    that is closest to the target coordinate.

    Args:
        x_target (float): The target x-coordinate.
        y_target (float): The target y-coordinate.
        X_grid (np.ndarray): The 2D NumPy array of X coordinates for the grid.
        Y_grid (np.ndarray): The 2D NumPy array of Y coordinates for the grid.

    Returns:
        tuple: A tuple containing the (row_index, column_index) corresponding
               to the closest point in the grid.
    """
    # 1. Extract the 1D axis arrays from the full 2D grids.
    #    The x-axis values are consistent across any row.
    #    The y-axis values are consistent down any column.
    x_axis_coords = X_grid[0, :]
    y_axis_coords = Y_grid[:, 0]

    # 2. Calculate the absolute difference between the target coordinate and
    #    every point on each axis.
    x_diff = np.abs(x_axis_coords - x_target)
    y_diff = np.abs(y_axis_coords - y_target)

    # 3. Use np.argmin() to find the index of the minimum difference.
    #    This index corresponds to the position of the closest grid point.
    col_index = np.argmin(x_diff)
    row_index = np.argmin(y_diff)

    return (row_index, col_index)


def save_to_netcdf(
    filepath,
    data_array,
    data_var_name,
    coords,
    dims,
    data_var_attrs=None,
    global_attrs=None
):
    """
    Saves a NumPy array to a user-friendly NetCDF file using xarray.

    This function is universal and not tied to any specific data structure.

    Args:
        filepath (str): The full path for the output .nc file.
        data_array (np.ndarray): The NumPy array containing the data to save.
        data_var_name (str): The name for the data variable inside the NetCDF file
                             (e.g., "temperature", "subsidence").
        coords (dict): A dictionary mapping dimension names to their 1D coordinate arrays.
                       Example: {'time': time_array, 'lat': lat_array, 'lon': lon_array}
        dims (tuple or list): A tuple or list of dimension names in the same order
                              as the data_array's axes. Example: ('time', 'lat', 'lon')
        data_var_attrs (dict, optional): A dictionary of attributes for the data
                                         variable (e.g., {'units': 'Celsius'}). Defaults to None.
        global_attrs (dict, optional): A dictionary of global attributes for the file
                                       (e.g., {'description': 'Weather data'}). Defaults to None.
    """
    if data_var_attrs is None:
        data_var_attrs = {}
    if global_attrs is None:
        global_attrs = {}

    try:
        # 1. Create an xarray.DataArray
        # This is the core step that bundles the raw data with its labels.
        xr_data_array = xr.DataArray(
            data=data_array,
            coords=coords,
            dims=dims,
            name=data_var_name,
            attrs=data_var_attrs
        )

        # 2. Create an xarray.Dataset
        # A Dataset is a container for one or more DataArrays.
        # We assign the global attributes to the dataset.
        dataset = xr_data_array.to_dataset(name=data_var_name)
        dataset.attrs = global_attrs

        # 3. Save the Dataset to a NetCDF file
        print(f"Saving data to '{filepath}'...")
        dataset.to_netcdf(filepath)
        print("Successfully saved file.")

    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


# --- Plotting Section ---
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(111, projection="3d")
# ax.set_title("Subsidence with Custom Boundaries", fontsize=20, fontweight="bold")

# Plot the final time step.
surface = ax.plot_surface(X, Y, Z[-1], cmap="turbo_r", edgecolor="none", alpha=0.9)
ax.set_xlabel("X", fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel("Y", fontsize=16, fontweight='bold', labelpad=10)
ax.set_zlabel("Z", fontsize=16, fontweight='bold', labelpad=10)
ax.tick_params(axis="both", which="major", labelsize=12)

# **FIX:** Set axis limits dynamically based on the input parameters.
ax.set_xlim(custom_sim_params['x_range'])
ax.set_ylim(custom_sim_params['y_range'])
ax.set_zlim(-custom_sim_params['A'] * 2.5, 0.5)

# Add color bar.
cbar = fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.05)
cbar.ax.tick_params(labelsize=14)
# Adjust the color bar limit to match the data's range for better visualization.
surface.set_clim(np.min(Z), np.max(Z))

# Set view angle.
ax.view_init(elev=30, azim=120)

plt.show()
