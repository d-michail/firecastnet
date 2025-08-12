import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.widgets import Slider
from cartopy.util import add_cyclic_point


def globe_data_visualization(data, variable, start_time_idx = 0, end_time_idx=None):
    """
    Create a global map visualization of the data with an interactive time slider
    when a range of time indices is provided.

    Parameters
    ----------
    data : xarray.DataArray
        The data to visualize
    variable : str
        Variable name
    start_time_idx : int, optional
        Starting time index
    end_time_idx : int, optional
        Ending time index. If provided, creates an interactive time slider.
    """
    if len(data.dims) != 3:
        raise ValueError("Data must have 3 dimensions: (latitude, longitude, time)")

    if not hasattr(data, 'time') or len(data.time.values) == 0:
        raise ValueError("Data must have a 'time' coordinate")

    # Get time values for labels
    time_values = data.time.values

    # Create the figure with Cartopy projection
    fig = plt.figure(figsize=(12, 10))  # Taller figure to accommodate slider

    # Create main map axis
    map_ax = fig.add_axes([0.05, 0.2, 0.9, 0.7], projection=ccrs.Robinson())

    # Add coastlines and gridlines
    map_ax.coastlines()
    map_ax.gridlines(draw_labels=True, alpha=0.45)

    # Initial time index to display
    current_time_idx = start_time_idx

    # Determine an appropriate title base
    if hasattr(data, 'long_name'):
        title_base = data.long_name
    else:
        title_base = variable

    # Function to update the plot for a specific time index
    def plot_time_step(time_idx):
        # Clear previous plot
        map_ax.clear()
        map_ax.coastlines()
        map_ax.gridlines(draw_labels=True, alpha=0.45)

        # Extract the data for the selected time
        data_array = data.isel(time=time_idx).values

        # If we have NaN values, use a masked array
        if np.isnan(data_array).any():
            data_array = np.ma.masked_invalid(data_array)

        # Get the min and max for a symmetric colorbar if data has positive and negative values
        if np.nanmin(data_array) < 0 and np.nanmax(data_array) > 0:
            vmax = max(abs(np.nanmin(data_array)), abs(np.nanmax(data_array)))
            vmin = -vmax
            cmap = cm.coolwarm
        else:
            vmin = np.nanmin(data_array)
            vmax = np.nanmax(data_array)
            cmap = cm.viridis

        # Add cyclic point to avoid a gap at the dateline
        cyclic_data, cyclic_lons = add_cyclic_point(data_array, coord=data.longitude.values)

        # Create the contourf plot
        contour = map_ax.contourf(
            cyclic_lons, data.latitude.values, cyclic_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=20
        )

        # Update title with time information
        time_str = str(time_values[time_idx])
        map_ax.set_title(f"{title_base} - {time_str}")

        return contour

    # Initial plot
    contour = plot_time_step(current_time_idx)

    # Add colorbar
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.04])
    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
    if hasattr(data, 'units'):
        cbar.set_label(f"{variable} {data.units}")
    else:
        cbar.set_label(variable)

    # If we have multiple time steps, add a slider
    if end_time_idx is not None:
        # Ensure end_time_idx is valid
        end_time_idx = min(end_time_idx, len(time_values))
        end_time_idx = max(end_time_idx, start_time_idx + 1)

        # Create slider axis
        slider_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])

        # Create the slider
        time_slider = Slider(
            slider_ax, 'Time',
            start_time_idx,
            end_time_idx - 1,
            valinit=current_time_idx,
            valstep=1
        )

        # Format slider tick labels with time values
        slider_positions = np.linspace(start_time_idx, end_time_idx-1, 
                                      min(5, end_time_idx-start_time_idx))
        slider_positions = [int(pos) for pos in slider_positions]

        slider_labels = [str(time_values[idx]).split('T')[0] for idx in slider_positions]
        slider_ax.set_xticks(slider_positions)
        slider_ax.set_xticklabels(slider_labels)

        # Update function for slider
        def update_time(val):
            idx = int(val)
            # Update the contour plot
            _ = plot_time_step(idx)
            # No need to update colorbar as limits are fixed
            fig.canvas.draw_idle()

        # Connect the update function to the slider
        time_slider.on_changed(update_time)

        # Add a description of the slider
        fig.text(0.5, 0.15, "Drag slider to change time", 
                ha='center', fontsize=10, style='italic')

    plt.show()

if __name__ == '__main__':
    from src.data.cube_data import extract_cube_data

    ds = extract_cube_data()
    print(ds)
    var = 'vpd'
    # Visualize a single time step
    globe_data_visualization(ds[var], var)

    # Visualize a range of time steps with a slider
    globe_data_visualization(ds[var], var, 0, 10)