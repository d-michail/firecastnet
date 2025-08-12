import xarray
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import colors
from matplotlib.widgets import Slider, Button
from src.data.cube_data import GlobeBounds as GB, extract_cube_data



class GlobeGraphDataVisualizer:
    """
    Interactive 3D visualization class that maps zarr dataset values 
    to latitude-longitude points on a 3D globe.
    """
    def __init__(self,
                 lat_lon_points: np.ndarray,
                 data: xarray.DataArray,
                 colormap: str = 'coolwarm',
                 point_size: int = 10,
                 fig_size: Tuple[int, int] = (12, 10),
                 title: str = None):
        """
        Initialize the 3D globe data visualizer.

        Parameters
        ----------
        lat_lon_points : np.ndarray
            Array of lat-lon points with shape (..., 2) where the last dimension
            contains [latitude, longitude] in degrees
        data : xarray.DataArray
            Zarr dataset containing the data of a specific variable to visualize
        colormap : str, optional
            Matplotlib colormap name, by default 'coolwarm'
        point_size : int, optional
            Size of data points, by default 10
        fig_size : tuple, optional
            Figure size in inches, by default (12, 10)
        title : str, optional
            Custom title for the plot, by default None
        """
        # Store input parameters
        self.lat_lon_points = lat_lon_points
        self.data = data
        self.colormap = colormap
        self.point_size = point_size
        self.fig_size = fig_size
        self.custom_title = title
        self.time_idx = 0
        self.precision = {
            "lat": int((lat_lon_points[1][0][0] - lat_lon_points[0][0][0]) / GB.PRECISION),
            "lon": int((lat_lon_points[0][1][1] - lat_lon_points[0][0][1]) / GB.PRECISION),
        }
        self.latlon_size = {
            "lat": data.shape[1] // self.precision['lat'],
            "lon": data.shape[2] // self.precision['lon']
        }

        # Create visualization attributes
        self.fig = None
        self.ax = None
        self.scatter = None
        self.colorbar = None
        self.point_values = None
        self.xyz_coords = None
        self.grid_lines = []
        self.sphere = None

        # Initialize data
        self._convert_latlon_to_xyz()
        self._map_data_to_points()
    
    def _get_title(self):
        """Get the title for the plot."""
        if self.custom_title:
            return self.custom_title
        if hasattr(self.data, 'long_name'):
            return self.data.long_name

    def _map_data_to_points(self):
        """Map data from the zarr grid to lat-lon points."""
        # Flatten the lat-lon points for processing
        flat_latlon = self.lat_lon_points.reshape(-1, 2)

        # Prepare container for mapped values
        self.point_values = np.full(flat_latlon.shape[0], np.nan)

        # Get coordinates from the dataset
        try:
            tdata = np.array(self.data.values[self.time_idx])
    
            for i in range(self.latlon_size['lat']):
                for j in range(self.latlon_size['lon']):
                    lat_idx = i * self.precision['lat']
                    lon_idx = j * self.precision['lon']
                    self.point_values[i * self.latlon_size['lon'] + j] = tdata[lat_idx, lon_idx]
        except Exception as e:
            print(f"Error mapping data to points: {e}")
            # Fill with placeholder values if mapping fails
            self.point_values = np.linspace(0, 1, flat_latlon.shape[0])

    def _convert_latlon_to_xyz(self):
        """Convert latitude-longitude coordinates to 3D Cartesian (x, y, z)."""
        # Flatten the lat-lon points
        flat_latlon = self.lat_lon_points.reshape(-1, 2)

        # Convert from degrees to radians
        lat_rad = np.radians(flat_latlon[:, 0])
        lon_rad = np.radians(flat_latlon[:, 1])

        # Convert to 3D Cartesian coordinates (unit sphere)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        # Store the coordinates
        self.xyz_coords = np.column_stack((x, y, z))

        return self.xyz_coords

    def create_visualization(self):
        """Create the interactive 3D visualization."""
        # Create the figure and 3D axis
        self.fig = plt.figure(figsize=self.fig_size)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Add a reference wireframe sphere
        self._add_reference_sphere(alpha=0.25)

        # Add coordinate grid lines
        self._add_grid_lines(alpha=0.45)

        # Set plot title
        self._update_title()

        # Plot the data points
        self._update_scatter()

        # Add a colorbar
        self.colorbar = self.fig.colorbar(
            self.scatter, 
            ax=self.ax, 
            orientation='vertical', 
            pad=0.05, 
            shrink=0.8,
            label=f"{getattr(self.data, 'units', '')}"
        )

        # Add interactive widgets
        self._add_interaction_controls()

        # Configure axes
        self._configure_axes()

        return self.fig

    def _update_scatter(self):
        """Create or update the scatter plot."""
        # Determine color range
        vmin = np.nanmin(self.point_values)
        vmax = np.nanmax(self.point_values)

        # Create color normalization
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # If scatter exists, remove it first
        if self.scatter:
            self.scatter.remove()

        # Plot the data points
        self.scatter = self.ax.scatter(
            self.xyz_coords[:, 0], 
            self.xyz_coords[:, 1], 
            self.xyz_coords[:, 2],
            c=self.point_values,
            cmap=self.colormap,
            norm=norm,
            s=self.point_size,
            alpha=0.8,
            edgecolors='none'
        )

        return self.scatter

    def _update_title(self):
        """Update the plot title based on current data."""
        title = self._get_title()
        # Add time information
        if hasattr(self.data, 'time') and self.data.time is not None:
            time_str = str(self.data.time.values[self.time_idx])
            title = f"{title} - {time_str}"

        self.ax.set_title(title)

    def _add_reference_sphere(self, alpha=0.1):
        """Add a reference wireframe sphere."""
        # Create a wireframe sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = 0.98 * np.outer(np.cos(u), np.sin(v))
        y = 0.98 * np.outer(np.sin(u), np.sin(v))
        z = 0.98 * np.outer(np.ones(np.size(u)), np.cos(v))

        # Add the sphere with a light color
        self.sphere = self.ax.plot_surface(
            x, y, z, 
            color='skyblue', 
            alpha=alpha, 
            linewidth=0,
            shade=True
        )

    def _add_grid_lines(self, alpha=0.45):
        """Add latitude and longitude grid lines."""
        # Clear existing grid lines
        for line in self.grid_lines:
            if line in self.ax.lines:
                self.ax.lines.remove(line)
        self.grid_lines = []

        # Calculate radius
        radius = 1.0

        # Add longitude lines (meridians)
        for phi in np.linspace(0, 2*np.pi, 12, endpoint=False):
            x = radius * np.cos(phi) * np.sin(np.linspace(0, np.pi, 50))
            y = radius * np.sin(phi) * np.sin(np.linspace(0, np.pi, 50))
            z = radius * np.cos(np.linspace(0, np.pi, 50))
            line, = self.ax.plot(x, y, z, 'gray', alpha=alpha, linewidth=0.5)
            self.grid_lines.append(line)

        # Add latitude lines (parallels)
        for theta in np.linspace(0.2, np.pi-0.2, 6):
            x = radius * np.cos(np.linspace(0, 2*np.pi, 50)) * np.sin(theta)
            y = radius * np.sin(np.linspace(0, 2*np.pi, 50)) * np.sin(theta)
            z = radius * np.cos(theta) * np.ones(50)
            line, = self.ax.plot(x, y, z, 'gray', alpha=alpha, linewidth=0.5)
            self.grid_lines.append(line)

    def _configure_axes(self):
        """Configure the axes for better visualization."""
        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])

        # Set axis limits
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_zlim(-1.1, 1.1)

        # Remove axis ticks and labels for cleaner look
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])

        # Remove background panes
        # self.ax.xaxis.pane.fill = False
        # self.ax.yaxis.pane.fill = False
        # self.ax.zaxis.pane.fill = False

        # Set grid off
        # self.ax.grid(False)

        # Add text indicators for orientation
        self.ax.text(0, 0, 1.15, "N", fontsize=12, ha='center')
        self.ax.text(0, 0, -1.15, "S", fontsize=12, ha='center')
        self.ax.text(1.15, 0, 0, "E", fontsize=12, ha='center')
        self.ax.text(-1.15, 0, 0, "W", fontsize=12, ha='center')

    def _add_interaction_controls(self):
        """Add interactive controls for the visualization."""

        # Add control panel area for point size slider
        time_ax = plt.axes([0.25, 0.95, 0.3, 0.03])

        # Add a slider to change the data time index
        time_slider = Slider(
            time_ax, 'Time Index',
            0, len(self.data.time) - 1,
            valinit=0,
            valstep=1,
        )

        # Define function to update time index
        def update_time(val):
            idx = int(val)
            self.update_data_by_time(time_idx=idx)

        # Connect the slider to the update function
        time_slider.on_changed(update_time)

        # Add a button to toggle grid lines
        grid_button_ax = plt.axes([0.65, 0.95, 0.1, 0.03])
        grid_button = Button(grid_button_ax, 'Toggle Grid', color='lightgray')

        # Define function to toggle grid lines
        def toggle_grid(event):
            for line in self.grid_lines:
                visible = not line.get_visible()
                line.set_visible(visible)
            self.fig.canvas.draw_idle()

        # Connect the button to the function
        grid_button.on_clicked(toggle_grid)

        self.time_slider = time_slider
        self.grid_button = grid_button

    def show(self):
        """Display the visualization."""
        if self.fig is None:
            self.create_visualization()
        plt.show()

    def update_data_by_time(self, time_idx=None):
        """
        Update the visualization with a new time index.
    
        Parameters
        ----------
        time_idx : int, optional
            New time index, by default None (keep current)
        """
        # Update parameters
        if time_idx is not None:
            self.time_idx = time_idx
    
        # Map data to points
        self._map_data_to_points()
    
        # Update the scatter plot
        if self.scatter is not None:
            # Update the scatter plot with new data
            self._update_scatter()
            
            # Update title
            self._update_title()
    
            # Redraw the figure
            if self.fig is not None:
                self.fig.canvas.draw_idle()

    def save(self, filename):
        """
        Save the visualization to a file.

        Parameters
        ----------
        filename : str
            Path to save the figure
        """
        if self.fig is None:
            self.create_visualization()

        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {filename}")


# Example usage
if __name__ == "__main__":
    # Example lat-lon grid
    np.set_printoptions(suppress=True, precision=3)
    lat_res = 0.5
    lon_res = 0.75
    lats = np.arange(GB.MIN_LAT, GB.MAX_LAT, lat_res)
    lons = np.arange(GB.MIN_LON, GB.MAX_LON, lon_res)
    lat_lon_grid = np.stack(np.meshgrid(lats, lons, indexing='ij'), axis=-1)

    data = extract_cube_data('vpd', 500, 505)

    # print(data.shape)

    # Create and show visualization
    visualizer = GlobeGraphDataVisualizer(
        lat_lon_grid,
        data,
        colormap='inferno',
        point_size=10
    )

    # Show the interactive visualization
    visualizer.show()