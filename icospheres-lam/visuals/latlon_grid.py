import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.utils.graph_utils import latlon2xyz_numpy

def visualize_latlon_grid_matplotlib(lat_lon_grid, sample_factor=10, marker_size=2):
    """
    Visualize a latitude-longitude grid on a sphere using Matplotlib.
    
    Parameters
    ----------
    lat_lon_grid : torch.Tensor
        A grid of latitude-longitude coordinates with shape [lat, lon, 2]
    sample_factor : int, optional
        Factor to downsample the grid for better visualization, by default 10
    marker_size : int, optional
        Size of markers in the visualization, by default 2
        
    Returns
    -------
    None
        Displays a matplotlib 3D visualization
    """
    # Convert lat-lon to 3D cartesian coordinates
    
    # Convert to numpy
    if torch.is_tensor(lat_lon_grid):
        lat_lon_grid = lat_lon_grid.numpy()
    
    # Sample the grid to avoid too many points
    sampled_grid = lat_lon_grid[::sample_factor, ::sample_factor]
    
    # Convert to cartesian coordinates
    xyz_coords = latlon2xyz_numpy(sampled_grid)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = xyz_coords[..., 0].flatten()
    y = xyz_coords[..., 1].flatten()
    z = xyz_coords[..., 2].flatten()
    
    # Plot grid points
    grid_scatter = ax.scatter(x, y, z, 
                             c=lat_lon_grid[::sample_factor, ::sample_factor, 0].flatten(),
                             cmap='viridis',
                             s=marker_size,
                             alpha=0.8)
    
    # Add colorbar for latitude
    cbar = plt.colorbar(grid_scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Latitude')
    
    # Draw wireframe sphere for reference
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = 0.98 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 0.98 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 0.98 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.15)
    
    # Add meridians and parallels
    # Meridians (lines of longitude)
    for lon in np.linspace(-180, 180, 13):  # Every 30 degrees
        if lon == -180 or lon == 180:
            continue  # Skip the repeated line
        lon_rad = np.radians(lon)
        lat_points = np.linspace(-90, 90, 100)
        lat_rad = np.radians(lat_points)
        x_merid = np.cos(lat_rad) * np.cos(lon_rad)
        y_merid = np.cos(lat_rad) * np.sin(lon_rad)
        z_merid = np.sin(lat_rad)
        ax.plot(x_merid, y_merid, z_merid, 'k-', alpha=0.2, linewidth=0.5)
    
    # Parallels (lines of latitude)
    for lat in np.linspace(-90, 90, 13):  # Every 15 degrees
        if lat == -90 or lat == 90:
            continue  # Skip the poles
        lat_rad = np.radians(lat)
        lon_points = np.linspace(-180, 180, 100)
        lon_rad = np.radians(lon_points)
        x_paral = np.cos(lat_rad) * np.cos(lon_rad)
        y_paral = np.cos(lat_rad) * np.sin(lon_rad)
        z_paral = np.sin(lat_rad) * np.ones_like(lon_rad)
        ax.plot(x_paral, y_paral, z_paral, 'k-', alpha=0.2, linewidth=0.5)
    
    # Axes labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Global Latitude-Longitude Grid Visualization')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set axis limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    
    # Add legend for reference
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
              markersize=5, label='Grid Points (colored by latitude)'),
        Line2D([0], [0], color='black', alpha=0.2, label='Meridians/Parallels')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Print some information about the grid
    print(f"Grid Shape: {lat_lon_grid.shape}")
    print(f"Latitude Range: {lat_lon_grid[..., 0].min():.2f} to {lat_lon_grid[..., 0].max():.2f}")
    print(f"Longitude Range: {lat_lon_grid[..., 1].min():.2f} to {lat_lon_grid[..., 1].max():.2f}")
    print(f"Resolution: {np.abs(lat_lon_grid[1, 0, 0] - lat_lon_grid[0, 0, 0]):.3f}° (latitude), " 
          f"{np.abs(lat_lon_grid[0, 1, 1] - lat_lon_grid[0, 0, 1]):.3f}° (longitude)")