import math
import numpy as np
import pandas as pd
from numpy.linalg import norm

ANTIMERIDIAN_THRESHOLD = 180.0  
GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0

def to_cartesian(lat_lon):
    """
    Convert latitude and longitude to 3D Cartesian coordinates.
    
    Args:
        lat_lon (np.ndarray): Array of shape (N, 2) representing the latitude and longitude in degrees.
        
    Returns:
        vertices (np.ndarray): Array of shape (N, 3) representing the vertices in 3D Cartesian coordinates.
    """
    lat_rad = np.radians(lat_lon[:, 0])
    lon_rad = np.radians(lat_lon[:, 1])
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    return np.column_stack((x, y, z))

def to_lat_lon(vertices):
    """
    Convert an array of 3D Cartesian coordinates to latitude and longitude.
    
    Parameters:
        vertices (np.ndarray): An (N, 3) array of 3D Cartesian coordinates (x, y, z).

    Returns:
        np.ndarray: An (N, 2) array of (latitude, longitude) in degrees.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    # Compute the hypotenuse on the x-y plane
    hyp = np.hypot(x, y)
    # Compute latitude and longitude in radians
    lat_rad = np.arctan2(z, hyp)
    lon_rad = np.arctan2(y, x)
    # Convert radians to degrees
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    # Stack into an (N, 2) array
    latlon = np.column_stack((lat_deg, lon_deg))
    return latlon

def to_sphere(vertices, radius=1, center=(0, 0, 0)):
    """Convert vertices to spherical coordinates."""
    length = norm(vertices, axis=1).reshape((-1, 1))
    return vertices / length * radius + center


def get_icosahedron_geometry():
    """Get the initial icosahedron vertices and faces."""
    r = GOLDEN_RATIO
    vertices = np.array([
        [-1.0,   r, 0.0], [ 1.0,   r, 0.0], [-1.0,  -r, 0.0], [ 1.0,  -r, 0.0],
        [0.0, -1.0,   r], [0.0,  1.0,   r], [0.0, -1.0,  -r], [0.0,  1.0,  -r],
        [  r, 0.0, -1.0], [  r, 0.0,  1.0], [ -r, 0.0, -1.0], [ -r, 0.0,  1.0],
    ], dtype=float)
    
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [5, 4, 9], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])
    
    return vertices, faces


def polygon_wraps_around_antimeridian(triangle_vertices_latlon):
    """Check if polygon crosses the antimeridian (180° longitude)."""
    lons = [lon for _, lon in triangle_vertices_latlon]
    max_diff = max(lons) - min(lons)
    return max_diff > ANTIMERIDIAN_THRESHOLD


def get_wkt(country_name, filename='src/structured_icospheres/csv/wkt.csv'):
    df = pd.read_csv(filename, encoding='latin1', keep_default_na=False)
    country = df[df['SU_A3'] == country_name]
    if country.empty:
        raise ValueError(f"Country {country_name} not found in the dataset.")
    return country.iloc[0]
