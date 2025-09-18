import math
import shapely
import numpy as np
from numpy.linalg import norm
from shapely import MultiPolygon, Polygon
from typing import List, Tuple, Union
from .PolygonStructure import PolygonStructure

MAX_LON = 180.0  
GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0

# GENERAL UTILS
def order(order, thing):
    return f"order_{order}_{thing}"

def load_yaml(filename: str):
    import yaml
    # Load the config from the specified path
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config

def generate_icosphere_file_code(polygon_structures: List[PolygonStructure], ref_order: int) -> str:
    structure_code = "icosphere_s"+str(ref_order)
    for p in polygon_structures:
        if p.target_code == "global":
            continue
        structure_code += f"_{p.target_code}_{p.refinement_order}{p.refinement_type[0]}"
    return structure_code

def gzip_file(filename: str):
    """ Compress a file using gzip and return the compressed file name.
    
    Args:
        filename (str): The name of the file to compress.
        
    Returns:
        str: The name of the compressed file.
    """
    import gzip
    import shutil
    
    compressed_filename = filename + '.gz'
    with open(filename, 'rb') as f_in:
        with gzip.open(compressed_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def mesh_to_dict(meshes: List) -> dict:
    """
    Convert an array of meshes to a dictionary format similar to generate_icosphere output.
    
    Args:
        meshes (List): List of PyMesh mesh objects representing different refinement orders.
        
    Returns:
        dict: Dictionary containing mesh data with keys for vertices, faces, and face centroids
              for each refinement order.
    """
    icospheres_dict = { "vertices":[], "faces":[] }

    for o, mesh in enumerate(meshes):
        if not mesh:
            continue
        # Add vertices for this order
        icospheres_dict[order(o, "vertices")] = mesh.vertices

        # Add faces for this order
        icospheres_dict[order(o, "faces")] = mesh.faces

        # Calculate and add face centroids for this order
        mesh.add_attribute("face_centroid")
        icospheres_dict[order(o, "face_centroid")] = mesh.get_face_attribute("face_centroid")

    # Convert numpy arrays to lists for JSON serialization
    result_dict = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value) 
        for key, value in icospheres_dict.items()
    }
    
    return result_dict

# CONVERSION UTILS
def to_cartesian(lat_lon: np.ndarray) -> np.ndarray:
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

def to_lat_lon(vertices: np.ndarray) -> np.ndarray:
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

def to_sphere(vertices, radius=1, center=(0, 0, 0)) -> np.ndarray:
    """Convert vertices to spherical coordinates."""
    length = norm(vertices, axis=1).reshape((-1, 1))
    return vertices / length * radius + center


def get_icosahedron_geometry() -> Tuple[np.ndarray, np.ndarray]:
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


# POLYGON UTILS
def flatten_polygons(polygons: List[Union[Polygon, MultiPolygon]]) -> List[Polygon]:
    """
    Flatten a list of polygons or multipolygons into a single list of polygons.
    
    :param polygons: List of polygons or multipolygons.
    
    :return: Flattened list of polygons.
    """
    flattened = []
    for polygon in polygons:
        if isinstance(polygon, MultiPolygon):
            flattened.extend(list(polygon.geoms))
        elif isinstance(polygon, Polygon):
            flattened.append(polygon)
        else:
            raise ValueError("Input must be a list of shapely Polygon or MultiPolygon objects.")
    return flattened

def undo_antimeridian_wrap(poly: Polygon) -> Polygon:
    # Undo the antimeridian wrap by adjusting longitudes that cross the antimeridian.
    coords = list(poly.exterior.coords)
    coords_len = len(coords)
    i = 0
    while i < coords_len:
        c1 = coords[i]
        c2 = coords[(i + 1) % coords_len]
        j = 0
        if abs(c1[1] - c2[1]) > 180:
            sign = -1 if c1[1] < 0 else 1
            for j in range(i+1, coords_len):
                if sign * coords[j][1] > 0:
                    break
                coords[j] = (coords[j][0], coords[j][1] + sign * 360)
        i += j + 1
    return shapely.Polygon(coords)

def polygon_crosses_antimeridian(polygon: Union[Polygon, List[Tuple[float, float]]]) -> bool:
    """Check if polygon crosses the antimeridian (180° longitude)."""
    if isinstance(polygon, Polygon):
        # For Shapely Polygon with (lat, lon) input format:
        # X coordinates store latitude, Y coordinates store longitude
        lons = list(polygon.exterior.coords.xy[1])  # Y coordinates = longitude
    elif isinstance(polygon, list):
        # For coordinate lists in (lat, lon) format, longitude is at index 1
        lons = [coord[1] for coord in polygon]
    else:
        raise ValueError("Input must be a shapely Polygon or a list of (lat, lon) tuples.")
    
    # Check if the lons exceed the maximum longitude threshold
    if max(lons) > MAX_LON or min(lons) < -MAX_LON:
        return True
    
    # Check if any edge crosses the antimeridian
    for i in range(len(lons)):
        lon1 = lons[i]
        lon2 = lons[(i + 1) % len(lons)]
        # If the longitude difference is greater than 180°, it likely crosses the antimeridian
        if abs(lon1 - lon2) > MAX_LON:
            return True
    
    return False