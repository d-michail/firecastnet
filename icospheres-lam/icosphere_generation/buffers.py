import math
import shapely
import shapely.wkt
from typing import Union
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform
from .utils import flatten_polygons, undo_antimeridian_wrap

def flip_coords(x, y):
    return y, x

def flip_transform(polygon):
    """
    Flips the coordinates of a Shapely Polygon from (lon, lat) to (lat, lon).
    
    Args:
        polygon (shapely.geometry.Polygon): The input polygon.
    
    Returns:
        shapely.geometry.Polygon: The polygon with flipped coordinates.
    """
    if isinstance(polygon, str):
        polygon = shapely.wkt.loads(polygon)
    if not isinstance(polygon, Polygon):
        raise ValueError("Input must be a shapely Polygon or WKT string representing a polygon.")
    return shapely_transform(flip_coords, polygon)

def buffer_polygon(polygon: Union[Polygon, MultiPolygon], buffer_factor: float, buffer_unit: str = "km") -> Union[Polygon, MultiPolygon]:
    """
    Buffers a Shapely Polygon by a given distance in kilometers or by percentage of its radius.

    The function transforms the polygon to an azimuthal equidistant projection
    centered at its centroid, performs the buffer operation in meters,
    and then transforms the result back to WGS84 (EPSG:4326).

    Args:
        polygon (Union[Polygon, MultiPolygon]): The input polygon,
                                                assumed to be in WGS84 (EPSG:4326).
        buffer_factor (float): The buffer distance in kilometers or percentage.
        buffer_unit (str): Either "km" for kilometers or "percent" for percentage buffering.

    Returns:
        Union[Polygon, MultiPolygon]: The buffered polygon(s) in WGS84 (EPSG:4326).
    """
    if isinstance(polygon, str):
        polygon = shapely.wkt.loads(polygon)
    
    if isinstance(polygon, MultiPolygon):
        # If input is a MultiPolygon, buffer each polygon individually
        buffered_polygons = flatten_polygons([buffer_polygon(p, buffer_factor, buffer_unit) for p in polygon.geoms])
        return MultiPolygon(buffered_polygons)
        
    if not isinstance(polygon, Polygon) or not polygon.is_valid:
        raise ValueError("Input must be a shapely Polygon or WKT string representing a polygon.")

    if buffer_unit not in ["km", "percent"]:
        raise ValueError("buffer_unit must be either 'km' or 'percent'")

    # Define the geographic CRS (WGS84)
    crs_wgs84 = CRS.from_epsg(4326)
    
    # Get the centroid of the polygon to center the projection
    centroid = polygon.centroid
    lon, lat = centroid.y, centroid.x
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise ValueError("Centroid coordinates are out of bounds for WGS84.")

    # Flip the polygon coordinates to (lon, lat) format, because CRS expects (lon, lat)
    polygon = flip_transform(polygon)

    # Define an azimuthal equidistant projection centered at the polygon's centroid
    # Units are in meters for buffering
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m"
    )

    # Create transformers
    transformer_to_aeqd = Transformer.from_crs(crs_wgs84, crs_aeqd, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(crs_aeqd, crs_wgs84, always_xy=True)

    # Transform the polygon to the azimuthal equidistant projection
    polygon_aeqd = shapely_transform(transformer_to_aeqd.transform, polygon)
    
    # Calculate buffer distance based on unit
    if buffer_unit == "km":
        buffer_distance_m = buffer_factor * 1000
    else:  # buffer_unit == "percent"
        area = polygon_aeqd.area
        radius = math.sqrt(area / math.pi)
        buffer_distance_m = (buffer_factor / 100.0) * radius
    
    # Buffer the polygon in meters
    buffered_polygon_aeqd = polygon_aeqd.buffer(buffer_distance_m, cap_style='round')
    
    # Transform the buffered polygon back to WGS84
    buffered_polygon_wgs84 = shapely_transform(transformer_to_wgs84.transform, buffered_polygon_aeqd)
    
    # Flip the coordinates back to (lat, lon) format
    buffered_polygon_wgs84 = flip_transform(buffered_polygon_wgs84)
    
    # PyProj will only wrap the points around, thus making the polygon structure either bad and/or invalid.
    # That's why this function is called to fix the polygon structure and have valid polygons between the antimeridian.
    buffered_polygon_wgs84 = undo_antimeridian_wrap(buffered_polygon_wgs84)
    
    return buffered_polygon_wgs84

def buffer_polygon_in_km(polygon: Union[Polygon, MultiPolygon], distance_km: float) -> Union[Polygon, MultiPolygon]:
    """Legacy function - use buffer_polygon instead."""
    return buffer_polygon(polygon, distance_km, "km")

def buffer_polygon_by_percent(polygon: Union[Polygon, MultiPolygon], percent: float) -> Union[Polygon, MultiPolygon]:
    """Legacy function - use buffer_polygon instead."""
    return buffer_polygon(polygon, percent, "percent")