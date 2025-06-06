import math
import shapely
import shapely.wkt
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform

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

def buffer_polygon_in_km(polygon: Polygon, distance_km: float) -> Polygon:
    """
    Buffers a Shapely Polygon by a given distance in kilometers.

    The function transforms the polygon to an azimuthal equidistant projection
    centered at its centroid, performs the buffer operation in meters,
    and then transforms the result back to WGS84 (EPSG:4326).

    Args:
        polygon (shapely.geometry.Polygon): The input polygon,
                                            assumed to be in WGS84 (EPSG:4326).
        distance_km (float): The buffer distance in kilometers.

    Returns:
        shapely.geometry.Polygon: The buffered polygon in WGS84 (EPSG:4326).
    """
    if isinstance(polygon, str):
        polygon = shapely.wkt.loads(polygon)
    if not isinstance(polygon, Polygon) or not polygon.is_valid:
        raise ValueError("Input must be a shapely Polygon or WKT string representing a polygon.")

    # Define the geographic CRS (WGS84)
    crs_wgs84 = CRS.from_epsg(4326)
    
    # Get the centroid of the polygon to center the projection
    centroid = polygon.centroid
    lon, lat = centroid.y, centroid.x
    print(f"Buffering polygon centered at: ({lat}, {lon})")
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
    
    # Buffer the polygon in meters
    distance_m = distance_km * 1000
    buffered_polygon_aeqd = polygon_aeqd.buffer(distance_m, cap_style='round')
    
    # Transform the buffered polygon back to WGS84
    buffered_polygon_wgs84 = shapely_transform(transformer_to_wgs84.transform, buffered_polygon_aeqd)
    
    # Flip the coordinates back to (lat, lon) format
    buffered_polygon_wgs84 = flip_transform(buffered_polygon_wgs84)
    return buffered_polygon_wgs84

def buffer_polygon_by_percent(polygon: Polygon, percent: float) -> Polygon:
    """
    Buffers a Shapely Polygon by a percentage of its approximate radius.

    The function transforms the polygon to an azimuthal equidistant projection
    centered at its centroid, calculates the buffer distance as a percentage
    of the polygon's approximate radius, performs the buffer operation in meters,
    and then transforms the result back to WGS84 (EPSG:4326).

    Args:
        polygon (shapely.geometry.Polygon): The input polygon,
                                            assumed to be in WGS84 (EPSG:4326).
        percent (float): The buffer percentage relative to the polygon's radius.

    Returns:
        shapely.geometry.Polygon: The buffered polygon in WGS84 (EPSG:4326).
    """
    if isinstance(polygon, str):
        polygon = shapely.wkt.loads(polygon)
    if not isinstance(polygon, Polygon) or not polygon.is_valid:
        raise ValueError("Input must be a shapely Polygon or WKT string representing a polygon.")

    # Define the geographic CRS (WGS84)
    crs_wgs84 = CRS.from_epsg(4326)
    
    # Get the centroid of the polygon to center the projection
    centroid = polygon.centroid
    lon, lat = centroid.y, centroid.x
    print(f"Buffering polygon centered at: ({lat}, {lon})")
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

    # Calculate buffer distance as percentage of approximate radius
    area = polygon_aeqd.area
    radius = math.sqrt(area / math.pi)
    buffer_distance_m = (percent / 100.0) * radius
    
    # Buffer the polygon in meters
    buffered_polygon_aeqd = polygon_aeqd.buffer(buffer_distance_m, cap_style='round')
    
    # Transform the buffered polygon back to WGS84
    buffered_polygon_wgs84 = shapely_transform(transformer_to_wgs84.transform, buffered_polygon_aeqd)
    
    # Flip the coordinates back to (lat, lon) format
    buffered_polygon_wgs84 = flip_transform(buffered_polygon_wgs84)
    return buffered_polygon_wgs84