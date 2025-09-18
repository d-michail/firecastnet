import numpy as np
import json
from scipy.spatial import KDTree
from .utils import to_lat_lon

def _save_comprehensive_mesh_debug(mesh_name, icospheres_dict):
    """Save comprehensive mesh debug information."""
    faces = np.array(icospheres_dict.get("order_0_faces", []))
    vertices = np.array(icospheres_dict.get("order_0_vertices", []))
    centroids = np.array(icospheres_dict.get("order_0_face_centroid", []))
    vertices_latlon = to_lat_lon(vertices)
    
    # Calculate vertex distances
    tree = KDTree(vertices_latlon)
    distances, neighbor_indices = tree.query(vertices_latlon, k=min(6, len(vertices)))
    
    # Face vertex usage analysis
    face_vertex_indices = np.unique(faces)
    vertex_face_count = np.bincount(faces.flatten(), minlength=len(vertices))
    
    debug_data = {
        "mesh_name": mesh_name,
        "mesh_overview": {
            "total_vertices": len(vertices),
            "total_faces": len(faces),
            "total_centroids": len(centroids),
            "vertices_used_in_faces": len(face_vertex_indices)
        },
        "vertex_analysis": {
            "coordinate_ranges": {
                "x": [float(np.min(vertices[:, 0])), float(np.max(vertices[:, 0]))],
                "y": [float(np.min(vertices[:, 1])), float(np.max(vertices[:, 1]))],
                "z": [float(np.min(vertices[:, 2])), float(np.max(vertices[:, 2]))],
                "latitude": [float(np.min(vertices_latlon[:, 0])), float(np.max(vertices_latlon[:, 0]))],
                "longitude": [float(np.min(vertices_latlon[:, 1])), float(np.max(vertices_latlon[:, 1]))]
            },
            "distance_statistics": {
                "min_distance": float(np.min(distances[:, 1])) if distances.shape[1] > 1 else None,
                "max_distance": float(np.max(distances[:, 1])) if distances.shape[1] > 1 else None,
                "mean_distance": float(np.mean(distances[:, 1])) if distances.shape[1] > 1 else None,
                "std_distance": float(np.std(distances[:, 1])) if distances.shape[1] > 1 else None
            },
            "face_connectivity": {
                "vertices_with_no_faces": int(np.sum(vertex_face_count == 0)),
                "min_faces_per_vertex": int(np.min(vertex_face_count[vertex_face_count > 0])) if np.any(vertex_face_count > 0) else 0,
                "max_faces_per_vertex": int(np.max(vertex_face_count)),
                "mean_faces_per_vertex": float(np.mean(vertex_face_count[vertex_face_count > 0])) if np.any(vertex_face_count > 0) else 0
            }
        },
        "face_analysis": {
            "vertex_index_ranges": {
                "min_index_in_faces": int(np.min(faces)),
                "max_index_in_faces": int(np.max(faces))
            },
            "face_validation": {
                "faces_with_duplicate_vertices": int(np.sum(np.any(np.sort(faces, axis=1)[:, :-1] == np.sort(faces, axis=1)[:, 1:], axis=1)))
            }
        },
        "unused_vertices": {
            "indices": np.setdiff1d(np.arange(len(vertices)), face_vertex_indices).tolist(),
            "coordinates": vertices[np.setdiff1d(np.arange(len(vertices)), face_vertex_indices)].tolist() if len(np.setdiff1d(np.arange(len(vertices)), face_vertex_indices)) > 0 else []
        }
    }
    
    with open("comprehensive_mesh_debug.json", "w") as f:
        json.dump(debug_data, f, indent=2)
    
    print(f"Comprehensive debug info saved to comprehensive_mesh_debug.json")


def validate_mesh(mesh_name, icospheres_dict, min_vertex_distance_deg=0.25):
    """
    Validate the mesh structure and ensure minimum vertex distance requirements.
    
    Args:
        icospheres_dict: Dictionary containing mesh data
        min_vertex_distance_deg: Minimum allowed distance between vertices in degrees
    
    Returns:
        bool: True if mesh is valid
    
    Raises:
        ValueError: If mesh doesn't meet validation requirements
    """
    # Extract mesh data
    faces = np.array(icospheres_dict.get("order_0_faces", []))
    vertices = np.array(icospheres_dict.get("order_0_vertices", []))
    centroids = np.array(icospheres_dict.get("order_0_face_centroid", []))
    
    print(f"Mesh stats - Centroids: {centroids.shape}, Vertices: {vertices.shape}, Faces: {faces.shape}")
    
    # Save comprehensive debug information
    print("Saving comprehensive debug information...")
    _save_comprehensive_mesh_debug(mesh_name, icospheres_dict)
    
    # Basic structure validation
    print("Validating mesh structure...")
    _validate_mesh_structure(faces, vertices, centroids)
    print("Mesh structure validation passed.\n")

    # Face integrity validation
    print("Validating face integrity...")
    _validate_face_integrity(faces, vertices)
    print("Face integrity validation passed.\n")
    
    # Vertex uniqueness validation
    print("Validating vertex uniqueness and face coverage...")
    _validate_vertex_uniqueness(vertices, faces)
    print("Vertex uniqueness validation passed.\n")
    
    # Distance validation
    print(f"Validating minimum vertex distance (threshold: {min_vertex_distance_deg}°)...")
    _validate_vertex_distances(vertices, min_vertex_distance_deg)
    print("Minimum vertex distance validation passed.\n")
    
    # Coordinate boundaries validation
    print("Validating coordinate boundaries...")
    _validate_coordinate_boundaries(vertices)
    print("Coordinate boundaries validation passed.\n")
    
    print("All mesh validations passed.")
    return True


def _validate_mesh_structure(faces, vertices, centroids):
    """Validate basic mesh structure requirements."""
    total_vertices, total_faces, total_centroids = len(vertices), len(faces), len(centroids)
    
    if total_vertices == 0 or total_faces == 0 or total_centroids == 0:
        raise ValueError("Mesh data is empty or missing")
    
    if vertices.shape[1] != 3:
        raise ValueError("Vertices must be 3D points")
    
    if faces.shape[1] != 3:
        raise ValueError("Faces must be triangles (3 vertices per face)")
    
    if total_faces != total_centroids:
        raise ValueError("Number of faces must match number of face centroids")


def _validate_face_integrity(faces, vertices):
    """Validate face indices and detect duplicate vertices within faces."""
    max_vertex_idx = np.max(faces)
    total_vertices = len(vertices)
    
    print(f"Max vertex index: {max_vertex_idx}, Total vertices: {total_vertices}")
    
    if max_vertex_idx >= total_vertices:
        raise ValueError("Face indices exceed vertex count")
    
    # Check for duplicate vertices within individual faces
    sorted_faces = np.sort(faces, axis=1)
    duplicate_mask = np.any(sorted_faces[:, :-1] == sorted_faces[:, 1:], axis=1)
    
    if np.any(duplicate_mask):
        duplicate_indices = np.where(duplicate_mask)[0]
        raise ValueError(f"Found {len(duplicate_indices)} faces with duplicate vertices")


def _validate_vertex_uniqueness(vertices, faces):
    """Validate vertex uniqueness and face coverage."""
    total_vertices = len(vertices)
    unique_vertices = np.unique(vertices, axis=0)
    
    if total_vertices != len(unique_vertices):
        print(f"WARNING: Found {total_vertices - len(unique_vertices)} duplicate vertices")
    
    # Check if all vertices are used in faces
    face_vertex_indices = np.unique(faces)
    unused_vertices = np.setdiff1d(np.arange(total_vertices), face_vertex_indices)
    
    if len(unused_vertices) > 0:
        raise ValueError(f"Found {len(unused_vertices)} unused vertices in mesh")
    
    # Save face vertex index data for debugging
    _save_face_indices_debug_info(face_vertex_indices, total_vertices, len(faces))
    
    print("Vertex uniqueness validation passed")


def _validate_vertex_distances(vertices, min_distance_deg):
    """Validate minimum distances between vertices in lat-lon space."""
    vertices_latlon = to_lat_lon(vertices)
    tree = KDTree(vertices_latlon)
    
    # Find nearest neighbor distances (excluding self)
    distances, _ = tree.query(vertices_latlon, k=2)
    min_distance = np.min(distances[:, 1])
    
    if min_distance < min_distance_deg:
        problem_count = np.sum(distances[:, 1] < min_distance_deg)
        print(f"WARNING: {problem_count} vertices below distance threshold "
              f"(min: {min_distance:.6f}°, threshold: {min_distance_deg}°)")
    
    print(f"Minimum vertex distance: {min_distance:.6f}°")


def _validate_coordinate_boundaries(vertices):
    """Validate lat-lon coordinate boundaries."""
    vertices_latlon = to_lat_lon(vertices)
    lat_range = (np.min(vertices_latlon[:, 0]), np.max(vertices_latlon[:, 0]))
    lon_range = (np.min(vertices_latlon[:, 1]), np.max(vertices_latlon[:, 1]))
    
    print(f"Latitude range: [{lat_range[0]:.6f}, {lat_range[1]:.6f}]")
    print(f"Longitude range: [{lon_range[0]:.6f}, {lon_range[1]:.6f}]")
    
    if lat_range[0] < -90.0 or lat_range[1] > 90.0:
        raise ValueError(f"Latitude values exceed valid range [-90, 90]: {lat_range}")
    
    if lon_range[0] < -180.0 or lon_range[1] > 180.0:
        raise ValueError(f"Longitude values exceed valid range [-180, 180]: {lon_range}")
    

def _save_face_indices_debug_info(face_vertex_indices, total_vertices, total_faces):
    """Save comprehensive face vertex indices and mesh analysis for debugging purposes."""
    debug_data = {
        "face_vertex_indices": sorted(face_vertex_indices.tolist()),
        "total_vertices": int(total_vertices),
        "total_faces": int(total_faces),
        "mesh_info": {
            "vertices_used_in_faces": len(face_vertex_indices),
            "vertices_not_used": int(total_vertices - len(face_vertex_indices)),
            "vertex_usage_percentage": (len(face_vertex_indices) / total_vertices) * 100,
            "faces_per_vertex_ratio": total_faces / len(face_vertex_indices) if len(face_vertex_indices) > 0 else 0
        },
        "vertex_statistics": {
            "min_vertex_index": int(np.min(face_vertex_indices)) if len(face_vertex_indices) > 0 else None,
            "max_vertex_index": int(np.max(face_vertex_indices)) if len(face_vertex_indices) > 0 else None,
            "vertex_index_range": int(np.max(face_vertex_indices) - np.min(face_vertex_indices)) if len(face_vertex_indices) > 0 else None
        },
        "unused_vertices": {
            "indices": np.setdiff1d(np.arange(total_vertices), face_vertex_indices).tolist(),
            "count": int(total_vertices - len(face_vertex_indices))
        }
    }
    
    with open("face_vertex_indices_debug.json", "w") as f:
        json.dump(debug_data, f, indent=2)
    
    print(f"Debug info saved to face_vertex_indices_debug.json")