"""
Math utility functions for 3D graphics operations.
Uses numpy for efficient matrix and vector calculations.
"""

import numpy as np
from typing import Tuple


def radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return np.deg2rad(degrees)


def perspective(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    """
    Create a perspective projection matrix.
    
    Args:
        fov_degrees: Field of view in degrees
        aspect: Aspect ratio (width / height)
        near: Near clipping plane
        far: Far clipping plane
    
    Returns:
        4x4 projection matrix
    """
    fov_rad = radians(fov_degrees)
    tan_half_fov = np.tan(fov_rad / 2.0)
    
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = 1.0 / (aspect * tan_half_fov)
    matrix[1, 1] = 1.0 / tan_half_fov
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -(2.0 * far * near) / (far - near)
    matrix[3, 2] = -1.0
    
    return matrix


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Create a view matrix using the lookAt method.
    
    Args:
        eye: Camera position
        center: Point the camera is looking at
        up: Up vector
    
    Returns:
        4x4 view matrix
    """
    # Forward direction (normalized)
    f = normalize(center - eye)
    # Right direction (normalized)
    r = normalize(np.cross(f, up))
    # True up direction (normalized)
    u = normalize(np.cross(r, f))
    
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = r
    matrix[1, :3] = u
    matrix[2, :3] = -f
    matrix[:3, 3] = -np.array([np.dot(r, eye), np.dot(u, eye), np.dot(-f, eye)])
    
    return matrix


def translate(position: np.ndarray) -> np.ndarray:
    """Create a translation matrix."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = position
    return matrix


def scale(scalar: float) -> np.ndarray:
    """Create a uniform scaling matrix."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] *= scalar
    return matrix


def rotate_y(angle_degrees: float) -> np.ndarray:
    """Create a rotation matrix around the Y axis."""
    angle_rad = radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = cos_a
    matrix[0, 2] = sin_a
    matrix[2, 0] = -sin_a
    matrix[2, 2] = cos_a
    
    return matrix


def normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def create_box_mesh(width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a box mesh with vertices and texture coordinates.
    
    Returns:
        vertices: (N, 8) array with position (x,y,z) and texcoord (u,v) and normal (nx,ny,nz)
        indices: Face indices for indexed drawing
    """
    hw, hh, hd = width / 2.0, height / 2.0, depth / 2.0
    
    # Define vertices with positions, texture coords, and normals
    # Format: x, y, z, u, v, nx, ny, nz
    vertices = np.array([
        # Front face (z = +hd)
        [-hw, -hh, hd, 0.0, 0.0, 0.0, 0.0, 1.0],
        [hw, -hh, hd, 1.0, 0.0, 0.0, 0.0, 1.0],
        [hw, hh, hd, 1.0, 1.0, 0.0, 0.0, 1.0],
        [-hw, hh, hd, 0.0, 1.0, 0.0, 0.0, 1.0],
        
        # Back face (z = -hd)
        [hw, -hh, -hd, 0.0, 0.0, 0.0, 0.0, -1.0],
        [-hw, -hh, -hd, 1.0, 0.0, 0.0, 0.0, -1.0],
        [-hw, hh, -hd, 1.0, 1.0, 0.0, 0.0, -1.0],
        [hw, hh, -hd, 0.0, 1.0, 0.0, 0.0, -1.0],
        
        # Top face (y = +hh)
        [-hw, hh, hd, 0.0, 0.0, 0.0, 1.0, 0.0],
        [hw, hh, hd, 1.0, 0.0, 0.0, 1.0, 0.0],
        [hw, hh, -hd, 1.0, 1.0, 0.0, 1.0, 0.0],
        [-hw, hh, -hd, 0.0, 1.0, 0.0, 1.0, 0.0],
        
        # Bottom face (y = -hh)
        [-hw, -hh, -hd, 0.0, 0.0, 0.0, -1.0, 0.0],
        [hw, -hh, -hd, 1.0, 0.0, 0.0, -1.0, 0.0],
        [hw, -hh, hd, 1.0, 1.0, 0.0, -1.0, 0.0],
        [-hw, -hh, hd, 0.0, 1.0, 0.0, -1.0, 0.0],
        
        # Right face (x = +hw)
        [hw, -hh, hd, 0.0, 0.0, 1.0, 0.0, 0.0],
        [hw, -hh, -hd, 1.0, 0.0, 1.0, 0.0, 0.0],
        [hw, hh, -hd, 1.0, 1.0, 1.0, 0.0, 0.0],
        [hw, hh, hd, 0.0, 1.0, 1.0, 0.0, 0.0],
        
        # Left face (x = -hw)
        [-hw, -hh, -hd, 0.0, 0.0, -1.0, 0.0, 0.0],
        [-hw, -hh, hd, 1.0, 0.0, -1.0, 0.0, 0.0],
        [-hw, hh, hd, 1.0, 1.0, -1.0, 0.0, 0.0],
        [-hw, hh, -hd, 0.0, 1.0, -1.0, 0.0, 0.0],
    ], dtype=np.float32)
    
    # Indices for each face (two triangles per face)
    indices = np.array([
        # Front
        0, 1, 2, 2, 3, 0,
        # Back
        4, 5, 6, 6, 7, 4,
        # Top
        8, 9, 10, 10, 11, 8,
        # Bottom
        12, 13, 14, 14, 15, 12,
        # Right
        16, 17, 18, 18, 19, 16,
        # Left
        20, 21, 22, 22, 23, 20,
    ], dtype=np.uint32)
    
    return vertices, indices


def create_plane_mesh(size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a plane mesh (flat quad) with vertices and texture coordinates.
    
    Returns:
        vertices: Position and texcoord data
        indices: Face indices
    """
    hs = size / 2.0
    
    vertices = np.array([
        [-hs, 0.0, -hs, 0.0, 0.0, 0.0, 1.0, 0.0],
        [hs, 0.0, -hs, 1.0, 0.0, 0.0, 1.0, 0.0],
        [hs, 0.0, hs, 1.0, 1.0, 0.0, 1.0, 0.0],
        [-hs, 0.0, hs, 0.0, 1.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    
    return vertices, indices
