"""
Collision detection utilities for the maze game.
Provides AABB and other collision detection functions.
"""

import numpy as np
from typing import Tuple, List


def aabb_collision_2d(
    pos_x: float,
    pos_z: float,
    radius: float,
    wall_x: float,
    wall_z: float,
    wall_width: float,
    wall_depth: float
) -> Tuple[bool, float, float]:
    """
    Check 2D AABB collision between player (circle) and wall (rectangle).
    
    Args:
        pos_x, pos_z: Player position
        radius: Player collision radius
        wall_x, wall_z: Wall center position
        wall_width, wall_depth: Wall dimensions
    
    Returns:
        (is_colliding, penetration_x, penetration_z)
    """
    # Expand wall bounds by player radius
    min_x = wall_x - wall_width / 2 - radius
    max_x = wall_x + wall_width / 2 + radius
    min_z = wall_z - wall_depth / 2 - radius
    max_z = wall_z + wall_depth / 2 + radius
    
    if min_x <= pos_x <= max_x and min_z <= pos_z <= max_z:
        # Calculate penetration depths
        left_dist = pos_x - min_x
        right_dist = max_x - pos_x
        top_dist = pos_z - min_z
        bottom_dist = max_z - pos_z
        
        min_dist = min(left_dist, right_dist, top_dist, bottom_dist)
        
        if min_dist == left_dist:
            return True, -left_dist, 0.0
        elif min_dist == right_dist:
            return True, right_dist, 0.0
        elif min_dist == top_dist:
            return True, 0.0, -top_dist
        else:
            return True, 0.0, bottom_dist
    
    return False, 0.0, 0.0


def check_point_in_rect(
    x: float,
    z: float,
    rect_center_x: float,
    rect_center_z: float,
    rect_width: float,
    rect_depth: float
) -> bool:
    """
    Check if a point is inside a rectangle.
    
    Args:
        x, z: Point to check
        rect_center_x, rect_center_z: Rectangle center
        rect_width, rect_depth: Rectangle dimensions
    
    Returns:
        True if point is inside rectangle
    """
    half_w = rect_width / 2
    half_d = rect_depth / 2
    
    return (rect_center_x - half_w <= x <= rect_center_x + half_w and
            rect_center_z - half_d <= z <= rect_center_z + half_d)


def resolve_collision(
    position: np.ndarray,
    velocity: np.ndarray,
    walls: List[Tuple[float, float, float, float]],
    radius: float
) -> np.ndarray:
    """
    Resolve collision against multiple walls.
    
    Args:
        position: Current position
        velocity: Movement velocity
        walls: List of (x, z, width, depth) wall bounds
        radius: Player radius
    
    Returns:
        Resolved position
    """
    new_pos = position.copy()
    
    for wall_x, wall_z, wall_w, wall_d in walls:
        collides, pen_x, pen_z = aabb_collision_2d(
            new_pos[0], new_pos[2], radius,
            wall_x, wall_z, wall_w, wall_d
        )
        
        if collides:
            # Apply resolution based on primary penetration axis
            if abs(pen_x) > abs(pen_z):
                new_pos[0] += pen_x
            else:
                new_pos[2] += pen_z
    
    return new_pos
