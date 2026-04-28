"""
Maze model that bridges the gap between maze generation and rendering.
Handles conversion of logical maze data to renderable 3D objects.
"""

import numpy as np
from typing import List, Tuple, Optional
from maze.generator import MazeGenerator, Cell


class MazeModel:
    """
    Represents the 3D maze structure for rendering and collision.
    Wraps MazeGenerator and provides game-ready data.
    """
    
    def __init__(
        self,
        size: int = 15,
        cell_size: float = 2.0,
        wall_height: float = 3.0,
        wall_thickness: float = 0.2
    ):
        """
        Initialize the maze model.
        
        Args:
            size: Grid size (NxN)
            cell_size: Size of each cell in world units
            wall_height: Height of walls in world units
            wall_thickness: Thickness of walls in world units
        """
        self.size = size
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        
        self.generator: Optional[MazeGenerator] = None
        self.collision_walls: List[Tuple[float, float, float, float]] = []
        self.world_bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)
        
        self._generate()
    
    def _generate(self):
        """Generate a new maze and compute derived data."""
        self.generator = MazeGenerator(self.size)
        self.generator.generate()
        
        # Compute collision data
        self.collision_walls = self.generator.get_collision_walls(self.cell_size)
        
        # Compute world bounds
        offset = (self.size * self.cell_size) / 2 - self.cell_size / 2
        half_extent = (self.size * self.cell_size) / 2
        self.world_bounds = (-half_extent, half_extent, -half_extent, half_extent)
    
    def regenerate(self):
        """Generate a new random maze."""
        self._generate()
    
    def get_wall_data_for_renderer(self) -> List[Tuple[np.ndarray, int]]:
        """
        Get wall positions and orientations for the renderer.
        
        Returns:
            List of (position, orientation) tuples
        """
        return self.generator.get_wall_positions(
            self.cell_size,
            self.wall_height,
            self.wall_thickness
        )
    
    def check_collision(
        self,
        position: np.ndarray,
        radius: float = 0.3
    ) -> Tuple[bool, np.ndarray]:
        """
        Check if a position collides with any wall.
        
        Args:
            position: Player position (x, y, z)
            radius: Player collision radius
        
        Returns:
            (is_colliding, resolution_vector)
            resolution_vector points away from the collision
        """
        x, z = position[0], position[2]
        collision = False
        resolution = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        for wall_x, wall_z, wall_w, wall_d in self.collision_walls:
            # AABB collision check (top-down, ignoring Y)
            # Expand wall bounds by player radius
            min_x = wall_x - wall_w / 2 - radius
            max_x = wall_x + wall_w / 2 + radius
            min_z = wall_z - wall_d / 2 - radius
            max_z = wall_z + wall_d / 2 + radius
            
            if min_x <= x <= max_x and min_z <= z <= max_z:
                collision = True
                
                # Calculate penetration depth and resolution direction
                # Determine which side of the wall we're hitting
                left_dist = x - min_x
                right_dist = max_x - x
                top_dist = z - min_z
                bottom_dist = max_z - z
                
                min_dist = min(left_dist, right_dist, top_dist, bottom_dist)
                
                if min_dist == left_dist:
                    resolution[0] = -left_dist
                elif min_dist == right_dist:
                    resolution[0] = right_dist
                elif min_dist == top_dist:
                    resolution[2] = -top_dist
                else:
                    resolution[2] = bottom_dist
        
        return collision, resolution
    
    def is_valid_position(
        self,
        position: np.ndarray,
        radius: float = 0.3
    ) -> bool:
        """
        Check if a position is valid (no collision).
        
        Args:
            position: Position to check
            radius: Player collision radius
        
        Returns:
            True if position is valid
        """
        collides, _ = self.check_collision(position, radius)
        return not collides
    
    def get_start_position(self) -> np.ndarray:
        """Get the starting position for the player."""
        return self.generator.get_start_position(self.cell_size)
    
    def get_end_position(self) -> np.ndarray:
        """Get the ending/goal position."""
        return self.generator.get_end_position(self.cell_size)
    
    def get_floor_position(self) -> np.ndarray:
        """Get the floor plane position."""
        offset = (self.size * self.cell_size) / 2 - self.cell_size / 2
        return np.array([
            0.0,
            -0.5,  # Slightly below eye level
            0.0
        ], dtype=np.float32)
    
    def get_ceiling_position(self) -> np.ndarray:
        """Get the ceiling plane position."""
        return np.array([
            0.0,
            self.wall_height - 0.5,
            0.0
        ], dtype=np.float32)
    
    def get_world_size(self) -> float:
        """Get the total size of the maze in world units."""
        return self.size * self.cell_size
    
    def check_goal_reached(
        self,
        position: np.ndarray,
        threshold: float = 1.0
    ) -> bool:
        """
        Check if the player has reached the goal.
        
        Args:
            position: Player position
            threshold: Distance threshold for "reaching" the goal
        
        Returns:
            True if goal is reached
        """
        goal = self.get_end_position()
        distance = np.linalg.norm(position[:3] - goal[:3])
        return distance < threshold
    
    def print_ascii(self):
        """Print ASCII representation of the maze."""
        if self.generator:
            self.generator.print_ascii()
