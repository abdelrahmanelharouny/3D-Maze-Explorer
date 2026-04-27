"""
Player controller for first-person movement in the maze.
Handles movement, collision response, and player state.
"""

import numpy as np
from typing import Optional
from engine.camera import Camera
from maze.maze_model import MazeModel


class Player:
    """
    Player controller that wraps the camera and handles
    collision detection and response with the maze.
    """
    
    def __init__(
        self,
        camera: Camera,
        maze: MazeModel,
        move_speed: float = 5.0,
        radius: float = 0.3,
        height: float = 1.7
    ):
        """
        Initialize the player.
        
        Args:
            camera: Camera component for view control
            maze: Maze model for collision detection
            move_speed: Movement speed (units per second)
            radius: Player collision radius
            height: Player eye height
        """
        self.camera = camera
        self.maze = maze
        self.move_speed = move_speed
        self.radius = radius
        self.height = height
        
        # Player state
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.on_ground = True
        
        # Set initial position
        self.respawn()
    
    def respawn(self):
        """Reset player to the starting position."""
        start_pos = self.maze.get_start_position()
        self.camera.position = start_pos.copy()
        self.velocity = np.zeros(3, dtype=np.float32)
    
    def update(self, delta_time: float):
        """
        Update player state and handle movement with collision.
        
        Args:
            delta_time: Time elapsed since last update (seconds)
        """
        # Calculate desired movement from input
        move_direction = np.zeros(3, dtype=np.float32)
        
        if self.camera.movement_flags[Camera.FORWARD]:
            move_direction += self.camera.front
        if self.camera.movement_flags[Camera.BACKWARD]:
            move_direction -= self.camera.front
        if self.camera.movement_flags[Camera.LEFT]:
            move_direction -= self.camera.right
        if self.camera.movement_flags[Camera.RIGHT]:
            move_direction += self.camera.right
        
        # Normalize to prevent faster diagonal movement
        if np.linalg.norm(move_direction) > 0:
            move_direction = move_direction / np.linalg.norm(move_direction)
        
        # Apply movement speed
        displacement = move_direction * self.move_speed * delta_time
        
        # Move with collision detection (X axis first)
        new_pos = self.camera.position.copy()
        new_pos[0] += displacement[0]
        
        collides, resolution = self.maze.check_collision(new_pos, self.radius)
        if collides:
            # Try moving only in X
            test_pos = self.camera.position.copy()
            test_pos[0] += displacement[0]
            if self.maze.is_valid_position(test_pos, self.radius):
                self.camera.position[0] = test_pos[0]
        else:
            self.camera.position[0] = new_pos[0]
        
        # Move with collision detection (Z axis)
        new_pos = self.camera.position.copy()
        new_pos[2] += displacement[2]
        
        collides, resolution = self.maze.check_collision(new_pos, self.radius)
        if collides:
            # Try moving only in Z
            test_pos = self.camera.position.copy()
            test_pos[2] += displacement[2]
            if self.maze.is_valid_position(test_pos, self.radius):
                self.camera.position[2] = test_pos[2]
        else:
            self.camera.position[2] = new_pos[2]
        
        # Keep player at correct height
        self.camera.position[1] = self.height
    
    def get_position(self) -> np.ndarray:
        """Get current player position."""
        return self.camera.position.copy()
    
    def get_direction(self) -> np.ndarray:
        """Get the direction the player is facing."""
        return self.camera.get_direction()
    
    def has_won(self) -> bool:
        """Check if the player has reached the goal."""
        return self.maze.check_goal_reached(self.get_position())
