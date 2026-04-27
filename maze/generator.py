"""
Maze generation using Recursive Backtracking (DFS) algorithm.
Produces a perfect maze with no loops and exactly one path between any two cells.
"""

import numpy as np
from typing import List, Tuple, Optional
import random


class Cell:
    """
    Represents a single cell in the maze grid.
    Tracks wall states for each direction.
    """
    
    # Wall directions
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3
    
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        # Walls: [top, right, bottom, left] - True means wall exists
        self.walls = [True, True, True, True]
        self.visited = False
    
    def has_wall(self, direction: int) -> bool:
        """Check if there's a wall in the given direction."""
        return self.walls[direction]
    
    def remove_wall(self, direction: int):
        """Remove the wall in the given direction."""
        self.walls[direction] = False
    
    def __repr__(self):
        walls_str = ''.join(['T' if self.walls[i] else ' ' for i in range(4)])
        return f"Cell({self.row}, {self.col}) [{walls_str}]"


class MazeGenerator:
    """
    Generates mazes using the Recursive Backtracking (DFS) algorithm.
    Creates perfect mazes with guaranteed solvability.
    """
    
    def __init__(self, size: int = 15):
        """
        Initialize the maze generator.
        
        Args:
            size: Size of the maze (NxN grid). Should be odd for best results.
        """
        # Ensure odd size for proper maze structure
        self.size = size if size % 2 == 1 else size + 1
        self.grid: List[List[Cell]] = []
        self.stack: List[Cell] = []
        
    def generate(self) -> List[List[Cell]]:
        """
        Generate a new maze using recursive backtracking.
        
        Returns:
            2D list of Cell objects representing the maze
        """
        # Initialize grid
        self.grid = [[Cell(r, c) for c in range(self.size)] for r in range(self.size)]
        self.stack = []
        
        # Start from top-left corner
        current = self.grid[0][0]
        current.visited = True
        self.stack.append(current)
        
        while self.stack:
            # Get unvisited neighbors
            neighbors = self._get_unvisited_neighbors(current)
            
            if neighbors:
                # Choose random unvisited neighbor
                next_cell, direction = random.choice(neighbors)
                
                # Remove walls between current and next
                self._remove_walls(current, next_cell, direction)
                
                # Mark next as visited and push to stack
                next_cell.visited = True
                self.stack.append(next_cell)
                current = next_cell
            else:
                # Backtrack
                current = self.stack.pop()
        
        return self.grid
    
    def _get_unvisited_neighbors(self, cell: Cell) -> List[Tuple[Cell, int]]:
        """
        Get all unvisited neighboring cells.
        
        Args:
            cell: Current cell
        
        Returns:
            List of (neighbor_cell, direction) tuples
        """
        neighbors = []
        row, col = cell.row, cell.col
        
        # Top neighbor
        if row > 0 and not self.grid[row - 1][col].visited:
            neighbors.append((self.grid[row - 1][col], Cell.TOP))
        
        # Right neighbor
        if col < self.size - 1 and not self.grid[row][col + 1].visited:
            neighbors.append((self.grid[row][col + 1], Cell.RIGHT))
        
        # Bottom neighbor
        if row < self.size - 1 and not self.grid[row + 1][col].visited:
            neighbors.append((self.grid[row + 1][col], Cell.BOTTOM))
        
        # Left neighbor
        if col > 0 and not self.grid[row][col - 1].visited:
            neighbors.append((self.grid[row][col - 1], Cell.LEFT))
        
        return neighbors
    
    def _remove_walls(self, current: Cell, next_cell: Cell, direction: int):
        """
        Remove walls between two adjacent cells.
        
        Args:
            current: Current cell
            next_cell: Neighbor cell
            direction: Direction from current to next
        """
        # Remove wall from current cell
        current.remove_wall(direction)
        
        # Remove opposite wall from next cell
        opposite = (direction + 2) % 4
        next_cell.remove_wall(opposite)
    
    def get_maze_array(self) -> np.ndarray:
        """
        Convert the maze to a numpy array representation.
        
        Returns:
            2D array where 1 represents a wall and 0 represents empty space
        """
        arr = np.zeros((self.size, self.size), dtype=np.int8)
        
        for row in range(self.size):
            for col in range(self.size):
                cell = self.grid[row][col]
                # Mark cell as occupied if it has walls
                # For collision purposes, we'll use this differently
                pass
        
        return arr
    
    def get_wall_positions(
        self,
        cell_size: float = 2.0,
        wall_height: float = 3.0,
        wall_thickness: float = 0.2
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Get world positions and orientations for all walls.
        
        Args:
            cell_size: Size of each cell in world units
            wall_height: Height of walls
            wall_thickness: Thickness of walls
        
        Returns:
            List of (position, orientation) tuples
            orientation: 0 = horizontal (top/bottom), 1 = vertical (left/right)
        """
        walls = []
        offset = (self.size * cell_size) / 2 - cell_size / 2
        
        for row in range(self.size):
            for col in range(self.size):
                cell = self.grid[row][col]
                
                # World position of cell center
                cx = col * cell_size - offset
                cz = row * cell_size - offset
                
                # Top wall
                if cell.has_wall(Cell.TOP):
                    pos = np.array([cx, wall_height / 2, cz - cell_size / 2], dtype=np.float32)
                    walls.append((pos, 0))  # Horizontal
                
                # Right wall
                if cell.has_wall(Cell.RIGHT):
                    pos = np.array([cx + cell_size / 2, wall_height / 2, cz], dtype=np.float32)
                    walls.append((pos, 1))  # Vertical
                
                # Bottom wall (only for last row)
                if row == self.size - 1 and cell.has_wall(Cell.BOTTOM):
                    pos = np.array([cx, wall_height / 2, cz + cell_size / 2], dtype=np.float32)
                    walls.append((pos, 0))
                
                # Left wall (only for first column)
                if col == 0 and cell.has_wall(Cell.LEFT):
                    pos = np.array([cx - cell_size / 2, wall_height / 2, cz], dtype=np.float32)
                    walls.append((pos, 1))
        
        return walls
    
    def get_collision_walls(
        self,
        cell_size: float = 2.0
    ) -> List[Tuple[float, float, float, float]]:
        """
        Get AABB collision bounds for all walls.
        
        Args:
            cell_size: Size of each cell in world units
        
        Returns:
            List of (x, z, width, depth) tuples for collision detection
        """
        collision_walls = []
        offset = (self.size * cell_size) / 2 - cell_size / 2
        wall_half_thickness = 0.1  # Half of wall_thickness (0.2)
        
        for row in range(self.size):
            for col in range(self.size):
                cell = self.grid[row][col]
                
                cx = col * cell_size - offset
                cz = row * cell_size - offset
                
                # Top wall
                if cell.has_wall(Cell.TOP):
                    x, z = cx, cz - cell_size / 2
                    collision_walls.append((x, z, cell_size, wall_half_thickness * 2))
                
                # Right wall
                if cell.has_wall(Cell.RIGHT):
                    x, z = cx + cell_size / 2, cz
                    collision_walls.append((x, z, wall_half_thickness * 2, cell_size))
                
                # Bottom wall
                if row == self.size - 1 and cell.has_wall(Cell.BOTTOM):
                    x, z = cx, cz + cell_size / 2
                    collision_walls.append((x, z, cell_size, wall_half_thickness * 2))
                
                # Left wall
                if col == 0 and cell.has_wall(Cell.LEFT):
                    x, z = cx - cell_size / 2, cz
                    collision_walls.append((x, z, wall_half_thickness * 2, cell_size))
        
        return collision_walls
    
    def get_start_position(self, cell_size: float = 2.0) -> np.ndarray:
        """Get the starting position (center of top-left cell)."""
        offset = (self.size * cell_size) / 2 - cell_size / 2
        return np.array([
            0 * cell_size - offset + cell_size / 2,
            1.7,  # Eye height
            0 * cell_size - offset + cell_size / 2
        ], dtype=np.float32)
    
    def get_end_position(self, cell_size: float = 2.0) -> np.ndarray:
        """Get the ending position (center of bottom-right cell)."""
        offset = (self.size * cell_size) / 2 - cell_size / 2
        return np.array([
            (self.size - 1) * cell_size - offset + cell_size / 2,
            1.7,
            (self.size - 1) * cell_size - offset + cell_size / 2
        ], dtype=np.float32)
    
    def print_ascii(self):
        """Print an ASCII representation of the maze."""
        for row in range(self.size):
            # Print top walls
            for col in range(self.size):
                cell = self.grid[row][col]
                print("+---" if cell.has_wall(Cell.TOP) else "+   ", end="")
            print("+")
            
            # Print left walls and cell contents
            for col in range(self.size):
                cell = self.grid[row][col]
                print("|   " if cell.has_wall(Cell.LEFT) else "    ", end="")
            print("|" if self.grid[row][self.size-1].has_wall(Cell.RIGHT) else "")
        
        # Print bottom border
        for col in range(self.size):
            print("+---", end="")
        print("+")
