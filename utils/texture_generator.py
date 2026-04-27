"""
Procedural texture generation for when image files are not available.
Creates brick, floor tile, and ceiling patterns programmatically.
"""

import numpy as np
from typing import Tuple


def create_brick_texture(
    width: int = 256,
    height: int = 256,
    brick_color: Tuple[int, int, int] = (180, 100, 80),
    mortar_color: Tuple[int, int, int] = (100, 100, 100),
    brick_rows: int = 8,
    brick_cols: int = 4
) -> np.ndarray:
    """
    Create a procedural brick wall texture.
    
    Args:
        width: Texture width in pixels
        height: Texture height in pixels
        brick_color: RGB color of bricks
        mortar_color: RGB color of mortar lines
        brick_rows: Number of brick rows
        brick_cols: Number of brick columns
    
    Returns:
        Nx3 uint8 array of pixel data
    """
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    
    brick_height = height // brick_rows
    brick_width = width // brick_cols
    mortar_size = max(2, brick_height // 10)
    
    for y in range(height):
        row = y // brick_height
        offset = (row % 2) * (brick_width // 2)  # Offset every other row
        
        for x in range(width):
            col_x = (x + offset) % width
            col = col_x // brick_width
            
            # Check if in mortar region
            in_horizontal_mortar = (y % brick_height) < mortar_size
            in_vertical_mortar = (col_x % brick_width) < mortar_size
            
            if in_horizontal_mortar or in_vertical_mortar:
                texture[y, x] = mortar_color
            else:
                # Add some variation to brick color
                noise = np.random.randint(-15, 15, 3)
                color = np.clip(np.array(brick_color) + noise, 0, 255)
                texture[y, x] = color
    
    return texture


def create_floor_texture(
    width: int = 256,
    height: int = 256,
    tile_color: Tuple[int, int, int] = (120, 120, 120),
    grout_color: Tuple[int, int, int] = (60, 60, 60),
    tile_rows: int = 4,
    tile_cols: int = 4
) -> np.ndarray:
    """
    Create a procedural floor tile texture.
    
    Args:
        width: Texture width in pixels
        height: Texture height in pixels
        tile_color: RGB color of tiles
        grout_color: RGB color of grout lines
        tile_rows: Number of tile rows
        tile_cols: Number of tile columns
    
    Returns:
        Nx3 uint8 array of pixel data
    """
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    
    tile_height = height // tile_rows
    tile_width = width // tile_cols
    grout_size = max(2, tile_height // 15)
    
    for y in range(height):
        for x in range(width):
            # Check if in grout region
            in_horizontal_grout = (y % tile_height) < grout_size
            in_vertical_grout = (x % tile_width) < grout_size
            
            if in_horizontal_grout or in_vertical_grout:
                texture[y, x] = grout_color
            else:
                # Add subtle variation to tile color
                noise = np.random.randint(-10, 10, 3)
                color = np.clip(np.array(tile_color) + noise, 0, 255)
                texture[y, x] = color
    
    return texture


def create_ceiling_texture(
    width: int = 256,
    height: int = 256,
    base_color: Tuple[int, int, int] = (200, 200, 190)
) -> np.ndarray:
    """
    Create a simple ceiling texture with subtle noise.
    
    Args:
        width: Texture width in pixels
        height: Texture height in pixels
        base_color: Base RGB color
    
    Returns:
        Nx3 uint8 array of pixel data
    """
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            # Add subtle noise for texture
            noise = np.random.randint(-20, 20, 3)
            color = np.clip(np.array(base_color) + noise, 0, 255)
            texture[y, x] = color
    
    return texture


def create_checkerboard_texture(
    width: int = 256,
    height: int = 256,
    color1: Tuple[int, int, int] = (255, 255, 255),
    color2: Tuple[int, int, int] = (0, 0, 0),
    squares: int = 8
) -> np.ndarray:
    """
    Create a checkerboard pattern texture.
    
    Args:
        width: Texture width in pixels
        height: Texture height in pixels
        color1: First color
        color2: Second color
        squares: Number of squares per row/column
    
    Returns:
        Nx3 uint8 array of pixel data
    """
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    
    square_w = width // squares
    square_h = height // squares
    
    for y in range(height):
        row = y // square_h
        for x in range(width):
            col = x // square_w
            if (row + col) % 2 == 0:
                texture[y, x] = color1
            else:
                texture[y, x] = color2
    
    return texture
