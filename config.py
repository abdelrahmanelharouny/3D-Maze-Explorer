"""
Configuration settings for the 3D Maze Explorer game.
All configurable parameters are centralized here for easy tuning.
"""

# Window Settings
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
WINDOW_TITLE = "3D Maze Explorer"
TARGET_FPS = 60

# Camera Settings
FOV = 75.0  # Field of view in degrees
NEAR_PLANE = 0.1
FAR_PLANE = 100.0
CAMERA_SPEED = 5.0  # Units per second
MOUSE_SENSITIVITY = 0.15
PITCH_CLAMP_MIN = -89.0
PITCH_CLAMP_MAX = 89.0

# Maze Settings
MAZE_SIZE = 15  # NxN grid (must be odd for recursive backtracking)
CELL_SIZE = 2.0  # World units per cell
WALL_HEIGHT = 3.0
WALL_THICKNESS = 0.2

# Lighting Settings
AMBIENT_STRENGTH = 0.3
DIFFUSE_STRENGTH = 0.8
SPECULAR_STRENGTH = 0.3
LIGHT_DIRECTION = [0.5, -1.0, 0.3]  # Normalized directional light

# Player Settings
PLAYER_HEIGHT = 1.7  # Eye level
PLAYER_RADIUS = 0.3  # Collision radius
JUMP_FORCE = 0.0  # Set > 0 to enable jumping
GRAVITY = 0.0  # Set > 0 to enable gravity

# Debug Settings
DEBUG_WIREFRAME = False
DEBUG_SHOW_FPS = True
DEBUG_SHOW_POSITION = True

# Texture Settings
TEXTURE_WALL = "assets/textures/wall.png"
TEXTURE_FLOOR = "assets/textures/floor.png"
TEXTURE_CEILING = "assets/textures/ceiling.png"

# Shader Paths
SHADER_VERTEX = "assets/shaders/vertex.glsl"
SHADER_FRAGMENT = "assets/shaders/fragment.glsl"
