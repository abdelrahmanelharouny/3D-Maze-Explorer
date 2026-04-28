"""
Main entry point for the 3D Maze Explorer game.
Initializes pygame, OpenGL, and runs the game loop.
"""

import sys
import time
import numpy as np
import pygame
from OpenGL.GL import *

# Import engine components
from engine.shader import Shader
from engine.mesh import Mesh
from engine.texture import Texture
from engine.camera import Camera
from engine.renderer import Renderer

# Import maze components
from maze.maze_model import MazeModel

# Import game components
from game.player import Player
from game.input_handler import InputHandler

# Import configuration
import config

# Import utility functions
from utils.math_utils import create_box_mesh, create_plane_mesh
from utils.texture_generator import (
    create_brick_texture,
    create_floor_texture,
    create_ceiling_texture
)


class Game:
    """
    Main game class that orchestrates all components.
    Handles initialization, game loop, and cleanup.
    """
    
    def __init__(self):
        self.running = False
        self.clock = None
        self.screen = None
        self.font = None
        
        # Game components
        self.camera: Camera = None
        self.renderer: Renderer = None
        self.maze: MazeModel = None
        self.player: Player = None
        self.input_handler: InputHandler = None
        
        # State
        self.delta_time = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.fps_update_timer = 0.0
        self.wireframe_mode = config.DEBUG_WIREFRAME
        self.game_won = False
    
    def initialize(self):
        """Initialize all game systems."""
        print("Initializing 3D Maze Explorer...")
        
        # Initialize pygame
        pygame.init()
        pygame.display.set_mode(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT),
            pygame.OPENGL | pygame.DOUBLEBUF
        )
        pygame.display.set_caption(config.WINDOW_TITLE)
        
        self.screen = pygame.display.get_surface()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize OpenGL
        self._init_opengl()
        
        # Create shader program
        print("Loading shaders...")
        self.shader = Shader().load_from_file(
            config.SHADER_VERTEX,
            config.SHADER_FRAGMENT
        )
        
        # Create camera
        print("Setting up camera...")
        self.camera = Camera(
            position=np.array([0.0, 1.7, 0.0], dtype=np.float32),
            fov=config.FOV,
            aspect=config.SCREEN_WIDTH / config.SCREEN_HEIGHT,
            near=config.NEAR_PLANE,
            far=config.FAR_PLANE,
            speed=config.CAMERA_SPEED,
            sensitivity=config.MOUSE_SENSITIVITY
        )
        
        # Generate maze
        print(f"Generating maze ({config.MAZE_SIZE}x{config.MAZE_SIZE})...")
        self.maze = MazeModel(
            size=config.MAZE_SIZE,
            cell_size=config.CELL_SIZE,
            wall_height=config.WALL_HEIGHT,
            wall_thickness=config.WALL_THICKNESS
        )
        
        # Set camera to start position
        start_pos = self.maze.get_start_position()
        self.camera.position = start_pos.copy()
        
        # Create player
        print("Creating player controller...")
        self.player = Player(
            camera=self.camera,
            maze=self.maze,
            move_speed=config.CAMERA_SPEED,
            radius=config.PLAYER_RADIUS,
            height=config.PLAYER_HEIGHT
        )
        
        # Create input handler
        self.input_handler = InputHandler(self.camera)
        
        # Setup renderer
        print("Setting up renderer...")
        self.renderer = Renderer()
        self.renderer.initialize(self.shader)
        
        # Create procedural textures
        print("Generating textures...")
        self._setup_textures()
        
        # Build wall instances for rendering
        self._build_maze_geometry()
        
        # Configure lighting
        self.renderer.set_lighting(
            direction=np.array(config.LIGHT_DIRECTION, dtype=np.float32),
            ambient=config.AMBIENT_STRENGTH,
            diffuse=config.DIFFUSE_STRENGTH,
            specular=config.SPECULAR_STRENGTH
        )
        
        self.running = True
        print("Initialization complete!")
        print("\nControls:")
        print("  W/A/S/D - Move")
        print("  Mouse - Look around")
        print("  Click - Capture mouse")
        print("  ESC - Release mouse")
        print("  F - Toggle wireframe")
        print("  R - Regenerate maze")
        print("  Q - Quit\n")
    
    def _init_opengl(self):
        """Configure OpenGL settings."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        
        # Enable blending for transparency if needed
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set clear color
        glClearColor(0.1, 0.1, 0.15, 1.0)
    
    def _setup_textures(self):
        """Create and load textures."""
        try:
            # Try to load from files first
            wall_tex = Texture().load(config.TEXTURE_WALL)
        except (ImportError, FileNotFoundError):
            # Fall back to procedural generation
            print("Using procedural textures...")
            wall_data = create_brick_texture()
            wall_tex = Texture()
            wall_tex.texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, wall_tex.texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB,
                256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE,
                wall_data
            )
            glGenerateMipmap(GL_TEXTURE_2D)
            wall_tex.width, wall_tex.height = 256, 256
            wall_tex.channels = 3
        
        try:
            floor_tex = Texture().load(config.TEXTURE_FLOOR)
        except (ImportError, FileNotFoundError):
            floor_data = create_floor_texture()
            floor_tex = Texture()
            floor_tex.texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, floor_tex.texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB,
                256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE,
                floor_data
            )
            glGenerateMipmap(GL_TEXTURE_2D)
            floor_tex.width, floor_tex.height = 256, 256
            floor_tex.channels = 3
        
        try:
            ceiling_tex = Texture().load(config.TEXTURE_CEILING)
        except (ImportError, FileNotFoundError):
            ceiling_data = create_ceiling_texture()
            ceiling_tex = Texture()
            ceiling_tex.texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, ceiling_tex.texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB,
                256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE,
                ceiling_data
            )
            glGenerateMipmap(GL_TEXTURE_2D)
            ceiling_tex.width, ceiling_tex.height = 256, 256
            ceiling_tex.channels = 3
        
        self.renderer.set_textures(
            wall=wall_tex,
            floor=floor_tex,
            ceiling=ceiling_tex
        )
    
    def _build_maze_geometry(self):
        """Build the maze geometry for rendering."""
        self.renderer.clear_walls()
        
        wall_data = self.maze.get_wall_data_for_renderer()
        
        for pos, orientation in wall_data:
            rotation = 90.0 if orientation == 1 else 0.0
            # Scale position based on cell size
            scaled_pos = pos.copy()
            self.renderer.add_wall_instance(scaled_pos, rotation)
        
        self.renderer.build_instance_buffer()
    
    def run(self):
        """Run the main game loop."""
        if not self.running:
            return
        
        print("Starting game loop...")
        
        while self.running:
            # Calculate delta time
            self.delta_time = self.clock.tick(config.TARGET_FPS) / 1000.0
            
            # Update FPS counter
            self._update_fps()
            
            # Process events
            self._handle_events()
            
            if not self.running:
                break
            
            # Update game state
            self._update()
            
            # Render
            self._render()
        
        self.shutdown()
    
    def _handle_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if not self.input_handler.handle_event(event):
                self.running = False
                return
            
            # Handle special keys
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.wireframe_mode = not self.wireframe_mode
                elif event.key == pygame.K_r:
                    self._regenerate_maze()
                elif event.key == pygame.K_q:
                    self.running = False
    
    def _update(self):
        """Update game state."""
        # Update player (handles movement with collision)
        self.player.update(self.delta_time)
        
        # Check win condition
        if not self.game_won and self.player.has_won():
            self.game_won = True
            print("\n🎉 Congratulations! You reached the goal! 🎉\n")
    
    def _render(self):
        """Render the scene."""
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Get window size for aspect ratio
        width, height = pygame.display.get_window_size()
        self.camera.set_aspect(width, height)
        
        # Render the maze
        self.renderer.render_scene(
            camera=self.camera,
            floor_pos=self.maze.get_floor_position(),
            ceiling_pos=self.maze.get_ceiling_position(),
            wireframe=self.wireframe_mode
        )
        
        # Render UI
        self._render_ui()
        
        # Swap buffers
        pygame.display.flip()
    
    def _render_ui(self):
        """Render 2D UI elements."""
        if config.DEBUG_SHOW_FPS:
            fps_text = f"FPS: {self.fps:.1f}"
            fps_surface = self.font.render(fps_text, True, (255, 255, 255))
            self.screen.blit(fps_surface, (10, 10))
        
        if config.DEBUG_SHOW_POSITION:
            pos = self.player.get_position()
            pos_text = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            pos_surface = self.font.render(pos_text, True, (255, 255, 255))
            self.screen.blit(pos_surface, (10, 40))
        
        if self.game_won:
            win_text = "🎉 GOAL REACHED! Press R for new maze 🎉"
            win_surface = self.font.render(win_text, True, (0, 255, 0))
            text_rect = win_surface.get_rect(center=(config.SCREEN_WIDTH // 2, 50))
            self.screen.blit(win_surface, text_rect)
        
        if not self.input_handler.is_mouse_captured():
            hint_text = "Click to capture mouse"
            hint_surface = self.font.render(hint_text, True, (200, 200, 200))
            text_rect = hint_surface.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT - 30))
            self.screen.blit(hint_surface, text_rect)
    
    def _update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        self.fps_update_timer += self.delta_time
        
        if self.fps_update_timer >= 0.5:  # Update every 0.5 seconds
            self.fps = self.frame_count / self.fps_update_timer
            self.frame_count = 0
            self.fps_update_timer = 0.0
    
    def _regenerate_maze(self):
        """Generate a new random maze."""
        print("Generating new maze...")
        self.maze.regenerate()
        self._build_maze_geometry()
        
        # Reset player to start
        start_pos = self.maze.get_start_position()
        self.camera.position = start_pos.copy()
        self.player.respawn()
        self.game_won = False
    
    def shutdown(self):
        """Clean up resources and exit."""
        print("Shutting down...")
        
        if self.renderer:
            self.renderer.cleanup()
        
        pygame.quit()
        print("Goodbye!")


def main():
    """Main entry point."""
    game = Game()
    
    try:
        game.initialize()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
