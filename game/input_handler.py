"""
Input handler for processing keyboard and mouse events.
Maps pygame events to game actions.
"""

import pygame
from engine.camera import Camera


class InputHandler:
    """
    Handles all input processing and maps it to camera/player actions.
    Supports keyboard movement and mouse look.
    """
    
    def __init__(self, camera: Camera):
        """
        Initialize the input handler.
        
        Args:
            camera: Camera to control with input
        """
        self.camera = camera
        self.mouse_captured = False
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Process a single pygame event.
        
        Args:
            event: Pygame event to process
        
        Returns:
            True if the game should continue, False to quit
        """
        if event.type == pygame.QUIT:
            return False
        
        elif event.type == pygame.KEYDOWN:
            self._handle_key_down(event.key)
        
        elif event.type == pygame.KEYUP:
            self._handle_key_up(event.key)
        
        elif event.type == pygame.MOUSEMOTION:
            if self.mouse_captured:
                xpos, ypos = event.pos
                self.camera.process_mouse_move(xpos, ypos)
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Capture mouse on first click
            if not self.mouse_captured:
                self.capture_mouse()
        
        return True
    
    def _handle_key_down(self, key: int):
        """Handle key press events."""
        if key == pygame.K_w:
            self.camera.process_keyboard(Camera.FORWARD, True)
        elif key == pygame.K_s:
            self.camera.process_keyboard(Camera.BACKWARD, True)
        elif key == pygame.K_a:
            self.camera.process_keyboard(Camera.LEFT, True)
        elif key == pygame.K_d:
            self.camera.process_keyboard(Camera.RIGHT, True)
        elif key == pygame.K_SPACE:
            self.camera.process_keyboard(Camera.UP, True)
        elif key == pygame.K_LSHIFT or key == pygame.K_RSHIFT:
            self.camera.process_keyboard(Camera.DOWN, True)
        elif key == pygame.K_ESCAPE:
            # Release mouse on escape
            self.release_mouse()
    
    def _handle_key_up(self, key: int):
        """Handle key release events."""
        if key == pygame.K_w:
            self.camera.process_keyboard(Camera.FORWARD, False)
        elif key == pygame.K_s:
            self.camera.process_keyboard(Camera.BACKWARD, False)
        elif key == pygame.K_a:
            self.camera.process_keyboard(Camera.LEFT, False)
        elif key == pygame.K_d:
            self.camera.process_keyboard(Camera.RIGHT, False)
        elif key == pygame.K_SPACE:
            self.camera.process_keyboard(Camera.UP, False)
        elif key == pygame.K_LSHIFT or key == pygame.K_RSHIFT:
            self.camera.process_keyboard(Camera.DOWN, False)
    
    def capture_mouse(self):
        """Capture and hide the mouse cursor for FPS controls."""
        pygame.mouse.set_visible(False)
        pygame.mouse.set_pos(
            pygame.display.get_window_size()[0] // 2,
            pygame.display.get_window_size()[1] // 2
        )
        self.mouse_captured = True
        self.camera.first_mouse = True
    
    def release_mouse(self):
        """Release the mouse cursor."""
        pygame.mouse.set_visible(True)
        self.mouse_captured = False
    
    def is_mouse_captured(self) -> bool:
        """Check if mouse is currently captured."""
        return self.mouse_captured
