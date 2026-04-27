"""
First-person camera system for 3D navigation.
Implements FPS-style controls with mouse look and WASD movement.
"""

import numpy as np
from utils.math_utils import radians, normalize, clamp


class Camera:
    """
    First-person camera with mouse look and keyboard movement.
    Manages view and projection matrices for 3D rendering.
    """
    
    # Movement directions
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5
    
    def __init__(
        self,
        position: np.ndarray = None,
        fov: float = 75.0,
        aspect: float = 16.0 / 9.0,
        near: float = 0.1,
        far: float = 100.0,
        speed: float = 5.0,
        sensitivity: float = 0.15
    ):
        """
        Initialize the camera.
        
        Args:
            position: Initial camera position (x, y, z)
            fov: Field of view in degrees
            aspect: Aspect ratio (width / height)
            near: Near clipping plane
            far: Far clipping plane
            speed: Movement speed (units per second)
            sensitivity: Mouse sensitivity
        """
        self.position = position if position is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Camera orientation (Euler angles)
        self.yaw = -90.0   # Facing -Z initially
        self.pitch = 0.0   # Level
        
        # Camera vectors
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Projection settings
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        
        # Movement settings
        self.speed = speed
        self.sensitivity = sensitivity
        
        # State
        self.first_mouse = True
        self.last_x = 0.0
        self.last_y = 0.0
        
        # Movement state
        self.movement_flags = {
            self.FORWARD: False,
            self.BACKWARD: False,
            self.LEFT: False,
            self.RIGHT: False,
            self.UP: False,
            self.DOWN: False
        }
        
        self._update_camera_vectors()
    
    def set_aspect(self, width: int, height: int):
        """Update aspect ratio when window is resized."""
        self.aspect = width / max(height, 1)
    
    def process_keyboard(self, direction: int, pressed: bool):
        """
        Handle keyboard input for movement.
        
        Args:
            direction: Movement direction constant
            pressed: Whether the key is pressed or released
        """
        self.movement_flags[direction] = pressed
    
    def process_mouse_move(self, xpos: float, ypos: float):
        """
        Handle mouse movement for camera rotation.
        
        Args:
            xpos: Mouse X position
            ypos: Mouse Y position
        """
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
            return
        
        # Calculate offset
        xoffset = (xpos - self.last_x) * self.sensitivity
        yoffset = (self.last_y - ypos) * self.sensitivity  # Invert Y for natural feel
        
        self.last_x = xpos
        self.last_y = ypos
        
        # Update yaw and pitch
        self.yaw += xoffset
        self.pitch += yoffset
        
        # Clamp pitch to prevent flipping
        self.pitch = clamp(self.pitch, -89.0, 89.0)
        
        self._update_camera_vectors()
    
    def process_scroll(self, yoffset: float):
        """
        Handle mouse scroll for FOV adjustment (zoom).
        
        Args:
            yoffset: Scroll amount
        """
        self.fov -= yoffset * 2.0  # Adjust zoom speed
        self.fov = clamp(self.fov, 1.0, 90.0)
    
    def _update_camera_vectors(self):
        """Recalculate front, right, and up vectors from Euler angles."""
        # Calculate new front vector from yaw and pitch
        yaw_rad = radians(self.yaw)
        pitch_rad = radians(self.pitch)
        
        front = np.array([
            np.cos(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad),
            np.sin(yaw_rad) * np.cos(pitch_rad)
        ], dtype=np.float32)
        
        self.front = normalize(front)
        self.right = normalize(np.cross(self.front, self.world_up))
        self.up = normalize(np.cross(self.right, self.front))
    
    def get_view_matrix(self) -> np.ndarray:
        """
        Create the view matrix using lookAt method.
        
        Returns:
            4x4 view matrix
        """
        target = self.position + self.front
        return self._look_at(self.position, target, self.up)
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Create the perspective projection matrix.
        
        Returns:
            4x4 projection matrix
        """
        return self._perspective(self.fov, self.aspect, self.near, self.far)
    
    def get_direction(self) -> np.ndarray:
        """Get the normalized forward direction vector."""
        return self.front.copy()
    
    def move(self, delta_time: float):
        """
        Update camera position based on active movement flags.
        
        Args:
            delta_time: Time elapsed since last update (seconds)
        """
        velocity = self.speed * delta_time
        
        if self.movement_flags[self.FORWARD]:
            self.position += self.front * velocity
        if self.movement_flags[self.BACKWARD]:
            self.position -= self.front * velocity
        if self.movement_flags[self.LEFT]:
            self.position -= self.right * velocity
        if self.movement_flags[self.RIGHT]:
            self.position += self.right * velocity
        if self.movement_flags[self.UP]:
            self.position += self.world_up * velocity
        if self.movement_flags[self.DOWN]:
            self.position -= self.world_up * velocity
    
    @staticmethod
    def _perspective(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix."""
        fov_rad = np.deg2rad(fov_degrees)
        tan_half_fov = np.tan(fov_rad / 2.0)
        
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = 1.0 / (aspect * tan_half_fov)
        matrix[1, 1] = 1.0 / tan_half_fov
        matrix[2, 2] = -(far + near) / (far - near)
        matrix[2, 3] = -(2.0 * far * near) / (far - near)
        matrix[3, 2] = -1.0
        
        return matrix
    
    @staticmethod
    def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create view matrix using lookAt method."""
        f = normalize(center - eye)
        r = normalize(np.cross(f, up))
        u = normalize(np.cross(r, f))
        
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, :3] = r
        matrix[1, :3] = u
        matrix[2, :3] = -f
        matrix[:3, 3] = -np.array([np.dot(r, eye), np.dot(u, eye), np.dot(-f, eye)])
        
        return matrix
