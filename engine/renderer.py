"""
Renderer module for the 3D maze explorer.
Handles OpenGL initialization, scene rendering, and draw calls.
"""

import OpenGL.GL as gl
import numpy as np
from typing import Optional, List, Dict, Any

from engine.shader import Shader
from engine.mesh import Mesh
from engine.texture import Texture
from engine.camera import Camera
from utils.math_utils import translate, scale


class Renderer:
    """
    Main renderer class that manages all OpenGL rendering operations.
    Handles scene setup, object batching, and draw calls.
    """
    
    def __init__(self):
        self.shader: Optional[Shader] = None
        self.wall_mesh: Optional[Mesh] = None
        self.floor_mesh: Optional[Mesh] = None
        self.ceiling_mesh: Optional[Mesh] = None
        
        # Textures
        self.wall_texture: Optional[Texture] = None
        self.floor_texture: Optional[Texture] = None
        self.ceiling_texture: Optional[Texture] = None
        
        # Instance data for batched rendering
        self.wall_instances: List[np.ndarray] = []
        self.wall_instance_matrices: Optional[np.ndarray] = None
        
        # Lighting
        self.light_direction = np.array([0.5, -1.0, 0.3], dtype=np.float32)
        self.ambient_strength = 0.3
        self.diffuse_strength = 0.8
        self.specular_strength = 0.3
    
    def initialize(self, shader: Shader):
        """
        Initialize the renderer with a shader program.
        Creates default meshes for walls, floor, and ceiling.
        
        Args:
            shader: Compiled shader program to use for rendering
        """
        self.shader = shader
        
        # Create wall mesh (thin box for walls)
        from utils.math_utils import create_box_mesh
        vertices, indices = create_box_mesh(
            width=1.0,
            height=1.0,
            depth=0.2  # Thin walls
        )
        self.wall_mesh = Mesh().create(vertices, indices)
        
        # Create floor mesh (flat plane)
        from utils.math_utils import create_plane_mesh
        vertices, indices = create_plane_mesh(size=1.0)
        self.floor_mesh = Mesh().create(vertices, indices)
        
        # Ceiling uses same mesh as floor
        self.ceiling_mesh = self.floor_mesh
    
    def set_textures(
        self,
        wall: Optional[Texture] = None,
        floor: Optional[Texture] = None,
        ceiling: Optional[Texture] = None
    ):
        """Set textures for different surface types."""
        self.wall_texture = wall
        self.floor_texture = floor
        self.ceiling_texture = ceiling
    
    def set_lighting(
        self,
        direction: np.ndarray,
        ambient: float = 0.3,
        diffuse: float = 0.8,
        specular: float = 0.3
    ):
        """Configure lighting parameters."""
        self.light_direction = direction
        self.ambient_strength = ambient
        self.diffuse_strength = diffuse
        self.specular_strength = specular
    
    def add_wall_instance(self, position: np.ndarray, rotation: float = 0.0):
        """
        Add a wall instance at the given position.
        
        Args:
            position: World position (x, y, z)
            rotation: Rotation around Y axis in degrees
        """
        # Create model matrix: scale -> rotate -> translate
        model = np.eye(4, dtype=np.float32)
        model = model @ translate(position)
        
        if rotation != 0.0:
            rot_rad = np.deg2rad(rotation)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            rot_matrix = np.array([
                [cos_r, 0, sin_r, 0],
                [0, 1, 0, 0],
                [-sin_r, 0, cos_r, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            model = model @ rot_matrix
        
        # Scale to wall dimensions (height=3, width=cell_size, thickness=0.2)
        scale_matrix = np.eye(4, dtype=np.float32)
        scale_matrix[0, 0] = 1.0  # Width (will be scaled by cell size externally)
        scale_matrix[1, 1] = 3.0  # Height
        scale_matrix[2, 2] = 0.2  # Thickness
        model = model @ scale_matrix
        
        self.wall_instances.append(model.flatten())
    
    def build_instance_buffer(self):
        """Build the instance buffer from collected wall instances."""
        if len(self.wall_instances) > 0:
            self.wall_instance_matrices = np.array(
                self.wall_instances,
                dtype=np.float32
            ).reshape(-1, 4, 4)
    
    def clear_walls(self):
        """Clear all wall instances."""
        self.wall_instances = []
        self.wall_instance_matrices = None
    
    def render_scene(
        self,
        camera: Camera,
        floor_pos: np.ndarray = None,
        ceiling_pos: np.ndarray = None,
        wireframe: bool = False
    ):
        """
        Render the entire scene.
        
        Args:
            camera: Camera to use for view/projection matrices
            floor_pos: Position of the floor plane
            ceiling_pos: Position of the ceiling plane
            wireframe: Whether to render in wireframe mode
        """
        if self.shader is None:
            return
        
        self.shader.use()
        
        # Set view and projection matrices
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix()
        self.shader.set_mat4('view', view)
        self.shader.set_mat4('projection', projection)
        
        # Set camera position for specular calculations
        self.shader.set_vec3('viewPos', camera.position)
        
        # Set lighting uniforms
        self.shader.set_vec3('light.direction', self.light_direction)
        self.shader.set_float('light.ambient', self.ambient_strength)
        self.shader.set_float('light.diffuse', self.diffuse_strength)
        self.shader.set_float('light.specular', self.specular_strength)
        
        # Render floor
        if self.floor_mesh and self.floor_texture:
            self.floor_texture.bind(0)
            self.shader.set_int('texture1', 0)
            
            model = np.eye(4, dtype=np.float32)
            if floor_pos is not None:
                model = model @ translate(floor_pos)
            else:
                model = model @ translate(np.array([0.0, -0.5, 0.0]))
            
            self.shader.set_mat4('model', model)
            
            if wireframe:
                self.floor_mesh.draw_wireframe()
            else:
                self.floor_mesh.draw()
        
        # Render ceiling
        if self.ceiling_mesh and self.ceiling_texture:
            self.ceiling_texture.bind(0)
            
            model = np.eye(4, dtype=np.float32)
            if ceiling_pos is not None:
                model = model @ translate(ceiling_pos)
            else:
                model = model @ translate(np.array([0.0, 2.5, 0.0]))
            
            self.shader.set_mat4('model', model)
            
            if wireframe:
                self.ceiling_mesh.draw_wireframe()
            else:
                self.ceiling_mesh.draw()
        
        # Render walls using instanced rendering
        if self.wall_mesh and self.wall_texture and self.wall_instance_matrices is not None:
            self.wall_texture.bind(0)
            self.shader.set_int('texture1', 0)
            
            # For instanced rendering, we need to pass the matrices differently
            # Since our Mesh class doesn't support instancing natively, we'll draw individually
            # This is less efficient but works with the current architecture
            
            for i, mat in enumerate(self.wall_instance_matrices):
                self.shader.set_mat4('model', mat)
                if wireframe:
                    self.wall_mesh.draw_wireframe()
                else:
                    self.wall_mesh.draw()
    
    def cleanup(self):
        """Delete all OpenGL resources."""
        if self.wall_mesh:
            self.wall_mesh.delete()
        if self.floor_mesh:
            self.floor_mesh.delete()
        if self.shader:
            self.shader.delete()
        if self.wall_texture:
            self.wall_texture.delete()
        if self.floor_texture:
            self.floor_texture.delete()
        if self.ceiling_texture:
            self.ceiling_texture.delete()
