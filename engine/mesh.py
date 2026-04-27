"""
Mesh module for creating and managing OpenGL vertex buffers.
Handles VAO, VBO, and EBO creation for efficient GPU rendering.
"""

import OpenGL.GL as gl
import numpy as np
from typing import Optional


class Mesh:
    """
    Manages vertex data and OpenGL buffer objects for 3D geometry.
    Supports indexed drawing with VAO, VBO, and EBO.
    """
    
    # Vertex attribute layout
    ATTRIB_POSITION = 0  # vec3 (x, y, z)
    ATTRIB_TEXCOORD = 1  # vec2 (u, v)
    ATTRIB_NORMAL = 2    # vec3 (nx, ny, nz)
    
    VERTEX_SIZE = 8  # floats per vertex (3 pos + 2 tex + 3 normal)
    VERTEX_STRIDE = VERTEX_SIZE * 4  # bytes per vertex
    
    def __init__(self):
        self.vao: Optional[int] = None
        self.vbo: Optional[int] = None
        self.ebo: Optional[int] = None
        self.index_count: int = 0
        self.vertex_count: int = 0
    
    def create(self, vertices: np.ndarray, indices: np.ndarray) -> 'Mesh':
        """
        Create VAO, VBO, and EBO from vertex and index data.
        
        Args:
            vertices: Nx8 array (position xyz, texcoord uv, normal xyz)
            indices: Face indices for indexed drawing
        
        Returns:
            Self for method chaining
        """
        self.vertex_count = len(vertices)
        self.index_count = len(indices)
        
        # Generate buffers
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)
        
        # Bind VAO first - all subsequent buffer calls affect this VAO
        gl.glBindVertexArray(self.vao)
        
        # Setup VBO (vertex data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, 
            vertices.nbytes, 
            vertices, 
            gl.GL_STATIC_DRAW
        )
        
        # Setup EBO (index data)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            indices,
            gl.GL_STATIC_DRAW
        )
        
        # Configure vertex attributes
        # Position attribute (location = 0)
        gl.glEnableVertexAttribArray(self.ATTRIB_POSITION)
        gl.glVertexAttribPointer(
            self.ATTRIB_POSITION,
            3,  # size (vec3)
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.VERTEX_STRIDE,
            None  # offset (start of vertex)
        )
        
        # Texture coordinate attribute (location = 1)
        gl.glEnableVertexAttribArray(self.ATTRIB_TEXCOORD)
        gl.glVertexAttribPointer(
            self.ATTRIB_TEXCOORD,
            2,  # size (vec2)
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.VERTEX_STRIDE,
            ctypes_offset(3 * 4)  # offset after 3 floats (position)
        )
        
        # Normal attribute (location = 2)
        gl.glEnableVertexAttribArray(self.ATTRIB_NORMAL)
        gl.glVertexAttribPointer(
            self.ATTRIB_NORMAL,
            3,  # size (vec3)
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.VERTEX_STRIDE,
            ctypes_offset(5 * 4)  # offset after 5 floats (position + texcoord)
        )
        
        # Unbind VAO
        gl.glBindVertexArray(0)
        
        return self
    
    def draw(self):
        """Render the mesh using indexed drawing."""
        if self.vao is None:
            return
        
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(
            gl.GL_TRIANGLES,
            self.index_count,
            gl.GL_UNSIGNED_INT,
            None
        )
        gl.glBindVertexArray(0)
    
    def draw_wireframe(self):
        """Render the mesh in wireframe mode."""
        if self.vao is None:
            return
        
        gl.glBindVertexArray(self.vao)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawElements(
            gl.GL_TRIANGLES,
            self.index_count,
            gl.GL_UNSIGNED_INT,
            None
        )
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glBindVertexArray(0)
    
    def delete(self):
        """Delete all OpenGL buffer objects."""
        if self.vao is not None:
            gl.glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        if self.vbo is not None:
            gl.glDeleteBuffers(1, [self.vbo])
            self.vbo = None
        if self.ebo is not None:
            gl.glDeleteBuffers(1, [self.ebo])
            self.ebo = None


def ctypes_offset(offset: int):
    """
    Create a ctypes void pointer for OpenGL offset calculations.
    
    Args:
        offset: Byte offset into the vertex data
    
    Returns:
        ctypes void pointer
    """
    import ctypes
    return ctypes.c_void_p(offset)


class InstancedMesh(Mesh):
    """
    Extended mesh that supports instanced rendering for drawing
    many copies of the same geometry efficiently.
    """
    
    def __init__(self):
        super().__init__()
        self.instance_vbo: Optional[int] = None
        self.instance_count: int = 0
    
    def set_instances(self, instance_data: np.ndarray) -> 'InstancedMesh':
        """
        Set per-instance data (e.g., model matrices).
        
        Args:
            instance_data: Nx16 array of model matrices (flattened)
        
        Returns:
            Self for method chaining
        """
        self.instance_count = len(instance_data)
        
        if self.instance_vbo is None:
            self.instance_vbo = gl.glGenBuffers(1)
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            instance_data.nbytes,
            instance_data,
            gl.GL_STATIC_DRAW
        )
        
        # Set up instanced attributes for mat4 (4 vec4 attributes)
        # These use locations 3, 4, 5, 6
        for i in range(4):
            loc = 3 + i
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(
                loc,
                4,  # vec4
                gl.GL_FLOAT,
                gl.GL_FALSE,
                16 * 4,  # stride (size of mat4)
                ctypes_offset(i * 4 * 4)  # offset within mat4
            )
            # This attribute advances once per instance, not per vertex
            gl.glVertexAttribDivisor(loc, 1)
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        
        return self
    
    def draw_instanced(self):
        """Render all instances with a single draw call."""
        if self.vao is None or self.instance_count == 0:
            return
        
        gl.glBindVertexArray(self.vao)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES,
            self.index_count,
            gl.GL_UNSIGNED_INT,
            None,
            self.instance_count
        )
        gl.glBindVertexArray(0)
