"""
Shader management module for loading and compiling GLSL shaders.
Handles vertex and fragment shader compilation, linking, and uniform management.
"""

import OpenGL.GL as gl
from typing import Optional, Dict, Any
import numpy as np


class Shader:
    """
    Manages GLSL shader programs including compilation, linking, and uniform access.
    """
    
    def __init__(self):
        self.program_id: Optional[int] = None
        self.uniform_cache: Dict[str, int] = {}
    
    def compile_shader(self, source: str, shader_type: int) -> int:
        """
        Compile a single shader from source code.
        
        Args:
            source: GLSL source code
            shader_type: gl.GL_VERTEX_SHADER or gl.GL_FRAGMENT_SHADER
        
        Returns:
            Shader object ID
        
        Raises:
            RuntimeError: If compilation fails
        """
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        
        # Check compilation status
        success = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if not success:
            info_log = gl.glGetShaderInfoLog(shader).decode('utf-8')
            gl.glDeleteShader(shader)
            raise RuntimeError(f"Shader compilation failed: {info_log}")
        
        return shader
    
    def load_from_source(self, vertex_source: str, fragment_source: str) -> 'Shader':
        """
        Load and compile shaders from source strings.
        
        Args:
            vertex_source: GLSL vertex shader source
            fragment_source: GLSL fragment shader source
        
        Returns:
            Self for method chaining
        """
        # Compile individual shaders
        vertex_shader = self.compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
        
        # Create program and attach shaders
        self.program_id = gl.glCreateProgram()
        gl.glAttachShader(self.program_id, vertex_shader)
        gl.glAttachShader(self.program_id, fragment_shader)
        gl.glLinkProgram(self.program_id)
        
        # Check linking status
        success = gl.glGetProgramiv(self.program_id, gl.GL_LINK_STATUS)
        if not success:
            info_log = gl.glGetProgramInfoLog(self.program_id).decode('utf-8')
            raise RuntimeError(f"Shader program linking failed: {info_log}")
        
        # Clean up compiled shaders (they're now linked to the program)
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        
        return self
    
    def load_from_file(self, vertex_path: str, fragment_path: str) -> 'Shader':
        """
        Load and compile shaders from files.
        
        Args:
            vertex_path: Path to vertex shader file
            fragment_path: Path to fragment shader file
        
        Returns:
            Self for method chaining
        """
        with open(vertex_path, 'r') as f:
            vertex_source = f.read()
        
        with open(fragment_path, 'r') as f:
            fragment_source = f.read()
        
        return self.load_from_source(vertex_source, fragment_source)
    
    def use(self) -> 'Shader':
        """Activate this shader program for rendering."""
        gl.glUseProgram(self.program_id)
        return self
    
    def get_uniform_location(self, name: str) -> int:
        """
        Get the location of a uniform variable.
        Uses caching to avoid repeated OpenGL queries.
        
        Args:
            name: Uniform variable name
        
        Returns:
            Uniform location (-1 if not found)
        """
        if name not in self.uniform_cache:
            self.uniform_cache[name] = gl.glGetUniformLocation(self.program_id, name)
        return self.uniform_cache[name]
    
    def set_bool(self, name: str, value: bool) -> 'Shader':
        """Set a bool uniform."""
        gl.glUniform1i(self.get_uniform_location(name), int(value))
        return self
    
    def set_int(self, name: str, value: int) -> 'Shader':
        """Set an int uniform."""
        gl.glUniform1i(self.get_uniform_location(name), value)
        return self
    
    def set_float(self, name: str, value: float) -> 'Shader':
        """Set a float uniform."""
        gl.glUniform1f(self.get_uniform_location(name), value)
        return self
    
    def set_vec3(self, name: str, value: np.ndarray) -> 'Shader':
        """Set a vec3 uniform."""
        gl.glUniform3fv(self.get_uniform_location(name), 1, value.astype(np.float32))
        return self
    
    def set_mat4(self, name: str, matrix: np.ndarray) -> 'Shader':
        """
        Set a mat4 uniform.
        
        Args:
            name: Uniform name
            matrix: 4x4 numpy array (will be transposed for OpenGL)
        """
        # OpenGL expects column-major order, numpy uses row-major
        gl.glUniformMatrix4fv(
            self.get_uniform_location(name), 
            1, 
            gl.GL_TRUE,  # transpose
            matrix.astype(np.float32)
        )
        return self
    
    def delete(self):
        """Delete the shader program and free GPU resources."""
        if self.program_id is not None:
            gl.glDeleteProgram(self.program_id)
            self.program_id = None
            self.uniform_cache.clear()
