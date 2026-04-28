"""
Texture loading and management module.
Handles loading images from files and creating OpenGL textures.
"""

import OpenGL.GL as gl
from typing import Optional, Tuple
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class Texture:
    """
    Manages OpenGL texture objects with support for various formats.
    Handles loading, filtering, and wrapping modes.
    """
    
    def __init__(self):
        self.texture_id: Optional[int] = None
        self.width: int = 0
        self.height: int = 0
        self.channels: int = 0
    
    def load(self, path: str, flip_v: bool = True) -> 'Texture':
        """
        Load a texture from an image file.
        
        Args:
            path: Path to the image file (PNG, JPG, etc.)
            flip_v: Whether to flip the image vertically (OpenGL expects origin at bottom-left)
        
        Returns:
            Self for method chaining
        
        Raises:
            ImportError: If PIL is not installed
            FileNotFoundError: If the image file doesn't exist
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL/Pillow is required for texture loading. "
                "Install it with: pip install Pillow"
            )
        
        # Load and process image
        img = Image.open(path)
        
        # Flip vertically for OpenGL (unless already flipped)
        if flip_v:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Convert to appropriate format
        if img.mode == 'RGB':
            self.channels = 3
            format = gl.GL_RGB
        elif img.mode == 'RGBA':
            self.channels = 4
            format = gl.GL_RGBA
        elif img.mode == 'L':
            self.channels = 1
            format = gl.GL_RED
        else:
            # Convert to RGBA as fallback
            img = img.convert('RGBA')
            self.channels = 4
            format = gl.GL_RGBA
        
        self.width, self.height = img.size
        
        # Convert to numpy array
        img_data = np.array(img, dtype=np.uint8)
        
        # Create OpenGL texture
        self._create_texture(img_data, format)
        
        return self
    
    def create_from_color(self, width: int, height: int, color: Tuple[int, int, int]) -> 'Texture':
        """
        Create a solid color texture programmatically.
        
        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            color: RGB color tuple (0-255)
        
        Returns:
            Self for method chaining
        """
        self.width = width
        self.height = height
        self.channels = 3
        
        # Create solid color image data
        img_data = np.full((height, width, 3), color, dtype=np.uint8)
        
        self._create_texture(img_data, gl.GL_RGB)
        
        return self
    
    def _create_texture(self, data: np.ndarray, format: int):
        """
        Internal method to create OpenGL texture from pixel data.
        
        Args:
            data: Pixel data as numpy array
            format: OpenGL format (GL_RGB, GL_RGBA, etc.)
        """
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        # Upload texture data
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,  # mipmap level
            format,  # internal format
            self.width,
            self.height,
            0,  # border (must be 0)
            format,  # format
            gl.GL_UNSIGNED_BYTE,
            data
        )
        
        # Generate mipmaps
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        
        # Unbind
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def bind(self, unit: int = 0):
        """
        Bind this texture to a texture unit.
        
        Args:
            unit: Texture unit (0-31 typically)
        """
        if self.texture_id is None:
            return
        
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
    
    def unbind(self):
        """Unbind the current texture from TEXTURE_2D."""
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def delete(self):
        """Delete the OpenGL texture and free GPU memory."""
        if self.texture_id is not None:
            gl.glDeleteTextures(1, [self.texture_id])
            self.texture_id = None


class TextureArray(Texture):
    """
    Texture array for efficiently storing multiple textures
    that can be sampled via an index in the shader.
    """
    
    def __init__(self):
        super().__init__()
        self.depth: int = 0  # Number of layers
    
    def load_multiple(self, paths: list) -> 'TextureArray':
        """
        Load multiple images into a texture array.
        All images must have the same dimensions.
        
        Args:
            paths: List of image file paths
        
        Returns:
            Self for method chaining
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for texture loading.")
        
        if len(paths) == 0:
            return self
        
        # Load first image to get dimensions
        img = Image.open(paths[0])
        if img.mode != 'RGB' and img.mode != 'RGBA':
            img = img.convert('RGBA')
        self.width, self.height = img.size
        self.channels = 4  # Always use RGBA for arrays
        self.depth = len(paths)
        
        # Load all images
        layers = []
        for path in paths:
            img = Image.open(path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            layers.append(np.array(img, dtype=np.uint8))
        
        # Stack into single array
        data = np.stack(layers, axis=0)
        
        # Create texture array
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.texture_id)
        
        # Set parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        # Upload data
        gl.glTexImage3D(
            gl.GL_TEXTURE_2D_ARRAY,
            0,
            gl.GL_RGBA,
            self.width,
            self.height,
            self.depth,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            data
        )
        
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
        
        return self
    
    def bind(self, unit: int = 0):
        """Bind the texture array."""
        if self.texture_id is None:
            return
        
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.texture_id)
