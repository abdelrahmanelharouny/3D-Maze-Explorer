#!/usr/bin/env python3
"""
3D Maze Explorer - Single File Version
A first-person 3D maze navigation game using pygame and OpenGL.

Controls:
  W/A/S/D - Move
  Mouse - Look around
  Click - Capture mouse
  ESC - Release mouse
  F - Toggle wireframe
  R - Regenerate maze
  Q - Quit

Requirements:
  pip install pygame PyOpenGL numpy
"""

import sys
import time
import random
import ctypes
import numpy as np
import pygame
from OpenGL.GL import *
from typing import Optional, Dict, Any, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Window Settings
    SCREEN_WIDTH = 1024
    SCREEN_HEIGHT = 768
    WINDOW_TITLE = "3D Maze Explorer"
    TARGET_FPS = 60
    
    # Camera Settings
    FOV = 75.0
    NEAR_PLANE = 0.1
    FAR_PLANE = 100.0
    CAMERA_SPEED = 5.0
    MOUSE_SENSITIVITY = 0.15
    PITCH_CLAMP_MIN = -89.0
    PITCH_CLAMP_MAX = 89.0
    
    # Maze Settings
    MAZE_SIZE = 15
    CELL_SIZE = 2.0
    WALL_HEIGHT = 3.0
    WALL_THICKNESS = 0.2
    
    # Lighting Settings
    AMBIENT_STRENGTH = 0.3
    DIFFUSE_STRENGTH = 0.8
    SPECULAR_STRENGTH = 0.3
    LIGHT_DIRECTION = [0.5, -1.0, 0.3]
    
    # Player Settings
    PLAYER_HEIGHT = 1.7
    PLAYER_RADIUS = 0.3
    JUMP_FORCE = 0.0
    GRAVITY = 0.0
    
    # Debug Settings
    DEBUG_WIREFRAME = False
    DEBUG_SHOW_FPS = True
    DEBUG_SHOW_POSITION = True
    
    # Shader Sources (embedded)
    SHADER_VERTEX = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""
    
    SHADER_FRAGMENT = """
#version 330 core
in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;

uniform sampler2D texture1;
uniform vec3 viewPos;

struct DirLight {
    vec3 direction;
    float ambient;
    float diffuse;
    float specular;
};
uniform DirLight light;

out vec4 FragColor;

void main()
{
    vec4 texColor = texture(texture1, TexCoord);
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-light.direction);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    vec3 ambient = light.ambient * vec3(texColor.rgb);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * vec3(texColor.rgb);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), 32.0);
    vec3 specular = light.specular * spec * vec3(1.0);
    
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, texColor.a);
}
"""


# ============================================================================
# MATH UTILITIES
# ============================================================================

def radians(degrees: float) -> float:
    return np.deg2rad(degrees)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def translate(position: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = position
    return matrix


def scale_matrix(scalar: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] *= scalar
    return matrix


def create_box_mesh(width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    hw, hh, hd = width / 2.0, height / 2.0, depth / 2.0
    
    vertices = np.array([
        # Front face (z = +hd)
        [-hw, -hh, hd, 0.0, 0.0, 0.0, 0.0, 1.0],
        [hw, -hh, hd, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [hw, hh, hd, 1.0, 1.0, 0.0, 0.0, 1.0],
        [-hw, hh, hd, 0.0, 1.0, 0.0, 0.0, 1.0],
        # Back face (z = -hd)
        [hw, -hh, -hd, 0.0, 0.0, 0.0, 0.0, -1.0],
        [-hw, -hh, -hd, 1.0, 0.0, 0.0, 0.0, -1.0],
        [-hw, hh, -hd, 1.0, 1.0, 0.0, 0.0, -1.0],
        [hw, hh, -hd, 0.0, 1.0, 0.0, 0.0, -1.0],
        # Top face (y = +hh)
        [-hw, hh, hd, 0.0, 0.0, 0.0, 1.0, 0.0],
        [hw, hh, hd, 1.0, 0.0, 0.0, 1.0, 0.0],
        [hw, hh, -hd, 1.0, 1.0, 0.0, 1.0, 0.0],
        [-hw, hh, -hd, 0.0, 1.0, 0.0, 1.0, 0.0],
        # Bottom face (y = -hh)
        [-hw, -hh, -hd, 0.0, 0.0, 0.0, -1.0, 0.0],
        [hw, -hh, -hd, 1.0, 0.0, 0.0, -1.0, 0.0],
        [hw, -hh, hd, 1.0, 1.0, 0.0, -1.0, 0.0],
        [-hw, -hh, hd, 0.0, 1.0, 0.0, -1.0, 0.0],
        # Right face (x = +hw)
        [hw, -hh, hd, 0.0, 0.0, 1.0, 0.0, 0.0],
        [hw, -hh, -hd, 1.0, 0.0, 1.0, 0.0, 0.0],
        [hw, hh, -hd, 1.0, 1.0, 1.0, 0.0, 0.0],
        [hw, hh, hd, 0.0, 1.0, 1.0, 0.0, 0.0],
        # Left face (x = -hw)
        [-hw, -hh, -hd, 0.0, 0.0, -1.0, 0.0, 0.0],
        [-hw, -hh, hd, 1.0, 0.0, -1.0, 0.0, 0.0],
        [-hw, hh, hd, 1.0, 1.0, -1.0, 0.0, 0.0],
        [-hw, hh, -hd, 0.0, 1.0, -1.0, 0.0, 0.0],
    ], dtype=np.float32)
    
    indices = np.array([
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8,
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16,
        20, 21, 22, 22, 23, 20,
    ], dtype=np.uint32)
    
    return vertices, indices


def create_plane_mesh(size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    hs = size / 2.0
    vertices = np.array([
        [-hs, 0.0, -hs, 0.0, 0.0, 0.0, 1.0, 0.0],
        [hs, 0.0, -hs, 1.0, 0.0, 0.0, 1.0, 0.0],
        [hs, 0.0, hs, 1.0, 1.0, 0.0, 1.0, 0.0],
        [-hs, 0.0, hs, 0.0, 1.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    return vertices, indices


def ctypes_offset(offset: int):
    return ctypes.c_void_p(offset)


# ============================================================================
# TEXTURE GENERATION
# ============================================================================

def create_brick_texture(width: int = 256, height: int = 256,
                         brick_color: Tuple[int, int, int] = (180, 100, 80),
                         mortar_color: Tuple[int, int, int] = (100, 100, 100),
                         brick_rows: int = 8, brick_cols: int = 4) -> np.ndarray:
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    brick_height = height // brick_rows
    brick_width = width // brick_cols
    mortar_size = max(2, brick_height // 10)
    
    for y in range(height):
        row = y // brick_height
        offset = (row % 2) * (brick_width // 2)
        for x in range(width):
            col_x = (x + offset) % width
            col = col_x // brick_width
            in_horizontal_mortar = (y % brick_height) < mortar_size
            in_vertical_mortar = (col_x % brick_width) < mortar_size
            if in_horizontal_mortar or in_vertical_mortar:
                texture[y, x] = mortar_color
            else:
                noise = np.random.randint(-15, 15, 3)
                color = np.clip(np.array(brick_color) + noise, 0, 255)
                texture[y, x] = color
    return texture


def create_floor_texture(width: int = 256, height: int = 256,
                         tile_color: Tuple[int, int, int] = (120, 120, 120),
                         grout_color: Tuple[int, int, int] = (60, 60, 60),
                         tile_rows: int = 4, tile_cols: int = 4) -> np.ndarray:
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    tile_height = height // tile_rows
    tile_width = width // tile_cols
    grout_size = max(2, tile_height // 15)
    
    for y in range(height):
        for x in range(width):
            in_horizontal_grout = (y % tile_height) < grout_size
            in_vertical_grout = (x % tile_width) < grout_size
            if in_horizontal_grout or in_vertical_grout:
                texture[y, x] = grout_color
            else:
                noise = np.random.randint(-10, 10, 3)
                color = np.clip(np.array(tile_color) + noise, 0, 255)
                texture[y, x] = color
    return texture


def create_ceiling_texture(width: int = 256, height: int = 256,
                           base_color: Tuple[int, int, int] = (200, 200, 190)) -> np.ndarray:
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            noise = np.random.randint(-20, 20, 3)
            color = np.clip(np.array(base_color) + noise, 0, 255)
            texture[y, x] = color
    return texture


# ============================================================================
# SHADER CLASS
# ============================================================================

class Shader:
    def __init__(self):
        self.program_id: Optional[int] = None
        self.uniform_cache: Dict[str, int] = {}
    
    def compile_shader(self, source: str, shader_type: int) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        success = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not success:
            info_log = glGetShaderInfoLog(shader).decode('utf-8')
            glDeleteShader(shader)
            raise RuntimeError(f"Shader compilation failed: {info_log}")
        return shader
    
    def load_from_source(self, vertex_source: str, fragment_source: str) -> 'Shader':
        vertex_shader = self.compile_shader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_source, GL_FRAGMENT_SHADER)
        self.program_id = glCreateProgram()
        glAttachShader(self.program_id, vertex_shader)
        glAttachShader(self.program_id, fragment_shader)
        glLinkProgram(self.program_id)
        success = glGetProgramiv(self.program_id, GL_LINK_STATUS)
        if not success:
            info_log = glGetProgramInfoLog(self.program_id).decode('utf-8')
            raise RuntimeError(f"Shader program linking failed: {info_log}")
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        return self
    
    def use(self) -> 'Shader':
        glUseProgram(self.program_id)
        return self
    
    def get_uniform_location(self, name: str) -> int:
        if name not in self.uniform_cache:
            self.uniform_cache[name] = glGetUniformLocation(self.program_id, name)
        return self.uniform_cache[name]
    
    def set_bool(self, name: str, value: bool) -> 'Shader':
        glUniform1i(self.get_uniform_location(name), int(value))
        return self
    
    def set_int(self, name: str, value: int) -> 'Shader':
        glUniform1i(self.get_uniform_location(name), value)
        return self
    
    def set_float(self, name: str, value: float) -> 'Shader':
        glUniform1f(self.get_uniform_location(name), value)
        return self
    
    def set_vec3(self, name: str, value: np.ndarray) -> 'Shader':
        glUniform3fv(self.get_uniform_location(name), 1, value.astype(np.float32))
        return self
    
    def set_mat4(self, name: str, matrix: np.ndarray) -> 'Shader':
        glUniformMatrix4fv(self.get_uniform_location(name), 1, GL_TRUE, matrix.astype(np.float32))
        return self
    
    def delete(self):
        if self.program_id is not None:
            glDeleteProgram(self.program_id)
            self.program_id = None
            self.uniform_cache.clear()


# ============================================================================
# MESH CLASS
# ============================================================================

class Mesh:
    ATTRIB_POSITION = 0
    ATTRIB_TEXCOORD = 1
    ATTRIB_NORMAL = 2
    VERTEX_SIZE = 8
    VERTEX_STRIDE = VERTEX_SIZE * 4
    
    def __init__(self):
        self.vao: Optional[int] = None
        self.vbo: Optional[int] = None
        self.ebo: Optional[int] = None
        self.index_count: int = 0
        self.vertex_count: int = 0
    
    def create(self, vertices: np.ndarray, indices: np.ndarray) -> 'Mesh':
        self.vertex_count = len(vertices)
        self.index_count = len(indices)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(self.ATTRIB_POSITION)
        glVertexAttribPointer(self.ATTRIB_POSITION, 3, GL_FLOAT, GL_FALSE, self.VERTEX_STRIDE, None)
        
        glEnableVertexAttribArray(self.ATTRIB_TEXCOORD)
        glVertexAttribPointer(self.ATTRIB_TEXCOORD, 2, GL_FLOAT, GL_FALSE, self.VERTEX_STRIDE, ctypes_offset(3 * 4))
        
        glEnableVertexAttribArray(self.ATTRIB_NORMAL)
        glVertexAttribPointer(self.ATTRIB_NORMAL, 3, GL_FLOAT, GL_FALSE, self.VERTEX_STRIDE, ctypes_offset(5 * 4))
        
        glBindVertexArray(0)
        return self
    
    def draw(self):
        if self.vao is None:
            return
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
    
    def draw_wireframe(self):
        if self.vao is None:
            return
        glBindVertexArray(self.vao)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBindVertexArray(0)
    
    def delete(self):
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        if self.vbo is not None:
            glDeleteBuffers(1, [self.vbo])
            self.vbo = None
        if self.ebo is not None:
            glDeleteBuffers(1, [self.ebo])
            self.ebo = None


# ============================================================================
# TEXTURE CLASS
# ============================================================================

class Texture:
    def __init__(self):
        self.texture_id: Optional[int] = None
        self.width: int = 0
        self.height: int = 0
        self.channels: int = 0
    
    def create_from_data(self, data: np.ndarray) -> 'Texture':
        self.width, self.height = data.shape[1], data.shape[0]
        self.channels = 3
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        return self
    
    def bind(self, unit: int = 0):
        if self.texture_id is None:
            return
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
    
    def unbind(self):
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def delete(self):
        if self.texture_id is not None:
            glDeleteTextures(1, [self.texture_id])
            self.texture_id = None


# ============================================================================
# CAMERA CLASS
# ============================================================================

class Camera:
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5
    
    def __init__(self, position: np.ndarray = None, fov: float = 75.0,
                 aspect: float = 16.0 / 9.0, near: float = 0.1, far: float = 100.0,
                 speed: float = 5.0, sensitivity: float = 0.15):
        self.position = position if position is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.speed = speed
        self.sensitivity = sensitivity
        self.first_mouse = True
        self.last_x = 0.0
        self.last_y = 0.0
        self.movement_flags = {self.FORWARD: False, self.BACKWARD: False, self.LEFT: False,
                               self.RIGHT: False, self.UP: False, self.DOWN: False}
        self._update_camera_vectors()
    
    def set_aspect(self, width: int, height: int):
        self.aspect = width / max(height, 1)
    
    def process_keyboard(self, direction: int, pressed: bool):
        self.movement_flags[direction] = pressed
    
    def process_mouse_move(self, xpos: float, ypos: float):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
            return
        xoffset = (xpos - self.last_x) * self.sensitivity
        yoffset = (self.last_y - ypos) * self.sensitivity
        self.last_x = xpos
        self.last_y = ypos
        self.yaw += xoffset
        self.pitch += yoffset
        self.pitch = clamp(self.pitch, -89.0, 89.0)
        self._update_camera_vectors()
    
    def _update_camera_vectors(self):
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
        target = self.position + self.front
        return self._look_at(self.position, target, self.up)
    
    def get_projection_matrix(self) -> np.ndarray:
        return self._perspective(self.fov, self.aspect, self.near, self.far)
    
    def get_direction(self) -> np.ndarray:
        return self.front.copy()
    
    def move(self, delta_time: float):
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
        f = normalize(center - eye)
        r = normalize(np.cross(f, up))
        u = normalize(np.cross(r, f))
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, :3] = r
        matrix[1, :3] = u
        matrix[2, :3] = -f
        matrix[:3, 3] = -np.array([np.dot(r, eye), np.dot(u, eye), np.dot(-f, eye)])
        return matrix


# ============================================================================
# MAZE GENERATOR
# ============================================================================

class Cell:
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3
    
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.walls = [True, True, True, True]
        self.visited = False
    
    def has_wall(self, direction: int) -> bool:
        return self.walls[direction]
    
    def remove_wall(self, direction: int):
        self.walls[direction] = False


class MazeGenerator:
    def __init__(self, size: int = 15):
        self.size = size if size % 2 == 1 else size + 1
        self.grid: List[List[Cell]] = []
        self.stack: List[Cell] = []
    
    def generate(self) -> List[List[Cell]]:
        self.grid = [[Cell(r, c) for c in range(self.size)] for r in range(self.size)]
        self.stack = []
        current = self.grid[0][0]
        current.visited = True
        self.stack.append(current)
        
        while self.stack:
            neighbors = self._get_unvisited_neighbors(current)
            if neighbors:
                next_cell, direction = random.choice(neighbors)
                self._remove_walls(current, next_cell, direction)
                next_cell.visited = True
                self.stack.append(next_cell)
                current = next_cell
            else:
                current = self.stack.pop()
        return self.grid
    
    def _get_unvisited_neighbors(self, cell: Cell) -> List[Tuple[Cell, int]]:
        neighbors = []
        row, col = cell.row, cell.col
        if row > 0 and not self.grid[row - 1][col].visited:
            neighbors.append((self.grid[row - 1][col], Cell.TOP))
        if col < self.size - 1 and not self.grid[row][col + 1].visited:
            neighbors.append((self.grid[row][col + 1], Cell.RIGHT))
        if row < self.size - 1 and not self.grid[row + 1][col].visited:
            neighbors.append((self.grid[row + 1][col], Cell.BOTTOM))
        if col > 0 and not self.grid[row][col - 1].visited:
            neighbors.append((self.grid[row][col - 1], Cell.LEFT))
        return neighbors
    
    def _remove_walls(self, current: Cell, next_cell: Cell, direction: int):
        current.remove_wall(direction)
        opposite = (direction + 2) % 4
        next_cell.remove_wall(opposite)
    
    def get_wall_positions(self, cell_size: float = 2.0, wall_height: float = 3.0,
                           wall_thickness: float = 0.2) -> List[Tuple[np.ndarray, int]]:
        walls = []
        offset = (self.size * cell_size) / 2 - cell_size / 2
        for row in range(self.size):
            for col in range(self.size):
                cell = self.grid[row][col]
                cx = col * cell_size - offset
                cz = row * cell_size - offset
                if cell.has_wall(Cell.TOP):
                    pos = np.array([cx, wall_height / 2, cz - cell_size / 2], dtype=np.float32)
                    walls.append((pos, 0))
                if cell.has_wall(Cell.RIGHT):
                    pos = np.array([cx + cell_size / 2, wall_height / 2, cz], dtype=np.float32)
                    walls.append((pos, 1))
                if row == self.size - 1 and cell.has_wall(Cell.BOTTOM):
                    pos = np.array([cx, wall_height / 2, cz + cell_size / 2], dtype=np.float32)
                    walls.append((pos, 0))
                if col == 0 and cell.has_wall(Cell.LEFT):
                    pos = np.array([cx - cell_size / 2, wall_height / 2, cz], dtype=np.float32)
                    walls.append((pos, 1))
        return walls
    
    def get_collision_walls(self, cell_size: float = 2.0) -> List[Tuple[float, float, float, float]]:
        collision_walls = []
        offset = (self.size * cell_size) / 2 - cell_size / 2
        wall_half_thickness = 0.1
        for row in range(self.size):
            for col in range(self.size):
                cell = self.grid[row][col]
                cx = col * cell_size - offset
                cz = row * cell_size - offset
                if cell.has_wall(Cell.TOP):
                    x, z = cx, cz - cell_size / 2
                    collision_walls.append((x, z, cell_size, wall_half_thickness * 2))
                if cell.has_wall(Cell.RIGHT):
                    x, z = cx + cell_size / 2, cz
                    collision_walls.append((x, z, wall_half_thickness * 2, cell_size))
                if row == self.size - 1 and cell.has_wall(Cell.BOTTOM):
                    x, z = cx, cz + cell_size / 2
                    collision_walls.append((x, z, cell_size, wall_half_thickness * 2))
                if col == 0 and cell.has_wall(Cell.LEFT):
                    x, z = cx - cell_size / 2, cz
                    collision_walls.append((x, z, wall_half_thickness * 2, cell_size))
        return collision_walls
    
    def get_start_position(self, cell_size: float = 2.0) -> np.ndarray:
        offset = (self.size * cell_size) / 2 - cell_size / 2
        return np.array([0 * cell_size - offset + cell_size / 2, 1.7, 0 * cell_size - offset + cell_size / 2], dtype=np.float32)
    
    def get_end_position(self, cell_size: float = 2.0) -> np.ndarray:
        offset = (self.size * cell_size) / 2 - cell_size / 2
        return np.array([(self.size - 1) * cell_size - offset + cell_size / 2, 1.7,
                         (self.size - 1) * cell_size - offset + cell_size / 2], dtype=np.float32)


# ============================================================================
# MAZE MODEL
# ============================================================================

class MazeModel:
    def __init__(self, size: int = 15, cell_size: float = 2.0,
                 wall_height: float = 3.0, wall_thickness: float = 0.2):
        self.size = size
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.generator: Optional[MazeGenerator] = None
        self.collision_walls: List[Tuple[float, float, float, float]] = []
        self.world_bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)
        self._generate()
    
    def _generate(self):
        self.generator = MazeGenerator(self.size)
        self.generator.generate()
        self.collision_walls = self.generator.get_collision_walls(self.cell_size)
        half_extent = (self.size * self.cell_size) / 2
        self.world_bounds = (-half_extent, half_extent, -half_extent, half_extent)
    
    def regenerate(self):
        self._generate()
    
    def get_wall_data_for_renderer(self) -> List[Tuple[np.ndarray, int]]:
        return self.generator.get_wall_positions(self.cell_size, self.wall_height, self.wall_thickness)
    
    def check_collision(self, position: np.ndarray, radius: float = 0.3) -> Tuple[bool, np.ndarray]:
        x, z = position[0], position[2]
        collision = False
        resolution = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for wall_x, wall_z, wall_w, wall_d in self.collision_walls:
            min_x = wall_x - wall_w / 2 - radius
            max_x = wall_x + wall_w / 2 + radius
            min_z = wall_z - wall_d / 2 - radius
            max_z = wall_z + wall_d / 2 + radius
            if min_x <= x <= max_x and min_z <= z <= max_z:
                collision = True
                left_dist = x - min_x
                right_dist = max_x - x
                top_dist = z - min_z
                bottom_dist = max_z - z
                min_dist = min(left_dist, right_dist, top_dist, bottom_dist)
                if min_dist == left_dist:
                    resolution[0] = -left_dist
                elif min_dist == right_dist:
                    resolution[0] = right_dist
                elif min_dist == top_dist:
                    resolution[2] = -top_dist
                else:
                    resolution[2] = bottom_dist
        return collision, resolution
    
    def is_valid_position(self, position: np.ndarray, radius: float = 0.3) -> bool:
        collides, _ = self.check_collision(position, radius)
        return not collides
    
    def get_start_position(self) -> np.ndarray:
        return self.generator.get_start_position(self.cell_size)
    
    def get_end_position(self) -> np.ndarray:
        return self.generator.get_end_position(self.cell_size)
    
    def get_floor_position(self) -> np.ndarray:
        return np.array([0.0, -0.5, 0.0], dtype=np.float32)
    
    def get_ceiling_position(self) -> np.ndarray:
        return np.array([0.0, self.wall_height - 0.5, 0.0], dtype=np.float32)
    
    def check_goal_reached(self, position: np.ndarray, threshold: float = 1.0) -> bool:
        goal = self.get_end_position()
        distance = np.linalg.norm(position[:3] - goal[:3])
        return distance < threshold


# ============================================================================
# RENDERER
# ============================================================================

class Renderer:
    def __init__(self):
        self.shader: Optional[Shader] = None
        self.wall_mesh: Optional[Mesh] = None
        self.floor_mesh: Optional[Mesh] = None
        self.ceiling_mesh: Optional[Mesh] = None
        self.wall_texture: Optional[Texture] = None
        self.floor_texture: Optional[Texture] = None
        self.ceiling_texture: Optional[Texture] = None
        self.wall_instances: List[np.ndarray] = []
        self.wall_instance_matrices: Optional[np.ndarray] = None
        self.light_direction = np.array([0.5, -1.0, 0.3], dtype=np.float32)
        self.ambient_strength = 0.3
        self.diffuse_strength = 0.8
        self.specular_strength = 0.3
    
    def initialize(self, shader: Shader):
        self.shader = shader
        vertices, indices = create_box_mesh(width=1.0, height=1.0, depth=0.2)
        self.wall_mesh = Mesh().create(vertices, indices)
        vertices, indices = create_plane_mesh(size=1.0)
        self.floor_mesh = Mesh().create(vertices, indices)
        self.ceiling_mesh = self.floor_mesh
    
    def set_textures(self, wall: Optional[Texture] = None, floor: Optional[Texture] = None,
                     ceiling: Optional[Texture] = None):
        self.wall_texture = wall
        self.floor_texture = floor
        self.ceiling_texture = ceiling
    
    def set_lighting(self, direction: np.ndarray, ambient: float = 0.3,
                     diffuse: float = 0.8, specular: float = 0.3):
        self.light_direction = direction
        self.ambient_strength = ambient
        self.diffuse_strength = diffuse
        self.specular_strength = specular
    
    def add_wall_instance(self, position: np.ndarray, rotation: float = 0.0):
        model = np.eye(4, dtype=np.float32)
        model = model @ translate(position)
        if rotation != 0.0:
            rot_rad = np.deg2rad(rotation)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            rot_matrix = np.array([[cos_r, 0, sin_r, 0], [0, 1, 0, 0], [-sin_r, 0, cos_r, 0], [0, 0, 0, 1]], dtype=np.float32)
            model = model @ rot_matrix
        scale_mat = np.eye(4, dtype=np.float32)
        scale_mat[0, 0] = 1.0
        scale_mat[1, 1] = 3.0
        scale_mat[2, 2] = 0.2
        model = model @ scale_mat
        self.wall_instances.append(model.flatten())
    
    def build_instance_buffer(self):
        if len(self.wall_instances) > 0:
            self.wall_instance_matrices = np.array(self.wall_instances, dtype=np.float32).reshape(-1, 4, 4)
    
    def clear_walls(self):
        self.wall_instances = []
        self.wall_instance_matrices = None
    
    def render_scene(self, camera: Camera, floor_pos: np.ndarray = None,
                     ceiling_pos: np.ndarray = None, wireframe: bool = False):
        if self.shader is None:
            return
        self.shader.use()
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix()
        self.shader.set_mat4('view', view)
        self.shader.set_mat4('projection', projection)
        self.shader.set_vec3('viewPos', camera.position)
        self.shader.set_vec3('light.direction', self.light_direction)
        self.shader.set_float('light.ambient', self.ambient_strength)
        self.shader.set_float('light.diffuse', self.diffuse_strength)
        self.shader.set_float('light.specular', self.specular_strength)
        
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
        
        if self.wall_mesh and self.wall_texture and self.wall_instance_matrices is not None:
            self.wall_texture.bind(0)
            self.shader.set_int('texture1', 0)
            for i, mat in enumerate(self.wall_instance_matrices):
                self.shader.set_mat4('model', mat)
                if wireframe:
                    self.wall_mesh.draw_wireframe()
                else:
                    self.wall_mesh.draw()
    
    def cleanup(self):
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


# ============================================================================
# INPUT HANDLER
# ============================================================================

class InputHandler:
    def __init__(self, camera: Camera):
        self.camera = camera
        self.mouse_captured = False
    
    def handle_event(self, event: pygame.event.Event) -> bool:
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
            if not self.mouse_captured:
                self.capture_mouse()
        return True
    
    def _handle_key_down(self, key: int):
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
            self.release_mouse()
    
    def _handle_key_up(self, key: int):
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
        pygame.mouse.set_visible(False)
        pygame.mouse.set_pos(pygame.display.get_window_size()[0] // 2, pygame.display.get_window_size()[1] // 2)
        self.mouse_captured = True
        self.camera.first_mouse = True
    
    def release_mouse(self):
        pygame.mouse.set_visible(True)
        self.mouse_captured = False
    
    def is_mouse_captured(self) -> bool:
        return self.mouse_captured


# ============================================================================
# PLAYER
# ============================================================================

class Player:
    def __init__(self, camera: Camera, maze: MazeModel, move_speed: float = 5.0,
                 radius: float = 0.3, height: float = 1.7):
        self.camera = camera
        self.maze = maze
        self.move_speed = move_speed
        self.radius = radius
        self.height = height
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.on_ground = True
        self.respawn()
    
    def respawn(self):
        start_pos = self.maze.get_start_position()
        self.camera.position = start_pos.copy()
        self.velocity = np.zeros(3, dtype=np.float32)
    
    def update(self, delta_time: float):
        move_direction = np.zeros(3, dtype=np.float32)
        if self.camera.movement_flags[Camera.FORWARD]:
            move_direction += self.camera.front
        if self.camera.movement_flags[Camera.BACKWARD]:
            move_direction -= self.camera.front
        if self.camera.movement_flags[Camera.LEFT]:
            move_direction -= self.camera.right
        if self.camera.movement_flags[Camera.RIGHT]:
            move_direction += self.camera.right
        if np.linalg.norm(move_direction) > 0:
            move_direction = move_direction / np.linalg.norm(move_direction)
        displacement = move_direction * self.move_speed * delta_time
        
        new_pos = self.camera.position.copy()
        new_pos[0] += displacement[0]
        collides, resolution = self.maze.check_collision(new_pos, self.radius)
        if collides:
            test_pos = self.camera.position.copy()
            test_pos[0] += displacement[0]
            if self.maze.is_valid_position(test_pos, self.radius):
                self.camera.position[0] = test_pos[0]
        else:
            self.camera.position[0] = new_pos[0]
        
        new_pos = self.camera.position.copy()
        new_pos[2] += displacement[2]
        collides, resolution = self.maze.check_collision(new_pos, self.radius)
        if collides:
            test_pos = self.camera.position.copy()
            test_pos[2] += displacement[2]
            if self.maze.is_valid_position(test_pos, self.radius):
                self.camera.position[2] = test_pos[2]
        else:
            self.camera.position[2] = new_pos[2]
        
        self.camera.position[1] = self.height
    
    def get_position(self) -> np.ndarray:
        return self.camera.position.copy()
    
    def get_direction(self) -> np.ndarray:
        return self.camera.get_direction()
    
    def has_won(self) -> bool:
        return self.maze.check_goal_reached(self.get_position())


# ============================================================================
# MAIN GAME CLASS
# ============================================================================

class Game:
    def __init__(self):
        self.running = False
        self.clock = None
        self.screen = None
        self.font = None
        self.camera: Camera = None
        self.renderer: Renderer = None
        self.maze: MazeModel = None
        self.player: Player = None
        self.input_handler: InputHandler = None
        self.delta_time = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.fps_update_timer = 0.0
        self.wireframe_mode = Config.DEBUG_WIREFRAME
        self.game_won = False
    
    def initialize(self):
        print("Initializing 3D Maze Explorer...")
        pygame.init()
        pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(Config.WINDOW_TITLE)
        self.screen = pygame.display.get_surface()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self._init_opengl()
        
        print("Loading shaders...")
        self.shader = Shader().load_from_source(Config.SHADER_VERTEX, Config.SHADER_FRAGMENT)
        
        print("Setting up camera...")
        self.camera = Camera(
            position=np.array([0.0, 1.7, 0.0], dtype=np.float32),
            fov=Config.FOV,
            aspect=Config.SCREEN_WIDTH / Config.SCREEN_HEIGHT,
            near=Config.NEAR_PLANE,
            far=Config.FAR_PLANE,
            speed=Config.CAMERA_SPEED,
            sensitivity=Config.MOUSE_SENSITIVITY
        )
        
        print(f"Generating maze ({Config.MAZE_SIZE}x{Config.MAZE_SIZE})...")
        self.maze = MazeModel(size=Config.MAZE_SIZE, cell_size=Config.CELL_SIZE,
                              wall_height=Config.WALL_HEIGHT, wall_thickness=Config.WALL_THICKNESS)
        
        start_pos = self.maze.get_start_position()
        self.camera.position = start_pos.copy()
        
        print("Creating player controller...")
        self.player = Player(camera=self.camera, maze=self.maze, move_speed=Config.CAMERA_SPEED,
                             radius=Config.PLAYER_RADIUS, height=Config.PLAYER_HEIGHT)
        
        self.input_handler = InputHandler(self.camera)
        
        print("Setting up renderer...")
        self.renderer = Renderer()
        self.renderer.initialize(self.shader)
        
        print("Generating textures...")
        self._setup_textures()
        self._build_maze_geometry()
        
        self.renderer.set_lighting(
            direction=np.array(Config.LIGHT_DIRECTION, dtype=np.float32),
            ambient=Config.AMBIENT_STRENGTH,
            diffuse=Config.DIFFUSE_STRENGTH,
            specular=Config.SPECULAR_STRENGTH
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
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.15, 1.0)
    
    def _setup_textures(self):
        print("Using procedural textures...")
        wall_data = create_brick_texture()
        wall_tex = Texture()
        wall_tex.create_from_data(wall_data)
        
        floor_data = create_floor_texture()
        floor_tex = Texture()
        floor_tex.create_from_data(floor_data)
        
        ceiling_data = create_ceiling_texture()
        ceiling_tex = Texture()
        ceiling_tex.create_from_data(ceiling_data)
        
        self.renderer.set_textures(wall=wall_tex, floor=floor_tex, ceiling=ceiling_tex)
    
    def _build_maze_geometry(self):
        self.renderer.clear_walls()
        wall_data = self.maze.get_wall_data_for_renderer()
        for pos, orientation in wall_data:
            rotation = 90.0 if orientation == 1 else 0.0
            self.renderer.add_wall_instance(pos.copy(), rotation)
        self.renderer.build_instance_buffer()
    
    def run(self):
        if not self.running:
            return
        print("Starting game loop...")
        while self.running:
            self.delta_time = self.clock.tick(Config.TARGET_FPS) / 1000.0
            self._update_fps()
            self._handle_events()
            if not self.running:
                break
            self._update()
            self._render()
        self.shutdown()
    
    def _handle_events(self):
        for event in pygame.event.get():
            if not self.input_handler.handle_event(event):
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.wireframe_mode = not self.wireframe_mode
                elif event.key == pygame.K_r:
                    self._regenerate_maze()
                elif event.key == pygame.K_q:
                    self.running = False
    
    def _update(self):
        self.player.update(self.delta_time)
        if not self.game_won and self.player.has_won():
            self.game_won = True
            print("\n🎉 Congratulations! You reached the goal! 🎉\n")
    
    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        width, height = pygame.display.get_window_size()
        self.camera.set_aspect(width, height)
        self.renderer.render_scene(
            camera=self.camera,
            floor_pos=self.maze.get_floor_position(),
            ceiling_pos=self.maze.get_ceiling_position(),
            wireframe=self.wireframe_mode
        )
        self._render_ui()
        pygame.display.flip()
    
    def _render_ui(self):
        if Config.DEBUG_SHOW_FPS:
            fps_text = f"FPS: {self.fps:.1f}"
            fps_surface = self.font.render(fps_text, True, (255, 255, 255))
            self.screen.blit(fps_surface, (10, 10))
        
        if Config.DEBUG_SHOW_POSITION:
            pos = self.player.get_position()
            pos_text = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            pos_surface = self.font.render(pos_text, True, (255, 255, 255))
            self.screen.blit(pos_surface, (10, 40))
        
        if self.game_won:
            win_text = "🎉 GOAL REACHED! Press R for new maze 🎉"
            win_surface = self.font.render(win_text, True, (0, 255, 0))
            text_rect = win_surface.get_rect(center=(Config.SCREEN_WIDTH // 2, 50))
            self.screen.blit(win_surface, text_rect)
        
        if not self.input_handler.is_mouse_captured():
            hint_text = "Click to capture mouse"
            hint_surface = self.font.render(hint_text, True, (200, 200, 200))
            text_rect = hint_surface.get_rect(center=(Config.SCREEN_WIDTH // 2, Config.SCREEN_HEIGHT - 30))
            self.screen.blit(hint_surface, text_rect)
    
    def _update_fps(self):
        self.frame_count += 1
        self.fps_update_timer += self.delta_time
        if self.fps_update_timer >= 0.5:
            self.fps = self.frame_count / self.fps_update_timer
            self.frame_count = 0
            self.fps_update_timer = 0.0
    
    def _regenerate_maze(self):
        print("Generating new maze...")
        self.maze.regenerate()
        self._build_maze_geometry()
        start_pos = self.maze.get_start_position()
        self.camera.position = start_pos.copy()
        self.player.respawn()
        self.game_won = False
    
    def shutdown(self):
        print("Shutting down...")
        if self.renderer:
            self.renderer.cleanup()
        pygame.quit()
        print("Goodbye!")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
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
