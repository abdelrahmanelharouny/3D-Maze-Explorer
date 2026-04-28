# 🧩 3D Maze Explorer

A production-quality, real-time 3D maze exploration game built with Python, Pygame, and PyOpenGL. Features procedural maze generation, first-person controls, textured rendering with lighting, and collision detection.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3+-green.svg)

---

## 🎮 Features

- **Procedural Maze Generation**: Recursive backtracking (DFS) algorithm creates unique, solvable mazes
- **First-Person Camera**: Smooth FPS-style controls with mouse look and WASD movement
- **3D Rendering**: OpenGL-based rendering with VAOs/VBOs, texture mapping, and lighting
- **Collision Detection**: AABB-based collision prevents walking through walls
- **Dynamic Lighting**: Ambient + diffuse + specular (Blinn-Phong) lighting model
- **Procedural Textures**: Auto-generated brick, floor tile, and ceiling textures
- **Goal System**: Navigate to the end of the maze to win
- **Debug Features**: Wireframe mode, FPS counter, position display

---

## 📋 Requirements

### Python Packages

```bash
pip install pygame PyOpenGL PyOpenGL_accelerate numpy Pillow
```

### System Requirements

- Python 3.10 or higher
- OpenGL 3.3+ compatible GPU
- Windowing system (X11, Wayland, Windows, macOS)

---

## 🚀 Quick Start

### Running the Game

```bash
cd /workspace
python main.py
```

### Controls

| Key/Action | Description |
|------------|-------------|
| **W** | Move forward |
| **S** | Move backward |
| **A** | Move left (strafe) |
| **D** | Move right (strafe) |
| **Mouse** | Look around |
| **Left Click** | Capture mouse |
| **ESC** | Release mouse |
| **Space** | Move up (fly) |
| **Shift** | Move down (fly) |
| **F** | Toggle wireframe mode |
| **R** | Regenerate maze |
| **Q** | Quit game |

---

## 🏗️ Architecture

### Project Structure

```
project_root/
│
├── main.py              # Game entry point and main loop
├── config.py            # Centralized configuration
│
├── engine/              # Core rendering engine
│   ├── renderer.py      # Scene rendering and draw calls
│   ├── camera.py        # First-person camera system
│   ├── shader.py        # GLSL shader management
│   ├── mesh.py          # VAO/VBO/EBO mesh handling
│   └── texture.py       # Texture loading and management
│
├── maze/                # Maze generation
│   ├── generator.py     # Recursive backtracking algorithm
│   └── maze_model.py    # 3D maze representation
│
├── game/                # Game logic
│   ├── player.py        # Player controller with collision
│   ├── collision.py     # Collision detection utilities
│   └── input_handler.py # Keyboard/mouse input processing
│
├── assets/
│   ├── textures/        # Texture images (optional)
│   └── shaders/         # GLSL shader files
│       ├── vertex.glsl
│       └── fragment.glsl
│
└── utils/
    ├── math_utils.py    # Matrix/vector math functions
    ├── texture_generator.py  # Procedural texture generation
    └── constants.py     # Shared constants
```

---

## 🔧 Configuration

Edit `config.py` to customize game settings:

```python
# Window Settings
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
WINDOW_TITLE = "3D Maze Explorer"

# Camera Settings
FOV = 75.0           # Field of view
CAMERA_SPEED = 5.0   # Movement speed
MOUSE_SENSITIVITY = 0.15

# Maze Settings
MAZE_SIZE = 15       # Grid size (NxN)
CELL_SIZE = 2.0      # Cell size in world units
WALL_HEIGHT = 3.0

# Lighting
AMBIENT_STRENGTH = 0.3
DIFFUSE_STRENGTH = 0.8
```

---

## 🎨 Technical Details

### Rendering Pipeline

1. **Vertex Shader**: Transforms vertices to clip space, passes UV coords and normals
2. **Fragment Shader**: Samples textures, applies Blinn-Phong lighting
3. **Geometry**: Cube meshes for walls, plane meshes for floor/ceiling
4. **Instancing**: Walls are batched for efficient rendering

### Maze Generation

The **Recursive Backtracking** algorithm:
1. Start with a grid of cells, all walls present
2. Mark current cell as visited
3. While there are unvisited cells:
   - If current has unvisited neighbors:
     - Choose random neighbor
     - Remove wall between them
     - Push current to stack, make neighbor current
   - Else pop from stack

This creates a **perfect maze** (no loops, exactly one path between any two points).

### Collision Detection

- Player represented as a circle (top-down) with configurable radius
- Walls are AABBs (Axis-Aligned Bounding Boxes)
- Collision resolved by separating axis with minimum penetration
- X and Z axes handled independently for smooth sliding

---

## 🛠️ Development

### Adding Custom Textures

Place image files in `assets/textures/` and update `config.py`:

```python
TEXTURE_WALL = "assets/textures/my_wall.png"
TEXTURE_FLOOR = "assets/textures/my_floor.png"
```

Supported formats: PNG, JPG, BMP (via Pillow)

### Extending the Engine

#### Adding New Mesh Types

```python
from engine.mesh import Mesh
from utils.math_utils import create_box_mesh

vertices, indices = create_box_mesh(width=2.0, height=1.0, depth=1.0)
mesh = Mesh().create(vertices, indices)
```

#### Creating Custom Shaders

1. Write GLSL files in `assets/shaders/`
2. Load with `Shader().load_from_file("vertex.glsl", "fragment.glsl")`

---

## 🐛 Troubleshooting

### "No module named 'pygame'"

```bash
pip install pygame
```

### "OpenGL not available"

Ensure you have proper GPU drivers installed and PyOpenGL:

```bash
pip install PyOpenGL PyOpenGL_accelerate
```

### Black screen / No textures

The game auto-generates procedural textures if image files aren't found. To use custom textures, ensure:
1. Image files exist at specified paths
2. Pillow is installed: `pip install Pillow`

### Low FPS

- Reduce `MAZE_SIZE` in config.py
- Lower `SCREEN_WIDTH` and `SCREEN_HEIGHT`
- Enable wireframe mode (F key) for debugging

---

## 📝 License

This project is provided as-is for educational purposes.

---

## 🎯 Future Enhancements

Potential features for extension:

- [ ] Mini-map overlay
- [ ] Enemy AI with pathfinding
- [ ] Collectibles and scoring
- [ ] Sound effects and music
- [ ] Fog effect for atmosphere
- [ ] Multiple maze themes
- [ ] Save/load system
- [ ] VR support

---

## 👨‍💻 Author

Built as a demonstration of modern OpenGL practices and game architecture in Python.

Enjoy exploring the maze! 🧭