"""
Game package initialization.
Provides player, collision, and input handling components.
"""

from game.player import Player
from game.collision import aabb_collision_2d, resolve_collision
from game.input_handler import InputHandler

__all__ = ['Player', 'InputHandler', 'aabb_collision_2d', 'resolve_collision']
