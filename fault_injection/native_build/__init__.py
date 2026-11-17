"""Reviewed native recipes, immutable builds and independent load checks.

No device library is imported and no device or Worker is created by this API.
Build success is deliberately separate from provider qualification.
"""
from .manager import BuildError, NativeBuildManager
from .recipes import recipes, resolve_recipe

__all__ = ["BuildError", "NativeBuildManager", "recipes", "resolve_recipe"]
