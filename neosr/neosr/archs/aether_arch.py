# neosr/neosr/archs/aether_arch.py

# This file acts as a bridge between the neosr framework and the aether_core factories.
# It imports the standardized factory functions and registers them so neosr can find them.

from neosr.utils.registry import ARCH_REGISTRY

# Import the factory functions from your single-source-of-truth file
from .aether_core import aether_small, aether_medium, aether_large

# Register each factory function with the framework.
# The decorator will automatically use the function's name as the key.
# So, neosr will know about "aether_small", "aether_medium", and "aether_large".
ARCH_REGISTRY.register()(aether_small)
ARCH_REGISTRY.register()(aether_medium)
ARCH_REGISTRY.register()(aether_large)

# You can add more here if you define them in aether_core.py, like aether_tiny:
#
# from .aether_core import aether_tiny
# ARCH_REGISTRY.register()(aether_tiny)