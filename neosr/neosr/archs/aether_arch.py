# neosr/neosr/archs/aether_arch.py

import sys
from pathlib import Path
# To import aether_core.py from the repository root, we need to add the root
# directory to Python's path. From this file's location, the root is 3 levels up.
# .../archs/ -> .../neosr/ -> .../neosr/ -> root
sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch
from torch import nn
from neosr.utils.registry import ArchRegistry

# Import the factory functions from your single source of truth (aether_core.py)
from aether_core import aether_small, aether_medium, aether_large

# The ArchRegistry decorator tells neosr that any model with `type: aether`
# in the YAML should be handled by this class.
@ArchRegistry.register()
class aether(nn.Module):
    """AetherNet wrapper for the neosr framework.

    This wrapper allows instantiating different variants of AetherNet (small, medium, large)
    and passes configuration options from the YAML file directly to the core architecture.

    YAML configuration example:
    network_g:
      type: aether
      variant: medium   # 'small', 'medium', or 'large'
      scale: 4          # The super-resolution scale
      # Optional parameters that will be passed to the core model:
      norm_type: 'layernorm'
      res_scale: 0.9
      use_channel_attn: true
    """
    def __init__(self, variant: str, scale: int, **kwargs):
        """
        Args:
            variant (str): The AetherNet variant to use ('small', 'medium', 'large').
            scale (int): The upsampling scale.
            **kwargs: Additional arguments passed to the AetherNet constructor,
                      such as 'norm_type', 'res_scale', etc.
        """
        super().__init__()

        # Log the received parameters for debugging
        print(f"Initializing AetherNet variant: {variant} with scale: {scale}")
        print(f"Additional kwargs: {kwargs}")

        # Choose the correct factory function based on the 'variant' string
        if variant == 'small':
            model_factory = aether_small
        elif variant == 'medium':
            model_factory = aether_medium
        elif variant == 'large':
            model_factory = aether_large
        else:
            raise ValueError(f"Unknown AetherNet variant: {variant}. Must be 'small', 'medium', or 'large'.")

        # Instantiate the actual network from aether_core.py, passing all parameters
        self.net = model_factory(scale=scale, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A simple forward pass that calls the underlying network."""
        return self.net(x)