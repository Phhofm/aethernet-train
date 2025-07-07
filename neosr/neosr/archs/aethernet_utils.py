"""
Utility functions for AetherNet model training, conversion, and deployment.

Includes:
- Version parsing
- Model presets
- Model saving/loading with metadata
- ONNX export
- FP16 stabilization
- Model compilation (for PyTorch 2.0+)
"""

import json
import math
import time
import warnings
import logging
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import torch
import numpy as np
import onnx
from packaging import version  # For version parsing
import torch.onnx
import torch.ao.quantization as tq

from aethernet_arch import AetherNet, DeploymentNorm, GatedConvFFN


def parse_version(ver_str: str) -> List[int]:
    """
    Robustly parse version strings with non-standard suffixes.
    
    Args:
        ver_str: Version string to parse
        
    Returns:
        List of 3 integers representing [major, minor, patch]
    """
    parts = []
    # Strip build metadata before splitting
    ver_str_clean = ver_str.split('+')[0]
    for part in ver_str_clean.split('.')[:3]:
        # Extract only numeric parts
        digits = ''.join(filter(str.isdigit, part))
        parts.append(int(digits) if digits else 0)
    # Pad to exactly 3 components
    return parts + [0] * (3 - len(parts))


def check_pytorch_version():
    """Verify PyTorch version compatibility and warn if outdated."""
    PT_VERSION = version.parse(torch.__version__)
    MIN_PT_VERSION = version.parse("1.10.0")
    REC_PT_VERSION = version.parse("2.0.0")
    
    if PT_VERSION < MIN_PT_VERSION:
        raise RuntimeError(f"PyTorch {MIN_PT_VERSION}+ required (detected {PT_VERSION})")
    if PT_VERSION < REC_PT_VERSION:
        warnings.warn(
            f"PyTorch {REC_PT_VERSION}+ recommended for optimal performance. "
            f"Detected version: {PT_VERSION}. Some features may be limited."
        )


def aether_tiny(scale: int, **kwargs) -> AetherNet:
    """
    Minimal version for real-time use (64 channels).
    
    Args:
        scale: Super-resolution scale factor
        **kwargs: Additional keyword arguments for AetherNet
    """
    return AetherNet(embed_dim=64, depths=(3, 3, 3), scale=scale,
                  use_spatial_attn=False, res_scale=0.2, **kwargs)


def aether_small(scale: int, **kwargs) -> AetherNet:
    """
    Small version (96 channels).
    
    Args:
        scale: Super-resolution scale factor
        **kwargs: Additional keyword arguments for AetherNet
    """
    return AetherNet(embed_dim=96, depths=(4, 4, 4, 4), scale=scale,
                  use_spatial_attn=False, res_scale=0.1, **kwargs)


def aether_medium(scale: int, **kwargs) -> AetherNet:
    """
    Balanced version (128 channels).
    
    Args:
        scale: Super-resolution scale factor
        **kwargs: Additional keyword arguments for AetherNet
    """
    return AetherNet(embed_dim=128, depths=(6, 6, 6, 6), scale=scale,
                  use_channel_attn=True, res_scale=0.1, **kwargs)


def aether_large(scale: int, **kwargs) -> AetherNet:
    """
    High-quality version (180 channels).
    
    Args:
        scale: Super-resolution scale factor
        **kwargs: Additional keyword arguments for AetherNet
    """
    return AetherNet(embed_dim=180, depths=(8, 8, 8, 8, 8), scale=scale,
                  use_channel_attn=True, use_spatial_attn=True, res_scale=0.1, **kwargs)


def save_optimized(model: AetherNet, filename: str, precision: str = 'fp32'):
    """
    Save optimized model with comprehensive metadata.
    
    Args:
        model: AetherNet model instance
        filename: Output filename
        precision: Model precision (fp32, fp16, int8)
        
    Raises:
        ValueError: If INT8 export is attempted on non-quantized model
    """
    if precision not in ('fp32', 'fp16', 'int8'):
        raise ValueError(f"Unsupported precision: {precision}. Choose from fp32, fp16, int8")
    
    model.eval()
    model.fuse_model()
    
    try:
        # Convert to requested precision
        if precision == 'fp16':
            model = model.half()
        elif precision == 'int8':
            if not model.is_quantized:
                raise ValueError("INT8 export requires a quantized model. Ensure it has been converted.")
        
        # Create comprehensive metadata
        metadata = {
            'model_version': model.MODEL_VERSION,
            'pt_version': torch.__version__,
            'scale': model.scale_tensor.item(),
            'in_chans': model.mean.shape[1],
            'img_range': model.img_range,
            'precision': precision,
            'architecture': model._get_architecture_name(),
            'quantized': model.is_quantized,
            'arch_config': model.get_config(),
            'creation_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save model state and metadata
        torch.save({
            'state_dict': model.state_dict(),
            'metadata': metadata
        }, filename)
    except Exception as e:
        raise RuntimeError(f"Failed to save optimized model: {str(e)}")


def load_optimized(filename: str, device='cuda') -> AetherNet:
    """
    Load optimized model with architecture reconstruction.
    
    Args:
        filename: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded AetherNet model
        
    Raises:
        ValueError: If arch_config is missing or invalid
    """
    try:
        checkpoint = torch.load(filename, map_location='cpu')
        metadata = checkpoint.get('metadata', {})
        arch_config = metadata.get('arch_config', None)
        
        if not arch_config:
            raise ValueError("Cannot load model: arch_config metadata not found.")
        
        # Ensure depths is a tuple for model constructor
        if 'depths' in arch_config and isinstance(arch_config['depths'], list):
            arch_config['depths'] = tuple(arch_config['depths'])
            
        model = AetherNet(**arch_config)
        
        is_qat_model = metadata.get('quantized', False)
        if is_qat_model:
            model.prepare_qat() # Prepare model structure for QAT state_dict
            model._set_quantization_flags(True)

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load optimized model: {str(e)}")


def stabilize_for_fp16(model: AetherNet):
    """
    Apply comprehensive FP16 stabilization techniques.
    
    Clamps parameters to prevent overflow in FP16 precision.
    Should be called on FP32 model before converting to .half().
    
    Args:
        model: AetherNet model instance
    """
    logger = logging.getLogger("AetherNet")
    logger.info("Applying FP16 stabilization...")
    for module in model.modules():
        # Stabilize fused normalization layers
        if isinstance(module, DeploymentNorm) and module.fused:
            module.weight.data.clamp_(min=-100.0, max=100.0)
            module.bias.data.clamp_(min=-100.0, max=100.0)
            
        # Stabilize FFN temperature parameters
        if isinstance(module, GatedConvFFN):
            module.temperature.data.clamp_(min=0.01, max=10.0)
            
    # General parameter stabilization
    for param in model.parameters():
        param.data.clamp_(min=-1000.0, max=1000.0)
    
    logger.info("FP16 stabilization complete.")


def export_onnx(
    model: AetherNet, 
    scale: int, 
    precision: str = 'fp32',
    output_path: str = None
) -> str:
    """
    Export model to ONNX format with optimizations and metadata.
    
    Args:
        model: AetherNet model instance
        scale: Super-resolution scale factor
        precision: Export precision (fp32, fp16, int8)
        output_path: Custom output path (optional)
        
    Returns:
        Path to exported ONNX file
        
    Raises:
        ValueError: If INT8 export is attempted on non-quantized model
        RuntimeError: If ONNX export fails
    """
    if precision not in ('fp32', 'fp16', 'int8'):
        raise ValueError(f"Unsupported precision: {precision}. Choose from fp32, fp16, int8")
    
    model.eval()
    model.fuse_model()

    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)
        
        # Handle precision conversion
        if precision == 'fp16':
            model.half()
            dummy_input = dummy_input.half()
        
        if precision == 'int8' and not model.is_quantized:
            raise ValueError("INT8 export requires a quantized model. Ensure it has been converted.")

        # Determine output filename
        model_name = model._get_architecture_name()
        onnx_filename = output_path or f"{model_name}_x{scale}_{precision}.onnx"
        
        # Configure dynamic axes for variable input sizes
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}
        }

        # Export to ONNX
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_filename, 
            opset_version=18,
            do_constant_folding=True, 
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes=dynamic_axes,
        )
        
        # Add metadata for Spandrel/ChaiNNer compatibility
        model_onnx = onnx.load(onnx_filename)
        meta = model_onnx.metadata_props
        # Add model name
        new_meta = onnx.StringStringEntryProto(key="model_name", value="AetherNet")
        meta.append(new_meta)
        # Add scale
        new_meta = onnx.StringStringEntryProto(key="scale", value=str(scale))
        meta.append(new_meta)
        # Add image range
        new_meta = onnx.StringStringEntryProto(key="img_range", value=str(model.img_range))
        meta.append(new_meta)
        # Add architecture configuration
        arch_str = json.dumps(model.get_config())
        new_meta = onnx.StringStringEntryProto(key="spandrel_config", value=arch_str)
        meta.append(new_meta)
        
        onnx.save(model_onnx, onnx_filename)
        logging.info(f"Added Spandrel/ChaiNNer metadata to {onnx_filename}")
        
        return onnx_filename
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {str(e)}")


def compile_model(model: AetherNet, mode: str = "max-autotune") -> torch.nn.Module:
    """
    Optimize model with torch.compile (PyTorch 2.0+ only).
    
    Args:
        model: AetherNet model instance
        mode: Compilation mode (default: "max-autotune")
        
    Returns:
        Compiled model for faster execution, or original model if not supported
    """
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        return torch.compile(model, mode=mode, fullgraph=True)
    warnings.warn("torch.compile requires PyTorch 2.0+. Returning original model.")
    return model