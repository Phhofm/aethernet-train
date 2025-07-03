# -*- coding: utf-8 -*-
# --- File Information ---
# Version: 1.0.0
# Author: Philip Hofmann
# License: MIT
# GitHub: https://github.com/phhofm/aethernet
# Description: Ultra-Fast Super-Resolution Network with QAT Support
# Features:
#   - Structural Reparameterization for Inference Efficiency
#   - Quantization-Aware Training (INT8, FP16)
#   - TensorRT/DML/ONNX Runtime Compatibility
#   - Version-Safe Deployment
#   - Multi-Backend Support (spandrel, chaiNNer, TRT, DML)

from copy import deepcopy
import json
import math
import time
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from typing import Tuple, List, Dict, Any, Optional, Union
import torch.ao.quantization as tq
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
import torch.onnx
from packaging import version  # For version parsing
import onnx

# Suppress quantization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")

# ------------------- Version Compatibility Setup ------------------- #
PT_VERSION = version.parse(torch.__version__)
MIN_PT_VERSION = version.parse("1.10.0")  # Minimum supported version
REC_PT_VERSION = version.parse("2.0.0")   # Recommended version for full features

if PT_VERSION < MIN_PT_VERSION:
    raise RuntimeError(f"PyTorch {MIN_PT_VERSION}+ required (detected {PT_VERSION})")

if PT_VERSION < REC_PT_VERSION:
    warnings.warn(
        f"PyTorch {REC_PT_VERSION}+ recommended for optimal performance and quantization features. "
        f"Detected version: {PT_VERSION}. Some features may be limited."
    )

# ------------------- Robust Version Parser ------------------- #
def parse_version(ver_str: str) -> List[int]:
    """Robustly parse version strings with non-standard suffixes
    
    Args:
        ver_str: Version string to parse
        
    Returns:
        List of 3 integers representing [major, minor, patch]
    """
    parts = []
    # Strip build metadata before splitting
    ver_str_clean = ver_str.split('+')[0]
    for part in ver_str_clean.split('.')[:3]:
        digits = ''.join(filter(str.isdigit, part))
        parts.append(int(digits) if digits else 0)
    # Pad to exactly 3 components
    return parts + [0] * (3 - len(parts))


# ------------------- Core Building Blocks ------------------- #

class DropPath(nn.Module):
    """
    Stochastic Depth implementation compatible with ONNX export.
    
    During training, randomly drops entire sample paths with given probability.
    During inference, acts as identity function.
    
    Args:
        drop_prob: Probability of dropping a path (0.0 = no drop)
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize to 0 or 1
        return x.div(keep_prob) * random_tensor


class ReparamLargeKernelConv(nn.Module):
    """
    Efficient large kernel convolution using structural reparameterization.
    
    Combines large and small kernel convolutions during training, fuses them
    into a single convolution for inference efficiency.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Main kernel size (must be odd)
        stride: Convolution stride
        groups: Number of groups
        small_kernel: Parallel small kernel size (must be odd)
        fused_init: Initialize in fused mode
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        if kernel_size % 2 == 0 or small_kernel % 2 == 0:
            raise ValueError("Kernel sizes must be odd for symmetrical padding")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        self.small_kernel = small_kernel
        self.fused = fused_init
        self.is_quantized = False

        if self.fused:
            # Directly initialize fused convolution
            self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                        padding=self.padding, groups=groups, bias=True)
        else:
            # Training path with separate convolutions
            self.lk_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                     self.padding, groups=groups, bias=False)
            self.sk_conv = nn.Conv2d(in_channels, out_channels, small_kernel, stride,
                                     small_kernel // 2, groups=groups, bias=False)
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.fused_conv(x)

        # Training forward pass combines both convolutions
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return (lk_out + self.lk_bias.view(1, -1, 1, 1) +
                sk_out + self.sk_bias.view(1, -1, 1, 1))

    def _fuse_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fused kernel and bias for deployment"""
        if self.fused:
            raise RuntimeError("Module is already fused")

        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = F.pad(self.sk_conv.weight, [pad] * 4)
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        return fused_kernel, fused_bias

    def fuse(self):
        """Fuse branches into single convolution for inference"""
        if self.fused:
            return

        fused_kernel, fused_bias = self._fuse_kernel()
        
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                    self.stride, padding=self.padding, groups=self.groups, bias=True)
                                    
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias

        # Preserve quantization config
        if self.is_quantized and hasattr(self.lk_conv, 'qconfig'):
            self.fused_conv.qconfig = self.lk_conv.qconfig
            
        # Cleanup training-specific parameters
        del self.lk_conv, self.sk_conv, self.lk_bias, self.sk_bias
        self.fused = True

class GatedConvFFN(nn.Module):
    """
    Gated Feed-Forward Network for enhanced feature transformation.
    
    Uses SiLU activation with temperature scaling and quantization support.
    
    Args:
        in_channels: Input channels
        mlp_ratio: Hidden dimension multiplier
        drop: Dropout probability
    """
    def __init__(self, in_channels: int, mlp_ratio: float = 1.5, drop: float = 0.):
        super().__init__()
        hidden_channels = int(in_channels * mlp_ratio)

        self.conv_gate = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv_main = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop2 = nn.Dropout(drop)
        self.temperature = nn.Parameter(torch.ones(1))
        self.is_quantized = False
        self.quant_mul = torch.nn.quantized.FloatFunctional()

        # Quantization stubs for SiLU activation
        self.act_dequant = tq.DeQuantStub()
        self.act_quant = tq.QuantStub()
        
        # Quantization stubs for temperature scaling
        self.temp_dequant = tq.DeQuantStub()
        self.temp_quant = tq.QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_unscaled = self.conv_gate(x)
        
        # Temperature scaling in float domain for quantization compatibility
        if self.is_quantized:
            gate_float = self.temp_dequant(gate_unscaled)
            gate_scaled = gate_float * self.temperature
            gate = self.temp_quant(gate_scaled)
        else:
            gate = gate_unscaled * self.temperature
        
        main = self.conv_main(x)
        
        # SiLU activation in float domain
        gate = self.act_dequant(gate)
        activated = self.act(gate)
        activated = self.act_quant(activated)
        
        # Handle mixed precision scenarios
        if x.dtype == torch.float16:
             x = activated.float() * main.float()
             x = x.half()
        elif self.is_quantized:
            x = self.quant_mul.mul(activated, main)
        else:
            x = activated * main

        x = self.drop1(x)
        x = self.conv_out(x)
        return x


class DynamicChannelScaling(nn.Module):
    """
    Efficient Channel Attention (Squeeze-and-Excitation) mechanism.
    
    Args:
        dim: Input dimension
        reduction: Channel reduction ratio
    """
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x))
        if self.is_quantized:
            return self.quant_mul.mul(x, scale)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Lightweight spatial attention module.
    
    Args:
        kernel_size: Convolution kernel size
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(concat))
        
        if self.is_quantized:
            return self.quant_mul.mul(x, attention_map)
        return x * attention_map


class DeploymentNorm(nn.Module):
    """
    Deployment-friendly normalization layer with fusion support.
    
    Maintains running statistics during training, converts to
    simple affine transform for inference.
    
    Args:
        channels: Number of channels
        eps: Numerical stability epsilon
    """
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
        self.fused = False
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return x * self.weight + self.bias

        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            with torch.no_grad():
                # Update running statistics with momentum
                self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

    def fuse(self):
        """Fuse normalization into affine transform for inference"""
        if self.fused:
            return
        
        # Compute fused scale and shift
        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        
        self.weight.data = scale
        self.bias.data = shift
        
        # Cleanup buffers
        del self.running_mean
        del self.running_var
        
        self.fused = True


class LayerNorm2d(nn.LayerNorm):
    """2D-adapted LayerNorm for 4D tensors."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class AetherBlock(nn.Module):
    """
    Core building block of AetherNet architecture.
    
    Combines large kernel convolution, gated FFN, and attention mechanisms
    with residual connection and quantization support.
    
    Args:
        dim: Feature dimension
        mlp_ratio: FFN expansion ratio
        drop: Dropout probability
        drop_path: Stochastic path probability
        lk_kernel: Large kernel size
        sk_kernel: Small kernel size
        fused_init: Initialize in fused mode
        quantize_residual: Quantize residual connection
        use_channel_attn: Enable channel attention
        use_spatial_attn: Enable spatial attention
        norm_layer: Normalization layer type
        res_scale: Residual scaling factor
    """
    def __init__(self, dim: int, mlp_ratio: float = 1.5, drop: float = 0.,
                 drop_path: float = 0., lk_kernel: int = 13, sk_kernel: int = 5,
                 fused_init: bool = False, quantize_residual: bool = True,
                 use_channel_attn: bool = True, use_spatial_attn: bool = False,
                 norm_layer: nn.Module = DeploymentNorm, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel,
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )
        self.norm = norm_layer(dim)
        self.ffn = GatedConvFFN(in_channels=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.channel_attn = DynamicChannelScaling(dim) if use_channel_attn else nn.Identity()
        self.spatial_attn = SpatialAttention() if use_spatial_attn else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Quantization utilities
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.quant_mul_scalar = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False
        
        # Quantization stubs for normalization
        self.norm_dequant = tq.DeQuantStub()
        self.norm_quant = tq.QuantStub()
        
        # Quantization stubs for residual scaling
        self.res_dequant = tq.DeQuantStub()
        self.res_quant = tq.QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)

        # Normalization in float domain
        x = self.norm_dequant(x)
        x = self.norm(x)
        x = self.norm_quant(x)
        
        x = self.ffn(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        
        residual_unscaled = self.drop_path(x)
        
        # Residual scaling in float domain
        if self.is_quantized:
            res_float = self.res_dequant(residual_unscaled)
            res_scaled = res_float * self.res_scale
            residual = self.res_quant(res_scaled)
        else:
            residual = residual_unscaled * self.res_scale

        # Add residual connection
        if self.is_quantized:
            return self.quant_add.add(shortcut, residual)
        else:
            return shortcut + residual


class QuantFusion(nn.Module):
    """
    Multi-scale feature fusion with quantization support.
    
    Aligns and concatenates features from multiple scales, applies
    convolution fusion with error compensation.
    
    Args:
        in_channels: Total input channels
        out_channels: Output channels
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fusion_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.error_comp = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if not features:
            raise ValueError("QuantFusion requires at least one feature")

        target_size = features[0].shape[-2:]
        aligned_features = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                aligned = F.interpolate(
                    feat, size=target_size, mode='nearest'
                )
                aligned_features.append(aligned)
            else:
                aligned_features.append(feat)

        x = torch.cat(aligned_features, dim=1)
        fused = self.fusion_conv(x)
        
        # Skip error compensation in quantized mode
        if self.is_quantized:
            return fused
        
        return fused + self.error_comp


class AdaptiveUpsample(nn.Module):
    """
    Resolution-aware upsampling module supporting powers of 2 and scale 3.
    
    Uses PixelShuffle for efficient upsampling with dynamic channel allocation.
    
    Args:
        scale: Upscaling factor
        in_channels: Input channels
    """
    def __init__(self, scale: int, in_channels: int):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.blocks = nn.ModuleList()
        # Ensure out_channels is multiple of 4 for PixelShuffle
        self.out_channels = max(32, (in_channels // max(1, scale // 2)) & -2)

        # Power of 2 scaling
        if (scale & (scale - 1)) == 0 and scale != 1:
            num_ups = int(math.log2(scale))
            current_channels = in_channels
            for i in range(num_ups):
                next_channels = self.out_channels if (i == num_ups - 1) else current_channels // 2
                self.blocks.append(nn.Conv2d(current_channels, 4 * next_channels, 3, 1, 1))
                self.blocks.append(nn.PixelShuffle(2))
                current_channels = next_channels
        # Scale 3
        elif scale == 3:
            self.blocks.append(nn.Conv2d(in_channels, 9 * self.out_channels, 3, 1, 1))
            self.blocks.append(nn.PixelShuffle(3))
        # Scale 1 (no upsampling)
        elif scale == 1:
            self.blocks.append(nn.Conv2d(in_channels, self.out_channels, 3, 1, 1))
        else:
            raise ValueError(f"Unsupported scale: {scale}. Only 1, 3 and powers of 2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

# ------------------- Main AetherNet Architecture ------------------- #

class AetherNet(nn.Module):
    """
    Production-Ready Super-Resolution Network with QAT support.
    
    Args:
        in_chans: Input channels (default: 3 for RGB)
        embed_dim: Base channel dimension
        depths: Number of blocks in each stage
        mlp_ratio: FFN expansion ratio
        drop: Dropout probability
        drop_path_rate: Stochastic path probability
        lk_kernel: Large kernel size
        sk_kernel: Small kernel size
        scale: Super-resolution scale factor
        img_range: Input image range (default: 1.0 for [0,1])
        fused_init: Initialize in fused mode
        quantize_residual: Quantize residual connections
        use_channel_attn: Enable channel attention
        use_spatial_attn: Enable spatial attention
        norm_type: Normalization type ('deployment' or 'layernorm')
        res_scale: Residual scaling factor
    """
    # Model version for compatibility tracking
    MODEL_VERSION = "3.1.1"
    
    def _init_weights(self, m: nn.Module):
        """Initialize weights using truncated normal distribution"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (DeploymentNorm, nn.LayerNorm, nn.GroupNorm)):
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
            if m.weight is not None: 
                nn.init.constant_(m.weight, 1.0)

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (4, 4, 4, 4),
        mlp_ratio: float = 1.5,
        drop: float = 0.0,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 13,
        sk_kernel: int = 5,
        scale: int = 4,
        img_range: float = 1.0,
        fused_init: bool = False,
        quantize_residual: bool = True,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = False,
        norm_type: str = 'deployment',
        res_scale: float = 0.1,
        **kwargs
    ):
        super().__init__()
        # Capture ALL constructor parameters
        self.arch_config = {
            'in_chans': in_chans, 'embed_dim': embed_dim, 'depths': depths,
            'mlp_ratio': mlp_ratio, 'drop': drop, 'drop_path_rate': drop_path_rate,
            'lk_kernel': lk_kernel, 'sk_kernel': sk_kernel, 'scale': scale,
            'img_range': img_range, 'fused_init': fused_init,
            'quantize_residual': quantize_residual, 'use_channel_attn': use_channel_attn,
            'use_spatial_attn': use_spatial_attn, 'norm_type': norm_type,
            'res_scale': res_scale, **kwargs
        }
        
        # Convert tuple parameters to lists for JSON serialization
        if isinstance(self.arch_config['depths'], tuple):
            self.arch_config['depths'] = list(self.arch_config['depths'])
        
        self.img_range = img_range
        self.register_buffer('scale_tensor', torch.tensor(scale, dtype=torch.int64))
        self.fused_init = fused_init
        self.embed_dim = embed_dim
        self.quantize_residual = quantize_residual
        self.num_stages = len(depths)
        self.is_quantized = False
        
        # Version tracking
        self.register_buffer('pt_version', torch.tensor(parse_version(torch.__version__)))
        self.register_buffer('model_version', torch.tensor(parse_version(self.MODEL_VERSION)))
        self.register_buffer('mean', torch.full((1, in_chans, 1, 1), 0.5))

        # Initial convolution
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Normalization layer selection
        if norm_type.lower() == 'deployment':
            norm_layer = DeploymentNorm
        elif norm_type.lower() == 'layernorm':
            norm_layer = LayerNorm2d
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        # Stage construction
        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        self.fusion_convs = nn.ModuleList()

        # Channel distribution for multi-scale fusion
        base_ch = embed_dim // self.num_stages
        remainder = embed_dim % self.num_stages
        fusion_out_channels = [base_ch + 1 if i < remainder else base_ch for i in range(self.num_stages)]
        
        assert sum(fusion_out_channels) == embed_dim, "Channel distribution error"

        # Build stages and fusion convolutions
        block_idx = 0
        for i, depth in enumerate(depths):
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(AetherBlock(
                    dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop,
                    drop_path=dpr[block_idx + j], lk_kernel=lk_kernel, sk_kernel=sk_kernel,
                    fused_init=fused_init, quantize_residual=quantize_residual,
                    use_channel_attn=use_channel_attn, use_spatial_attn=use_spatial_attn,
                    norm_layer=norm_layer, res_scale=res_scale))
            self.stages.append(nn.Sequential(*stage_blocks))
            block_idx += depth
            self.fusion_convs.append(nn.Conv2d(embed_dim, fusion_out_channels[i], 1))

        # Feature fusion and reconstruction
        self.quant_fusion_layer = QuantFusion(embed_dim, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Upsampling path
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), 
            nn.LeakyReLU(inplace=True))
        self.upsample = AdaptiveUpsample(scale, embed_dim)
        self.conv_last = nn.Conv2d(self.upsample.out_channels, in_chans, 3, 1, 1)

        # Initialize weights if not fused
        if not self.fused_init:
            self.apply(self._init_weights)

        # Quantization utilities
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.body_norm_dequant = tq.DeQuantStub()
        self.body_norm_quant = tq.QuantStub()
        self.quant_add = torch.nn.quantized.FloatFunctional()
        
        # Upsampling float island stubs
        self.upsample_dequant = tq.DeQuantStub()
        self.upsample_quant = tq.QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalization
        x_in = x / self.img_range if self.img_range != 1.0 else x
        x_in = x_in - self.mean
        
        # Quantization entry point
        x = self.quant(x_in)
        x_first = self.conv_first(x)

        # Feature extraction stages
        features = []
        out = x_first
        for stage in self.stages:
            out = stage(out)
            features.append(self.fusion_convs[len(features)](out))

        # Multi-scale feature fusion
        fused_features = self.quant_fusion_layer(features)
        
        # Normalization in float domain
        body_out = self.body_norm_dequant(fused_features)
        body_out = self.norm(body_out)
        body_out = self.body_norm_quant(body_out)
        
        # Residual connections
        if self.is_quantized:
            body_out = self.quant_add.add(body_out, x_first)
            body_out = self.quant_add.add(self.conv_after_body(body_out), body_out)
        else:
            body_out = body_out + x_first
            body_out = self.conv_after_body(body_out) + body_out
        
        # Reconstruction
        recon = self.conv_before_upsample(body_out)
        
        # Upsampling in float domain for quantization compatibility
        if self.is_quantized:
            recon = self.upsample_dequant(recon)
            recon = self.upsample(recon)
            recon = self.upsample_quant(recon)
        else:
            recon = self.upsample(recon)
            
        recon = self.conv_last(recon)
        output = self.dequant(recon)

        # Output denormalization
        output = output + self.mean
        output = output * self.img_range if self.img_range != 1.0 else output
        return output

    def fuse_model(self):
        """Fuse reparameterizable components for inference"""
        if self.fused_init:
            return
        for module in self.modules():
            if hasattr(module, 'fuse') and callable(module.fuse):
                module.fuse()
        self.fused_init = True

    def prepare_qat(self, per_channel: bool = False):
        """
        Prepare model for Quantization-Aware Training.
        
        Configures quantization observers and fuses model components.
        Explicitly excludes upsampling module from quantization.
        
        Args:
            per_channel: Use per-channel quantization for weights
        """
        if per_channel and PT_VERSION < version.parse("1.10.0"):
            warnings.warn("Per-channel quantization requires PyTorch 1.10+. Disabling.")
            per_channel = False

        # Configure quantization observers
        activation_observer = MovingAverageMinMaxObserver.with_args(
            qscheme=torch.per_tensor_affine,
            dtype=torch.quint8,
            reduce_range=False
        )
        
        weight_observer = (
            MovingAveragePerChannelMinMaxObserver.with_args(
                qscheme=torch.per_channel_symmetric, dtype=torch.qint8
            ) if per_channel else
            MovingAverageMinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
            )
        )
        
        qconfig = tq.QConfig(activation=activation_observer, weight=weight_observer)
        self.qconfig = qconfig
        
        self.fuse_model()
        
        # Exclude upsampling module from quantization
        self.upsample.qconfig = None

        tq.prepare_qat(self, inplace=True)
        self._set_quantization_flags(True)
        

    def _set_quantization_flags(self, status: bool):
        """Set quantization status flag on all relevant modules"""
        for module in self.modules():
            if hasattr(module, 'is_quantized'):
                module.is_quantized = status

    def convert_to_quantized(self) -> nn.Module:
        """Convert QAT model to fully quantized INT8 model"""
        if not self.is_quantized:
            raise RuntimeError("Model must be prepared with prepare_qat() first")
        
        self.eval()
        quantized_model = tq.convert(self, inplace=False)
        quantized_model._set_quantization_flags(True)
        return quantized_model

    def verify_quantization(self):
        """Check quantization status of all layers"""
        non_quantized = []
        for name, module in self.named_modules():
            # Check for standard layers that should be quantized
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                 # In a converted model, it should be a child of QuantizedModule
                if 'quantized' not in type(module).__name__.lower():
                    # Check if it has a fake_quant attribute (if it's a QAT model)
                    if not hasattr(module, 'weight_fake_quant'):
                        non_quantized.append(name)
        
        if non_quantized:
            print("Warning: Non-quantized layers detected:")
            for name in non_quantized:
                print(f"  - {name}")
        else:
            print("All convertible layers appear to be quantized.")
        
        return len(non_quantized) == 0
    
    def get_config(self) -> Dict[str, Any]:
        """Return the complete architecture configuration"""
        return deepcopy(self.arch_config)

    def save_optimized(self, filename: str, precision: str = 'fp32'):
        """
        Save optimized model with comprehensive metadata.
            
        Args:
            filename: Output filename
            precision: Model precision (fp32, fp16, int8)
        """
        self.eval()
        self.fuse_model()
            
        # Convert to requested precision
        if precision == 'fp16':
            model = self.half()
        elif precision == 'int8':
            if not self.is_quantized:
                raise ValueError("Model must be quantized for INT8 export")
            model = self
        else:  # fp32
            model = self
                
        # Create comprehensive metadata
        metadata = {
            'model_version': self.MODEL_VERSION,
            'pt_version': torch.__version__,
            'scale': self.scale_tensor.item(),
            'in_chans': self.mean.shape[1],
            'img_range': self.img_range,
            'precision': precision,
            'architecture': self._get_architecture_name(),
            'quantized': self.is_quantized,
            'arch_config': self.arch_config,
            'creation_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
            
        torch.save({
            'state_dict': model.state_dict(),
            'metadata': metadata
        }, filename)

    @classmethod
    def load_optimized(cls, filename: str, device='cuda'):
        """Load optimized model with architecture reconstruction"""
        checkpoint = torch.load(filename, map_location='cpu')
        metadata = checkpoint.get('metadata', {})
        arch_config = metadata.get('arch_config', None)
        
        if not arch_config:
            raise ValueError("Cannot load model: arch_config metadata not found.")
        
        # Ensure depths is a tuple for model constructor
        if 'depths' in arch_config and isinstance(arch_config['depths'], list):
            arch_config['depths'] = tuple(arch_config['depths'])
            
        model = cls(**arch_config)
        
        is_qat_model = metadata.get('quantized', False)
        if is_qat_model:
            model.prepare_qat() # Prepare model structure for QAT state_dict
            model._set_quantization_flags(True)

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model.to(device)
        
    def _get_architecture_name(self) -> str:
        """Get human-readable architecture name based on embed_dim"""
        if self.embed_dim <= 64: return "aether_tiny"
        if self.embed_dim <= 96: return "aether_small"
        if self.embed_dim <= 128: return "aether_medium"
        if self.embed_dim <= 180: return "aether_large"
        return "custom"
        
    def stabilize_for_fp16(self):
        """
        Apply comprehensive FP16 stabilization techniques.
        
        Clamps parameters to prevent overflow in FP16 precision.
        Should be called on FP32 model before converting to .half().
        """
        print("Applying FP16 stabilization...")
        for module in self.modules():
            # Stabilize fused normalization layers
            if isinstance(module, DeploymentNorm) and module.fused:
                module.weight.data.clamp_(min=-100.0, max=100.0)
                module.bias.data.clamp_(min=-100.0, max=100.0)
                
            # Stabilize FFN temperature parameters
            if isinstance(module, GatedConvFFN):
                module.temperature.data.clamp_(min=0.01, max=10.0)
                
        # General parameter stabilization
        for param in self.parameters():
            param.data.clamp_(min=-1000.0, max=1000.0)
        
        print("FP16 stabilization complete.")

# ------------------- PyTorch 2.0+ Optimizations ------------------- 
if PT_VERSION >= version.parse("2.0.0"):
    def compile_model(model: AetherNet, mode: str = "max-autotune"):
        """
        Optimize model with torch.compile (PyTorch 2.0+ only)
        
        Args:
            model: AetherNet model instance
            mode: Compilation mode (default: max-autotune)
            
        Returns:
            Compiled model for faster execution
        """
        return torch.compile(model, mode=mode, fullgraph=True)
else:
    def compile_model(model: AetherNet, mode: str = None):
        """
        Dummy function for older PyTorch versions
        
        Returns:
            Original model unchanged
        """
        warnings.warn("torch.compile requires PyTorch 2.0+. Returning original model.")
        return model

# ------------------- Network Presets ------------------- 

def aether_tiny(scale: int, **kwargs) -> AetherNet:
    """Minimal version for real-time use (64 channels)"""
    return AetherNet(embed_dim=64, depths=(3, 3, 3), scale=scale,
                  use_spatial_attn=False, res_scale=0.2, **kwargs)

def aether_small(scale: int, **kwargs) -> AetherNet:
    """Small version (96 channels)"""
    return AetherNet(embed_dim=96, depths=(4, 4, 4, 4), scale=scale,
                  use_spatial_attn=False, res_scale=0.1, **kwargs)

def aether_medium(scale: int, **kwargs) -> AetherNet:
    """Balanced version (128 channels)"""
    return AetherNet(embed_dim=128, depths=(6, 6, 6, 6), scale=scale,
                  use_channel_attn=True, res_scale=0.1, **kwargs)

def aether_large(scale: int, **kwargs) -> AetherNet:
    """High-quality version (180 channels)"""
    return AetherNet(embed_dim=180, depths=(8, 8, 8, 8, 8), scale=scale,
                  use_channel_attn=True, use_spatial_attn=True, res_scale=0.1, **kwargs)

# ------------------- ONNX Export ------------------- 

def export_onnx(
    model: nn.Module, 
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
    """
    model.eval()
    model.fuse_model()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    device = next(model.parameters(), None)
    if device is None:
        device = 'cpu'
    else:
        device = device.device
    dummy_input = dummy_input.to(device)
    
    # Handle precision conversion
    if precision == 'fp16':
        model.half()
        dummy_input = dummy_input.half()
    
    if precision == 'int8' and not model.is_quantized:
        raise ValueError("INT8 export requires a quantized model. Ensure it has been converted.")

    # Determine output filename
    model_name = model._get_architecture_name() if hasattr(model, '_get_architecture_name') else "aether"
    onnx_filename = output_path or f"{model_name}_x{scale}_{precision}.onnx"
    
    # Configure dynamic axes for variable input sizes
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}
    }

    # Export to ONNX
    torch.onnx.export(
        model, dummy_input, onnx_filename, opset_version=18,
        do_constant_folding=True, input_names=['input'], output_names=['output'],
        dynamic_axes=dynamic_axes,
    )
    
    # Add metadata for Spandrel/ChaiNNer compatibility
    try:
        model_onnx = onnx.load(onnx_filename)
        
        meta = model_onnx.metadata_props
        # Use add() to create new entries
        new_meta = meta.add()
        new_meta.key = "model_name"
        new_meta.value = "AetherNet"
        
        new_meta = meta.add()
        new_meta.key = "scale"
        new_meta.value = str(scale)
        
        new_meta = meta.add()
        new_meta.key = "img_range"
        new_meta.value = str(model.img_range)
        
        if hasattr(model, 'arch_config'):
            arch_str = json.dumps(model.get_config())
            new_meta = meta.add()
            new_meta.key = "spandrel_config"
            new_meta.value = arch_str
            
        onnx.save(model_onnx, onnx_filename)
        print(f"Successfully added Spandrel/ChaiNNer metadata to {onnx_filename}")
    except Exception as e:
        print(f"Warning: Could not add ONNX metadata. Error: {e}")

    print(f"Exported {precision.upper()} model to {onnx_filename}")
    return onnx_filename