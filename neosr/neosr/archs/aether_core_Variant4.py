# -*- coding: utf-8 -*-
# --- File Information ---
# Version: 3.1.0 (Production-Ready)
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
    for part in ver_str.split('.')[:3]:  # Only consider up to 3 components
        # Strip non-digit characters and trailing metadata
        clean_part = part.split('+')[0].split('-')[0]
        # Extract numeric components
        digits = ''.join(filter(str.isdigit, clean_part))
        # Convert to integer if valid, otherwise 0
        parts.append(int(digits) if digits else 0)
    # Pad to exactly 3 components if needed
    return parts + [0] * (3 - len(parts))


# ------------------- Core Building Blocks ------------------- #

class DropPath(nn.Module):
    """
    Stochastic Depth with ONNX-compatible implementation.
    
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
    Efficient large kernel convolution with structural reparameterization.
    
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
            self.explicit_pad = nn.ZeroPad2d(self.padding)
            self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                        padding=0, groups=groups, bias=True)
        else:
            # Training path
            self.lk_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                     self.padding, groups=groups, bias=False)
            self.sk_conv = nn.Conv2d(in_channels, out_channels, small_kernel, stride,
                                     small_kernel // 2, groups=groups, bias=False)
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.fused_conv(self.explicit_pad(x))

        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return (lk_out + self.lk_bias.view(1, -1, 1, 1) +
                sk_out + self.sk_bias.view(1, -1, 1, 1))

    def _fuse_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fused kernel and bias for deployment."""
        if self.fused:
            raise RuntimeError("Module is already fused")

        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = F.pad(self.sk_conv.weight, [pad] * 4)
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        return fused_kernel, fused_bias

    def fuse(self):
        """Fuse branches into single convolution."""
        if self.fused:
            return

        fused_kernel, fused_bias = self._fuse_kernel()
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                    self.stride, padding=0, groups=self.groups, bias=True)
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias
        self.explicit_pad = nn.ZeroPad2d(self.padding)

        # Preserve quantization config
        if self.is_quantized and hasattr(self.lk_conv, 'qconfig'):
            self.fused_conv.qconfig = self.lk_conv.qconfig
            
        # Cleanup
        del self.lk_conv, self.sk_conv, self.lk_bias, self.sk_bias
        self.fused = True


class GatedConvFFN(nn.Module):
    """
    Gated Feed-Forward Network for enhanced feature transformation.
    
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
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.conv_gate(x) * self.temperature
        main = self.conv_main(x)
        activated = self.act(gate)

        if self.is_quantized:
            x = self.quant_mul.mul(activated, main)
        else:
            x = activated * main

        x = self.drop1(x)
        x = self.conv_out(x)
        return self.drop2(x)


class DynamicChannelScaling(nn.Module):
    """
    Efficient Channel Attention (Squeeze-and-Excitation).
    
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
    Deployment-friendly normalization layer.
    
    Args:
        channels: Number of channels
        eps: Numerical stability epsilon
    """
    def __init__(self, channels: int, eps: float = 1e-5):
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
                self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

    def fuse(self):
        """Fuse normalization into affine transform."""
        if self.fused:
            return

        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        self.weight.data = scale
        self.bias.data = shift
        self.fused = True


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for 4D tensors."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class AetherBlock(nn.Module):
    """
    Core building block of AetherNet.
    
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

        # Quantization-aware residual
        self.quantize_residual_flag = quantize_residual
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False
        self.residual_quant = tq.QuantStub() if quantize_residual else nn.Identity()
        self.residual_dequant = tq.DeQuantStub() if quantize_residual else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.ffn(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        residual = self.drop_path(x) * self.res_scale

        if self.is_quantized:
            if self.quantize_residual_flag:
                shortcut_q = self.residual_quant(shortcut)
                output = self.quant_add.add(shortcut_q, residual)
                return self.residual_dequant(output)
            return self.quant_add.add(shortcut, residual)

        return shortcut + residual


class QuantFusion(nn.Module):
    """
    Multi-scale feature fusion with quantization support.
    
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
                    feat, size=target_size, mode='bilinear', 
                    align_corners=False
                )
                aligned_features.append(aligned)
            else:
                aligned_features.append(feat)

        x = torch.cat(aligned_features, dim=1)
        fused = self.fusion_conv(x)
        
        if self.is_quantized:
            return self.quant_add.add(fused, self.error_comp)
        return fused + self.error_comp


class AdaptiveUpsample(nn.Module):
    """
    Resolution-aware upsampling module.
    
    Args:
        scale: Upscaling factor
        in_channels: Input channels
    """
    def __init__(self, scale: int, in_channels: int):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.blocks = nn.ModuleList()
        self.out_channels = max(32, in_channels // max(1, scale // 2))

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
    Production-Ready Super-Resolution Network.
    
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
    MODEL_VERSION = "3.1.0"
    
    def _init_weights(self, m: nn.Module):
        """Initialize weights for various layer types."""
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
            'in_chans': in_chans,
            'embed_dim': embed_dim,
            'depths': depths,
            'mlp_ratio': mlp_ratio,
            'drop': drop,
            'drop_path_rate': drop_path_rate,
            'lk_kernel': lk_kernel,
            'sk_kernel': sk_kernel,
            'scale': scale,
            'img_range': img_range,
            'fused_init': fused_init,
            'quantize_residual': quantize_residual,
            'use_channel_attn': use_channel_attn,
            'use_spatial_attn': use_spatial_attn,
            'norm_type': norm_type,
            'res_scale': res_scale,
            **kwargs
        }
        
        # Convert tuple parameters to lists for JSON serialization
        if isinstance(self.arch_config['depths'], tuple):
            self.arch_config['depths'] = list(self.arch_config['depths'])
        
        self.img_range = img_range
        self.register_buffer('scale_tensor', torch.tensor(scale))
        self.fused_init = fused_init
        self.embed_dim = embed_dim
        self.quantize_residual = quantize_residual
        self.num_stages = len(depths)
        self.is_quantized = False
        
        # Store creation environment
        self.register_buffer('pt_version', 
                            torch.tensor(parse_version(torch.__version__)))
        self.register_buffer('model_version', 
                            torch.tensor(parse_version(self.MODEL_VERSION)))
        
        # Input normalization
        self.register_buffer('mean', torch.full((1, in_chans, 1, 1), 0.5))

        # Input normalization
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Normalization layer selection
        if norm_type.lower() == 'deployment':
            norm_layer = DeploymentNorm
        elif norm_type.lower() == 'layernorm':
            norm_layer = LayerNorm2d
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        # Build stages
        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        self.fusion_convs = nn.ModuleList()

        # Channel distribution for fusion
        base_ch = embed_dim // self.num_stages
        remainder = embed_dim % self.num_stages
        fusion_out_channels = []
        for i in range(self.num_stages):
            fusion_out_channels.append(base_ch + 1 if i < remainder else base_ch)
        
        assert sum(fusion_out_channels) == embed_dim, "Channel distribution error"

        # Build stages
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

        # Feature fusion and processing
        self.quant_fusion_layer = QuantFusion(embed_dim, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Reconstruction
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), 
            nn.LeakyReLU(inplace=True))
        self.upsample = AdaptiveUpsample(scale, embed_dim)
        self.conv_last = nn.Conv2d(self.upsample.out_channels, in_chans, 3, 1, 1)

        # Initialize weights
        if not self.fused_init:
            self.apply(self._init_weights)

        # Quantization stubs
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalization
        x = (x - self.mean) * self.img_range
        x = self.quant(x) if self.is_quantized else x

        # Initial feature extraction
        x_first = self.conv_first(x)

        # Process through stages
        features = []
        out = x_first
        for stage, fusion_conv in zip(self.stages, self.fusion_convs):
            out = stage(out)
            features.append(fusion_conv(out))

        # Fuse and process features
        fused_features = self.quant_fusion_layer(features)
        body_out = fused_features + x_first
        body_out = self.conv_after_body(self.norm(body_out)) + body_out

        # Reconstruction
        recon = self.conv_before_upsample(body_out)
        recon = self.upsample(recon)
        recon = self.conv_last(recon)

        # Output processing
        output = self.dequant(recon) if self.is_quantized else recon
        return output / self.img_range + self.mean

    def fuse_model(self):
        """Fuse reparameterizable layers for inference."""
        if self.fused_init:
            return

        for module in self.modules():
            if hasattr(module, 'fuse') and callable(module.fuse):
                if not isinstance(module, LayerNorm2d):
                    module.fuse()
        self.fused_init = True

    def prepare_qat(self, per_channel: bool = False):
        """
        Prepare model for Quantization-Aware Training.
        
        Args:
            per_channel: Use per-channel quantization for weights
        """
        # Version guard for per-channel quantization
        if per_channel and PT_VERSION < version.parse("1.10.0"):
            warnings.warn("Per-channel quantization requires PyTorch 1.10+. Disabling.")
            per_channel = False

        # Configure quantization observers
        activation_observer = MovingAverageMinMaxObserver.with_args(
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            reduce_range=False
        )
        
        if per_channel:
            try:
                weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(
                    qscheme=torch.per_channel_symmetric,
                    dtype=torch.qint8,
                    reduce_range=False
                )
            except AttributeError:
                print("Per-channel not available, falling back to per-tensor")
                weight_observer = MovingAverageMinMaxObserver.with_args(
                    qscheme=torch.per_tensor_symmetric,
                    dtype=torch.qint8,
                    reduce_range=False
                )
        else:
            weight_observer = MovingAverageMinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric,
                dtype=torch.qint8,
                reduce_range=False
            )
        
        qconfig = tq.QConfig(
            activation=activation_observer,
            weight=weight_observer
        )
        
        # Apply configuration
        self.qconfig = qconfig
        self.apply(lambda m: setattr(m, 'qconfig', qconfig) if hasattr(m, 'qconfig') else None)
        
        # Exclude sensitive layers from quantization
        layers_to_float = ['conv_first', 'conv_last', 'conv_before_upsample.0']
        for name, module in self.named_modules():
            if name in layers_to_float:
                module.qconfig = None
        
        # Prepare model
        self.train()
        self.fuse_model()
        tq.prepare_qat(self, inplace=True)
        
        # Set quantization flags
        self._set_quantization_flags(True)

    def _set_quantization_flags(self, status: bool):
        """Recursively set quantization flags for all modules."""
        for module in self.modules():
            if hasattr(module, 'is_quantized'):
                module.is_quantized = status

    def convert_to_quantized(self) -> nn.Module:
        """Convert QAT model to fully quantized INT8 model."""
        if not self.is_quantized:
            raise RuntimeError("Model must be prepared with prepare_qat() first")

        quantized_model = tq.convert(self, inplace=False)
        quantized_model.is_quantized = True
        return quantized_model

    def verify_quantization(self):
        """Check quantization status of all layers."""
        non_quantized = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and not hasattr(module, 'weight_fake_quant'):
                non_quantized.append(name)
        
        if non_quantized:
            print("Non-quantized layers detected:")
            for name in non_quantized:
                print(f"  - {name}")
        else:
            print("All layers quantized successfully")
        
        return len(non_quantized) == 0
    
    def get_config(self) -> Dict[str, Any]:
        """Return the complete architecture configuration."""
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
        """
        Load optimized model with enhanced compatibility checks.
            
        Args:
            filename: Model filename
            device: Target device
                
        Returns:
            AetherNet: Loaded model instance
        """
        checkpoint = torch.load(filename, map_location='cpu')
        state_dict = checkpoint['state_dict']
        metadata = checkpoint.get('metadata', {})
            
        # Version compatibility checks
        current_pt = version.parse(torch.__version__)
        saved_pt = version.parse(metadata.get('pt_version', '1.0.0'))
            
        if current_pt.major != saved_pt.major:
            warnings.warn(f"PyTorch major version mismatch: "
                        f"Saved with {saved_pt}, running {current_pt}. "
                        "Quantization may be affected.")
            
        # Model initialization using stored architecture config
        if 'arch_config' in metadata:
            config = metadata['arch_config']
            # Convert depths back to tuple
            if 'depths' in config and isinstance(config['depths'], list):
                config['depths'] = tuple(config['depths'])
            model = cls(**config)
        else:
            # Fallback for older models without arch_config
            model = cls(
                scale=metadata['scale'],
                in_chans=metadata['in_chans'],
                img_range=metadata['img_range'],
                quantize_residual=metadata.get('quantized', False)
            )
            
        model.load_state_dict(state_dict)
            
        # Set quantization flags if needed
        if metadata.get('quantized', False):
            model._set_quantization_flags(True)
                
        return model.to(device)
        
    def _get_architecture_name(self) -> str:
        """Identify model preset programmatically."""
        if self.embed_dim == 64: return "aether_tiny"
        if self.embed_dim == 96: return "aether_small"
        if self.embed_dim == 128: return "aether_medium"
        if self.embed_dim == 180: return "aether_large"
        return "custom"

# ------------------- PyTorch 2.0+ Optimizations ------------------- 
if PT_VERSION >= version.parse("2.0.0"):
    def compile_model(model: AetherNet, mode: str = "max-autotune"):
        """
        Optimize model with torch.compile (PyTorch 2.0+ only)
        
        Args:
            model: AetherNet model instance
            mode: Compilation mode (default: max-autotune)
            
        Returns:
            Compiled model for faster training
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
    """Minimal version for real-time use (64 channels)."""
    return AetherNet(embed_dim=64, depths=(3, 3, 3), scale=scale,
                  use_spatial_attn=False, res_scale=0.2, **kwargs)

def aether_small(scale: int, **kwargs) -> AetherNet:
    """Small version (96 channels)."""
    return AetherNet(embed_dim=96, depths=(4, 4, 4, 4), scale=scale,
                  use_spatial_attn=False, res_scale=0.1, **kwargs)

def aether_medium(scale: int, **kwargs) -> AetherNet:
    """Balanced version (128 channels)."""
    return AetherNet(embed_dim=128, depths=(6, 6, 6, 6), scale=scale,
                  use_channel_attn=True, res_scale=0.1, **kwargs)

def aether_large(scale: int, **kwargs) -> AetherNet:
    """High-quality version (180 channels)."""
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
        model: Model to export
        scale: Super-resolution scale factor
        precision: Export precision (fp32, fp16, int8)
        output_path: Custom output path
        
    Returns:
        str: Path to exported ONNX file
    """
    model.eval()
    model.fuse_model()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    device = next(model.parameters()).device if next(model.parameters(), None) else 'cpu'
    dummy_input = dummy_input.to(device)
    
    # Handle precision
    if precision == 'fp16':
        model = model.half()
        dummy_input = dummy_input.half()
    
    # Validate INT8 model
    if precision == 'int8' and not model.is_quantized:
        raise ValueError("INT8 export requires a quantized model")

    # Create filename
    model_name = model._get_architecture_name() if hasattr(model, '_get_architecture_name') else "aether"
    onnx_filename = output_path or f"{model_name}_x{scale}_{precision}.onnx"
    
    # Dynamic axes configuration
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}
    }

    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )
    
    # Add metadata for spandrel/chaiNNer
    try:
        import onnx
        from onnx import helper
        
        model_onnx = onnx.load(onnx_filename)
        meta = [
            helper.make_tensor('scale', onnx.TensorProto.INT64, [1], [scale]),
            helper.make_tensor('img_range', onnx.TensorProto.FLOAT, [1], [model.img_range]),
            helper.make_tensor('mean', onnx.TensorProto.FLOAT, model.mean.shape, model.mean.cpu().numpy().flatten())
        ]
        
        # Add architecture metadata if available
        if hasattr(model, 'arch_config'):
            # Convert to string for ONNX metadata
            arch_str = str(model.arch_config)
            meta.append(
                helper.make_tensor(
                    'architecture', 
                    onnx.TensorProto.STRING, 
                    [1], 
                    [arch_str.encode('utf-8')]
                )
            )
            
        model_onnx.graph.initializer.extend(meta)
        onnx.save(model_onnx, onnx_filename)
    except ImportError:
        print("ONNX metadata not added - install onnx package to enable")

    print(f"Exported {precision.upper()} model to {onnx_filename}")
    return onnx_filename