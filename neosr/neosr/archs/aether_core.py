# --- File Information ---
# Version: 1.2.0
# Author: Philip Hofmann
# License: MIT
# GitHub: https://github.com/phhofm/aethernet
# Description: Production-ready super-resolution network with quantization support

"""
AetherNet Core Architecture - Production-Ready Implementation

Overview:
    A high-performance super-resolution network designed for optimal balance of quality,
    speed, and deployment flexibility. Features quantization-aware training (QAT) support
    and ONNX export capabilities for production deployment.

Key Features:
    - Structural Reparameterization: Large kernel convolutions fused for inference efficiency
    - Gated Feed-Forward Networks: Enhanced feature transformation
    - Multi-Stage Feature Fusion: Combines shallow and deep features
    - Deployment-Optimized Normalization: ONNX/TensorRT friendly
    - Quantization Support: Full INT8 quantization workflow
    - Attention Mechanisms: Lightweight channel and spatial attention

Usage:
    1. Instantiate model: model = aether_medium(scale=4)
    2. Train: Standard PyTorch training loop
    3. Prepare for deployment: model.fuse_model()
    4. For quantization:
        - model.prepare_qat()  # Prepare for QAT
        - Continue training
        - quant_model = model.convert_to_quantized()  # Convert to INT8
    5. Export to ONNX: export_onnx(model, scale=4, precision='int8')

Release Notes:
    Version 1.2.0:
    - Enhanced documentation and code comments
    - Improved quantization safety checks
    - Added per-channel quantization verification
    - Better type hints and error handling
    - Optimized ONNX export compatibility
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from typing import Tuple, List, Dict, Any, Optional, Union
import torch.ao.quantization as tq
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
import warnings

# Suppress quantization warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")

# ------------------- Core Building Blocks ------------------- #

class DropPath(nn.Module):
    """
    Stochastic Depth with ONNX-compatible implementation.
    
    Randomly drops entire residual blocks during training for regularization.
    
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
    
    Uses parallel large and small kernels during training, fused to a single
    large kernel convolution for deployment.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Size of the main kernel (must be odd)
        stride: Convolution stride
        groups: Number of groups (depthwise when groups=in_channels)
        small_kernel: Size of the parallel small kernel (must be odd)
        fused_init: Initialize in fused mode (for deployment)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        # Validate kernel sizes
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
        self.is_quantized = False  # Tracks quantization state

        if self.fused:
            # Deployment path: single convolution with explicit padding
            self.explicit_pad = nn.ZeroPad2d(self.padding)
            self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                        padding=0, groups=groups, bias=True)
        else:
            # Training path: parallel convolutions
            self.lk_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                     self.padding, groups=groups, bias=False)
            self.sk_conv = nn.Conv2d(in_channels, out_channels, small_kernel, stride,
                                     small_kernel // 2, groups=groups, bias=False)
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.fused_conv(self.explicit_pad(x))

        # Training forward pass
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return (lk_out + self.lk_bias.view(1, -1, 1, 1) +
                sk_out + self.sk_bias.view(1, -1, 1, 1))

    def _fuse_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fused kernel and bias for deployment."""
        if self.fused:
            raise RuntimeError("Module is already fused")

        # Pad small kernel to match large kernel size
        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = F.pad(self.sk_conv.weight, [pad] * 4)

        # Combine kernels and biases
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        return fused_kernel, fused_bias

    def fuse(self):
        """Fuse branches into a single convolution for deployment."""
        if self.fused:
            return

        fused_kernel, fused_bias = self._fuse_kernel()

        # Create fused convolution
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                    self.stride, padding=0, groups=self.groups, bias=True)
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias
        self.explicit_pad = nn.ZeroPad2d(self.padding)

        # Preserve quantization config if present
        if self.is_quantized and hasattr(self.lk_conv, 'qconfig'):
            self.fused_conv.qconfig = self.lk_conv.qconfig

        # Cleanup unused parameters
        del self.lk_conv, self.sk_conv, self.lk_bias, self.sk_bias
        self.fused = True


class GatedConvFFN(nn.Module):
    """
    Gated Feed-Forward Network using 1x1 convolutions.
    
    Enhanced feature transformation with gated linear unit (GLU) variant.
    
    Args:
        in_channels: Input channels
        mlp_ratio: Hidden dimension multiplier
        drop: Dropout probability
    """
    def __init__(self, in_channels: int, mlp_ratio: float = 2.0, drop: float = 0.):
        super().__init__()
        hidden_channels = int(in_channels * mlp_ratio)

        # Convolutional components
        self.conv_gate = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv_main = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop2 = nn.Dropout(drop)

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))

        # Quantization support
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.conv_gate(x) * self.temperature
        main = self.conv_main(x)
        activated = self.act(gate)

        # Quantization-aware multiplication
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
    
    Lightweight channel attention mechanism.
    
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
        # Generate spatial descriptors
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
    
    Fuses to simple affine transform for optimized inference.
    
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

        # Register buffers for EMA statistics
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return x * self.weight + self.bias

        if self.training:
            # Calculate batch statistics
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            # Update running stats with EMA
            with torch.no_grad():
                self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize and affine transform
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

    def fuse(self):
        """Fuse normalization into scale and shift for deployment."""
        if self.fused:
            return

        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale

        self.weight.data = scale
        self.bias.data = shift
        self.fused = True


class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm implementation for 4D tensors.
    
    Provides stable normalization for GAN training.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)


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
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.,
                 drop_path: float = 0., lk_kernel: int = 11, sk_kernel: int = 3,
                 fused_init: bool = False, quantize_residual: bool = True,
                 use_channel_attn: bool = True, use_spatial_attn: bool = True,
                 norm_layer: nn.Module = DeploymentNorm, res_scale: float = 1.0):
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

        # Quantization-aware residual connection
        self.quantize_residual_flag = quantize_residual
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

        # Quantization stubs
        self.residual_quant = tq.QuantStub() if quantize_residual else nn.Identity()
        self.residual_dequant = tq.DeQuantStub() if quantize_residual else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.ffn(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        residual = self.drop_path(x)

        # Scaled residual addition
        scaled_residual = residual * self.res_scale

        if self.is_quantized:
            # Handle quantized tensors with proper scale/zero-point
            if self.quantize_residual_flag:
                shortcut_q = self.residual_quant(shortcut)
                output = self.quant_add.add(shortcut_q, scaled_residual)
                return self.residual_dequant(output)
            else:
                return self.quant_add.add(shortcut, scaled_residual)

        return shortcut + scaled_residual


class QuantFusion(nn.Module):
    """
    Multi-scale feature fusion with quantization support.
    
    Args:
        in_channels: Total input channels from all features
        out_channels: Output channels
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fusion_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.error_comp = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if not features:
            raise ValueError("QuantFusion requires at least one input feature")

        # Align feature resolutions
        target_size = features[0].shape[-2:]
        aligned_features = [
            F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            if feat.shape[-2:] != target_size else feat
            for feat in features
        ]

        x = torch.cat(aligned_features, dim=1)

        # Quantization-aware error compensation
        if self.is_quantized:
            x = self.quant_add.add(x, self.error_comp)
        else:
            x = x + self.error_comp

        return self.fusion_conv(x)


class AdaptiveUpsample(nn.Module):
    """
    Resolution-aware upsampling module.
    
    Handles various scaling factors using PixelShuffle.
    
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

class aether(nn.Module):
    """
    AetherNet: Production-Ready Super-Resolution Network.
    
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
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 11,
        sk_kernel: int = 3,
        scale: int = 4,
        img_range: float = 1.0,
        fused_init: bool = False,
        quantize_residual: bool = True,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
        norm_type: str = 'deployment',
        res_scale: float = 1.0,
    ):
        super().__init__()
        self.img_range = img_range
        self.register_buffer('scale', torch.tensor(scale))
        self.fused_init = fused_init
        self.embed_dim = embed_dim
        self.quantize_residual = quantize_residual
        self.num_stages = len(depths)
        self.is_quantized = False  # Global quantization flag

        # Input normalization (mean for [0,1] range)
        self.register_buffer('mean', torch.full((1, in_chans, 1, 1), 0.5))

        # --- 1. Initial Feature Extraction ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # --- 2. Deep Feature Processing ---
        # Normalization layer selection
        if norm_type.lower() == 'deployment':
            norm_layer = DeploymentNorm
        elif norm_type.lower() == 'layernorm':
            norm_layer = LayerNorm2d
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Choose 'deployment' or 'layernorm'.")

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
        
        # Ensure channel distribution is valid
        assert sum(fusion_out_channels) == embed_dim, "Channel distribution error"

        # Build stages
        block_idx = 0
        for i, depth in enumerate(depths):
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(AetherBlock(
                    dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop,
                    drop_path=dpr[block_idx + j], lk_kernel=lk_kernel, sk_kernel=sk_kernel,
                    fused_init=fused_init, quantize_residual=self.quantize_residual,
                    use_channel_attn=use_channel_attn, use_spatial_attn=use_spatial_attn,
                    norm_layer=norm_layer, res_scale=res_scale))
            self.stages.append(nn.Sequential(*stage_blocks))
            block_idx += depth
            # Fusion convolution for this stage
            self.fusion_convs.append(nn.Conv2d(embed_dim, fusion_out_channels[i], 1))

        # Feature fusion and processing
        self.quant_fusion_layer = QuantFusion(embed_dim, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- 3. Reconstruction ---
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), 
            nn.LeakyReLU(inplace=True))
        self.upsample = AdaptiveUpsample(scale, embed_dim)
        self.conv_last = nn.Conv2d(self.upsample.out_channels, in_chans, 3, 1, 1)

        # Initialize weights unless in fused mode
        if not self.fused_init:
            self.apply(self._init_weights)
        else:
            print("Skipping weight init for fused model - weights from checkpoint")

        # Quantization stubs
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AetherNet.
        
        Args:
            x: Input tensor in [0, 1] range (shape: [B, C, H, W])
            
        Returns:
            torch.Tensor: Output tensor in [0, 1] range
        """
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
        """Fuse reparameterizable layers for optimized inference."""
        if self.fused_init:
            print("Model already fused")
            return

        print("Fusing modules for inference...")
        for module in self.modules():
            if hasattr(module, 'fuse') and callable(module.fuse):
                if not isinstance(module, LayerNorm2d):
                    module.fuse()
        self.fused_init = True
        print("Fusion complete")

    def prepare_qat(self):
        """
        Prepare model for Quantization-Aware Training (QAT).
        
        Configures per-tensor quantization for compatibility and stability.
        """
        if any(isinstance(m, LayerNorm2d) for m in self.modules()):
            print("Warning: QAT with LayerNorm may affect stability")

        print("Configuring QAT with per-tensor quantization...")
        
        try:
            # Modern PyTorch (2.6+)
            from torch.ao.quantization import get_default_qat_qconfig_mapping
            qconfig_mapping = get_default_qat_qconfig_mapping()
            
            from torch.ao.quantization import default_fake_quant, default_per_tensor_weight_fake_quant
            per_tensor_qconfig = tq.QConfig(
                activation=default_fake_quant,
                weight=default_per_tensor_weight_fake_quant
            )
            qconfig_mapping = qconfig_mapping.set_global(per_tensor_qconfig)
            self.qconfig = per_tensor_qconfig
            print("Using modern per-tensor QAT configuration")
        except (ImportError, AttributeError):
            # Fallback for older PyTorch
            print("Using PyTorch 2.5.1 compatible QAT configuration")
            
            per_tensor_observer = MovingAverageMinMaxObserver.with_args(
                qscheme=torch.per_tensor_affine,
                reduce_range=False
            )
            
            # Custom fake quantization classes
            class SafePerTensorFakeQuantize(tq.FakeQuantize):
                def __init__(self, *args, **kwargs):
                    kwargs['observer'] = per_tensor_observer
                    kwargs['quant_min'] = 0
                    kwargs['quant_max'] = 255
                    super().__init__(*args, **kwargs)
                    
            class SafePerTensorWeightFakeQuantize(tq.FakeQuantize):
                def __init__(self, *args, **kwargs):
                    kwargs['observer'] = per_tensor_observer
                    kwargs['quant_min'] = -128
                    kwargs['quant_max'] = 127
                    kwargs['dtype'] = torch.qint8
                    super().__init__(*args, **kwargs)
            
            self.qconfig = tq.QConfig(
                activation=SafePerTensorFakeQuantize,
                weight=SafePerTensorWeightFakeQuantize
            )
            print("Using custom per-tensor QAT for compatibility")

        # Prepare model
        print("Preparing QAT model...")
        self.train()
        self.fuse_model()
        tq.prepare_qat(self, inplace=True)

        # Set quantization flags
        self.is_quantized = True
        for module in self.modules():
            if hasattr(module, 'is_quantized'):
                module.is_quantized = True

        # Exclude sensitive layers from quantization
        layers_to_float = ['conv_first', 'conv_last', 'conv_before_upsample.0']
        for name, module in self.named_modules():
            if name in layers_to_float:
                module.qconfig = None
                print(f"  - Excluded from quantization: {name}")

        print("Model ready for QAT")

    def convert_to_quantized(self) -> nn.Module:
        """
        Convert QAT model to fully quantized INT8 model.
        
        Returns:
            nn.Module: Quantized model ready for inference
        """
        if not self.is_quantized:
            raise RuntimeError("Model must be prepared with prepare_qat() first")

        self.eval()
        quantized_model = tq.convert(self, inplace=False)
        quantized_model.is_quantized = True
        
        # Verify no per-channel quantization remains
        print("Verifying quantization parameters...")
        for name, module in quantized_model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                scale = module.weight_fake_quant.scale
                if isinstance(scale, torch.Tensor) and scale.numel() > 1:
                    print(f"  ⚠️ Per-channel quantization detected in {name}")
                    # Convert to per-tensor
                    module.weight_fake_quant.scale = scale.mean()
                    zp = module.weight_fake_quant.zero_point
                    if isinstance(zp, torch.Tensor):
                        module.weight_fake_quant.zero_point = zp.mean().round().clamp(-128, 127)
        
        print("Converted to fully quantized INT8 model")
        return quantized_model

# ------------------- Network Presets ------------------- #

def aether_tiny(scale: int, **kwargs) -> aether:
    """Minimal version for real-time/mobile use (64 channels, 3 stages)."""
    return aether(embed_dim=64, depths=(3, 3, 3), scale=scale,
                  use_channel_attn=False, use_spatial_attn=False, **kwargs)

def aether_small(scale: int, **kwargs) -> aether:
    """Small version (96 channels, 4 stages)."""
    return aether(embed_dim=96, depths=(4, 4, 4, 4), scale=scale,
                  use_channel_attn=False, use_spatial_attn=False, **kwargs)

def aether_medium(scale: int, **kwargs) -> aether:
    """Balanced version (128 channels, 4 stages, attention)."""
    return aether(embed_dim=128, depths=(6, 6, 6, 6), scale=scale,
                  use_channel_attn=True, use_spatial_attn=True, **kwargs)

def aether_large(scale: int, **kwargs) -> aether:
    """High-quality version (180 channels, 5 stages, attention)."""
    return aether(embed_dim=180, depths=(8, 8, 8, 8, 8), scale=scale,
                  use_channel_attn=True, use_spatial_attn=True, **kwargs)

# ------------------- ONNX Export ------------------- #

def export_onnx(
    model: nn.Module, 
    scale: int, 
    precision: str = 'fp32'
) -> str:
    """
    Export model to ONNX format with optimizations.
    
    Args:
        model: Model to export
        scale: Super-resolution scale factor
        precision: Export precision ('fp32', 'fp16', 'int8')
        
    Returns:
        str: Path to exported ONNX file
    """
    model.eval()
    
    # Fuse model if not already fused
    if hasattr(model, 'fuse_model'):
        model.fuse_model()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    dummy_input = dummy_input.to(device)

    # Handle precision
    if precision == 'fp16':
        if next(model.parameters()).dtype != torch.float16:
            model = model.half()
        dummy_input = dummy_input.half()
    
    # Validate INT8 model
    if precision == 'int8':
        # Check if model is truly quantized
        is_truly_quantized = any(
            m.__class__.__module__.startswith('torch.ao.nn.quantized') 
            for m in model.modules()
        )
        if not is_truly_quantized:
            raise ValueError("INT8 export requires a fully quantized model")

    # Export
    onnx_filename = f"aether_net_x{scale}_{precision}.onnx"
    print(f"Exporting to {onnx_filename}...")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}
        },
        export_params=True,
        keep_initializers_as_inputs=True if precision != 'fp32' else None,
        verbose=False,
    )

    print(f"Successfully exported {precision.upper()} model")
    return onnx_filename