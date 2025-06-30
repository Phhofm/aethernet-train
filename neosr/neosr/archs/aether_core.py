# --- File Information ---
# Version: 1.1.0
# Author: Philip Hofmann (Original), with AI Assistant contributions
# License: MIT
# GitHub: https://github.com/phhofm/aethernet

"""
AetherNet Core Architecture - Single Source of Truth

Overview:
    This file contains the complete, self-contained implementation of the AetherNet
    super-resolution architecture. It is designed to be a single, reliable source
    file that can be easily integrated into various training and inference frameworks
    such as traiNNer-redux, neosr, or used directly with model-loading libraries
    like Spandrel.

Design & Architecture:
    AetherNet is a high-performance super-resolution network designed for an
    optimal balance of image quality, inference speed, and deployment flexibility.

    Key Architectural Features:
    - Reparameterizable Large Kernels (`ReparamLargeKernelConv`): Employs structural
      reparameterization to get the quality benefits of large convolutional kernels
      while fusing into a single, fast convolution for inference.
    - Gated Feed-Forward Network (`GatedConvFFN`): Uses a gated convolutional FFN
      (GLU variant) for more effective feature transformation than standard MLPs.
    - Multi-Stage Feature Fusion (`QuantFusion`): The network body is composed of
      multiple stages. Features from each stage are fused, allowing the network
      to combine shallow and deep features for superior reconstruction.
    - Deployment-Optimized Normalization (`DeploymentNorm`, `LayerNorm2d`): Features
      a custom normalization layer that fuses into a simple scale-and-shift
      operation for maximum ONNX/TensorRT performance. It also offers standard
      LayerNorm for enhanced training stability, crucial for GANs.
    - Attention Mechanisms (`DynamicChannelScaling`, `SpatialAttention`): Optional,
      lightweight channel and spatial attention modules to refine features with
      minimal computational overhead.
    - End-to-End Quantization Support: Built from the ground up to be compatible
      with PyTorch's Quantization-Aware Training (QAT) workflow, enabling high-
      performance INT8 deployment.

Strengths:
    - Performance: Fuses modules into a simple, linear structure that is extremely
      fast for inference on modern hardware.
    - Quality: The combination of large kernels, attention, and multi-stage fusion
      delivers high-quality image reconstruction.
    - Deployability: Includes a robust ONNX export function with support for FP32,
      FP16, and INT8 precision, ready for engines like TensorRT or ONNX Runtime.
    - Flexibility: Provides factory functions (`aether_small`, `aether_large`, etc.)
      and architecture options (`norm_type`, `res_scale`) to tailor the model to
      specific quality vs. speed requirements.

How to Use:
    1. Instantiation:
       >>> from aether_core import aether_medium
       >>> model = aether_medium(scale=4, norm_type='layernorm')

    2. Training:
       Train the model as a standard PyTorch nn.Module. For GAN training,
       using `norm_type='layernorm'` is recommended for stability.

    3. Preparing for Deployment (Fusion):
       >>> model.eval()
       >>> model.fuse_model() # Fuses layers for inference

    4. Quantization-Aware Training (QAT):
       >>> model.train()
       >>> model.prepare_qat()
       # ... continue training for a few epochs ...
       >>> quantized_model = model.convert_to_quantized()

    5. Exporting to ONNX:
       >>> from aether_core import export_onnx
       >>> export_onnx(model, scale=4, precision='fp32')
       >>> export_onnx(quantized_model, scale=4, precision='int8')
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from typing import Tuple, List, Dict, Any, Optional
import torch.ao.quantization as tq
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
import warnings


# Ignore quantization warnings for a cleaner user experience
warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")

# ------------------- Core Building Blocks ------------------- #

class DropPath(nn.Module):
    """
    Stochastic Depth with an ONNX-compatible implementation.

    This acts as a form of regularization, randomly dropping entire residual blocks
    during training.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize to 0 or 1
        return x.div(keep_prob) * random_tensor


class ReparamLargeKernelConv(nn.Module):
    """
    Efficient large kernel convolution using structural reparameterization.

    During training, it uses a large kernel and a parallel small kernel branch.
    For deployment, these are fused into a single, faster large-kernel convolution.
    This design is optimized for TensorRT inference.

    Input shape: (B, C_in, H, W)
    Output shape: (B, C_out, H_out, W_out)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        if kernel_size % 2 == 0 or small_kernel % 2 == 0:
            raise ValueError("Kernel sizes must be odd numbers for symmetrical padding.")

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
            # Deployment path: single convolution with explicit padding
            self.explicit_pad = nn.ZeroPad2d(self.padding)
            self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                        padding=0, groups=groups, bias=True)
        else:
            # Training path: two parallel convolutions
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
        """Internal method to compute the fused kernel and bias."""
        if self.fused:
            raise RuntimeError("This module is already fused.")

        # Pad the small kernel to the size of the large kernel
        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = F.pad(self.sk_conv.weight, [pad] * 4)

        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        return fused_kernel, fused_bias

    def fuse(self):
        """Fuses the large and small kernel branches into a single convolution."""
        if self.fused:
            return

        fused_kernel, fused_bias = self._fuse_kernel()

        # Create the new fused convolution layer
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                    self.stride, padding=0, groups=self.groups, bias=True)
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias
        self.explicit_pad = nn.ZeroPad2d(self.padding)

        if self.is_quantized and hasattr(self.lk_conv, 'qconfig'):
            self.fused_conv.qconfig = self.lk_conv.qconfig

        # Remove old parameters
        del self.lk_conv, self.sk_conv, self.lk_bias, self.sk_bias
        self.fused = True


class GatedConvFFN(nn.Module):
    """
    Gated Feed-Forward Network using 1x1 convolutions.

    This replaces the standard MLP layer with a gated linear unit (GLU) variant,
    which can be more effective for vision tasks. A trainable `temperature`
    parameter is included to scale the gate activation, potentially improving
    training stability and control over the information flow.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
    """
    def __init__(self, in_channels: int, mlp_ratio: float = 2.0, drop: float = 0.):
        super().__init__()
        hidden_channels = int(in_channels * mlp_ratio)

        self.conv_gate = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv_main = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop2 = nn.Dropout(drop)

        # A trainable scalar to control the gate's sensitivity
        self.temperature = nn.Parameter(torch.ones(1))

        # Quantization-aware multiplication
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
    Efficient Channel Attention (Squeeze-and-Excitation) using 1x1 Convolutions.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
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
    Lightweight and efficient spatial attention module.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
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
    A deployment-friendly normalization layer designed for ONNX compatibility.

    Behaves like a standard LayerNorm (normalizing across spatial dimensions
    per channel) but uses EMA statistics like BatchNorm for inference. This makes
    it stable and allows it to be fused into a simple affine transformation
    (scale and shift) for maximum inference speed.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
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
            # Calculate batch statistics across spatial and batch dims
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            # Update running stats with EMA
            with torch.no_grad():
                self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

    def fuse(self):
        """Fuses the normalization into a scale and shift for deployment."""
        if self.fused:
            return

        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale

        self.weight.data = scale
        self.bias.data = shift
        self.fused = True


class LayerNorm2d(nn.LayerNorm):
    """
    A LayerNorm implementation for 4D tensors (B, C, H, W).

    This provides a stable normalization alternative to DeploymentNorm,
    especially useful for GAN training. It is not fusable.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x


class AetherBlock(nn.Module):
    """
    The core building block of AetherNet.

    It features a reparameterizable convolution, a gated FFN, optional channel
    and spatial attention, and a robustly implemented quantized residual connection.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
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

        # --- Quantization-aware residual connection ---
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
        residual = self.drop_path(x)

        # The scaled residual is added to the shortcut.
        scaled_residual = residual * self.res_scale

        if self.is_quantized:
            # In quantized mode, use the functional `add` which correctly handles
            # tensors with different scales and zero-points.
            if self.quantize_residual_flag:
                shortcut_q = self.residual_quant(shortcut)
                output = self.quant_add.add(shortcut_q, scaled_residual)
                return self.residual_dequant(output)
            else:
                return self.quant_add.add(shortcut, scaled_residual)

        return shortcut + scaled_residual


class QuantFusion(nn.Module):
    """
    Multi-scale feature fusion layer with quantization support.

    This layer takes a list of feature maps (from different network stages),
    aligns their resolutions, concatenates them, and then fuses them using a
    1x1 convolution.

    Input: List of tensors [ (B, C_i, H_i, W_i) ]
    Output: (B, C_out, H, W) where H, W = size of first tensor
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

        target_size = features[0].shape[-2:]
        aligned_features = [
            F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            if feat.shape[-2:] != target_size else feat
            for feat in features
        ]

        x = torch.cat(aligned_features, dim=1)

        # Apply quantization-aware error compensation
        if self.is_quantized:
            x = self.quant_add.add(x, self.error_comp)
        else:
            x = x + self.error_comp

        return self.fusion_conv(x)


class AdaptiveUpsample(nn.Module):
    """
    Resolution-aware upsampling module using PixelShuffle.

    Handles different integer scaling factors (powers of 2 and 3) while
    managing channel dimensions to maintain computational efficiency.

    Input shape: (B, C, H, W)
    Output shape: (B, C_out, H*scale, W*scale)
    """
    def __init__(self, scale: int, in_channels: int):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.blocks = nn.ModuleList()
        self.out_channels = max(32, in_channels // max(1, scale // 2))

        if (scale & (scale - 1)) == 0 and scale != 1:  # Power of 2
            num_ups = int(math.log2(scale))
            current_channels = in_channels
            for i in range(num_ups):
                next_channels = self.out_channels if (i == num_ups - 1) else current_channels // 2
                self.blocks.append(nn.Conv2d(current_channels, 4 * next_channels, 3, 1, 1))
                self.blocks.append(nn.PixelShuffle(2))
                current_channels = next_channels
        elif scale == 3:
            self.blocks.append(nn.Conv2d(in_channels, 9 * self.out_channels, 3, 1, 1))
            self.blocks.append(nn.PixelShuffle(3))
        elif scale == 1:
            self.blocks.append(nn.Conv2d(in_channels, self.out_channels, 3, 1, 1))
        else:
            raise ValueError(f"Unsupported scale: {scale}. Only 1, 3 and powers of 2 are supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

# ------------------- Main AetherNet Architecture ------------------- #

class aether(nn.Module):
    """
    AetherNet: A Production-Ready Super-Resolution Network.

    The model expects input tensors to be in the [0, 1] range.
    For full documentation, see the docstring at the top of this file.
    """
    def _init_weights(self, m: nn.Module):
        """Initializes weights for various layer types."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (DeploymentNorm, nn.LayerNorm, nn.GroupNorm)):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)

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

        # Input normalization buffer, dynamically created based on in_chans
        self.register_buffer('mean', torch.full((1, in_chans, 1, 1), 0.5))

        # --- 1. Initial Feature Extraction ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # --- 2. Deep Feature Processing ---
        # Select normalization layer based on user choice
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

        # Robustly calculate output channels for each fusion conv to handle non-divisible cases
        base_ch = embed_dim // self.num_stages
        remainder = embed_dim % self.num_stages
        fusion_out_channels = []
        for i in range(self.num_stages):
            if i < remainder:
                fusion_out_channels.append(base_ch + 1)
            else:
                fusion_out_channels.append(base_ch)
        
        # This ensures sum(fusion_out_channels) == embed_dim
        assert sum(fusion_out_channels) == embed_dim, "Channel distribution logic is flawed."

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
            # Use the pre-calculated channel count for this fusion convolution
            self.fusion_convs.append(nn.Conv2d(embed_dim, fusion_out_channels[i], 1))

        self.quant_fusion_layer = QuantFusion(embed_dim, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- 3. Reconstruction ---
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = AdaptiveUpsample(scale, embed_dim)
        self.conv_last = nn.Conv2d(self.upsample.out_channels, in_chans, 3, 1, 1)

        if not self.fused_init:
            self.apply(self._init_weights)
        else:
            print("Skipping weight init for fused model - weights expected from checkpoint.")

        # Quantization stubs for input/output
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AetherNet.
        Args:
            x (torch.Tensor): Input low-resolution image tensor in range [0, 1].
        Returns:
            torch.Tensor: Output high-resolution image tensor in range [0, 1].
        """
        x = (x - self.mean) * self.img_range
        x = self.quant(x) if self.is_quantized else x

        x_first = self.conv_first(x)

        features = []
        out = x_first
        for stage, fusion_conv in zip(self.stages, self.fusion_convs):
            out = stage(out)
            features.append(fusion_conv(out))

        fused_features = self.quant_fusion_layer(features)
        body_out = fused_features + x_first

        body_out = self.conv_after_body(self.norm(body_out)) + body_out

        recon = self.conv_before_upsample(body_out)
        recon = self.upsample(recon)
        recon = self.conv_last(recon)

        output = self.dequant(recon) if self.is_quantized else recon
        return output / self.img_range + self.mean

    def fuse_model(self):
        """Fuses reparameterizable and normalizable layers for deployment."""
        if self.fused_init:
            print("Model is already fused.")
            return

        print("Fusing modules for optimal inference...")
        for module in self.modules():
            if hasattr(module, 'fuse') and callable(module.fuse):
                # Only fuse if it's not a LayerNorm2d, which is not fusable
                if not isinstance(module, LayerNorm2d):
                    module.fuse()
        self.fused_init = True
        print("Fusion complete.")

    def prepare_qat(self):
            """Prepares the model for Quantization-Aware Training (QAT)."""
            if any(isinstance(m, LayerNorm2d) for m in self.modules()):
                print("Warning: QAT is enabled with LayerNorm, which is not typically quantized.")

            # --- START OF THE ULTIMATE WORKAROUND ---
            print("INFO: Creating a custom QAT configuration with PER-TENSOR weight quantization for maximum compatibility.")
            
            try:
                # Modern PyTorch (2.6+): Use QConfigMapping API
                from torch.ao.quantization import get_default_qat_qconfig_mapping
                qconfig_mapping = get_default_qat_qconfig_mapping()
                
                # Create a new per-tensor QConfig
                from torch.ao.quantization import default_fake_quant, default_per_tensor_weight_fake_quant
                per_tensor_qconfig = tq.QConfig(
                    activation=default_fake_quant,
                    weight=default_per_tensor_weight_fake_quant
                )
                qconfig_mapping = qconfig_mapping.set_global(per_tensor_qconfig)
                self.qconfig = per_tensor_qconfig
                print("Using modern per-tensor QAT configuration (QConfigMapping).")
            except (ImportError, AttributeError):
                # Fallback for PyTorch 2.5.1
                print("Warning: Using PyTorch 2.5.1 compatible QAT configuration.")
                
                # Create a safe per-tensor observer
                per_tensor_observer = MovingAverageMinMaxObserver.with_args(
                    qscheme=torch.per_tensor_affine,
                    reduce_range=False
                )
                
                # Create custom FakeQuantize classes
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
                print("Using custom per-tensor QAT configuration for PyTorch 2.5.1")
            # --- END OF THE ULTIMATE WORKAROUND ---

            print("Preparing model for Quantization-Aware Training...")
            # Move these calls INSIDE the method
            self.train()
            self.fuse_model()
            
            # Apply QAT preparation
            tq.prepare_qat(self, inplace=True)

            self.is_quantized = True
            for module in self.modules():
                if hasattr(module, 'is_quantized'):
                    module.is_quantized = True

            # Disable quantization for sensitive layers
            layers_to_float = ['conv_first', 'conv_last', 'conv_before_upsample.0']
            for name, module in self.named_modules():
                if name in layers_to_float:
                    module.qconfig = None
                    print(f"  - Disabled quantization for sensitive layer: {name}")

            print("AetherNet prepared for QAT.")

    def convert_to_quantized(self) -> nn.Module:
        """Converts a QAT-trained model to a true integer-based quantized model."""
        if not self.is_quantized:
            raise RuntimeError("Model must be prepared with prepare_qat() before conversion.")

        self.eval()
        quantized_model = tq.convert(self, inplace=False)
        quantized_model.is_quantized = True
        print("Converted to a fully quantized INT8 model.")
        return quantized_model

# ------------------- Network Options ------------------- #

def aether_tiny(scale: int, **kwargs) -> aether:
    """A minimal and extremely fast version of AetherNet for real-time or mobile use."""
    return aether(embed_dim=64, depths=(3, 3, 3), scale=scale,
                  use_channel_attn=False, use_spatial_attn=False, **kwargs)

def aether_small(scale: int, **kwargs) -> aether:
    """A small and fast version of AetherNet."""
    return aether(embed_dim=96, depths=(4, 4, 4, 4), scale=scale,
                  use_channel_attn=False, use_spatial_attn=False, **kwargs)

def aether_medium(scale: int, **kwargs) -> aether:
    """A balanced version of AetherNet (default)."""
    return aether(embed_dim=128, depths=(6, 6, 6, 6), scale=scale,
                  use_channel_attn=True, use_spatial_attn=True, **kwargs)

def aether_large(scale: int, **kwargs) -> aether:
    """A larger, more powerful version of AetherNet for higher quality."""
    return aether(embed_dim=180, depths=(8, 8, 8, 8, 8), scale=scale,
                  use_channel_attn=True, use_spatial_attn=True, **kwargs)

# ------------------- Deployment Utilities ------------------- #

def export_onnx(
    model,  # Remove type annotation to avoid import issues
    scale: int,
    precision: str = 'fp32',
):
    """
    Exports the AetherNet model to the ONNX format with optimizations.
    """
    import torch  # Import inside function to ensure availability
    from torch import nn  # Import inside function
    
    model.eval()
    if hasattr(model, 'fuse_model'):
        model.fuse_model()

    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    dummy_input = dummy_input.to(device)

    if precision == 'fp16':
        if next(model.parameters()).dtype != torch.float16:
            model = model.half()
        dummy_input = dummy_input.half()
    
    if precision == 'int8':
        # Check if model is truly quantized
        is_truly_quantized = any(
            m.__class__.__module__.startswith('torch.ao.nn.quantized') 
            for m in model.modules()
        )
        if not is_truly_quantized:
            raise ValueError("To export to INT8, the model must be a fully converted quantized model.")

    onnx_filename = f"aether_net_x{scale}_{precision}.onnx"
    print(f"Exporting model to {onnx_filename}...")

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

    print(f"Successfully exported {precision.upper()} model to {onnx_filename}")
    return onnx_filename