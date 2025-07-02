# --- FILE: aether_core.py ---

# -*- coding: utf-8 -*-
# --- File Information ---
# Version: 1.0.0
# Author: Philip Hofmann
# License: MIT
# GitHub: https://github.com/phhofm/aethernet
# Description: Ultra-Fast Super-Resolution Network with QAT Support

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
from packaging import version
import onnx

# Suppress quantization warnings for a cleaner user experience.
warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")

# --- Version Compatibility Setup ---
PT_VERSION = version.parse(torch.__version__)
MIN_PT_VERSION = version.parse("1.10.0")
REC_PT_VERSION = version.parse("2.0.0")

if PT_VERSION < MIN_PT_VERSION:
    raise RuntimeError(f"PyTorch {MIN_PT_VERSION}+ required (detected {PT_VERSION})")
if PT_VERSION < REC_PT_VERSION:
    warnings.warn(
        f"PyTorch {REC_PT_VERSION}+ recommended for optimal performance and quantization features. "
        f"Detected version: {PT_VERSION}. Some features may be limited."
    )

def parse_version(ver_str: str) -> List[int]:
    parts = []
    ver_str_clean = ver_str.split('+')[0]
    for part in ver_str_clean.split('.')[:3]:
        digits = ''.join(filter(str.isdigit, part))
        parts.append(int(digits) if digits else 0)
    return parts + [0] * (3 - len(parts))

# --- Core Building Blocks ---

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ReparamLargeKernelConv(nn.Module):
    """
    Implements a large-kernel convolution via structural reparameterization.
    For inference, the parallel branches (large kernel, small kernel, biases)
    are fused into a single, standard convolution layer for maximum speed.
    This fused convolution uses a standard `padding` argument to be compatible
    with all backends, including the INT8 ONNX exporter.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        if kernel_size % 2 == 0 or small_kernel % 2 == 0:
            raise ValueError("Kernel sizes must be odd for symmetrical padding")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.groups = kernel_size, stride, groups
        self.padding = kernel_size // 2
        self.small_kernel = small_kernel
        self.fused = fused_init
        self.is_quantized = False

        if self.fused:
            self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                        padding=self.padding, groups=groups, bias=True)
        else:
            self.lk_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                     self.padding, groups=groups, bias=False)
            self.sk_conv = nn.Conv2d(in_channels, out_channels, small_kernel, stride,
                                     small_kernel // 2, groups=groups, bias=False)
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.fused_conv(x)
        
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return (lk_out + self.lk_bias.view(1, -1, 1, 1) +
                sk_out + self.sk_bias.view(1, -1, 1, 1))

    def _fuse_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fused: raise RuntimeError("Module is already fused")
        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = F.pad(self.sk_conv.weight, [pad] * 4)
        return self.lk_conv.weight + sk_kernel_padded, self.lk_bias + self.sk_bias

    def fuse(self):
        if self.fused: return
        fused_kernel, fused_bias = self._fuse_kernel()
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                    self.stride, padding=self.padding, groups=self.groups, bias=True)
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias
        
        if self.is_quantized and hasattr(self, 'lk_conv') and hasattr(self.lk_conv, 'qconfig'):
            self.fused_conv.qconfig = self.lk_conv.qconfig
        
        # Clean up training-time layers
        for attr in ['lk_conv', 'sk_conv', 'lk_bias', 'sk_bias', 'explicit_pad']:
            if hasattr(self, attr):
                delattr(self, attr)
        self.fused = True

class GatedConvFFN(nn.Module):
    """Gated Feed-Forward Network using 1x1 convolutions."""
    def __init__(self, in_channels: int, mlp_ratio: float = 1.5, drop: float = 0.):
        super().__init__()
        hidden_channels = int(in_channels * mlp_ratio)
        self.conv_gate = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv_main = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop1, self.drop2 = nn.Dropout(drop), nn.Dropout(drop)
        self.temperature = nn.Parameter(torch.ones(1))
        self.is_quantized = False
        self.quant_mul = torch.nn.quantized.FloatFunctional()

        # Float islands for operators not supported by the INT8 ONNX exporter
        self.act_dequant = tq.DeQuantStub()
        self.act_quant = tq.QuantStub()
        self.temp_dequant = tq.DeQuantStub()
        self.temp_quant = tq.QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_unscaled = self.conv_gate(x)
        
        # The temperature multiplication is not supported in the quantized domain by ONNX.
        # We create a "float island": dequantize -> multiply -> requantize.
        if self.is_quantized:
            gate_float = self.temp_dequant(gate_unscaled)
            gate_scaled = gate_float * self.temperature
            gate = self.temp_quant(gate_scaled)
        else:
            gate = gate_unscaled * self.temperature
        
        main = self.conv_main(x)
        
        # The SiLU activation is also not supported in the INT8 quantized domain.
        gate = self.act_dequant(gate)
        activated = self.act(gate)
        activated = self.act_quant(activated)
        
        # The element-wise multiplication can be unstable in FP16.
        # We cast to FP32 for the operation and then cast back.
        if x.dtype == torch.float16:
             x = activated.float() * main.float()
             x = x.half()
        elif self.is_quantized:
            x = self.quant_mul.mul(activated, main)
        else:
            x = activated * main

        x = self.drop1(x)
        x = self.conv_out(x)
        return self.drop2(x)

class DynamicChannelScaling(nn.Module):
    """Squeeze-and-Excitation channel attention module."""
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid())
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x))
        if self.is_quantized:
            return self.quant_mul.mul(x, scale)
        return x * scale

class SpatialAttention(nn.Module):
    """Lightweight spatial attention module."""
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
    A LayerNorm-like layer that can be fused into a simple affine transformation
    (scale and shift) for efficient inference.
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
            mean, var = self.running_mean, self.running_var
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

    def fuse(self):
        if self.fused: return
        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        self.weight.data, self.bias.data = scale, shift
        if hasattr(self, 'running_mean'): del self.running_mean
        if hasattr(self, 'running_var'): del self.running_var
        self.fused = True

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for 4D tensors."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class AetherBlock(nn.Module):
    """Core building block of AetherNet."""
    def __init__(self, dim: int, mlp_ratio: float, drop: float,
                 drop_path: float, lk_kernel: int, sk_kernel: int,
                 fused_init: bool, norm_layer: nn.Module, res_scale: float, **_):
        super().__init__()
        self.res_scale = res_scale
        self.conv = ReparamLargeKernelConv(dim, dim, lk_kernel, 1, dim, sk_kernel, fused_init)
        self.norm = norm_layer(dim)
        self.ffn = GatedConvFFN(dim, mlp_ratio, drop)
        self.channel_attn = DynamicChannelScaling(dim)
        self.spatial_attn = SpatialAttention()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

        # Float islands for operators not supported by the INT8 ONNX exporter.
        self.norm_dequant = tq.DeQuantStub()
        self.norm_quant = tq.QuantStub()
        self.res_dequant = tq.DeQuantStub()
        self.res_quant = tq.QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)
        # Float island for the custom normalization layer.
        x = self.norm_dequant(x)
        x = self.norm(x)
        x = self.norm_quant(x)
        x = self.ffn(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        residual_unscaled = self.drop_path(x)
        
        # Float island for the residual scaling multiplication.
        if self.is_quantized:
            res_float = self.res_dequant(residual_unscaled)
            res_scaled = res_float * self.res_scale
            residual = self.res_quant(res_scaled)
        else:
            residual = residual_unscaled * self.res_scale

        # Use quantization-aware addition for quantized models.
        if self.is_quantized:
            return self.quant_add.add(shortcut, residual)
        else:
            return shortcut + residual

class QuantFusion(nn.Module):
    """Multi-scale feature fusion with quantization support."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fusion_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.error_comp = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.is_quantized = False

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if not features: raise ValueError("QuantFusion requires at least one feature")
        target_size = features[0].shape[-2:]
        aligned_features = [F.interpolate(feat, size=target_size, mode='nearest') if feat.shape[-2:] != target_size else feat for feat in features]
        x = torch.cat(aligned_features, dim=1)
        fused = self.fusion_conv(x)
        
        # The float `error_comp` parameter cannot be added in the quantized path.
        # It is bypassed for INT8 models but included for FP32/FP16 models.
        if self.is_quantized:
            return fused
        return fused + self.error_comp

class AdaptiveUpsample(nn.Module):
    """
    Resolution-aware upsampling module using PixelShuffle. This module is
    kept as a float-only module during quantization to avoid ONNX export
    issues with quantized PixelShuffle (DepthToSpace).
    """
    def __init__(self, scale: int, in_channels: int):
        super().__init__()
        self.scale, self.in_channels = scale, in_channels
        self.blocks = nn.ModuleList()
        self.out_channels = max(32, (in_channels // max(1, scale // 2)) & -2)
        
        if (scale & (scale - 1)) == 0 and scale != 1: # Power of 2
            for i in range(int(math.log2(scale))):
                in_ch = in_channels if i == 0 else self.out_channels
                out_ch = self.out_channels
                self.blocks.append(nn.Conv2d(in_ch, 4 * out_ch, 3, 1, 1))
                self.blocks.append(nn.PixelShuffle(2))
        elif scale == 3:
            self.blocks.append(nn.Conv2d(in_channels, 9 * self.out_channels, 3, 1, 1))
            self.blocks.append(nn.PixelShuffle(3))
        elif scale == 1:
            self.blocks.append(nn.Conv2d(in_channels, self.out_channels, 3, 1, 1))
        else:
            raise ValueError(f"Unsupported scale: {scale}. Only 1, 3 and powers of 2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

# --- Main AetherNet Architecture ---

class AetherNet(nn.Module):
    """
    AetherNet: An ultra-fast, production-ready super-resolution network.

    This architecture is designed for high-speed inference across multiple
    hardware backends. It leverages structural reparameterization to fuse
    multiple convolutional branches into a single, efficient layer for deployment.
    It is fully compatible with PyTorch's Quantization-Aware Training (QAT)
    workflow, enabling high-performance INT8 inference with minimal quality loss.

    The key design principles are:
    1.  **Inference-Time Speed:** All complex structures exist only at training
        time and are fused into simple, fast layers before deployment.
    2.  **Quantization-Friendly:** The architecture is designed with quantization
        in mind, using float islands for unsupported operations to ensure
        successful conversion to INT8.
    3.  **Deployment-Centric:** Includes built-in mechanisms for versioning,
        configuration saving, and robust ONNX export with metadata for
        frameworks like Spandrel and ChaiNNer.
    """
    MODEL_VERSION = "3.2.0"
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (DeploymentNorm, nn.LayerNorm, nn.GroupNorm)):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)

    def __init__(self, in_chans: int = 3, embed_dim: int = 96, depths: Tuple[int, ...] = (4, 4, 4, 4),
                 mlp_ratio: float = 1.5, drop: float = 0.0, drop_path_rate: float = 0.1,
                 lk_kernel: int = 13, sk_kernel: int = 5, scale: int = 4, img_range: float = 1.0,
                 fused_init: bool = False, norm_type: str = 'deployment', **kwargs):
        super().__init__()
        self.arch_config = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
        if isinstance(self.arch_config['depths'], tuple):
            self.arch_config['depths'] = list(self.arch_config['depths'])
        
        self.img_range, self.fused_init, self.embed_dim = img_range, fused_init, embed_dim
        self.num_stages = len(depths)
        self.is_quantized = False
        
        self.register_buffer('pt_version', torch.tensor(parse_version(torch.__version__)))
        self.register_buffer('model_version', torch.tensor(parse_version(self.MODEL_VERSION)))
        self.register_buffer('mean', torch.full((1, in_chans, 1, 1), 0.5))
        self.register_buffer('scale_tensor', torch.tensor(scale, dtype=torch.int64))

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        norm_layer = DeploymentNorm if norm_type.lower() == 'deployment' else LayerNorm2d

        self.stages, self.fusion_convs = nn.ModuleList(), nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        block_idx = 0
        for i, depth in enumerate(depths):
            stage_blocks = [AetherBlock(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop,
                              drop_path=dpr[block_idx + j], lk_kernel=lk_kernel, sk_kernel=sk_kernel,
                              fused_init=fused_init, norm_layer=norm_layer, **self.arch_config)
                            for j in range(depth)]
            self.stages.append(nn.Sequential(*stage_blocks))
            block_idx += depth
            self.fusion_convs.append(nn.Conv2d(embed_dim, embed_dim // self.num_stages, 1))

        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.quant_fusion_layer = QuantFusion(embed_dim, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = AdaptiveUpsample(scale, embed_dim)
        self.conv_last = nn.Conv2d(self.upsample.out_channels, in_chans, 3, 1, 1)

        if not self.fused_init: self.apply(self._init_weights)

        self.quant, self.dequant = tq.QuantStub(), tq.DeQuantStub()
        self.body_norm_dequant, self.body_norm_quant = tq.DeQuantStub(), tq.QuantStub()
        self.upsample_dequant, self.upsample_quant = tq.DeQuantStub(), tq.QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = (x - self.mean) * self.img_range
        x = self.quant(x_in)
        x_first = self.conv_first(x)
        features = [fusion_conv(stage(x if i == 0 else features[-1])) for i, (stage, fusion_conv) in enumerate(zip(self.stages, self.fusion_convs))]
        fused_features = self.quant_fusion_layer(features)
        
        body_out = self.body_norm_dequant(fused_features)
        body_out = self.norm(body_out)
        body_out = self.body_norm_quant(body_out)
        
        if self.is_quantized:
            body_out_res = self.quant_add.add(body_out, x_first)
            body_out = self.quant_add.add(self.conv_after_body(body_out_res), body_out_res)
        else:
            body_out_res = body_out + x_first
            body_out = self.conv_after_body(body_out_res) + body_out_res
        
        recon = self.conv_before_upsample(body_out)
        
        if self.is_quantized:
            recon = self.upsample_dequant(recon)
            recon = self.upsample(recon)
            recon = self.upsample_quant(recon)
        else:
            recon = self.upsample(recon)
            
        recon = self.conv_last(recon)
        output = self.dequant(recon)
        return output / self.img_range + self.mean

    def fuse_model(self):
        if self.fused_init: return
        for module in self.modules():
            if hasattr(module, 'fuse') and callable(module.fuse):
                module.fuse()
        self.fused_init = True

    def prepare_qat(self, per_channel: bool = False):
        if per_channel and PT_VERSION < version.parse("1.10.0"):
            warnings.warn("Per-channel quantization requires PyTorch 1.10+. Disabling.")
            per_channel = False
        
        act_obs = MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8)
        wt_obs = MovingAveragePerChannelMinMaxObserver.with_args(qscheme=torch.per_channel_symmetric, dtype=torch.qint8) if per_channel else MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
        self.qconfig = tq.QConfig(activation=act_obs, weight=wt_obs)
        
        self.fuse_model()
        self.upsample.qconfig = None # Exclude upsampler from quantization
        tq.prepare_qat(self, inplace=True)
        self._set_quantization_flags(True)

    def _set_quantization_flags(self, status: bool):
        for module in self.modules():
            if hasattr(module, 'is_quantized'):
                module.is_quantized = status

    def convert_to_quantized(self) -> nn.Module:
        if not self.is_quantized: raise RuntimeError("Model must be prepared with prepare_qat() first")
        self.eval()
        quantized_model = tq.convert(self, inplace=False)
        quantized_model._set_quantization_flags(True)
        return quantized_model

    def verify_quantization(self):
        """Check quantization status of all layers."""
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
        
        pass

    @classmethod
    def load_optimized(cls, filename: str, device='cuda'):
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
        
        if is_qat_model:
            model.eval() # Important after loading a QAT state
        
        return model.to(device)
        
    def _get_architecture_name(self) -> str:
        if self.embed_dim <= 64: return "aether_tiny"
        if self.embed_dim <= 96: return "aether_small"
        if self.embed_dim <= 128: return "aether_medium"
        if self.embed_dim <= 180: return "aether_large"
        return "custom"
        
    def stabilize_for_fp16(self):
        """
        Apply comprehensive FP16 stabilization techniques.
        This should be called on the FP32 model *before* converting to .half().
        """
        print("Applying FP16 stabilization...")
        for module in self.modules():
            # Stabilize DeploymentNorm layers post-fusion
            if isinstance(module, DeploymentNorm) and module.fused:
                # After fusion, 'weight' is the scale and 'bias' is the shift.
                # Clamp these to prevent extreme values that cause FP16 overflow.
                module.weight.data.clamp_(min=-100.0, max=100.0)
                module.bias.data.clamp_(min=-100.0, max=100.0)
                
            # Stabilize GatedConvFFN temperature
            if isinstance(module, GatedConvFFN):
                module.temperature.data.clamp_(min=0.01, max=10.0)
                
        # Also clamp general weights and biases to a safe range
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
    """
    model.eval()
    model.fuse_model()

    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    device = next(model.parameters(), None)
    if device is None:
        device = 'cpu'
    else:
        device = device.device
    dummy_input = dummy_input.to(device)
    
    # Handle precision
    if precision == 'fp16':
        model.half()
        dummy_input = dummy_input.half()
    
    if precision == 'int8' and not model.is_quantized:
        raise ValueError("INT8 export requires a quantized model. Ensure it has been converted.")

    model_name = model._get_architecture_name() if hasattr(model, '_get_architecture_name') else "aether"
    onnx_filename = output_path or f"{model_name}_x{scale}_{precision}.onnx"
    
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}
    }

    torch.onnx.export(
        model, dummy_input, onnx_filename, opset_version=18,
        do_constant_folding=True, input_names=['input'], output_names=['output'],
        dynamic_axes=dynamic_axes,
    )
    
    try:
        model_onnx = onnx.load(onnx_filename)
        
        meta = model_onnx.metadata_props
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