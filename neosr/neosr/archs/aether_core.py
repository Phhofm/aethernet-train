# --- File Information ---
# Version: 2.0.0 (Definitive, Clean Architecture)
# Author: Philip Hofmann
# License: MIT
# GitHub: https://github.com/phhofm/aethernet
# Description: Production-ready super-resolution network with quantization support.

import math
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.ao import quantization as tq
from torch.nn.quantized import FloatFunctional
from torch.quantization.observer import MovingAverageMinMaxObserver

# Suppress quantization warnings for cleaner output during QAT
warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")

# --- Core Building Blocks ---

class DropPath(nn.Module):
    """Stochastic Depth with ONNX-compatible implementation."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()

class ReparamLargeKernelConv(nn.Module):
    """Efficient large kernel convolution using structural reparameterization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, small_kernel=3, fused_init=False):
        super().__init__()
        self.kernel_size, self.small_kernel = kernel_size, small_kernel
        self.in_channels, self.out_channels = in_channels, out_channels
        self.stride, self.groups, self.padding = stride, groups, (kernel_size - 1) // 2
        if kernel_size % 2 == 0 or small_kernel % 2 == 0: raise ValueError("Kernel sizes must be odd.")
        
        if fused_init:
            self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=True)
        else:
            self.lk_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=True)
            self.sk_conv = nn.Conv2d(in_channels, out_channels, small_kernel, stride, (small_kernel - 1) // 2, groups=groups, bias=True)
            
    def forward(self, x):
        if hasattr(self, 'fused_conv'): return self.fused_conv(x)
        return self.lk_conv(x) + self.sk_conv(x)

    def _fuse_kernel(self):
        if not hasattr(self, 'lk_conv'): return self.fused_conv.weight.data, self.fused_conv.bias.data
        pad = (self.kernel_size - self.small_kernel) // 2
        sk_padded = F.pad(self.sk_conv.weight, [pad] * 4)
        return self.lk_conv.weight + sk_padded, self.lk_conv.bias + self.sk_conv.bias

    def fuse(self):
        if hasattr(self, 'fused_conv'): return
        fused_k, fused_b = self._fuse_kernel()
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups, bias=True)
        if hasattr(self.lk_conv, 'qconfig'): self.fused_conv.qconfig = self.lk_conv.qconfig
        self.fused_conv.weight.data.copy_(fused_k); self.fused_conv.bias.data.copy_(fused_b)
        for attr in ['lk_conv', 'sk_conv']: self.__delattr__(attr)

class GatedConvFFN(nn.Module):
    """Gated Feed-Forward Network using 1x1 convolutions."""
    def __init__(self, in_channels, mlp_ratio=2., drop=0.):
        super().__init__()
        hidden = int(in_channels * mlp_ratio)
        self.conv_gate = nn.Conv2d(in_channels, hidden, 1)
        self.conv_main = nn.Conv2d(in_channels, hidden, 1)
        self.conv_out = nn.Conv2d(hidden, in_channels, 1)
        self.drop, self.act, self.temperature = nn.Dropout(drop), nn.GELU(), nn.Parameter(torch.ones(1))
        self.quant_mul = FloatFunctional()
    def forward(self, x):
        gate, main = self.conv_gate(x) * self.temperature, self.conv_main(x)
        return self.conv_out(self.drop(self.quant_mul.mul(self.act(gate), main)))

class DynamicChannelScaling(nn.Module):
    """Efficient Channel Attention (Squeeze-and-Excitation)."""
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(dim, dim // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(dim // reduction, dim, bias=False), nn.Sigmoid())
    def forward(self, x): b, c, _, _ = x.size(); return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """Lightweight spatial attention module."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv, self.sigmoid = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False), nn.Sigmoid()
    def forward(self, x): return x * self.sigmoid(self.conv(torch.cat([torch.max(x, 1, True)[0], torch.mean(x, 1, True)], 1)))

class DeploymentNorm(nn.Module):
    """Deployment-friendly normalization layer."""
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight, self.bias = nn.Parameter(torch.ones(channels)), nn.Parameter(torch.zeros(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean, var = x.mean(dim=(0, 2, 3), keepdim=True), x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            self.running_mean.copy_(self.running_mean * 0.9 + mean.squeeze() * 0.1)
            self.running_var.copy_(self.running_var * 0.9 + var.squeeze() * 0.1)
            stats_mean, stats_var = mean, var
        else: stats_mean, stats_var = self.running_mean.view(1,-1,1,1), self.running_var.view(1,-1,1,1)
        return ((x - stats_mean) / torch.sqrt(stats_var + self.eps)) * self.weight.view(1,-1,1,1) + self.bias.view(1,-1,1,1)
    def fuse(self): pass

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm implementation for 4D tensors."""
    def forward(self, x): return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class AetherBlock(nn.Module):
    """Core building block of AetherNet."""
    def __init__(self, dim, mlp_ratio, drop, drop_path, lk_kernel, sk_kernel, fused_init, quantize, use_ca, use_sa, norm_layer, res_scale):
        super().__init__()
        self.res_scale = res_scale
        self.conv = ReparamLargeKernelConv(dim, dim, lk_kernel, 1, dim, sk_kernel, fused_init)
        self.norm = norm_layer(dim)
        self.ffn = GatedConvFFN(dim, mlp_ratio, drop)
        self.channel_attn = DynamicChannelScaling(dim) if use_ca else nn.Identity()
        self.spatial_attn = SpatialAttention() if use_sa else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.quantize_residual_flag = quantize
        if quantize: self.residual_quant = tq.QuantStub()
        self.quant_add = FloatFunctional()
    def forward(self, x):
        shortcut = self.residual_quant(x) if self.quantize_residual_flag else x
        residual = self.drop_path(self.spatial_attn(self.channel_attn(self.ffn(self.norm(self.conv(x))))))
        return self.quant_add.add(shortcut, residual * self.res_scale)

class QuantFusion(nn.Module):
    """Multi-scale feature fusion with quantization support."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.error_comp = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 1e-5)
        self.quant_add = FloatFunctional()
    def forward(self, features):
        target_size = features[0].shape[-2:]
        aligned = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) if f.shape[-2:] != target_size else f for f in features]
        return self.quant_add.add(self.conv(torch.cat(aligned, dim=1)), self.error_comp.to(features[0].dtype))

class AdaptiveUpsample(nn.Module):
    """Resolution-aware upsampling module using PixelShuffle."""
    def __init__(self, scale, in_channels):
        super().__init__()
        self.upsample = nn.Sequential()
        if scale == 1: self.upsample.add_module('identity', nn.Identity())
        elif (scale & (scale - 1)) == 0:
            num_ups, current_channels = int(math.log2(scale)), in_channels
            for i in range(num_ups):
                next_channels = in_channels if (i == num_ups - 1) else current_channels // 2
                self.upsample.add_module(f'up_{i}', nn.Sequential(nn.Conv2d(current_channels, 4 * next_channels, 3, 1, 1), nn.PixelShuffle(2)))
                current_channels = next_channels
        elif scale == 3: self.upsample.add_module('up_3x', nn.Sequential(nn.Conv2d(in_channels, 9 * in_channels, 3, 1, 1), nn.PixelShuffle(3)))
        else: raise ValueError(f"Unsupported scale factor: {scale}")
    def forward(self, x): return self.upsample(x)


class aether(nn.Module):
    """
    AetherNet: Production-Ready Super-Resolution Network.
    Designed for a balance of performance, speed, and production-readiness,
    featuring Quantization-Aware Training (QAT) support.
    """
    def __init__(self, in_chans=3, embed_dim=128, depths=[6,6,6,6], mlp_ratio=2., drop=0., drop_path_rate=0.,
                 lk_kernel=13, sk_kernel=3, scale=4, img_range=1., fused_init=False, quantize_residual=True,
                 use_channel_attn=True, use_spatial_attn=False, norm_type='deployment', res_scale=0.1, **kwargs):
        super().__init__()
        self.scale, self.img_range, self.is_quantized = scale, img_range, False
        self.mean = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), requires_grad=False)
        self.quant, self.dequant = tq.QuantStub(), tq.DeQuantStub()
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        if norm_type == 'deployment': norm_layer = DeploymentNorm
        elif norm_type == 'layernorm': norm_layer = LayerNorm2d
        else: raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.num_stages = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        base_ch, rem = divmod(embed_dim, self.num_stages)
        fusion_out = [base_ch + 1 if i < rem else base_ch for i in range(self.num_stages)]

        self.stages, self.fusion_convs = nn.ModuleList(), nn.ModuleList()
        block_idx = 0
        for i in range(self.num_stages):
            self.stages.append(nn.Sequential(*[AetherBlock(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=dpr[block_idx+j], lk_kernel=lk_kernel, sk_kernel=sk_kernel, fused_init=fused_init, quantize=quantize_residual, use_ca=use_channel_attn, use_sa=use_spatial_attn, norm_layer=norm_layer, res_scale=res_scale) for j in range(depths[i])]))
            block_idx += depths[i]
            self.fusion_convs.append(nn.Conv2d(embed_dim, fusion_out[i], 1))

        self.quant_fusion_layer = QuantFusion(embed_dim, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsample = AdaptiveUpsample(scale, embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, 3, 3, 1, 1)
        self.add1, self.add2 = FloatFunctional(), FloatFunctional()
        if not fused_init: self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, ReparamLargeKernelConv)) and hasattr(m, 'weight') and m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x_norm = (x - self.mean.to(x.dtype)) * self.img_range
        x_quant = self.quant(x_norm) if self.is_quantized else x_norm
        x_first = self.conv_first(x_quant)
        features, out = [], x_first
        for stage, fusion_conv in zip(self.stages, self.fusion_convs):
            out = stage(out)
            features.append(fusion_conv(out))
        fused = self.quant_fusion_layer(features)
        body = self.add1.add(fused, x_first)
        body = self.add2.add(self.conv_after_body(self.norm(body)), body)
        recon = self.conv_last(self.upsample(self.conv_before_upsample(body)))
        output = self.dequant(recon) if self.is_quantized else recon
        return output / self.img_range + self.mean

    def fuse_model(self): self.eval(); [m.fuse() for m in self.modules() if hasattr(m, 'fuse')]

    def prepare_qat(self):
        self.fuse_model(); self.train()
        torch.backends.quantized.engine = 'qnnpack'
        try:
            qat_qconfig = tq.get_default_qat_qconfig('qnnpack')
            weight_obs = MovingAverageMinMaxObserver.with_args(quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
            self.qconfig = tq.QConfig(activation=qat_qconfig.activation, weight=weight_obs)
        except AttributeError: self.qconfig = tq.get_default_qat_qconfig('qnnpack')
        for name, module in self.named_modules():
            if any(ln in name for ln in ['conv_first', 'conv_last', 'conv_before_upsample']): module.qconfig = None
        tq.prepare_qat(self, inplace=True); return self

    def convert_to_quantized(self):
        model_copy = deepcopy(self); model_copy.eval(); model_copy.fuse_model()
        torch.backends.quantized.engine = 'qnnpack'
        quantized_model = tq.convert(model_copy, inplace=False)
        quantized_model.eval(); quantized_model.is_quantized = True
        return quantized_model

# --- Network Presets ---
def aether_tiny(scale: int, **kwargs): return aether(embed_dim=64, depths=[3, 3, 3], res_scale=0.2, scale=scale, **kwargs)
def aether_small(scale: int, **kwargs): return aether(embed_dim=96, depths=[6, 6, 6, 6], res_scale=0.1, scale=scale, **kwargs)
def aether_medium(scale: int, **kwargs): return aether(embed_dim=128, depths=[6, 6, 6, 6], use_channel_attn=True, res_scale=0.1, scale=scale, **kwargs)
def aether_large(scale: int, **kwargs): return aether(embed_dim=180, depths=[8, 8, 8, 8, 8], use_channel_attn=True, use_spatial_attn=True, res_scale=0.1, scale=scale, **kwargs)

# --- ONNX Export Utility ---
def export_onnx(model, scale, precision="fp32", output_path=None):
    model.eval()
    if not hasattr(model, 'fused_conv'): model.fuse_model()
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    try: device = next(model.parameters()).device
    except StopIteration: device = torch.device('cpu')
    model.to(device); dummy_input = dummy_input.to(device)
    if precision == "fp16": model, dummy_input = model.half(), dummy_input.half()
    elif precision == "int8" and not (hasattr(model, 'is_quantized') and model.is_quantized):
        raise ValueError("For INT8 export, the model must be converted.")
    onnx_filename = output_path if output_path else f"aether_net_x{scale}_{precision}.onnx"
    print(f"Exporting model to {onnx_filename} with opset 18...")
    training_mode = torch.onnx.TrainingMode.EVAL
    torch.onnx.export(model, dummy_input, onnx_filename, input_names=["input"], output_names=["output"],
        export_params=True, opset_version=18, do_constant_folding=True,
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}}, training=training_mode)
    print(f"Successfully exported to {onnx_filename}")