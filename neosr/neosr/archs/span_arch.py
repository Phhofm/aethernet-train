from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """Re-write convolution layer for adaptive `padding`."""
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def sequential(*args):
    """Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            msg = "sequential does not support OrderedDict input."
            raise NotImplementedError(msg)
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules = list(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    """Upsample features according to `upscale_factor`."""
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, s=1, bias=True):
        super().__init__()
        self.bias = bias
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )

        if self.training is False:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False
            self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        sk_w = self.sk.weight.data.clone().detach()

        if self.bias:
            b1 = self.conv[0].bias.data.clone().detach()
            b2 = self.conv[1].bias.data.clone().detach()
            b3 = self.conv[2].bias.data.clone().detach()
            b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
            self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
            sk_b = self.sk.bias.data.clone().detach()

        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
        )
        self.weight_concat = self.weight_concat + sk_w
        self.eval_conv.weight.data = self.weight_concat
        if self.bias:
            self.bias_concat = self.bias_concat + sk_b
            self.eval_conv.bias.data = self.bias_concat

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)
        return out


class SPAB(nn.Module):
    def __init__(
        self, in_channels, mid_channels=None, out_channels=None, bias=False, fast=False
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1, bias=bias)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1, bias=bias)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1, bias=bias)
        if not fast:
            self.act1 = torch.nn.SiLU(inplace=True)
        else:
            self.act1 = torch.nn.Mish(inplace=True)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        out3 = self.c3_r(out2_act)
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        return out, out1, sim_att


@ARCH_REGISTRY.register()
class span(nn.Module):
    """Swift Parameter-free Attention Network for Efficient Super-Resolution"""

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='linear')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=48,
        upscale=upscale,
        bias=True,
        norm=False,
        fast=False,
        img_range=1.0,
        rgb_mean=(0.5, 0.5, 0.5),
        **kwargs,  # noqa: ARG002
    ):
        super().__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.fast = fast
        self.upscale = upscale
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.no_norm: torch.Tensor | None
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None

        self.block_1 = SPAB(feature_channels, bias=bias, fast=fast)
        self.block_2 = SPAB(feature_channels, bias=bias, fast=fast)
        self.block_3 = SPAB(feature_channels, bias=bias, fast=fast)
        self.block_4 = SPAB(feature_channels, bias=bias, fast=fast)
        self.block_5 = SPAB(feature_channels, bias=bias, fast=fast)

        if not self.fast:
            self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
            self.block_6 = SPAB(feature_channels, bias=bias, fast=fast)
        else:
            self.conv_9x9 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=feature_channels,
                kernel_size=9,
                padding=4,
                bias=bias,
            )
            self.refine_conv = nn.Conv2d(
                in_channels=feature_channels,
                out_channels=feature_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            )

        self.conv_cat = conv_layer(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.conv_2 = Conv3XC(
            feature_channels, feature_channels, gain1=2, s=1, bias=bias
        )

        self.upsampler = pixelshuffle_block(
            feature_channels, out_channels, upscale_factor=upscale
        )

        if self.fast:
            self.apply(self._init_weights)

    @property
    def is_norm(self):
        return self.no_norm is None

    def forward(self, x):
        if self.is_norm:
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range

        if not self.fast:
            out_feature = self.conv_1(x)
        else:
            # large kernel conv
            out_feature = self.conv_9x9(x)
            # add conv to refine feature maps
            out_feature = self.refine_conv(out_feature)

        out_b1, _, _att1 = self.block_1(out_feature)
        out_b2, _, _att2 = self.block_2(out_b1)
        out_b3, _, _att3 = self.block_3(out_b2)
        out_b4, _, _att4 = self.block_4(out_b3)

        if not self.fast:
            out_b5, _, _att5 = self.block_5(out_b4)
            out_b6, out_b5_2, _att6 = self.block_6(out_b5)
            out_b6 = self.conv_2(out_b6)
            out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        else:
            out_b5, out_b4_2, _att5 = self.block_5(out_b4)
            out_b5 = self.conv_2(out_b5)
            out = self.conv_cat(torch.cat([out_feature, out_b5, out_b1, out_b4_2], 1))

        return self.upsampler(out)


@ARCH_REGISTRY.register()
def span_fast(**kwargs):
    return span(feature_channels=32, bias=False, fast=True, **kwargs)
