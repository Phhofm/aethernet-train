import copy
from pathlib import Path
from urllib import request

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50

from neosr.utils.registry import LOSS_REGISTRY


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0, normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu
        self.normalize_before = normalize_before

    def forward(self, src):
        src2 = self.norm1(src)
        q = k = src2
        src2, __ = self.self_attn(q, k, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        return src + self.dropout2(src2)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0, normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu
        self.normalize_before = normalize_before

    def forward(self, tgt, memory):
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)
        tgt2, __ = self.multihead_attn(query=tgt2, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return tgt + self.dropout3(tgt2)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory)
        return output


class resnet_features(nn.Module):
    def __init__(self):
        super().__init__()
        # load pretrained
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = nn.Identity()
        self.fc = nn.Identity()

    def forward(self, x):
        features = []
        # initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features.append(x)  # 64 channels

        x = self.layer1(x)
        features.append(x)  # 256 channels

        x = self.layer2(x)
        features.append(x)  # 512 channels

        x = self.layer3(x)
        features.append(x)  # 1024 channels

        x = self.layer4(x)
        features.append(x)  # 2048 channels

        return features


@LOSS_REGISTRY.register()
class topiq(nn.Module):
    """Inference for TOP-IQ metric, proposed in:
    https://arxiv.org/abs/2308.03060
    Code adapted from:
    https://github.com/chaofengc/IQA-PyTorch
    """

    def _init_linear(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def __init__(
        self,
        num_class=1,
        inter_dim=256,
        num_heads=4,
        num_attn_layers=1,
        dprate=0,
        default_mean=(0.485, 0.456, 0.406),
        default_std=(0.229, 0.224, 0.225),
        loss_weight=1.0,
    ):
        super().__init__()

        self.loss_weight = loss_weight
        self.num_class = num_class
        self.semantic_model = resnet_features()
        self.fix_bn(self.semantic_model)
        feature_dim_list = [64, 256, 512, 1024, 2048]

        self.default_mean = torch.tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.tensor(default_std).view(1, 3, 1, 1)

        # define self-attention and cross scale attention blocks
        ca_layers = sa_layers = num_attn_layers
        self.act_layer = nn.GELU()
        dim_feedforward = min(4 * inter_dim, 2048)

        # gated local pooling and self-attention
        tmp_layer = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
        )
        self.sa_attn_blks = nn.ModuleList()
        self.dim_reduce = nn.ModuleList()
        self.weight_pool = nn.ModuleList()
        for dim in feature_dim_list:
            dim = dim * 3  # noqa: PLW2901
            self.weight_pool.append(
                nn.Sequential(
                    nn.Conv2d(dim // 3, 64, 1, stride=1),
                    self.act_layer,
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    self.act_layer,
                    nn.Conv2d(64, 1, 3, stride=1, padding=1),
                    nn.Sigmoid(),
                )
            )

            self.dim_reduce.append(
                nn.Sequential(nn.Conv2d(dim, inter_dim, 1, 1), self.act_layer)
            )

            self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))

        # cross scale attention
        self.attn_blks = nn.ModuleList()
        tmp_layer = TransformerDecoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
        )
        for _i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))

        # attention pooling and MLP layers
        self.attn_pool = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
        )

        linear_dim = inter_dim
        self.score_linear = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, self.num_class),
        ]

        self.score_linear = nn.Sequential(*self.score_linear)
        self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
        self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))

        # init
        nn.init.trunc_normal_(self.h_emb.data, std=0.02)
        nn.init.trunc_normal_(self.w_emb.data, std=0.02)
        self._init_linear(self.dim_reduce)
        self._init_linear(self.sa_attn_blks)
        self._init_linear(self.attn_blks)
        self._init_linear(self.attn_pool)

        # load topiq_fr weights
        model_path = Path(__file__).parent / "topiq_fr_weights.pth"
        try:
            if not model_path.exists():
                url = "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/cfanet_fr_kadid_res50-2c4cc61d.pth?download=true"
                request.urlretrieve(url, model_path)  # noqa: S310
        except:
            msg = "Could not download TOPIQ weights."
            raise ValueError(msg)

        checkpoint = torch.load(model_path, map_location="cuda", weights_only=True)
        self.load_state_dict(checkpoint["params"], strict=True)
        self.eval()

    def preprocess(self, x):
        return (x - self.default_mean.to(x)) / self.default_std.to(x)

    def fix_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def dist_func(self, x, y, eps=1e-12):
        return torch.sqrt((x - y) ** 2 + eps)

    def forward_cross_attention(self, x, y):
        x = self.preprocess(x)
        y = self.preprocess(y)

        dist_feat_list = self.semantic_model(x)
        ref_feat_list = self.semantic_model(y)
        self.fix_bn(self.semantic_model)
        self.semantic_model.eval()

        start_level = 0
        end_level = len(dist_feat_list)

        __, __, th, tw = dist_feat_list[end_level - 1].shape
        pos_emb = torch.cat(
            (
                self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]),
                self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1),
            ),
            dim=1,
        )

        token_feat_list = []
        for i in reversed(range(start_level, end_level)):
            tmp_dist_feat = dist_feat_list[i]
            # gated local pooling
            tmp_ref_feat = ref_feat_list[i]
            diff = self.dist_func(tmp_dist_feat, tmp_ref_feat)

            tmp_feat = torch.cat([tmp_dist_feat, tmp_ref_feat, diff], dim=1)
            weight = self.weight_pool[i](diff)
            tmp_feat = tmp_feat * weight

            if tmp_feat.shape[2] > th and tmp_feat.shape[3] > tw:
                tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))
            # self attention
            tmp_pos_emb = F.interpolate(
                pos_emb, size=tmp_feat.shape[2:], mode="bicubic", align_corners=False
            )
            tmp_pos_emb = tmp_pos_emb.flatten(2).permute(2, 0, 1)
            tmp_feat = self.dim_reduce[i](tmp_feat)
            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)
            tmp_feat = tmp_feat + tmp_pos_emb
            tmp_feat = self.sa_attn_blks[i](tmp_feat)
            token_feat_list.append(tmp_feat)

        # high level -> low level: coarse to fine
        query = token_feat_list[0]
        query_list = [query]
        for i in range(len(token_feat_list) - 1):
            key_value = token_feat_list[i + 1]
            query = self.attn_blks[i](query, key_value)
            query_list.append(query)
        final_feat = self.attn_pool(query)
        return self.score_linear(final_feat.mean(dim=0))

    def forward(self, x, y):
        return self.forward_cross_attention(x, y).mean() * self.loss_weight
