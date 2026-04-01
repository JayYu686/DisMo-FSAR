#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
Internal D2ST-strength anchor for CLIP ViT-B/16.

This head stays intentionally close to the local D2ST ViT path:
1. CLIP ViT-B/16 visual stem.
2. D2ST-style adapter inserted into every transformer block.
3. CLS-sequence based Bi-MHM / OTAM few-shot matching.

It is used as an internal calibration anchor, not as a claimed method novelty.
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

import utils.logging as logging
from models.base.base_blocks import HEAD_REGISTRY

logger = logging.get_logger(__name__)


def otam_cum_dist(dists, lbda=0.5):
    dists = F.pad(dists, (1, 1), "constant", 0)
    cum_dists = torch.zeros_like(dists)

    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    for l in range(1, dists.shape[2]):
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(-cum_dists[:, :, l - 1, 0] / lbda)
            + torch.exp(-cum_dists[:, :, l - 1, 1] / lbda)
            + torch.exp(-cum_dists[:, :, l, 0] / lbda)
        )
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(-cum_dists[:, :, l - 1, m - 1] / lbda)
                + torch.exp(-cum_dists[:, :, l, m - 1] / lbda)
            )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(-cum_dists[:, :, l - 1, -2] / lbda)
            + torch.exp(-cum_dists[:, :, l - 1, -1] / lbda)
            + torch.exp(-cum_dists[:, :, l, -2] / lbda)
        )
    return cum_dists[:, :, -1, -1]


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, "b c t h w -> b t h w c")
        x = self.norm(x)
        x = rearrange(x, "b t h w c -> b c t h w")
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ViTDeformAttention(nn.Module):
    def __init__(self, cfg, dim, heads, groups, kernel_size, stride, padding):
        super().__init__()
        self.args = cfg
        self.dim = dim
        self.heads = heads
        self.head_channels = dim // heads
        self.scale = self.head_channels ** -0.5
        self.groups = groups
        self.group_channels = self.dim // self.groups
        self.factor = 2.0

        self.conv_offset = nn.Sequential(
            nn.Conv3d(
                in_channels=self.group_channels,
                out_channels=self.group_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=self.group_channels,
            ),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv3d(
                in_channels=self.group_channels,
                out_channels=3,
                kernel_size=(1, 1, 1),
                bias=False,
            ),
        )

        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

    @torch.no_grad()
    def _get_ref_points(self, T, H, W, B, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T - 0.5, T, dtype=dtype, device=device),
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            indexing="ij",
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)
        ref[..., 0].div_(T).mul_(2).sub_(1)
        ref[..., 1].div_(H).mul_(2).sub_(1)
        ref[..., 2].div_(W).mul_(2).sub_(1)
        return ref[None, ...].expand(B * self.groups, -1, -1, -1, -1)

    def forward(self, x):
        n, bt, c = x.shape
        t = self.args.DATA.NUM_INPUT_FRAMES
        b = bt // t
        h = round(math.sqrt(n - 1))
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = rearrange(q[1:, :, :], "(h w) (b t) c -> b c t h w", h=h, t=t)
        q_off = rearrange(
            q_off,
            "b (g c) t h w -> (b g) c t h w",
            g=self.groups,
            c=self.group_channels,
        )
        offset = self.conv_offset(q_off)
        tp, hp, wp = offset.size(2), offset.size(3), offset.size(4)

        offset_range = torch.tensor(
            [min(1.0, self.factor / tp), min(1.0, self.factor / hp), min(1.0, self.factor / wp)],
            device=device,
            dtype=dtype,
        ).reshape(1, 3, 1, 1, 1)
        offset = offset.tanh().mul(offset_range)
        offset = rearrange(offset, "b p t h w -> b t h w p")
        reference = self._get_ref_points(tp, hp, wp, b, dtype, device)
        pos = offset + reference

        x_sampled = rearrange(x[1:, :, :], "(h w) (b t) c -> b c t h w", h=h, t=t)
        x_sampled = rearrange(x_sampled, "b (g c) t h w -> (b g) c t h w", g=self.groups)
        x_sampled = F.grid_sample(
            input=x_sampled,
            grid=pos[..., (2, 1, 0)],
            mode="bilinear",
            align_corners=True,
        )
        x_sampled = rearrange(x_sampled, "(b g) c t h w -> b (g c) t h w", g=self.groups)
        x_sampled = rearrange(x_sampled, "b c t h w -> b (t h w) c")

        q = rearrange(q, "n (b t) c -> b c (t n)", b=b)
        q = rearrange(q, "b (heads c) n -> (b heads) c n", heads=self.heads)

        k = self.proj_k(x_sampled)
        k = rearrange(k, "b n (heads c) -> (b heads) c n", heads=self.heads)
        v = self.proj_v(x_sampled)
        v = rearrange(v, "b n (heads c) -> (b heads) c n", heads=self.heads)

        attn = einsum("b c m, b c n -> b m n", q, k).mul(self.scale)
        attn = F.softmax(attn, dim=-1)

        out = einsum("b m n, b c n -> b c m", attn, v)
        out = rearrange(out, "(b heads) c n -> b (heads c) n", heads=self.heads)
        out = rearrange(out, "b c (t n) -> n (b t) c", t=t)
        return self.proj_out(out)


class ViTD2STAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        width = cfg.ADAPTER.WIDTH
        adapter_scale = float(cfg.ADAPTER.ADAPTER_SCALE)
        self.adapter_channels = int(width * adapter_scale)

        self.down = nn.Linear(width, self.adapter_channels)
        self.gelu1 = nn.GELU()

        self.pos_embed = nn.Conv3d(
            in_channels=self.adapter_channels,
            out_channels=self.adapter_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            groups=self.adapter_channels,
        )
        self.s_ln = nn.LayerNorm(self.adapter_channels)
        self.s_attn = ViTDeformAttention(
            cfg=cfg,
            dim=self.adapter_channels,
            heads=4,
            groups=4,
            kernel_size=(4, 5, 5),
            stride=(4, 3, 3),
            padding=(0, 0, 0),
        )
        self.t_ln = nn.LayerNorm(self.adapter_channels)
        self.t_attn = ViTDeformAttention(
            cfg=cfg,
            dim=self.adapter_channels,
            heads=4,
            groups=4,
            kernel_size=(1, 7, 7),
            stride=(1, 7, 7),
            padding=(0, 0, 0),
        )
        self.gelu = nn.GELU()
        self.up = nn.Linear(self.adapter_channels, width)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        n, bt, _ = x.shape
        h = round(math.sqrt(n - 1))
        residual = x

        x = self.gelu1(self.down(x))
        cls = x[0, :, :].unsqueeze(0)
        x = x[1:, :, :]

        x = rearrange(x, "(h w) (b t) c -> b c t h w", t=self.args.DATA.NUM_INPUT_FRAMES, h=h)
        x = x + self.pos_embed(x)
        x = rearrange(x, "b c t h w -> (h w) (b t) c")
        x = torch.cat([cls, x], dim=0)

        xs = x + self.s_attn(self.s_ln(x))
        xt = x + self.t_attn(self.t_ln(x))
        x = self.gelu((xs + xt) / 2.0)

        x = self.gelu2(self.up(x))
        return x + residual


class ResidualAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = int(cfg.ADAPTER.WIDTH)
        n_head = int(cfg.ADAPTER.HEADS)
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.Adapter = ViTD2STAdapter(cfg)

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return self.Adapter(x)


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(cfg) for _ in range(int(cfg.ADAPTER.LAYERS))]
        )

    def forward(self, x):
        return self.resblocks(x)


@HEAD_REGISTRY.register()
class ViT_D2ST_Compat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "The 'clip' package is required for ViT_D2ST_Compat."
            ) from exc

        self.args = cfg
        self.pretrained = str(getattr(cfg.ADAPTER, "PRETRAINED", "ViT-B/16"))
        self.width = int(cfg.ADAPTER.WIDTH)
        self.patch_size = int(cfg.ADAPTER.PATCH_SIZE)
        self.layers = int(cfg.ADAPTER.LAYERS)
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        self.distance_type = str(getattr(cfg.TRAIN, "DISTANCE_TYPE", "bi_mhm")).lower()
        if self.distance_type not in {"bi_mhm", "otam"}:
            raise ValueError(
                f"ViT_D2ST_Compat supports DISTANCE_TYPE in {{'bi_mhm','otam'}}, got {self.distance_type}"
            )
        self.encode_chunk_size = max(1, int(getattr(cfg.ADAPTER, "ENCODE_CHUNK_SIZE", 8)))

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(self.width))
        num_patches = (cfg.DATA.TRAIN_CROP_SIZE // self.patch_size) ** 2 + 1
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches, self.width))
        self.ln_pre = LayerNorm(self.width)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames, self.width))
        self.transformer = Transformer(cfg)
        self.ln_post = LayerNorm(self.width)

        self.classification_layer = None
        if bool(getattr(cfg.TRAIN, "USE_CLASSIFICATION", False)):
            self.classification_layer = nn.Linear(self.width, int(cfg.TRAIN.NUM_CLASS))

        self._init_weights(clip)
        self._freeze_non_adapter_params()

    def _init_weights(self, clip_module):
        logger.info("Loading CLIP visual weights for D2ST anchor from %s", self.pretrained)
        clip_model, _ = clip_module.load(self.pretrained, device="cpu", jit=False)
        pretrain_dict = clip_model.visual.state_dict()
        del clip_model
        pretrain_dict.pop("proj", None)
        msg = self.load_state_dict(pretrain_dict, strict=False)
        logger.info("Missing keys: %s", msg.missing_keys)
        logger.info("Unexpected keys: %s", msg.unexpected_keys)

        for name, module in self.named_modules():
            if "Adapter" in name:
                for child_name, child_module in module.named_modules():
                    if child_name.endswith("up") and isinstance(child_module, nn.Linear):
                        nn.init.constant_(child_module.weight, 0)
                        nn.init.constant_(child_module.bias, 0)

    def _freeze_non_adapter_params(self):
        for name, param in self.named_parameters():
            trainable = (
                ("class_embedding" in name)
                or ("temporal_embedding" in name)
                or ("Adapter" in name)
                or ("ln_post" in name)
                or ("classification_layer" in name)
            )
            param.requires_grad = trainable

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def _prepare_video_tensor(self, x):
        if isinstance(x, dict):
            x = x["video"]
        if x.dim() == 5 and x.shape[1] != 3:
            x = rearrange(x, "b t c h w -> b c t h w")
        if x.dim() == 5:
            x = rearrange(x, "b c t h w -> (b t) c h w")
        if x.dim() != 4:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")
        if x.shape[0] % self.num_frames != 0:
            raise ValueError(
                f"Input has {x.shape[0]} frames, not divisible by NUM_INPUT_FRAMES={self.num_frames}"
            )
        return x.float()

    def _encode_frames(self, frames):
        x = self.conv1(frames)
        x = rearrange(x, "b c h w -> b (h w) c")
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, "(b t) n c -> (b n) t c", t=self.num_frames)
        x = x + self.temporal_embedding[:, : self.num_frames].to(x.dtype)
        x = rearrange(x, "(b n) t c -> (b t) n c", n=n)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x[:, 0, :]

    def get_feat(self, x):
        x = self._prepare_video_tensor(x)
        frames_per_chunk = self.encode_chunk_size * self.num_frames
        if x.shape[0] <= frames_per_chunk:
            return self._encode_frames(x)
        outputs = []
        for start in range(0, x.shape[0], frames_per_chunk):
            outputs.append(self._encode_frames(x[start : start + frames_per_chunk]))
        return torch.cat(outputs, dim=0)

    def get_sequence_features(self, x):
        return self.get_feat(x).reshape(-1, self.num_frames, self.width)

    def get_episode_features(self, support_images, query_images):
        support_features = self.get_sequence_features(support_images)
        query_features = self.get_sequence_features(query_images)
        return support_features, query_features

    def get_classification_logits(self, support_features, query_features):
        if self.classification_layer is None:
            return None
        pooled = torch.cat(
            [torch.mean(support_features, dim=1), torch.mean(query_features, dim=1)],
            dim=0,
        )
        return self.classification_layer(pooled)

    def forward(self, inputs):
        support_images = inputs["support_set"]
        query_images = inputs["target_set"]
        support_labels = inputs["support_labels"]
        if torch.is_tensor(support_labels) and support_labels.dim() > 1:
            support_labels = support_labels[0]

        support_features, query_features = self.get_episode_features(support_images, query_images)
        unique_labels = torch.unique(support_labels)

        class_logits = self.get_classification_logits(support_features, query_features)

        support_prototypes = [
            torch.mean(
                torch.index_select(
                    support_features,
                    0,
                    self._extract_class_indices(support_labels, c),
                ),
                dim=0,
            )
            for c in unique_labels
        ]
        support_prototypes = torch.stack(support_prototypes, dim=0)

        support_num = support_prototypes.shape[0]
        query_num = query_features.shape[0]

        support_prototypes = support_prototypes.unsqueeze(0).repeat(query_num, 1, 1, 1)
        support_prototypes = rearrange(support_prototypes, "q s t c -> q (s t) c")

        frame_sim = torch.matmul(
            F.normalize(support_prototypes, dim=2),
            F.normalize(query_features, dim=2).permute(0, 2, 1),
        ).reshape(query_num, support_num, self.num_frames, self.num_frames)
        dist = 1.0 - frame_sim

        if self.distance_type == "bi_mhm":
            class_dist = dist.min(3)[0].sum(2) + dist.min(2)[0].sum(2)
        else:
            class_dist = otam_cum_dist(dist) + otam_cum_dist(rearrange(dist, "q s n m -> q s m n"))

        return {"logits": -class_dist, "class_logits": class_logits}
