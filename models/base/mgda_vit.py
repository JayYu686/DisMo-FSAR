#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
MGDA few-shot mainline.

This module provides a benchmark-clean CLIP ViT-B/16 few-shot head with:
1. Frozen CLIP image backbone.
2. Motion-Gated Dual-Path Adapter (MGDA) inserted into later ViT blocks.
3. Lightweight OTAM matching head.

The design is intentionally simpler than D2ST-style deformable attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.base.base_blocks import HEAD_REGISTRY


def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
    return torch.reshape(class_mask_indices, (-1,))


def otam_cum_dist(dists, lbda=0.1):
    """Soft OTAM cumulative distance in one direction."""
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


class MotionGatedDualPathAdapter(nn.Module):
    """Lightweight dual-path adapter with explicit frame-difference gating."""

    def __init__(
        self,
        embed_dim=768,
        adapter_dim=192,
        spatial_kernel_size=3,
        temporal_kernel_size=3,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim
        self.spatial_kernel_size = spatial_kernel_size
        self.temporal_kernel_size = temporal_kernel_size

        self.down_proj = nn.Linear(embed_dim, adapter_dim)

        self.spatial_dw = nn.Conv2d(
            adapter_dim,
            adapter_dim,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            groups=adapter_dim,
            bias=False,
        )
        self.spatial_pw = nn.Conv2d(adapter_dim, adapter_dim, kernel_size=1, bias=False)
        self.spatial_norm = nn.GroupNorm(1, adapter_dim)

        self.diff_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.temporal_dw = nn.Conv1d(
            adapter_dim,
            adapter_dim,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
            groups=adapter_dim,
            bias=False,
        )
        self.temporal_pw = nn.Conv1d(adapter_dim, adapter_dim, kernel_size=1, bias=False)
        self.temporal_norm = nn.GroupNorm(1, adapter_dim)

        self.gate = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim),
            nn.Sigmoid(),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(adapter_dim, embed_dim)
        self.cls_up_proj = nn.Linear(adapter_dim, embed_dim)

        # Start from near-identity behavior for stable adapter tuning.
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.cls_up_proj.weight)
        nn.init.zeros_(self.cls_up_proj.bias)

    def forward(self, x):
        """x: [B, T, N+1, C]"""
        cls_token = x[:, :, :1, :]
        patch_tokens = x[:, :, 1:, :]
        b, t, n, _ = patch_tokens.shape
        hw = int(math.sqrt(n))
        if hw * hw != n:
            raise ValueError(f"MGDA expects square patch tokens, got N={n}")

        patch_hidden = self.down_proj(patch_tokens)

        spatial = rearrange(patch_hidden, "b t (h w) c -> (b t) c h w", h=hw, w=hw)
        spatial = self.spatial_dw(spatial)
        spatial = self.spatial_norm(spatial)
        spatial = self.activation(spatial)
        spatial = self.spatial_pw(spatial)
        spatial = rearrange(spatial, "(b t) c h w -> b t (h w) c", b=b, t=t)

        diff = torch.zeros_like(patch_hidden)
        if t > 1:
            diff[:, 1:] = patch_hidden[:, 1:] - patch_hidden[:, :-1]
        diff = diff + self.diff_proj(diff)

        temporal = patch_hidden + diff
        temporal = rearrange(temporal, "b t n c -> (b n) c t")
        temporal = self.temporal_dw(temporal)
        temporal = self.temporal_norm(temporal)
        temporal = self.activation(temporal)
        temporal = self.temporal_pw(temporal)
        temporal = rearrange(temporal, "(b n) c t -> b t n c", b=b, n=n)

        gate_input = diff.abs().mean(dim=2).mean(dim=1)
        gate = self.gate(gate_input).view(b, 1, 1, self.adapter_dim)

        fused = gate * temporal + (1.0 - gate) * spatial
        fused = self.dropout(fused)

        patch_out = patch_tokens + self.up_proj(fused)
        cls_context = fused.mean(dim=2, keepdim=True)
        cls_out = cls_token + self.cls_up_proj(cls_context)
        return torch.cat([cls_out, patch_out], dim=2)


class CLIPVisionMGDABackbone(nn.Module):
    """Video wrapper around CLIP ViT-B/16 with MGDA adapters."""

    def __init__(self, cfg):
        super().__init__()
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "The 'clip' package is required for CNN_FS_MGDA_ViT. "
                "Please install it in the active environment."
            ) from exc

        self.args = cfg
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        self.freeze_clip = bool(getattr(cfg.VIDEO.BACKBONE, "FREEZE", True))
        self.use_mgda = bool(getattr(cfg.MGDA, "ENABLE", True))
        self.use_temporal_embed = bool(getattr(cfg.MGDA, "USE_TEMPORAL_EMBED", self.use_mgda))
        insert_blocks = getattr(cfg.MGDA, "INSERT_BLOCKS", [6, 7, 8, 9, 10, 11])
        self.insert_blocks = sorted({int(i) for i in insert_blocks})

        clip_model_name = "ViT-B/16"
        full_model, _ = clip.load(clip_model_name, device="cpu", jit=False)
        self.visual = full_model.visual.float()
        self.width = self.visual.conv1.out_channels
        self.output_dim = self.visual.proj.shape[1] if self.visual.proj is not None else self.width
        self.grid_size = self.visual.input_resolution // self.visual.conv1.kernel_size[0]

        if self.freeze_clip:
            for param in self.visual.parameters():
                param.requires_grad = False

        if self.use_temporal_embed:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames, 1, self.width))
            nn.init.normal_(self.temporal_embedding, std=0.02)
        else:
            self.register_parameter("temporal_embedding", None)

        adapter_dim = int(getattr(cfg.MGDA, "ADAPTER_DIM", 192))
        spatial_kernel = int(getattr(cfg.MGDA, "SPATIAL_KERNEL_SIZE", 3))
        temporal_kernel = int(getattr(cfg.MGDA, "TEMPORAL_KERNEL_SIZE", 3))
        adapter_dropout = float(getattr(cfg.MGDA, "DROPOUT", 0.0))

        self.adapters = nn.ModuleDict()
        if self.use_mgda:
            for block_idx in self.insert_blocks:
                self.adapters[str(block_idx)] = MotionGatedDualPathAdapter(
                    embed_dim=self.width,
                    adapter_dim=adapter_dim,
                    spatial_kernel_size=spatial_kernel,
                    temporal_kernel_size=temporal_kernel,
                    dropout=adapter_dropout,
                )

    def _reshape_video(self, x):
        if isinstance(x, dict):
            x = x["video"]
        if x.dim() == 4:
            b_times_t, c, h, w = x.shape
            if b_times_t % self.num_frames != 0:
                raise ValueError(
                    f"Input has {b_times_t} frames, not divisible by NUM_INPUT_FRAMES={self.num_frames}"
                )
            b = b_times_t // self.num_frames
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=self.num_frames)
        elif x.dim() == 5 and x.shape[1] != 3:
            x = rearrange(x, "b t c h w -> b c t h w")
        elif x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")
        return x.float()

    def forward(self, x):
        x = self._reshape_video(x)
        b, c, t, h, w = x.shape
        frames = rearrange(x, "b c t h w -> (b t) c h w")

        tokens = self.visual.conv1(frames)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
        cls = self.visual.class_embedding.to(tokens.dtype) + torch.zeros(
            tokens.shape[0], 1, tokens.shape[-1], dtype=tokens.dtype, device=tokens.device
        )
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.visual.positional_embedding.to(tokens.dtype)
        tokens = self.visual.ln_pre(tokens)
        tokens = rearrange(tokens, "(b t) n c -> b t n c", b=b, t=t)

        if self.use_temporal_embed and self.temporal_embedding is not None:
            tokens = tokens + self.temporal_embedding[:, :t].to(tokens.dtype)

        for idx, block in enumerate(self.visual.transformer.resblocks):
            tokens = rearrange(tokens, "b t n c -> n (b t) c")
            tokens = block(tokens)
            tokens = rearrange(tokens, "n (b t) c -> b t n c", b=b, t=t)
            adapter_key = str(idx)
            if adapter_key in self.adapters:
                tokens = self.adapters[adapter_key](tokens)

        cls_tokens = self.visual.ln_post(tokens[:, :, 0, :])
        if self.visual.proj is not None:
            cls_tokens = cls_tokens @ self.visual.proj

        cls_tokens = rearrange(cls_tokens, "b t d -> b d t")
        return cls_tokens.unsqueeze(-1).unsqueeze(-1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_clip:
            self.visual.eval()
        return self


@HEAD_REGISTRY.register()
class CNN_FS_MGDA_ViT(nn.Module):
    """Benchmark-clean few-shot head with CLIP ViT-B/16 + MGDA + OTAM."""

    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        self.backbone = CLIPVisionMGDABackbone(cfg)
        self.mid_dim = self.backbone.output_dim
        self.encode_chunk_size = max(1, int(getattr(cfg.MGDA, "ENCODE_CHUNK_SIZE", 4)))
        self.backbone_requires_grad = (
            (not self.backbone.freeze_clip)
            or self.backbone.use_mgda
            or self.backbone.use_temporal_embed
        )
        # Keep a minimal trainable parameter for the frozen-backbone B0 baseline.
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.distance_type = str(getattr(cfg.TRAIN, "DISTANCE_TYPE", "otam")).lower()
        if self.distance_type not in {"otam", "hausdorff"}:
            raise ValueError(
                f"CNN_FS_MGDA_ViT supports DISTANCE_TYPE in {{'otam', 'hausdorff'}}, got {self.distance_type}"
            )

    def _prepare_video_tensor(self, video_tensor):
        if torch.is_tensor(video_tensor) and video_tensor.dim() > 5:
            video_tensor = video_tensor[0]
        return video_tensor

    def _encode_video_batch(self, video_tensor):
        """Encode an episodic video batch in chunks to control CLIP ViT memory usage."""
        def _encode_chunk(chunk):
            if self.backbone_requires_grad:
                return self.backbone(chunk)
            with torch.no_grad():
                return self.backbone(chunk)

        if video_tensor.dim() == 4:
            frames_per_chunk = self.encode_chunk_size * self.num_frames
            if video_tensor.shape[0] <= frames_per_chunk:
                return _encode_chunk(video_tensor)
            encoded = []
            for start in range(0, video_tensor.shape[0], frames_per_chunk):
                end = start + frames_per_chunk
                encoded.append(_encode_chunk(video_tensor[start:end]))
            return torch.cat(encoded, dim=0)

        if video_tensor.shape[0] <= self.encode_chunk_size:
            return _encode_chunk(video_tensor)

        encoded = []
        for start in range(0, video_tensor.shape[0], self.encode_chunk_size):
            end = start + self.encode_chunk_size
            encoded.append(_encode_chunk(video_tensor[start:end]))
        return torch.cat(encoded, dim=0)

    def get_feats(self, support_images, target_images):
        support_images = self._prepare_video_tensor(support_images)
        target_images = self._prepare_video_tensor(target_images)

        support_features = self._encode_video_batch(support_images).squeeze(-1).squeeze(-1)
        target_features = self._encode_video_batch(target_images).squeeze(-1).squeeze(-1)

        support_features = rearrange(support_features, "b d t -> b t d")
        target_features = rearrange(target_features, "b d t -> b t d")
        return support_features, target_features

    def _compute_pairwise_distances(self, support_features, target_features):
        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]
        support_expand = support_features.unsqueeze(0).expand(n_queries, -1, -1, -1)
        target_expand = target_features.unsqueeze(1).expand(-1, n_support, -1, -1)

        frame_sim = torch.matmul(
            F.normalize(target_expand, dim=-1),
            F.normalize(support_expand, dim=-1).transpose(-1, -2),
        )
        frame_dists = 1.0 - frame_sim

        if self.distance_type == "otam":
            return otam_cum_dist(frame_dists) + otam_cum_dist(
                rearrange(frame_dists, "q s tq ts -> q s ts tq")
            )

        return frame_dists.min(dim=3)[0].sum(dim=2) + frame_dists.min(dim=2)[0].sum(dim=2)

    def forward(self, inputs):
        support_images = inputs["support_set"]
        support_labels = inputs["support_labels"]
        target_images = inputs["target_set"]

        if torch.is_tensor(support_labels) and support_labels.dim() > 1:
            support_labels = support_labels[0]

        support_features, target_features = self.get_feats(support_images, target_images)
        unique_labels = torch.unique(support_labels)

        pairwise_dists = self._compute_pairwise_distances(support_features, target_features)
        class_dists = []
        for cls in unique_labels:
            cls_mask = extract_class_indices(support_labels, cls)
            class_dists.append(torch.index_select(pairwise_dists, 1, cls_mask).mean(dim=1))
        class_dists = torch.stack(class_dists, dim=1)
        return {"logits": -class_dists * self.logit_scale.exp()}

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
