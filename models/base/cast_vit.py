#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
CAST few-shot mainline.

This module provides a benchmark-clean CLIP ViT-B/16 few-shot head with:
1. Frozen CLIP image backbone.
2. Contact-Aware State-Transition Adapter (CAST) inserted into later ViT blocks.
3. Lightweight OTAM matching head.

The design targets fine-grained local state changes in few-shot action recognition.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from models.base.base_blocks import HEAD_REGISTRY


def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
    return torch.reshape(class_mask_indices, (-1,))


def otam_cum_dist(dists, lbda=0.1):
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


class ContactAwareStateTransitionAdapter(nn.Module):
    """Local-state, transition, and relation-anchor adapter."""

    def __init__(
        self,
        embed_dim=768,
        adapter_dim=192,
        topk_patches=8,
        spatial_kernel_size=3,
        temporal_kernel_size=3,
        dropout=0.0,
        anchor_heads=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim
        self.topk_patches = int(topk_patches)

        self.down_proj = nn.Linear(embed_dim, adapter_dim)

        self.local_dw = nn.Conv2d(
            adapter_dim,
            adapter_dim,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            groups=adapter_dim,
            bias=False,
        )
        self.local_pw = nn.Conv2d(adapter_dim, adapter_dim, kernel_size=1, bias=False)
        self.local_norm = nn.LayerNorm(adapter_dim)

        self.transition_dw = nn.Conv1d(
            adapter_dim,
            adapter_dim,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
            groups=adapter_dim,
            bias=False,
        )
        self.transition_pw = nn.Conv1d(adapter_dim, adapter_dim, kernel_size=1, bias=False)
        self.transition_mlp = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim * 2),
            nn.GELU(),
            nn.Linear(adapter_dim * 2, adapter_dim),
        )
        self.transition_norm = nn.LayerNorm(adapter_dim)

        self.anchor_attn = nn.MultiheadAttention(adapter_dim, anchor_heads, batch_first=True)
        self.anchor_proj = nn.Linear(adapter_dim, adapter_dim)
        self.anchor_norm = nn.LayerNorm(adapter_dim)

        self.state_gate = nn.Sequential(
            nn.Linear(adapter_dim * 2, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim * 3),
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(adapter_dim, embed_dim)
        self.cls_up_proj = nn.Linear(adapter_dim, embed_dim)

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.cls_up_proj.weight)
        nn.init.zeros_(self.cls_up_proj.bias)

    def _local_path(self, patch_hidden, hw):
        b, t, n, c = patch_hidden.shape
        local = rearrange(patch_hidden, "b t (h w) c -> (b t) c h w", h=hw, w=hw)
        local = self.local_dw(local)
        local = self.activation(local)
        local = self.local_pw(local)
        local = rearrange(local, "(b t) c h w -> b t (h w) c", b=b, t=t)
        return self.local_norm(local)

    def _transition_path(self, delta):
        b, t, n, c = delta.shape
        transition = rearrange(delta, "b t n c -> (b n) c t")
        transition = self.transition_dw(transition)
        transition = self.activation(transition)
        transition = self.transition_pw(transition)
        transition = rearrange(transition, "(b n) c t -> b t n c", b=b, n=n)
        transition = self.transition_mlp(transition)
        return self.transition_norm(transition)

    def _relation_anchor_path(self, patch_hidden, delta):
        b, t, n, c = patch_hidden.shape
        k = min(self.topk_patches, n)
        saliency = delta.abs().mean(dim=-1)
        anchor_idx = torch.topk(saliency, k=k, dim=-1).indices
        gather_index = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, c)

        anchor_tokens = torch.gather(patch_hidden + delta, 2, gather_index)
        anchor_tokens = rearrange(anchor_tokens, "b t k c -> (b k) t c")
        anchor_tokens, _ = self.anchor_attn(anchor_tokens, anchor_tokens, anchor_tokens, need_weights=False)
        anchor_tokens = self.anchor_proj(anchor_tokens)
        anchor_tokens = rearrange(anchor_tokens, "(b k) t c -> b t k c", b=b, k=k)

        anchor_map = torch.zeros_like(patch_hidden)
        anchor_map.scatter_add_(2, gather_index, anchor_tokens)
        return self.anchor_norm(anchor_map), anchor_tokens

    def forward(self, x):
        """x: [B, T, N+1, C]"""
        cls_token = x[:, :, :1, :]
        patch_tokens = x[:, :, 1:, :]
        b, t, n, _ = patch_tokens.shape
        hw = int(math.sqrt(n))
        if hw * hw != n:
            raise ValueError(f"CAST expects square patch tokens, got N={n}")

        patch_hidden = self.down_proj(patch_tokens)
        delta = torch.zeros_like(patch_hidden)
        if t > 1:
            delta[:, 1:] = patch_hidden[:, 1:] - patch_hidden[:, :-1]

        local = self._local_path(patch_hidden, hw)
        transition = self._transition_path(delta)
        anchor_map, anchor_tokens = self._relation_anchor_path(patch_hidden, delta)

        delta_pooled = delta.abs().mean(dim=(1, 2))
        anchor_pooled = anchor_tokens.mean(dim=(1, 2))
        gate_logits = self.state_gate(torch.cat([delta_pooled, anchor_pooled], dim=-1))
        gate_logits = gate_logits.view(b, 3, self.adapter_dim)
        gate_weights = torch.softmax(gate_logits, dim=1).view(b, 3, 1, 1, self.adapter_dim)

        fused = (
            gate_weights[:, 0] * local
            + gate_weights[:, 1] * transition
            + gate_weights[:, 2] * anchor_map
        )
        fused = self.dropout(fused)

        patch_out = patch_tokens + self.up_proj(fused)
        cls_context = fused.mean(dim=2, keepdim=True)
        cls_out = cls_token + self.cls_up_proj(cls_context)
        return torch.cat([cls_out, patch_out], dim=2)


class CLIPVisionCASTBackbone(nn.Module):
    """Video wrapper around CLIP ViT-B/16 with CAST adapters."""

    def __init__(self, cfg):
        super().__init__()
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "The 'clip' package is required for CNN_FS_CAST_ViT. "
                "Please install it in the active environment."
            ) from exc

        self.args = cfg
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        self.freeze_clip = bool(getattr(cfg.VIDEO.BACKBONE, "FREEZE", True))
        self.use_cast = bool(getattr(cfg.CAST, "ENABLE", True))
        self.use_temporal_embed = bool(getattr(cfg.CAST, "USE_TEMPORAL_EMBED", self.use_cast))
        self.use_checkpoint = bool(getattr(cfg.CAST, "USE_CHECKPOINT", True))
        self.train_layernorms = bool(getattr(cfg.CAST, "TRAIN_LAYERNORMS", True))
        insert_blocks = getattr(cfg.CAST, "INSERT_BLOCKS", [4, 5, 6, 7, 8, 9, 10, 11])
        self.insert_blocks = sorted({int(i) for i in insert_blocks})
        self.first_trainable_block = min(self.insert_blocks) if (self.use_cast and self.insert_blocks) else None

        full_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        self.visual = full_model.visual.float()
        self.width = self.visual.conv1.out_channels
        self.output_dim = self.visual.proj.shape[1] if self.visual.proj is not None else self.width

        if self.freeze_clip:
            for param in self.visual.parameters():
                param.requires_grad = False

        if self.use_temporal_embed:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames, 1, self.width))
            nn.init.normal_(self.temporal_embedding, std=0.02)
        else:
            self.register_parameter("temporal_embedding", None)

        adapter_dim = int(getattr(cfg.CAST, "ADAPTER_DIM", 256))
        topk_patches = int(getattr(cfg.CAST, "TOPK_PATCHES", 16))
        spatial_kernel = int(getattr(cfg.CAST, "SPATIAL_KERNEL_SIZE", 3))
        temporal_kernel = int(getattr(cfg.CAST, "TEMPORAL_KERNEL_SIZE", 3))
        adapter_dropout = float(getattr(cfg.CAST, "DROPOUT", 0.0))

        self.adapters = nn.ModuleDict()
        if self.use_cast:
            for block_idx in self.insert_blocks:
                self.adapters[str(block_idx)] = ContactAwareStateTransitionAdapter(
                    embed_dim=self.width,
                    adapter_dim=adapter_dim,
                    topk_patches=topk_patches,
                    spatial_kernel_size=spatial_kernel,
                    temporal_kernel_size=temporal_kernel,
                    dropout=adapter_dropout,
                )

        if self.freeze_clip and self.train_layernorms and self.insert_blocks:
            for block_idx in self.insert_blocks:
                block = self.visual.transformer.resblocks[block_idx]
                for param in block.ln_1.parameters():
                    param.requires_grad = True
                for param in block.ln_2.parameters():
                    param.requires_grad = True

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

    def _forward_block(self, tokens, block):
        b, t = tokens.shape[:2]
        tokens = rearrange(tokens, "b t n c -> n (b t) c")
        tokens = block(tokens)
        return rearrange(tokens, "n (b t) c -> b t n c", b=b, t=t)

    def _forward_trainable_block(self, tokens, block, adapter: Optional[nn.Module]):
        tokens = self._forward_block(tokens, block)
        if adapter is not None:
            tokens = adapter(tokens)
        return tokens

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

        blocks = self.visual.transformer.resblocks
        for idx, block in enumerate(blocks):
            adapter = self.adapters[str(idx)] if str(idx) in self.adapters else None
            if self.first_trainable_block is not None and idx < self.first_trainable_block:
                with torch.no_grad():
                    tokens = self._forward_block(tokens, block)
                continue
            if idx == self.first_trainable_block and self.use_temporal_embed and self.temporal_embedding is not None:
                tokens = tokens + self.temporal_embedding[:, :t].to(tokens.dtype)

            if self.use_checkpoint and self.training and tokens.requires_grad and (adapter is not None or idx in self.insert_blocks):
                tokens = checkpoint(
                    lambda inp, blk=block, adp=adapter: self._forward_trainable_block(inp, blk, adp),
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = self._forward_trainable_block(tokens, block, adapter)

        cls_tokens = self.visual.ln_post(tokens[:, :, 0, :])
        if self.visual.proj is not None:
            cls_tokens = cls_tokens @ self.visual.proj

        cls_tokens = rearrange(cls_tokens, "b t d -> b d t")
        return cls_tokens.unsqueeze(-1).unsqueeze(-1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_clip:
            self.visual.eval()
            if self.train_layernorms:
                for block_idx in self.insert_blocks:
                    block = self.visual.transformer.resblocks[block_idx]
                    block.ln_1.train(mode)
                    block.ln_2.train(mode)
        return self


@HEAD_REGISTRY.register()
class CNN_FS_CAST_ViT(nn.Module):
    """CLIP ViT-B/16 + CAST + OTAM few-shot head."""

    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        self.backbone = CLIPVisionCASTBackbone(cfg)
        self.mid_dim = self.backbone.output_dim
        self.encode_chunk_size = max(1, int(getattr(cfg.CAST, "ENCODE_CHUNK_SIZE", 8)))
        self.backbone_requires_grad = any(param.requires_grad for param in self.backbone.parameters())
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.distance_type = str(getattr(cfg.TRAIN, "DISTANCE_TYPE", "otam")).lower()
        if self.distance_type not in {"otam", "hausdorff"}:
            raise ValueError(
                f"CNN_FS_CAST_ViT supports DISTANCE_TYPE in {{'otam', 'hausdorff'}}, got {self.distance_type}"
            )

    def _prepare_support_labels(self, support_labels):
        if not torch.is_tensor(support_labels):
            raise TypeError("support_labels must be a tensor.")
        if support_labels.dim() == 1:
            support_labels = support_labels.unsqueeze(0)
        return support_labels.long()

    def _reshape_episodic_videos(self, video_tensor, expected_items=None):
        if not torch.is_tensor(video_tensor):
            raise TypeError("Expected episodic video tensor.")

        if video_tensor.dim() == 4:
            total_frames = video_tensor.shape[0]
            if total_frames % self.num_frames != 0:
                raise ValueError(
                    f"Input has {total_frames} frames, not divisible by NUM_INPUT_FRAMES={self.num_frames}"
                )
            num_tasks = 1
            num_items = total_frames // self.num_frames
            video_tensor = rearrange(video_tensor, "(n t) c h w -> n c t h w", t=self.num_frames)
        elif video_tensor.dim() == 5:
            num_tasks, total_frames, c, h, w = video_tensor.shape
            if total_frames % self.num_frames != 0:
                raise ValueError(
                    f"Input has {total_frames} frames per task, not divisible by NUM_INPUT_FRAMES={self.num_frames}"
                )
            num_items = total_frames // self.num_frames
            video_tensor = rearrange(video_tensor, "m (n t) c h w -> (m n) c t h w", t=self.num_frames)
        else:
            raise ValueError(f"Expected episodic 4D/5D tensor, got shape {tuple(video_tensor.shape)}")

        if expected_items is not None and num_items != int(expected_items):
            raise ValueError(
                f"Episodic tensor implies {num_items} videos per task, but expected {expected_items}"
            )
        return video_tensor, num_tasks, num_items

    def _encode_video_batch(self, video_tensor):
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
                encoded.append(_encode_chunk(video_tensor[start:start + frames_per_chunk]))
            return torch.cat(encoded, dim=0)

        if video_tensor.shape[0] <= self.encode_chunk_size:
            return _encode_chunk(video_tensor)

        encoded = []
        for start in range(0, video_tensor.shape[0], self.encode_chunk_size):
            encoded.append(_encode_chunk(video_tensor[start:start + self.encode_chunk_size]))
        return torch.cat(encoded, dim=0)

    def get_feats(self, support_images, target_images, support_labels):
        support_labels = self._prepare_support_labels(support_labels)
        support_images, num_tasks, num_support = self._reshape_episodic_videos(
            support_images, expected_items=support_labels.shape[1]
        )
        target_images, target_num_tasks, num_queries = self._reshape_episodic_videos(target_images)
        if target_num_tasks != num_tasks:
            raise ValueError(
                f"Support/query task batch mismatch: support has {num_tasks}, query has {target_num_tasks}"
            )

        support_features = self._encode_video_batch(support_images).squeeze(-1).squeeze(-1)
        target_features = self._encode_video_batch(target_images).squeeze(-1).squeeze(-1)

        support_features = rearrange(support_features, "(m s) d t -> m s t d", m=num_tasks, s=num_support)
        target_features = rearrange(target_features, "(m q) d t -> m q t d", m=num_tasks, q=num_queries)
        return support_features, target_features, support_labels

    def _compute_pairwise_distances(self, support_features, target_features):
        num_tasks, n_support = support_features.shape[:2]
        n_queries = target_features.shape[1]
        support_expand = support_features.unsqueeze(1)
        target_expand = target_features.unsqueeze(2)

        frame_sim = torch.matmul(
            F.normalize(target_expand, dim=-1),
            F.normalize(support_expand, dim=-1).transpose(-1, -2),
        )
        frame_dists = 1.0 - frame_sim

        if self.distance_type == "otam":
            flat_dists = rearrange(frame_dists, "m q s tq ts -> (m q) s tq ts")
            flat_out = otam_cum_dist(flat_dists) + otam_cum_dist(
                rearrange(flat_dists, "mq s tq ts -> mq s ts tq")
            )
            return rearrange(flat_out, "(m q) s -> m q s", m=num_tasks, q=n_queries)

        return frame_dists.min(dim=3)[0].sum(dim=2) + frame_dists.min(dim=2)[0].sum(dim=2)

    def forward(self, inputs):
        support_images = inputs["support_set"]
        support_labels = inputs["support_labels"]
        target_images = inputs["target_set"]

        support_features, target_features, support_labels = self.get_feats(
            support_images, target_images, support_labels
        )

        pairwise_dists = self._compute_pairwise_distances(support_features, target_features)
        way = max(int(getattr(self.args.TRAIN, "WAY", 0)), int(support_labels.max().item()) + 1)
        class_dists = []
        for cls in range(way):
            cls_mask = (support_labels == cls).float()
            cls_count = cls_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            cls_dist = (pairwise_dists * cls_mask.unsqueeze(1)).sum(dim=2) / cls_count
            class_dists.append(cls_dist)
        class_dists = torch.stack(class_dists, dim=-1)
        logits = -class_dists.reshape(-1, way) * self.logit_scale.exp()
        return {"logits": logits}

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].reshape(-1).long())
