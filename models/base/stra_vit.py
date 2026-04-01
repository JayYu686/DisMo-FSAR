#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
STRA few-shot mainline.

This module provides a benchmark-clean CLIP ViT-B/16 few-shot head with:
1. Frozen CLIP image backbone.
2. State-Transition Relation Adapter (STRA) inserted into deeper ViT blocks.
3. Bi-MHM as the default few-shot matcher over cls/anchor token sequences.

The design targets fine-grained local state transitions and support-query relation cues.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from models.base.base_blocks import HEAD_REGISTRY


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


def bi_mhm_dist(dists):
    if dists.dim() == 4:
        return dists.min(dim=3)[0].sum(dim=2) + dists.min(dim=2)[0].sum(dim=2)
    if dists.dim() == 5:
        forward = dists.min(dim=4)[0].sum(dim=3)
        backward = dists.min(dim=3)[0].sum(dim=3)
        return forward + backward
    raise ValueError(f"Bi-MHM expects 4D or 5D distances, got shape {tuple(dists.shape)}")


class StateTransitionRelationAdapter(nn.Module):
    """State-transition relation adapter with anchor-centric token updates."""

    def __init__(
        self,
        embed_dim=768,
        adapter_dim=192,
        topk_patches=12,
        stem_kernel_size=3,
        dropout=0.0,
        anchor_heads=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim
        self.topk_patches = int(topk_patches)

        self.down_proj = nn.Linear(embed_dim, adapter_dim)
        self.cls_down_proj = nn.Linear(embed_dim, adapter_dim)

        self.state_stem_dw = nn.Conv3d(
            adapter_dim,
            adapter_dim,
            kernel_size=stem_kernel_size,
            padding=stem_kernel_size // 2,
            groups=adapter_dim,
            bias=False,
        )
        self.state_stem_pw = nn.Conv3d(adapter_dim, adapter_dim, kernel_size=1, bias=False)
        self.state_norm = nn.LayerNorm(adapter_dim)

        self.spatial_anchor_norm = nn.LayerNorm(adapter_dim)
        self.spatial_anchor_attn = nn.MultiheadAttention(adapter_dim, anchor_heads, batch_first=True)

        self.temporal_anchor_norm = nn.LayerNorm(adapter_dim)
        self.temporal_anchor_attn = nn.MultiheadAttention(adapter_dim, anchor_heads, batch_first=True)

        self.cls_anchor_norm = nn.LayerNorm(adapter_dim)
        self.cls_anchor_attn = nn.MultiheadAttention(adapter_dim, anchor_heads, batch_first=True)

        self.anchor_summary_proj = nn.Linear(adapter_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, embed_dim)
        self.cls_up_proj = nn.Linear(adapter_dim, embed_dim)

        # Pure zero-init makes early STRA blocks effectively invisible to the loss
        # because the residual branch is multiplied away at initialization.
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.normal_(self.cls_up_proj.weight, std=1e-3)
        nn.init.zeros_(self.cls_up_proj.bias)

    @staticmethod
    def _normalize_scores(scores):
        mean = scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (scores - mean) / std

    def _state_stem(self, patch_hidden, hw):
        state = rearrange(patch_hidden, "b t (h w) c -> b c t h w", h=hw, w=hw)
        state = self.state_stem_dw(state)
        state = self.activation(state)
        state = self.state_stem_pw(state)
        state = rearrange(state, "b c t h w -> b t (h w) c")
        return self.state_norm(patch_hidden + state)

    def _select_anchors(self, state_hidden, delta):
        motion_saliency = delta.abs().mean(dim=-1)
        appearance_variance = state_hidden.var(dim=-1, unbiased=False)
        saliency = self._normalize_scores(motion_saliency) + self._normalize_scores(appearance_variance)

        b, t, n = saliency.shape
        k = min(self.topk_patches, n)
        anchor_idx = torch.topk(saliency, k=k, dim=-1).indices
        gather_index = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, state_hidden.shape[-1])
        anchors = torch.gather(state_hidden, 2, gather_index)
        return anchors, gather_index

    def _spatial_anchor_relation(self, anchors):
        b, t, k, c = anchors.shape
        anchor_seq = rearrange(anchors, "b t k c -> (b t) k c")
        normed = self.spatial_anchor_norm(anchor_seq)
        updated, _ = self.spatial_anchor_attn(normed, normed, normed, need_weights=False)
        anchor_seq = anchor_seq + updated
        return rearrange(anchor_seq, "(b t) k c -> b t k c", b=b, t=t)

    def _temporal_anchor_transition(self, anchors):
        b, t, k, c = anchors.shape
        anchor_seq = rearrange(anchors, "b t k c -> (b k) t c")
        normed = self.temporal_anchor_norm(anchor_seq)
        updated, _ = self.temporal_anchor_attn(normed, normed, normed, need_weights=False)
        anchor_seq = anchor_seq + updated
        return rearrange(anchor_seq, "(b k) t c -> b t k c", b=b, k=k)

    def _cls_anchor_update(self, cls_token, anchors):
        b, t, _, _ = cls_token.shape
        cls_hidden = self.cls_down_proj(cls_token.squeeze(2))
        cls_query = rearrange(cls_hidden, "b t c -> (b t) 1 c")
        anchor_seq = rearrange(anchors, "b t k c -> (b t) k c")
        q = self.cls_anchor_norm(cls_query)
        kv = self.cls_anchor_norm(anchor_seq)
        updated, _ = self.cls_anchor_attn(q, kv, kv, need_weights=False)
        cls_query = cls_query + updated
        return rearrange(cls_query, "(b t) 1 c -> b t 1 c", b=b, t=t)

    def forward(self, x):
        """x: [B, T, N+1, C]"""
        cls_token = x[:, :, :1, :]
        patch_tokens = x[:, :, 1:, :]
        b, t, n, _ = patch_tokens.shape
        hw = int(math.sqrt(n))
        if hw * hw != n:
            raise ValueError(f"STRA expects square patch tokens, got N={n}")

        patch_hidden = self.down_proj(patch_tokens)
        state_hidden = self._state_stem(patch_hidden, hw)

        delta = torch.zeros_like(state_hidden)
        if t > 1:
            delta[:, 1:] = state_hidden[:, 1:] - state_hidden[:, :-1]

        anchors, gather_index = self._select_anchors(state_hidden, delta)
        anchors = self._spatial_anchor_relation(anchors)
        anchors = self._temporal_anchor_transition(anchors)
        cls_hidden = self._cls_anchor_update(cls_token, anchors)

        anchor_map = torch.zeros_like(state_hidden)
        anchor_map.scatter_add_(2, gather_index, anchors)

        fused_patches = self.dropout(state_hidden + anchor_map)
        patch_out = patch_tokens + self.up_proj(fused_patches)
        cls_out = cls_token + self.cls_up_proj(self.dropout(cls_hidden))
        anchor_summary = self.anchor_summary_proj(anchors.mean(dim=2))
        return torch.cat([cls_out, patch_out], dim=2), anchor_summary


class CLIPVisionSTRABackbone(nn.Module):
    """Video wrapper around CLIP ViT-B/16 with STRA adapters."""

    def __init__(self, cfg):
        super().__init__()
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "The 'clip' package is required for CNN_FS_STRA_ViT. "
                "Please install it in the active environment."
            ) from exc

        self.args = cfg
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        self.freeze_clip = bool(getattr(cfg.VIDEO.BACKBONE, "FREEZE", True))
        self.use_stra = bool(getattr(cfg.STRA, "ENABLE", True))
        self.use_temporal_embed = bool(getattr(cfg.STRA, "USE_TEMPORAL_EMBED", self.use_stra))
        self.train_layernorms = bool(getattr(cfg.STRA, "TRAIN_LAYERNORMS", True))
        self.use_checkpoint = bool(getattr(cfg.STRA, "USE_CHECKPOINT", True))
        insert_blocks = getattr(cfg.STRA, "INSERT_BLOCKS", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        checkpoint_blocks = getattr(cfg.STRA, "CHECKPOINT_BLOCKS", [2, 3, 4, 5, 6, 7])
        self.insert_blocks = sorted({int(i) for i in insert_blocks})
        self.checkpoint_blocks = {int(i) for i in checkpoint_blocks}
        self.first_trainable_block = min(self.insert_blocks) if (self.use_stra and self.insert_blocks) else None

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

        adapter_dim = int(getattr(cfg.STRA, "ADAPTER_DIM", 192))
        topk_patches = int(getattr(cfg.STRA, "TOPK_PATCHES", 12))
        stem_kernel = int(getattr(cfg.STRA, "STEM_KERNEL_SIZE", 3))
        adapter_dropout = float(getattr(cfg.STRA, "DROPOUT", 0.0))
        anchor_heads = int(getattr(cfg.STRA, "ANCHOR_HEADS", 4))

        self.adapters = nn.ModuleDict()
        if self.use_stra:
            for block_idx in self.insert_blocks:
                self.adapters[str(block_idx)] = StateTransitionRelationAdapter(
                    embed_dim=self.width,
                    adapter_dim=adapter_dim,
                    topk_patches=topk_patches,
                    stem_kernel_size=stem_kernel,
                    dropout=adapter_dropout,
                    anchor_heads=anchor_heads,
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
        anchor_summary = None
        if adapter is not None:
            tokens, anchor_summary = adapter(tokens)
        return tokens, anchor_summary

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
        anchor_summaries = []
        for idx, block in enumerate(blocks):
            adapter = self.adapters[str(idx)] if str(idx) in self.adapters else None
            if self.first_trainable_block is not None and idx < self.first_trainable_block:
                with torch.no_grad():
                    tokens = self._forward_block(tokens, block)
                continue
            if idx == self.first_trainable_block and self.use_temporal_embed and self.temporal_embedding is not None:
                tokens = tokens + self.temporal_embedding[:, :t].to(tokens.dtype)

            if (
                adapter is not None
                and self.use_checkpoint
                and self.training
                and idx in self.checkpoint_blocks
                and tokens.requires_grad
            ):
                tokens, anchor_summary = checkpoint(
                    lambda inp, blk=block, adp=adapter: self._forward_trainable_block(inp, blk, adp),
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens, anchor_summary = self._forward_trainable_block(tokens, block, adapter)

            if anchor_summary is not None:
                anchor_summaries.append(anchor_summary)

        cls_tokens = self.visual.ln_post(tokens[:, :, 0, :])
        if anchor_summaries:
            anchor_tokens = torch.stack(anchor_summaries, dim=0).mean(dim=0)
        else:
            anchor_tokens = tokens[:, :, 1:, :].mean(dim=2)
        anchor_tokens = self.visual.ln_post(anchor_tokens)
        if self.visual.proj is not None:
            cls_tokens = cls_tokens @ self.visual.proj
            anchor_tokens = anchor_tokens @ self.visual.proj

        cls_tokens = rearrange(cls_tokens, "b t d -> b d t")
        anchor_tokens = rearrange(anchor_tokens, "b t d -> b d t")
        return cls_tokens.unsqueeze(-1).unsqueeze(-1), anchor_tokens.unsqueeze(-1).unsqueeze(-1)

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
class CNN_FS_STRA_ViT(nn.Module):
    """CLIP ViT-B/16 + STRA + Bi-MHM few-shot head."""

    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        self.backbone = CLIPVisionSTRABackbone(cfg)
        self.mid_dim = self.backbone.output_dim
        self.encode_chunk_size = max(1, int(getattr(cfg.STRA, "ENCODE_CHUNK_SIZE", 8)))
        self.backbone_requires_grad = any(param.requires_grad for param in self.backbone.parameters())
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.distance_type = str(getattr(cfg.TRAIN, "DISTANCE_TYPE", "bi_mhm")).lower()
        if self.distance_type not in {"bi_mhm", "otam"}:
            raise ValueError(
                f"CNN_FS_STRA_ViT supports DISTANCE_TYPE in {{'bi_mhm', 'otam'}}, got {self.distance_type}"
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
            cls_parts, anchor_parts = [], []
            for start in range(0, video_tensor.shape[0], frames_per_chunk):
                cls_chunk, anchor_chunk = _encode_chunk(video_tensor[start:start + frames_per_chunk])
                cls_parts.append(cls_chunk)
                anchor_parts.append(anchor_chunk)
            return torch.cat(cls_parts, dim=0), torch.cat(anchor_parts, dim=0)

        if video_tensor.shape[0] <= self.encode_chunk_size:
            return _encode_chunk(video_tensor)

        cls_parts, anchor_parts = [], []
        for start in range(0, video_tensor.shape[0], self.encode_chunk_size):
            cls_chunk, anchor_chunk = _encode_chunk(video_tensor[start:start + self.encode_chunk_size])
            cls_parts.append(cls_chunk)
            anchor_parts.append(anchor_chunk)
        return torch.cat(cls_parts, dim=0), torch.cat(anchor_parts, dim=0)

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

        support_cls, support_anchor = self._encode_video_batch(support_images)
        target_cls, target_anchor = self._encode_video_batch(target_images)
        support_cls = support_cls.squeeze(-1).squeeze(-1)
        support_anchor = support_anchor.squeeze(-1).squeeze(-1)
        target_cls = target_cls.squeeze(-1).squeeze(-1)
        target_anchor = target_anchor.squeeze(-1).squeeze(-1)

        support_cls = rearrange(support_cls, "(m s) d t -> m s t d", m=num_tasks, s=num_support)
        support_anchor = rearrange(support_anchor, "(m s) d t -> m s t d", m=num_tasks, s=num_support)
        target_cls = rearrange(target_cls, "(m q) d t -> m q t d", m=num_tasks, q=num_queries)
        target_anchor = rearrange(target_anchor, "(m q) d t -> m q t d", m=num_tasks, q=num_queries)
        return support_cls, support_anchor, target_cls, target_anchor, support_labels

    @staticmethod
    def _build_dual_token_sequence(cls_tokens, anchor_tokens):
        seq = torch.stack([cls_tokens, anchor_tokens], dim=3)
        return rearrange(seq, "m n t two d -> m n (t two) d")

    def _compute_pairwise_distances(self, support_seq, target_seq):
        num_tasks, n_queries = target_seq.shape[:2]
        support_expand = support_seq.unsqueeze(1)
        target_expand = target_seq.unsqueeze(2)
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

        return bi_mhm_dist(frame_dists)

    def forward(self, inputs):
        support_cls, support_anchor, target_cls, target_anchor, support_labels = self.get_feats(
            inputs["support_set"],
            inputs["target_set"],
            inputs["support_labels"],
        )

        support_seq = self._build_dual_token_sequence(support_cls, support_anchor)
        target_seq = self._build_dual_token_sequence(target_cls, target_anchor)
        pairwise_dists = self._compute_pairwise_distances(support_seq, target_seq)

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
