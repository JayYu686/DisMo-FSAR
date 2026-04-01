#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
Motion Module for DiSMo: Few-Shot Action Recognition.

Enhanced motion modeling with:
1. Multi-scale frame difference encoding (stride 1, 2, 4) for temporal dynamics
2. Deeper motion encoder with temporal attention
3. Motion autodecoder for reconstruction auxiliary loss
4. Long-short contrastive learning for global-local consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiScaleFrameDiff(nn.Module):
    """
    Compute frame differences at multiple temporal strides.

    For stride *s* the difference is ``x[:, :, s:] - x[:, :, :-s]``,
    zero-padded at the beginning to keep the time dimension unchanged.
    The multi-scale differences are fused via a learnable 1×1 projection.

    Args:
        dim: Feature dimension.
        strides: Tuple of temporal strides (default ``(1, 2, 4)``).
    """

    def __init__(self, dim, strides=(1, 2, 4)):
        super().__init__()
        self.strides = strides
        # 1x1 conv to fuse concatenated multi-scale diffs back to dim
        self.fusion = nn.Sequential(
            nn.Conv1d(dim * len(strides), dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: [B, D, T] temporal features
        Returns:
            fused_diff: [B, D, T] fused multi-scale frame differences
        """
        diffs = []
        for s in self.strides:
            T = x.shape[2]
            if s >= T:
                # stride larger than sequence – fall back to zeros
                diffs.append(torch.zeros_like(x))
                continue
            diff = x[:, :, s:] - x[:, :, :-s]          # [B, D, T-s]
            diff = F.pad(diff, (s, 0), mode='constant', value=0)  # [B, D, T]
            diffs.append(diff)
        # Concatenate along channel dim and fuse
        multi = torch.cat(diffs, dim=1)  # [B, D*S, T]
        return self.fusion(multi)         # [B, D, T]


class MotionEncoder(nn.Module):
    """
    Encodes frame differences into motion features.

    Deeper architecture: 3 conv layers with residual shortcut,
    hidden_dim defaults to ``dim // 2`` for higher capacity.
    """

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),
        )
        # Residual shortcut (identity since in/out dim match)
        self.residual = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, D, T] frame difference features
        Returns:
            motion_feat: [B, D, T] encoded motion features
        """
        return self.encoder(x) + self.residual(x)


class TemporalAttentionBlock(nn.Module):
    """
    Lightweight temporal self-attention applied along the time axis.

    Lets the motion branch capture non-local temporal dependencies
    that pure convolution misses.
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, D, T] temporal features
        Returns:
            out: [B, D, T]
        """
        # Transpose to [B, T, D] for attention
        xt = rearrange(x, 'b d t -> b t d')
        # Self-attention with residual
        normed = self.norm(xt)
        att_out, _ = self.attn(normed, normed, normed)
        xt = xt + att_out
        # FFN with residual
        xt = xt + self.ffn(self.norm2(xt))
        return rearrange(xt, 'b t d -> b d t')


class MotionDecoder(nn.Module):
    """
    Decodes motion features back to frame differences for reconstruction loss.
    """

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim // 2

        self.decoder = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, D, T] motion features
        Returns:
            recon: [B, D, T] reconstructed frame differences
        """
        return self.decoder(x)


class LongShortContrastive(nn.Module):
    """
    Long-short contrastive learning module.

    Maximizes agreement between local frame features and
    global video representation for temporal coherence.
    """

    def __init__(self, dim):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.local_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        """
        Args:
            x: [B, D, T] temporal features
        Returns:
            loss: scalar contrastive loss
        """
        B, D, T = x.shape

        # Global token: [B, D]
        global_token = self.global_pool(x).squeeze(-1)
        global_token = self.global_proj(global_token)
        global_token = F.normalize(global_token, dim=-1)

        # Local tokens: [B, T, D]
        local_tokens = rearrange(x, 'b d t -> b t d')
        local_tokens = self.local_proj(local_tokens)
        local_tokens = F.normalize(local_tokens, dim=-1)

        # Cosine similarity: [B, T]
        global_expanded = global_token.unsqueeze(1)
        similarity = (local_tokens * global_expanded).sum(dim=-1)

        # Loss: maximize similarity
        loss = (1 - similarity.mean())

        return loss


class MotionModule(nn.Module):
    """
    Enhanced motion modeling module for DiSMo.

    Key improvements over v1:
    - Multi-scale frame differences (stride 1/2/4) to capture motion at
      different temporal granularities.
    - Deeper MotionEncoder with residual shortcut and hidden_dim = dim//2.
    - Optional lightweight temporal self-attention in the motion branch.
    - Larger initial residual_scale (0.5) so motion features have meaningful
      influence from the start of training.

    Args:
        dim: Feature dimension.
        num_frames: Number of frames (for informational purposes).
        use_autodecoder: Whether to use motion reconstruction loss.
        use_long_short: Whether to use long-short contrastive.
        hidden_dim: Hidden dimension for encoder/decoder (default: dim//2).
        use_temporal_attn: Whether to add temporal self-attention in the
            motion branch (default: True).
        diff_strides: Multi-scale difference strides (default: (1, 2, 4)).
    """

    def __init__(
        self,
        dim,
        num_frames=8,
        use_autodecoder=True,
        use_long_short=True,
        hidden_dim=None,
        use_temporal_attn=True,
        diff_strides=(1, 2, 4),
    ):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.use_autodecoder = use_autodecoder
        self.use_long_short = use_long_short
        self.use_temporal_attn = use_temporal_attn

        # Multi-scale frame difference
        self.multi_scale_diff = MultiScaleFrameDiff(dim, strides=diff_strides)

        # Motion encoder (deeper, wider)
        self.motion_encoder = MotionEncoder(dim, hidden_dim)

        # Optional temporal self-attention
        if use_temporal_attn:
            self.temporal_attn = TemporalAttentionBlock(dim, num_heads=4)

        # Motion autodecoder (optional)
        if use_autodecoder:
            self.motion_decoder = MotionDecoder(dim, hidden_dim)

        # Long-short contrastive (optional)
        if use_long_short:
            self.long_short = LongShortContrastive(dim)

        # Residual scale for motion feature addition
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)

    def compute_frame_diff(self, x):
        """
        Compute temporal frame differences (stride-1 only, used as
        reconstruction target for the autodecoder).

        Args:
            x: [B, D, T] temporal features
        Returns:
            x_diff: [B, D, T] frame differences (zero-padded at start)
        """
        x_diff = x[:, :, 1:] - x[:, :, :-1]  # [B, D, T-1]
        x_diff = F.pad(x_diff, (1, 0), mode='constant', value=0)  # [B, D, T]
        return x_diff

    def forward(self, x, compute_loss=True):
        """
        Forward pass with optional auxiliary losses.

        Args:
            x: [B, D, T] temporal features
            compute_loss: Whether to compute auxiliary losses (False for inference)

        Returns:
            enhanced_feat: [B, D, T] motion-enhanced features
            losses: dict of auxiliary losses (empty if compute_loss=False)
        """
        B, D, T = x.shape
        losses = {}

        # 1. Multi-scale frame differences
        ms_diff = self.multi_scale_diff(x)   # [B, D, T]

        # 2. Encode motion features (with residual inside encoder)
        motion_feat = self.motion_encoder(ms_diff)

        # 3. Optional temporal self-attention over motion features
        if self.use_temporal_attn:
            motion_feat = self.temporal_attn(motion_feat)

        # 4. Autodecoder reconstruction loss (against stride-1 diff)
        if self.use_autodecoder and compute_loss and self.training:
            x_diff_s1 = self.compute_frame_diff(x)
            motion_recon = self.motion_decoder(motion_feat)
            recon_loss = F.mse_loss(motion_recon, x_diff_s1)
            losses['motion_recon'] = recon_loss

        # 5. Long-short contrastive loss
        if self.use_long_short and compute_loss and self.training:
            ls_loss = self.long_short(x)
            losses['long_short_contrast'] = ls_loss

        # 6. Add motion features to appearance features
        enhanced_feat = x + self.residual_scale * motion_feat

        return enhanced_feat, losses

    def get_motion_only(self, x):
        """
        Get pure motion features without residual connection.
        Useful for visualization or separate motion branch.

        Args:
            x: [B, D, T] temporal features
        Returns:
            motion_feat: [B, D, T] motion features
        """
        ms_diff = self.multi_scale_diff(x)
        motion_feat = self.motion_encoder(ms_diff)
        if self.use_temporal_attn:
            motion_feat = self.temporal_attn(motion_feat)
        return motion_feat


class GatedFusion(nn.Module):
    """
    Gated fusion of static and motion features.

    Uses a learned gate to control the contribution of each modality.
    """

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, static_feat, motion_feat):
        """
        Args:
            static_feat: [B, D, T] static appearance features
            motion_feat: [B, D, T] motion features
        Returns:
            fused: [B, D, T] gated fusion result
        """
        B, D, T = static_feat.shape

        static_t = rearrange(static_feat, 'b d t -> (b t) d')
        motion_t = rearrange(motion_feat, 'b d t -> (b t) d')

        combined = torch.cat([static_t, motion_t], dim=-1)  # [B*T, 2D]
        gate = self.gate(combined)  # [B*T, D]
        gate = rearrange(gate, '(b t) d -> b d t', b=B, t=T)

        fused = static_feat + gate * motion_feat
        return fused
