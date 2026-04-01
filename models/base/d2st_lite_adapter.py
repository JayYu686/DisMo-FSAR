#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""Lightweight temporal patch adapters for D2ST-lite experiments."""

import torch
import torch.nn as nn
from einops import rearrange


class TemporalAdapterBlock(nn.Module):
    """Residual bottleneck adapter over temporal patch-token sequences."""

    def __init__(
        self,
        dim: int,
        adapter_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        init_scale: float = 0.5,
    ):
        super().__init__()
        resolved_heads = max(1, int(num_heads))
        while adapter_dim % resolved_heads != 0 and resolved_heads > 1:
            resolved_heads -= 1

        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, adapter_dim)
        self.attn = nn.MultiheadAttention(
            adapter_dim,
            num_heads=resolved_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_dwconv = nn.Conv1d(
            adapter_dim,
            adapter_dim,
            kernel_size=3,
            padding=1,
            groups=adapter_dim,
        )
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1) * float(init_scale))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, T, N, D]
        Returns:
            [B, T, N, D]
        """
        if patch_tokens.dim() != 4:
            raise ValueError(
                "TemporalAdapterBlock expects [B, T, N, D], "
                f"got {tuple(patch_tokens.shape)}"
            )

        residual = patch_tokens
        x = rearrange(patch_tokens, "b t n d -> (b n) t d")
        x = self.down(self.norm(x))

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + self.dropout(attn_out)

        conv_out = self.temporal_dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.dropout(conv_out)

        x = self.up(self.act(x))
        x = rearrange(x, "(b n) t d -> b t n d", b=residual.shape[0], n=residual.shape[2])
        return residual + self.scale * x


class TemporalPatchAdapter(nn.Module):
    """Stack of lightweight temporal patch adapters."""

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        adapter_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        init_scale: float = 0.5,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TemporalAdapterBlock(
                    dim=dim,
                    adapter_dim=adapter_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    init_scale=init_scale,
                )
                for _ in range(max(1, int(depth)))
            ]
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            patch_tokens = block(patch_tokens)
        return patch_tokens
