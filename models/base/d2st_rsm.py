#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
Residual Segment Matcher (RSM) on top of the fixed D2ST-compatible backbone.

The backbone stays unchanged and provides frame-level CLS sequences.
RSM only adds a conservative segment-level residual correction on top of the
stable global Bi-MHM score.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.base_blocks import HEAD_REGISTRY
from models.base.d2st_compat_vit import ViT_D2ST_Compat


def bi_mhm_dist(dists):
    return dists.min(dim=3)[0].sum(dim=2) + dists.min(dim=2)[0].sum(dim=2)


@HEAD_REGISTRY.register()
class CNN_FS_D2ST_RSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.backbone = ViT_D2ST_Compat(cfg)
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        self.feature_dim = int(self.backbone.width)

        windows = getattr(cfg.RSM, "WINDOWS", [[0, 4], [2, 6], [4, 8]])
        self.windows = [(int(w[0]), int(w[1])) for w in windows]
        self.pooling = str(getattr(cfg.RSM, "POOLING", "attention")).lower()
        if self.pooling not in {"attention", "ordered_mean"}:
            raise ValueError("RSM.POOLING must be 'attention' or 'ordered_mean'.")
        self.use_query_condition = bool(getattr(cfg.RSM, "USE_QUERY_CONDITION", False))
        self.residual_alpha = float(getattr(cfg.RSM, "RESIDUAL_ALPHA", 0.1))
        self.residual_beta = float(getattr(cfg.RSM, "RESIDUAL_BETA", 0.2))
        self.residual_clip = float(getattr(cfg.RSM, "RESIDUAL_CLIP", 0.2))
        if self.residual_alpha < 0 or self.residual_beta < 0 or self.residual_clip <= 0:
            raise ValueError("RSM residual hyperparameters must be non-negative with positive clip.")

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    @staticmethod
    def _pairwise_sequence_distance(query_seq, support_seq):
        query_expand = query_seq.unsqueeze(1).expand(-1, support_seq.shape[0], -1, -1)
        support_expand = support_seq.unsqueeze(0).expand(query_seq.shape[0], -1, -1, -1)
        sim = torch.matmul(
            F.normalize(query_expand, dim=-1),
            F.normalize(support_expand, dim=-1).transpose(-1, -2),
        )
        return 1.0 - sim

    def _build_class_sequence_prototypes(self, support_features, support_labels):
        unique_labels = torch.unique(support_labels)
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
        return unique_labels, torch.stack(support_prototypes, dim=0)

    def _extract_segments(self, features):
        segments = []
        for start, end in self.windows:
            if start < 0 or end > features.shape[1] or end <= start:
                raise ValueError(
                    f"Invalid segment window [{start}, {end}) for sequence length {features.shape[1]}"
                )
            segments.append(features[:, start:end, :])
        return torch.stack(segments, dim=1)

    def _pool_segment_tokens(self, tokens, query_vec=None):
        if self.pooling == "ordered_mean":
            return tokens.mean(dim=0)

        if self.use_query_condition and query_vec is not None:
            cond = F.normalize(query_vec, dim=0)
        else:
            cond = F.normalize(tokens.mean(dim=0), dim=0)

        scores = torch.matmul(tokens, cond.unsqueeze(-1)).squeeze(-1) / math.sqrt(tokens.shape[-1])
        weights = F.softmax(scores, dim=0)
        return torch.sum(weights.unsqueeze(-1) * tokens, dim=0)

    def _segment_tokens(self, features):
        segments = self._extract_segments(features)
        pooled = []
        for seg_idx in range(segments.shape[1]):
            seg_tokens = []
            for sample_idx in range(segments.shape[0]):
                seg_tokens.append(self._pool_segment_tokens(segments[sample_idx, seg_idx]))
            pooled.append(torch.stack(seg_tokens, dim=0))
        return torch.stack(pooled, dim=1)

    def _build_class_segment_prototypes(self, support_features, support_labels):
        support_segment_tokens = self._segment_tokens(support_features)
        unique_labels = torch.unique(support_labels)
        class_segment_prototypes = []
        for cls in unique_labels:
            class_support = torch.index_select(
                support_segment_tokens,
                0,
                self._extract_class_indices(support_labels, cls),
            )
            class_segment_prototypes.append(class_support.mean(dim=0))
        return unique_labels, torch.stack(class_segment_prototypes, dim=0)

    def _segment_relation_score(self, support_features, query_features, support_labels):
        unique_labels, class_segment_prototypes = self._build_class_segment_prototypes(
            support_features,
            support_labels,
        )
        query_segments = self._extract_segments(query_features)

        num_query = query_segments.shape[0]
        num_classes = class_segment_prototypes.shape[0]
        num_windows = query_segments.shape[1]
        query_segment_tokens = []
        for q_idx in range(num_query):
            per_query = []
            for seg_idx in range(num_windows):
                proto_hint = class_segment_prototypes[:, seg_idx].mean(dim=0) if self.use_query_condition else None
                per_query.append(self._pool_segment_tokens(query_segments[q_idx, seg_idx], query_vec=proto_hint))
            query_segment_tokens.append(torch.stack(per_query, dim=0))
        query_segment_tokens = torch.stack(query_segment_tokens, dim=0)

        segment_score = query_segment_tokens.new_zeros((num_query, num_classes))
        for seg_idx in range(num_windows):
            q_seg = F.normalize(query_segment_tokens[:, seg_idx], dim=-1)
            s_seg = F.normalize(class_segment_prototypes[:, seg_idx], dim=-1)
            segment_score += torch.matmul(q_seg, s_seg.transpose(0, 1))
        return segment_score / num_windows

    def _build_residual(self, segment_score, global_score):
        delta = segment_score - global_score.detach()
        delta = delta - delta.mean(dim=1, keepdim=True)
        delta_std = delta.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        delta = delta / delta_std
        delta = self.residual_beta * delta
        return torch.clamp(delta, min=-self.residual_clip, max=self.residual_clip)

    def forward(self, inputs):
        support_images = inputs["support_set"]
        query_images = inputs["target_set"]
        support_labels = inputs["support_labels"]
        if torch.is_tensor(support_labels) and support_labels.dim() > 1:
            support_labels = support_labels[0]

        support_features, query_features = self.backbone.get_episode_features(
            support_images,
            query_images,
        )
        _, support_prototypes = self._build_class_sequence_prototypes(support_features, support_labels)
        class_logits = self.backbone.get_classification_logits(support_features, query_features)

        global_dist = bi_mhm_dist(self._pairwise_sequence_distance(query_features, support_prototypes))
        global_score = -global_dist

        segment_score = self._segment_relation_score(support_features, query_features, support_labels)
        segment_residual = self._build_residual(segment_score, global_score)
        final_logits = global_score + self.residual_alpha * segment_residual

        return {
            "logits": final_logits,
            "class_logits": class_logits,
            "global_score": global_score,
            "segment_score": segment_score,
            "segment_residual": segment_residual,
        }
