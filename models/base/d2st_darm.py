#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
DARM matcher on top of the fixed D2ST-compatible backbone.

The backbone is kept unchanged and only exposes frame-level CLS sequences.
The head performs:
1. global Bi-MHM over full 8-frame sequences
2. decisive segment relation matching over three overlapping windows
3. dataset-aware softmax fusion between the two branches
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
class CNN_FS_D2ST_DARM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.backbone = ViT_D2ST_Compat(cfg)
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        self.feature_dim = int(self.backbone.width)

        windows = getattr(cfg.DARM, "SEG_WINDOWS", [[0, 4], [2, 6], [4, 8]])
        self.windows = [(int(w[0]), int(w[1])) for w in windows]
        self.topk_seg_pairs = int(getattr(cfg.DARM, "TOPK_SEG_PAIRS", 2))
        if self.topk_seg_pairs <= 0:
            raise ValueError("DARM.TOPK_SEG_PAIRS must be positive.")

        self.mode = str(getattr(cfg.DARM, "MODE", "temporal")).lower()
        if self.mode not in {"temporal", "spatial"}:
            raise ValueError("DARM.MODE must be 'temporal' or 'spatial'.")

        self.fusion_mode = str(getattr(cfg.DARM, "FUSION", "learnable_softmax")).lower()
        if self.fusion_mode != "learnable_softmax":
            raise ValueError("DARM.FUSION must be 'learnable_softmax'.")

        init_global = float(getattr(cfg.DARM, "INIT_GLOBAL_WEIGHT", 0.5))
        init_segment = float(getattr(cfg.DARM, "INIT_SEGMENT_WEIGHT", 0.5))
        if init_global <= 0 or init_segment <= 0:
            raise ValueError("DARM initial fusion weights must be positive.")
        init_weights = torch.tensor([init_global, init_segment], dtype=torch.float32)
        self.fusion_logits = nn.Parameter(init_weights.log())

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

    @staticmethod
    def _attention_pool(tokens, query_vec):
        scores = torch.matmul(tokens, query_vec.unsqueeze(-1)).squeeze(-1) / math.sqrt(tokens.shape[-1])
        weights = F.softmax(scores, dim=0)
        return torch.sum(weights.unsqueeze(-1) * tokens, dim=0)

    def _segment_relation_scores(self, support_features, query_features, support_labels):
        support_segments = self._extract_segments(support_features)
        query_segments = self._extract_segments(query_features)
        unique_labels = torch.unique(support_labels)

        num_query = query_segments.shape[0]
        num_classes = unique_labels.shape[0]
        num_windows = query_segments.shape[1]

        score_s2q = query_segments.new_zeros((num_query, num_classes, num_windows, num_windows))
        score_q2s = query_segments.new_zeros((num_query, num_classes, num_windows, num_windows))

        for class_idx, cls in enumerate(unique_labels):
            class_support = torch.index_select(
                support_segments,
                0,
                self._extract_class_indices(support_labels, cls),
            )

            for q_idx in range(num_query):
                for q_seg_idx in range(num_windows):
                    query_tokens = query_segments[q_idx, q_seg_idx]
                    query_cond = F.normalize(query_tokens.mean(dim=0), dim=0)

                    for s_seg_idx in range(num_windows):
                        support_tokens = class_support[:, s_seg_idx].reshape(-1, self.feature_dim)
                        support_cond = F.normalize(support_tokens.mean(dim=0), dim=0)

                        pooled_support = self._attention_pool(support_tokens, query_cond)
                        pooled_query = self._attention_pool(query_tokens, support_cond)

                        score_s2q[q_idx, class_idx, q_seg_idx, s_seg_idx] = F.cosine_similarity(
                            query_cond.unsqueeze(0),
                            pooled_support.unsqueeze(0),
                            dim=1,
                        ).squeeze(0)
                        score_q2s[q_idx, class_idx, q_seg_idx, s_seg_idx] = F.cosine_similarity(
                            pooled_query.unsqueeze(0),
                            support_cond.unsqueeze(0),
                            dim=1,
                        ).squeeze(0)

        flat_s2q = score_s2q.reshape(num_query, num_classes, -1)
        flat_q2s = score_q2s.reshape(num_query, num_classes, -1)
        topk = min(self.topk_seg_pairs, flat_s2q.shape[-1])

        topk_s2q = flat_s2q.topk(k=topk, dim=-1, largest=True).values.mean(dim=-1)
        topk_q2s = flat_q2s.topk(k=topk, dim=-1, largest=True).values.mean(dim=-1)
        segment_score = 0.5 * (topk_s2q + topk_q2s)
        return segment_score

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

        segment_score = self._segment_relation_scores(support_features, query_features, support_labels)

        fusion = F.softmax(self.fusion_logits, dim=0)
        final_logits = fusion[0] * global_score + fusion[1] * segment_score

        return {
            "logits": final_logits,
            "class_logits": class_logits,
            "global_score": global_score,
            "segment_score": segment_score,
        }
