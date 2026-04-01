#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
Matcher-first mainline on top of the MGDA backbone.

This head reuses the current B1 CLIP ViT-B/16 + MGDA feature extractor and
replaces the pure OTAM classifier with a segmented relation matcher:
1. global Bi-MHM over full frame sequences
2. support-to-query segment relation
3. query-to-support segment relation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.base_blocks import HEAD_REGISTRY
from models.base.mgda_vit import CLIPVisionMGDABackbone, extract_class_indices


def bi_mhm_dist(dists):
    return dists.min(dim=3)[0].sum(dim=2) + dists.min(dim=2)[0].sum(dim=2)


@HEAD_REGISTRY.register()
class CNN_FS_MGDA_SEGREL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.backbone = CLIPVisionMGDABackbone(cfg)
        self.mid_dim = self.backbone.output_dim
        self.encode_chunk_size = max(1, int(getattr(cfg.MGDA, "ENCODE_CHUNK_SIZE", 4)))
        self.backbone_requires_grad = (
            (not self.backbone.freeze_clip)
            or self.backbone.use_mgda
            or self.backbone.use_temporal_embed
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        windows = getattr(cfg.SEGREL, "WINDOWS", [[0, 4], [2, 6], [4, 8]])
        self.windows = [(int(w[0]), int(w[1])) for w in windows]
        self.fusion_mode = str(getattr(cfg.SEGREL, "FUSION", "learnable_softmax")).lower()
        if self.fusion_mode != "learnable_softmax":
            raise ValueError(
                f"CNN_FS_MGDA_SEGREL only supports SEGREL.FUSION='learnable_softmax', got {self.fusion_mode}"
            )
        self.fusion_logits = nn.Parameter(torch.zeros(3))

        self.classification_layer = None
        if bool(getattr(cfg.TRAIN, "USE_CLASSIFICATION", False)):
            self.classification_layer = nn.Linear(self.mid_dim, int(cfg.TRAIN.NUM_CLASS))

    def _prepare_video_tensor(self, video_tensor):
        if torch.is_tensor(video_tensor) and video_tensor.dim() > 5:
            video_tensor = video_tensor[0]
        return video_tensor

    def _encode_video_batch(self, video_tensor):
        def _encode_chunk(chunk):
            if self.backbone_requires_grad:
                return self.backbone(chunk)
            with torch.no_grad():
                return self.backbone(chunk)

        if video_tensor.dim() == 4:
            frames_per_chunk = self.encode_chunk_size * int(self.args.DATA.NUM_INPUT_FRAMES)
            if video_tensor.shape[0] <= frames_per_chunk:
                return _encode_chunk(video_tensor)
            encoded = []
            for start in range(0, video_tensor.shape[0], frames_per_chunk):
                encoded.append(_encode_chunk(video_tensor[start : start + frames_per_chunk]))
            return torch.cat(encoded, dim=0)

        if video_tensor.shape[0] <= self.encode_chunk_size:
            return _encode_chunk(video_tensor)

        encoded = []
        for start in range(0, video_tensor.shape[0], self.encode_chunk_size):
            encoded.append(_encode_chunk(video_tensor[start : start + self.encode_chunk_size]))
        return torch.cat(encoded, dim=0)

    def get_feats(self, support_images, target_images):
        support_images = self._prepare_video_tensor(support_images)
        target_images = self._prepare_video_tensor(target_images)

        support_features = self._encode_video_batch(support_images).squeeze(-1).squeeze(-1)
        target_features = self._encode_video_batch(target_images).squeeze(-1).squeeze(-1)

        support_features = support_features.permute(0, 2, 1).contiguous()
        target_features = target_features.permute(0, 2, 1).contiguous()
        return support_features, target_features

    @staticmethod
    def _pairwise_sequence_distance(query_seq, support_seq):
        query_expand = query_seq.unsqueeze(1).expand(-1, support_seq.shape[0], -1, -1)
        support_expand = support_seq.unsqueeze(0).expand(query_seq.shape[0], -1, -1, -1)
        sim = torch.matmul(
            F.normalize(query_expand, dim=-1),
            F.normalize(support_expand, dim=-1).transpose(-1, -2),
        )
        return 1.0 - sim

    def _segment_tokens(self, features):
        segments = []
        total_frames = features.shape[1]
        for start, end in self.windows:
            if start < 0 or end > total_frames or end <= start:
                raise ValueError(
                    f"Invalid SEGREL window [{start}, {end}) for sequence length {total_frames}"
                )
            segments.append(features[:, start:end, :].mean(dim=1))
        return torch.stack(segments, dim=1)

    def _build_class_prototypes(self, support_features, support_labels):
        unique_labels = torch.unique(support_labels)
        class_prototypes = []
        for cls in unique_labels:
            cls_mask = extract_class_indices(support_labels, cls)
            class_prototypes.append(torch.index_select(support_features, 0, cls_mask).mean(dim=0))
        return unique_labels, torch.stack(class_prototypes, dim=0)

    def forward(self, inputs):
        support_images = inputs["support_set"]
        target_images = inputs["target_set"]
        support_labels = inputs["support_labels"]
        if torch.is_tensor(support_labels) and support_labels.dim() > 1:
            support_labels = support_labels[0]

        support_features, target_features = self.get_feats(support_images, target_images)
        _, support_proto = self._build_class_prototypes(support_features, support_labels)

        class_logits = None
        if self.classification_layer is not None:
            pooled = torch.cat(
                [support_features.mean(dim=1), target_features.mean(dim=1)],
                dim=0,
            )
            class_logits = self.classification_layer(pooled)

        global_dist = bi_mhm_dist(self._pairwise_sequence_distance(target_features, support_proto))

        support_segments = self._segment_tokens(support_features)
        target_segments = self._segment_tokens(target_features)
        _, support_seg_proto = self._build_class_prototypes(support_segments, support_labels)
        seg_dist = self._pairwise_sequence_distance(target_segments, support_seg_proto)

        seg_q2s = seg_dist.min(dim=3)[0].sum(dim=2)
        seg_s2q = seg_dist.min(dim=2)[0].sum(dim=2)

        fusion = F.softmax(self.fusion_logits, dim=0)
        fused_dist = fusion[0] * global_dist + fusion[1] * seg_s2q + fusion[2] * seg_q2s
        fused_dist = fused_dist * self.logit_scale.exp()

        return {
            "logits": -fused_dist,
            "class_logits": class_logits,
            "logits_global": -global_dist,
            "logits_s2q": -seg_s2q,
            "logits_q2s": -seg_q2s,
        }
