#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
Semantic Residual Calibration (SRC) on top of the fixed D2ST-compatible backbone.

SRC keeps the visual few-shot path unchanged:
1. frame-level CLS sequence from ViT_D2ST_Compat
2. visual Bi-MHM logits remain the primary decision signal
3. semantic branch only adds a clipped residual calibration based on
   frozen text prototypes and query visual summaries
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.base_blocks import HEAD_REGISTRY
from models.base.d2st_compat_vit import ViT_D2ST_Compat
from models.base.semantic_module import TextEncoder


def bi_mhm_dist(dists: torch.Tensor) -> torch.Tensor:
    return dists.min(dim=3)[0].sum(dim=2) + dists.min(dim=2)[0].sum(dim=2)


def _normalize_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\[\]]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _humanize_name(name: str) -> str:
    text = str(name or "").strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    return " ".join(text.split())


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split()).strip()
    return " ".join(str(value).split()).strip()


def _clean_list(value: Iterable) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_clean_text(value)] if _clean_text(value) else []
    items = []
    for item in value:
        text = _clean_text(item)
        if text:
            items.append(text)
    deduped = []
    seen = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped


def _load_description_db(path: str) -> Dict[str, dict]:
    if not path:
        return {}
    desc_path = Path(path)
    if not desc_path.exists():
        raise FileNotFoundError(f"SEMCAL.TEXT_DESC_PATH not found: {path}")
    with desc_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {_normalize_name(k): v for k, v in raw.items()}


def _build_global_text(class_name: str, entry) -> str:
    if isinstance(entry, str):
        return _clean_text(entry) or f"A video of {_humanize_name(class_name)}."
    if not isinstance(entry, dict):
        return f"A video of {_humanize_name(class_name)}."
    for key in ("global_description", "action_anchor", "description", "label_text"):
        text = _clean_text(entry.get(key))
        if text:
            return text
    return f"A video of {_humanize_name(class_name)}."


def _build_local_text(class_name: str, entry) -> str:
    if isinstance(entry, str):
        return f"Key visual cues for {_humanize_name(class_name)}."
    if not isinstance(entry, dict):
        return f"Key visual cues for {_humanize_name(class_name)}."

    local = _clean_text(entry.get("local_description"))
    if local:
        return local

    objects = _clean_list(entry.get("objects") or entry.get("entity_priors"))
    attributes = _clean_list(entry.get("attribute_cues"))
    state_change = _clean_list(entry.get("state_change") or entry.get("phase_cues"))

    parts = []
    if objects:
        parts.append("Objects: " + ", ".join(objects[:6]))
    if attributes:
        parts.append("Cues: " + ", ".join(attributes[:6]))
    if state_change:
        parts.append("State change: " + "; ".join(state_change[:3]))

    if parts:
        return " ".join(parts)
    return _build_global_text(class_name, entry)


@HEAD_REGISTRY.register()
class CNN_FS_D2ST_SRC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.backbone = ViT_D2ST_Compat(cfg)
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        self.feature_dim = int(self.backbone.width)

        self.enable_semcal = bool(getattr(cfg.SEMCAL, "ENABLE", False))
        self.text_desc_path = str(getattr(cfg.SEMCAL, "TEXT_DESC_PATH", ""))
        self.text_encoder_name = str(getattr(cfg.SEMCAL, "TEXT_ENCODER", "clip_vitb16"))
        self.use_local_desc = bool(getattr(cfg.SEMCAL, "USE_LOCAL_DESC", False))
        self.residual_weight = float(getattr(cfg.SEMCAL, "RESIDUAL_WEIGHT", 0.0))
        self.residual_clip = float(getattr(cfg.SEMCAL, "RESIDUAL_CLIP", 0.1))
        self.gate_enable = bool(getattr(cfg.SEMCAL, "GATE_ENABLE", True))
        self.gate_margin = float(getattr(cfg.SEMCAL, "GATE_MARGIN", 0.15))
        self.mode = str(getattr(cfg.SEMCAL, "MODE", "temporal")).lower()
        if self.mode not in {"temporal", "spatial"}:
            raise ValueError("SEMCAL.MODE must be 'temporal' or 'spatial'.")

        self.train_class_names = list(getattr(cfg.TRAIN, "CLASS_NAME", []))
        eval_names = list(getattr(cfg.TEST, "CLASS_NAME", []))
        if not eval_names and hasattr(cfg.TEST, "CLASS_NAME_VAL"):
            eval_names = list(getattr(cfg.TEST, "CLASS_NAME_VAL", []))
        self.eval_class_names = eval_names

        self.register_buffer("visual_to_text_projection", torch.empty(0, 0), persistent=False)
        self.register_buffer("train_global_text_features", torch.empty(0, 0), persistent=False)
        self.register_buffer("train_local_text_features", torch.empty(0, 0), persistent=False)
        self.register_buffer("eval_global_text_features", torch.empty(0, 0), persistent=False)
        self.register_buffer("eval_local_text_features", torch.empty(0, 0), persistent=False)

        if self.enable_semcal:
            self._precompute_text_features()

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

    def _precompute_text_features(self):
        desc_db = _load_description_db(self.text_desc_path)
        encoder = TextEncoder(
            model_name=self.text_encoder_name,
            device="cpu",
            allow_random_fallback=False,
        )
        if "clip" not in self.text_encoder_name.lower():
            raise ValueError("SRC currently requires SEMCAL.TEXT_ENCODER to use a CLIP text encoder.")
        if not hasattr(encoder.encoder, "visual") or getattr(encoder.encoder.visual, "proj", None) is None:
            raise RuntimeError("Failed to access CLIP visual.proj for visual-text alignment in SRC.")
        self.visual_to_text_projection = encoder.encoder.visual.proj.detach().float().cpu()

        def build_bank(class_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            if len(class_names) == 0:
                empty = torch.empty(0, self.visual_to_text_projection.shape[1], dtype=torch.float32)
                return empty, empty

            global_texts = []
            local_texts = []
            for class_name in class_names:
                entry = desc_db.get(_normalize_name(class_name), class_name)
                global_texts.append(_build_global_text(class_name, entry))
                local_texts.append(_build_local_text(class_name, entry))

            global_features = encoder.encode(global_texts).cpu()
            local_features = encoder.encode(local_texts).cpu()
            return global_features, local_features

        train_global, train_local = build_bank(self.train_class_names)
        eval_global, eval_local = build_bank(self.eval_class_names)

        self.train_global_text_features = train_global
        self.train_local_text_features = train_local
        self.eval_global_text_features = eval_global
        self.eval_local_text_features = eval_local

        del encoder

    def _select_text_bank(self):
        if self.training:
            return self.train_class_names, self.train_global_text_features, self.train_local_text_features
        return self.eval_class_names, self.eval_global_text_features, self.eval_local_text_features

    def _episode_text_features(self, inputs, unique_labels):
        class_names, global_bank, local_bank = self._select_text_bank()
        if len(class_names) == 0 or global_bank.numel() == 0:
            raise ValueError("SEMCAL requires non-empty class names and precomputed text features.")

        batch_class_list = inputs.get("batch_class_list", None)
        if batch_class_list is None:
            episode_indices = [int(v.item()) for v in unique_labels]
        else:
            if batch_class_list.dim() > 1:
                batch_class_list = batch_class_list[0]
            episode_indices = [int(batch_class_list[int(v.item())].item()) for v in unique_labels]

        global_features = []
        local_features = []
        for class_idx in episode_indices:
            if class_idx < 0 or class_idx >= len(class_names):
                raise IndexError(
                    f"Episode class index {class_idx} is out of range for {len(class_names)} configured class names."
                )
            global_features.append(global_bank[class_idx])
            local_features.append(local_bank[class_idx])
        return torch.stack(global_features, dim=0), torch.stack(local_features, dim=0)

    def _semantic_score(self, query_features, global_text_features, local_text_features):
        query_summary = query_features.detach().mean(dim=1)
        projection = self.visual_to_text_projection.to(query_summary.device)
        query_summary = F.normalize(torch.matmul(query_summary, projection), dim=-1)
        global_text_features = F.normalize(global_text_features.to(query_summary.device), dim=-1)
        semantic_score = torch.matmul(query_summary, global_text_features.t())

        if self.use_local_desc and local_text_features.numel() > 0:
            local_text_features = F.normalize(local_text_features.to(query_summary.device), dim=-1)
            local_score = torch.matmul(query_summary, local_text_features.t())
            if self.mode == "spatial":
                semantic_score = 0.4 * semantic_score + 0.6 * local_score
            else:
                semantic_score = 0.7 * semantic_score + 0.3 * local_score
        return semantic_score

    def _semantic_residual(self, semantic_score):
        residual = semantic_score - semantic_score.mean(dim=1, keepdim=True)
        residual_std = residual.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        residual = residual / residual_std
        return torch.clamp(residual, min=-self.residual_clip, max=self.residual_clip)

    def _semantic_gate(self, visual_logits):
        if not self.gate_enable:
            return visual_logits.new_ones((visual_logits.shape[0], 1))
        probs = F.softmax(visual_logits.detach(), dim=1)
        top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
        if top2.shape[1] < 2:
            return visual_logits.new_ones((visual_logits.shape[0], 1))
        margin = top2[:, 0] - top2[:, 1]
        if self.gate_margin <= 0:
            return visual_logits.new_ones((visual_logits.shape[0], 1))
        gate = (self.gate_margin - margin).clamp(min=0.0) / self.gate_margin
        return gate.unsqueeze(1)

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
        unique_labels, support_prototypes = self._build_class_sequence_prototypes(support_features, support_labels)
        class_logits = self.backbone.get_classification_logits(support_features, query_features)

        global_dist = bi_mhm_dist(self._pairwise_sequence_distance(query_features, support_prototypes))
        visual_logits = -global_dist

        if not self.enable_semcal or self.residual_weight <= 0:
            return {
                "logits": visual_logits,
                "class_logits": class_logits,
                "visual_logits": visual_logits,
            }

        global_text_features, local_text_features = self._episode_text_features(inputs, unique_labels)
        semantic_score = self._semantic_score(query_features, global_text_features, local_text_features)
        semantic_residual = self._semantic_residual(semantic_score)
        gate = self._semantic_gate(visual_logits)
        final_logits = visual_logits + gate * self.residual_weight * semantic_residual

        return {
            "logits": final_logits,
            "class_logits": class_logits,
            "visual_logits": visual_logits,
            "semantic_score": semantic_score,
            "semantic_residual": semantic_residual,
            "semantic_gate": gate,
        }
