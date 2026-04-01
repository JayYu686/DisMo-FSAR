#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
Semantic Knowledge Module for DiSMo: Few-Shot Action Recognition.

This module supports two description formats:
1. Legacy single-paragraph descriptions per class.
2. Structured descriptions with action/entity/phase fields.

The structured path is designed for motion-guided semantic modulation and
phase-level semantic supervision while preserving backward compatibility.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """
    Text encoder wrapper supporting multiple backends.

    Supports:
    - SentenceTransformer (default, lightweight)
    - CLIP text encoder (optional, visually aligned)
    """

    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda', allow_random_fallback=False):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.allow_random_fallback = allow_random_fallback
        self.encoder = None
        self.embed_dim = None

        self._load_encoder()

    def _load_encoder(self):
        """Load the text encoder based on model name."""
        if 'clip' in self.model_name.lower():
            self._load_clip_encoder()
        else:
            self._load_sentence_transformer()

    @staticmethod
    def _sentence_transformer_fallback_name() -> str:
        local_model = "./data/pretrained/all-MiniLM-L6-v2"
        return local_model if os.path.exists(local_model) else "all-MiniLM-L6-v2"

    def _load_sentence_transformer(self, model_name: Optional[str] = None):
        """Load SentenceTransformer model."""
        model_name = model_name or self.model_name
        try:
            from sentence_transformers import SentenceTransformer

            print(f"[DiSMo] Loading SentenceTransformer: {model_name}")
            self.encoder = SentenceTransformer(model_name)
            self.embed_dim = self.encoder.get_sentence_embedding_dimension()
            self.model_name = model_name

            for param in self.encoder.parameters():
                param.requires_grad = False
        except Exception as exc:
            if self.allow_random_fallback:
                print(
                    "[DiSMo] WARNING: failed to initialize sentence-transformers "
                    f"({exc}). Using random embeddings as fallback."
                )
                self.encoder = None
                self.embed_dim = 384
                return
            raise RuntimeError(
                "Failed to initialize SentenceTransformer '{}': {}. "
                "Please verify model path/network and dependencies, "
                "or set SEMANTIC.ALLOW_RANDOM_FALLBACK=true to force random fallback.".format(
                    model_name, exc
                )
            ) from exc

    def _load_clip_encoder(self):
        """Load CLIP text encoder."""
        try:
            import clip

            print(f"[DiSMo] Loading CLIP text encoder: {self.model_name}")
            model_name = "ViT-B/16"
            if "rn50" in self.model_name.lower():
                model_name = "RN50"
            elif "vit-b/32" in self.model_name.lower() or "vitb32" in self.model_name.lower():
                model_name = "ViT-B/32"
            elif "vit-l/14" in self.model_name.lower() or "vitl14" in self.model_name.lower():
                model_name = "ViT-L/14"

            model, _ = clip.load(model_name, device=self.device, jit=False)
            self.encoder = model
            self.embed_dim = getattr(model, "text_projection", None).shape[-1]

            for param in self.encoder.parameters():
                param.requires_grad = False
        except Exception as exc:
            fallback_name = self._sentence_transformer_fallback_name()
            print(
                "[DiSMo] WARNING: failed to initialize CLIP text encoder "
                f"({exc}). Falling back to SentenceTransformer backend: {fallback_name}."
            )
            self._load_sentence_transformer(fallback_name)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts to normalized embeddings.
        """
        if len(texts) == 0:
            return torch.empty(0, self.embed_dim, dtype=torch.float32)

        if self.encoder is None:
            embeddings = torch.randn(len(texts), self.embed_dim)
            return F.normalize(embeddings, dim=-1)

        if 'clip' in self.model_name.lower():
            import clip

            tokens = clip.tokenize(texts, truncate=True).to(self.device)
            embeddings = self.encoder.encode_text(tokens).float()
        else:
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        return F.normalize(embeddings.float(), dim=-1)


class SemanticModule(nn.Module):
    """
    Semantic knowledge module for vision-language alignment.

    Features:
    1. Legacy single-vector class semantics
    2. Structured multi-bank text features (global/entity/phase)
    3. Learnable vision-to-semantic projection
    4. Global and phase-level semantic supervision
    """

    def __init__(
        self,
        visual_dim: int = 1024,
        semantic_dim: Optional[int] = None,
        text_model: str = 'all-MiniLM-L6-v2',
        descriptions_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        device: str = 'cuda',
        num_classes: Optional[int] = None,
        allow_random_fallback: bool = False,
        description_format: str = 'legacy',
        use_structured_text: bool = False,
        num_phases: int = 3,
        strict_class_coverage: bool = False,
        dataset_name: str = '',
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.allow_random_fallback = allow_random_fallback
        self.description_format = description_format
        self.use_structured_text = use_structured_text
        self.num_phases = num_phases
        self.strict_class_coverage = strict_class_coverage
        self.dataset_name = str(dataset_name or '').lower()
        self._warned_label_mismatch = False
        self.descriptions_path = descriptions_path or ""

        self.text_encoder = TextEncoder(
            model_name=text_model,
            device=self.device,
            allow_random_fallback=self.allow_random_fallback,
        )
        self.semantic_dim = semantic_dim or self.text_encoder.embed_dim

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_dim // 2),
            nn.LayerNorm(visual_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(visual_dim // 2, self.semantic_dim),
            nn.LayerNorm(self.semantic_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.class_names = class_names or []
        self.descriptions: Dict[str, Any] = {}
        self.normalized_descriptions: Dict[str, Any] = {}
        self.normalized_desc_keys: Dict[str, str] = {}
        if descriptions_path and os.path.exists(descriptions_path):
            self._load_descriptions(descriptions_path)

        self.register_buffer('text_features', None)
        self.register_buffer('text_global_features', None)
        self.register_buffer('text_entity_features', None)
        self.register_buffer('text_phase_features', None)
        self.register_buffer('text_features_initialized', torch.tensor(False))

        init_classes = class_names or []
        if len(init_classes) == 0 and len(self.descriptions) > 0:
            init_classes = list(self.descriptions.keys())
        if num_classes is not None and num_classes > 0 and len(init_classes) == 0:
            init_classes = [f"class_{idx}" for idx in range(num_classes)]
        if len(init_classes) > 0:
            self.precompute_text_features(init_classes)

    def _load_descriptions(self, path: str):
        """Load class descriptions from JSON file."""
        try:
            (
                self.descriptions,
                self.normalized_descriptions,
                self.normalized_desc_keys,
            ) = self._load_descriptions_from_path(path)
            print(f"[DiSMo] Loaded {len(self.descriptions)} class descriptions from {path}")
        except Exception as exc:
            print(f"[DiSMo] WARNING: Failed to load descriptions: {exc}")
            self.descriptions = {}
            self.normalized_descriptions = {}
            self.normalized_desc_keys = {}

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        """Normalize class names so different naming styles map to the same key."""
        if not isinstance(name, str):
            return ""
        normalized = name.strip()
        if not normalized:
            return ""

        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", normalized)
        normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", normalized)
        normalized = normalized.replace("_", " ").replace("-", " ").replace("/", " ")
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\[\]]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _to_clean_text(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    @staticmethod
    def _humanize_class_name(name: str) -> str:
        text = str(name).strip()
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
        text = text.replace("_", " ").replace("-", " ").replace("/", " ")
        return " ".join(text.split())

    def _normalize_list_field(self, value: Any, max_items: Optional[int] = None) -> List[str]:
        if value is None:
            items: List[str] = []
        elif isinstance(value, list):
            items = [self._to_clean_text(v) for v in value if self._to_clean_text(v)]
        elif isinstance(value, str):
            if ',' in value:
                items = [self._to_clean_text(v) for v in value.split(',') if self._to_clean_text(v)]
            else:
                clean = self._to_clean_text(value)
                sentence_items = [self._to_clean_text(v) for v in re.split(r"(?<=[\.\!\?;])\s+", clean) if self._to_clean_text(v)]
                items = sentence_items if len(sentence_items) > 1 else ([clean] if clean else [])
        else:
            clean = self._to_clean_text(value)
            items = [clean] if clean else []

        deduped: List[str] = []
        seen = set()
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            deduped.append(item)
            seen.add(key)
        items = deduped
        if max_items is not None:
            items = items[:max_items]
        return items

    def _default_phase_texts(self, class_name: str, anchor: str) -> List[str]:
        base = anchor or f"a person performing {class_name}"
        return [
            f"start of {base}",
            f"core interaction of {base}",
            f"end state of {base}",
        ]

    def _heuristic_label_text(self, class_name: str) -> str:
        words = self._humanize_class_name(class_name).split()
        if len(words) <= 6:
            return " ".join(words)
        return " ".join(words[:6])

    def _extract_scene_priors(self, class_name: str, seed_text: str = "", max_items: int = 4) -> List[str]:
        scene_keywords = {
            "table", "surface", "floor", "ground", "camera", "water",
            "kitchen", "road", "field", "court", "pool", "room",
            "wall", "chair", "bed", "desk", "shelf", "container", "box",
        }
        source = f"{self._humanize_class_name(class_name)} {self._to_clean_text(seed_text)}".lower()
        items = [token for token in scene_keywords if re.search(rf"\b{re.escape(token)}\b", source)]
        return self._normalize_list_field(items, max_items=max_items)

    def _extract_attribute_cues(self, class_name: str, seed_text: str = "", max_items: int = 4) -> List[str]:
        attribute_keywords = {
            "left", "right", "up", "down", "into", "out", "inside", "outside",
            "open", "close", "opening", "closing", "upright", "side", "slanted",
            "falling", "rolling", "spinning", "tilting", "twisting", "pulling",
            "pushing", "touching", "moving", "dropping", "catching", "support",
            "supported", "unsupported", "containment", "contact", "collision",
            "revealing", "covering", "uncovering", "empty", "full", "wet", "dry",
        }
        source = f"{self._humanize_class_name(class_name)} {self._to_clean_text(seed_text)}".lower()
        items = [token for token in attribute_keywords if re.search(rf"\b{re.escape(token)}\b", source)]
        return self._normalize_list_field(items, max_items=max_items)

    def _normalize_description_entry(self, class_name: str, entry: Any) -> Dict[str, Any]:
        """
        Normalize legacy/structured_v1/structured_v2 entries to a unified record.
        """
        if isinstance(entry, dict):
            if any(key in entry for key in (
                'label_text', 'entity_priors', 'scene_priors',
                'attribute_cues', 'phase_cues', 'confusion_cues'
            )):
                label_text = self._to_clean_text(entry.get('label_text')) or self._heuristic_label_text(class_name)
                action_anchor = self._to_clean_text(
                    entry.get('action_anchor') or entry.get('label') or label_text
                )
                entity_priors = self._normalize_list_field(entry.get('entity_priors'), max_items=6)
                scene_priors = self._normalize_list_field(entry.get('scene_priors'), max_items=4)
                attribute_cues = self._normalize_list_field(entry.get('attribute_cues'), max_items=4)
                phase_cues = self._normalize_list_field(entry.get('phase_cues'))
                confusion_cues = self._normalize_list_field(entry.get('confusion_cues'), max_items=2)
            else:
                action_anchor = self._to_clean_text(
                    entry.get('action_anchor')
                    or entry.get('Action Label')
                    or entry.get('label')
                    or self._heuristic_label_text(class_name)
                )
                entity_priors = self._normalize_list_field(
                    entry.get('key_entities') or entry.get('Scene Description') or entry.get('entities'),
                    max_items=6,
                )
                phase_cues = self._normalize_list_field(
                    entry.get('motion_phases') or entry.get('Sub-Action Description') or entry.get('sub_actions')
                )
                confusion_cues = self._normalize_list_field(
                    entry.get('disambiguation') or entry.get('negative_cues'),
                    max_items=2,
                )
                seed_text = " ".join([action_anchor] + phase_cues + confusion_cues)
                label_text = self._heuristic_label_text(class_name)
                scene_priors = self._extract_scene_priors(class_name, seed_text)
                attribute_cues = self._extract_attribute_cues(class_name, seed_text)
        else:
            seed_text = self._to_clean_text(entry)
            action_anchor = seed_text or f"a person performing {self._humanize_class_name(class_name)}"
            label_text = self._heuristic_label_text(class_name)
            entity_priors = []
            scene_priors = self._extract_scene_priors(class_name, seed_text)
            attribute_cues = self._extract_attribute_cues(class_name, seed_text)
            phase_cues = []
            confusion_cues = []

        if not action_anchor:
            action_anchor = f"a person performing {self._humanize_class_name(class_name)}"
        if 'label_text' not in locals() or not label_text:
            label_text = self._heuristic_label_text(class_name)
        if len(entity_priors) == 0:
            entity_priors = self._normalize_list_field(
                self._extract_scene_priors(class_name, action_anchor) + [self._humanize_class_name(class_name)],
                max_items=6,
            )
        if len(phase_cues) == 0:
            phase_cues = self._default_phase_texts(class_name, action_anchor)
        if len(phase_cues) < self.num_phases:
            phase_cues = phase_cues + [phase_cues[-1]] * (self.num_phases - len(phase_cues))
        phase_cues = phase_cues[:self.num_phases]

        return {
            'label_text': label_text,
            'action_anchor': action_anchor,
            'entity_priors': entity_priors[:6],
            'scene_priors': scene_priors[:4],
            'attribute_cues': attribute_cues[:4],
            'phase_cues': phase_cues,
            'confusion_cues': confusion_cues[:2],
        }

    def _compose_entity_text(self, class_name: str, record: Dict[str, Any], use_structured_v2: bool = False) -> str:
        entities = list(record.get('entity_priors', []))
        scenes = list(record.get('scene_priors', []))
        attributes = list(record.get('attribute_cues', []))
        if use_structured_v2:
            is_ssv2 = 'ssv2' in self.dataset_name or '[' in class_name or ']' in class_name
            parts = entities + ([] if is_ssv2 else scenes) + attributes
        else:
            parts = entities
        if len(parts) == 0:
            return f"typical visual entities for {self._humanize_class_name(class_name)}"
        return ", ".join(parts)

    def _compose_fused_text(self, class_name: str, record: Dict[str, Any]) -> str:
        anchor = record.get('action_anchor') or f"a person performing {self._humanize_class_name(class_name)}"
        entities = record.get('entity_priors', [])
        if len(entities) == 0:
            return anchor
        return f"{anchor}. Typical visual entities: {', '.join(entities)}."

    def _load_descriptions_from_path(
        self,
        path: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
        with open(path, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
        if not isinstance(descriptions, dict):
            raise ValueError(f"Description file must be a JSON object: {path}")
        normalized_descriptions, normalized_desc_keys = self._build_normalized_description_index_from_source(descriptions)
        return descriptions, normalized_descriptions, normalized_desc_keys

    def _build_normalized_description_index(self):
        """Build normalized key -> description mapping for robust class-name lookup."""
        self.normalized_descriptions, self.normalized_desc_keys = self._build_normalized_description_index_from_source(
            self.descriptions
        )

    def _build_normalized_description_index_from_source(
        self,
        descriptions: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        normalized_descriptions = {}
        normalized_desc_keys = {}
        collisions = {}

        for key, desc in descriptions.items():
            normalized = self._normalize_class_name(key)
            if not normalized:
                continue
            if normalized in normalized_descriptions:
                prev_key = normalized_desc_keys[normalized]
                if prev_key != key:
                    collisions.setdefault(normalized, [prev_key]).append(key)
                continue
            normalized_descriptions[normalized] = desc
            normalized_desc_keys[normalized] = key

        if collisions:
            print(
                "[DiSMo] WARNING: normalized class-name collisions found for "
                f"{len(collisions)} keys. Keeping the first key for each collision."
            )
        return normalized_descriptions, normalized_desc_keys

    def _resolve_structured_description(
        self,
        class_name: str,
        descriptions: Optional[Dict[str, Any]] = None,
        normalized_descriptions: Optional[Dict[str, Any]] = None,
        normalized_desc_keys: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve a class to a structured description record with normalization fallback.
        """
        descriptions = self.descriptions if descriptions is None else descriptions
        normalized_descriptions = self.normalized_descriptions if normalized_descriptions is None else normalized_descriptions
        normalized_desc_keys = self.normalized_desc_keys if normalized_desc_keys is None else normalized_desc_keys

        if class_name in descriptions:
            record = self._normalize_description_entry(class_name, descriptions[class_name])
            return {'record': record, 'mode': 'exact', 'matched_key': class_name}

        normalized = self._normalize_class_name(class_name)
        if normalized in normalized_descriptions:
            matched_key = normalized_desc_keys[normalized]
            record = self._normalize_description_entry(class_name, normalized_descriptions[normalized])
            return {'record': record, 'mode': 'normalized', 'matched_key': matched_key}

        record = self._normalize_description_entry(class_name, None)
        return {'record': record, 'mode': 'fallback', 'matched_key': ''}

    def _resolve_class_text(self, class_name: str) -> Dict[str, str]:
        """Resolve class text for backward-compatible single-vector use."""
        resolved = self._resolve_structured_description(class_name)
        return {
            'text': self._compose_fused_text(class_name, resolved['record']),
            'mode': resolved['mode'],
            'matched_key': resolved['matched_key'],
        }

    def _validate_class_coverage(
        self,
        class_names: List[str],
        stats: Dict[str, Any],
        enforce_num_classes: bool = True,
    ):
        """Strict validation for structured-semantic experiments."""
        if enforce_num_classes and self.num_classes is not None and len(class_names) != self.num_classes:
            raise ValueError(
                f"NUM_CLASS={self.num_classes} but got {len(class_names)} class names. "
                "Fix TRAIN.NUM_CLASS / TRAIN.CLASS_NAME before running semantic experiments."
            )
        if stats['fallback_hits'] > 0:
            missing = ", ".join(stats['fallback_examples'][:10])
            raise ValueError(
                "Structured semantic descriptions do not cover all classes. "
                f"Missing examples: {missing}"
            )

    def build_text_feature_banks(
        self,
        class_names: List[str],
        strict_coverage: Optional[bool] = None,
        descriptions_path: Optional[str] = None,
        enforce_num_classes: bool = True,
    ) -> Dict[str, Any]:
        """
        Build fused/global/entity/phase text banks for a list of class names.
        """
        strict = self.strict_class_coverage if strict_coverage is None else strict_coverage
        descriptions = self.descriptions
        normalized_descriptions = self.normalized_descriptions
        normalized_desc_keys = self.normalized_desc_keys
        if descriptions_path:
            descriptions, normalized_descriptions, normalized_desc_keys = self._load_descriptions_from_path(descriptions_path)

        label_texts = []
        global_texts = []
        entity_texts = []
        phase_texts = []
        records = []
        exact_hits = 0
        normalized_hits = 0
        fallback_hits = 0
        normalized_examples = []
        fallback_examples = []

        for name in class_names:
            resolved = self._resolve_structured_description(
                name,
                descriptions=descriptions,
                normalized_descriptions=normalized_descriptions,
                normalized_desc_keys=normalized_desc_keys,
            )
            record = resolved['record']
            records.append(record)
            label_texts.append(record['label_text'])
            global_texts.append(record['action_anchor'])
            entity_texts.append(
                self._compose_entity_text(
                    name,
                    record,
                    use_structured_v2=(self.description_format == 'structured_v2'),
                )
            )
            phase_texts.extend(record['phase_cues'])

            if resolved['mode'] == 'exact':
                exact_hits += 1
            elif resolved['mode'] == 'normalized':
                normalized_hits += 1
                if len(normalized_examples) < 5:
                    normalized_examples.append(f"{name} -> {resolved['matched_key']}")
            else:
                fallback_hits += 1
                if len(fallback_examples) < 10:
                    fallback_examples.append(name)

        global_features = self.text_encoder.encode(global_texts)
        entity_features = self.text_encoder.encode(entity_texts)
        phase_features = self.text_encoder.encode(phase_texts).view(-1, self.num_phases, self.semantic_dim)
        if self.description_format == 'structured_v2':
            label_features = self.text_encoder.encode(label_texts)
            global_features = F.normalize(label_features * 0.7 + global_features * 0.3, dim=-1)
            phase_mean = phase_features.mean(dim=1)
            fused_features = F.normalize(
                global_features * 0.45 + entity_features * 0.20 + phase_mean * 0.35,
                dim=-1,
            )
        else:
            fused_features = F.normalize((global_features + entity_features) * 0.5, dim=-1)

        stats = {
            'exact_hits': exact_hits,
            'normalized_hits': normalized_hits,
            'fallback_hits': fallback_hits,
            'normalized_examples': normalized_examples,
            'fallback_examples': fallback_examples,
        }
        if strict:
            self._validate_class_coverage(class_names, stats, enforce_num_classes=enforce_num_classes)

        return {
            'fused': fused_features,
            'global': global_features,
            'entity': entity_features,
            'phase': phase_features,
            'records': records,
            'stats': stats,
        }

    def precompute_text_features(self, class_names: List[str]):
        """
        Precompute and cache text features for given classes.
        """
        self.class_names = class_names
        banks = self.build_text_feature_banks(class_names)

        text_features = banks['fused']
        text_global_features = banks['global']
        text_entity_features = banks['entity']
        text_phase_features = banks['phase']
        stats = banks['stats']

        if self.num_classes is not None and self.num_classes > 0 and not self.strict_class_coverage:
            if text_features.shape[0] < self.num_classes:
                pad_texts = [f"a video of action class {i}" for i in range(text_features.shape[0], self.num_classes)]
                pad_global = self.text_encoder.encode(pad_texts)
                pad_phase = pad_global.unsqueeze(1).repeat(1, self.num_phases, 1)
                text_features = torch.cat([text_features, pad_global], dim=0)
                text_global_features = torch.cat([text_global_features, pad_global], dim=0)
                text_entity_features = torch.cat([text_entity_features, pad_global], dim=0)
                text_phase_features = torch.cat([text_phase_features, pad_phase], dim=0)
                print(
                    "[DiSMo] WARNING: text prototype count smaller than NUM_CLASS; "
                    f"padding from {banks['fused'].shape[0]} to {self.num_classes}."
                )
            elif text_features.shape[0] > self.num_classes:
                text_features = text_features[:self.num_classes]
                text_global_features = text_global_features[:self.num_classes]
                text_entity_features = text_entity_features[:self.num_classes]
                text_phase_features = text_phase_features[:self.num_classes]
                print(
                    "[DiSMo] WARNING: text prototype count larger than NUM_CLASS; "
                    f"truncating from {banks['fused'].shape[0]} to {self.num_classes}."
                )

        self.text_features = text_features.detach()
        self.text_global_features = text_global_features.detach()
        self.text_entity_features = text_entity_features.detach()
        self.text_phase_features = text_phase_features.detach()
        self.text_features_initialized.fill_(True)

        print(f"[DiSMo] Precomputed text features for {self.text_features.shape[0]} classes")
        if len(self.descriptions) > 0:
            print(
                "[DiSMo] Description match stats: "
                f"exact={stats['exact_hits']}, normalized={stats['normalized_hits']}, "
                f"fallback={stats['fallback_hits']}"
            )
            if stats['normalized_examples']:
                print(
                    "[DiSMo] Normalized mapping examples: "
                    + "; ".join(stats['normalized_examples'])
                )

    def project_visual_features(self, visual_feat: torch.Tensor) -> torch.Tensor:
        """
        Project visual features into the semantic space.

        Supports [B, D] and [B, P, D] inputs.
        """
        if visual_feat.dim() == 2:
            vis_emb = self.visual_proj(visual_feat)
            return F.normalize(vis_emb, dim=-1)

        if visual_feat.dim() == 3:
            bsz, phases, dim = visual_feat.shape
            flat = visual_feat.reshape(bsz * phases, dim)
            flat_emb = self.visual_proj(flat)
            flat_emb = F.normalize(flat_emb, dim=-1)
            return flat_emb.reshape(bsz, phases, -1)

        raise ValueError(f"Unsupported visual feature shape for projection: {tuple(visual_feat.shape)}")

    def compute_phase_alignment_loss(
        self,
        phase_visual_feat: torch.Tensor,
        class_indices: torch.Tensor,
        text_phase_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Phase-wise semantic classification loss.

        Args:
            phase_visual_feat: [B, P, D]
            class_indices: [B]
            text_phase_features: optional [C, P, D_sem]
        """
        if self.text_phase_features is None and text_phase_features is None:
            raise RuntimeError("Phase text features not initialized")

        phase_vis_emb = self.project_visual_features(phase_visual_feat)  # [B, P, D_sem]
        phase_bank = text_phase_features if text_phase_features is not None else self.text_phase_features
        phase_bank = phase_bank.to(phase_visual_feat.device)
        class_indices = class_indices.to(phase_visual_feat.device).long()

        logit_scale = self.logit_scale.exp().clamp(max=100)
        loss = phase_visual_feat.new_tensor(0.0)
        for phase_idx in range(min(phase_vis_emb.shape[1], phase_bank.shape[1])):
            logits = logit_scale * phase_vis_emb[:, phase_idx, :] @ phase_bank[:, phase_idx, :].transpose(0, 1)
            loss = loss + F.cross_entropy(logits, class_indices)
        loss = loss / float(min(phase_vis_emb.shape[1], phase_bank.shape[1]))
        return loss

    def forward(
        self,
        visual_feat: torch.Tensor,
        class_indices: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for global semantic alignment.
        """
        if (not self.text_features_initialized.item()) and class_names is not None:
            self.precompute_text_features(class_names)

        if self.text_features is None:
            raise RuntimeError(
                "Text features not initialized. Call precompute_text_features() first "
                "or provide class_names argument."
            )

        vis_emb = self.project_visual_features(visual_feat)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        text_features = self.text_features.to(visual_feat.device)
        logits = logit_scale * vis_emb @ text_features.t()

        result = {
            'logits': logits,
            'vis_emb': vis_emb,
            'loss': None,
        }

        if class_indices is not None:
            class_indices = class_indices.to(visual_feat.device)
            if class_indices.max().item() < logits.shape[1]:
                result['loss'] = F.cross_entropy(logits, class_indices.long())
            elif not self._warned_label_mismatch:
                print(
                    "[DiSMo] WARNING: semantic labels exceed available text prototypes. "
                    "Skipping semantic loss for this run."
                )
                self._warned_label_mismatch = True

        return result

    def get_text_prototype(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Get fused text prototypes for specified classes."""
        if self.text_features is None:
            raise RuntimeError("Text features not initialized")
        return self.text_features[class_indices.long()].to(class_indices.device)

    def get_global_text_prototype(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Get global action text prototypes for specified classes."""
        if self.text_global_features is None:
            raise RuntimeError("Global text features not initialized")
        return self.text_global_features[class_indices.long()].to(class_indices.device)

    def get_entity_text_prototype(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Get entity text prototypes for specified classes."""
        if self.text_entity_features is None:
            raise RuntimeError("Entity text features not initialized")
        return self.text_entity_features[class_indices.long()].to(class_indices.device)

    def get_phase_text_prototype(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Get phase text prototypes for specified classes."""
        if self.text_phase_features is None:
            raise RuntimeError("Phase text features not initialized")
        return self.text_phase_features[class_indices.long()].to(class_indices.device)

    def get_all_text_features(self) -> torch.Tensor:
        """Get all fused text features."""
        if self.text_features is None:
            raise RuntimeError("Text features not initialized")
        return self.text_features


class EpisodeSFM(nn.Module):
    """
    Episode-level Semantic Feature Modulation via FiLM.
    """

    def __init__(self, text_dim: int = 384, visual_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, visual_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, visual_dim),
        )
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(text_emb)
        beta = self.beta_net(text_emb)
        if x.dim() != 3:
            raise ValueError(f"EpisodeSFM expects a 3D tensor, got {tuple(x.shape)}")

        if x.shape[1] == gamma.shape[-1]:
            # x: [B, D, T]
            gamma = gamma.unsqueeze(0).unsqueeze(-1)
            beta = beta.unsqueeze(0).unsqueeze(-1)
        elif x.shape[-1] == gamma.shape[-1]:
            # x: [B, T, D]
            gamma = gamma.unsqueeze(0).unsqueeze(0)
            beta = beta.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(
                f"EpisodeSFM feature dim mismatch: x={tuple(x.shape)}, "
                f"text_emb projects to {gamma.shape[-1]}"
            )
        return (1.0 + gamma) * x + beta


class ClassSFM(nn.Module):
    """
    Legacy per-class semantic channel attention.
    """

    def __init__(self, text_dim: int = 384, visual_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, visual_dim),
        )
        nn.init.zeros_(self.attn_net[-1].weight)
        nn.init.constant_(self.attn_net[-1].bias, 5.0)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(self.attn_net(text_emb))
        return x * mask


class PhaseAwareClassSFM(nn.Module):
    """
    Phase-aware class semantic modulation.

    Generates one channel-wise gate per semantic phase, then mixes the phase
    gates with per-frame soft phase weights from MotionPhaseRouter.
    """

    def __init__(self, text_dim: int = 384, visual_dim: int = 1024, num_phases: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.num_phases = num_phases
        self.attn_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, visual_dim),
        )
        nn.init.zeros_(self.attn_net[-1].weight)
        nn.init.constant_(self.attn_net[-1].bias, 5.0)

    def forward(self, x: torch.Tensor, phase_text_emb: torch.Tensor, phase_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Q, S, T, D] or [Q, T, D]
            phase_text_emb: [P, D_text]
            phase_weights: [Q, S, T, P] or [Q, T, P]
        """
        if phase_text_emb.dim() != 2:
            raise ValueError(f"phase_text_emb must be [P, D_text], got {tuple(phase_text_emb.shape)}")

        if x.dim() == 4:
            if phase_weights.dim() != 4:
                raise ValueError(f"Expected phase_weights [Q, S, T, P], got {tuple(phase_weights.shape)}")
        elif x.dim() == 3:
            if phase_weights.dim() != 3:
                raise ValueError(f"Expected phase_weights [Q, T, P], got {tuple(phase_weights.shape)}")
        else:
            raise ValueError(f"Unsupported feature shape for PhaseAwareClassSFM: {tuple(x.shape)}")

        phase_masks = torch.sigmoid(self.attn_net(phase_text_emb))  # [P, D]

        if x.dim() == 4:
            mixed_mask = torch.einsum('qstp,pd->qstd', phase_weights, phase_masks)
        else:
            mixed_mask = torch.einsum('qtp,pd->qtd', phase_weights, phase_masks)
        return x * mixed_mask


class SemanticFusion(nn.Module):
    """
    Fuse visual and semantic features for enhanced prototype.
    """

    def __init__(self, visual_dim: int, semantic_dim: int):
        super().__init__()
        self.semantic_proj = nn.Linear(semantic_dim, visual_dim)
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.ReLU(inplace=True),
            nn.Linear(visual_dim, visual_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.Sigmoid(),
        )

    def forward(self, visual_feat: torch.Tensor, semantic_feat: torch.Tensor) -> torch.Tensor:
        semantic_proj = self.semantic_proj(semantic_feat)
        combined = torch.cat([visual_feat, semantic_proj], dim=-1)
        gate = self.gate(combined)
        fused = visual_feat + gate * self.fusion(combined)
        return fused
