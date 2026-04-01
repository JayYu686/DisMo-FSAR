#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
LLM Description Generator for DiSMo.

This script generates semantic descriptions for action classes using LLMs.
Supports OpenAI API, local models (via transformers), or manual templates.

Usage:
    python scripts/generate_descriptions.py --dataset kinetics100 --output data/kinetics100_descriptions.json
    python scripts/generate_descriptions.py --dataset ssv2 --method template --output data/ssv2_descriptions.json
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# Template-based descriptions (fallback when no LLM available)
ACTION_TEMPLATES = [
    "a video showing a person {action}",
    "a person is performing the action of {action}",
    "someone doing {action} in a video",
    "a video clip of {action} being performed",
]

TEMPORAL_TEMPLATES = {
    "start": "The action begins with {detail}",
    "middle": "During the action, {detail}",
    "end": "The action ends with {detail}",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in video action recognition annotation. "
    "Write discriminative, strictly visual descriptions that maximize class separability."
)

DEFAULT_STRUCTURED_SYSTEM_PROMPT = (
    "You are an expert in video action recognition annotation. "
    "Output only valid JSON with concise, visually grounded action semantics."
)

DEFAULT_REASONER_STRUCTURED_SYSTEM_PROMPT = (
    "You are an expert in video action recognition annotation. "
    "Return only one compact JSON object. No markdown. No explanation."
)

DEFAULT_USER_PROMPT_TEMPLATE = """Write one detailed visual description for the action class: "{class_name}".

Requirements:
1) English only, one paragraph, 95-140 words.
2) Use only observable visual evidence: body parts, pose transitions, motion trajectory, speed, rhythm, force, contact changes.
3) Use explicit temporal structure: beginning -> middle -> ending.
4) Include object interaction and scene cues only when visually typical.
5) Add at least two contrastive cues with explicit comparisons using "unlike ...".
6) If the class implies direction/state change (left/right, up/down, in/out, on/off, open/close), state it explicitly.
7) Avoid generic openings like "a video showing" or "someone doing".
8) Do not mention datasets, labels, model names, or "this class".
9) Do not output bullets, numbering, headings, quotes, or markdown.
"""

DEFAULT_STRUCTURED_USER_PROMPT_TEMPLATE = """Generate one JSON object for the action class "{class_name}".

Return JSON only. Do not add markdown fences, comments, or extra prose.

Required schema:
{{
  "action_anchor": "one concise canonical sentence",
  "key_entities": ["entity 1", "entity 2"],
  "motion_phases": [
    "phase 1 short visual cue",
    "phase 2 short visual cue",
    "phase 3 short visual cue"
  ],
  "disambiguation": ["not similar action a", "not similar action b"]
}}

Requirements:
1) English only.
2) Use only observable visual evidence.
3) Keep action_anchor to one short sentence.
4) key_entities must contain at most 6 short noun phrases.
5) motion_phases must contain exactly {num_phases} short visual cues in temporal order.
6) disambiguation must contain at most 2 short contrastive cues.
7) Preserve placeholders like [something], [part], [number of] exactly if they appear in the class name.
8) Do not mention datasets, labels, model names, or "this class".
"""

DEFAULT_REASONER_STRUCTURED_USER_PROMPT_TEMPLATE = """Return one JSON object for "{class_name}".

Schema:
{{
  "action_anchor": "one short sentence",
  "key_entities": ["entity 1", "entity 2"],
  "motion_phases": [
    "phase 1",
    "phase 2",
    "phase 3"
  ],
  "disambiguation": ["cue 1", "cue 2"]
}}

Rules:
1) JSON only.
2) English only.
3) Keep placeholders like [something], [part], [number of] exactly.
4) action_anchor: one short visual sentence.
5) key_entities: at most 6 short noun phrases.
6) motion_phases: exactly {num_phases} short cues in time order.
7) disambiguation: at most 2 short cues.
8) Keep every field concise.
"""

DEFAULT_STRUCTURED_V2_SYSTEM_PROMPT = (
    "You are an expert in few-shot action recognition annotation. "
    "Output only valid JSON with compact, visually grounded semantic fields."
)

DEFAULT_REASONER_STRUCTURED_V2_SYSTEM_PROMPT = (
    "You are an expert in few-shot action recognition annotation. "
    "Return one compact JSON object only. No markdown. No explanation."
)

DEFAULT_STRUCTURED_V2_USER_PROMPT_TEMPLATE = """Generate one JSON object for the action class "{class_name}".

Return JSON only. Do not add markdown fences, comments, or extra prose.

Required schema:
{{
  "label_text": "canonical short class phrase",
  "action_anchor": "one short visual sentence",
  "entity_priors": ["entity 1", "entity 2"],
  "scene_priors": ["scene cue 1", "scene cue 2"],
  "attribute_cues": ["attribute cue 1", "attribute cue 2"],
  "phase_cues": [
    "phase 1 short visual cue",
    "phase 2 short visual cue",
    "phase 3 short visual cue"
  ],
  "confusion_cues": ["easy confusion 1", "easy confusion 2"]
}}

Requirements:
1) English only.
2) Use only observable visual evidence.
3) label_text must be a short canonical action phrase with 2-6 words and no explanation.
4) action_anchor must be one short visual sentence.
5) entity_priors must contain at most 6 short noun phrases.
6) scene_priors must contain at most 4 short scene cues and may be empty.
7) attribute_cues must contain at most 4 short direction/state/contact cues and may be empty.
8) phase_cues must contain exactly {num_phases} short cues in temporal order.
9) confusion_cues must contain at most 2 short contrastive cues.
10) Preserve placeholders like [something], [part], [number of], [somewhere] exactly if they appear in the class name.
11) Do not mention datasets, labels, model names, or "this class".
"""

DEFAULT_REASONER_STRUCTURED_V2_USER_PROMPT_TEMPLATE = """Return one JSON object for "{class_name}".

Schema:
{{
  "label_text": "short action phrase",
  "action_anchor": "one short visual sentence",
  "entity_priors": ["entity 1", "entity 2"],
  "scene_priors": ["scene cue 1", "scene cue 2"],
  "attribute_cues": ["attribute cue 1", "attribute cue 2"],
  "phase_cues": [
    "phase 1",
    "phase 2",
    "phase 3"
  ],
  "confusion_cues": ["cue 1", "cue 2"]
}}

Rules:
1) JSON only.
2) English only.
3) Keep placeholders like [something], [part], [number of], [somewhere] exactly.
4) label_text: 2-6 words, canonical action phrase.
5) action_anchor: one short visual sentence.
6) entity_priors: at most 6 short noun phrases.
7) scene_priors: at most 4 short scene cues.
8) attribute_cues: at most 4 short direction/state/contact cues.
9) phase_cues: exactly {num_phases} short cues in time order.
10) confusion_cues: at most 2 short cues.
11) Keep every field concise.
"""

STRUCTURED_OUTPUT_FORMATS = {"structured_v1", "structured_v2"}

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "into", "onto", "from", "with",
    "without", "while", "until", "that", "it", "is", "are", "be", "being", "been",
    "does", "do", "doing", "did", "so", "but", "then", "than", "your", "you",
    "someone", "person", "video", "action", "class", "showing", "show", "showing",
    "something", "somewhere", "part", "number", "many", "similar", "actually",
}

ACTION_WORDS = {
    "apply", "approaching", "attach", "attaching", "bend", "bending", "blowing",
    "brush", "brushing", "burying", "catch", "catching", "close", "closing",
    "covering", "dropping", "folding", "holding", "jumping", "letting", "lifting",
    "moving", "opening", "picking", "piling", "plugging", "poking", "pouring",
    "pretending", "pulling", "pushing", "putting", "removing", "rolling", "showing",
    "spilling", "spinning", "spreading", "squeezing", "stacking", "stuffing",
    "taking", "tearing", "tilting", "tipping", "touching", "turning", "twisting",
    "uncovering", "unfolding", "wiping", "wringing",
}

ATTRIBUTE_KEYWORDS = [
    "left", "right", "up", "down", "into", "out", "inside", "outside",
    "open", "close", "opening", "closing", "upright", "side", "slanted",
    "falling", "rolling", "spinning", "tilting", "twisting", "pulling",
    "pushing", "touching", "moving", "dropping", "catching", "support",
    "supported", "unsupported", "containment", "contact", "collision",
    "revealing", "covering", "uncovering", "empty", "full", "wet", "dry",
]

SCENE_KEYWORDS = [
    "table", "surface", "floor", "ground", "camera", "water", "kitchen",
    "road", "field", "court", "pool", "room", "wall", "chair", "bed",
    "desk", "shelf", "container", "box",
]

SSV2_ATTRIBUTE_CANONICAL = [
    "pretend", "fail", "almost no motion",
    "left", "right", "towards", "away", "up", "down",
    "into", "out of", "behind", "in front of", "over", "under",
    "contact", "no contact", "release", "grasp", "touch", "collision",
    "supported", "unsupported", "contained", "not contained", "balanced", "tilted",
    "open", "closed", "covered", "uncovered", "empty", "full", "upright", "sideways",
    "broken", "intact", "rolling", "spinning", "sliding", "falling", "lifting",
    "dropping", "twisting", "separating", "revealed", "hidden",
]

SSV2_GENERIC_SCENES = {
    "kitchen", "dining area", "indoor setting", "indoor", "room",
    "tabletop", "counter", "home interior", "dining table",
}

SSV2_PHASE_WORD_LIMIT = 8

HMDB_GENERIC_SCENES = {
    "home interior", "bathroom mirror", "bedroom vanity", "dressing table",
    "indoor/outdoor", "room", "open space", "dining table", "restaurant",
    "court-like area", "well-lit mirror", "living room", "outdoor setting",
}

UCF_GENERIC_SCENES = {
    "crowded stands", "sports arena", "competition venue", "outdoor field",
    "well-lit mirror", "dressing room", "living room floor", "play area",
    "indoor court", "outdoor court", "open field", "stadium seating",
}

K100_GENERIC_SCENES = {
    "living room", "concert venue", "bedroom", "stage", "bar", "gym",
    "home", "music venue", "street performance area", "open area",
    "indoor space", "home environment", "stadium", "competition venue",
    "gathering", "performance area", "stage area", "informal gathering",
}

GENERIC_ENTITY_TOKENS = {
    "person", "people", "man", "woman", "human", "actor", "someone",
}

GENERIC_ATTRIBUTE_PHRASES = {
    "team coordination", "background motion", "crowd presence", "environment context",
    "scene context", "camera framing", "general movement", "outdoor setting",
}

HMDB_STABLE_SCENE_HINTS = [
    (r"\bclimb stairs\b", ["staircase"]),
    (r"\bdive\b", ["swimming pool"]),
    (r"\bgolf\b", ["golf course"]),
    (r"\bfencing\b", ["fencing piste"]),
    (r"\bride bike\b", ["bike lane"]),
    (r"\bride horse\b", ["riding arena"]),
    (r"\bdribble\b|\bshoot ball\b", ["basketball court"]),
    (r"\bswing baseball\b", ["baseball field"]),
]

UCF_STABLE_SCENE_HINTS = [
    (r"\bbasketball\b", ["basketball court"]),
    (r"\bbalance beam\b", ["balance beam"]),
    (r"\barchery\b", ["archery range"]),
    (r"\bbench press\b", ["weight bench"]),
    (r"\bbreast stroke\b|\bfront crawl\b|\bswimming\b", ["swimming pool"]),
    (r"\bdiving\b|\bcliff diving\b", ["diving platform"]),
    (r"\bhorse riding\b|\bhorse race\b", ["horse track"]),
    (r"\bboxing\b", ["boxing ring"]),
    (r"\bparallel bars\b|\buneven bars\b", ["gym bar"]),
    (r"\bpommel horse\b", ["pommel horse"]),
]

K100_STABLE_SCENE_HINTS = [
    (r"\bbasketball\b", ["basketball court"]),
    (r"\bski\b|\bsnowboard\b", ["ski slope"]),
    (r"\bswimming\b|\bdiving\b", ["swimming pool"]),
    (r"\bboxing\b|\barm wrestling\b", ["combat mat"]),
    (r"\bice skating\b", ["ice rink"]),
    (r"\bplaying tennis\b", ["tennis court"]),
]


def _resolve_ssl_verify(ca_bundle: str = "", insecure_ssl: bool = False):
    """
    Resolve SSL verification mode for httpx/OpenAI client.
    Returns:
      - False: disable verification (unsafe; debug only)
      - str: custom CA bundle path
      - True: default verification
    """
    if insecure_ssl:
        return False
    if ca_bundle:
        return ca_bundle
    env_bundle = (
        os.getenv("REQUESTS_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("CURL_CA_BUNDLE")
    )
    if env_bundle:
        return env_bundle
    return True


def _is_ssv2_dataset(dataset: str = "", class_name: str = "") -> bool:
    dataset = str(dataset or "").lower()
    if "ssv2" in dataset:
        return True
    return bool(re.search(r"\[[^\]]+\]", str(class_name)))


def _dataset_key(dataset: str = "") -> str:
    return str(dataset or "").strip().lower()


def load_class_names(dataset: str) -> List[str]:
    """Load class names for a dataset."""
    preferred_local_files = {
        "kinetics100": "data/kinetics100_classes_all.txt",
        "hmdb51": "data/hmdb51_classes.txt",
        "ucf101": "data/ucf101_classes.txt",
        "ssv2": "data/ssv2_full_classes.txt",
    }
    local_file = preferred_local_files.get(dataset)
    if local_file and os.path.exists(local_file):
        return load_class_names_from_file(local_file)

    # Fallback builtin examples when no explicit class file is provided.
    dataset_classes = {
        "kinetics100": [
            "running", "walking", "jumping", "sitting", "standing",
            "waving", "clapping", "pointing", "pushing", "pulling",
            "throwing", "catching", "kicking", "punching", "dancing",
            "climbing", "swimming", "cycling", "driving", "cooking",
            "eating", "drinking", "reading", "writing", "typing",
            "talking", "laughing", "crying", "hugging", "handshake",
            "high_five", "playing_guitar", "playing_piano", "playing_drums",
            "singing", "exercising", "yoga", "stretching", "lifting_weights",
            "playing_basketball", "playing_soccer", "playing_tennis",
            "skateboarding", "surfing", "skiing", "snowboarding",
            "ice_skating", "bowling", "golf_swing", "archery",
            "fencing", "martial_arts", "boxing", "wrestling",
            "gymnastics", "diving", "rowing", "canoeing", "fishing",
            "gardening", "cleaning", "ironing", "sewing", "painting", "drawing"
        ],
        "ssv2": [
            "pushing something from left to right",
            "pushing something from right to left", 
            "moving something up",
            "moving something down",
            "pulling something from left to right",
            "pulling something from right to left",
            "putting something into something",
            "taking something out of something",
            "opening something",
            "closing something",
            "turning something upside down",
            "putting something on a surface",
            "picking something up",
            "dropping something",
            "throwing something",
            "catching something",
            "rolling something",
            "folding something",
            "unfolding something",
            "covering something with something",
        ],
        "hmdb51": [
            "brush_hair", "cartwheel", "catch", "chew", "clap",
            "climb", "climb_stairs", "dive", "draw_sword", "dribble",
            "drink", "eat", "fall_floor", "fencing", "flic_flac",
            "golf", "handstand", "hit", "hug", "jump",
            "kick", "kick_ball", "kiss", "laugh", "pick",
            "pour", "pullup", "punch", "push", "pushup",
            "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball",
            "shoot_bow", "shoot_gun", "sit", "situp", "smile",
            "smoke", "somersault", "stand", "swing_baseball", "sword",
            "sword_exercise", "talk", "throw", "turn", "walk", "wave"
        ],
        "ucf101": [
            "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling",
            "BalanceBeam", "BandMarching", "BaseballPitch", "Basketball",
            "BasketballDunk", "BenchPress", "Biking", "Billiards",
            "BlowDryHair", "BlowingCandles", "BodyWeightSquats", "Bowling",
            "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
            "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot",
            "CuttingInKitchen", "Diving", "Drumming", "Fencing", "FieldHockeyPenalty",
            "FloorGymnastics", "FrisbeeCatch", "FrontCrawl", "GolfSwing",
            "Haircut", "HammerThrow", "Hammering", "HandstandPushups",
            "HandstandWalking", "HeadMassage", "HighJump", "HorseRace",
            "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
            "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking"
        ]
    }
    return dataset_classes.get(dataset, [])


def load_class_names_from_file(path: str) -> List[str]:
    """Load class names from txt/json file.

    Supported formats:
    - txt: one class name per line
    - json list: ["class_a", "class_b", ...]
    - json dict:
      - {"class_a": "...description...", ...} -> use dict keys
      - {"0": "class_a", "1": "class_b", ...} -> use dict values (id->name)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"class names file not found: {path}")

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, dict):
            keys_are_int_like = True
            for k in obj.keys():
                if not str(k).isdigit():
                    keys_are_int_like = False
                    break
            if keys_are_int_like:
                pairs = sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])
                return [str(v).strip() for _, v in pairs if str(v).strip()]
            return [str(k).strip() for k in obj.keys() if str(k).strip()]

    raise ValueError(f"Unsupported class names file format: {path}")


def load_prompt_template(path: str) -> str:
    """Load user prompt template from txt/md file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompt template file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        template = f.read().strip()
    if not template:
        raise ValueError(f"prompt template file is empty: {path}")
    return template


def generate_template_description(class_name: str, templates: List[str] = None) -> str:
    """Generate description using simple templates."""
    if templates is None:
        templates = ACTION_TEMPLATES
    
    # 格式化类名
    action = class_name.replace("_", " ").replace("-", " ").lower()
    
    # 使用第一个模板
    description = templates[0].format(action=action)
    return description


def generate_detailed_description(class_name: str) -> str:
    """Generate more detailed description with temporal structure."""
    action = class_name.replace("_", " ").replace("-", " ").lower()
    
    description = (
        f"A video showing a person performing the action of {action}. "
        f"The action involves specific body movements and may include "
        f"interaction with objects or other people. "
        f"Key visual features include the motion patterns and poses associated with {action}."
    )
    return description


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def _humanize_class_name(class_name: str) -> str:
    text = str(class_name).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    return _normalize_text(text)


def _normalize_lookup_key(text: str) -> str:
    normalized = _humanize_class_name(text).lower()
    normalized = re.sub(r"[^a-z0-9\[\]]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _split_sentences(text: str) -> List[str]:
    clean = _normalize_text(text)
    if not clean:
        return []
    parts = re.split(r"(?<=[\.\!\?;])\s+", clean)
    sentences = []
    for part in parts:
        part = _normalize_text(part.strip(" .;"))
        if part:
            sentences.append(part)
    return sentences


def _extract_placeholders(text: str) -> List[str]:
    return [p.strip() for p in re.findall(r"\[[^\]]+\]", text)]


def _extract_entities(class_name: str, seed_text: str = "", max_items: int = 6) -> List[str]:
    entities: List[str] = []
    seen = set()

    for placeholder in _extract_placeholders(class_name):
        if placeholder not in seen:
            entities.append(placeholder)
            seen.add(placeholder)

    token_source = _humanize_class_name(class_name)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9']+", token_source):
        norm = token.lower()
        if norm in STOPWORDS or norm in ACTION_WORDS or len(norm) < 3:
            continue
        if norm.endswith("ing") or norm.endswith("ed"):
            continue
        if norm not in seen:
            entities.append(norm)
            seen.add(norm)
        if len(entities) >= max_items:
            break
    return entities[:max_items]


def _default_phase_texts(class_name: str, anchor: str, num_phases: int) -> List[str]:
    action = anchor or f"A person performs {_humanize_class_name(class_name)}."
    if num_phases <= 1:
        return [f"Core interaction during {action}"]
    defaults = [
        f"Beginning of {action}",
        f"Core interaction during {action}",
        f"Ending state of {action}",
    ]
    if num_phases <= len(defaults):
        return defaults[:num_phases]
    while len(defaults) < num_phases:
        defaults.append(defaults[-1])
    return defaults


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_normalize_text(x) for x in value if _normalize_text(x)]
    if isinstance(value, str):
        if "," in value:
            return [_normalize_text(x) for x in value.split(",") if _normalize_text(x)]
        sentences = _split_sentences(value)
        if len(sentences) > 1:
            return sentences
        clean = _normalize_text(value)
        return [clean] if clean else []
    clean = _normalize_text(value)
    return [clean] if clean else []


def _dedupe_keep_order(items: List[str], max_items: Optional[int] = None) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in items:
        clean = _normalize_text(item)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        deduped.append(clean)
        seen.add(key)
        if max_items is not None and len(deduped) >= max_items:
            break
    return deduped


def _extract_scene_priors(class_name: str, seed_text: str = "", max_items: int = 4) -> List[str]:
    source = f"{_humanize_class_name(class_name)} {_normalize_text(seed_text)}".lower()
    scenes = []
    for token in SCENE_KEYWORDS:
        if re.search(rf"\b{re.escape(token)}\b", source):
            scenes.append(token)
    return _dedupe_keep_order(scenes, max_items=max_items)


def _extract_attribute_cues(class_name: str, seed_text: str = "", max_items: int = 4) -> List[str]:
    source = f"{_humanize_class_name(class_name)} {_normalize_text(seed_text)}".lower()
    cues = []
    for token in ATTRIBUTE_KEYWORDS:
        if re.search(rf"\b{re.escape(token)}\b", source):
            cues.append(token)
    return _dedupe_keep_order(cues, max_items=max_items)


def _ssv2_role_placeholders(class_name: str) -> List[str]:
    placeholders = _extract_placeholders(class_name)
    if not placeholders:
        return []
    has_duplicates = len(set(placeholders)) < len(placeholders)
    role_names = ["first", "second", "third", "fourth"]
    resolved = []
    for idx, placeholder in enumerate(placeholders):
        if has_duplicates and idx < len(role_names):
            resolved.append(f"{role_names[idx]} {placeholder}")
        else:
            resolved.append(placeholder)
    return resolved


def _ssv2_trim_phrase(text: str, max_words: int = SSV2_PHASE_WORD_LIMIT) -> str:
    text = _normalize_text(text)
    if not text:
        return text
    placeholders = _extract_placeholders(text)
    masked = text
    restore_map = {}
    for idx, placeholder in enumerate(placeholders):
        token = f"__PH_{idx}__"
        masked = masked.replace(placeholder, token)
        restore_map[token] = placeholder
    words = masked.split()
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words])
    for token, placeholder in restore_map.items():
        trimmed = trimmed.replace(token, placeholder)
    return trimmed


def _ssv2_dedup(items: List[str], max_items: Optional[int] = None) -> List[str]:
    deduped = []
    seen = set()
    for item in items:
        clean = _normalize_text(item)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        deduped.append(clean)
        seen.add(key)
        if max_items is not None and len(deduped) >= max_items:
            break
    return deduped


def _canonicalize_ssv2_attribute_cue(cue: str) -> str:
    clean = _normalize_text(cue).lower()
    if not clean:
        return ""

    alias_rules = [
        (r"\bno touching\b|\bwithout touching\b|\bwithout contact\b|\bno contact\b", "no contact"),
        (r"\bmoves? closer\b|\bcloser\b|\bapproach(?:es|ing)?\b|\bnearer\b", "towards"),
        (r"\bmoves? away\b|\bfarther\b|\bfurther away\b", "away"),
        (r"\bfalls? out\b|\boutside\b|\boutward\b", "out of"),
        (r"\binside\b|\bin target\b|\bin container\b", "contained"),
        (r"\bnot inside\b|\bstays outside\b|\boutside target\b", "not contained"),
        (r"\bset on its side\b|\bon its side\b|\bfalls on its side\b|\bsideways\b", "sideways"),
        (r"\bkept upright\b|\bremains upright\b|\bupright\b", "upright"),
        (r"\bopening reduces\b|\bfully closes\b|\bclose[ds]?\b", "closed"),
        (r"\bopens?\b|\bunfolds?\b|\bopening\b", "open"),
        (r"\buncover(?:ed|ing)?\b", "uncovered"),
        (r"\bcover(?:ed|ing)?\b", "covered"),
        (r"\breveal(?:ed|ing)?\b|\bvisible\b", "revealed"),
        (r"\bhidden\b|\bobscured\b", "hidden"),
        (r"\bnot supported\b|\bsupport is lost\b|\bfalls off\b", "unsupported"),
        (r"\bdoesn't fall\b|\bdoesn't glide\b|\bremains balanced\b|\bstays in place\b", "balanced"),
        (r"\bsupported\b|\bmaintained support\b", "supported"),
        (r"\btilt(?:ed|ing)?\b|\bslanted\b", "tilted"),
        (r"\bgrips?\b|\bholding\b|\bholds\b|\bgrasps?\b", "grasp"),
        (r"\breleases?\b|\blets go\b|\bfingers open\b", "release"),
        (r"\btouch(?:es|ing)?\b", "touch"),
        (r"\bcollid(?:e|es|ing)\b", "collision"),
        (r"\broll(?:s|ing)?\b", "rolling"),
        (r"\bspin(?:s|ning)?\b", "spinning"),
        (r"\bslide(?:s|ing)?\b|\bglide(?:s|ing)?\b", "sliding"),
        (r"\bfall(?:s|ing)?\b", "falling"),
        (r"\blift(?:s|ing)?\b|\braise(?:s|d)?\b", "lifting"),
        (r"\bdrop(?:s|ping)?\b", "dropping"),
        (r"\btwist(?:s|ing)?\b|\bwring(?:ing)?\b", "twisting"),
        (r"\bseparat(?:e|es|ing)\b|\bpulls apart\b", "separating"),
        (r"\bpretend(?:s|ing)?\b", "pretend"),
        (r"\bfail(?:s|ed|ing)?\b|\bdoes not fit\b", "fail"),
        (r"\balmost doesn't move\b|\balmost no motion\b|\bbarely moves\b|\bslightly moves\b", "almost no motion"),
        (r"\bin front of\b", "in front of"),
        (r"\bbehind\b", "behind"),
        (r"\bunderneath\b|\bunder\b", "under"),
        (r"\bover\b", "over"),
        (r"\binto\b", "into"),
        (r"\bleft\b", "left"),
        (r"\bright\b", "right"),
        (r"\btowards\b", "towards"),
        (r"\baway\b", "away"),
        (r"\bup\b", "up"),
        (r"\bdown\b", "down"),
        (r"\bcontact\b", "contact"),
        (r"\bempty\b", "empty"),
        (r"\bfull\b", "full"),
        (r"\bbroken\b|\bsnaps apart\b", "broken"),
        (r"\bintact\b", "intact"),
    ]
    for pattern, canonical in alias_rules:
        if re.search(pattern, clean):
            return canonical
    return clean if clean in SSV2_ATTRIBUTE_CANONICAL else ""


def _ssv2_add_cue(cues: List[str], cue: str):
    canonical = _canonicalize_ssv2_attribute_cue(cue)
    if canonical and canonical not in cues:
        cues.append(canonical)


def _build_ssv2_label_text(class_name: str, existing: str = "") -> str:
    base = _normalize_text(existing) or _humanize_class_name(class_name)
    replacements = {
        "[something in it]": "content",
        "[something that cannot actually stand upright]": "unstable object",
        "[one of many similar things on the table]": "one object",
        "[number of]": "multiple",
        "[some substance]": "substance",
        "[somewhere]": "place",
        "[part]": "part",
        "[something]": "object",
    }
    base = base.lower()
    for src, dst in replacements.items():
        base = base.replace(src.lower(), dst)
    base = base.replace("pretending or trying and failing to", "fail")
    base = base.replace("trying and failing to", "fail")
    base = base.replace("pretending to", "pretend")
    base = base.replace("without actually", "without")
    base = re.sub(r"\s+", " ", base).strip(" .")
    words = base.split()
    if len(words) > 6:
        base = " ".join(words[:6])
    return base


def _build_ssv2_scene_priors(class_name: str) -> List[str]:
    lower = _humanize_class_name(class_name).lower()
    if "slanted surface" in lower:
        return ["slanted surface"]
    if "flat surface" in lower:
        return ["flat surface"]
    if "edge of" in lower:
        return ["surface edge"]
    return []


def _build_ssv2_entity_priors(class_name: str, existing_entities: Optional[List[str]] = None) -> List[str]:
    lower = _humanize_class_name(class_name).lower()
    placeholders = _ssv2_role_placeholders(class_name)
    entities: List[str] = []

    if len(placeholders) >= 2:
        entities.extend(placeholders[:2])
    elif "camera" in lower:
        entities.append("camera")
        entities.extend(placeholders[:1])
    else:
        entities.append("hand")
        entities.extend(placeholders)

    if not placeholders:
        if "stack" in lower:
            entities.append("stack")
        else:
            entities.append("object")
    if any(token in lower for token in ("slanted surface", "flat surface", "surface", "table", "edge of")):
        entities.append("surface")
    elif any(token in lower for token in ("into", "out of", "pour", "spill", "container")):
        entities.append("container")
    if existing_entities:
        for entity in existing_entities:
            if any(term in entity.lower() for term in ("hand", "camera", "surface", "container")):
                entities.append(entity)
    return _ssv2_dedup(entities, max_items=2)


def _build_ssv2_attribute_cues(class_name: str, record: Dict[str, Any]) -> List[str]:
    lower = _humanize_class_name(class_name).lower()
    cues: List[str] = []
    is_camera = "camera" in lower
    is_fail = "trying and failing" in lower or "failing" in lower or "does not fit" in lower
    is_pretend = "pretending" in lower

    def has_word(token: str) -> bool:
        return re.search(rf"\b{re.escape(token)}\b", lower) is not None

    if is_pretend:
        _ssv2_add_cue(cues, "pretend")
    if is_fail:
        _ssv2_add_cue(cues, "fail")
    if any(token in lower for token in ("almost doesn't move", "slightly moves", "almost falls off but doesn't", "so lightly")):
        _ssv2_add_cue(cues, "almost no motion")

    for relation in ("left", "right", "up", "down"):
        if has_word(relation):
            _ssv2_add_cue(cues, relation)
    if "towards the camera" in lower or "approaching" in lower or "moves towards" in lower:
        _ssv2_add_cue(cues, "towards")
    if "away from" in lower or "moving away" in lower:
        _ssv2_add_cue(cues, "away")
    if re.search(r"\binto\b", lower):
        _ssv2_add_cue(cues, "into")
    if any(token in lower for token in ("out of", "falls out", "taking out")):
        _ssv2_add_cue(cues, "out of")
    if "behind" in lower:
        _ssv2_add_cue(cues, "behind")
    if "in front of" in lower:
        _ssv2_add_cue(cues, "in front of")
    if re.search(r"\bover\b", lower):
        _ssv2_add_cue(cues, "over")
    if "underneath" in lower or re.search(r"\bunder\b", lower):
        _ssv2_add_cue(cues, "under")

    if any(token in lower for token in ("poking", "putting", "attaching", "plugging", "dropping onto", "covering", "opening", "closing", "touching")):
        _ssv2_add_cue(cues, "contact")
    if any(token in lower for token in ("pretending to poke", "pretending to put", "pretending to open", "pretending to spread air", "pretending or failing")):
        _ssv2_add_cue(cues, "no contact")
    if any(token in lower for token in ("touching", "touching (without moving)")):
        _ssv2_add_cue(cues, "touch")
    if any(token in lower for token in ("holding", "lifting", "pulling", "scooping", "taking", "putting", "attaching", "plugging", "moving [part]")):
        _ssv2_add_cue(cues, "grasp")
    if any(token in lower for token in ("dropping", "throwing", "falls out", "fingers open", "release")):
        _ssv2_add_cue(cues, "release")
    if "collide" in lower:
        _ssv2_add_cue(cues, "collision")

    if any(token in lower for token in ("falls off", "not supported", "edge of")):
        _ssv2_add_cue(cues, "unsupported")
    if any(token in lower for token in ("doesn't glide down", "doesn't fall down", "without the stack collapsing")):
        _ssv2_add_cue(cues, "balanced")
    if "without letting it drop down" in lower or "remains upright" in lower:
        _ssv2_add_cue(cues, "supported")
    if any(token in lower for token in ("tilting", "tipping", "slanted surface")):
        _ssv2_add_cue(cues, "tilted")
    if re.search(r"\binto\b", lower) and not is_fail:
        _ssv2_add_cue(cues, "contained")
    if any(token in lower for token in ("out of", "falls out", "revealing", "taking out")):
        _ssv2_add_cue(cues, "not contained")
    if is_fail and re.search(r"\binto\b", lower):
        _ssv2_add_cue(cues, "not contained")

    if any(token in lower for token in ("opening", "unfolding")):
        _ssv2_add_cue(cues, "open")
    if "closing" in lower:
        _ssv2_add_cue(cues, "closed")
    if "covering" in lower:
        _ssv2_add_cue(cues, "covered")
    if any(token in lower for token in ("uncovering", "revealing", "showing behind")):
        _ssv2_add_cue(cues, "uncovered")
        _ssv2_add_cue(cues, "revealed")
    if "empty" in lower:
        _ssv2_add_cue(cues, "empty")
    if "upright" in lower:
        _ssv2_add_cue(cues, "upright")
    if any(token in lower for token in ("on its side", "falls on its side", "side, not upright", "upside down")):
        _ssv2_add_cue(cues, "sideways")
    if "break" in lower:
        _ssv2_add_cue(cues, "broken")

    if "roll" in lower:
        _ssv2_add_cue(cues, "rolling")
    if "spin" in lower:
        _ssv2_add_cue(cues, "spinning")
    if any(token in lower for token in ("slide", "glide")):
        _ssv2_add_cue(cues, "sliding")
    if "fall" in lower:
        _ssv2_add_cue(cues, "falling")
    if "lift" in lower:
        _ssv2_add_cue(cues, "lifting")
    if "drop" in lower:
        _ssv2_add_cue(cues, "dropping")
    if any(token in lower for token in ("twist", "wringing")):
        _ssv2_add_cue(cues, "twisting")
    if "separate" in lower:
        _ssv2_add_cue(cues, "separating")

    if "moving [part] of" in lower:
        _ssv2_add_cue(cues, "contact")
        _ssv2_add_cue(cues, "grasp")
    if "showing" in lower and "camera" in lower:
        _ssv2_add_cue(cues, "revealed")

    if is_camera:
        cues = [cue for cue in cues if cue not in {"contact", "grasp", "release"}]
    if is_pretend and "contact" in cues:
        cues = [cue for cue in cues if cue != "contact"]
    if is_fail and "contained" in cues:
        cues = [cue for cue in cues if cue != "contained"]
    if "sideways" in cues and "upright" in cues:
        cues = [cue for cue in cues if cue != "upright"]
    if "no contact" in cues and "contact" in cues:
        cues = [cue for cue in cues if cue != "contact"]
    if "upright" in cues and "right" in cues and not has_word("right"):
        cues = [cue for cue in cues if cue != "right"]

    if len(cues) < 2 and "contact" not in cues and "no contact" not in cues and "grasp" not in cues and not is_camera:
        _ssv2_add_cue(cues, "contact")
    if len(cues) < 2 and not any(cue in cues for cue in ("rolling", "spinning", "sliding", "falling", "lifting", "dropping", "twisting", "separating")):
        if "moving" in lower and not is_camera:
            _ssv2_add_cue(cues, "almost no motion")

    return _ssv2_dedup(cues, max_items=4)


def _ssv2_roles(class_name: str) -> Tuple[str, str]:
    lower = _humanize_class_name(class_name).lower()
    placeholders = _ssv2_role_placeholders(class_name)
    first = placeholders[0] if placeholders else ("stack" if "stack" in lower else ("camera" if "camera" in lower else "object"))
    second = placeholders[1] if len(placeholders) > 1 else ("surface" if any(token in lower for token in ("surface", "table", "edge of")) else "target")
    return first, second


def _build_ssv2_phase_cues(class_name: str, record: Dict[str, Any], num_phases: int) -> List[str]:
    lower = _humanize_class_name(class_name).lower()
    first_obj, second_obj = _ssv2_roles(class_name)

    if "moving [part] of" in lower:
        phases = [f"[part] on {second_obj}", f"[part] moves on {second_obj}", "[part] reaches new position"]
    elif "touching (without moving) [part] of" in lower:
        phases = ["hand approaches [part]", "[part] is touched", "[part] stays still"]
    elif "turning the camera upwards" in lower:
        phases = [f"camera below {first_obj}", "camera tilts upward", f"{first_obj} higher in view"]
    elif "turning the camera downwards" in lower:
        phases = [f"camera above {first_obj}", "camera tilts downward", f"{first_obj} lower in view"]
    if "turning the camera left" in lower:
        phases = [f"camera faces {first_obj}", "camera turns left", f"{first_obj} shifts right"]
    elif "moving away from" in lower and "camera" in lower:
        phases = [f"camera near {first_obj}", "camera moves away", f"{first_obj} farther in view"]
    elif "approaching" in lower and "camera" in lower:
        phases = [f"camera far from {first_obj}", "camera moves towards", f"{first_obj} larger in view"]
    elif "showing" in lower and "camera" in lower:
        phases = [f"hand holds {first_obj}", f"{first_obj} faces camera", f"{first_obj} centered in view"]
    elif "pretending to poke" in lower:
        phases = [f"hand approaches {first_obj}", "finger stops near target", "no contact made"]
    elif "pretending to open" in lower:
        phases = [f"hand near {first_obj}", "opening motion mimed", f"{first_obj} stays closed"]
    elif "pretending to put" in lower and any(token in lower for token in ("onto", "on a surface", "onto a")):
        phases = [f"hand holds {first_obj}", f"{first_obj} moves near {second_obj}", f"{first_obj} not placed"]
    elif "pretending to put" in lower and any(token in lower for token in ("underneath", "under")):
        phases = [f"hand holds {first_obj}", f"{first_obj} moves under {second_obj}", f"{first_obj} not placed"]
    elif "pretending to turn" in lower and "upside down" in lower:
        phases = [f"hand holds {first_obj}", f"{first_obj} rotates partially", f"{first_obj} not inverted"]
    elif any(token in lower for token in ("trying and failing", "does not fit")):
        phases = [f"hand aligns {first_obj}", f"{first_obj} presses at opening", f"{first_obj} stays outside"]
    elif "without the stack collapsing" in lower:
        phases = ["finger approaches stack", "finger contacts stack", "stack stays upright"]
    elif "without letting it drop down" in lower:
        phases = [f"{first_obj} supported at one end", f"{first_obj} lifts upward", f"{first_obj} stays supported"]
    elif "doesn't glide down" in lower:
        phases = [f"{first_obj} on slanted surface", "surface tilts slightly", f"{first_obj} stays in place"]
    elif "without it falling down" in lower:
        phases = [f"{first_obj} on surface", f"{first_obj} moves slightly", f"{first_obj} stays on surface"]
    elif "pouring" in lower and "into" in lower:
        phases = [f"{first_obj} above {second_obj}", f"{first_obj} tilts into {second_obj}", f"{first_obj} inside {second_obj}"]
    elif any(token in lower for token in ("putting", "dropping")) and "into" in lower:
        phases = [f"hand holds {first_obj}", f"{first_obj} moves into {second_obj}", f"{first_obj} inside {second_obj}"]
    elif any(token in lower for token in ("taking", "out of")):
        phases = [f"{first_obj} inside {second_obj}", f"{first_obj} moves out of {second_obj}", f"{first_obj} outside {second_obj}"]
    elif "falls out" in lower:
        phases = [f"{first_obj} inside {second_obj}", f"{first_obj} moves out of {second_obj}", f"{first_obj} outside {second_obj}"]
    elif "behind" in lower:
        final_phrase = f"{first_obj} behind {second_obj}"
        if "dropping" in lower or "spilling" in lower:
            final_phrase = f"{first_obj} lands behind {second_obj}"
        phases = [f"hand holds {first_obj}", f"{first_obj} moves behind {second_obj}", final_phrase]
    elif "in front of" in lower:
        final_phrase = f"{first_obj} in front of {second_obj}"
        if "dropping" in lower:
            final_phrase = f"{first_obj} lands in front"
        phases = [f"hand holds {first_obj}", f"{first_obj} moves in front of {second_obj}", final_phrase]
    elif "next to" in lower:
        phases = [f"hand holds {first_obj}", f"{first_obj} moves next to {second_obj}", f"{first_obj} stays next to {second_obj}"]
    elif re.search(r"\bover\b", lower):
        phases = [f"hand holds {first_obj}", f"{first_obj} moves over {second_obj}", f"{first_obj} stays over {second_obj}"]
    elif "underneath" in lower or re.search(r"\bunder\b", lower):
        phases = [f"hand holds {first_obj}", f"{first_obj} moves under {second_obj}", f"{first_obj} under {second_obj}"]
    elif "upright on the table" in lower and any(token in lower for token in ("falls on its side", "not upright")):
        phases = [f"{first_obj} off table", "base contacts table", f"{first_obj} rests on side"]
    elif "upright on the table" in lower:
        phases = [f"{first_obj} off table", "base contacts table", f"{first_obj} remains upright"]
    elif any(token in lower for token in ("on its side", "not upright", "falls on its side")):
        phases = [f"{first_obj} above surface", f"{first_obj} laid sideways", f"{first_obj} rests on side"]
    elif "edge of" in lower and "falls down" in lower:
        phases = [f"{first_obj} at surface edge", "support is lost", f"{first_obj} falls down"]
    elif "falls off the table" in lower:
        phases = [f"{first_obj} on table", f"{first_obj} moves off edge", f"{first_obj} falls down"]
    elif "across a surface until it falls down" in lower:
        phases = [f"{first_obj} on surface", f"{first_obj} moves across surface", f"{first_obj} falls off edge"]
    elif "across a surface without it falling down" in lower:
        phases = [f"{first_obj} on surface", f"{first_obj} moves across surface", f"{first_obj} stays on surface"]
    elif "lifting a surface" in lower and "starts sliding down" in lower:
        phases = [f"{first_obj} on surface", "surface tilts upward", f"{first_obj} starts sliding"]
    elif "so lightly that it doesn't or almost doesn't move" in lower:
        phases = [f"finger approaches {first_obj}", "brief light contact", f"{first_obj} barely moves"]
    elif "so that it spins around" in lower:
        phases = [f"finger approaches {first_obj}", "side contact is made", f"{first_obj} spins"]
    elif "so it slightly moves" in lower:
        phases = [f"finger approaches {first_obj}", "contact is made", f"{first_obj} shifts slightly"]
    elif "poking a stack" in lower and "collapses" in lower:
        phases = ["finger approaches stack", "finger hits stack", "stack collapses"]
    elif "poking a hole" in lower:
        phases = [f"hand approaches {first_obj}", "tool presses into surface", "hole appears"]
    elif "touching (without moving)" in lower:
        phases = [f"hand approaches {first_obj}", f"{first_obj} is touched", f"{first_obj} stays still"]
    elif "collide with each other" in lower or ("colliding with" in lower and "deflected" in lower):
        phases = [f"{first_obj} and {second_obj} apart", f"{first_obj} moves into {second_obj}", "objects collide"]
    elif "opening" in lower:
        phases = [f"{first_obj} closed", f"{first_obj} opens", f"{first_obj} stays open"]
    elif "closing" in lower:
        phases = [f"{first_obj} open", f"{first_obj} closes", f"{first_obj} stays closed"]
    elif any(token in lower for token in ("uncovering", "revealing", "showing behind")):
        phases = ["target hidden", "cover moves away", "target becomes visible"]
    elif "covering" in lower:
        phases = [f"{first_obj} visible", f"{second_obj} moves over {first_obj}", f"{first_obj} becomes covered"]
    elif "unfolding" in lower:
        phases = [f"{first_obj} folded", f"{first_obj} opens outward", f"{first_obj} unfolded"]
    elif "roll down a slanted surface" in lower:
        phases = [f"{first_obj} on slanted surface", f"{first_obj} rolls downward", f"{first_obj} lower on surface"]
    elif "roll up a slanted surface" in lower:
        phases = [f"{first_obj} rolls uphill", f"{first_obj} slows near top", f"{first_obj} rolls back down"]
    elif "roll along a flat surface" in lower or "rolling" in lower:
        phases = [f"{first_obj} on flat surface", f"{first_obj} rolls forward", f"{first_obj} farther along surface"]
    elif "spinning" in lower:
        phases = [f"{first_obj} at rest", f"{first_obj} spins quickly", f"{first_obj} stops spinning"]
    elif any(token in lower for token in ("twisting", "wringing")):
        if "water comes out" in lower:
            phases = [f"hand grips {first_obj}", f"{first_obj} twists tightly", "water leaves object"]
        else:
            phases = [f"hand grips {first_obj}", f"{first_obj} twists around axis", f"{first_obj} stays twisted"]
    elif "squeezing" in lower:
        phases = [f"hand grips {first_obj}", f"{first_obj} compresses", f"{first_obj} stays squeezed"]
    elif "bending" in lower and "break" in lower:
        phases = [f"hand grips {first_obj}", f"{first_obj} bends deeply", f"{first_obj} breaks apart"]
    elif "throwing" in lower and "catching" in lower:
        phases = [f"{first_obj} in hand", f"{first_obj} moves upward", f"{first_obj} caught again"]
    elif "stacking" in lower or "putting [number of]" in lower:
        phases = ["items separate", "items placed together", "stack formed"]
    elif "piling" in lower:
        phases = ["items separate", "items gather together", "pile formed"]
    elif any(token in lower for token in ("attaching", "plugging")):
        phases = [f"{first_obj} separate from {second_obj}", f"{first_obj} moves into contact", f"{first_obj} attached to {second_obj}"]
    elif "spreading" in lower:
        phases = [f"{first_obj} held above {second_obj}", f"{first_obj} spreads across {second_obj}", f"{second_obj} becomes covered"]
    elif "wiping" in lower and "off of" in lower:
        phases = [f"{first_obj} on {second_obj}", "wiping contact begins", f"{first_obj} removed from {second_obj}"]
    else:
        phases = list(record.get("phase_cues") or [])
        if len(phases) < num_phases:
            phases = [
                f"{first_obj} before action",
                f"{first_obj} changes state",
                f"{first_obj} in final state",
            ]

    phases = [_ssv2_trim_phrase(phase) for phase in phases[:num_phases]]
    if len(phases) < num_phases:
        phases = phases + [phases[-1]] * (num_phases - len(phases))
    return phases[:num_phases]


def _build_ssv2_confusion_cues(class_name: str, record: Dict[str, Any]) -> List[str]:
    lower = _humanize_class_name(class_name).lower()
    confusion = list(record.get("confusion_cues") or [])

    if "pretending" in lower:
        confusion.append("actual action completion")
    if "trying and failing" in lower or "does not fit" in lower:
        confusion.append("successful completion")
    if "behind" in lower:
        confusion.append("in front of")
    if "in front of" in lower:
        confusion.append("behind")
    if "opening" in lower:
        confusion.append("closing [something]")
    if "closing" in lower:
        confusion.append("opening [something]")
    if any(token in lower for token in ("uncovering", "revealing")):
        confusion.append("covering [something]")
    if "without the stack collapsing" in lower:
        confusion.append("stack collapses")
    return _ssv2_dedup(confusion, max_items=2)


def _postprocess_ssv2_structured_v2(class_name: str, record: Dict[str, Any], num_phases: int) -> Dict[str, Any]:
    action_anchor = _normalize_text(record.get("action_anchor") or "")
    if not action_anchor:
        action_anchor = _humanize_class_name(class_name)
    action_anchor = re.sub(
        r"\b(kitchen|dining area|indoor setting|room|tabletop|counter)\b",
        "",
        action_anchor,
        flags=re.IGNORECASE,
    )
    action_anchor = _normalize_text(action_anchor.strip(" ,.;"))
    if action_anchor and not action_anchor.endswith("."):
        action_anchor += "."

    scene_priors = [
        scene for scene in _build_ssv2_scene_priors(class_name)
        if scene.lower() not in SSV2_GENERIC_SCENES
    ]

    return {
        "label_text": _build_ssv2_label_text(class_name, record.get("label_text", "")),
        "action_anchor": action_anchor,
        "entity_priors": _build_ssv2_entity_priors(class_name, record.get("entity_priors", [])),
        "scene_priors": _ssv2_dedup(scene_priors, max_items=1),
        "attribute_cues": _build_ssv2_attribute_cues(class_name, record),
        "phase_cues": _build_ssv2_phase_cues(class_name, record, num_phases),
        "confusion_cues": _build_ssv2_confusion_cues(class_name, record),
    }


def _short_phrase(text: str, max_words: int = 7) -> str:
    return _ssv2_trim_phrase(text, max_words=max_words)


def _clean_anchor_text(anchor: str, banned_terms: set[str], max_words: int = 16) -> str:
    clean = _normalize_text(anchor or "")
    if not clean:
        return clean
    for term in sorted(banned_terms, key=len, reverse=True):
        clean = re.sub(rf"\b{re.escape(term)}\b", "", clean, flags=re.IGNORECASE)
    clean = _normalize_text(clean.strip(" ,.;"))
    clean = _short_phrase(clean, max_words=max_words)
    if clean and not clean.endswith("."):
        clean += "."
    return clean


def _pattern_hints(class_name: str, hint_rules: List[Tuple[str, List[str]]], max_items: int) -> List[str]:
    lower = _humanize_class_name(class_name).lower()
    hints: List[str] = []
    for pattern, values in hint_rules:
        if re.search(pattern, lower):
            hints.extend(values)
    return _dedupe_keep_order(hints, max_items=max_items)


def _filtered_scene_priors(
    class_name: str,
    existing: Optional[List[str]],
    banned_terms: set[str],
    hint_rules: List[Tuple[str, List[str]]],
    allow_tokens: set[str],
    max_items: int,
) -> List[str]:
    scenes = _pattern_hints(class_name, hint_rules, max_items=max_items)
    for scene in existing or []:
        clean = _normalize_text(scene)
        if not clean:
            continue
        low = clean.lower()
        if any(term in low for term in banned_terms):
            continue
        if any(token in low for token in allow_tokens):
            scenes.append(clean)
    return _dedupe_keep_order(scenes, max_items=max_items)


def _filtered_entities(
    class_name: str,
    existing: Optional[List[str]],
    max_items: int,
    drop_person_terms: bool = True,
) -> List[str]:
    candidates = _extract_entities(class_name, max_items=8) + list(existing or [])
    entities: List[str] = []
    for entity in candidates:
        clean = _normalize_text(entity)
        if not clean:
            continue
        low = clean.lower()
        if drop_person_terms and low in GENERIC_ENTITY_TOKENS:
            continue
        if any(term in low for term in HMDB_GENERIC_SCENES | UCF_GENERIC_SCENES | K100_GENERIC_SCENES):
            continue
        entities.append(clean)
    return _dedupe_keep_order(entities, max_items=max_items)


def _filtered_confusions(existing: Optional[List[str]], max_items: int = 2) -> List[str]:
    return _dedupe_keep_order([_short_phrase(item, max_words=4) for item in (existing or [])], max_items=max_items)


def _filtered_attribute_cues(
    class_name: str,
    existing: Optional[List[str]],
    pattern_rules: List[Tuple[str, List[str]]],
    max_items: int,
) -> List[str]:
    cues = _pattern_hints(class_name, pattern_rules, max_items=max_items)
    for cue in existing or []:
        clean = _short_phrase(cue, max_words=4)
        low = clean.lower()
        if not clean or low in GENERIC_ATTRIBUTE_PHRASES:
            continue
        if any(term in low for term in ("crowd", "background", "scene", "environment", "venue")):
            continue
        cues.append(clean)
    return _dedupe_keep_order(cues, max_items=max_items)


def _clean_phase_cues(
    existing: Optional[List[str]],
    fallback: List[str],
    num_phases: int,
    banned_terms: set[str],
    max_words: int,
) -> List[str]:
    phases = list(existing or [])
    if not phases:
        phases = list(fallback)
    cleaned: List[str] = []
    for phase in phases[:num_phases]:
        text = _normalize_text(phase)
        for term in sorted(banned_terms, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(term)}\b", "", text, flags=re.IGNORECASE)
        text = _normalize_text(text.strip(" ,.;"))
        cleaned.append(_short_phrase(text, max_words=max_words))
    while len(cleaned) < num_phases:
        cleaned.append(cleaned[-1] if cleaned else fallback[min(len(cleaned), len(fallback) - 1)])
    return cleaned[:num_phases]


HMDB_ATTRIBUTE_HINTS = [
    (r"\bbrush hair\b", ["repetitive strokes", "object contact"]),
    (r"\bclimb stairs\b", ["upward motion", "alternating steps"]),
    (r"\bride horse\b|\bride bike\b", ["mounted posture"]),
    (r"\bhug\b|\bkiss\b|\bshake hands\b", ["two-person contact"]),
    (r"\bpushup\b|\bsitup\b|\bpullup\b|\bhandstand\b", ["ground contact"]),
    (r"\bkick", ["kick motion"]),
    (r"\bpunch", ["punch motion"]),
    (r"\bswing baseball\b|\bgolf\b|\bsword\b|\bdraw sword\b", ["body rotation", "object contact"]),
]

UCF_ATTRIBUTE_HINTS = [
    (r"\bbasketball\b|\bbasketball dunk\b", ["ball control", "jump"]),
    (r"\barchery\b", ["object contact", "upward release"]),
    (r"\bbreast stroke\b|\bfront crawl\b|\bswimming\b|\bdiving\b", ["water contact"]),
    (r"\bparallel bars\b|\buneven bars\b|\bpommel horse\b|\bbalance beam\b", ["bar support", "rotation"]),
    (r"\bplaying\b|\bdrumming\b", ["instrument playing"]),
    (r"\bbiking\b|\bhorse riding\b|\bkayaking\b", ["vehicle riding"]),
    (r"\bgolf swing\b|\btable tennis shot\b|\bcricket shot\b|\bfield hockey penalty\b", ["racket swing"]),
]

K100_ATTRIBUTE_HINTS = [
    (
        r"\bair drumming\b|\bplaying accordion\b|\bplaying didgeridoo\b|\bplaying keyboard\b|"
        r"\bplaying ukulele\b|\bplaying xylophone\b|\bplaying drums\b|\bplaying trumpet\b",
        ["instrument playing"],
    ),
    (r"\bbike\b|\bcycling\b|\briding\b", ["vehicle riding"]),
    (r"\bsurfing\b|\bskateboarding\b|\bsnowboarding\b|\bskiing\b", ["board riding"]),
    (r"\bswimming\b|\bdiving\b|\browing\b|\bcanoeing\b|\bfishing\b", ["water interaction"]),
    (r"\bbasketball\b|\bsoccer\b|\btennis\b|\bgolf\b|\bbowling\b", ["ball interaction"]),
    (r"\bclimbing\b|\brope\b", ["climb ascent"]),
    (r"\bboxing\b|\bwrestling\b|\bfencing\b|\bmartial arts\b", ["impact"]),
    (r"\bcooking\b|\bgardening\b|\bcleaning\b|\bironing\b|\bsewing\b|\bpainting\b|\bdrawing\b", ["tool use"]),
]


def _postprocess_hmdb51_structured_v2(class_name: str, record: Dict[str, Any], num_phases: int) -> Dict[str, Any]:
    label_text = _normalize_label_text(record.get("label_text") or _heuristic_label_text(class_name), class_name).lower()
    anchor = _clean_anchor_text(record.get("action_anchor") or _humanize_class_name(class_name), HMDB_GENERIC_SCENES, max_words=14)
    scenes = _filtered_scene_priors(
        class_name,
        record.get("scene_priors", []),
        HMDB_GENERIC_SCENES,
        HMDB_STABLE_SCENE_HINTS,
        {"stair", "pool", "golf", "bike", "horse", "fencing", "field", "court"},
        max_items=1,
    )
    phases = _clean_phase_cues(
        record.get("phase_cues", []),
        ["initial posture", "core body motion", "ending state"],
        num_phases,
        HMDB_GENERIC_SCENES,
        max_words=7,
    )
    return {
        "label_text": label_text,
        "action_anchor": anchor,
        "entity_priors": _filtered_entities(class_name, record.get("entity_priors", []), max_items=3, drop_person_terms=True),
        "scene_priors": scenes,
        "attribute_cues": _filtered_attribute_cues(class_name, record.get("attribute_cues", []), HMDB_ATTRIBUTE_HINTS, max_items=4),
        "phase_cues": phases,
        "confusion_cues": _filtered_confusions(record.get("confusion_cues", []), max_items=2),
    }


def _postprocess_ucf101_structured_v2(class_name: str, record: Dict[str, Any], num_phases: int) -> Dict[str, Any]:
    label_text = _normalize_label_text(record.get("label_text") or _heuristic_label_text(class_name), class_name).lower()
    anchor = _clean_anchor_text(record.get("action_anchor") or _humanize_class_name(class_name), UCF_GENERIC_SCENES, max_words=14)
    scenes = _filtered_scene_priors(
        class_name,
        record.get("scene_priors", []),
        UCF_GENERIC_SCENES,
        UCF_STABLE_SCENE_HINTS,
        {"court", "pool", "beam", "range", "ring", "board", "track", "bar", "horse", "gym"},
        max_items=2,
    )
    phases = _clean_phase_cues(
        record.get("phase_cues", []),
        ["ready stance", "main action", "result posture"],
        num_phases,
        UCF_GENERIC_SCENES,
        max_words=7,
    )
    return {
        "label_text": label_text,
        "action_anchor": anchor,
        "entity_priors": _filtered_entities(class_name, record.get("entity_priors", []), max_items=4, drop_person_terms=True),
        "scene_priors": scenes,
        "attribute_cues": _filtered_attribute_cues(class_name, record.get("attribute_cues", []), UCF_ATTRIBUTE_HINTS, max_items=4),
        "phase_cues": phases,
        "confusion_cues": _filtered_confusions(record.get("confusion_cues", []), max_items=2),
    }


def _postprocess_kinetics100_structured_v2(class_name: str, record: Dict[str, Any], num_phases: int) -> Dict[str, Any]:
    label_text = _normalize_label_text(record.get("label_text") or _heuristic_label_text(class_name), class_name).lower()
    anchor = _clean_anchor_text(record.get("action_anchor") or _humanize_class_name(class_name), K100_GENERIC_SCENES, max_words=14)
    scenes = _filtered_scene_priors(
        class_name,
        record.get("scene_priors", []),
        K100_GENERIC_SCENES,
        K100_STABLE_SCENE_HINTS,
        {"court", "pool", "slope", "track", "ring", "rink", "gym"},
        max_items=2,
    )
    phases = _clean_phase_cues(
        record.get("phase_cues", []),
        ["action begins", "main interaction", "action result"],
        num_phases,
        K100_GENERIC_SCENES,
        max_words=7,
    )
    attributes = _filtered_attribute_cues(class_name, record.get("attribute_cues", []), K100_ATTRIBUTE_HINTS, max_items=4)
    lower = _humanize_class_name(class_name).lower()
    if not re.search(
        r"\bair drumming\b|\bplaying accordion\b|\bplaying didgeridoo\b|\bplaying keyboard\b|"
        r"\bplaying ukulele\b|\bplaying xylophone\b|\bplaying drums\b|\bplaying trumpet\b",
        lower,
    ):
        attributes = [cue for cue in attributes if cue != "instrument playing"]
    return {
        "label_text": label_text,
        "action_anchor": anchor,
        "entity_priors": _filtered_entities(class_name, record.get("entity_priors", []), max_items=4, drop_person_terms=True),
        "scene_priors": scenes,
        "attribute_cues": attributes,
        "phase_cues": phases,
        "confusion_cues": _filtered_confusions(record.get("confusion_cues", []), max_items=2),
    }


def _heuristic_label_text(class_name: str) -> str:
    base = _humanize_class_name(class_name)
    words = base.split()
    if len(words) <= 6:
        return base
    return " ".join(words[:6])


def _normalize_label_text(label_text: str, class_name: str) -> str:
    text = _normalize_text(label_text or _heuristic_label_text(class_name))
    if not text:
        text = _heuristic_label_text(class_name)

    special_cases = {
        "handstand": "handstand pose",
        "kiss": "kissing action",
        "kayaking": "kayaking action",
        "beatboxing": "beatboxing performance",
        "breakdancing": "breakdancing routine",
        "paragliding": "paragliding flight",
    }
    low = text.lower()
    if low in special_cases:
        return special_cases[low]

    words = text.split()
    if len(words) == 1:
        word = words[0]
        if word.endswith("ing"):
            return f"{word} action"
        return f"{word} action"
    if len(words) > 6:
        return " ".join(words[:6])
    return text


def _normalize_structured_v1_entry(class_name: str, entry: Any, num_phases: int) -> Dict[str, Any]:
    if isinstance(entry, dict):
        anchor = _normalize_text(
            entry.get("action_anchor")
            or entry.get("Action Label")
            or entry.get("label")
            or entry.get("action")
            or f"A person performs {_humanize_class_name(class_name)}."
        )
        key_entities = _coerce_list(
            entry.get("key_entities") or entry.get("Scene Description") or entry.get("entities")
        )[:6]
        motion_phases = _coerce_list(
            entry.get("motion_phases")
            or entry.get("Sub-Action Description")
            or entry.get("sub_actions")
            or entry.get("phase_cues")
        )
        disambiguation = _coerce_list(
            entry.get("disambiguation")
            or entry.get("negative_cues")
            or entry.get("confusion_cues")
        )[:2]
    else:
        seed_text = _normalize_text(entry)
        anchor = ""
        sentences = _split_sentences(seed_text)
        if sentences:
            anchor = sentences[0]
        if not anchor:
            anchor = f"A person performs {_humanize_class_name(class_name)}."
        key_entities = _extract_entities(class_name, seed_text)
        if len(sentences) >= num_phases:
            if num_phases == 3 and len(sentences) >= 3:
                motion_phases = [
                    sentences[0],
                    " ".join(sentences[1:-1]) or sentences[1],
                    sentences[-1],
                ]
            else:
                indices = [
                    round(i * (len(sentences) - 1) / max(1, num_phases - 1))
                    for i in range(num_phases)
                ]
                motion_phases = [sentences[idx] for idx in indices]
        else:
            motion_phases = _default_phase_texts(class_name, anchor, num_phases)
        disambiguation = [
            sentence for sentence in sentences
            if "unlike " in sentence.lower() or "rather than" in sentence.lower()
        ][:2]

    if not anchor:
        anchor = f"A person performs {_humanize_class_name(class_name)}."
    if not key_entities:
        key_entities = _extract_entities(class_name, anchor)
    if not motion_phases:
        motion_phases = _default_phase_texts(class_name, anchor, num_phases)
    if len(motion_phases) < num_phases:
        motion_phases = motion_phases + [motion_phases[-1]] * (num_phases - len(motion_phases))

    return {
        "action_anchor": anchor,
        "key_entities": _dedupe_keep_order(key_entities, max_items=6),
        "motion_phases": motion_phases[:num_phases],
        "disambiguation": _dedupe_keep_order(disambiguation, max_items=2),
    }


def _normalize_structured_v2_entry(class_name: str, entry: Any, num_phases: int) -> Dict[str, Any]:
    if isinstance(entry, dict):
        if "label_text" in entry or "entity_priors" in entry or "phase_cues" in entry:
            label_text = _normalize_text(entry.get("label_text") or _heuristic_label_text(class_name))
            action_anchor = _normalize_text(
                entry.get("action_anchor")
                or entry.get("label")
                or f"A person performs {_humanize_class_name(class_name)}."
            )
            entity_priors = _coerce_list(entry.get("entity_priors"))[:6]
            scene_priors = _coerce_list(entry.get("scene_priors"))[:4]
            attribute_cues = _coerce_list(entry.get("attribute_cues"))[:4]
            phase_cues = _coerce_list(entry.get("phase_cues"))
            confusion_cues = _coerce_list(entry.get("confusion_cues"))[:2]
        else:
            v1 = _normalize_structured_v1_entry(class_name, entry, num_phases)
            seed_text = " ".join(
                [v1["action_anchor"]]
                + v1["motion_phases"]
                + v1["disambiguation"]
            )
            label_text = _heuristic_label_text(class_name)
            action_anchor = v1["action_anchor"]
            entity_priors = v1["key_entities"]
            scene_priors = _extract_scene_priors(class_name, seed_text)
            attribute_cues = _extract_attribute_cues(class_name, seed_text)
            phase_cues = v1["motion_phases"]
            confusion_cues = v1["disambiguation"]
    else:
        seed_text = _normalize_text(entry)
        sentences = _split_sentences(seed_text)
        label_text = _heuristic_label_text(class_name)
        action_anchor = sentences[0] if sentences else f"A person performs {_humanize_class_name(class_name)}."
        entity_priors = _extract_entities(class_name, seed_text)
        scene_priors = _extract_scene_priors(class_name, seed_text)
        attribute_cues = _extract_attribute_cues(class_name, seed_text)
        if len(sentences) >= num_phases:
            if num_phases == 3 and len(sentences) >= 3:
                phase_cues = [
                    sentences[0],
                    " ".join(sentences[1:-1]) or sentences[1],
                    sentences[-1],
                ]
            else:
                indices = [
                    round(i * (len(sentences) - 1) / max(1, num_phases - 1))
                    for i in range(num_phases)
                ]
                phase_cues = [sentences[idx] for idx in indices]
        else:
            phase_cues = _default_phase_texts(class_name, action_anchor, num_phases)
        confusion_cues = [
            sentence for sentence in sentences
            if "unlike " in sentence.lower() or "rather than" in sentence.lower()
        ][:2]

    if not label_text:
        label_text = _heuristic_label_text(class_name)
    if not action_anchor:
        action_anchor = f"A person performs {_humanize_class_name(class_name)}."
    if not entity_priors:
        entity_priors = _extract_entities(class_name, action_anchor)
    if len(phase_cues) < num_phases:
        phase_cues = phase_cues + [phase_cues[-1]] * (num_phases - len(phase_cues))

    return {
        "label_text": label_text,
        "action_anchor": action_anchor,
        "entity_priors": _dedupe_keep_order(entity_priors, max_items=6),
        "scene_priors": _dedupe_keep_order(scene_priors, max_items=4),
        "attribute_cues": _dedupe_keep_order(attribute_cues, max_items=4),
        "phase_cues": phase_cues[:num_phases],
        "confusion_cues": _dedupe_keep_order(confusion_cues, max_items=2),
    }


def _normalize_structured_entry(
    class_name: str,
    entry: Any,
    num_phases: int,
    output_format: str = "structured_v1",
    dataset: str = "",
) -> Dict[str, Any]:
    if output_format == "structured_v2":
        normalized = _normalize_structured_v2_entry(class_name, entry, num_phases)
        dataset_name = _dataset_key(dataset)
        if _is_ssv2_dataset(dataset=dataset, class_name=class_name):
            normalized = _postprocess_ssv2_structured_v2(class_name, normalized, num_phases)
        elif dataset_name == "hmdb51":
            normalized = _postprocess_hmdb51_structured_v2(class_name, normalized, num_phases)
        elif dataset_name == "ucf101":
            normalized = _postprocess_ucf101_structured_v2(class_name, normalized, num_phases)
        elif dataset_name == "kinetics100":
            normalized = _postprocess_kinetics100_structured_v2(class_name, normalized, num_phases)
        return normalized
    return _normalize_structured_v1_entry(class_name, entry, num_phases)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _parse_structured_response(
    content: str,
    class_name: str,
    num_phases: int,
    output_format: str = "structured_v1",
    dataset: str = "",
) -> Dict[str, Any]:
    raw = _strip_code_fences(content)
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and class_name in payload and isinstance(payload[class_name], dict):
            payload = payload[class_name]
        return _normalize_structured_entry(
            class_name,
            payload,
            num_phases,
            output_format=output_format,
            dataset=dataset,
        )
    except Exception:
        return _normalize_structured_entry(
            class_name,
            raw,
            num_phases,
            output_format=output_format,
            dataset=dataset,
        )


def generate_structured_description(
    class_name: str,
    seed_text: str = "",
    num_phases: int = 3,
    output_format: str = "structured_v1",
    dataset: str = "",
) -> Dict[str, Any]:
    """Generate a structured description entry from seed text or class name."""
    return _normalize_structured_entry(
        class_name,
        seed_text,
        num_phases,
        output_format=output_format,
        dataset=dataset,
    )


def _load_seed_descriptions(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"seed descriptions file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"seed descriptions file must be a JSON object mapping class -> description: {path}")
    return obj


def _build_normalized_seed_map(seed_descriptions: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in seed_descriptions.items():
        norm = _normalize_lookup_key(key)
        if norm and norm not in normalized:
            normalized[norm] = value
    return normalized


def _build_normalized_key_index(mapping: Dict[str, Any]) -> Dict[str, str]:
    normalized = {}
    for key in mapping.keys():
        norm = _normalize_lookup_key(key)
        if norm and norm not in normalized:
            normalized[norm] = key
    return normalized


def _load_alias_manifest(path: str) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"alias manifest file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"alias manifest must be a JSON object: {path}")
    manifest = {}
    for alias, canonical in obj.items():
        if not _normalize_text(alias) or not _normalize_text(canonical):
            continue
        manifest[_normalize_text(alias)] = _normalize_text(canonical)
        manifest[_normalize_lookup_key(alias)] = _normalize_text(canonical)
    return manifest


def _resolve_alias_name(class_name: str, alias_manifest: Dict[str, str]) -> str:
    if not alias_manifest:
        return class_name
    return (
        alias_manifest.get(class_name)
        or alias_manifest.get(_normalize_text(class_name))
        or alias_manifest.get(_normalize_lookup_key(class_name))
        or class_name
    )


def _derive_split_descriptions(
    full_descriptions: Dict[str, Any],
    class_names: List[str],
    output_format: str,
    num_phases: int,
    dataset: str = "",
    alias_manifest: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    alias_manifest = alias_manifest or {}
    normalized_full = _build_normalized_key_index(full_descriptions)
    split_descriptions: Dict[str, Any] = {}
    stats = {
        "exact_hits": 0,
        "alias_hits": 0,
        "normalized_hits": 0,
        "fallback_hits": 0,
        "fallback_examples": [],
    }

    for class_name in class_names:
        entry = None
        aliased = _resolve_alias_name(class_name, alias_manifest)
        if class_name in full_descriptions:
            entry = full_descriptions[class_name]
            stats["exact_hits"] += 1
        elif aliased in full_descriptions:
            entry = full_descriptions[aliased]
            stats["alias_hits"] += 1
        else:
            norm_key = _normalize_lookup_key(aliased)
            matched_key = normalized_full.get(norm_key)
            if matched_key is not None:
                entry = full_descriptions[matched_key]
                stats["normalized_hits"] += 1

        if entry is None:
            stats["fallback_hits"] += 1
            if len(stats["fallback_examples"]) < 10:
                stats["fallback_examples"].append(class_name)
            split_descriptions[class_name] = generate_structured_description(
                class_name,
                seed_text=generate_detailed_description(class_name),
                num_phases=num_phases,
                output_format=output_format,
                dataset=dataset,
            ) if output_format in STRUCTURED_OUTPUT_FORMATS else generate_detailed_description(class_name)
            continue

        if output_format in STRUCTURED_OUTPUT_FORMATS:
            split_descriptions[class_name] = _normalize_structured_entry(
                class_name,
                entry,
                num_phases,
                output_format=output_format,
                dataset=dataset,
            )
        elif isinstance(entry, dict):
            split_descriptions[class_name] = _normalize_text(
                entry.get("action_anchor") or entry.get("label_text") or class_name
            )
        else:
            split_descriptions[class_name] = _normalize_text(entry)
    return split_descriptions, stats


def _save_json(data: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _render_prompt_template(template: str, class_name: str, num_phases: int) -> str:
    """
    Render prompt templates while allowing raw JSON braces in external prompt files.

    Supports both:
    - escaped braces used by in-code templates: {{ ... }}
    - raw braces used by prompt files: { ... }
    """
    rendered = template.replace("{{", "__LBRACE__").replace("}}", "__RBRACE__")
    rendered = rendered.replace("{class_name}", str(class_name))
    rendered = rendered.replace("{num_phases}", str(num_phases))
    rendered = rendered.replace("__LBRACE__", "{").replace("__RBRACE__", "}")
    return rendered


def _extract_text_from_chat_response(response: Any) -> str:
    """Extract plain text from OpenAI-compatible chat response."""
    try:
        choice = response.choices[0]
        message = choice.message
    except Exception:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return _normalize_text(content)

    # Some providers may return content as a list of typed parts.
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "output_text"):
                    txt = item.get("text") or item.get("value")
                    if txt:
                        parts.append(str(txt))
            else:
                txt = getattr(item, "text", None) or getattr(item, "value", None)
                if txt:
                    parts.append(str(txt))
        if parts:
            return _normalize_text(" ".join(parts))

    return ""


def _extract_reasoning_text_from_chat_response(response: Any) -> str:
    """Extract provider-specific reasoning content when available."""
    try:
        choice = response.choices[0]
        message = choice.message
    except Exception:
        return ""

    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str):
        return _normalize_text(reasoning_content)

    if isinstance(reasoning_content, list):
        parts = []
        for item in reasoning_content:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("value")
                if txt:
                    parts.append(str(txt))
            else:
                txt = getattr(item, "text", None) or getattr(item, "value", None)
                if txt:
                    parts.append(str(txt))
        if parts:
            return _normalize_text(" ".join(parts))

    return ""


def generate_llm_description(
    class_name: str,
    model: str = "deepseek-chat",
    temperature: float = 0.2,
    max_tokens: int = 320,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    output_format: str = "legacy",
    num_phases: int = 3,
    proxy_url: str = "",
    openai_base_url: str = "",
    timeout_sec: float = 30.0,
    retries: int = 2,
    ca_bundle: str = "",
    insecure_ssl: bool = False,
    dataset: str = "",
) -> Any:
    """Generate description using OpenAI API (supports old/new SDKs)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if not hasattr(generate_llm_description, "_warned_missing_key"):
            print("[LLM] OPENAI_API_KEY is not set. Falling back to rule-based descriptions.")
            generate_llm_description._warned_missing_key = True
        if output_format in STRUCTURED_OUTPUT_FORMATS:
            return generate_structured_description(
                class_name,
                seed_text=generate_detailed_description(class_name),
                num_phases=num_phases,
                output_format=output_format,
                dataset=dataset,
            )
        return generate_detailed_description(class_name)

    prompt = _render_prompt_template(user_prompt_template, class_name, num_phases)
    ssl_verify = _resolve_ssl_verify(ca_bundle=ca_bundle, insecure_ssl=insecure_ssl)
    is_reasoner = "deepseek-reasoner" in str(model).lower()
    if (
        is_reasoner
        and output_format in STRUCTURED_OUTPUT_FORMATS
        and (
            (output_format == "structured_v1"
             and system_prompt == DEFAULT_STRUCTURED_SYSTEM_PROMPT
             and user_prompt_template == DEFAULT_STRUCTURED_USER_PROMPT_TEMPLATE)
            or
            (output_format == "structured_v2"
             and system_prompt == DEFAULT_STRUCTURED_V2_SYSTEM_PROMPT
             and user_prompt_template == DEFAULT_STRUCTURED_V2_USER_PROMPT_TEMPLATE)
        )
    ):
        if output_format == "structured_v2":
            system_prompt = DEFAULT_REASONER_STRUCTURED_V2_SYSTEM_PROMPT
            user_prompt_template = DEFAULT_REASONER_STRUCTURED_V2_USER_PROMPT_TEMPLATE
        else:
            system_prompt = DEFAULT_REASONER_STRUCTURED_SYSTEM_PROMPT
            user_prompt_template = DEFAULT_REASONER_STRUCTURED_USER_PROMPT_TEMPLATE
        prompt = _render_prompt_template(user_prompt_template, class_name, num_phases)

    reasoner_min_tokens = 4096 if output_format in STRUCTURED_OUTPUT_FORMATS else 2048
    reasoner_max_tokens = 8192 if output_format in STRUCTURED_OUTPUT_FORMATS else 4096

    if is_reasoner and max_tokens < reasoner_min_tokens:
        if not hasattr(generate_llm_description, "_warned_reasoner_tokens"):
            print(
                "[LLM] deepseek-reasoner usually needs a larger max_tokens budget. "
                f"Auto-adjusting max_tokens to {reasoner_min_tokens}."
            )
            generate_llm_description._warned_reasoner_tokens = True
        max_tokens = reasoner_min_tokens
    if is_reasoner and not hasattr(generate_llm_description, "_warned_reasoner_temp"):
        print(
            "[LLM] deepseek-reasoner: temperature/top_p generally have no effect. "
            "Keeping provided value for compatibility."
        )
        generate_llm_description._warned_reasoner_temp = True

    # New SDK path (openai>=1.0)
    for attempt in range(max(1, retries + 1)):
        current_max_tokens = max_tokens
        if is_reasoner:
            # Reasoning models may consume many completion tokens before final answer.
            current_max_tokens = min(max_tokens * (2 ** attempt), reasoner_max_tokens)

        custom_http_client = None
        try:
            from openai import OpenAI
            import httpx

            client_kwargs = {
                "api_key": api_key,
                "timeout": timeout_sec,
            }
            if openai_base_url:
                client_kwargs["base_url"] = openai_base_url

            if proxy_url or ssl_verify is not True:
                http_client_kwargs = {
                    "timeout": timeout_sec,
                    "verify": ssl_verify,
                }
                if proxy_url:
                    http_client_kwargs["proxy"] = proxy_url
                custom_http_client = httpx.Client(**http_client_kwargs)
                client_kwargs["http_client"] = custom_http_client

            client = OpenAI(**client_kwargs)
            request_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": current_max_tokens,
            }
            if not is_reasoner:
                request_kwargs["temperature"] = temperature
            if output_format in STRUCTURED_OUTPUT_FORMATS and "deepseek" in (
                f"{openai_base_url} {model}".lower()
            ):
                request_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**request_kwargs)
            content = _extract_text_from_chat_response(response)
            if content:
                if output_format in STRUCTURED_OUTPUT_FORMATS:
                    return _parse_structured_response(
                        content,
                        class_name,
                        num_phases,
                        output_format=output_format,
                        dataset=dataset,
                    )
                return content

            finish_reason = None
            reasoning_content = ""
            try:
                finish_reason = response.choices[0].finish_reason
            except Exception:
                pass
            try:
                reasoning_content = _extract_reasoning_text_from_chat_response(response)
            except Exception:
                reasoning_content = ""
            print(
                f"[LLM] Empty content for {class_name} "
                f"(attempt {attempt + 1}/{max(1, retries + 1)}, "
                f"finish_reason={finish_reason}, max_tokens={current_max_tokens}, "
                f"reasoning_chars={len(reasoning_content)})."
            )
        except Exception as e_new:
            print(
                f"[LLM] New OpenAI SDK path failed for {class_name}: {e_new} "
                f"(proxy={proxy_url or 'env'}, base_url={openai_base_url or 'default'}, "
                f"verify={'off' if ssl_verify is False else ('custom_ca' if isinstance(ssl_verify, str) else 'default')}, "
                f"attempt={attempt + 1}/{max(1, retries + 1)}, max_tokens={current_max_tokens})"
            )
        finally:
            if custom_http_client is not None:
                custom_http_client.close()

        if attempt < max(1, retries + 1) - 1:
            time.sleep(min(2.0, 0.5 * (attempt + 1)))

    # Legacy SDK path (openai<1.0). Skip if ChatCompletion is unavailable.
    try:
        import openai
        version = getattr(openai, "__version__", "0")
        try:
            major = int(str(version).split(".")[0])
        except Exception:
            major = 0
        if major >= 1:
            raise RuntimeError("Skipping legacy ChatCompletion path because openai>=1.")
        if not hasattr(openai, "ChatCompletion"):
            raise RuntimeError("Legacy ChatCompletion API not available in current openai package.")
        openai.api_key = api_key

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        message = response.choices[0].message
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = message.content
        if content is not None and str(content).strip() != "":
            if output_format in STRUCTURED_OUTPUT_FORMATS:
                return _parse_structured_response(
                    str(content),
                    class_name,
                    num_phases,
                    output_format=output_format,
                    dataset=dataset,
                )
            return " ".join(str(content).split())
    except Exception as e_old:
        print(f"[LLM] Legacy OpenAI SDK path skipped/failed for {class_name}: {e_old}")

    print(f"[LLM] Fallback to rule-based description for {class_name}")
    if output_format in STRUCTURED_OUTPUT_FORMATS:
        return generate_structured_description(
            class_name,
            seed_text=generate_detailed_description(class_name),
            num_phases=num_phases,
            output_format=output_format,
            dataset=dataset,
        )
    return generate_detailed_description(class_name)


def generate_descriptions(
    class_names: List[str],
    method: str = "template",
    output_format: str = "legacy",
    num_phases: int = 3,
    llm_model: str = "deepseek-chat",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 320,
    llm_retries: int = 2,
    sleep_sec: float = 0.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    proxy_url: str = "",
    openai_base_url: str = "",
    timeout_sec: float = 30.0,
    ca_bundle: str = "",
    insecure_ssl: bool = False,
    seed_descriptions: Optional[Dict[str, Any]] = None,
    alias_manifest: Optional[Dict[str, str]] = None,
    dataset: str = "",
) -> Dict[str, Any]:
    """Generate descriptions for all classes."""
    descriptions = {}
    normalized_seed_descriptions = _build_normalized_seed_map(seed_descriptions or {})
    alias_manifest = alias_manifest or {}
    
    for i, class_name in enumerate(class_names):
        print(f"Processing {i+1}/{len(class_names)}: {class_name}")
        seed_entry = None
        if seed_descriptions:
            seed_entry = seed_descriptions.get(class_name)
            if seed_entry is None:
                alias_name = _resolve_alias_name(class_name, alias_manifest)
                seed_entry = seed_descriptions.get(alias_name)
            if seed_entry is None:
                seed_entry = normalized_seed_descriptions.get(
                    _normalize_lookup_key(_resolve_alias_name(class_name, alias_manifest))
                )
        if seed_entry is not None:
            if output_format in STRUCTURED_OUTPUT_FORMATS:
                descriptions[class_name] = _normalize_structured_entry(
                    class_name,
                    seed_entry,
                    num_phases,
                    output_format=output_format,
                    dataset=dataset,
                )
            else:
                if isinstance(seed_entry, dict):
                    descriptions[class_name] = _normalize_structured_entry(
                        class_name,
                        seed_entry,
                        num_phases,
                        output_format="structured_v1",
                        dataset=dataset,
                    )["action_anchor"]
                else:
                    descriptions[class_name] = _normalize_text(seed_entry)
            continue
        
        if method == "llm":
            desc = generate_llm_description(
                class_name,
                model=llm_model,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                output_format=output_format,
                num_phases=num_phases,
                proxy_url=proxy_url,
                openai_base_url=openai_base_url,
                timeout_sec=timeout_sec,
                retries=llm_retries,
                ca_bundle=ca_bundle,
                insecure_ssl=insecure_ssl,
                dataset=dataset,
            )
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        elif method == "detailed":
            if output_format in STRUCTURED_OUTPUT_FORMATS:
                desc = generate_structured_description(
                    class_name,
                    seed_text=generate_detailed_description(class_name),
                    num_phases=num_phases,
                    output_format=output_format,
                    dataset=dataset,
                )
            else:
                desc = generate_detailed_description(class_name)
        else:  # template
            if output_format in STRUCTURED_OUTPUT_FORMATS:
                desc = generate_structured_description(
                    class_name,
                    seed_text=generate_template_description(class_name),
                    num_phases=num_phases,
                    output_format=output_format,
                    dataset=dataset,
                )
            else:
                desc = generate_template_description(class_name)
        
        descriptions[class_name] = desc
    
    return descriptions


def main():
    parser = argparse.ArgumentParser(description="Generate action descriptions for DiSMo")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["kinetics100", "ssv2", "hmdb51", "ucf101"],
                        help="Dataset to generate descriptions for (used as fallback class list source)")
    parser.add_argument("--class_names_file", type=str, default="",
                        help="Optional class list file (.txt/.json). Recommended for full dataset coverage.")
    parser.add_argument("--method", type=str, default="detailed",
                        choices=["template", "detailed", "llm"],
                        help="Generation method")
    parser.add_argument("--output_format", type=str, default="legacy",
                        choices=["legacy", "structured_v1", "structured_v2"],
                        help="Output schema for saved descriptions")
    parser.add_argument("--num_phases", type=int, default=3,
                        help="Number of motion phases for structured output")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--train_class_names_file", type=str, default="",
                        help="Optional train split class file for deriving train split view from canonical full output.")
    parser.add_argument("--test_class_names_file", type=str, default="",
                        help="Optional test split class file for deriving test split view from canonical full output.")
    parser.add_argument("--output_train", type=str, default="",
                        help="Optional output path for derived train split descriptions.")
    parser.add_argument("--output_test", type=str, default="",
                        help="Optional output path for derived test split descriptions.")
    parser.add_argument("--alias_manifest_file", type=str, default="",
                        help="Optional JSON mapping alias -> canonical class name for seed/split resolution.")
    parser.add_argument("--llm_model", type=str, default="deepseek-chat",
                        help="LLM model to use (if method=llm)")
    parser.add_argument("--llm_temperature", type=float, default=0.2,
                        help="LLM sampling temperature (if method=llm)")
    parser.add_argument("--llm_max_tokens", type=int, default=320,
                        help="LLM max output tokens (if method=llm)")
    parser.add_argument("--llm_retries", type=int, default=2,
                        help="Retry attempts for each class in LLM mode")
    parser.add_argument("--sleep_sec", type=float, default=0.0,
                        help="Sleep seconds between LLM calls to avoid rate limits")
    parser.add_argument("--system_prompt", type=str, default="",
                        help="System prompt for LLM generation")
    parser.add_argument("--prompt_template_file", type=str, default="",
                        help="Optional file containing user prompt template. Use {class_name} placeholder.")
    parser.add_argument("--seed_descriptions_file", type=str, default="",
                        help="Optional existing JSON mapping class -> description to convert/reuse.")
    parser.add_argument("--proxy_url", type=str, default="",
                        help="Explicit proxy URL, e.g. http://127.0.0.1:8888")
    parser.add_argument("--ca_bundle", type=str, default=os.getenv("REQUESTS_CA_BUNDLE", ""),
                        help="Path to custom CA bundle for TLS verification (proxy/self-signed cert).")
    parser.add_argument("--insecure_ssl", action="store_true",
                        help="Disable TLS certificate verification (unsafe; debug only).")
    parser.add_argument("--openai_base_url", type=str, default=os.getenv("OPENAI_BASE_URL", ""),
                        help="Optional OpenAI-compatible base URL")
    parser.add_argument("--timeout_sec", type=float, default=30.0,
                        help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # 加载类别名（优先使用显式文件）
    if args.class_names_file:
        class_names = load_class_names_from_file(args.class_names_file)
        print(f"Loaded {len(class_names)} classes from file: {args.class_names_file}")
    else:
        class_names = load_class_names(args.dataset)
        print(f"Loaded {len(class_names)} classes from builtin list for {args.dataset}")

    # 去重并保持顺序
    seen = set()
    class_names = [x for x in class_names if not (x in seen or seen.add(x))]
    print(f"Loaded {len(class_names)} classes for {args.dataset}")
    if len(class_names) == 0:
        raise ValueError("No class names found. Provide --class_names_file or check --dataset.")

    if args.output_format == "structured_v1":
        user_prompt_template = DEFAULT_STRUCTURED_USER_PROMPT_TEMPLATE
    elif args.output_format == "structured_v2":
        user_prompt_template = DEFAULT_STRUCTURED_V2_USER_PROMPT_TEMPLATE
    else:
        user_prompt_template = DEFAULT_USER_PROMPT_TEMPLATE
    if args.prompt_template_file:
        user_prompt_template = load_prompt_template(args.prompt_template_file)
    system_prompt = (
        args.system_prompt
        or (
            DEFAULT_STRUCTURED_SYSTEM_PROMPT
            if args.output_format == "structured_v1"
            else (
                DEFAULT_STRUCTURED_V2_SYSTEM_PROMPT
                if args.output_format == "structured_v2"
                else DEFAULT_SYSTEM_PROMPT
            )
        )
    )
    seed_descriptions = None
    if args.seed_descriptions_file:
        seed_descriptions = _load_seed_descriptions(args.seed_descriptions_file)
    alias_manifest = _load_alias_manifest(args.alias_manifest_file) if args.alias_manifest_file else {}
    
    # 生成描述
    descriptions = generate_descriptions(
        class_names,
        method=args.method,
        output_format=args.output_format,
        num_phases=args.num_phases,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_retries=args.llm_retries,
        sleep_sec=args.sleep_sec,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        proxy_url=args.proxy_url,
        openai_base_url=args.openai_base_url,
        timeout_sec=args.timeout_sec,
        ca_bundle=args.ca_bundle,
        insecure_ssl=args.insecure_ssl,
        seed_descriptions=seed_descriptions,
        alias_manifest=alias_manifest,
        dataset=args.dataset,
    )
    _save_json(descriptions, args.output)
    print(f"Saved {len(descriptions)} descriptions to {args.output}")

    if args.output_train:
        if not args.train_class_names_file:
            raise ValueError("--output_train requires --train_class_names_file")
        train_class_names = load_class_names_from_file(args.train_class_names_file)
        train_descriptions, train_stats = _derive_split_descriptions(
            descriptions,
            train_class_names,
            output_format=args.output_format,
            num_phases=args.num_phases,
            dataset=args.dataset,
            alias_manifest=alias_manifest,
        )
        _save_json(train_descriptions, args.output_train)
        print(
            f"Saved {len(train_descriptions)} train split descriptions to {args.output_train} "
            f"(exact={train_stats['exact_hits']}, alias={train_stats['alias_hits']}, "
            f"normalized={train_stats['normalized_hits']}, fallback={train_stats['fallback_hits']})"
        )
        if train_stats["fallback_examples"]:
            print("[Split] Train fallback examples: " + ", ".join(train_stats["fallback_examples"]))

    if args.output_test:
        if not args.test_class_names_file:
            raise ValueError("--output_test requires --test_class_names_file")
        test_class_names = load_class_names_from_file(args.test_class_names_file)
        test_descriptions, test_stats = _derive_split_descriptions(
            descriptions,
            test_class_names,
            output_format=args.output_format,
            num_phases=args.num_phases,
            dataset=args.dataset,
            alias_manifest=alias_manifest,
        )
        _save_json(test_descriptions, args.output_test)
        print(
            f"Saved {len(test_descriptions)} test split descriptions to {args.output_test} "
            f"(exact={test_stats['exact_hits']}, alias={test_stats['alias_hits']}, "
            f"normalized={test_stats['normalized_hits']}, fallback={test_stats['fallback_hits']})"
        )
        if test_stats["fallback_examples"]:
            print("[Split] Test fallback examples: " + ", ".join(test_stats["fallback_examples"]))


if __name__ == "__main__":
    main()
