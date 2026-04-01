#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
CNN_DiSMo: Main Few-Shot Action Recognition Model

DiSMo = DINOv2 + Semantic + Motion
Combines:
1. DINOv2 ViT backbone for powerful visual features
2. LLM-based semantic knowledge injection
3. MoLo-style motion modeling with long-short contrastive
4. HyRSM-style hybrid relation module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import torchvision.models as tv_models

from models.base.base_blocks import HEAD_REGISTRY
from models.base.dinov2_backbone import DINOv2Backbone
from models.base.motion_module import MotionModule
from models.base.semantic_module import (
    SemanticModule,
    EpisodeSFM,
    ClassSFM,
    PhaseAwareClassSFM,
)


def extract_class_indices(labels, which_class):
    """Helper to extract indices of elements with specified label."""
    class_mask = torch.eq(labels, which_class)
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
    return torch.reshape(class_mask_indices, (-1,))


def cos_sim(x, y, epsilon=0.01):
    """Cosine similarity between last dimensions of two tensors."""
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    return torch.div(numerator, denominator)


def otam_distance(query_feat, support_feat):
    """
    Ordered Temporal Alignment Module (OTAM) distance.

    Uses Dynamic Time Warping (DTW) with cumulative minimum to
    compute an order-preserving alignment distance between two frame
    sequences.  Critical for temporally sensitive datasets like SSV2
    where action direction matters (e.g. "opening" vs "closing").

    Args:
        query_feat:   [T, D] query frame features (L2-normalised)
        support_feat: [T, D] support frame features (L2-normalised)

    Returns:
        distance: scalar, normalised DTW alignment cost
    """
    T_q = query_feat.shape[0]
    T_s = support_feat.shape[0]

    # Frame-wise cosine distance matrix  [T_s, T_q]
    dist = 1.0 - torch.mm(
        F.normalize(support_feat, dim=-1),
        F.normalize(query_feat, dim=-1).t()
    )

    # --- Vectorised DTW via anti-diagonal sweep -------------------------
    # D[i, j] = dist[i, j] + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    # Out-of-place to preserve autograd graph.
    first_row = [dist[0, 0]]
    for j in range(1, T_q):
        first_row.append(first_row[j - 1] + dist[0, j])
    prev = torch.stack(first_row)   # [T_q]

    for i in range(1, T_s):
        cols = [prev[0] + dist[i, 0]]
        for j in range(1, T_q):
            cols.append(dist[i, j] + torch.minimum(
                prev[j], torch.minimum(cols[j - 1], prev[j - 1])
            ))
        prev = torch.stack(cols)    # [T_q]

    return prev[-1] / (T_s + T_q)   # normalise by path length


def batch_otam_distance(support_feat, query_feat):
    """
    Batched OTAM distance between support and query frame sequences.

    Args:
        support_feat: [Q, S, T, D]  support frame features
        query_feat:   [Q, T, D]     query frame features

    Returns:
        cum_dists: [Q, S]  OTAM distance for each (query, support) pair
    """
    Q, S, T_s, D = support_feat.shape
    T_q = query_feat.shape[1]

    # Cosine distance: (query vs each support video)
    # support_feat: [Q, S, T_s, D],  query_feat: [Q, T_q, D]
    q_norm = F.normalize(query_feat, dim=-1)          # [Q, T_q, D]
    s_norm = F.normalize(support_feat, dim=-1)         # [Q, S, T_s, D]

    # Reshape to [Q*S, T, D] for batched matmul
    s_flat = rearrange(s_norm, 'q s t d -> (q s) t d')       # [Q*S, T_s, D]
    q_expand = q_norm.unsqueeze(1).expand(-1, S, -1, -1)     # [Q, S, T_q, D]
    q_flat = rearrange(q_expand, 'q s t d -> (q s) t d')     # [Q*S, T_q, D]

    # [Q*S, T_s, T_q]
    dist = 1.0 - torch.bmm(s_flat, q_flat.transpose(1, 2))

    # --- Vectorised DTW (two-row, batched over Q*S) ---------
    # All operations are out-of-place to preserve autograd graph.

    # Initialise first row: D[0, j] = sum(dist[0, 0..j])
    first_row = [dist[:, 0, 0]]                     # list of [QS] tensors
    for j in range(1, T_q):
        first_row.append(first_row[j - 1] + dist[:, 0, j])
    prev = torch.stack(first_row, dim=1)             # [QS, T_q]

    for i in range(1, T_s):
        cols = [prev[:, 0] + dist[:, i, 0]]          # column 0
        for j in range(1, T_q):
            val = dist[:, i, j] + torch.minimum(
                prev[:, j], torch.minimum(cols[j - 1], prev[:, j - 1])
            )
            cols.append(val)
        prev = torch.stack(cols, dim=1)               # [QS, T_q]

    # prev[:, -1] is the DTW cost for each (query, support) pair
    dtw_cost = prev[:, -1] / (T_s + T_q)
    cum_dists = dtw_cost.reshape(Q, S)
    return cum_dists


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model=1024, max_seq_len=20, dropout=0.1, 
                 A_scale=10., B_scale=1.):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.A_scale = A_scale
        self.B_scale = B_scale
        
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * np.sqrt(self.d_model / self.A_scale)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len].detach()
        x = x + self.B_scale * pe
        return self.dropout(x)


class Attention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PreNormAttention(nn.Module):
    """Pre-LayerNorm attention with residual."""
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class MultiHeadAttention(nn.Module):
    """Multi-head attention for cross-attention."""
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.temperature = np.power(d_k, 0.5)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class MotionPhaseRouter(nn.Module):
    """
    Predict soft phase assignments from motion-enhanced temporal features.

    Input:
        x: [B, D, T]
    Output:
        phase_weights: [B, T, P]
    """

    def __init__(self, dim, num_phases=3, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 4, 64)
        self.num_phases = num_phases
        self.router = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, num_phases, kernel_size=1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, x):
        logits = self.router(x)  # [B, P, T]
        return F.softmax(logits.transpose(1, 2), dim=-1)  # [B, T, P]


class _ResNetVideoBackbone(nn.Module):
    """
    Wrapper that turns a torchvision ResNet (2-D, per-frame) into a video
    backbone with the same I/O contract as DINOv2Backbone:

        Input:  [B, C, T, H, W]
        Output: [B, D, T, 1, 1]

    The wrapper strips the final FC layer (keeps through avgpool) and
    processes each frame independently.
    """

    def __init__(self, factory_fn, freeze=False):
        super().__init__()
        # Try loading pretrained weights; fall back to random init if download fails.
        try:
            full_net = factory_fn(weights="IMAGENET1K_V1")
            print("[DiSMo] ResNet backbone: loaded ImageNet pretrained weights")
        except Exception as e:
            import ssl
            # Retry with SSL verification disabled (common in corporate/proxy envs)
            old_ctx = ssl._create_default_https_context
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
                full_net = factory_fn(weights="IMAGENET1K_V1")
                print("[DiSMo] ResNet backbone: loaded ImageNet pretrained weights (SSL unverified)")
            except Exception:
                ssl._create_default_https_context = old_ctx
                full_net = factory_fn(weights=None)
                print(f"[DiSMo] WARNING: Could not download pretrained weights ({e}). "
                      "Using random initialization.")
            finally:
                ssl._create_default_https_context = old_ctx
        # Keep everything up to and including avgpool (removes the fc layer).
        self.encoder = nn.Sequential(*list(full_net.children())[:-1])
        self.freeze = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            print("[DiSMo] ResNet backbone: frozen")
        else:
            print("[DiSMo] ResNet backbone: trainable")

    # ------------------------------------------------------------------
    def forward(self, x):
        """x: [B, C, T, H, W] -> [B, D, T, 1, 1]"""
        if isinstance(x, dict):
            x = x['video']
        if x.dim() == 4:
            # Already [B*T, C, H, W]
            B, C, H, W = x.shape
            T = 1
            frames = x
        elif x.dim() == 5:
            B, C, T, H, W = x.shape
            frames = rearrange(x, 'b c t h w -> (b t) c h w')
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        if self.freeze:
            with torch.no_grad():
                feats = self.encoder(frames)          # [(B*T), D, 1, 1]
        else:
            feats = self.encoder(frames)

        feats = feats.squeeze(-1).squeeze(-1)         # [(B*T), D]
        feats = rearrange(feats, '(b t) d -> b d t', b=B, t=T)
        return feats.unsqueeze(-1).unsqueeze(-1)      # [B, D, T, 1, 1]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self


class _CLIPVideoBackbone(nn.Module):
    """
    Wrapper that turns a CLIP visual encoder into a video backbone
    with the same I/O contract as DINOv2Backbone:

        Input:  [B, C, T, H, W]
        Output: [B, D, T, 1, 1]

    Supports:
    - CLIP ViT-B/16  → 512-d features (after projection)
    - CLIP RN50      → 1024-d features (after attention pooling)

    Standard practice in FSAR: freeze the CLIP visual encoder and
    extract per-frame features.
    """

    # Map user-facing names to (clip model name, output dim)
    CLIP_MODELS = {
        'clip_vitb16': ('ViT-B/16', 512),
        'clip_vitb32': ('ViT-B/32', 512),
        'clip_rn50':   ('RN50',     1024),
        'clip_rn101':  ('RN101',    512),
    }

    def __init__(self, clip_model_name, freeze=True, device='cpu'):
        super().__init__()
        import clip as _clip
        self.freeze = freeze

        print(f"[DiSMo] Loading CLIP visual encoder: {clip_model_name}")
        full_model, _ = _clip.load(clip_model_name, device=device, jit=False)
        # Keep only the visual encoder; discard text encoder to save memory
        self.encoder = full_model.visual
        # CLIP uses float16 by default on GPU; cast to float32 for stable training
        self.encoder = self.encoder.float()

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            print("[DiSMo] CLIP visual encoder: frozen")
        else:
            print("[DiSMo] CLIP visual encoder: trainable")

    def forward(self, x):
        """x: [B, C, T, H, W] -> [B, D, T, 1, 1]"""
        if isinstance(x, dict):
            x = x['video']
        if x.dim() == 4:
            B, C, H, W = x.shape
            T = 1
            frames = x
        elif x.dim() == 5:
            B, C, T, H, W = x.shape
            frames = rearrange(x, 'b c t h w -> (b t) c h w')
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        if self.freeze:
            with torch.no_grad():
                feats = self.encoder(frames.float())  # [(B*T), D]
        else:
            feats = self.encoder(frames.float())

        feats = feats.float()  # ensure float32
        feats = rearrange(feats, '(b t) d -> b d t', b=B, t=T)
        return feats.unsqueeze(-1).unsqueeze(-1)      # [B, D, T, 1, 1]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self


@HEAD_REGISTRY.register()
class CNN_DiSMo(nn.Module):
    """
    DiSMo: DINO + Semantic + Motion for Few-Shot Action Recognition
    
    Key innovations:
    1. DINOv2 ViT-L backbone for powerful visual features
    2. LLM-based semantic knowledge injection via contrastive learning
    3. MoLo-style motion modeling with autodecoder and long-short contrast
    4. HyRSM-style hybrid relation module with gated fusion
    
    Args:
        cfg: Configuration object with model settings
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        
        # ===== 1. Backbone (configurable) =====
        self.backbone, self.mid_dim = self._build_backbone(cfg)
        
        # ===== 2. Motion Module =====
        motion_cfg = cfg.MOTION if hasattr(cfg, 'MOTION') else cfg
        self.use_motion = getattr(motion_cfg, 'ENABLE', True)
        self.phase_router_source = getattr(motion_cfg, 'ROUTER_SOURCE', 'enhanced_motion')
        if self.phase_router_source not in ('enhanced_motion', 'backbone'):
            raise ValueError(
                f"Unsupported MOTION.ROUTER_SOURCE='{self.phase_router_source}'. "
                "Choose from: enhanced_motion, backbone"
            )
        
        if self.use_motion:
            # Parse multi-scale diff strides from config (default: 1,2,4)
            diff_strides_raw = getattr(motion_cfg, 'DIFF_STRIDES', [1, 2, 4])
            diff_strides = tuple(diff_strides_raw) if isinstance(diff_strides_raw, (list, tuple)) else (1, 2, 4)

            self.motion_module = MotionModule(
                dim=self.mid_dim,
                num_frames=cfg.DATA.NUM_INPUT_FRAMES,
                use_autodecoder=getattr(motion_cfg, 'USE_AUTODECODER', True),
                use_long_short=getattr(motion_cfg, 'USE_LONG_SHORT', True),
                use_temporal_attn=getattr(motion_cfg, 'USE_TEMPORAL_ATTN', True),
                diff_strides=diff_strides,
            )
            self.motion_weight = getattr(motion_cfg, 'LOSS_WEIGHT', 0.1)
        
        # ===== 3. Semantic Module =====
        semantic_cfg = cfg.SEMANTIC if hasattr(cfg, 'SEMANTIC') else cfg
        self.use_semantic = getattr(semantic_cfg, 'ENABLE', True)
        self.use_structured_text = getattr(semantic_cfg, 'USE_STRUCTURED_TEXT', False)
        self.description_format = getattr(semantic_cfg, 'DESCRIPTION_FORMAT', 'legacy')
        self.num_phases = getattr(semantic_cfg, 'NUM_PHASES', 3)
        self.strict_class_coverage = getattr(semantic_cfg, 'STRICT_CLASS_COVERAGE', False)
        self.use_phase_router = getattr(semantic_cfg, 'USE_PHASE_ROUTER', False)
        self.phase_loss_weight = getattr(semantic_cfg, 'PHASE_LOSS_WEIGHT', 0.15)
        self.phase_dist_weight = getattr(semantic_cfg, 'PHASE_DIST_WEIGHT', 0.15)
        self.confidence_gated_fusion = getattr(semantic_cfg, 'CONFIDENCE_GATED_FUSION', False)
        self.semantic_descriptions_path_train = getattr(
            semantic_cfg,
            'DESCRIPTIONS_PATH_TRAIN',
            getattr(semantic_cfg, 'DESCRIPTIONS_PATH', None),
        )
        self.semantic_descriptions_path_test = getattr(
            semantic_cfg,
            'DESCRIPTIONS_PATH_TEST',
            getattr(semantic_cfg, 'DESCRIPTIONS_PATH', None),
        )
        self.semantic_descriptions_path_val = getattr(
            semantic_cfg,
            'DESCRIPTIONS_PATH_VAL',
            self.semantic_descriptions_path_test,
        )
        num_classes = getattr(cfg.TRAIN, 'NUM_CLASS', 64)
        train_class_names = getattr(cfg.TRAIN, 'CLASS_NAME', None)
        self.semantic_inference_weight = 0.0
        self.max_semantic_inference_weight = 0.0
        self.register_buffer('test_text_features', None, persistent=False)
        self.register_buffer('test_global_text_features', None, persistent=False)
        self.register_buffer('test_entity_text_features', None, persistent=False)
        self.register_buffer('test_phase_text_features', None, persistent=False)

        if self.use_semantic:
            self.semantic_module = SemanticModule(
                visual_dim=self.mid_dim,
                text_model=getattr(semantic_cfg, 'TEXT_MODEL', 'all-MiniLM-L6-v2'),
                descriptions_path=self.semantic_descriptions_path_train,
                class_names=train_class_names,
                num_classes=num_classes,
                allow_random_fallback=getattr(semantic_cfg, 'ALLOW_RANDOM_FALLBACK', False),
                description_format=self.description_format,
                use_structured_text=self.use_structured_text,
                num_phases=self.num_phases,
                strict_class_coverage=self.strict_class_coverage,
                dataset_name=getattr(cfg.TRAIN, 'DATASET_FEW', getattr(cfg.TRAIN, 'DATASET', '')),
            )
            self.semantic_weight = getattr(semantic_cfg, 'LOSS_WEIGHT', 0.5)
            self.semantic_inference_weight = getattr(semantic_cfg, 'INFERENCE_WEIGHT', 0.0)
            self.max_semantic_inference_weight = getattr(
                semantic_cfg, 'MAX_INFERENCE_WEIGHT', self.semantic_inference_weight
            )

            # Precompute text features for test classes (inference-time semantic fusion)
            test_class_names = self._get_eval_class_names(cfg)
            if test_class_names and len(test_class_names) > 0:
                self._precompute_test_text_features(test_class_names)

        # ===== SFM: Semantic Feature Modulation =====
        self.use_sfm = False
        self.use_phase_router = self.use_semantic and self.use_phase_router
        if self.use_phase_router:
            self.phase_router = MotionPhaseRouter(
                self.mid_dim,
                num_phases=self.num_phases,
            )

        if self.use_semantic:
            sfm_enable = getattr(semantic_cfg, 'USE_SFM', False)
            if sfm_enable:
                self.use_sfm = True
                _text_dim = self.semantic_module.semantic_dim  # typically 384
                self.episode_sfm = EpisodeSFM(
                    text_dim=_text_dim,
                    visual_dim=self.mid_dim,
                    hidden_dim=min(_text_dim, 256),
                )
                if self.use_phase_router and self.use_structured_text:
                    self.class_sfm = PhaseAwareClassSFM(
                        text_dim=_text_dim,
                        visual_dim=self.mid_dim,
                        num_phases=self.num_phases,
                        hidden_dim=min(_text_dim, 256),
                    )
                else:
                    self.class_sfm = ClassSFM(
                        text_dim=_text_dim,
                        visual_dim=self.mid_dim,
                        hidden_dim=min(_text_dim, 256),
                    )
                print(f"[DiSMo] SFM enabled: text_dim={_text_dim}, visual_dim={self.mid_dim}")

        position_a = getattr(cfg.TRAIN, 'POSITION_A', self.mid_dim)
        position_b = getattr(cfg.TRAIN, 'POSITION_B', 0.01)
        
        self.pe = PositionalEncoder(
            d_model=self.mid_dim,
            dropout=0.1,
            A_scale=position_a,
            B_scale=position_b
        )
        
        num_heads = getattr(cfg.TRAIN, 'HEAD', 8)
        self.temporal_atte_before = PreNormAttention(
            self.mid_dim,
            Attention(self.mid_dim, heads=num_heads, 
                     dim_head=self.mid_dim // num_heads, dropout=0.2)
        )
        self.temporal_atte = MultiHeadAttention(
            num_heads, self.mid_dim,
            self.mid_dim // num_heads, self.mid_dim // num_heads,
            dropout=0.05
        )
        
        # ===== 5. Fusion Layer =====
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.mid_dim * 2, self.mid_dim, kernel_size=1, padding=0),
        )
        
        # ===== 6. Classification Head (auxiliary) =====
        self.classification_layer = nn.Linear(self.mid_dim, num_classes)
        self.use_classification = getattr(cfg.TRAIN, 'USE_CLASSIFICATION', True)
        self.cls_weight = getattr(cfg.TRAIN, 'USE_CLASSIFICATION_VALUE', 0.4)
        self.use_local = getattr(cfg.TRAIN, 'USE_LOCAL', True)
        
        self.relu = nn.ReLU(inplace=True)
        
        # ===== 7. Distance metric =====
        self.distance_type = getattr(cfg.TRAIN, 'DISTANCE_TYPE', 'hausdorff').lower()
        
        _sem_infer_w = self.semantic_inference_weight if self.use_semantic else 0.0
        _bb_name = getattr(cfg.VIDEO.HEAD, 'BACKBONE_NAME', 'dinov2')
        print(f"[DiSMo] Initialized with backbone={_bb_name}, dim={self.mid_dim}, "
              f"motion={self.use_motion}, semantic={self.use_semantic}, "
              f"sfm={self.use_sfm}, structured={self.use_structured_text}, "
              f"phase_router={self.use_phase_router}, distance={self.distance_type}, "
              f"sem_infer_w={_sem_infer_w}")
    
    @staticmethod
    def _build_backbone(cfg):
        """
        Build the visual backbone based on config.

        Supports:
        - 'dinov2' / 'dinov2_vits14' / 'dinov2_vitb14' / 'dinov2_vitl14' / 'dinov2_vitg14'
        - 'resnet50'  (torchvision, ImageNet pretrained)
        - 'resnet34'
        - 'resnet18'

        Returns:
            backbone: nn.Module  – forward(x: [B, C, T, H, W]) -> [B, D, T, 1, 1]
            mid_dim:  int        – feature dimension D
        """
        backbone_name = getattr(cfg.VIDEO.HEAD, 'BACKBONE_NAME', 'dinov2').lower()

        # ---- DINOv2 family ----
        if backbone_name.startswith('dinov2'):
            dino_model = getattr(cfg.VIDEO.BACKBONE, 'DINO_MODEL', 'dinov2_vitl14')
            dim_map = {
                'dinov2_vits14': 384,
                'dinov2_vitb14': 768,
                'dinov2_vitl14': 1024,
                'dinov2_vitg14': 1536,
            }
            mid_dim = dim_map.get(dino_model, 1024)
            backbone = DINOv2Backbone(cfg)
            return backbone, mid_dim

        # ---- CLIP visual encoder family ----
        if backbone_name in _CLIPVideoBackbone.CLIP_MODELS:
            clip_name, mid_dim = _CLIPVideoBackbone.CLIP_MODELS[backbone_name]
            freeze = getattr(cfg.VIDEO.BACKBONE, 'FREEZE', True)
            backbone = _CLIPVideoBackbone(clip_name, freeze=freeze)
            return backbone, mid_dim

        # ---- torchvision ResNet family ----
        resnet_map = {
            'resnet18':  (tv_models.resnet18,  512),
            'resnet34':  (tv_models.resnet34,  512),
            'resnet50':  (tv_models.resnet50,  2048),
        }
        if backbone_name in resnet_map:
            factory, mid_dim = resnet_map[backbone_name]
            freeze = getattr(cfg.VIDEO.BACKBONE, 'FREEZE', False)
            backbone = _ResNetVideoBackbone(factory, freeze=freeze)
            return backbone, mid_dim

        raise ValueError(
            f"Unsupported BACKBONE_NAME='{backbone_name}'. "
            f"Choose from: dinov2*, clip_vitb16, clip_vitb32, clip_rn50, clip_rn101, "
            f"resnet18, resnet34, resnet50"
        )

    @torch.no_grad()
    def _precompute_test_text_features(self, test_class_names):
        """
        Precompute text features for test/val classes.
        Used for inference-time semantic prototype fusion (P2).

        Args:
            test_class_names: list of str, class names from TEST.CLASS_NAME
        """
        # Keep train-time fail-fast strict, but allow test-time fallback text for
        # unseen classes so existing episodic zero-shot splits remain runnable.
        banks = self.semantic_module.build_text_feature_banks(
            test_class_names,
            strict_coverage=self.strict_class_coverage,
            descriptions_path=self._get_eval_descriptions_path(),
            enforce_num_classes=False,
        )
        self.test_text_features = banks['fused'].detach()
        self.test_global_text_features = banks['global'].detach()
        self.test_entity_text_features = banks['entity'].detach()
        self.test_phase_text_features = banks['phase'].detach()
        print(f"[DiSMo] Precomputed test text features for {len(test_class_names)} classes "
              f"(inference semantic fusion)")
        if banks['stats']['fallback_hits'] > 0:
            print(
                "[DiSMo] Test semantic fallback examples: "
                + ", ".join(banks['stats']['fallback_examples'][:5])
            )

    @staticmethod
    def _get_eval_class_names(cfg):
        """Select benchmark-compatible evaluation class names."""
        return getattr(cfg.TEST, "CLASS_NAME", None)

    def _get_eval_descriptions_path(self):
        """Select benchmark-compatible semantic description file for evaluation."""
        if self.semantic_descriptions_path_test:
            return self.semantic_descriptions_path_test
        return self.semantic_descriptions_path_train

    @staticmethod
    def _gather_episode_class_ids(batch_class_list=None, support_labels=None, real_support_labels=None):
        """Recover real class ids for the current episode."""
        if batch_class_list is not None:
            return batch_class_list.long()
        if real_support_labels is not None and support_labels is not None:
            unique_locals = torch.unique(support_labels).long()
            ids = []
            for ll in unique_locals:
                mask = (support_labels == ll.float())
                ids.append(real_support_labels[mask][0].long())
            return torch.stack(ids)
        return None

    def _get_active_text_banks(self):
        """Select train/test text banks depending on current mode."""
        if self.training or self.test_text_features is None:
            return {
                'fused': self.semantic_module.text_features,
                'global': self.semantic_module.text_global_features,
                'entity': self.semantic_module.text_entity_features,
                'phase': self.semantic_module.text_phase_features,
            }
        return {
            'fused': self.test_text_features,
            'global': self.test_global_text_features,
            'entity': self.test_entity_text_features,
            'phase': self.test_phase_text_features,
        }

    def _get_episode_semantic_banks(self, batch_class_list,
                                    support_labels=None,
                                    real_support_labels=None):
        """
        Retrieve precomputed semantic banks for the WAY classes in this episode.

        Returns:
            dict with fused/global/entity/phase banks or None
        """
        class_ids = self._gather_episode_class_ids(batch_class_list, support_labels, real_support_labels)
        if class_ids is None:
            return None

        banks = self._get_active_text_banks()
        if banks['fused'] is None:
            return None

        episode_banks = {'class_ids': class_ids}
        target_device = support_labels.device if torch.is_tensor(support_labels) else class_ids.device
        for key, value in banks.items():
            if value is None:
                episode_banks[key] = None
                continue
            class_ids_clamped = class_ids.clamp(0, value.shape[0] - 1)
            episode_banks[key] = value[class_ids_clamped].to(target_device)
        return episode_banks

    @staticmethod
    def _pool_phase_features(video_feat, phase_weights):
        """
        Weighted temporal pooling into phase prototypes.

        Args:
            video_feat: [B, T, D]
            phase_weights: [B, T, P]
        Returns:
            pooled: [B, P, D]
        """
        denom = phase_weights.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(1e-6)
        pooled = torch.einsum('btp,btd->bpd', phase_weights, video_feat)
        return pooled / denom

    @staticmethod
    def _phase_distance(query_phase_proto, class_phase_proto):
        """
        Cosine distance between query and class phase prototypes.

        Args:
            query_phase_proto: [Q, P, D]
            class_phase_proto: [Q, P, D]
        Returns:
            dist: [Q]
        """
        sim = F.cosine_similarity(
            F.normalize(query_phase_proto, dim=-1),
            F.normalize(class_phase_proto, dim=-1),
            dim=-1,
        )
        return (1.0 - sim).mean(dim=-1)

    def get_feats(self, support_images, target_images, support_labels=None, episode_text_emb=None):
        """
        Extract and enhance features from support and target videos.
        
        Args:
            support_images: [B_s, 3, T, H, W] support set images
            target_images: [B_q, 3, T, H, W] query set images
            support_labels: [B_s] support labels (optional)
            episode_text_emb: [text_dim] episode-level text embedding for SFM (optional)
            
        Returns:
            support_features: [Q, S, T, D] support features
            target_features: [Q, T, D] target features
            class_logits: [B, num_classes] classification logits
            aux_losses: dict of auxiliary losses
            support_phase_weights: [S, T, P] or None
            target_phase_weights: [Q, T, P] or None
        """
        T = self.args.DATA.NUM_INPUT_FRAMES
        aux_losses = {}
        support_phase_weights = None
        target_phase_weights = None

        # Few-shot dataset returns flattened frames: [B*T, C, H, W].
        # Rebuild per-video clips for the ViT backbone.
        if support_images.dim() == 4:
            if support_images.shape[0] % T != 0:
                raise ValueError(
                    f"Support tensor has {support_images.shape[0]} frames, "
                    f"not divisible by NUM_INPUT_FRAMES={T}."
                )
            support_images = rearrange(support_images, '(b t) c h w -> b c t h w', t=T)
        elif support_images.dim() == 5 and support_images.shape[1] == T:
            support_images = rearrange(support_images, 'b t c h w -> b c t h w')

        if target_images.dim() == 4:
            if target_images.shape[0] % T != 0:
                raise ValueError(
                    f"Target tensor has {target_images.shape[0]} frames, "
                    f"not divisible by NUM_INPUT_FRAMES={T}."
                )
            target_images = rearrange(target_images, '(b t) c h w -> b c t h w', t=T)
        elif target_images.dim() == 5 and target_images.shape[1] == T:
            target_images = rearrange(target_images, 'b t c h w -> b c t h w')
        
        # ===== Backbone Features =====
        # [B, D, T, 1, 1] -> [B, D, T] -> [B, T, D]
        support_feat = self.backbone(support_images).squeeze(-1).squeeze(-1)
        target_feat = self.backbone(target_images).squeeze(-1).squeeze(-1)
        support_backbone_feat = support_feat
        target_backbone_feat = target_feat
        
        num_support = support_feat.shape[0]
        num_query = target_feat.shape[0]
        
        # ===== Motion Enhancement =====
        if self.use_motion:
            # MotionModule returns enhanced features (original + scaled motion)
            support_feat, supp_motion_loss = self.motion_module(
                support_feat, compute_loss=self.training
            )
            target_feat, tgt_motion_loss = self.motion_module(
                target_feat, compute_loss=self.training
            )
            
            if self.training:
                if 'motion_recon' in supp_motion_loss:
                    aux_losses['motion_recon'] = (
                        supp_motion_loss['motion_recon'] + 
                        tgt_motion_loss.get('motion_recon', 0)
                    ) * self.motion_weight
                if 'long_short_contrast' in supp_motion_loss:
                    aux_losses['long_short'] = (
                        supp_motion_loss['long_short_contrast'] + 
                        tgt_motion_loss.get('long_short_contrast', 0)
                    ) * 0.1

        if self.use_phase_router:
            router_support_feat = support_feat if self.phase_router_source == 'enhanced_motion' else support_backbone_feat
            router_target_feat = target_feat if self.phase_router_source == 'enhanced_motion' else target_backbone_feat
            support_phase_weights = self.phase_router(router_support_feat)
            target_phase_weights = self.phase_router(router_target_feat)
        
        # ===== SFM Stage 1: Episode-level FiLM =====
        if self.use_sfm and episode_text_emb is not None:
            support_feat = self.episode_sfm(support_feat, episode_text_emb)
            target_feat = self.episode_sfm(target_feat, episode_text_emb)

        # ===== Reshape: [B, D, T] -> [B, T, D] =====
        support_feat = rearrange(support_feat, 'b d t -> b t d')
        target_feat = rearrange(target_feat, 'b d t -> b t d')
        
        # ===== Temporal Relation (HyRSM style) =====
        # Positional encoding + self-attention
        support_feat = self.relu(self.temporal_atte_before(self.pe(support_feat)))
        target_feat = self.relu(self.temporal_atte_before(self.pe(target_feat)))
        
        # Classification logits (auxiliary task)
        class_logits = self.classification_layer(
            torch.cat([support_feat, target_feat], dim=0)
        ).reshape(-1, self.classification_layer.out_features)
        
        # Cross-video attention
        # support_ext: [Q, S, T, D]
        support_ext = support_feat.unsqueeze(0).repeat(num_query, 1, 1, 1)
        target_ext = target_feat.unsqueeze(1)  # [Q, 1, T, D]
        
        # Mean pooling over time for attention
        feature_in = torch.cat([
            support_ext.mean(2),  # [Q, S, D]
            target_ext.mean(2)    # [Q, 1, D]
        ], dim=1)  # [Q, S+1, D]
        
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in))
        
        # Fuse cross-attention info back to frame features
        support_out = torch.cat([
            support_ext,
            feature_in[:, :-1, :].unsqueeze(2).repeat(1, 1, T, 1)
        ], dim=-1)  # [Q, S, T, 2D]
        
        support_out = self.layer2(
            rearrange(support_out, 'q s t d -> (q s) d t')
        )
        support_out = rearrange(support_out, '(q s) d t -> q s t d', q=num_query)
        
        target_out = torch.cat([
            rearrange(target_feat, 'q t d -> q d t'),
            feature_in[:, -1:, :].permute(0, 2, 1).repeat(1, 1, T)
        ], dim=1)
        target_out = self.layer2(target_out)
        target_out = rearrange(target_out, 'q d t -> q t d')
        
        return support_out, target_out, class_logits, aux_losses, support_phase_weights, target_phase_weights
    
    def forward(self, inputs):
        """
        Forward pass for few-shot classification.
        
        Args:
            inputs: dict with keys:
                - support_set: [B_s, 3, T, H, W]
                - support_labels: [B_s]
                - target_set: [B_q, 3, T, H, W]
                - real_support_labels: [B_s] (optional)
                - real_target_labels: [B_q] (optional)
                
        Returns:
            dict with:
                - logits: [Q, C] classification scores
                - class_logits: [B, num_classes] auxiliary classification
                - aux_losses: dict of auxiliary losses
        """
        support_images = inputs['support_set']
        support_labels = inputs['support_labels']
        target_images = inputs['target_set']

        # CPU/single-process fallback can leave a task-batch dimension.
        # Keep behavior consistent with the trainer's single-task update.
        if torch.is_tensor(support_images) and support_images.dim() > 4:
            support_images = support_images[0]
        if torch.is_tensor(target_images) and target_images.dim() > 4:
            target_images = target_images[0]
        if torch.is_tensor(support_labels) and support_labels.dim() > 1:
            support_labels = support_labels[0]

        for key in ["target_labels", "real_target_labels", "real_support_labels", "batch_class_list"]:
            if key in inputs and torch.is_tensor(inputs[key]) and inputs[key].dim() > 1:
                inputs[key] = inputs[key][0]
        
        # ===== Structured semantics for the current episode =====
        batch_class_list = inputs.get('batch_class_list', None)
        episode_semantic_banks = None
        episode_text_emb = None
        if self.use_semantic:
            _real_supp = inputs.get('real_support_labels', None)
            episode_semantic_banks = self._get_episode_semantic_banks(
                batch_class_list, support_labels, _real_supp
            )
            if self.use_sfm and episode_semantic_banks is not None:
                episode_context_feats = episode_semantic_banks['global']
                if episode_semantic_banks['entity'] is not None:
                    episode_context_feats = F.normalize(
                        (episode_context_feats + episode_semantic_banks['entity']) * 0.5,
                        dim=-1,
                    )
                episode_text_emb = episode_context_feats.mean(dim=0)  # [text_dim]

        # Get enhanced features
        support_feat, target_feat, class_logits, aux_losses, support_phase_weights, target_phase_weights = self.get_feats(
            support_images, target_images, support_labels,
            episode_text_emb=episode_text_emb,
        )
        
        unique_labels = torch.unique(support_labels)
        n_queries = target_feat.shape[0]
        n_support = support_feat.shape[1]
        T = support_feat.shape[2]
        
        # ===== Semantic Enhancement (optional) =====
        if self.use_semantic and self.training:
            # Video-level feature for semantic alignment
            query_video_feat = target_feat.mean(dim=1)  # [Q, D]
            
            real_target_labels = inputs.get('real_target_labels', None)
            semantic_result = self.semantic_module(
                query_video_feat,
                class_indices=real_target_labels
            )
            
            if semantic_result['loss'] is not None:
                aux_losses['semantic'] = semantic_result['loss'] * self.semantic_weight
            if (
                self.use_phase_router
                and episode_semantic_banks is not None
                and target_phase_weights is not None
                and real_target_labels is not None
                and self.semantic_module.text_phase_features is not None
            ):
                max_label = int(real_target_labels.max().item())
                if max_label < self.semantic_module.text_phase_features.shape[0]:
                    query_phase_feat = self._pool_phase_features(target_feat, target_phase_weights)
                    phase_loss = self.semantic_module.compute_phase_alignment_loss(
                        query_phase_feat,
                        real_target_labels.long(),
                    )
                    aux_losses['phase_semantic'] = phase_loss * self.phase_loss_weight
        
        # ===== Distance Computation =====
        episode_fused_text = episode_semantic_banks['fused'] if episode_semantic_banks is not None else None
        episode_phase_text = episode_semantic_banks['phase'] if episode_semantic_banks is not None else None

        if self.use_sfm and episode_fused_text is not None:
            # ---- SFM path: per-class distance with semantic channel attention ----
            class_dists = []
            for idx, c in enumerate(unique_labels):
                class_mask = extract_class_indices(support_labels, c)

                # Symmetric channel attention on BOTH support & query
                class_support = torch.index_select(support_feat, 1, class_mask)  # [Q, shots, T, D]
                if (
                    self.use_phase_router
                    and episode_phase_text is not None
                    and support_phase_weights is not None
                    and target_phase_weights is not None
                    and isinstance(self.class_sfm, PhaseAwareClassSFM)
                ):
                    class_phase_text = episode_phase_text[idx]
                    class_support_phase_weights = torch.index_select(
                        support_phase_weights, 0, class_mask
                    ).unsqueeze(0).expand(n_queries, -1, -1, -1)
                    class_support_m = self.class_sfm(
                        class_support, class_phase_text, class_support_phase_weights
                    )
                    target_m = self.class_sfm(
                        target_feat, class_phase_text, target_phase_weights
                    )
                else:
                    class_text_emb = episode_fused_text[idx]  # [text_dim]
                    class_support_m = self.class_sfm(class_support, class_text_emb)
                    target_m = self.class_sfm(target_feat, class_text_emb)           # [Q, T, D]

                if self.distance_type == 'otam':
                    dist = batch_otam_distance(class_support_m, target_m).mean(dim=1)
                else:
                    shots = class_support_m.shape[1]
                    cs_flat = rearrange(class_support_m, 'q s t d -> q (s t) d')
                    frame_sim = torch.matmul(
                        F.normalize(cs_flat, dim=-1),
                        F.normalize(target_m, dim=-1).transpose(-1, -2)
                    )
                    frame_sim = frame_sim.reshape(n_queries, shots, T, T)
                    frame_dists = 1 - frame_sim
                    dist = (frame_dists.min(dim=3)[0].sum(dim=2)
                            + frame_dists.min(dim=2)[0].sum(dim=2)).mean(dim=1)

                if (
                    self.use_phase_router
                    and support_phase_weights is not None
                    and target_phase_weights is not None
                ):
                    shots = class_support_m.shape[1]
                    class_support_phase_weights = torch.index_select(
                        support_phase_weights, 0, class_mask
                    ).unsqueeze(0).expand(n_queries, -1, -1, -1)
                    support_phase_proto = self._pool_phase_features(
                        class_support_m.reshape(n_queries * shots, T, -1),
                        class_support_phase_weights.reshape(n_queries * shots, T, self.num_phases),
                    ).reshape(n_queries, shots, self.num_phases, -1).mean(dim=1)
                    query_phase_proto = self._pool_phase_features(target_m, target_phase_weights)
                    dist = dist + self.phase_dist_weight * self._phase_distance(
                        query_phase_proto, support_phase_proto
                    )
                class_dists.append(dist)
        else:
            # ---- Original path (no SFM) ----
            if self.distance_type == 'otam':
                cum_dists = batch_otam_distance(support_feat, target_feat)
            else:
                support_flat = rearrange(support_feat, 'q s t d -> q (s t) d')
                frame_sim = torch.matmul(
                    F.normalize(support_flat, dim=-1),
                    F.normalize(target_feat, dim=-1).transpose(-1, -2)
                )
                frame_sim = frame_sim.reshape(n_queries, n_support, T, T)
                frame_dists = 1 - frame_sim
                cum_dists = frame_dists.min(dim=3)[0].sum(dim=2) + frame_dists.min(dim=2)[0].sum(dim=2)

            class_dists = []
            for c in unique_labels:
                class_mask = extract_class_indices(support_labels, c)
                class_dist = torch.mean(
                    torch.index_select(cum_dists, 1, class_mask), dim=1
                )
                if (
                    self.use_phase_router
                    and support_phase_weights is not None
                    and target_phase_weights is not None
                ):
                    class_support = torch.index_select(support_feat, 1, class_mask)
                    shots = class_support.shape[1]
                    class_support_phase_weights = torch.index_select(
                        support_phase_weights, 0, class_mask
                    ).unsqueeze(0).expand(n_queries, -1, -1, -1)
                    support_phase_proto = self._pool_phase_features(
                        class_support.reshape(n_queries * shots, T, -1),
                        class_support_phase_weights.reshape(n_queries * shots, T, self.num_phases),
                    ).reshape(n_queries, shots, self.num_phases, -1).mean(dim=1)
                    query_phase_proto = self._pool_phase_features(target_feat, target_phase_weights)
                    class_dist = class_dist + self.phase_dist_weight * self._phase_distance(
                        query_phase_proto, support_phase_proto
                    )
                class_dists.append(class_dist)

        class_dists = torch.stack(class_dists, dim=1)  # [Q, C]

        visual_logits = -class_dists  # [Q, C]  higher = more similar
        
        # ===== Inference-time Semantic Fusion (P2) =====
        sem_fusion_weight = (
            self.max_semantic_inference_weight if self.confidence_gated_fusion
            else self.semantic_inference_weight
        )
        if (
            self.use_semantic
            and not self.training
            and sem_fusion_weight > 0
            and self.test_text_features is not None
        ):
            if episode_fused_text is None:
                _real_supp = inputs.get('real_support_labels', None)
                episode_semantic_banks = self._get_episode_semantic_banks(
                    batch_class_list, support_labels, _real_supp
                )
                episode_fused_text = (
                    episode_semantic_banks['fused'] if episode_semantic_banks is not None else None
                )
            if episode_fused_text is not None:
                query_video_feat = target_feat.mean(dim=1)  # [Q, D]
                vis_emb = self.semantic_module.project_visual_features(query_video_feat)

                logit_scale = self.semantic_module.logit_scale.exp().clamp(max=100)
                semantic_logits = logit_scale * vis_emb @ episode_fused_text.t()  # [Q, C]

                vis_probs = F.softmax(visual_logits, dim=-1)
                sem_probs = F.softmax(semantic_logits, dim=-1)
                if self.confidence_gated_fusion:
                    entropy = -(vis_probs * torch.log(vis_probs + 1e-8)).sum(dim=-1)
                    max_entropy = torch.log(
                        torch.tensor(float(vis_probs.shape[-1]), device=vis_probs.device)
                    ).clamp_min(1e-6)
                    alpha = (self.max_semantic_inference_weight * entropy / max_entropy).clamp(
                        0.0, self.max_semantic_inference_weight
                    ).unsqueeze(-1)
                else:
                    alpha = self.semantic_inference_weight
                fused_probs = (1 - alpha) * vis_probs + alpha * sem_probs
                visual_logits = torch.log(fused_probs + 1e-8)
        
        return_dict = {
            'logits': visual_logits,
            'class_logits': class_logits,
            'aux_losses': aux_losses,
        }
        
        return return_dict
    
    def loss(self, task_dict, model_dict):
        """
        Compute total loss including all auxiliary losses.
        
        Args:
            task_dict: dict with ground truth labels
            model_dict: dict with model outputs
            
        Returns:
            total_loss: scalar loss value
        """
        # Main few-shot loss
        main_loss = F.cross_entropy(
            model_dict['logits'],
            task_dict['target_labels'].long()
        )
        
        # Classification auxiliary loss
        if self.use_classification and 'class_logits' in model_dict:
            real_labels = torch.cat([
                task_dict['real_support_labels'],
                task_dict['real_target_labels']
            ], dim=0)
            
            if self.use_local:
                real_labels = real_labels.unsqueeze(1).repeat(
                    1, self.args.DATA.NUM_INPUT_FRAMES
                ).reshape(-1)
            
            cls_loss = F.cross_entropy(model_dict['class_logits'], real_labels.long())
            main_loss = main_loss + self.cls_weight * cls_loss
        
        # Add auxiliary losses from modules
        for loss_name, loss_val in model_dict.get('aux_losses', {}).items():
            if loss_val is not None and isinstance(loss_val, torch.Tensor):
                main_loss = main_loss + loss_val
        
        return main_loss
    
    def distribute_model(self):
        """Distribute model across GPUs."""
        if self.args.TRAIN.DDP_GPU > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(
                self.backbone,
                device_ids=[i for i in range(0, self.args.TRAIN.DDP_GPU)]
            )
