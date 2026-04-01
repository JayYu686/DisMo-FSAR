#!/usr/bin/env python3
# Copyright (C) DiSMo Authors.

"""
DINOv2 Backbone for DiSMo: Few-Shot Action Recognition.

This module wraps DINOv2 ViT models for video understanding,
handling the conversion from video format (B, C, T, H, W) to 
frame-level ViT input (B*T, C, H, W).
"""

import os
import sys

import torch
import torch.nn as nn
from einops import rearrange
from utils.registry import Registry

# 尝试导入现有的 BACKBONE_REGISTRY，如果失败则创建新的
try:
    from models.base.backbone import BACKBONE_REGISTRY
except ImportError:
    BACKBONE_REGISTRY = Registry("Backbone")


@BACKBONE_REGISTRY.register()
class DINOv2Backbone(nn.Module):
    """
    DINOv2 ViT backbone for video understanding.
    
    Handles frame-level feature extraction and temporal reshaping.
    Supports ViT-S/B/L/G variants with optional freezing and AMP.
    
    Args:
        cfg: Configuration object with VIDEO.BACKBONE settings:
            - DINO_MODEL: Model variant (dinov2_vits14, dinov2_vitb14, 
                          dinov2_vitl14, dinov2_vitg14)
            - FREEZE: Whether to freeze backbone weights (default: True)
            - RETURN_PATCHES: Return patch tokens instead of CLS (default: False)
    
    Input:
        x: Video tensor [B, C, T, H, W] or dict with 'video' key
        
    Output:
        features: [B, D, T, 1, 1] to match CNN backbone output format
    """
    
    # 支持的模型及其输出维度
    SUPPORTED_MODELS = {
        'dinov2_vits14': 384,   # ViT-S/14
        'dinov2_vitb14': 768,   # ViT-B/14
        'dinov2_vitl14': 1024,  # ViT-L/14
        'dinov2_vitg14': 1536,  # ViT-G/14
    }
    
    def __init__(self, cfg):
        super().__init__()
        
        # 获取配置
        backbone_cfg = cfg.VIDEO.BACKBONE if hasattr(cfg.VIDEO, 'BACKBONE') else cfg.VIDEO
        
        # 模型选择
        self.model_name = getattr(backbone_cfg, 'DINO_MODEL', 'dinov2_vitl14')
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported DINO model: {self.model_name}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        self.embed_dim = self.SUPPORTED_MODELS[self.model_name]
        self.pretrained = getattr(backbone_cfg, 'PRETRAINED', True)
        self.repo = getattr(backbone_cfg, 'DINO_REPO', 'facebookresearch/dinov2')
        self.source = str(getattr(backbone_cfg, 'DINO_SOURCE', 'auto')).lower()
        self.local_repo = os.path.expanduser(
            getattr(backbone_cfg, 'DINO_LOCAL_REPO', '~/.cache/torch/hub/facebookresearch_dinov2_main')
        )

        self._check_runtime_compatibility()
        
        # 加载预训练模型
        print(f"[DiSMo] Loading DINOv2 backbone: {self.model_name} (dim={self.embed_dim})")
        self.backbone = self._load_backbone()
        
        # 冻结策略
        self.freeze = getattr(backbone_cfg, 'FREEZE', True)
        if self.freeze:
            print("[DiSMo] Freezing DINOv2 backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        # 是否返回 patch tokens (用于空间细粒度任务)
        self.return_patches = getattr(backbone_cfg, 'RETURN_PATCHES', False)
        
        # 是否使用混合精度
        self.use_amp = getattr(cfg.TRAIN, 'AMP_ENABLE', False) if hasattr(cfg, 'TRAIN') else False

    def _check_runtime_compatibility(self):
        """Validate runtime requirements before touching torch.hub."""
        if sys.version_info < (3, 10):
            py_ver = "{}.{}.{}".format(*sys.version_info[:3])
            print(
                "[DiSMo] WARNING: running on Python {}. Upstream DINOv2 may fail on "
                "PEP604 type annotations. Prefer Python 3.10+ or use a patched local cache.".format(py_ver)
            )
        if self.source not in {'auto', 'local', 'github'}:
            raise ValueError("VIDEO.BACKBONE.DINO_SOURCE must be one of: auto, local, github")

    def _patch_local_repo_for_legacy_python(self):
        """Patch local DINOv2 cache for Python<3.10 annotation compatibility."""
        if sys.version_info >= (3, 10):
            return
        patch_targets = [
            os.path.join(self.local_repo, "dinov2/layers/attention.py"),
            os.path.join(self.local_repo, "dinov2/layers/block.py"),
        ]
        for file_path in patch_targets:
            try:
                if not os.path.exists(file_path):
                    continue
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                future_line = "from __future__ import annotations"
                if future_line in content:
                    continue
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(future_line + "\n\n" + content)
                print(f"[DiSMo] Applied Python<3.10 compatibility patch: {file_path}")
            except Exception as exc:
                print(f"[DiSMo] WARNING: failed to patch {file_path}: {exc}")

    def _load_backbone(self):
        """Load DINOv2 from local cache first, then GitHub when allowed."""
        errors = []

        if self.source in {'auto', 'local'} and os.path.isdir(self.local_repo):
            self._patch_local_repo_for_legacy_python()
            try:
                print(f"[DiSMo] Trying local DINOv2 repo: {self.local_repo}")
                return torch.hub.load(
                    self.local_repo,
                    self.model_name,
                    source='local',
                    pretrained=self.pretrained,
                )
            except Exception as exc:
                errors.append(f"local load failed: {exc}")

        if self.source in {'auto', 'github'}:
            try:
                print(f"[DiSMo] Trying GitHub DINOv2 repo: {self.repo}")
                return torch.hub.load(
                    self.repo,
                    self.model_name,
                    pretrained=self.pretrained,
                )
            except Exception as exc:
                errors.append(f"github load failed: {exc}")

        detail = " | ".join(errors) if len(errors) > 0 else "no valid DINOv2 load source found"
        if "unsupported operand type(s) for |" in detail:
            raise RuntimeError(
                "DINOv2 source uses Python 3.10+ type syntax. "
                "Please run with Python >= 3.10."
            )
        raise RuntimeError(f"Failed to load DINOv2 backbone '{self.model_name}': {detail}")
        
    @torch.no_grad()
    def _extract_features_frozen(self, x):
        """Extract features without gradient computation when frozen."""
        if self.return_patches:
            # 返回所有 patch tokens [B, N, D]
            # DINOv2 forward_features 返回包含多个键的字典
            out = self.backbone.forward_features(x)
            if isinstance(out, dict):
                return out.get('x_norm_patchtokens', out.get('x_patchtokens'))
            return out
        else:
            # 仅返回 CLS token [B, D]
            return self.backbone(x)
    
    def _extract_features(self, x):
        """Extract features with gradient computation (for fine-tuning)."""
        if self.return_patches:
            out = self.backbone.forward_features(x)
            if isinstance(out, dict):
                return out.get('x_norm_patchtokens', out.get('x_patchtokens'))
            return out
        else:
            return self.backbone(x)
    
    def forward(self, x):
        """
        Forward pass for video input.
        
        Args:
            x: Video tensor [B, C, T, H, W] or dict with 'video' key
            
        Returns:
            features: [B, D, T, 1, 1] to match CNN backbone output format
                      If return_patches=True: [B, D, T, N] where N is num patches
        """
        # 处理 dict 输入
        if isinstance(x, dict):
            x = x['video']
        
        # 验证输入维度
        if x.dim() == 4:
            # 已经是 [B*T, C, H, W] 格式
            B, C, H, W = x.shape
            T = 1
            need_reshape = False
        elif x.dim() == 5:
            B, C, T, H, W = x.shape
            need_reshape = True
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")
        
        # 展平时间维度: [B, C, T, H, W] -> [B*T, C, H, W]
        if need_reshape:
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        
        # 提取特征
        if self.freeze:
            features = self._extract_features_frozen(x)
        else:
            features = self._extract_features(x)
        
        # 恢复时间维度
        if self.return_patches:
            # features: [B*T, N, D] -> [B, T, N, D] -> [B, D, T, N]
            features = rearrange(features, '(b t) n d -> b d t n', b=B, t=T)
        else:
            # features: [B*T, D] -> [B, D, T] -> [B, D, T, 1, 1]
            features = rearrange(features, '(b t) d -> b d t', b=B, t=T)
            features = features.unsqueeze(-1).unsqueeze(-1)
        
        return features
    
    def train(self, mode=True):
        """Override train to keep backbone frozen if needed."""
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self
    
    @property
    def output_dim(self):
        """Return the output embedding dimension."""
        return self.embed_dim


# 为了兼容性，也注册为 Identity + DINOv2 的工厂函数
def build_dinov2_backbone(cfg):
    """Factory function to create DINOv2 backbone."""
    return DINOv2Backbone(cfg)
