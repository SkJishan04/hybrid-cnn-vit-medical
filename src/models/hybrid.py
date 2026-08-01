"""
Hybrid CNN + Vision Transformer with cross-attention fusion.

Architecture:
    Input image
      -> CNN backbone (ResNet18, first 3 stages only — local feature extraction)
      -> Feature maps [B, C, H, W]
      -> Flatten to patch tokens + positional embedding + [CLS] token
      -> Transformer encoder, each layer using CROSS-attention:
           Query = transformer tokens (evolving global representation)
           Key/Value = the raw CNN feature map (un-abstracted local detail)
         This lets the transformer repeatedly "look back" at local CNN
         features rather than only reasoning over its own tokens.
      -> Dual-attention fusion gate: a learned per-location scalar deciding
         how much to trust CNN features vs. transformer output at each
         spatial position, rather than a fixed concatenation/average.
      -> Global average pool -> classifier head

Why a CNN backbone this small (ResNet18, first 3 stages) rather than the
full ResNet50 used in the baseline: the point of this branch is *local*
feature extraction to feed the transformer, not classification on its own.
A full-depth, full-width CNN backbone would let the CNN branch dominate
and make the transformer/fusion mechanism largely decorative.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    First 3 stages of a ResNet18 (stem + layer1 + layer2 + layer3),
    stopping before layer4 to keep spatial resolution higher (14x14
    instead of 7x7 at 224 input) — more tokens for the transformer to
    work with, and shallower features that are more "local" in nature.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = timm.create_model("resnet18", pretrained=pretrained, features_only=True)
        # features_only resnet18 stages: [stem, layer1, layer2, layer3, layer4]
        # channel counts at each stage: [64, 64, 128, 256, 512]
        self.stem = resnet.conv1
        self.bn1 = resnet.bn1
        self.act1 = resnet.act1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.out_channels = 256  # output channels after layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # [B, 256, 14, 14] for 224x224 input


class CrossAttentionBlock(nn.Module):
    """
    One transformer encoder layer, but using cross-attention to the CNN
    feature map instead of self-attention only. Query comes from the
    evolving token sequence; Key/Value come from the (fixed) CNN features,
    so every layer can re-attend to un-abstracted local detail.
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, cnn_kv: torch.Tensor) -> torch.Tensor:
        # cross-attention: tokens attend to CNN feature map
        q = self.norm1(tokens)
        kv = self.norm_kv(cnn_kv)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        tokens = tokens + attn_out
        # standard MLP block
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class DualAttentionFusionGate(nn.Module):
    """
    Learns a per-token scalar gate alpha in [0,1] deciding how much to
    trust the CNN-derived feature vs. the transformer-derived feature at
    that spatial location, instead of a fixed blend (e.g. concatenation
    or averaging). This is the project's core novelty component.

        output = alpha * cnn_feature + (1 - alpha) * transformer_feature
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, cnn_feat: torch.Tensor, transformer_feat: torch.Tensor) -> torch.Tensor:
        # both inputs: [B, N, dim]
        combined = torch.cat([cnn_feat, transformer_feat], dim=-1)
        alpha = self.gate_mlp(combined)  # [B, N, 1]
        fused = alpha * cnn_feat + (1 - alpha) * transformer_feat
        return fused, alpha


class HybridCNNViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        pretrained_cnn: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn_backbone = CNNBackbone(pretrained=pretrained_cnn)
        cnn_channels = self.cnn_backbone.out_channels  # 256

        # project CNN features to embed_dim if they don't already match
        self.cnn_proj = (
            nn.Identity() if cnn_channels == embed_dim
            else nn.Conv2d(cnn_channels, embed_dim, kernel_size=1)
        )

        # learnable [CLS] token + positional embeddings for the transformer tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 14x14 = 196 spatial tokens + 1 CLS token, for 224x224 input
        num_patches = 14 * 14
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.fusion_gate = DualAttentionFusionGate(embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        B = x.shape[0]

        # 1. CNN backbone -> local feature map
        cnn_feat_map = self.cnn_backbone(x)          # [B, C, H, W]
        cnn_feat_map = self.cnn_proj(cnn_feat_map)     # [B, embed_dim, H, W]
        H, W = cnn_feat_map.shape[-2:]

        # 2. flatten CNN feature map into token sequence (used as cross-attn K/V)
        cnn_tokens = cnn_feat_map.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]

        # 3. build the transformer's own token sequence: [CLS] + same spatial tokens,
        #    with positional embeddings, as the starting point for cross-attention queries
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, cnn_tokens], dim=1)   # [B, 1+H*W, embed_dim]
        tokens = tokens + self.pos_embed

        # 4. cross-attention transformer encoder: tokens repeatedly attend
        #    back to the raw (un-positional-embedded) CNN feature map
        for block in self.transformer_blocks:
            tokens = block(tokens, cnn_kv=cnn_tokens)

        # 5. dual-attention fusion: blend CNN-branch tokens with transformer-branch
        #    tokens at each spatial position (skip the CLS token for this, then
        #    re-attach it since CNN has no direct analogue for CLS)
        cls_out, spatial_out = tokens[:, :1], tokens[:, 1:]
        fused_spatial, gate_alpha = self.fusion_gate(cnn_tokens, spatial_out)

        # 6. pool: CLS token (global summary) + mean of fused spatial tokens
        pooled = cls_out.squeeze(1) + fused_spatial.mean(dim=1)
        pooled = self.norm(pooled)
        logits = self.head(pooled)

        if return_gate:
            return logits, gate_alpha
        return logits


def build_hybrid(num_classes: int = 7, pretrained: bool = True, drop_rate: float = 0.2,
                  embed_dim: int = 256, depth: int = 4, num_heads: int = 8) -> nn.Module:
    return HybridCNNViT(
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        pretrained_cnn=pretrained,
        dropout=drop_rate,
    )


if __name__ == "__main__":
    model = build_hybrid(pretrained=False)  # pretrained=False for a quick offline shape check
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect [2, 7]
    logits, gate = model(dummy, return_gate=True)
    print(f"Gate alpha shape: {gate.shape}")  # expect [2, 196, 1]
    print(f"Gate alpha mean: {gate.mean().item():.4f}")  # sanity check: should be in (0, 1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")