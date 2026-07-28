"""
Pure Vision Transformer baseline — second comparison point, alongside
the ResNet50 baseline. ViT-Small is a reasonable choice for a mid-sized
dataset like HAM10000 (~10k images): large enough to fine-tune well,
small enough to not need enormous compute.
"""
import timm
import torch.nn as nn


def build_vit(num_classes: int = 7, pretrained: bool = True, drop_rate: float = 0.2) -> nn.Module:
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


if __name__ == "__main__":
    import torch

    model = build_vit()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect [2, 7]