"""
Pure CNN baseline. This is your first comparison point — get this
trained and evaluated before touching the ViT or hybrid model.
"""
import timm
import torch.nn as nn


def build_resnet50(num_classes: int = 7, pretrained: bool = True, drop_rate: float = 0.2) -> nn.Module:
    model = timm.create_model(
        "resnet50",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


if __name__ == "__main__":
    import torch

    model = build_resnet50()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect [2, 7]