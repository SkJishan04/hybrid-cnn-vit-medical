
"""
Loads a trained checkpoint and evaluates it on the held-out TEST split
(never seen during training or checkpointing — val was used for that).
This is the number you actually report.

Usage:
    python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/resnet50_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import HAM10000Dataset
from models.baseline_resnet import build_resnet50
from utils.metrics import evaluate, format_results


def build_model(cfg: dict) -> torch.nn.Module:
    name = cfg["model"]["name"]
    if name == "resnet50":
        return build_resnet50(
            num_classes=cfg["model"]["num_classes"],
            pretrained=False,  # loading trained weights next, no need for imagenet init
            drop_rate=cfg["model"]["drop_rate"],
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["classes"]

    test_ds = HAM10000Dataset(
        cfg["data"]["processed_dir"], classes, "test", cfg["data"]["image_size"]
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = evaluate(model, test_loader, device, classes)
    print(f"\n=== TEST SET RESULTS ({len(test_ds)} images) ===")
    print(format_results(results))
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(classes)
    print(results["confusion_matrix"])


if __name__ == "__main__":
    main()