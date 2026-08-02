"""
Runs evaluation and saves everything in report-ready formats:
  results/{model_name}/metrics.json         — full numeric results, machine-readable
  results/{model_name}/metrics.csv          — per-class table, opens in Excel/Sheets
  results/{model_name}/confusion_matrix.png — actual plotted image, not a raw array
  results/{model_name}/summary.md           — ready-to-paste markdown table for your README

Usage:
    python src/save_results.py --config configs/config.yaml --checkpoint checkpoints/hybrid_best.pt --model_name hybrid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import HAM10000Dataset
from models.baseline_resnet import build_resnet50
from utils.metrics import evaluate


def build_model(cfg: dict) -> torch.nn.Module:
    name = cfg["model"]["name"]
    if name == "resnet50":
        return build_resnet50(
            num_classes=cfg["model"]["num_classes"],
            pretrained=False,
            drop_rate=cfg["model"]["drop_rate"],
        )
    elif name == "vit_small_patch16_224":
        from models.baseline_vit import build_vit
        return build_vit(
            num_classes=cfg["model"]["num_classes"],
            pretrained=False,
            drop_rate=cfg["model"]["drop_rate"],
        )
    elif name == "hybrid":
        from models.hybrid import build_hybrid
        return build_hybrid(
            num_classes=cfg["model"]["num_classes"],
            pretrained=False,
            drop_rate=cfg["model"]["drop_rate"],
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def save_confusion_matrix(cm: np.ndarray, classes: list, out_path: Path, model_name: str):
    plt.figure(figsize=(8, 6))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        cbar_kws={"label": "fraction of true class"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {model_name} (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
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
    results = evaluate(model, test_loader, device, classes)

    out_dir = Path("results") / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_safe = {k: v for k, v in results.items() if k != "confusion_matrix"}
    json_safe["confusion_matrix"] = results["confusion_matrix"].tolist()
    json_safe["test_set_size"] = len(test_ds)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(json_safe, f, indent=2)

    with open(out_dir / "metrics.csv", "w") as f:
        f.write("class,f1_score\n")
        for cls, f1 in results["per_class_f1"].items():
            f.write(f"{cls},{f1:.4f}\n")
        f.write(f"\nmetric,value\n")
        f.write(f"accuracy,{results['accuracy']:.4f}\n")
        f.write(f"precision_macro,{results['precision_macro']:.4f}\n")
        f.write(f"recall_macro,{results['recall_macro']:.4f}\n")
        f.write(f"f1_macro,{results['f1_macro']:.4f}\n")
        f.write(f"auc_roc_macro,{results['auc_roc_macro']:.4f}\n")

    save_confusion_matrix(
        results["confusion_matrix"], classes,
        out_dir / "confusion_matrix.png", args.model_name
    )

    with open(out_dir / "summary.md", "w") as f:
        f.write(f"### {args.model_name} — Test Set Results ({len(test_ds)} images)\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Accuracy | {results['accuracy']:.4f} |\n")
        f.write(f"| Precision (macro) | {results['precision_macro']:.4f} |\n")
        f.write(f"| Recall (macro) | {results['recall_macro']:.4f} |\n")
        f.write(f"| F1 (macro) | {results['f1_macro']:.4f} |\n")
        f.write(f"| AUC-ROC (macro) | {results['auc_roc_macro']:.4f} |\n\n")
        f.write("**Per-class F1:**\n\n")
        f.write("| Class | F1 |\n|---|---|\n")
        for cls, f1 in results["per_class_f1"].items():
            f.write(f"| {cls} | {f1:.4f} |\n")
        f.write(f"\n![confusion matrix](results/{args.model_name}/confusion_matrix.png)\n")

    print(f"Saved everything to {out_dir}/")


if __name__ == "__main__":
    main()