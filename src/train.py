"""
Generic training loop — works for the ResNet baseline today, and for the
ViT baseline / hybrid model later, since `model.name` in the config is
the only thing that changes.

Usage:
    python src/train.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# lets this run as `python src/train.py` from the repo root, not just from inside src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import HAM10000Dataset
from models.baseline_resnet import build_resnet50
from utils.losses import FocalLoss, compute_class_weights
from utils.metrics import evaluate, format_results


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> torch.nn.Module:
    name = cfg["model"]["name"]
    if name == "resnet50":
        return build_resnet50(
            num_classes=cfg["model"]["num_classes"],
            pretrained=cfg["model"]["pretrained"],
            drop_rate=cfg["model"]["drop_rate"],
        )
    elif name == "vit_small_patch16_224":
        raise NotImplementedError(
            "ViT baseline not wired up yet — this is next on the list."
        )
    elif name == "hybrid":
        raise NotImplementedError(
            "Hybrid model not wired up yet — build the CNN and ViT baselines first."
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(dataloader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        mode=cfg["wandb"]["mode"],
        config=cfg,
        name=f"{cfg['model']['name']}_run",
    )

    classes = cfg["classes"]

    train_ds = HAM10000Dataset(
        cfg["data"]["processed_dir"], classes, "train", cfg["data"]["image_size"]
    )
    val_ds = HAM10000Dataset(
        cfg["data"]["processed_dir"], classes, "val", cfg["data"]["image_size"]
    )

    # NOTE: weighted sampler removed — combined with focal loss it was
    # overcorrecting, tanking F1 on the majority class (nv). Focal loss
    # alone handles the imbalance; plain shuffling works better here.
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )

    model = build_model(cfg).to(device)

    class_weights = compute_class_weights(train_ds.class_counts()).to(device)
    criterion = FocalLoss(gamma=cfg["train"]["focal_gamma"], class_weights=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1 = 0.0
    patience_counter = 0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        scheduler.step()

        val_results = evaluate(model, val_loader, device, classes)
        print(f"\nEpoch {epoch + 1}/{cfg['train']['epochs']}  train_loss={train_loss:.4f}")
        print(format_results(val_results))

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_results["accuracy"],
            "val_f1_macro": val_results["f1_macro"],
            "val_auc_roc_macro": val_results["auc_roc_macro"],
            "lr": scheduler.get_last_lr()[0],
        })

        if val_results["f1_macro"] > best_f1:
            best_f1 = val_results["f1_macro"]
            patience_counter = 0
            torch.save(
                model.state_dict(),
                ckpt_dir / f"{cfg['model']['name']}_best.pt",
            )
            print(f"  -> new best model saved (val F1 = {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stop_patience"]:
                print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for "
                      f"{cfg['train']['early_stop_patience']} epochs)")
                break

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()