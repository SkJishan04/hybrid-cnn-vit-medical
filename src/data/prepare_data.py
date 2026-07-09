"""
Prepares HAM10000 for training.

Why this script exists and isn't just "put images in folders":
HAM10000 has multiple photos of the SAME lesion (same lesion_id).
A naive random split leaks the same lesion into both train and val/test,
which inflates your validation accuracy and invalidates your comparison
between models. This script splits at the lesion_id level instead.

Usage:
    1. Download from Kaggle first (requires a kaggle.json API token in ~/.kaggle/):
       kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip

    2. Then run:
       python src/data/prepare_data.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_image_path(image_id: str, raw_dir: Path) -> Path | None:
    """HAM10000 ships images split across two folders (part_1 / part_2)."""
    for sub in ["HAM10000_images_part_1", "HAM10000_images_part_2", ""]:
        candidate = raw_dir / sub / f"{image_id}.jpg"
        if candidate.exists():
            return candidate
    return None


def lesion_level_split(df: pd.DataFrame, val_split: float, test_split: float, seed: int):
    """
    Split by lesion_id, not by row, so the same lesion never appears
    in two different splits.
    """
    lesion_ids = df["lesion_id"].unique()

    train_ids, temp_ids = train_test_split(
        lesion_ids, test_size=(val_split + test_split), random_state=seed
    )
    relative_test_size = test_split / (val_split + test_split)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=relative_test_size, random_state=seed
    )

    train_df = df[df["lesion_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["lesion_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


