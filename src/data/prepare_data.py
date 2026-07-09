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


def copy_split(df: pd.DataFrame, raw_dir: Path, out_dir: Path, split_name: str):
    missing = 0
    rows = []
    for _, row in df.iterrows():
        src = find_image_path(row["image_id"], raw_dir)
        if src is None:
            missing += 1
            continue
        class_dir = out_dir / split_name / row["dx"]
        class_dir.mkdir(parents=True, exist_ok=True)
        dst = class_dir / f"{row['image_id']}.jpg"
        if not dst.exists():
            shutil.copy2(src, dst)
        rows.append(row)

    if missing:
        print(f"  [warn] {missing} images listed in metadata but not found on disk")

    split_df = pd.DataFrame(rows)
    split_df.to_csv(out_dir / f"{split_name}_labels.csv", index=False)
    return split_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    metadata_path = raw_dir / cfg["data"]["metadata_csv"]

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find {metadata_path}. Download HAM10000 first — see the "
            f"docstring at the top of this file for the kaggle CLI command."
        )

    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {len(df)} images, {df['lesion_id'].nunique()} unique lesions")
    print("Class distribution:\n", df["dx"].value_counts())

    train_df, val_df, test_df = lesion_level_split(
        df,
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        seed=cfg["data"]["seed"],
    )

    print(f"\nSplit sizes (by image, after lesion-level split):")
    print(f"  train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\nCopying {name} split...")
        copy_split(split_df, raw_dir, processed_dir, name)

    print(f"\nDone. Processed data written to {processed_dir}/")
    print("Folder layout: processed_dir/{train,val,test}/{class_name}/*.jpg")


if __name__ == "__main__":
    main()

