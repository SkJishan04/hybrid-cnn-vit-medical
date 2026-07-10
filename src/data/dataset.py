"""
PyTorch Dataset for HAM10000, plus:
  - hair-artifact removal (a lot of HAM10000 photos have hair covering
    part of the lesion — this hurts a CNN more than a ViT since CNNs
    key on local texture, so it's worth normalizing away)
  - class-weighted sampling to help with the ~67% nevus class imbalance
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler


def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    Morphological black-hat filter + inpainting to remove hair artifacts.
    Standard preprocessing step for dermoscopic images (see DullRazor algorithm).
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return inpainted


def build_transforms(image_size: int, split: str) -> A.Compose:
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=25, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class HAM10000Dataset(Dataset):
    """
    Expects the folder layout produced by prepare_data.py:
        processed_dir/{split}/{class_name}/*.jpg
    """

    def __init__(self, root_dir: str, classes: list[str], split: str,
                 image_size: int = 224, use_hair_removal: bool = True):
        self.root_dir = Path(root_dir) / split
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.use_hair_removal = use_hair_removal
        self.transform = build_transforms(image_size, split)

        self.samples: list[tuple[Path, int]] = []
        for class_name in classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root_dir}. Did you run prepare_data.py?"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_hair_removal:
            image = remove_hair(image)

        transformed = self.transform(image=image)
        return transformed["image"], label

    def class_counts(self) -> torch.Tensor:
        counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Inverse-frequency sampling so each training batch sees a more
        balanced mix of classes, rather than being dominated by nevi (~67%).
        """
        counts = self.class_counts()
        class_weights = 1.0 / counts.clamp(min=1)
        sample_weights = torch.tensor(
            [class_weights[label] for _, label in self.samples]
        )
        return WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )