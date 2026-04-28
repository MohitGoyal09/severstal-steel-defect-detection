"""
Quick visual inspection script for DefectPatchDataset.

Usage:
    python -m gan.inspect_patches
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from gan.dataset import DefectPatchDataset


def save_image_grid_from_tensors(tensor_batch, path, nrow=4):
    """Save a batch of (C,H,W) tensors as a single image grid."""
    # tensor_batch: (B, C, H, W) in range [0, 1]
    grid = make_grid(tensor_batch, nrow=nrow, padding=2)
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    Image.fromarray(ndarr.squeeze() if ndarr.shape[2] == 1 else ndarr).save(path)


def main():
    csv_path = "data/train.csv"
    image_root = "data/train_images"
    out_path = Path("gan_samples/debug_patches.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = DefectPatchDataset(
        csv_path=csv_path,
        image_root=image_root,
        patch_size=(256, 256),
        min_defect_area=50,
        return_mask=False,  # we only want patch + condition for viz
    )

    print(f"Total patches in dataset: {len(dataset)}")

    # Class distribution
    class_counts = np.zeros(4, dtype=int)
    size_counts = np.zeros(3, dtype=int)
    for sample in dataset.samples:
        class_counts[sample["class_id"]] += 1
        size_counts[sample["size_bucket"]] += 1

    print("Class distribution:")
    for i in range(4):
        print(f"  Class {i + 1}: {class_counts[i]}")

    print("Size distribution:")
    labels = ["small (<500px)", "medium (500-5000px)", "large (>5000px)"]
    for i in range(3):
        print(f"  {labels[i]}: {size_counts[i]}")

    # Sample 16 random patches
    n_show = 16
    indices = random.sample(range(len(dataset)), min(n_show, len(dataset)))
    patches = []
    for idx in indices:
        patch, cond = dataset[idx]
        patches.append(patch)

    patches_tensor = torch.stack(patches)  # (B, 1, H, W)
    save_image_grid_from_tensors(patches_tensor, str(out_path), nrow=4)
    print(f"Saved {len(patches)} sample patches to {out_path}")


if __name__ == "__main__":
    main()
