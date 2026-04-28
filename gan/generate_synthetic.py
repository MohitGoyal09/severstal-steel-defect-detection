"""
Generate synthetic defect patches from a trained GAN and filter by discriminator score.

Usage:
    python -m gan.generate_synthetic \
        --generator saved/gan/G_final.pth \
        --critic saved/gan/D_final.pth \
        --output synthetic/ \
        --n_samples 2000
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image

from gan.models import ConditionalGenerator, PatchGANCritic
from gan.utils import set_seed


def generate_synthetic(
    generator_path,
    critic_path,
    output_root,
    n_samples,
    z_dim=100,
    cond_dim=7,
    ngf=64,
    ndf=64,
    batch_size=32,
    device=None,
    seed=42,
    score_lower_pct=30,
    score_upper_pct=80,
):
    """
    Generate synthetic defect images and filter by discriminator score.

    Args:
        generator_path: Path to trained generator weights (.pth)
        critic_path: Path to trained critic weights (.pth)
        output_root: Directory to save synthetic/images, synthetic/masks
        n_samples: Total number of samples to generate
        score_lower_pct: Lower percentile for quality filtering
        score_upper_pct: Upper percentile for quality filtering
    """
    set_seed(seed)

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    # --- Load models ---
    G = ConditionalGenerator(
        z_dim=z_dim, cond_dim=cond_dim, ngf=ngf, output_channels=1
    ).to(device)
    D = PatchGANCritic(input_channels=1, cond_dim=cond_dim, ndf=ndf).to(device)

    G.load_state_dict(torch.load(generator_path, map_location=device))
    D.load_state_dict(torch.load(critic_path, map_location=device))
    G.eval()
    D.eval()

    # --- Setup output dirs ---
    out_root = Path(output_root)
    img_dir = out_root / "images"
    mask_dir = out_root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution: oversample rare classes
    # Severstal approx distribution: class 3 is rarest, then 1, 2, 4
    # We'll generate equal numbers per class by default, but can customize
    samples_per_class = n_samples // 4
    class_counts = {
        0: samples_per_class,
        1: samples_per_class,
        2: samples_per_class,
        3: samples_per_class,
    }

    all_records = []
    global_idx = 0

    with torch.no_grad():
        for cls_id in range(4):
            count = class_counts[cls_id]
            print(f"Generating {count} samples for class {cls_id + 1}...")

            cls_records = []

            for batch_start in range(0, count, batch_size):
                batch_end = min(batch_start + batch_size, count)
                current_batch = batch_end - batch_start

                z = torch.randn(current_batch, z_dim, device=device)
                cond = torch.zeros(current_batch, cond_dim, device=device)
                cond[:, cls_id] = 1.0
                # Random size bucket
                size_buckets = torch.randint(0, 3, (current_batch,))
                for i, sb in enumerate(size_buckets):
                    cond[i, 4 + sb] = 1.0

                fake = G(z, cond)
                scores = D(fake, cond).view(-1).cpu().numpy()
                fake = fake.cpu().numpy()

                for i in range(current_batch):
                    img_arr = fake[i, 0]  # (H, W)

                    # Denormalize from [-1, 1] or [0, 1] to [0, 255]
                    if img_arr.min() < 0:
                        img_arr = (img_arr + 1) / 2
                    img_arr = np.clip(img_arr * 255, 0, 255).astype(np.uint8)

                    # Approximate mask from generated image
                    # Defects are usually darker than steel background
                    # Use Otsu or simple threshold
                    _, mask_arr = cv2.threshold(
                        img_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )

                    fname = f"img_{global_idx:06d}.png"
                    cls_dir = f"cls_{cls_id + 1}"
                    (img_dir / cls_dir).mkdir(exist_ok=True)
                    (mask_dir / cls_dir).mkdir(exist_ok=True)

                    Image.fromarray(img_arr).save(img_dir / cls_dir / fname)
                    Image.fromarray(mask_arr).save(mask_dir / cls_dir / fname)

                    cls_records.append(
                        {
                            "filename": f"{cls_dir}/{fname}",
                            "class_id": cls_id + 1,
                            "score": float(scores[i]),
                        }
                    )
                    global_idx += 1

            all_records.extend(cls_records)

    # --- Quality filtering ---
    df = pd.DataFrame(all_records)
    df.to_csv(out_root / "metadata_raw.csv", index=False)

    print(f"Generated {len(df)} raw samples.")
    print(
        f"Score stats: min={df.score.min():.4f}, max={df.score.max():.4f}, "
        f"mean={df.score.mean():.4f}, median={df.score.median():.4f}"
    )

    # Per-class filtering
    filtered = []
    for cls_id in range(4):
        cls_df = df[df["class_id"] == cls_id + 1]
        if len(cls_df) == 0:
            continue
        low = cls_df["score"].quantile(score_lower_pct / 100.0)
        high = cls_df["score"].quantile(score_upper_pct / 100.0)
        kept = cls_df[(cls_df["score"] >= low) & (cls_df["score"] <= high)]
        filtered.append(kept)
        print(
            f"Class {cls_id + 1}: kept {len(kept)}/{len(cls_df)} "
            f"(scores in [{low:.4f}, {high:.4f}])"
        )

    filtered_df = pd.concat(filtered, ignore_index=True)
    filtered_df.to_csv(out_root / "metadata_filtered.csv", index=False)
    print(
        f"Filtered dataset: {len(filtered_df)} samples saved to {out_root / 'metadata_filtered.csv'}"
    )

    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic defects from trained GAN"
    )
    parser.add_argument(
        "--generator", type=str, required=True, help="Path to G_final.pth"
    )
    parser.add_argument("--critic", type=str, required=True, help="Path to D_final.pth")
    parser.add_argument(
        "--output", type=str, default="synthetic", help="Output directory"
    )
    parser.add_argument(
        "--n_samples", type=int, default=2000, help="Total samples to generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for generation"
    )
    parser.add_argument("--z_dim", type=int, default=100, help="Latent dimension")
    parser.add_argument("--cond_dim", type=int, default=7, help="Condition dimension")
    parser.add_argument("--ngf", type=int, default=64, help="Generator base features")
    parser.add_argument("--ndf", type=int, default=64, help="Critic base features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--score_lower_pct", type=int, default=30, help="Lower score percentile"
    )
    parser.add_argument(
        "--score_upper_pct", type=int, default=80, help="Upper score percentile"
    )
    args = parser.parse_args()

    generate_synthetic(
        generator_path=args.generator,
        critic_path=args.critic,
        output_root=args.output,
        n_samples=args.n_samples,
        z_dim=args.z_dim,
        cond_dim=args.cond_dim,
        ngf=args.ngf,
        ndf=args.ndf,
        batch_size=args.batch_size,
        seed=args.seed,
        score_lower_pct=args.score_lower_pct,
        score_upper_pct=args.score_upper_pct,
    )


if __name__ == "__main__":
    main()
