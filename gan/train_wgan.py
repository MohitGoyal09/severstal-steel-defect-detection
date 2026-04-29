"""
WGAN-GP training script for conditional steel defect generation.

Usage:
    python -m gan.train_wgan --config gan/config.yml

Or programmatically:
    from gan.train_wgan import train_gan
    train_gan(config_dict)
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from gan.dataset import DefectPatchDataset
from gan.models import ConditionalGenerator, PatchGANCritic
from gan.utils import gradient_penalty, save_image_grid, set_seed


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent GAN checkpoint in the directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def resume_from_checkpoint(checkpoint_path, G, D, optimizer_G, optimizer_D, device):
    """Load model and optimizer states from a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(
            f"Warning: checkpoint {checkpoint_path} not found. Starting from scratch."
        )
        return 0

    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    G.load_state_dict(checkpoint["generator_state_dict"])
    D.load_state_dict(checkpoint["critic_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    print(f"Resumed training from epoch {start_epoch}")
    return start_epoch


def train_gan(config):
    """Main WGAN-GP training loop."""
    # --- Setup ---
    seed = config.get("seed", 42)
    set_seed(seed)

    device_str = config.get("device", "auto")
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")

    # --- Hyperparameters ---
    z_dim = config.get("z_dim", 100)
    cond_dim = config.get("cond_dim", 7)
    ngf = config.get("ngf", 64)
    ndf = config.get("ndf", 64)
    lr_g = config.get("lr_g", 0.0002)
    lr_d = config.get("lr_d", 0.0002)
    beta1 = config.get("beta1", 0.5)
    beta2 = config.get("beta2", 0.999)
    batch_size = config.get("batch_size", 16)
    n_epochs = config.get("n_epochs", 200)
    n_critic = config.get("n_critic", 5)
    lambda_gp = config.get("lambda_gp", 10.0)
    sample_interval = config.get("sample_interval", 10)
    checkpoint_interval = config.get("checkpoint_interval", 20)

    # --- Dataset ---
    csv_path = config["csv_path"]
    image_root = config["image_root"]
    patch_size = tuple(config.get("patch_size", [256, 256]))

    dataset = DefectPatchDataset(
        csv_path=csv_path,
        image_root=image_root,
        patch_size=patch_size,
        min_defect_area=config.get("min_defect_area", 50),
        return_mask=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    print(f"Dataset size: {len(dataset)} patches")

    # --- Models ---
    G = ConditionalGenerator(
        z_dim=z_dim, cond_dim=cond_dim, ngf=ngf, output_channels=1
    ).to(device)
    D = PatchGANCritic(input_channels=1, cond_dim=cond_dim, ndf=ndf).to(device)

    # --- Optimizers ---
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))

    # --- Logging ---
    save_dir = Path(config.get("save_dir", "saved/gan"))
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = save_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Fixed noise for consistent sample generation
    fixed_z = torch.randn(64, z_dim, device=device)
    # Sample random conditions for fixed samples
    fixed_cond = torch.zeros(64, cond_dim, device=device)
    for i in range(64):
        cls = i % 4
        size = (i // 4) % 3
        fixed_cond[i, cls] = 1.0
        fixed_cond[i, 4 + size] = 1.0

    # --- Resume ---
    start_epoch = 0
    if config.get("resume"):
        resume_path = config["resume"]
        if resume_path.lower() == "latest":
            resume_path = find_latest_checkpoint(checkpoint_dir)
        if resume_path:
            start_epoch = resume_from_checkpoint(
                resume_path, G, D, optimizer_G, optimizer_D, device
            )

    # --- Training Loop ---
    g_losses = []
    d_losses = []

    for epoch in range(start_epoch, n_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # batch: (patch, mask, condition)
            if len(batch) == 3:
                real_patches, real_masks, conditions = batch
            else:
                real_patches, conditions = batch
                real_masks = None

            real_patches = real_patches.to(device)
            conditions = conditions.to(device)
            batch_size_current = real_patches.size(0)

            # ---------------------
            # Train Critic (D)
            # ---------------------
            for _ in range(n_critic):
                optimizer_D.zero_grad()

                # Real patches
                d_real = D(real_patches, conditions)
                d_real_loss = d_real.mean()

                # Fake patches
                z = torch.randn(batch_size_current, z_dim, device=device)
                fake_patches = G(z, conditions)
                d_fake = D(fake_patches.detach(), conditions)
                d_fake_loss = d_fake.mean()

                # Gradient penalty
                gp = gradient_penalty(
                    D,
                    real_patches,
                    fake_patches.detach(),
                    conditions,
                    device,
                    lambda_gp,
                )

                # Total D loss
                d_loss = d_fake_loss - d_real_loss + gp
                d_loss.backward()
                optimizer_D.step()

            # ---------------------
            # Train Generator (G)
            # ---------------------
            optimizer_G.zero_grad()

            z = torch.randn(batch_size_current, z_dim, device=device)
            fake_patches = G(z, conditions)
            d_fake_for_g = D(fake_patches, conditions)
            g_loss = -d_fake_for_g.mean()

            # Optional: reconstruction loss using real masks to guide location
            if real_masks is not None and config.get("use_reconstruction_loss", False):
                # Heuristic: encourage generated intensity where mask is present
                # This is a simple L1 on masked region intensity variance
                rec_loss = ((fake_patches * real_masks).abs().mean()) * config.get(
                    "recon_weight", 10.0
                )
                g_loss = g_loss + rec_loss

            g_loss.backward()
            optimizer_G.step()

            # Logging
            if batch_idx % 50 == 0:
                print(
                    f"[Epoch {epoch}/{n_epochs}] [Batch {batch_idx}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                    f"[GP: {gp.item():.4f}]"
                )

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        # Save sample images
        if epoch % sample_interval == 0 or epoch == n_epochs - 1:
            G.eval()
            with torch.no_grad():
                fake_samples = G(fixed_z, fixed_cond)
                save_image_grid(
                    fake_samples, str(sample_dir / f"epoch_{epoch:04d}.png"), nrow=8
                )
            G.train()
            print(f"Saved samples to {sample_dir / f'epoch_{epoch:04d}.png'}")

        # Save checkpoints
        if epoch % checkpoint_interval == 0 or epoch == n_epochs - 1:
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": G.state_dict(),
                    "critic_state_dict": D.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "config": config,
                },
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
            )
            print(
                f"Saved checkpoint to {checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'}"
            )

    # Save final models
    torch.save(G.state_dict(), save_dir / "G_final.pth")
    torch.save(D.state_dict(), save_dir / "D_final.pth")
    print(f"Training complete. Final models saved to {save_dir}")

    return G, D


def main():
    parser = argparse.ArgumentParser(
        description="Train conditional WGAN-GP for steel defects"
    )
    parser.add_argument(
        "--config", type=str, default="gan/config.yml", help="Path to config YAML"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from. Use 'latest' to auto-pick the most recent.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.resume:
        config["resume"] = args.resume

    train_gan(config)


if __name__ == "__main__":
    main()
