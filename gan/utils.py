"""
Utility functions for GAN training.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


def gradient_penalty(critic, real_data, fake_data, conditions, device, lambda_gp=10.0):
    """
    Compute WGAN-GP gradient penalty.

    Args:
        critic: Discriminator/critic network
        real_data: (B, C, H, W) real patches
        fake_data: (B, C, H, W) fake patches
        conditions: (B, cond_dim) condition vectors
        device: torch device
        lambda_gp: Gradient penalty coefficient

    Returns:
        Scalar gradient penalty tensor
    """
    batch_size = real_data.size(0)

    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    # Critic score on interpolates
    d_interpolates = critic(interpolates, conditions)

    # Gradients w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return penalty


def weights_init(m):
    """Custom weights initialization for GAN modules."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_image_grid(tensor_batch, path, nrow=8, normalize=True):
    """
    Save a batch of tensors as an image grid.

    Args:
        tensor_batch: (B, C, H, W) tensor in range [-1, 1] or [0, 1]
        path: Output file path
        nrow: Number of images per row
        normalize: If True, scale [-1, 1] to [0, 255]
    """
    from torchvision.utils import make_grid

    if normalize:
        tensor_batch = (tensor_batch + 1) / 2  # [-1,1] -> [0,1]
    tensor_batch = tensor_batch.clamp(0, 1)

    grid = make_grid(tensor_batch, nrow=nrow, padding=2)
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    im.save(path)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    import os

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
