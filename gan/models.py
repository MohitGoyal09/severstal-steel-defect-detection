"""
Conditional GAN models for steel defect synthesis.

Architecture:
  - ConditionalGenerator: DCGAN-style with condition injection
  - PatchGANCritic: PatchGAN discriminator for WGAN-GP
"""

import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    """
    Conditional DCGAN-style generator for 256x256 defect patches.

    Args:
        z_dim: Dimension of latent noise vector
        cond_dim: Dimension of condition vector (e.g., 7 for 4 classes + 3 sizes)
        ngf: Base number of generator features
        output_channels: 1 for grayscale defects, 2 if also outputting mask
    """

    def __init__(self, z_dim=100, cond_dim=7, ngf=64, output_channels=1):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.output_channels = output_channels

        # Project and reshape: (z + cond) -> 4x4 spatial
        self.project = nn.Sequential(
            nn.Linear(z_dim + cond_dim, ngf * 8 * 4 * 4),
            nn.ReLU(True),
        )

        # Upsample blocks: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.block1 = self._upblock(ngf * 8, ngf * 8)  # 4 -> 8
        self.block2 = self._upblock(ngf * 8, ngf * 4)  # 8 -> 16
        self.block3 = self._upblock(ngf * 4, ngf * 2)  # 16 -> 32
        self.block4 = self._upblock(ngf * 2, ngf)  # 32 -> 64
        self.block5 = self._upblock(ngf, ngf // 2)  # 64 -> 128
        self.block6 = self._upblock(ngf // 2, ngf // 4)  # 128 -> 256

        self.to_rgb = nn.Sequential(
            nn.Conv2d(ngf // 4, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _upblock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z, cond):
        """
        Args:
            z: (B, z_dim) noise
            cond: (B, cond_dim) condition vector
        Returns:
            (B, output_channels, 256, 256) generated patch
        """
        x = torch.cat([z, cond], dim=1)  # (B, z_dim + cond_dim)
        x = self.project(x)  # (B, ngf*8*4*4)
        x = x.view(x.size(0), -1, 4, 4)  # (B, ngf*8, 4, 4)

        x = self.block1(x)  # 8
        x = self.block2(x)  # 16
        # TODO: Add ASPP bottleneck here at 16x16 for multi-scale context
        x = self.block3(x)  # 32
        x = self.block4(x)  # 64
        # TODO: Add spatial attention before final upsampling
        x = self.block5(x)  # 128
        x = self.block6(x)  # 256

        out = self.to_rgb(x)
        return out


class PatchGANCritic(nn.Module):
    """
    PatchGAN discriminator / critic for WGAN-GP.
    Outputs a scalar score per patch.

    Args:
        input_channels: 1 for grayscale
        ndf: Base number of discriminator features
    """

    def __init__(self, input_channels=1, cond_dim=7, ndf=64):
        super().__init__()
        self.cond_dim = cond_dim

        # Condition projection to spatial
        self.cond_proj = nn.Linear(cond_dim, 256 * 256)

        # Downsample blocks: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.block1 = self._downblock(
            input_channels + 1, ndf, normalize=False
        )  # 256 -> 128
        self.block2 = self._downblock(ndf, ndf * 2)  # 128 -> 64
        self.block3 = self._downblock(ndf * 2, ndf * 4)  # 64 -> 32
        self.block4 = self._downblock(ndf * 4, ndf * 8)  # 32 -> 16
        self.block5 = self._downblock(ndf * 8, ndf * 8)  # 16 -> 8
        self.block6 = self._downblock(ndf * 8, ndf * 8, normalize=False)  # 8 -> 4

        self.to_score = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0)

        self._init_weights()

    def _downblock(self, in_ch, out_ch, normalize=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x, cond):
        """
        Args:
            x: (B, 1, 256, 256) image patch
            cond: (B, cond_dim) condition vector
        Returns:
            (B, 1, 1, 1) scalar score
        """
        # Project condition to spatial and concatenate as extra channel
        cond_spatial = self.cond_proj(cond).view(-1, 1, 256, 256)
        x = torch.cat([x, cond_spatial], dim=1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        score = self.to_score(x)
        return score


class UNetGenerator(nn.Module):
    """
    U-Net style conditional generator (alternative / future upgrade).
    Encoder-decoder with skip connections for better mask-conditioned generation.

    Args:
        z_dim: Latent dimension
        cond_dim: Condition vector dimension
        ngf: Base features
        output_channels: 1 or 2
    """

    def __init__(self, z_dim=100, cond_dim=7, ngf=64, output_channels=1):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        # Condition embedding
        self.cond_embed = nn.Linear(cond_dim, 256 * 256)
        self.z_proj = nn.Linear(z_dim, 256 * 256)

        in_ch = 1 + 1 + 1  # image channel + cond channel + z channel

        # Encoder
        self.enc1 = self._conv_block(in_ch, ngf)  # 256
        self.enc2 = self._conv_block(ngf, ngf * 2)  # 128
        self.enc3 = self._conv_block(ngf * 2, ngf * 4)  # 64
        self.enc4 = self._conv_block(ngf * 4, ngf * 8)  # 32
        self.enc5 = self._conv_block(ngf * 8, ngf * 8)  # 16

        # Decoder with skip connections
        self.dec5 = self._up_block(ngf * 8, ngf * 8)  # 16 -> 32
        self.dec4 = self._up_block(ngf * 16, ngf * 4)  # 32 -> 64
        self.dec3 = self._up_block(ngf * 8, ngf * 2)  # 64 -> 128
        self.dec2 = self._up_block(ngf * 4, ngf)  # 128 -> 256
        self.dec1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, output_channels, 3, padding=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True),
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z, cond):
        cond_spatial = self.cond_embed(cond).view(-1, 1, 256, 256)
        z_spatial = self.z_proj(z).view(-1, 1, 256, 256)

        # Start with zeros or mean image; inject z and cond as channels
        x = torch.zeros(z.size(0), 1, 256, 256, device=z.device)
        x = torch.cat([x, cond_spatial, z_spatial], dim=1)

        e1 = self.enc1(x)  # 128
        e2 = self.enc2(e1)  # 64
        e3 = self.enc3(e2)  # 32
        e4 = self.enc4(e3)  # 16
        e5 = self.enc5(e4)  # 8

        d5 = self.dec5(e5)  # 16
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.dec4(d5)  # 32
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4)  # 64
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)  # 128
        d2 = torch.cat([d2, e1], dim=1)
        out = self.dec1(d2)  # 256

        return out
