"""
Convolutional Denoising Autoencoder for MNIST.

Architecture
------------
Encoder:
    Conv(1→32, 3, pad=1) + BN + ReLU → MaxPool(2)   [32×14×14]
    Conv(32→64, 3, pad=1) + BN + ReLU → MaxPool(2)  [64×7×7]
    Conv(64→64, 3, pad=1) + BN + ReLU               [64×7×7]

Decoder:
    ConvTranspose(64→64, 3, pad=1) + BN + ReLU              [64×7×7]
    ConvTranspose(64→32, 3, stride=2, pad=1, out_pad=1) + BN + ReLU  [32×14×14]
    ConvTranspose(32→1,  3, stride=2, pad=1, out_pad=1) + Sigmoid    [1×28×28]

Input : noisy MNIST image  (N, 1, 28, 28)
Output: reconstructed clean image (N, 1, 28, 28)
Loss  : MSE(output, clean_image)
PSNR  : 10 * log10(1 / MSE)  [images in [0,1]]
"""
import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                      # → 32×14×14

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                      # → 64×7×7

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
