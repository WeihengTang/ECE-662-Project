"""
Convolutional Denoising Autoencoder with DCF or FB encoder conv layers.

The three Conv2d layers in the *encoder* are replaced by Conv_DCF.
The decoder uses standard ConvTranspose2d layers throughout (typical in
practice because DCF decomposition is defined for Conv2d).

Controlling the variant:
    DenoisingAutoencoderDCF(num_bases=4, bases_grad=True,  initializer='random')
        → DCF: both atoms ψ_k and coefficients a_k are learned
    DenoisingAutoencoderDCF(num_bases=4, bases_grad=False, initializer='FB')
        → FB:  atoms are fixed Fourier-Bessel bases; only a_k is learned
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from dcf_layer import Conv_DCF


class DenoisingAutoencoderDCF(nn.Module):
    """
    Denoising autoencoder whose encoder conv layers use DCF or FB decomposition.

    Args:
        num_bases   (int) : K — number of atoms per conv layer.
        bases_grad  (bool): True  → learnable atoms (DCF from scratch).
                            False → fixed atoms (FB initialisation).
        initializer (str) : 'FB' or 'random'.
    """
    def __init__(self, num_bases: int = 8, bases_grad: bool = False,
                 initializer: str = 'FB'):
        super().__init__()

        def dcf(in_ch, out_ch):
            return Conv_DCF(in_ch, out_ch, kernel_size=3, padding=1,
                            num_bases=num_bases, bias=False,
                            bases_grad=bases_grad, initializer=initializer,
                            mode='mode1')

        self.encoder = nn.Sequential(
            dcf(1, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            dcf(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            dcf(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Decoder keeps standard transposed convolutions
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
        return self.decoder(self.encoder(x))
