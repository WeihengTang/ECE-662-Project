"""
AlexNet-FB for MNIST.

Identical architecture to AlexNetDCF, but the filter atoms are fixed
Fourier-Bessel (FB) bases — only the coefficients a_k are learned.

This is a thin wrapper around AlexNetDCF with bases_grad=False and
initializer='FB'.  It exists as a separate class for clarity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.alexnet_dcf import AlexNetDCF


class AlexNetFB(AlexNetDCF):
    """
    AlexNet with fixed Fourier-Bessel filter atoms.

    Args:
        num_bases  (int): K — number of FB atoms (≤ max_fb_bases(3) = 8 for 3×3).
        num_classes (int): Output classes.
    """
    def __init__(self, num_bases: int = 8, num_classes: int = 10):
        super().__init__(
            num_bases=num_bases,
            bases_grad=False,
            initializer='FB',
            num_classes=num_classes,
        )
