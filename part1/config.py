import os

# Path to the precomputed Bessel function zero table (from DCFNet reference repo)
path_to_bessel = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'DCFNet-Pytorch', 'bessel.npy'
)
