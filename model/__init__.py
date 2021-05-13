from .linear_vae import VAE
from .dac import MNISTNetwork
def get_model(name):
    return {
        'vae':VAE,
        'dac_mnist':MNISTNetwork
    }[name]