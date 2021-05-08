from .linear_vae import VAE

def get_model(name):
    return {
        'vae':VAE
    }[name]