from .torchvision import MNISTDataset
def get_dataset(name):
    return {
        'mnist': MNISTDataset
    }[name]