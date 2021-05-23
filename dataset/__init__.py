from .torchvision import MNISTDataset

def get_dataset(dataset_name):
    return {
        'mnist': MNISTDataset
    }[dataset_name]