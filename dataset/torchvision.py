from torchvision import transforms

from torchvision.datasets import MNIST
from torch.utils.data.dataset import Dataset as TorchDataset

class _AbstractTorchvisionDataset(TorchDataset):

    dataset_class = NotImplementedError
    name = NotImplementedError
    n_classes = NotImplementedError
    n_channels = NotImplementedError
    img_size = NotImplementedError
    test_split_only = False
    label_shift = 0
    n_samples = None
    def __init__(self):
        pass 
class MNISTDataset(_AbstractTorchvisionDataset):
    dataset_class = MNIST
    name = 'mnist'
    n_classes = 10
    n_channels = 1
    img_size = (28, 28)

