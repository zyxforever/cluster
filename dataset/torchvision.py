from torchvision.transforms import ToTensor, Compose
from torchvision.datasets import FashionMNIST, MNIST, SVHN, USPS

class _AbstractTorchvisionDataset:

    dataset_class = NotImplementedError
    name = NotImplementedError
    label_shift = 0
    def __init__(self,cfg):
        dataset = self.dataset_class(root=cfg['dataset']['path'],transforms=self.transform)

    @property
    def transform(self):
        return Compose([ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label + self.label_shift

class MNISTDataset(_AbstractTorchvisionDataset):
    dataset_class = MNIST
    name = 'mnist'
    n_classes = 10
    n_channels = 1
    img_size = (28, 28)