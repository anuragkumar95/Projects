import os
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader


class MNIST_Dataset(Dataset):
    """Custom MNIST Dataset"""
    def __init__(self, root_dir, set='Train', transforms=[], filter_labels = []):
        """
        Args:
            root_dir  : Root directory where MNIST is stored. If the directory 
                        doesn't exist, it will create a new directory and 
                        download the MNIST dataset.
            transforms: List of transforms to be applied.
        """
        self.set = set
        self.transforms = transforms
        if not os.path.exists(root_dir):
            download = True
        else:
            download = False
        if set == 'Train':
            self.mnist = datasets.MNIST(root=root_dir, download=download, train=True)  
            self.filter_set(filter_labels)
            indices = range(min(50000, int(0.9*len(self.mnist))))
            self.mnist = [self.mnist[i] for i in indices]
        
        if set == 'Valid':
            self.mnist = datasets.MNIST(root=root_dir, download=download, train=True)
            self.filter_set(filter_labels)
            ind_start = min(50000, int(0.9*len(self.mnist)))
            self.mnist = [self.mnist[i+ind_start] for i in range(len(self.mnist) - ind_start)]
                            
        if set == 'Test':
            self.mnist = datasets.MNIST(root=root_dir, download=download, train=False)
            self.filter_set(filter_labels)
           
        
    def filter_set(self, filter_labels):
        dataset = []
        if len(filter_labels) == 0:
            return
        for i, (img, label) in enumerate(self.mnist):
            if label in filter_labels:
                dataset.append((img, label))
        self.mnist = dataset
        
    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        for i, transform in enumerate(self.transforms):
            img = transform(img)
        return img, label