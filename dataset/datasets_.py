from __future__ import print_function
import torch.utils.data as data
import torchvision.transforms as transforms

class DomainNetDataset(data.Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.labels = label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]

        if len(img.split()) != 3:
            img = transforms.Grayscale(num_output_channels=3)(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

class PACSDataset(data.Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.labels = label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]

        if len(img.split()) != 3:
            img = transforms.Grayscale(num_output_channels=3)(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

class OfficeHomeDataset(data.Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.labels = label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]

        if len(img.split()) != 3:
            img = transforms.Grayscale(num_output_channels=3)(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

class DatasetSplit(data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label