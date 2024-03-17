from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import random_split


transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_label, transform=None):
        self.path_label = path_label
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class ImageDataset(pl.LightningDataModule):
    def __init__(self, path_label, batch_size=32, transform = transform):
        super().__init__()
        self.path_label = path_label
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        dataset = CustomDataset(self.path_label, self.transform)
        dataset_size = len(dataset)
        print(dataset_size)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size

        self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        self.test_dataset = torch.utils.data.Subset(dataset, range(train_size, dataset_size))

    def __len__(self):
        if self.train_dataset is not None:
            return len(self.train_dataset)
        elif self.test_dataset is not None:
            return len(self.test_dataset)
        else:
            return 0

    def __getitem__(self, index):
        if self.train_dataset is not None:
            return self.train_dataset[index]
        elif self.test_dataset is not None:
            return self.test_dataset[index]
        else:
            raise IndexError("Index out of range. The dataset is empty.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class DataModule(pl.LightningDataModule):

    def __init__(self, dataset, transform=transform, batch_size=32):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.dataset = dataset

    def setup(self, stage=None):
        n_data = len(self.dataset)
        n_train = int(0.8 * n_data)
        n_test = n_data - n_train

        torch.manual_seed(12345)
        train_dataset, test_dataset = random_split(self.dataset, [n_train, n_test])
        torch.manual_seed(torch.seed())

        self.train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = DataLoader(test_dataset, batch_size=self.batch_size)

    def train_dataloader(self):
        return self.train_dataset

    def test_dataloader(self):
        return self.test_dataset
