import random
import pandas as pd
import torch
import pytorch_lightning as pl

from age-classification.network import ConvolutionalNetwork
from age-classification.datasets import ImageDataset, DataModule
from age-classification.create_path_label_list import create_path_label_list

class_names = ['YOUNG', 'MIDDLE', 'OLD']


def main():
    data = pd.read_csv('data.csv')
    path_label = create_path_label_list(data)
    random.seed(42)
    path_label = random.sample(path_label, 1000)

    dataset = ImageDataset(path_label)
    dataset.setup()
    datamodule = DataModule(dataset)
    datamodule.setup()

    model = ConvolutionalNetwork()
    trainer = pl.Trainer(max_epochs=30)
    trainer.fit(model, datamodule)

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
