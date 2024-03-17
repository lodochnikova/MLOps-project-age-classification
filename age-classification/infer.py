import torch
import random
from pytorch_lightning import Trainer
from sklearn.metrics import classification_report
import pandas as pd

from create_path_label_list import create_path_label_list
from network import ConvolutionalNetwork
from datasets import ImageDataset, DataModule

class_names = ['YOUNG', 'MIDDLE', 'OLD']


def main():
    model = ConvolutionalNetwork()
    model.load_state_dict(torch.load("model.pth"))

    data = pd.read_csv('data.csv')
    path_label = create_path_label_list(data)
    random.seed(42)
    path_label = random.sample(path_label, 1000)

    dataset = ImageDataset(path_label)
    dataset.setup()
    datamodule = DataModule(dataset)
    datamodule.setup()

    trainer = Trainer()
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    trainer.test(model=model, dataloaders=test_loader)

    device = torch.device("cpu")

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for test_data in datamodule.test_dataloader():
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


if __name__ == "__main__":
    main()
