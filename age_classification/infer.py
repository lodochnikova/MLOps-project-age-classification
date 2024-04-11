import random

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from sklearn.metrics import classification_report
from torchvision import transforms

from age_classification.create_path_label_list import create_path_label_list
from age_classification.datasets import DataModule, ImageDataset
from age_classification.network import ConvolutionalNetwork


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    class_names = cfg.data.class_names

    model = ConvolutionalNetwork(cfg)
    model.load_state_dict(torch.load(cfg.model.path_to_model))

    data = pd.read_csv(cfg.data.path_to_data)
    path_label = create_path_label_list(data)
    random.seed(cfg.data.seed)
    path_label = random.sample(path_label, cfg.data.num_samples)

    transform = transforms.Compose(
        [
            transforms.Resize(cfg.transform.resize),
            transforms.CenterCrop(cfg.transform.center_crop),
            transforms.ToTensor(),
            transforms.Normalize(
                cfg.transform.normalize_mean, cfg.transform.normalize_std
            ),
        ]
    )

    dataset = ImageDataset(
        path_label,
        batch_size=cfg.train.batch_size,
        train_test_split=cfg.train.train_test_split,
        transform=transform,
    )
    dataset.setup()
    datamodule = DataModule(
        dataset,
        transform=transform,
        batch_size=cfg.train.batch_size,
        train_test_split=cfg.train.train_test_split,
        seed=cfg.data.seed,
    )
    datamodule.setup()

    trainer = Trainer()
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    trainer.test(model=model, dataloaders=test_loader)

    device = torch.device(cfg.infer.device)

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

    print(
        classification_report(
            y_true, y_pred, target_names=class_names, digits=cfg.infer.digits
        )
    )


if __name__ == "__main__":
    main()
