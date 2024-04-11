import random

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torchvision import transforms

from age_classification.create_path_label_list import create_path_label_list
from age_classification.datasets import DataModule, ImageDataset
from age_classification.network import ConvolutionalNetwork


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
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
        seed=cfg.data.seed_data_module,
    )
    datamodule.setup()

    model = ConvolutionalNetwork(cfg)
    trainer = pl.Trainer(max_epochs=cfg.train.max_epoch)
    trainer.fit(model, datamodule)

    torch.save(model.state_dict(), cfg.model.path_to_model)


if __name__ == "__main__":
    main()
