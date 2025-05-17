import numpy as np
from requests import get
from sklearn.utils import compute_class_weight
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

import torchvision

from PIL import Image


class DatasetLoader:
    def __init__(
        self,
        transforms,
        batch_size=16,
        num_workers=2,
    ):
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self, data_dir):
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=self.transforms
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "valid"), transform=self.transforms
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "test"), transform=self.transforms
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image).unsqueeze(0)
        image = image.to(self.device)

        return image
    
def get_class_weights(data_dir):
    dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train")
    )

    y = dataset.targets

    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    weights=torch.tensor(weights,dtype=torch.float)
    
    return weights


if __name__ == "__main__":
    pass
    # print(torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms())
