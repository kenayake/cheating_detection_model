from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim

from roboflow import Roboflow

from models.classification import base_model
from models.classification.base_model import BaseModel, multi_train
from models.classification.focal_loss import FocalLoss

def train_efnet(batch):
    models_dict = {
        "efficientnetb0": models.efficientnet_b0,
        # "efficientnet_b1": models.efficientnet_b1,
        # "efficientnet_b2": models.efficientnet_b2,
    }
    base_weights = [
        models.EfficientNet_B0_Weights.DEFAULT,
        # models.EfficientNet_B1_Weights.DEFAULT,
        # models.EfficientNet_B2_Weights.DEFAULT,
    ]
    optims = lambda params: {
        "adamw_1e-4": optim.AdamW(params, lr=1e-5, weight_decay=1e-4),
    }
    epochs = 30
    multi_train("no test set", 6, models_dict, base_weights, optims, epochs, batch_size=batch)

if __name__ == "__main__":
    train_efnet()