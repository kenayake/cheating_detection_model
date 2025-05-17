from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim

from roboflow import Roboflow

from models.classification import base_model
from models.classification.base_model import BaseModel, multi_train
from models.classification.focal_loss import FocalLoss
from wakepy import keep


def train_alexnet():
    models_dict = {
        "alexnet": models.alexnet,
    }
    base_weights = [
        models.AlexNet_Weights.DEFAULT,
    ]
    optims = lambda params: {
        "adamw_1e-5": optim.AdamW(params, lr=1e-4, weight_decay=1e-4),
    }
    
    epochs = 100
    dt_version = 1
    model_type = "alexnet"
    
    multi_train(model_type, dt_version, models_dict, base_weights, optims, epochs)

if __name__ == "__main__":
    train_alexnet()