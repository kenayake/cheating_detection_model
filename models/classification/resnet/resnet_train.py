import os
import sys
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim

from roboflow import Roboflow

from models.classification import base_model
from models.classification.base_model import BaseModel, multi_train
from models.classification.focal_loss import FocalLoss
from wakepy import keep

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def train_resnet():
    model_group = "no test set"
    models_dict = {
        "resnet18": models.resnet18,
    }
    base_weights = [
        models.ResNet18_Weights.DEFAULT,
    ]
    optims = lambda params: {
        "adamw_batch128": optim.AdamW(params, weight_decay=1e-4),
        # "adamw_1e-4": optim.AdamW(params, lr=1e-4),
        # "adam_1e-5": optim.Adam(params, lr=1e-5),
    }
    epochs = 30
    dt_version = 6
    
    multi_train(model_group, dt_version, models_dict, base_weights, optims, epochs, batch_size=128)

if __name__ == "__main__":
    train_resnet()
    
    # weights = models.ResNet50_Weights.DEFAULT
    # base_model = models.resnet50(weights=weights)
    # optimizer = optim.A(base_model.parameters(), momentum=0.9, lr=1e-4)
    # dataset = initialize_roboflow()
    # data_dir = dataset.location
    # savedir = "saved_checkpoints/resnet/resnet50_v1_adam"
    # criterion = FocalLoss(alpha=[0.75, 0.6, 1.0, 0.69], task_type="multi-class", num_classes=4)
    # batch_size = 16
    
    # trainer = BaseModel(base_model, weights, data_dir, savedir, optimizer, criterion, batch_size=batch_size)
    # trainer.train(epochs)