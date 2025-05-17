from colorama import init
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim

from roboflow import Roboflow

from models.classification import base_model
from models.classification.base_model import BaseModel, initialize_roboflow, multi_train
from models.classification.focal_loss import FocalLoss
from wakepy import keep

def train_vit():
    models_dict = {
        "vit_b_16": models.vit_b_16,
    }
    base_weights = [
        models.ViT_B_16_Weights.DEFAULT,
    ]
    optims = lambda params: {
        "sgd_1e-3": optim.SGD(params, lr=1e-3, momentum=0.9),
        # "sgd_5e-4": optim.SGD(params, lr=5e-4, momentum=0.9),
    }
    epochs = 100
    dt_version = 5
    model_type = "vit"
    
    multi_train(model_type, dt_version, models_dict, base_weights, optims, epochs)

if __name__ == "__main__":
    train_vit()
    
    # weights = models.ViT_B_16_Weights.DEFAULT
    # base_model = models.vit_b_32(weights=weights)
    # criterion = FocalLoss(alpha=[0.75, 0.6, 1.0, 0.69], task_type="multi-class", num_classes=4)
    # optimizer = optim.SGD(base_model.parameters(), momentum=0.9, lr=1e-4)
    # dataset = initialize_roboflow(1)
    