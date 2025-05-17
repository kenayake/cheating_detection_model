from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim

from roboflow import Roboflow

from models.classification import base_model
from models.classification.base_model import BaseModel, multi_train
from models.classification.focal_loss import FocalLoss
from wakepy import keep


def train_densenet():
    models_dict = {
        "densenet121": models.densenet121,
    }
    base_weights = [
        models.DenseNet121_Weights.DEFAULT,
    ]
    optims = lambda params: {
        "sgd_nesterov_1e-3": optim.SGD(params, lr=1e-3, momentum=0.9, nesterov=True),
        "sgd_nesterov_5e-4": optim.SGD(params, lr=5e-4, momentum=0.9, nesterov=True),
        "sgd_nesterov_1e-4": optim.SGD(params, lr=1e-4, momentum=0.9, nesterov=True),
        "sgd_nesterov_5e-5": optim.SGD(params, lr=5e-5, momentum=0.9, nesterov=True),
        "sgd_nesterov_1e-5": optim.SGD(params, lr=1e-5, momentum=0.9, nesterov=True),
        "adam_1e-3": optim.Adam(params, lr=1e-3),
        "adam_5e-4": optim.Adam(params, lr=5e-4),
        "adam_1e-4": optim.Adam(params, lr=1e-4),
        "adam_5e-5": optim.Adam(params, lr=5e-5),
        "adam_1e-5": optim.Adam(params, lr=1e-5),
        "sgd_1e-3": optim.SGD(params, lr=1e-3, momentum=0.9),
        "sgd_5e-4": optim.SGD(params, lr=5e-4, momentum=0.9),
        "sgd_1e-4": optim.SGD(params, lr=1e-4, momentum=0.9),
        "sgd_5e-5": optim.SGD(params, lr=5e-5, momentum=0.9),
        "sgd_1e-5": optim.SGD(params, lr=1e-5, momentum=0.9),
    }
    epochs = 100
    dt_version = 1
    model_type = "densenet"
    
    multi_train(model_type, dt_version, models_dict, base_weights, optims, epochs)

if __name__ == "__main__":
    train_densenet()