from collections import Counter
import numpy as np
from sklearn.utils import compute_class_weight
from wakepy import keep
import sys
import os
from models.classification.dataset_loader import DatasetLoader
import models.classification.resnet.resnet_train as resnet_train
import models.classification.ViT.vit_train as vit_train
import models.classification.densenet.densenet_train as densenet_train
import models.classification.alexnet.alexnet_train as alexnet_train
import models.classification.efficientnet.efnet_train as efficientnet_train
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim

from roboflow import Roboflow

from models.classification import base_model
from models.classification.base_model import BaseModel, initialize_roboflow, multi_train
from models.classification.focal_loss import FocalLoss
from wakepy import keep

from torchvision import datasets

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def initialize_roboflow(dt_version):
    rf = Roboflow(api_key="m2wBNOl1tzsXZBfl8Lwv")
    project = rf.workspace("skripsi-gfit0").project("pointy-sticks")
    version = project.version(dt_version)
    dataset = version.download(
        "folder",
        location=f"dataset/classification/cheating-classification-v{dt_version}",
    )
    return dataset

if __name__ == "__main__":
    models_dict = {
        "vit_b_32": models.vit_b_32,
        "vit_l_32": models.vit_l_32,
    }
    base_weights = [
        models.ViT_B_32_Weights.DEFAULT,
        models.ViT_L_32_Weights.DEFAULT,
    ]
    optims = lambda params: {
        "adamw_1e-5": optim.AdamW(params,lr=1e-5, weight_decay=1e-4),
    }
    epochs = 30
    dt_version = 3
    group = "nyoba"
    
    multi_train(group, dt_version, models_dict, base_weights, optims, epochs, batch_size=16)
