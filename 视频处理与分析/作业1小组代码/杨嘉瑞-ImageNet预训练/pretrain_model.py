import torch
import torch.nn as nn
from torchvision import models


def get_pretrain_model(mode):
    model = models.resnet101(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 8)
    if mode == 1:
        for name, p in model.named_parameters():
            if 'fc' not in name:
                p.requires_grad = False
    return model