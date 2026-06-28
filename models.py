import torch
import torch.nn as nn
from torchvision import models

def get_baseline_cnn(num_classes=3):
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 56 * 56, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return model

def get_mobilenet_v2(num_classes=3):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
