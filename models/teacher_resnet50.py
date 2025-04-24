import torch.nn as nn
import torchvision.models as models

def get_teacher_model(num_classes=10):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model