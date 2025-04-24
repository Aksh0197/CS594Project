import torch.nn as nn
import torchvision.models as models
from models.student_resnet18 import get_student_model

student = get_student_model()

def get_student_model(num_classes=10, pretrained=False):
    """
    Returns a ResNet-18 student model compatible with CIFAR-10.

    Args:
        num_classes (int): Number of output classes (10 for CIFAR-10).
        pretrained (bool): Whether to use ImageNet pretrained weights.

    Returns:
        model (nn.Module): A ResNet-18 model with a custom output head.
    """
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    
    # Replace the classifier head to match CIFAR-10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

