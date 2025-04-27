# Script to train teacher models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models.teacher_resnet50 import get_teacher_model
from data_loader.cifar10_loader import get_cifar10_dataloaders

def train_teacher(epochs=20, batch_size=128, lr=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_teacher_model().to(device)
    trainloader, testloader = get_cifar10_dataloaders(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.2f} | Accuracy: {acc:.2f}%")
        scheduler.step()

    torch.save(model.state_dict(), "teacher_resnet50.pth")
    print("✅ Teacher model saved to 'teacher_resnet50.pth'.")

if __name__ == "__main__":
    train_teacher()
# This script trains a teacher model on CIFAR-10 using ResNet-50 architecture.
# It uses SGD optimizer with momentum and a step learning rate scheduler.
# The model is saved after training.
# The training process includes loss calculation and accuracy tracking.
# The script is designed to be run as a standalone program.
# It uses PyTorch for model training and data loading.
# The CIFAR-10 dataset is used for training and testing.
# The script is modular and can be easily modified for different datasets or models.
# The training loop includes model evaluation and learning rate adjustment.