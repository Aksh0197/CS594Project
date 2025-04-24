# Script to train student using KD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models.teacher_resnet50 import get_teacher_model
from models.student_resnet18 import get_student_model
from data_loader.cifar10_loader import get_cifar10_dataloaders
from distillation.soft_label_kd import SoftLabelDistillationLoss

def train_student_with_kd(epochs=20, batch_size=128, lr=0.1, alpha=0.7, temperature=4.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-10 data
    trainloader, testloader = get_cifar10_dataloaders(batch_size)

    # Load teacher and freeze
    teacher = get_teacher_model().to(device)
    teacher.load_state_dict(torch.load("teacher_resnet50.pth", map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Load student
    student = get_student_model().to(device)

    # Loss and optimizer
    criterion = SoftLabelDistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        student.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Teacher outputs (no grad!)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            # Student outputs
            student_outputs = student(inputs)

            # Calculate KD loss
            loss = criterion(student_outputs, teacher_outputs, labels)

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"[Epoch {epoch+1}] Loss: {running_loss:.2f} | Train Acc: {acc:.2f}%")
        scheduler.step()

    # Save student model
    torch.save(student.state_dict(), "student_resnet18_kd.pth")
    print("✅ Student model saved to 'student_resnet18_kd.pth'.")

if __name__ == "__main__":
    train_student_with_kd()
# This script trains a student model using knowledge distillation from a pre-trained teacher model.
# It uses the CIFAR-10 dataset and ResNet-18 architecture for the student model.
# The training process includes loss calculation and accuracy tracking.
# The script is designed to be run as a standalone program.
# It uses PyTorch for model training and data loading.
# The CIFAR-10 dataset is used for training and testing.
# The script is modular and can be easily modified for different datasets or models.
# The training loop includes model evaluation and learning rate adjustment.
# The student model is saved after training.
# The script uses soft label knowledge distillation for training the student model.
# The script includes detailed logging of the training process.
# The script uses SGD optimizer with momentum and a step learning rate scheduler.
# The script includes hyperparameters for temperature and alpha for knowledge distillation.
# The script uses a custom loss function for knowledge distillation.