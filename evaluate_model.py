import torch
import os
from models.teacher_resnet50 import get_teacher_model
from data_loader.cifar10_loader import get_cifar10_dataloaders

def evaluate_teacher(model_path="teacher_resnet50.pth", batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔒 Check if model file exists and is not empty
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' does not exist.")
        return
    elif os.path.getsize(model_path) < 1000:
        print(f"❌ Error: Model file '{model_path}' seems to be corrupted or incomplete.")
        return

    # Load test set only
    _, testloader = get_cifar10_dataloaders(batch_size)

    # Load and evaluate the model
    model = get_teacher_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"✅ Teacher Model Test Accuracy: {acc:.2f}%")

def evaluate_student(model_path="student_resnet18_kd.pth", batch_size=128):
    import torch
    from models.student_model import get_student_model
    from data_loader.cifar10_loader import get_cifar10_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader = get_cifar10_dataloaders(batch_size)

    model = get_student_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"✅ Student Model Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    evaluate_teacher()
    evaluate_student()
# This script evaluates the teacher and student models on the CIFAR-10 test set.
# It checks if the model files exist and are not corrupted before loading.
# It uses PyTorch for model evaluation and data loading.
# The script prints the test accuracy for both models.
# The CIFAR-10 dataset is used for evaluation.
# The script is designed to be run as a standalone program.
# It can be easily modified to evaluate different models or datasets.
# The evaluation process includes loading the model, running inference, and calculating accuracy.
# The script uses the CIFAR-10 data loader to fetch test data.