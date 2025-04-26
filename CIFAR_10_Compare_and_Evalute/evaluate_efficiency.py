import os
import time
import torch
from models.teacher_resnet50 import get_teacher_model
from models.student_model import get_student_model
from data_loader.cifar10_loader import get_cifar10_dataloaders

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("⚠️ CodeCarbon not installed. Carbon tracking will be skipped.")


def get_model_size(filepath):
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_size=(1, 3, 32, 32), device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()
    inputs = torch.randn(input_size).to(device)

    # Warm-up
    for _ in range(10):
        _ = model(inputs)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(inputs)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / 100
    return avg_inference_time * 1000  # milliseconds

# def evaluate_efficiency():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load teacher
#     teacher = get_teacher_model()
#     teacher.load_state_dict(torch.load("teacher_resnet50.pth", map_location=device))

#     # Load student
#     student = get_student_model()
#     student.load_state_dict(torch.load("student_resnet18_kd.pth", map_location=device))

#     # Model sizes
#     teacher_size = get_model_size("teacher_resnet50.pth")
#     student_size = get_model_size("student_resnet18_kd.pth")

#     # Parameter counts
#     teacher_params = count_parameters(teacher)
#     student_params = count_parameters(student)

#     # Inference speeds
#     teacher_latency = measure_inference_time(teacher)
#     student_latency = measure_inference_time(student)

#     # Carbon tracking (optional)
#     if CODECARBON_AVAILABLE:
#         tracker = EmissionsTracker(project_name="KD_Student_Inference")
#         tracker.start()
#         _ = measure_inference_time(student)
#         emissions = tracker.stop()
#     else:
#         emissions = None

#     # Report
#     print("\n📊 Teacher vs Student Efficiency Comparison")
#     print("-------------------------------------------")
#     print(f"Teacher Model Size: {teacher_size:.2f} MB")
#     print(f"Student Model Size: {student_size:.2f} MB\n")
    
#     print(f"Teacher Parameters: {teacher_params:,}")
#     print(f"Student Parameters: {student_params:,}\n")

#     print(f"Teacher Inference Time: {teacher_latency:.2f} ms")
#     print(f"Student Inference Time: {student_latency:.2f} ms\n")

#     if emissions is not None:
#         print(f"🌍 Carbon Emissions during Student Inference: {emissions:.6f} kg CO₂")
#     else:
#         print("🌍 Carbon Emissions: (CodeCarbon not available)")



def evaluate_efficiency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load teacher
    teacher = get_teacher_model()
    teacher.load_state_dict(torch.load("teacher_resnet50.pth", map_location=device))

    # Load student
    student = get_student_model()
    student.load_state_dict(torch.load("student_resnet18_kd.pth", map_location=device))

    # Model sizes
    teacher_size = get_model_size("teacher_resnet50.pth")
    student_size = get_model_size("student_resnet18_kd.pth")

    # Parameter counts
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)

    # Inference speeds
    teacher_latency = measure_inference_time(teacher)
    student_latency = measure_inference_time(student)

    # Carbon tracking separately
    emissions_teacher = None
    emissions_student = None

    if CODECARBON_AVAILABLE:
        # Track Teacher emissions
        tracker_teacher = EmissionsTracker(project_name="KD_Teacher_Inference")
        tracker_teacher.start()
        _ = measure_inference_time(teacher)
        emissions_teacher = tracker_teacher.stop()

        # Track Student emissions
        tracker_student = EmissionsTracker(project_name="KD_Student_Inference")
        tracker_student.start()
        _ = measure_inference_time(student)
        emissions_student = tracker_student.stop()

    # Report
    print("\n📊 Teacher vs Student Efficiency Comparison")
    print("-------------------------------------------")
    print(f"Teacher Model Size: {teacher_size:.2f} MB")
    print(f"Student Model Size: {student_size:.2f} MB\n")
    
    print(f"Teacher Parameters: {teacher_params:,}")
    print(f"Student Parameters: {student_params:,}\n")

    print(f"Teacher Inference Time: {teacher_latency:.2f} ms")
    print(f"Student Inference Time: {student_latency:.2f} ms\n")

    if emissions_teacher is not None and emissions_student is not None:
        print(f"🌍 Teacher Inference Carbon Emissions: {emissions_teacher:.6f} kg CO₂")
        print(f"🌍 Student Inference Carbon Emissions: {emissions_student:.6f} kg CO₂")
    else:
        print("🌍 Carbon Emissions: (CodeCarbon not available)")

if __name__ == "__main__":
    evaluate_efficiency()

