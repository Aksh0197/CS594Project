import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler

from models.teacher_bert import get_teacher_model
from models.student_distilbert import get_student_model
from data_loader.sst2_loader import get_sst2_dataloaders

from distillation.soft_label_kd import SoftLabelDistillationLoss

def train_student_nlp(epochs=3, batch_size=32, lr=5e-5, temperature=4.0, alpha=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SST-2 data
    trainloader, valloader = get_sst2_dataloaders(batch_size=batch_size)

    # Load teacher model and freeze it
    teacher = get_teacher_model()
    teacher.load_state_dict(torch.load("teacher_bert_sst2.pth", map_location=device))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Load student model
    student = get_student_model()
    student.to(device)

    # Loss and optimizer
    criterion = SoftLabelDistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = optim.AdamW(student.parameters(), lr=lr)
    num_training_steps = epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training loop
    student.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for batch in trainloader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Teacher predictions (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=inputs, attention_mask=masks).logits

            # Student predictions
            student_outputs = student(input_ids=inputs, attention_mask=masks).logits

            # KD loss
            loss = criterion(student_outputs, teacher_outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, preds = torch.max(student_outputs, dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"[Epoch {epoch+1}] Loss: {running_loss:.2f} | Train Acc: {acc:.2f}%")

    # Save student model
    torch.save(student.state_dict(), "student_distilbert_kd_sst2.pth")
    print("✅ Student model (DistilBERT) saved as 'student_distilbert_kd_sst2.pth'.")

if __name__ == "__main__":
    train_student_nlp()