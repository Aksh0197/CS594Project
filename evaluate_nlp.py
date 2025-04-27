import torch
from models.teacher_bert import get_teacher_model
from models.student_distilbert import get_student_model
from data_loader.sst2_loader import get_sst2_dataloaders

def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=inputs, attention_mask=masks)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    acc = 100. * correct / total
    return acc

def evaluate_nlp_models(batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SST-2 Validation Data
    _, val_loader = get_sst2_dataloaders(batch_size=batch_size)

    # Load and evaluate Teacher (BERT)
    teacher = get_teacher_model()
    teacher.load_state_dict(torch.load("teacher_bert_sst2.pth", map_location=device))
    teacher_acc = evaluate_model(teacher, val_loader, device)

    # Load and evaluate Student (DistilBERT)
    student = get_student_model()
    student.load_state_dict(torch.load("student_distilbert_kd_sst2.pth", map_location=device))
    student_acc = evaluate_model(student, val_loader, device)

    print("\n📊 NLP Model Evaluation on SST-2 (Validation Set)")
    print("----------------------------------------------")
    print(f"✅ Teacher (BERT) Accuracy: {teacher_acc:.2f}%")
    print(f"✅ Student (DistilBERT) Accuracy: {student_acc:.2f}%")

if __name__ == "__main__":
    evaluate_nlp_models()