import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler

from models.teacher_bert import get_teacher_model
from data_loader.sst2_loader import get_sst2_dataloaders

def train_teacher_nlp(epochs=3, batch_size=32, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load SST-2 Data
    trainloader, valloader = get_sst2_dataloaders(batch_size=batch_size)
    
    # Load Teacher Model
    model = get_teacher_model()
    model.to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for batch in trainloader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=masks)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"[Epoch {epoch+1}] Loss: {running_loss:.2f} | Train Acc: {acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), "teacher_bert_sst2.pth")
    print("✅ Teacher model (BERT) saved as 'teacher_bert_sst2.pth'.")

if __name__ == "__main__":
    train_teacher_nlp()
