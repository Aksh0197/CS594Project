 # Dataset loading and preprocessing using SST2 dataset for Natural Langauge Processing
from datasets import load_dataset
from transformers import AutoTokenizer

def get_sst2_dataloaders(model_name='bert-base-uncased', batch_size=32, max_length=128):
    # Load SST-2 dataset
    dataset = load_dataset("glue", "sst2")
    # Limit training data to first 10,000 examples for faster training
    dataset['train'] = dataset['train'].select(range(10000))
    dataset['validation'] = dataset['validation'].select(range(1000))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenization function
    def tokenize_function(example):
        return tokenizer(
            example['sentence'],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # Apply tokenization
    encoded_dataset = dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    encoded_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # Create DataLoaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(encoded_dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(encoded_dataset['validation'], batch_size=batch_size)

    return train_loader, val_loader