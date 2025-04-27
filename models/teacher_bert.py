from transformers import BertForSequenceClassification

def get_teacher_model(num_labels=2):
    """
    Loads BERT-base-uncased for sequence classification.

    Args:
        num_labels (int): Number of output classes (2 for SST-2).

    Returns:
        model: BERT model ready for fine-tuning.
    """
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels
    )
    return model