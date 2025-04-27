from transformers import DistilBertForSequenceClassification

def get_student_model(num_labels=2):
    """
    Loads DistilBERT-base-uncased for sequence classification.

    Args:
        num_labels (int): Number of output classes (2 for SST-2).

    Returns:
        model: DistilBERT model ready for fine-tuning.
    """
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels
    )
    return model
