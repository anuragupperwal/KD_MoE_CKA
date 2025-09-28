from transformers import AutoModelForSequenceClassification

def get_teacher_model(model_name: str = "roberta-large", num_labels: int = 4):
    """
    Loads the teacher model (RoBERTa-large by default).
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model
