from transformers import AutoModelForSequenceClassification

def get_student_model(model_name: str = "distilbert-base-uncased", num_labels: int = 4):
    """
    Loads the student model (DistilBERT by default).
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model
