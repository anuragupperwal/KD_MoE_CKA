import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import json, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "Teacher": "anuragupperwal/teacher-finetuned-roberta",
    "Student": "anuragupperwal/student-finetuned-distilbert-base"
}

MAX_LENGTH = 256
BATCH_SIZE = 32
RESULTS_DIR = "./model_comparison_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading AG News test dataset...")
dataset = load_dataset("ag_news")["test"]

texts = dataset["text"]
labels = dataset["label"]
label_names = ["World", "Sports", "Business", "Sci/Tech"]

def evaluate_model(model_name, model_id):
    print(f"\nEvaluating: {model_name} ({model_id})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_labels = labels[i : i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec_macro = precision_score(all_labels, all_preds, average="macro")
    rec_macro = recall_score(all_labels, all_preds, average="macro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    report = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    # ROC-AUC (macro, one-vs-rest)
    try:
        auc_macro = roc_auc_score(
            np.eye(len(label_names))[all_labels], all_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        auc_macro = None

    #STORE RESULTS
    result = {
        "Model": model_name,
        "HF_Repo": model_id,
        "Accuracy": acc,
        "Precision(Macro)": prec_macro,
        "Recall(Macro)": rec_macro,
        "F1(Macro)": f1_macro,
        "F1(Weighted)": f1_weighted,
        "ROC-AUC(Macro)": auc_macro,
        "Classification_Report": report,
        "Confusion_Matrix": cm.tolist(),
    }

    with open(f"{RESULTS_DIR}/{model_name.lower()}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

#RUN EVALUATION FOR BOTH MODELS
results = []
for name, repo in MODELS.items():
    results.append(evaluate_model(name, repo))

#CREATE COMPARISON TABLE
df = pd.DataFrame(results)
summary_cols = [
    "Model", "Accuracy", "Precision(Macro)", "Recall(Macro)",
    "F1(Macro)", "F1(Weighted)", "ROC-AUC(Macro)"
]
summary_df = df[summary_cols]
summary_df.to_csv(f"{RESULTS_DIR}/comparison_summary.csv", index=False)

# Markdown table for reports/presentation
md_table = summary_df.to_markdown(index=False)
with open(f"{RESULTS_DIR}/comparison_summary.md", "w") as f:
    f.write(md_table)

print("\nMODEL PERFORMANCE COMPARISON")
print(md_table)
print("\nDetailed metrics saved to:", RESULTS_DIR)