import os
import sys
from dotenv import load_dotenv

# --- Ensure root path is available ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))     # src/evaluate
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))   # src
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))      # project root
sys.path.insert(0, ROOT_DIR)

# --- Load environment variables ---
env_path = os.path.join(ROOT_DIR, ".env")
if not load_dotenv(env_path):
    raise FileNotFoundError(f".env file not found at {env_path}")


from src.models.teacher_model import GeminiTeacher
from src.data.dataset_loader import load_synthetic_dataset
from src.trainers.utils import (
    compute_bleu, compute_rouge, compute_accuracy,
    compute_bertscore, compute_final_score
)

DATA_PATH = os.path.join(ROOT_DIR, "data", "synthetic_dataset.jsonl")
OUTPUT_PATH = os.path.join(SRC_DIR, "evaluate", "teacher_baseline_results.json")


import json
# import os
from tqdm import tqdm
# from src.models.teacher_model import GeminiTeacher
# from src.data.dataset_loader import load_synthetic_dataset
# from src.trainers.utils import compute_bleu, compute_rouge, compute_accuracy, compute_bertscore
from dotenv import load_dotenv

load_dotenv()
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

def main():
    api_key = os.getenv("GEMINI_API_KEY") 

    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in environment. Please set it in your .env file.")

    teacher = GeminiTeacher(api_key)
    dataset = load_synthetic_dataset(DATA_PATH)
    print(f"Loaded {len(dataset)} records for evaluation from {DATA_PATH}")

    results = []
    for record in tqdm(dataset, desc="Evaluating Teacher (Gemini 2.5 Flash)"):
        try:
            pred = teacher.predict(record)
        except Exception as e:
            print(f"Skipping ID {record['id']} due to error: {e}")
            continue

        results.append({
            "id": record.get("id", None),
            "true_category": record.get("category", None),
            "pred_category": pred.get("pred_category", None),
            "true_advisor_response": record.get("advisor_response", None),
            "pred_advisor_response": pred.get("pred_advisor_response", None),
            "true_rationale": record.get("model_thinking", None),
            "pred_rationale": pred.get("pred_rationale", None),
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved baseline results to {OUTPUT_PATH}")

    acc = compute_accuracy([r["true_category"] for r in results],
                           [r["pred_category"] for r in results])
    bleu = compute_bleu([r["true_advisor_response"] for r in results],
                        [r["pred_advisor_response"] for r in results])
    rouge = compute_rouge([r["true_advisor_response"] for r in results],
                          [r["pred_advisor_response"] for r in results])
    bert = compute_bertscore([r["true_rationale"] for r in results],
                             [r["pred_rationale"] for r in results])
    scores = compute_final_score(
        true_labels=[r["true_category"] for r in results],
        pred_labels=[r["pred_category"] for r in results],
        true_rationale=[r["true_rationale"] for r in results],
        pred_rationale=[r["pred_rationale"] for r in results],
        true_advisor=[r["true_advisor_response"] for r in results],
        pred_advisor=[r["pred_advisor_response"] for r in results],
    )
    reasoningScore = scores["ReasoningScore"]
    DomainScore = scores["DomainScore"]
    LanguageScore = scores["LanguageScore"]
    FinalScore = scores["FinalScore"]

    metrics = {
        "accuracy": acc,
        "bleu": bleu,
        "rougeL": rouge,
        "bertscore": bert,
        "reasoning_score": reasoningScore,
        "domain_score": DomainScore,
        "language_score": LanguageScore,
    }
        # "final_score": FinalScore
    with open("teacher_baseline_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\\nTeacher Baseline Metrics")
    print(f"Classification Accuracy: {acc:.4f}")
    print(f"Advisory BLEU: {bleu:.4f}")
    print(f"Advisory ROUGE-L: {rouge:.4f}")
    print(f"Rationale BERTScore: {bert:.4f}")
    print(f"Reaasoning Score:  {reasoningScore:.4f}")
    print(f"Domain Score:: {DomainScore:.4f}")
    print(f"Language Score: {LanguageScore:.4f}")
    print(f"Final Score: {FinalScore:.4f}")


if __name__ == "__main__":
    main()