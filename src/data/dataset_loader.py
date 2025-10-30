# dataset_loader.py

import json

def load_synthetic_dataset(jsonl_path):
    """
    Load the synthetic multi-task reasoning dataset for baseline evaluation.
    Each entry follows:
    {
      "id": int,
      "crop": str,
      "region": str,
      "season": str,
      "farmer_query": str,
      "advisor_response": str,
      "category": str,
      "rationale": str,
      "task_labels": {
          "classification": str,
          "generation_target": str,
          "rationale_target": str
      }
    }
    """
    dataset = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                dataset.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return dataset