from datasets import load_dataset
from transformers import AutoTokenizer

# def get_dataset_and_tokenizer(model_name: str = "roberta-large", max_length: int = 128):
def get_dataset_and_tokenizer(
                      model_name: str = "roberta-large",
                      max_length: int = 128,
                      train_size: int = None,
                      val_size: int = None,
                      seed: int = 42
                  ):

    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Subsample if needed
    if train_size:
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(train_size))
    if val_size:
        dataset["test"] = dataset["test"].shuffle(seed=seed).select(range(val_size))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized["train"], tokenized["test"], tokenized["test"], tokenizer
