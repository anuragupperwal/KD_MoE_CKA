from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataset_and_tokenizer(
                model_name: str = "roberta-large",
                max_length: int = 256,
                train_split_ratio: float = 0.1,
                seed: int = 42
            ):

    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_valid = dataset["train"].train_test_split(
        test_size=train_split_ratio,
        seed=seed
    )

    train_ds = train_valid["train"]
    val_ds = train_valid["test"]
    test_ds = dataset["test"]  # untouched, official AG News test set

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Train: {len(train_ds)} | Validation: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds, tokenizer




