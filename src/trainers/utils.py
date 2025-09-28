import torch
from torch.utils.data import DataLoader

def get_dataloaders(train_ds, val_ds, batch_size=16):
    """
    Returns PyTorch DataLoaders for train and validation sets.
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

def check_batch(dataloader, tokenizer):
    """
    Prints one batch to verify correctness.
    """
    batch = next(iter(dataloader))
    print("Keys:", batch.keys())
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Attention Mask shape:", batch["attention_mask"].shape)
    print("Labels shape:", batch["labels"].shape)
    print("Decoded[0]:", tokenizer.decode(batch["input_ids"][0]))
