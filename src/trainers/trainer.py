import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import os
from torch.nn import CrossEntropyLoss
import shutil, glob

def train_model(model, train_loader, val_loader, epochs, lr, device, save_path=None, save_every=2000, resume_checkpoint=None):
    """
    Training loop with epoch + step checkpointing.
    Saves and resumes model, optimizer, scheduler, global_step, epoch.
    Tracks train loss, val accuracy, val loss.
    """

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    start_epoch = 0
    global_step = 0

    # Resume if checkpoint provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        global_step = checkpoint['global_step']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {resume_checkpoint} at epoch {start_epoch}, step {global_step}")

    model.to(device)
    # Ensure save directory exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            progress_bar.set_postfix({"train_loss": loss.item()})


            # Step-level checkpoint
            if save_path and global_step % save_every == 0:
                step_dir = f"{save_path}_step{global_step}"
                os.makedirs(step_dir, exist_ok=True)

                # Hugging Face style
                # model.save_pretrained(step_dir)

                # PyTorch style
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch
                }, os.path.join(step_dir, "checkpoint.pt"))

                print(f"Checkpoint saved at {step_dir}")

                # Delete old checkpoints (keep last 2)
                all_ckpts = sorted(glob.glob(f"{save_path}_step*"), key=os.path.getmtime)
                if len(all_ckpts) > 2:
                    shutil.rmtree(all_ckpts[0])  # delete oldest
                    print(f"Deleted old checkpoint: {all_ckpts[0]}")



        # Average train loss
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        val_acc, val_loss = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Epoch-level checkpoint
        if save_path:
            epoch_dir = f"{save_path}_epoch{epoch+1}"
            os.makedirs(epoch_dir, exist_ok=True)

            # PyTorch style save
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'global_step': global_step,
                'epoch': epoch
            }, os.path.join(epoch_dir, "checkpoint.pt"))

            print(f"Checkpoint saved at {epoch_dir}")

            # Delete old checkpoints (keep last 2 epochs only)
            all_epochs = sorted(glob.glob(f"{save_path}_epoch*"), key=os.path.getmtime)
            if len(all_epochs) > 2:
                shutil.rmtree(all_epochs[0])
                print(f"Deleted old epoch checkpoint: {all_epochs[0]}")

                
        if not val_accuracies:
            val_acc, val_loss = evaluate_model(model, val_loader, device)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            print(f"Final Validation | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model, train_losses, val_accuracies, val_losses


def evaluate_model(model, dataloader, device):
    """
    Evaluate accuracy AND loss on validation set.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    loss_fn = CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            total_loss += loss_fn(outputs.logits, batch["labels"]).item()

    val_acc = correct / total
    val_loss = total_loss / len(dataloader)
    return val_acc, val_loss
