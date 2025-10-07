import os, glob, shutil
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from huggingface_hub import HfApi

def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    save_path=None,
    save_every=2000,
    resume_checkpoint=None,
    use_amp=True,
    tokenizer=None,
    repo_id=None,  
):
    """
    Training loop with epoch + step checkpointing.
    Saves/resumes model, optimizer, scheduler, scaler (if AMP), global_step, epoch.
    Tracks train loss, val accuracy, val loss, val F1 (macro, weighted).
    """

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # scaler = torch.cuda.amp.GradScaler(enabled=use_amp) #depriciated
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    start_epoch = 0
    global_step = 0

    # Resume if checkpoint provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"] + 1
        if "scaler_state_dict" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"Resumed from checkpoint: {resume_checkpoint} at epoch {start_epoch}, step {global_step}")

    model.to(device)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_losses = []
    val_accuracies = []
    val_losses = []
    val_f1_macro = []
    val_f1_weighted = []

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss

            if loss.dim() > 0:
                loss = loss.mean()

            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)   
            lr_scheduler.step()

            global_step += 1
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

            #print loss
            if global_step % 100 == 0:
                print(f"Step {global_step}, Loss = {loss.item():.4f}")


            # # Step-level checkpoint
            # if save_path and global_step % save_every == 0:
                # step_dir = f"{save_path}_step{global_step}"
                # os.makedirs(step_dir, exist_ok=True)
                # torch.save({
                #     "model_state_dict": model.state_dict(),
                #     "optimizer_state_dict": optimizer.state_dict(),
                #     "scheduler_state_dict": lr_scheduler.state_dict(),
                #     "scaler_state_dict": scaler.state_dict() if use_amp else None,
                #     "global_step": global_step,
                #     "epoch": epoch
                # }, os.path.join(step_dir, "checkpoint.pt"))
                # print(f"Checkpoint saved at {step_dir}")

                # # keep last 2 step checkpoints
                # all_ckpts = sorted(glob.glob(f"{save_path}_step*"), key=os.path.getmtime)
                # if len(all_ckpts) > 2:
                #     shutil.rmtree(all_ckpts[0])
                #     print(f"Deleted old checkpoint: {all_ckpts[0]}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        val_acc, val_loss, f1_macro, f1_weight = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        val_f1_macro.append(f1_macro)
        val_f1_weighted.append(f1_weight)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | F1(macro): {f1_macro:.4f} | F1(weighted): {f1_weight:.4f}"
        )

        # Epoch-level checkpoint
        if save_path:
            epoch_dir = save_path 
            # epoch_dir = f"{save_path}_epoch{epoch+1}"
            os.makedirs(epoch_dir, exist_ok=True)
            # torch.save({
            #     "model_state_dict": model.state_dict(),
            #     "optimizer_state_dict": optimizer.state_dict(),
            #     "scheduler_state_dict": lr_scheduler.state_dict(),
            #     "scaler_state_dict": scaler.state_dict() if use_amp else None,
            #     "global_step": global_step,
            #     "epoch": epoch,
            # }, os.path.join(epoch_dir, "checkpoint.pt"))
            # print(f"Checkpoint saved for epoch {epoch} at {epoch_dir}")

            if isinstance(model, torch.nn.DataParallel):
                model.module.save_pretrained(epoch_dir)
            else:
                model.save_pretrained(epoch_dir)
            if tokenizer:
                tokenizer.save_pretrained(epoch_dir)

            if repo_id and (epoch == epochs - 1):
                # push to HuggingFace Hub
                print(f"Uploading model to Hugging Face Hub → {repo_id}")
                api = HfApi()
                api.upload_folder(
                    folder_path=save_path,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"Pushed epoch {epoch+1} to HuggingFace Hub")


            # keep last 2 epoch checkpoints
            all_epochs = sorted(glob.glob(f"{save_path}_epoch*"), key=os.path.getmtime)
            if len(all_epochs) > 2:
                shutil.rmtree(all_epochs[0])
                print(f"Deleted old epoch checkpoint: {all_epochs[0]}")

    return model, train_losses, val_accuracies, val_losses, val_f1_macro, val_f1_weighted

def evaluate_model(model, dataloader, device):
    """
    Evaluate accuracy, loss, macro/weighted F1 on validation set.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    loss_fn = CrossEntropyLoss()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

            total_loss += loss_fn(logits, batch["labels"]).item()

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(batch["labels"].detach().cpu().tolist())

    val_acc = correct / total
    val_loss = total_loss / len(dataloader)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    return val_acc, val_loss, f1_macro, f1_weighted
