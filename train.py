import wandb
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm

def train_model(train_loader, model_obj, ignore_index, args):
    """Train the model using only the training set."""
    accelerator = Accelerator()
    device = accelerator.device
    model = model_obj.to(device)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    elif args.load_pretrained:
        checkpoint = os.path.join(args.out_dir, "open_ended_latest.pt")
        if args.verbose:
            print(f">> Loading pre-trained model {checkpoint}!", flush=True)
        if os.path.exists(checkpoint):
            model.load_state_dict(
                torch.load(checkpoint, map_location=device), strict=False
            )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.98))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_loader),
    )

    # Prepare model, optimizer, and data for distributed training
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    n_epochs = args.epochs
    accelerator.wait_for_everyone()
    shift = 10 if args.setting in ["p_tuning", "prompttuning"] else 0  

    for epoch in tqdm(range(n_epochs)):
        start_time = time.time()
        model.train()  # Ensure model is in training mode
        total_loss = 0.0

        for i, (prefix, tokens, mask, q_len) in tqdm(enumerate(train_loader), total=len(train_loader)):
            with accelerator.accumulate(model):
                prefix = prefix.to(accelerator.device).float()
                tokens = tokens.to(accelerator.device).long()
                mask = mask.to(accelerator.device).float()
                q_len = q_len.to(accelerator.device).long()

                outputs = model(prefix, tokens, mask, q_len)
                logits = outputs.logits

                loss = 0.0
                for b in range(logits.size(0)):
                    condensed_tokens = tokens[b, model.prefix_length + 1:]
                    condensed_logits = logits[b, shift + model.prefix_length:-1]
                    loss += nnf.cross_entropy(
                        condensed_logits.reshape(-1, logits.shape[-1]),
                        condensed_tokens.flatten(),
                        ignore_index=ignore_index
                    )
                loss = loss / logits.size(0)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                avg_loss = total_loss / (i + 1)
                if (i + 1) % 10 == 0:
                    wandb.log({"train_loss": avg_loss, "batch": i + 1, "epoch": epoch + 1})

        scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.out_dir, "open_ended_latest.pt"))

    return model  #  Ensure the trained model is returned
