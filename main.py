import wandb
import sys
sys.path.append('/home/twang/OPERA')

import os
import argparse
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, random_split

from predict import eval_llm_open_ended
from models.models import AudioQAModel
from models.audio_encoder import initialize_pretrained_model
from dataloader import AudioQADataset
from train import train_model
from huggingface_hub import login

login("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")

def set_random_seeds(seed=0):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_type", type=str, default="deepseek-ai/DeepSeek-R1", help="Model ID for LLaMA")
    parser.add_argument("--setting", type=str, default="lora", choices=("lora", "frozen", 'prefixtuning', "p_tuning", "prompttuning", "unfrozen"))
    parser.add_argument("--mapping_type", type=str, default="Transformer")
    parser.add_argument("--prefix_length", type=int, default=8)
    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--data_csv", type=str, default="/home/twang/qas.csv", help="Path to QA pairs CSV file")
    parser.add_argument("--dataset_path", type=str, default="/home/twang/audio", help="Path to audio dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--opera_checkpoint_path", type=str, default="/home/twang/encoder-operaCE.ckpt")  
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters_to_accumulate", type=int, default=4)
    parser.add_argument("--validation_step", type=int, default=100)
    parser.add_argument("--out_dir", default="/home/twang/checkpoints")
    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--load_pretrained", type=bool, default=False) 
    args = parser.parse_args()
    set_random_seeds(args.seed)

    return args

if __name__ == "__main__":
    args = parse_argument()
    wandb.init(
        project="my-awesome-project",
        config={
            "learning_rate": args.lr,
            "architecture": "Phi with LoRA",
            "dataset": "AudioQA",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
         }
    )

    suffix = f"audioqa_prefixlength_{args.prefix_length}_mapping_{args.mapping_type}_seed_{args.seed}_modeltype_{args.llm_type.replace('/', '')}_setting_{args.setting}"
    args.out_dir = os.path.join(args.out_dir, suffix)

    # Load the dataset and split into train and val
    full_dataset = AudioQADataset(
        data_dir=args.dataset_path,
        csv_data_path=args.data_csv,
        prefix_length=args.prefix_length, 
        model_type=args.llm_type,
    )

    # Split the dataset
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_dataloader = DataLoader(dataset=train_dataset, num_workers=2, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, num_workers=2, batch_size=args.batch_size, shuffle=False, drop_last=True)

    opera_model = initialize_pretrained_model(pretrain="operaCE") 
    model = AudioQAModel(
        llm_type=args.llm_type,
        opera_checkpoint_path=args.opera_checkpoint_path,
        prefix_length=args.prefix_length,
        clip_length=args.clip_length,
        setting=args.setting,
        mapping_type=args.mapping_type,
        fine_tune_opera=False,
        args=args,
    )

    ignore_index = full_dataset.ignore_index

    # Train or Evaluate
    if not args.eval:
        model = train_model(train_dataloader, model, ignore_index, args)
    else:
        checkpoint = os.path.join(args.out_dir, "open_ended_latest.pt")
        if args.verbose:
            print(f">> Loading pre-trained model {checkpoint}!")
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cuda")), strict=False)
        else:
            raise ValueError("Please provide a valid path for loading checkpoint")
            
        eval_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        eval_llm_open_ended(model, eval_dataloader, args)
