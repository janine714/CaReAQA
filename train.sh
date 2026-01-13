#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1   
#SBATCH --ntasks=2
#SBATCH --partition=gpu_a100             
#SBATCH --gres=gpu:1                     
#SBATCH --cpus-per-task=4               
#SBATCH --mem=32G                        
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out
set -x  # Enable script debugging

echo "Loading module 2023..."
module load 2023

echo "Loading CUDA/12.1.1..."
module load CUDA/12.1.1
module load cudnn/8.6  # Ensure cuDNN module is loaded if separate

echo "Updating LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$HOME/myenv/lib:$LD_LIBRARY_PATH

echo "Activating virtual environment..."
source /gpfs/home6/twang/myenv/bin/activate

echo "Verifying Python and CUDA..."
which python
python --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Devices:', torch.cuda.device_count())"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting training script..."
python -u main.py \
    --llm_type "meta-llama/Llama-3.2-3B" \
    --dataset_path "/home/twang/combined_audio" \
    --data_csv "/home/twang/combined.csv" \
    --batch_size 32 \
    --epochs 50 \
    --setting "lora" \
    --out_dir "/home/twang/lora_checkpoints " \
    --lr 2e-5 \
    --warmup_steps 400
    --eval \
    --verbose       
