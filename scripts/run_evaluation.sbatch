#!/bin/bash
#SBATCH --job-name=rlm_eval
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=rlm_eval_%j.out
#SBATCH --error=rlm_eval_%j.err

echo "RLM Evaluation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export PYTORCH_ALLOC_CONF=expandable_segments:True

source venv/bin/activate

nvidia-smi --query-gpu=index,name,memory.total --format=csv

python experiments/evaluate.py \
    --num_examples 10 \
    --split validation \
    --controller_gpu cuda:0 \
    --worker_gpu cuda:1

echo "Job completed with exit code: $?"
