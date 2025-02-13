#!/bin/bash
poetry shell
# Create output directory for evaluation results
mkdir -p results/evaluation

# Evaluate base model
python src/utils/evaluate.py \
    -m "meta-llama/Llama-3.1-8B-Instruct" \
    --output_dir results/evaluation/base

# Evaluate all DPO checkpoints
python src/utils/evaluate.py \
    -m "meta-llama/Llama-3.1-8B-Instruct" \
    -d "ckpts/dpo/Llama-3.1-8B-Instruct" \
    --output_dir results/evaluation/dpo

# Evaluate all SSFT checkpoints
python src/utils/evaluate.py \
    -m "meta-llama/Llama-3.1-8B-Instruct" \
    -d "ckpts/ssft/Llama-3.1-8B-Instruct" \
    --output_dir results/evaluation/ssft

echo "Strong reject evaluation completed!" 