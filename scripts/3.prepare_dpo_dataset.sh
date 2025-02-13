#!/bin/bash
poetry shell

# Create output directory if it doesn't exist
mkdir -p data/dpo

# Generate DPO dataset from all jailbreak samples
python src/utils/dpo_dataset.py \
    --datasets trigger_removal pair simple gptfuzz \
    flip gcg renellm codechameleon \
    --output_dir data/dpo

echo "DPO dataset preparation completed!" 