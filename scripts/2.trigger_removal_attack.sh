#!/bin/bash
poetry shell

# Create output directory if it doesn't exist
mkdir -p data/trigger_removal

# Run trigger removal attack
python src/jailbreaks/trigger_removal.py \
    --victim "meta-llama/Llama-3.1-8B-Instruct" \
    --resample "nousresearch/hermes-3-llama-3.1-70b"

echo "Trigger removal attack completed!" 