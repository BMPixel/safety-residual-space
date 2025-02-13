#!/bin/bash
poetry shell

# Create output directory for analysis results
mkdir -p results/analysis

# Analyze SSFT model
echo "Analyzing SSFT model..."
python src/experiments/rank_safety_residual_space.py \
    --base_model "meta-llama/Llama-3.1-8B-Instruct" \
    --finetuned_model "ckpts/ssft/t160_n1" \
    --output_dir "results/analysis/ssft"

# Analyze DPO model
echo "Analyzing DPO model..."
python src/experiments/rank_safety_residual_space.py \
    --base_model "meta-llama/Llama-3.1-8B-Instruct" \
    --finetuned_model "ckpts/dpo/t160_n1" \
    --output_dir "results/analysis/dpo"

# Compare components with probe vectors
echo "Comparing components with probe vectors..."
python src/experiments/output_prediction_with_components.py \
    --base_model "meta-llama/Llama-Guard-3-8B" \
    --finetuned_model "ckpts/ssft/t160_n1" \
    --output_dir "results/analysis/components"

# Apply PLRP for token-wise relevance
echo "Applying PLRP for token-wise relevance..."
python src/experiments/plrp_relevance_tokens.py \
    --base_model "meta-llama/Llama-Guard-3-8B" \
    --finetuned_model "ckpts/ssft/t160_n1" \
    --layer_idx 14 \
    --output_dir "results/analysis/plrp_tokens"

# Apply PLRP for early layer components
echo "Applying PLRP for early layer components..."
python src/experiments/plrp_relevance_components.py \
    --base_model "meta-llama/Llama-Guard-3-8B" \
    --finetuned_model "ckpts/ssft/t160_n1" \
    --target_layer 14 \
    --source_layer 13 \
    --output_dir "results/analysis/plrp_components"

# Run intervention experiments
echo "Running intervention experiments..."
python src/experiments/intervention.py \
    --base_model "meta-llama/Llama-Guard-3-8B" \
    --finetuned_model "ckpts/ssft/t160_n1" \
    --layer_idx 14 \
    --comp_idx 5 \
    --alpha 1.15 \
    --output_dir "results/analysis/intervention"

# Evaluate perplexity
echo "Evaluating perplexity..."
python src/experiments/perplexity_evaluation.py \
    --run_type intervention \
    --layer_idx 15 \
    --component_idx 6 \
    --alpha 1.15 \
    --output_dir "results/analysis/perplexity"

echo "Safety residual space analysis completed!" 