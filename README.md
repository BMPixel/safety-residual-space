# Safety Residual Space

This anonymous repository contains source code, datasets, and experimental results to reproduce findings from the research paper *The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Analysis of Orthogonal Safety Directions*

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Setup](#data-setup)
  - [Performing SSFT and DPO](#performing-ssft-and-dpo)
  - [Analyzing Safety Residual Space](#analyzing-safety-residual-space)
- [Dataset and Artifacts](#dataset-and-artifacts)

## Installation

### 1. Environment Setup with Poetry

First, ensure you have Poetry installed on your system. Poetry is a dependency management tool that will help set up the virtual environment with all required packages:

```bash
# Install all dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### 2. Openrouter API Configuration

> Note: This step is only necessary if you want to generate jailbreak samples from scratch. Pre-generated samples are already provided in the `data` directory.

We utilize the Openrouter API to conveniently access multiple language models (including GPT-4, Llama 3 70B/405B, and Hermes 3 70B/405B). Set up your API key as follows:

```bash
export OPENROUTER_API_KEY=sk-or-v1-xxxxx
```

### 3. Multi-GPU Training Environment Setup

Our implementation uses the `accelerate` library to support full-weight fine-tuning of DPO and SSFT on Llama 3.1 8B. Please note:
- Training Llama 3.1 8B requires approximately 600GB of VRAM
- For systems with limited resources, you can alternatively train on Llama 3.1 3B using a single GPU

Configure your accelerator settings with:

```bash
accelerate config
```

## Usage

### Data Setup

Below we detail the process of generating the training and test datasets used in safety fine-tuning. For convenience, all pre-generated samples are provided in the `data` directory. The training and experiment scripts will automatically look for these samples in this location. Since the generation process involves API calls to third-party model providers, rerunning the scripts may produce slightly different results each time.

#### 1. Jailbreak Baselines:
We provide implementations of different jailbreak techniques in the `src/jailbreaks` directory. To generate jailbreak samples using these techniques, run the following script:

```bash
# scripts/1.make_jailbreaks.sh
for jailbreak in gcg gptfuzz pair flipattack codechameleon renellm simple strong_reject; do
    python src/jailbreaks/$jailbreak.py
done
```

#### 2. Trigger Removal Attack:
The trigger removal attack identifies and removes safety-triggering tokens from prompts. The implementation is in `src/jailbreaks/trigger_removal.py`. This attack requires running a safety-aligned LLM as the victim model to identify trigger tokens. Run the attack using:

```bash
# scripts/2.trigger_removal_attack.sh
python src/jailbreaks/trigger_removal.py \
    --victim "meta-llama/Llama-3.1-8B-Instruct" \
    --resample "nousresearch/hermes-3-llama-3.1-70b"
```

#### 3. Preference Optimization Dataset:
For DPO training, we generate pairs of preferred and rejected outputs for each input query. The implementation is in `src/utils/dpo_dataset.py`. Generate the dataset with:

```bash
# scripts/3.prepare_dpo_dataset.sh
python src/utils/dpo_dataset.py \
    --datasets trigger_removal pair simple gptfuzz \
    flip gcg renellm codechameleon
```

#### 4. Evaluate Strong Reject Score:
We provide a script to evaluate the Strong Reject Score of jailbreak attacks on any given model. When `-d` points to a directory of checkpoints, this script will evaluate all variants of K-SHOT safety training:

```bash
# scripts/4.evaluate_strong_reject.sh
python src/utils/evaluate.py -m "meta-llama/Llama-3.1-8B-Instruct" -d "ckpts/dpo/Llama-3.1-8B-Instruct"
```

### Performing SSFT and DPO

To derive the safety residual space, we perform SSFT and DPO training on Llama 3.1 8B. The training code is in `src/utils/safety_sft.py` and `src/utils/safety_dpo.py`. Trained checkpoints are saved to `ckpts/ssft` and `ckpts/dpo` directories.

```bash
# scripts/5.run_safety_fine_tuning.sh
for size in 10 20 40 80 160; do
    # SSFT
    accelerate launch src/utils/safety_ssft.py --train_size_per_type $size
    # DPO
    accelerate launch src/utils/safety_dpo.py --train_size_per_type $size
done
```

### Analyzing Safety Residual Space

After SSFT and DPO training, we perform analysis on the safety residual space. The implementation is in `src/experiments`:

```bash
# scripts/6.rank_safety_residual_space.sh

# Visualize rank on SSFT model
python src/experiments/rank_safety_residual_space.py \
    --base_model "meta-llama/Llama-3.1-8B-Instruct" \
    --finetuned_model "ckpts/ssft/t160_n1"

# Visualize rank on DPO model
python src/experiments/rank_safety_residual_space.py \
    --base_model "meta-llama/Llama-3.1-8B-Instruct" \
    --finetuned_model "ckpts/dpo/t160_n1"

# Compare components with probe vectors
python src/experiments/output_prediction_with_components.py \
    --base_model "meta-llama/Llama-Guard-3-8B" \
    --finetuned_model "ckpts/ssft/t160_n1"

# Apply PLRP to interpret components with token-wise relevance score
python src/experiments/plrp_relevance_tokens.py \
    --base_model "meta-llama/Llama-Guard-3-8B" \
    --finetuned_model "ckpts/ssft/t160_n1" \
    --layer_idx 14

# Apply PLRP to interpret components with early layer components
python src/experiments/plrp_relevance_components.py \
    --base_model "meta-llama/Llama-Guard-3-8B" \
    --finetuned_model "ckpts/ssft/t160_n1" \
    --target_layer 14 \
    --source_layer 13

# Intervention on non-dominant component to affect model refuse behavior
# Recommend trying different components and alpha values as the intervention is sensitive to them
python src/experiments/intervention.py \
    --base_model "meta-llama/Llama-Guard-3-8B" \
    --finetuned_model "ckpts/ssft/t160_n1" \
    --layer_idx 14 \
    --comp_idx 5 \
    --alpha 1.15

# Evaluate perplexity on alpaca dataset with given model and direction intervention
python src/experiments/perplexity_evaluation.py \
    --run_type intervention \
    --layer_idx 15 \
    --component_idx 6 \
    --alpha 1.15
```

## Dataset and Artifacts

We provide the dataset used in the paper in the `data` directory. In addition to the generated samples, we provide intermediate results as artifacts in the `artifacts` directory. Below is a brief description of the contents of these artifacts:

**Measuring Linear Properties of Safety Residual Space**
- **[activation_norms.png](artifacts/activation_norms.png)**: Norms of activations across different layers for base, ssft model and their residual differences.
- **[eigenvalues.pdf](artifacts/eigenvalues.pdf)**: Eigenvalues of the SVD of the safety residual space on different layers.
- **[fig_rank.pdf](artifacts/fig_rank.pdf)**: Effective rank of the safety residual space across different layers.

**Comparing Components with Probe Vectors**
- **[all_comp_accuracies.json](artifacts/all_comp_accuracies.json)**: Classification accuracies of model refuse behavior using all components as linear classifier on activation space.
- **[all_comp_loglikelihoods.json](artifacts/all_comp_loglikelihoods.json)**: Log-likelihood scores of model refuse behavior using all components as linear classifier on activation space.
- **[first_comp_accuracies.json](artifacts/first_comp_accuracies.json)**: Accuracy scores for dominant components across layers.
- **[best_comp_indices.json](artifacts/best_comp_indices.json)**: Indices of best performing components for each layer on classification accuracy.
- **[fig_cls_acc.pdf](artifacts/fig_cls_acc.pdf)**: Figure showing classification accuracy results.
- **[probe_similarity_to_dominant.png](artifacts/probe_similarity_to_dominant.png)**: Cosine similarity between probes and dominant components.

**PLRP Analysis**
- **[average_relevance_by_layer.json](artifacts/average_relevance_by_layer.json)**: Relevance scores between adjacent layers in the ssft model.
- **[relevance.pt](artifacts/relevance.pt)**: Relevance scores between adjacent layers in the ssft model (PyTorch format).
- **[relevance_heatmap.pdf](artifacts/relevance_heatmap.pdf)**: Heatmap visualization of adjacent layer relevance scores.
- **[network_0-3.pdf](artifacts/network_0.pdf)**: A more intuitive visualization of the adjacent layer relevance scores. Drawn from `average_relevance_by_layer.json`.

**Intervention Results**
- **[fig_supression.pdf](artifacts/fig_supression.pdf)**: Results of suppression L14C6 on ssft model.
- **[component_projections.pdf](artifacts/component_projections.pdf)**: Visualization of non-dominant components suppression on ssft model, compared with other methods.

**Safety Fine-tuning Evaluation**
- **[base-dpo_metrics_by_exposure.csv](artifacts/base-dpo_metrics_by_exposure.csv)**: Strong Reject Score of each jailbreak attack on DPO model with K-SHOT safety training.
- **[base-dpo_metrics_by_exposure.pdf](artifacts/base-dpo_metrics_by_exposure.pdf)**: Visualization of Strong Reject Score of each jailbreak attack on DPO model with K-SHOT safety training.
- **[base-ssft_metrics_by_exposure_cls.csv](artifacts/base-ssft_metrics_by_exposure_cls.csv)**: Classification metrics of model refuse behavior using ssft model with K-SHOT safety training. Classification metrics are computed using the risk type identifier generated by ssft model.
- **[base-ssft_metrics_by_exposure_cls.png](artifacts/base-ssft_metrics_by_exposure_cls.png)**: Visualization of classification metrics of model refuse behavior using ssft model with K-SHOT safety training.
- **[base-ssft_metrics_by_exposure.csv](artifacts/base-ssft_metrics_by_exposure.csv)**: Refusal classification metrics of ssft model with K-SHOT safety training.
- **[passing_rate_by_exposure.png](artifacts/passing_rate_by_exposure.png)**: Plot showing passing rates at different exposure levels.
