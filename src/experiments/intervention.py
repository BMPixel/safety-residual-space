import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.extract import (
    calculate_aligned_diffs,
    calculate_components,
    get_activations,
    modify_model_with_direction,
)
from src.utils.ssft_dataset import make_dataset

parser = argparse.ArgumentParser(description="Model Relevance Analysis")
parser.add_argument(
    "--base-model",
    type=str,
    default="meta-llama/Llama-3.1-8B-Instruct",
    help="Path to the base model for analysis.",
)
parser.add_argument(
    "--finetuned-model",
    type=str,
    default="ckpts/ssft/t160_n1",
    help="Path to the fine-tuned model for comparison.",
)
parser.add_argument(
    "--layer_idx", type=int, default=15, help="Index of the target layer for intervention."
)
parser.add_argument("--comp_idx", type=int, default=5, help="Index of the component to intervene.")
parser.add_argument("--alpha", type=float, default=-1.0, help="Alpha value for the intervention.")
parser.add_argument(
    "--use_proj", action="store_true", help="Whether to use projection for the intervention."
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Device to run the model on (e.g., cuda:0, cpu)."
)

args = parser.parse_args()

BASE_MODEL = args.base_model
FINETUNED_MODEL = args.finetuned_model
DEVICE = args.device
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
output_base = f"artifacts"
os.makedirs(output_base, exist_ok=True)
data = make_dataset(50, 0, 700, 256)["train"]
samples = data["input"]
layer_idx = args.layer_idx
comp_idx = args.comp_idx
alpha = args.alpha
use_proj = args.use_proj

# Get indices where moderation_truth is non-zero
harmful_ids = [i for i in range(len(data["moderation_truth"])) if data["moderation_truth"][i] != 0]
safe_ids = [i for i in range(len(data["moderation_truth"])) if data["moderation_truth"][i] == 0]
pair_ids = [i for i in range(len(data["type"])) if data["type"][i] == "pair"]


def calculate_diff_components(BASE_MODEL_PATH, FINETUNED_MODEL_PATH, template="instruct"):

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    base_activations, base_logits = get_activations(
        model=AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map=DEVICE),
        tokenizer=tokenizer,
        texts=samples,
    )

    ft_activations, ft_logits = get_activations(
        model=AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH, device_map=DEVICE),
        tokenizer=tokenizer,
        texts=samples,
    )

    if template == "guard":
        unsafe_token_idx = 39257
    elif template == "instruct":
        unsafe_token_idx = 40

    base_output_label = base_logits.argmax(axis=1) == unsafe_token_idx
    ft_output_label = ft_logits.argmax(axis=1) == unsafe_token_idx

    # Calculate differences using specified method
    noise, transform_mat, diff_activations = calculate_aligned_diffs(
        base_acts=base_activations, ft_acts=ft_activations, method="matrix"
    )

    num_layers = base_activations.shape[1]

    diff_components_by_layer = []
    diff_eigenvalues_by_layer = []

    for layer in tqdm(range(num_layers), desc="Calculating components"):
        diff_comps, diff_eigs = calculate_components(diff_activations[:, layer, :])
        diff_components_by_layer.append(diff_comps)
        diff_eigenvalues_by_layer.append(diff_eigs)

    return diff_components_by_layer, diff_eigenvalues_by_layer, base_activations, base_output_label


diff_components_by_layer, diff_eigenvalues_by_layer, guard_base_acts, guard_base_labels = (
    calculate_diff_components(BASE_MODEL, FINETUNED_MODEL, template="guard")
)

if "model" in locals():
    del model
model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL, device_map=DEVICE)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

test_data = make_dataset(50, 60, 700, 256)["test"]
test_pair_samples = [d["input"] for d in test_data if d["type"] == "pair"]
test_flip_samples = [d["input"] for d in test_data if d["type"] == "flip"]
test_simple_samples = [d["input"] for d in test_data if d["type"] == "simple"]
test_srj_samples = [d["input"] for d in test_data if d["type"] == "strong_reject"]
test_gptfuzz_samples = [d["input"] for d in test_data if d["type"] == "gptfuzz"]

sample_sets = {
    "pair": test_pair_samples,
    "flip": test_flip_samples,
    "simple": test_simple_samples,
    "gptfuzz": test_gptfuzz_samples[:4],
}

outputs = {}

try:
    hooks = modify_model_with_direction(
        model,
        direction=diff_components_by_layer[layer_idx][comp_idx].astype(np.float32),
        alpha=alpha,
        last_token_only=True,  # Optional: only modify last two layers
        use_proj=use_proj,
    )
    for sample_type, samples in sample_sets.items():
        print(f"Sample type {sample_type}")
        # Run your model
        acts, logs = get_activations(model, tokenizer, samples)
        indices = tokenizer.convert_ids_to_tokens(logs.argmax(axis=1))
        outputs[sample_type] = indices
finally:
    # Always remove hooks
    for hook in hooks:
        hook.remove()

# Print results
results = {}
print("-" * 50)
results["intervention"] = {}  # Single direction results
for sample_type, responses in outputs.items():
    results["intervention"][sample_type] = {}
    print(f"\nSample type: {sample_type}")
    # Count frequency of each unique response
    unique_responses = {}
    total_count = len(responses)
    for resp in responses:
        if resp not in unique_responses:
            unique_responses[resp] = 0
        unique_responses[resp] += 1

    # Print percentage for each unique response
    for resp, count in unique_responses.items():
        percentage = count / total_count
        results["intervention"][sample_type][resp] = percentage
        print(f"{resp}: {percentage*100:.1f}%")

# Set up the plot with improved style
plt.figure(figsize=(5.5, 2.5), dpi=300)
ax = plt.gca()

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Set width of bars and positions of the bars
bar_width = 0.2  # Increased bar width since we have fewer bars
models = ["Original", f"L{layer_idx}C{comp_idx}\nα={alpha}"]  # Updated labels
sample_types = ["pair", "flip", "simple"]

sample_type_to_name = {
    "pair": "PAIR",
    "flip": "FLIPAttack",
    "simple": "StrongReject",
}

# Calculate positions for each group of bars
x = np.arange(len(models))
# Define colors using matplotlib colormap
cmap = plt.cm.gnuplot2
colors = [cmap(i) for i in np.linspace(0.4, 0.8, len(sample_types))]

# Create bars for each sample type
for i, sample_type in enumerate(sample_types):
    safe_percentages = [
        0,  # Placeholder for original model (you might want to compute this)
        results["intervention"][sample_type].get("safe", 0) * 100,
    ]

    plt.bar(
        x + i * bar_width,
        safe_percentages,
        bar_width,
        color=colors[i],
        label=f"{sample_type_to_name[sample_type]}",
    )

# Customize the plot
plt.ylabel("Attack Pass Rate (%)", fontsize=10)
plt.xticks(x + bar_width, models, fontsize=9)
plt.legend(fontsize=8, frameon=False)

# Add gridlines only for y-axis with lighter style
plt.grid(True, axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(
    f"../../artifacts/intervention_L{layer_idx}C{comp_idx}_a{alpha}.pdf", bbox_inches="tight"
)
plt.show()
