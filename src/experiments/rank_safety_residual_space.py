import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("module://imgcat")
import argparse

from src.utils.extract import calculate_aligned_diffs, calculate_components, get_activations
from src.utils.ssft_dataset import make_dataset

DEVICE = "cuda:6"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
output_base = f"artifacts"
os.makedirs(output_base, exist_ok=True)
data = make_dataset(50, 0, 700, 256)["train"]
samples = data["input"]

# Get indices where moderation_truth is non-zero
harmful_ids = [i for i in range(len(data["moderation_truth"])) if data["moderation_truth"][i] != 0]
safe_ids = [i for i in range(len(data["moderation_truth"])) if data["moderation_truth"][i] == 0]
pair_ids = [i for i in range(len(data["type"])) if data["type"][i] == "pair"]

# Add argument parsing at the top
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--finetuned_model", type=str, default="ckpts/dpo/t160_n1")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--save_name", type=str, default="fig_rank.pdf")
args = parser.parse_args()

# Update these constants to use command line arguments
LLAMA_BASE_PATH = args.base_model
FINETUNED_PATH = args.finetuned_model
DEVICE = args.device


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

    diff_activations, transform_mat, transform_vec = calculate_aligned_diffs(
        base_acts=base_activations, ft_acts=ft_activations, method="identity"
    )

    num_layers = base_activations.shape[1]

    diff_components_by_layer = []
    diff_eigenvalues_by_layer = []

    for layer in tqdm(range(num_layers), desc="Calculating components"):
        diff_comps, diff_eigs = calculate_components(diff_activations[:, layer, :])
        diff_components_by_layer.append(diff_comps)
        diff_eigenvalues_by_layer.append(diff_eigs)

    return diff_components_by_layer, diff_eigenvalues_by_layer, base_activations, ft_activations


# Remove Guard components calculation and only keep Llama
llama_components, llama_eigenvalues, llama_base_acts, llama_base_labels = (
    calculate_diff_components(LLAMA_BASE_PATH, FINETUNED_PATH, template="instruct")
)


# Calculate effective rank of the difference activations
def calculate_effective_rank(eigenvalues, threshold=0.99):
    """Calculate effective rank by summing eigenvalues until they reach threshold of the total."""
    # Filter eigenvalues greater than 0.01 first
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0
    return np.argmax(np.cumsum(eigenvalues) / np.sum(eigenvalues) >= threshold / 100) + 1


# Calculate effective ranks for different thresholds
thresholds = [40, 60, 70, 80, 90]
diff_ranks_by_threshold_llama = {}
# Create a purple colormap for different thresholds
colors = plt.cm.gnuplot2(np.linspace(0.5, 0.8, len(thresholds)))
num_layers = len(llama_components)

# Calculate ranks for both models
for threshold in thresholds:
    llama_ranks = []
    for layer in tqdm(range(num_layers), desc=f"Calculating ranks (threshold={threshold}%)"):
        llama_rank = calculate_effective_rank(llama_eigenvalues[layer], threshold)
        llama_ranks.append(llama_rank)
    diff_ranks_by_threshold_llama[threshold] = llama_ranks

# Update plotting section to use single plot
plt.figure(figsize=(3.5, 2.5), dpi=300)
ax = plt.gca()

# Plot Llama model with different thresholds
for threshold, color in list(zip(thresholds, colors))[::-1]:
    ax.plot(
        range(num_layers),
        diff_ranks_by_threshold_llama[threshold],
        color=color,
        linewidth=2,
        label=f"$\\tau = {threshold / 100:.1f}$",
    )
    print(f"Rank of {threshold}%: {diff_ranks_by_threshold_llama[threshold]}")

# Configure plot
ax.set_ylabel("Effective Rank")
ax.set_xlabel("Layer")
ax.grid(axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
ax.set_ylim(0.9, 100)
ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])

# Adjust legend placement
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

plt.tight_layout()
plt.savefig(f"artifacts/{args.save_name}", bbox_inches="tight")
plt.show()
