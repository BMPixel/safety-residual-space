import argparse
import gc
import os

import diskcache as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from numpy.linalg import norm
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.extract import calculate_aligned_diffs, calculate_components, get_activations
from src.utils.ssft_dataset import make_dataset

# Add argument parsing at the start of executable code
parser = argparse.ArgumentParser(description="Model Analysis")
parser.add_argument(
    "--finetuned_model", type=str, default="ckpts/ssft/t160_n1", help="Path to finetuned model"
)
parser.add_argument(
    "--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path to base model"
)
parser.add_argument("--device", type=str, default="cuda", help="CUDA device to use")
parser.add_argument(
    "--save_name", type=str, default="fig_cls_acc.pdf", help="Filename for saving the plot"
)
args = parser.parse_args()

# Update these constants to use command line arguments
BASE_PATH = args.base_model
FINETUNED_PATH = args.finetuned_model
DEVICE = args.device
RANDOM_SEED = 42

# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def cleanup_memory():
    """Clean up memory by calling garbage collector and clearing CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


# Generate output filenames based on dataset
output_base = f"../../artifacts/residual/"
os.makedirs(output_base, exist_ok=True)
# read in the dataset
data = make_dataset(50, 0, 700, 256)["train"]
samples = data["input"]
print("Total samples: ", len(samples))
print(samples[2])

# Get indices where moderation_truth is non-zero
harmful_ids = [i for i in range(len(data["moderation_truth"])) if data["moderation_truth"][i] != 0]
safe_ids = [i for i in range(len(data["moderation_truth"])) if data["moderation_truth"][i] == 0]
pair_ids = [i for i in range(len(data["type"])) if data["type"][i] == "pair"]


tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)

base_model = AutoModelForCausalLM.from_pretrained(BASE_PATH, device_map=DEVICE)
ft_model = AutoModelForCausalLM.from_pretrained(FINETUNED_PATH, device_map=DEVICE)
base_activations, base_logits = get_activations(base_model, tokenizer, samples)
ft_activations, ft_logits = get_activations(ft_model, tokenizer, samples)
del base_model, ft_model
cleanup_memory()

noise, transform_mat, diff_activations = calculate_aligned_diffs(
    base_activations, ft_activations, method="matrix"
)

# Check the logits
# base_logits_max = base_logits.argmax(axis=1)
unsafe_token_idx = 39257
safe_token_idx = 19193
base_output_label = base_logits.argmax(axis=1) == unsafe_token_idx
ft_output_label = ft_logits.argmax(axis=1) == unsafe_token_idx

# Train the refusal direction

# Train refusal direction for each layer separately
num_layers = base_activations.shape[1]
refusal_directions = []
train_accuracies = []
test_accuracies = []

# Create labels (1 for harmful, 0 for safe)
labels = np.array([1 if i in harmful_ids else 0 for i in range(len(base_activations))])

# Split data into train/test sets
from sklearn.model_selection import train_test_split

indices = np.arange(len(base_activations))
train_idx, test_idx = train_test_split(indices, test_size=0.5, random_state=42, stratify=labels)

# Train logistic regression for each layer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

for layer in range(num_layers):
    # Get activations for this layer
    layer_diffs = base_activations[:, layer, :]  # (n_samples, hidden_dim)

    # Split into train/test
    X_train = layer_diffs[train_idx]
    X_test = layer_diffs[test_idx]
    y_train = base_output_label[train_idx]
    y_test = base_output_label[test_idx]

    # Train model
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)

    # Get predictions and accuracy for train and test
    train_preds = lr_model.predict(X_train)
    test_preds = lr_model.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # Store refusal direction for this layer
    refusal_directions.append(lr_model.coef_[0])  # (hidden_dim,)

refusal_directions = np.array(refusal_directions)  # (n_layers, hidden_dim)
print(f"Refusal directions shape: {refusal_directions.shape}")
print("\nAccuracies by layer:")
print("Layer\tTrain\tTest")
print("-" * 30)
for layer, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
    print(f"Layer {layer}:\t{train_acc:.4f}\t{test_acc:.4f}")
print(f"\nAverage train accuracy: {np.mean(train_accuracies):.4f}")
print(f"Average test accuracy: {np.mean(test_accuracies):.4f}")

# Create mapping of type to list of indices
type2ids = {}
for i, example in enumerate(data):
    type_key = example["type"]
    if type_key not in type2ids:
        type2ids[type_key] = []
    type2ids[type_key].append(i)

# Train a single refusal direction that can be used for all layers
from collections import defaultdict

# First create train/test split for the samples
sample_indices = np.arange(len(labels))
print(len(sample_indices))
train_mask, test_mask = train_test_split(
    sample_indices,
    test_size=0.5,
    random_state=42,
    # stratify=labels
)

# Split activations and labels into train/test sets first
base_activations_train = base_activations[train_mask]  # (n_train_samples, n_layers, hidden_dim)
base_activations_test = base_activations[test_mask]  # (n_test_samples, n_layers, hidden_dim)
labels_train = base_output_label[train_mask]
labels_test = base_output_label[test_mask]

# Flatten the train activations across layers
flattened_train_diffs = base_activations_train.reshape(
    -1, base_activations_train.shape[-1]
)  # (n_train_samples * n_layers, hidden_dim)
flattened_train_labels = np.repeat(labels_train, num_layers)  # Repeat labels for each layer

# Train a single refusal direction on all layers together
lr_model_all = LogisticRegression(random_state=42, fit_intercept=False)
lr_model_all.fit(flattened_train_diffs, flattened_train_labels)

# Get the single refusal direction
single_refusal_direction = lr_model_all.coef_[0]  # (hidden_dim,)

# Test accuracy when applying this direction to each layer separately
train_accuracies = []
single_test_accuracies = []
type_train_accs = defaultdict(list)
type_test_accs = defaultdict(list)

for layer in range(num_layers):
    # Get activations for this layer
    layer_diffs_train = base_activations_train[:, layer, :]
    layer_diffs_test = base_activations_test[:, layer, :]

    # Get predictions
    train_preds = lr_model_all.predict(layer_diffs_train)
    test_preds = lr_model_all.predict(layer_diffs_test)

    # Calculate overall accuracies
    train_acc = accuracy_score(labels_train, train_preds)
    test_acc = accuracy_score(labels_test, test_preds)

    train_accuracies.append(train_acc)
    single_test_accuracies.append(test_acc)

    # Calculate accuracies by type
    for type_key in type2ids:
        # Get indices for this type in train set
        type_train_indices = [
            i for i in range(len(train_mask)) if train_mask[i] in type2ids[type_key]
        ]
        if len(type_train_indices) > 0:
            type_train_acc = accuracy_score(
                labels_train[type_train_indices], train_preds[type_train_indices]
            )
            type_train_accs[type_key].append(type_train_acc)

        # Get indices for this type in test set
        type_test_indices = [
            i for i in range(len(test_mask)) if test_mask[i] in type2ids[type_key]
        ]
        if len(type_test_indices) > 0:
            type_test_acc = accuracy_score(
                labels_test[type_test_indices], test_preds[type_test_indices]
            )
            type_test_accs[type_key].append(type_test_acc)

print("\nSingle refusal direction results:")
print(f"Direction shape: {single_refusal_direction.shape}")
print("\nAccuracy by layer when using single direction:")
print("Layer\tTrain\tTest")
print("-" * 30)
for layer, (train_acc, test_acc) in enumerate(zip(train_accuracies, single_test_accuracies)):
    print(f"Layer {layer}:\t{train_acc:.4f}\t{test_acc:.4f}")
print(f"\nAverage train accuracy: {np.mean(train_accuracies):.4f}")
print(f"Average test accuracy: {np.mean(single_test_accuracies):.4f}")

print("\nAccuracies by type:")
for type_key in type_train_accs:
    print(f"\n{type_key}:")
    print(f"Average train accuracy: {np.mean(type_train_accs[type_key]):.4f}")
    print(f"Average test accuracy: {np.mean(type_test_accs[type_key]):.4f}")


diff_components_by_layer = []
diff_eigenvalues_by_layer = []
for layer in tqdm(range(num_layers), desc="Calculating components"):
    diff_comps, diff_eigs = calculate_components(diff_activations[:, layer, :])
    diff_components_by_layer.append(diff_comps)
    diff_eigenvalues_by_layer.append(diff_eigs)


# Calculate classification accuracy and log likelihood between components and labels for each layer
def calculate_cls_acc_and_loglikelihood(
    base_activations, diff_components_by_layer, base_output_label
):
    first_comp_accuracies = []  # accuracies for first component
    max_accuracies = []  # max accuracies across all components
    best_comp_indices = []  # indices of best performing components
    all_comp_accuracies = []
    all_comp_loglikelihoods = []  # new: store log likelihoods
    for layer in tqdm(range(num_layers), desc="Calculating accuracies and log likelihoods"):
        layer_accuracies = []
        layer_loglikelihoods = []  # new: for this layer

        # Calculate metrics for each of the top 100 components
        for comp_idx in range(100):
            comp = diff_components_by_layer[layer][comp_idx]

            # Project activations onto component
            layer_activations = base_activations[:, layer, :]
            projections = np.dot(layer_activations, -comp)

            # Use threshold based on label ratio
            ratio_labels = base_output_label.sum() / len(base_output_label)
            threshold = np.percentile(projections, 100 - ratio_labels * 100)
            predictions = (projections > threshold).astype(int)

            # Calculate accuracy
            accuracy = np.mean(predictions == base_output_label)
            layer_accuracies.append(accuracy)

            # Calculate log likelihood
            # Normalize projections for numerical stability
            proj_normalized = projections - projections.mean()
            log_likelihood = np.mean(
                proj_normalized * (2 * base_output_label - 1)  # Convert 0/1 to -1/+1
            )
            if comp_idx < 5:
                print("Layer: ", layer, "Comp: ", comp_idx, "Log Likelihood: ", log_likelihood)
            layer_loglikelihoods.append(log_likelihood)

        # Store metrics for this layer
        first_comp_accuracies.append(layer_accuracies[0])
        max_acc = max(layer_accuracies)
        max_accuracies.append(max_acc)
        best_idx = layer_accuracies.index(max_acc)
        best_comp_indices.append(best_idx)
        all_comp_accuracies.append(layer_accuracies)
        all_comp_loglikelihoods.append(layer_loglikelihoods)  # new: store log likelihoods

    return (
        first_comp_accuracies,
        max_accuracies,
        best_comp_indices,
        all_comp_accuracies,
        all_comp_loglikelihoods,
    )


(
    first_comp_accuracies,
    max_accuracies,
    best_comp_indices,
    all_comp_accuracies,
    all_comp_loglikelihoods,
) = calculate_cls_acc_and_loglikelihood(
    base_activations, diff_components_by_layer, base_output_label
)
# Print best component indices by layer
print("\nBest performing component indices by layer:")
for layer, idx in enumerate(best_comp_indices):
    print(f"Layer {layer}: component {idx}")

# Save results
import json

json.dump(all_comp_accuracies, open("../../artifacts/all_comp_accuracies.json", "w"))
json.dump(
    all_comp_loglikelihoods, open("../../artifacts/all_comp_loglikelihoods.json", "w")
)  # new: save log likelihoods

# Update the plotting section to only use SSFT results
plt.figure(figsize=(5, 3.0), dpi=300)
colors = plt.cm.gnuplot2(np.linspace(0.3, 0.8, 4))

plt.plot(
    range(num_layers),
    max_accuracies,
    "--",
    color=colors[0],
    label="Best-of-N SSFT",
    linewidth=2,
    alpha=0.4,
)

plt.plot(
    range(num_layers),
    single_test_accuracies,
    "-",
    color=colors[3],
    label="Probe Vector",
    linewidth=2,
)

plt.plot(
    range(num_layers), first_comp_accuracies, "-", color=colors[0], label="Dom. SSFT", linewidth=2
)

plt.xlabel("Layer")
plt.xlim(2, 30)
plt.ylabel("Accuracy")
plt.legend(
    frameon=True,
    facecolor="white",
    edgecolor="none",
    framealpha=1.0,
    bbox_to_anchor=(0.5, -0.5),  # Position legend below plot
    loc="center",  # Center horizontally
    ncol=3,  # Split into 2 rows (5 items ÷ 3 columns = 2 rows)
    bbox_transform=plt.gca().transAxes,
)
plt.grid(axis="y")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"../../artifacts/{args.save_name}", bbox_inches="tight")
