import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("module://imgcat")
import json

from matplotlib.colors import ListedColormap

WEIGHTS = torch.load("artifacts/relevance.pt")
NODES = json.load(open("artifacts/all_comp_loglikelihoods.json", "r"))

# Create figure with 3 subplots stacked vertically
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(5, 4), dpi=300)

# Add a larger ylabel for the entire figure
fig.text(0.02, 0.5, "Component", va="center", rotation="vertical")

K = 5

# Prepare data for heatmaps
comp1_data = np.zeros((K, 30))
comp2_data = np.zeros((K, 30))
comp3_data = np.zeros((K, 30))
nodes_data = np.zeros((K, 30))


def normalize(data, no_negative=True):
    if no_negative:
        data = np.clip(data, 0, None)
    return data / np.sum(data)


# Fill data arrays
for layer in range(30):
    comp1_data[:, layer] = normalize(WEIGHTS[layer + 2][:K, 0].numpy())
    comp2_data[:, layer] = normalize(WEIGHTS[layer + 2][:K, 1].numpy())
    comp3_data[:, layer] = normalize(WEIGHTS[layer + 2][:K, 2].numpy())
    nodes_data[:, layer] = NODES[layer + 2][:K]

# Get the base colormap
base_cmap = plt.get_cmap("gnuplot2")
# Create a new colormap using only values between 0.2 and 0.8
n_bins = 256  # number of color gradients
original_values = np.linspace(0, 1, n_bins)
subset_values = np.linspace(0.15, 0.9, n_bins)
cmap = ListedColormap(base_cmap(subset_values))

# Plot heatmaps
im1 = ax1.imshow(comp1_data, aspect="auto", cmap=cmap)
im2 = ax2.imshow(comp2_data, aspect="auto", cmap=cmap)
im3 = ax3.imshow(comp3_data, aspect="auto", cmap=cmap)
im4 = ax4.imshow(nodes_data, aspect="auto", cmap=cmap)

# Set y-ticks for all subplots
ax1.set_yticks(range(K))
ax1.set_yticklabels(range(1, K + 1))
for ax in [ax2, ax3, ax4]:
    ax.set_yticks([0, 4])
    ax.set_yticklabels([1, 5])

# Add colorbars with increased width
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.1, pad=0.04, ticks=[0, 1], label="Rel.\nComp. 1")
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.1, pad=0.04, ticks=[0, 1], label="Rel.\nComp. 2")
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.1, pad=0.04, ticks=[0, 1], label="Rel.\nComp. 3")
cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.1, pad=0.04, ticks=[0, 9], label="Log-\nlikelihood")

# Remove x-axis ticks for the first three subplots
ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])

# Set titles and labels
# ax4.set_title('Node Log-likelihoods')

# Set only x label (y label is now for whole figure)
ax4.set_xlabel("Layer")

# Adjust spacing between subplots
# plt.subplots_adjust(hspace=10)

# plt.tight_layout()
plt.savefig("artifacts/relevance_heatmap.pdf")
plt.show()
