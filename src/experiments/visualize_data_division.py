import matplotlib.pyplot as plt
import numpy as np

# Data
labels = [
    "Ours",
    "PAIR",
    "ReNellm",
    "GPTFuzz",
    "GCG",
    "Simple",
    "CodeChaeleom",
    "Flip",
    "or-bench",
]
sizes = [60, 60, 60, 60, 60, 60, 60, 60, 480]

# Color settings
colors = plt.cm.Blues(np.linspace(0.4, 0.8, 8))  # Blue gradient
colors = np.vstack((colors, [1, 0.7, 0.3, 1]))  # Add orange

# Set font size
plt.rcParams.update({"font.size": 15})

# Create pie chart
plt.figure(figsize=(10, 8))
plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.8,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 18},
    labeldistance=1.05,
)  # Adjust label distance closer to center

# Add title
plt.title(
    "Test Set Partition Distribution\nTotal: 960 samples", pad=20, fontsize=20
)  # Set title font size

# Ensure pie chart is circular
plt.axis("equal")

# Add legend
plt.legend(
    labels,
    title="Sources",
    loc="center left",
    bbox_to_anchor=(0.8, 0.3, 0.5, 1),
    fontsize=14,  # Set legend font size
    title_fontsize=17,
)  # Set legend title font size

# Save figure
plt.savefig("helper/perplexity/result.pdf", bbox_inches="tight", dpi=300)
