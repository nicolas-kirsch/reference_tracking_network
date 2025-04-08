import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Define paths
sensitivity_dir = "sensitivity_study"
figures_dir = os.path.join(sensitivity_dir, "Figures")
os.makedirs(figures_dir, exist_ok=True)

# Load JSON file
epochs_rollouts_file = os.path.join(sensitivity_dir, "epochs_rollouts7.json")

with open(epochs_rollouts_file, "r") as f:
    epochs_rollouts_data = json.load(f)

# Convert JSON data to DataFrame
epochs_rollouts_df = pd.DataFrame(epochs_rollouts_data)

# Get the current timestamp
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ------------------ Plot 1a: Heat Map for Collisions ------------------
collisions_pivot = epochs_rollouts_df.pivot(index="num_rollouts", columns="epochs", values="percentage_test_collisions")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(collisions_pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax, cbar_kws={'label': 'Percentage of Collisions (%)'})
ax.set_title("Percentage of Collisions")
ax.set_xlabel("Number of Epochs")
ax.set_ylabel("Number of Rollouts")
ax.invert_yaxis()  # Invert the y-axis

# Save the plot as SVG
collisions_svg_path = os.path.join(figures_dir, f"epochs_collisions_heatmap_{current_datetime}.svg")
plt.tight_layout()
plt.savefig(collisions_svg_path, format="svg")
print(f"Collisions heat map saved to {collisions_svg_path}")

# ------------------ Plot 1b: Heat Map for Obstacle Loss ------------------
loss_pivot = epochs_rollouts_df.pivot(index="num_rollouts", columns="epochs", values="test_obst_loss")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(loss_pivot, annot=True, fmt=".1f", cmap="Blues", ax=ax, cbar_kws={'label': 'Obstacle Loss'})
ax.set_title("Obstacle Loss")
ax.set_xlabel("Number of Epochs")
ax.set_ylabel("Number of Rollouts")
ax.invert_yaxis()  # Invert the y-axis

# Save the plot as SVG
loss_svg_path = os.path.join(figures_dir, f"epochs_loss_heatmap_{current_datetime}.svg")
plt.tight_layout()
plt.savefig(loss_svg_path, format="svg")
print(f"Obstacle loss heat map saved to {loss_svg_path}")

plt.show()