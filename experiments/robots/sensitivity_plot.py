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

# Load JSON files
dim_file = os.path.join(sensitivity_dir, "dim5.json")
rollouts_file = os.path.join(sensitivity_dir, "rollouts5.json")

with open(dim_file, "r") as f:
    dim_data = json.load(f)

with open(rollouts_file, "r") as f:
    rollouts_data = json.load(f)

# Convert JSON data to DataFrames
dim_df = pd.DataFrame(dim_data)
rollouts_df = pd.DataFrame(rollouts_data)

# Get the current timestamp
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ------------------ Plot 1a: Heat Map for Collisions ------------------
collisions_pivot = dim_df.pivot(index="dim_nl", columns="dim_internal", values="percentage_test_collisions")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(collisions_pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax, cbar_kws={'label': 'Percentage of Collisions (%)'})
ax.set_title(f"Percentage of Collisions (Rollouts: {dim_df['num_rollouts'].iloc[0]})")
ax.set_xlabel("dim_internal")
ax.set_ylabel("dim_nl")
ax.invert_yaxis()  # Invert the y-axis

# Save the plot as SVG
collisions_svg_path = os.path.join(figures_dir, f"dim_collisions_heatmap_{current_datetime}.svg")
plt.tight_layout()
plt.savefig(collisions_svg_path, format="svg")
print(f"Collisions heat map saved to {collisions_svg_path}")

# ------------------ Plot 1b: Heat Map for Obstacle Loss ------------------
loss_pivot = dim_df.pivot(index="dim_nl", columns="dim_internal", values="test_obst_loss")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(loss_pivot, annot=True, fmt=".1f", cmap="Blues", ax=ax, cbar_kws={'label': 'Obstacle Loss'})
ax.set_title(f"Obstacle Loss (Rollouts: {dim_df['num_rollouts'].iloc[0]})")
ax.set_xlabel("dim_internal")
ax.set_ylabel("dim_nl")
ax.invert_yaxis()  # Invert the y-axis

# Save the plot as SVG
loss_svg_path = os.path.join(figures_dir, f"dim_loss_heatmap_{current_datetime}.svg")
plt.tight_layout()
plt.savefig(loss_svg_path, format="svg")
print(f"Obstacle loss heat map saved to {loss_svg_path}")
# ------------------ Plot 2: Number of Rollouts vs Collisions and Loss ------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot number of rollouts vs collisions
ax1.set_xlabel("Number of Rollouts")
ax1.set_ylabel("Test Collisions [%] (on 500 rollouts)", color="tab:red")
ax1.plot(rollouts_df["num_rollouts"], rollouts_df["percentage_test_collisions"], label="Percentage of test collisions", color="tab:red", marker="o")
ax1.tick_params(axis="y", labelcolor="tab:red")

# Add a second y-axis for the loss
ax2 = ax1.twinx()
ax2.set_ylabel("Obstacle Loss", color="tab:blue")
ax2.plot(rollouts_df["num_rollouts"], rollouts_df["test_obst_loss"], label="Obstacle Loss", color="tab:blue", marker="o")
ax2.tick_params(axis="y", labelcolor="tab:blue")

# Add a third y-axis for the total loss
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis to avoid overlap
ax3.set_ylabel("Total Loss", color="tab:green")
ax3.plot(rollouts_df["num_rollouts"], rollouts_df["test_loss"], label="Total Loss", color="tab:green", marker="o")
ax3.tick_params(axis="y", labelcolor="tab:green")

# Add title and grid
plt.title("Sensitivity Study: Number of Rollouts vs Collisions and Loss")
fig.tight_layout()

# Save the plot
rollouts_pdf_path = os.path.join(figures_dir, f"rollouts_vs_collisions_and_loss_{current_datetime}.svg")
plt.savefig(rollouts_pdf_path, format="svg")
print(f"Rollouts vs Collisions and Loss plot saved to {rollouts_pdf_path}")


plt.show()