import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Define paths
sensitivity_dir = "sensitivity_study"
figures_dir = os.path.join(sensitivity_dir, "Figures")
os.makedirs(figures_dir, exist_ok=True)

# Load JSON file
epochs_file = os.path.join(sensitivity_dir, "rt_epochs12.json")

with open(epochs_file, "r") as f:
    epochs_data = json.load(f)

# Convert JSON data to DataFrame
epochs_df = pd.DataFrame(epochs_data)

# Get the current timestamp
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Create a single figure with three subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)  # Wide figure with shared x-axis

# ------------------ Subplot 1: Total Loss ------------------
axes[0].set_ylabel("Total Loss", color="tab:green", fontsize=14)
axes[0].plot(epochs_df["rt_epochs"], epochs_df["test_loss"], label="Total Loss", color="tab:green", marker="o")
axes[0].tick_params(axis="y", labelcolor="tab:green", labelsize=12)
axes[0].tick_params(axis="x", labelsize=12)
# axes[0].set_title("Metrics vs rt-epochs", fontsize=16)

# ------------------ Subplot 2: Collisions and Obstacle Loss ------------------
axes[1].set_ylabel("Collisions [%]", color="tab:red", fontsize=14)
axes[1].plot(epochs_df["rt_epochs"], epochs_df["percentage_test_collisions"], label="Collisions [%]", color="tab:red", marker="o")
axes[1].tick_params(axis="y", labelcolor="tab:red", labelsize=12)

# Add a second y-axis for obstacle loss
ax2 = axes[1].twinx()
ax2.set_ylabel("Obstacle Loss", color="tab:blue", fontsize=14)
ax2.plot(epochs_df["rt_epochs"], epochs_df["test_obst_loss"], label="Obstacle Loss", color="tab:blue", marker="s")
ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=12)

# Add plain text annotation to the second plot
axes[1].text(
    0.5, 0.9, "% of collisions on 500 test rollouts",
    transform=axes[1].transAxes, fontsize=12, ha="center", va="center", color="tab:red"
)

# ------------------ Subplot 3: Metric ------------------
axes[2].set_ylabel("Metric", color="tab:purple", fontsize=14)
axes[2].plot(epochs_df["rt_epochs"], epochs_df["test_metric"], label="Metric", color="tab:purple", marker="o")
axes[2].tick_params(axis="y", labelcolor="tab:purple", labelsize=12)

# Add plain text annotation to the third plot (at the bottom)
axes[2].text(
    0.5, 0.1, "Metric = Birds flight distance / Travelled Distance",
    transform=axes[2].transAxes, fontsize=12, ha="center", va="center", color="tab:purple"
)

# Add a single x-axis label
fig.text(0.5, 0.04, "backward / forward epochs", ha="center", fontsize=14)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the shared x-axis label
combined_svg_path = os.path.join(figures_dir, f"combined_metrics_vs_rt_epochs_{current_datetime}.svg")
plt.savefig(combined_svg_path, format="svg")
print(f"Combined metrics plot saved to {combined_svg_path}")
plt.show()





# # ------------------ Plot 1: rt_epochs vs percentage_test_collisions and test_obst_loss ------------------
# fig, ax1 = plt.subplots(figsize=(8, 5))  # Set figure size
# ax1.set_xlabel("rt-epochs/forward epochs", fontsize=14)
# ax1.set_ylabel("Collisions [%]", color="tab:red", fontsize=14)
# ax1.plot(epochs_df["rt_epochs"], epochs_df["percentage_test_collisions"], label="Percentage of Collisions", color="tab:red", marker="o")
# ax1.tick_params(axis="y", labelcolor="tab:red", labelsize=12)
# ax1.tick_params(axis="x", labelsize=12)

# # Add a second y-axis for test_obst_loss
# ax1_2 = ax1.twinx()
# ax1_2.set_ylabel("Obstacle Loss", color="tab:blue", fontsize=14)
# ax1_2.plot(epochs_df["rt_epochs"], epochs_df["test_obst_loss"], label="Obstacle Loss", color="tab:blue", marker="s")
# ax1_2.tick_params(axis="y", labelcolor="tab:blue", labelsize=12)
# # ax1.set_title("rt_epochs vs Collisions and Obstacle Loss", fontsize=14)

# # Save the plot as SVG
# collisions_loss_svg_path = os.path.join(figures_dir, f"rt_epochs_vs_collisions_and_loss_{current_datetime}.svg")
# plt.tight_layout()
# plt.savefig(collisions_loss_svg_path, format="svg")
# print(f"rt_epochs vs Collisions and Obstacle Loss plot saved to {collisions_loss_svg_path}")
# # plt.close(fig)

# # ------------------ Plot 2: rt_epochs vs total_loss and test_metric ------------------
# fig, ax2 = plt.subplots(figsize=(8, 5))  # Set figure size
# ax2.set_xlabel("rt-epochs/forward epochs", fontsize=14)
# ax2.set_ylabel("Total Loss", color="tab:green", fontsize=14)
# ax2.plot(epochs_df["rt_epochs"], epochs_df["test_loss"], label="Total Loss", color="tab:green", marker="o")
# ax2.tick_params(axis="y", labelcolor="tab:green", labelsize=12)
# ax2.tick_params(axis="x", labelsize=12)

# # Add a second y-axis for test_metric
# ax2_2 = ax2.twinx()
# ax2_2.set_ylabel("Straight-Line Distance / Distance Covered", color="tab:purple", fontsize=14)
# ax2_2.plot(epochs_df["rt_epochs"], epochs_df["test_metric"], label="Test Metric", color="tab:purple", marker="o")
# ax2_2.tick_params(axis="y", labelcolor="tab:purple", labelsize=12)
# # ax2.set_title("rt_epochs vs Total Loss and Test Metric", fontsize=14)

# # Save the plot as SVG
# total_loss_metric_svg_path = os.path.join(figures_dir, f"rt_epochs_vs_total_loss_and_metric_{current_datetime}.svg")
# plt.tight_layout()
# plt.savefig(total_loss_metric_svg_path, format="svg")
# print(f"rt_epochs vs Total Loss and Test Metric plot saved to {total_loss_metric_svg_path}")
# plt.show()
# # plt.close(fig)