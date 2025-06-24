import os
import json
import pandas as pd
import seaborn as sns
# import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
# plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
# plt.rcParams['text.latex.preamble'] = r'\usepackage{color}'  # Allow color in LaTeX

# Define paths
sensitivity_dir = "sensitivity_study"
figures_dir = os.path.join(sensitivity_dir, "Figures")
os.makedirs(figures_dir, exist_ok=True)

# Load JSON files
dim_file = os.path.join(sensitivity_dir, "dim10.json")
rollouts_file = os.path.join(sensitivity_dir, "rollouts12.json")

with open(dim_file, "r") as f:
    dim_data = json.load(f)

with open(rollouts_file, "r") as f:
    rollouts_data = json.load(f)

# Convert JSON data to DataFrames
dim_df = pd.DataFrame(dim_data)
rollouts_df = pd.DataFrame(rollouts_data)

# Get the current timestamp
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# # ------------------ Plot 1a: Heat Map for Collisions ------------------
# collisions_pivot = dim_df.pivot(index="dim_nl", columns="dim_internal", values="percentage_test_collisions")

# fig, ax = plt.subplots(figsize=(5, 4))  # Slightly smaller square size
# sns.heatmap(collisions_pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax, cbar_kws={'label': 'Percentage of Collisions (%)'})
# # ax.set_title(f"Percentage of Collisions (Rollouts: {dim_df['num_rollouts'].iloc[0]})", fontsize=16)
# ax.set_xlabel("dim-internal", fontsize=14)
# ax.set_ylabel("dim-nl", fontsize=14)
# ax.tick_params(axis="both", labelsize=12)
# ax.invert_yaxis()  # Invert the y-axis

# # Save the plot as SVG
# collisions_svg_path = os.path.join(figures_dir, f"dim_collisions_heatmap_{current_datetime}.svg")
# plt.tight_layout()
# plt.savefig(collisions_svg_path, format="svg")
# print(f"Collisions heat map saved to {collisions_svg_path}")

# # ------------------ Plot 1b: Heat Map for Obstacle Loss ------------------
# loss_pivot = dim_df.pivot(index="dim_nl", columns="dim_internal", values="test_obst_loss")

# fig, ax = plt.subplots(figsize=(5, 4))  # Slightly smaller square size
# sns.heatmap(loss_pivot, annot=True, fmt=".1f", cmap="Blues", ax=ax, cbar_kws={'label': 'Obstacle Loss'})
# # ax.set_title(f"Obstacle Loss (Rollouts: {dim_df['num_rollouts'].iloc[0]})", fontsize=16)
# ax.set_xlabel("dim-internal", fontsize=14)
# ax.set_ylabel("dim-nl", fontsize=14)
# ax.tick_params(axis="both", labelsize=12)
# ax.invert_yaxis()  # Invert the y-axis

# # Save the plot as SVG
# loss_svg_path = os.path.join(figures_dir, f"dim_loss_heatmap_{current_datetime}.svg")
# plt.tight_layout()
# plt.savefig(loss_svg_path, format="svg")
# print(f"Obstacle loss heat map saved to {loss_svg_path}")

# ------------------ Plot 1c: Heat Map for Number of Parameters ------------------
# params_pivot = dim_df.pivot(index="dim_nl", columns="dim_internal", values="num_parameters")

# fig, ax = plt.subplots(figsize=(5, 4))  # Slightly smaller square size
# sns.heatmap(params_pivot, annot=True, fmt=".1e", cmap="Greens", ax=ax, cbar_kws={'label': 'Number of Parameters'})
# # ax.set_title(f"Number of Parameters (Rollouts: {dim_df['num_rollouts'].iloc[0]})", fontsize=16)
# ax.set_xlabel("dim-internal", fontsize=14)
# ax.set_ylabel("dim-nl", fontsize=14)
# ax.tick_params(axis="both", labelsize=12)
# ax.invert_yaxis()  # Invert the y-axis

# # Save the plot as SVG
# params_svg_path = os.path.join(figures_dir, f"dim_params_heatmap_{current_datetime}.svg")
# plt.tight_layout()
# plt.savefig(params_svg_path, format="svg")
# print(f"Number of parameters heat map saved to {params_svg_path}")


# Limit rollouts to 800
rollouts_df = rollouts_df[rollouts_df["num_rollouts"] <= 800]
# Create a single figure with three subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)  # Wide figure with shared x-axis

# ------------------ Subplot 1: Total Loss ------------------
axes[0].set_ylabel("Total Loss", color="tab:green", fontsize=14)
axes[0].plot(rollouts_df["num_rollouts"], rollouts_df["test_loss"], label="Total Loss", color="tab:green", marker="o")
axes[0].tick_params(axis="y", labelcolor="tab:green", labelsize=12)
axes[0].tick_params(axis="x", labelsize=12)
# axes[0].set_title("Metrics vs Number of Rollouts", fontsize=16)

# ------------------ Subplot 2: Collisions and Obstacle Loss ------------------
axes[1].set_ylabel("Collisions [%]", color="tab:red", fontsize=14)
axes[1].plot(rollouts_df["num_rollouts"], rollouts_df["percentage_test_collisions"], label="Collisions [%]", color="tab:red", marker="o")
axes[1].tick_params(axis="y", labelcolor="tab:red", labelsize=12)

# Add a second y-axis for obstacle loss
ax2 = axes[1].twinx()
ax2.set_ylabel("Obstacle Loss", color="tab:blue", fontsize=14)
ax2.plot(rollouts_df["num_rollouts"], rollouts_df["test_obst_loss"], label="Obstacle Loss", color="tab:blue", marker="s")
ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=12)
# Add LaTeX text annotation to the second plot
axes[1].text(
    0.5, 0.9, "% of collisions on 500 test rollouts",
    transform=axes[1].transAxes, fontsize=12, ha="center", va="center", color ="tab:red"
)
# ------------------ Subplot 3: Metric ------------------
axes[2].set_ylabel("Metric", color="tab:purple", fontsize=14)
axes[2].plot(rollouts_df["num_rollouts"], rollouts_df["test_metric"], label="Metric", color="tab:purple", marker="o")
axes[2].tick_params(axis="y", labelcolor="tab:purple", labelsize=12)

# Add LaTeX text annotation to the third plot
axes[2].text(
    0.5, 0.1, "Metric = Birds flight distance / Travelled Distance",
    transform=axes[2].transAxes, fontsize=12, ha="center", va="center", color ="tab:purple"
)


# Add a single x-axis label
fig.text(0.5, 0.04, "Number of Rollouts", ha="center", fontsize=14)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the shared x-axis label
combined_svg_path = os.path.join(figures_dir, f"combined_metrics_vs_num_rollouts_{current_datetime}.svg")
plt.savefig(combined_svg_path, format="svg")
print(f"Combined metrics plot saved to {combined_svg_path}")
plt.show()







# # ------------------ Plot 1: Total Loss vs Number of Rollouts ------------------
# fig, ax = plt.subplots(figsize=(10, 3))  # Wide but short
# ax.set_xlabel("Number of Rollouts", fontsize=14)
# ax.set_ylabel("Total Loss", color="tab:green", fontsize=14)
# ax.plot(rollouts_df["num_rollouts"], rollouts_df["test_loss"], label="Total Loss", color="tab:green", marker="o")
# ax.tick_params(axis="y", labelcolor="tab:green", labelsize=12)
# ax.tick_params(axis="x", labelsize=12)
# # ax.set_title("Total Loss vs Number of Rollouts", fontsize=14)

# # Save the plot as SVG
# total_loss_svg_path = os.path.join(figures_dir, f"total_loss_vs_num_rollouts_{current_datetime}.svg")
# plt.tight_layout()
# plt.savefig(total_loss_svg_path, format="svg")
# print(f"Total Loss vs Number of Rollouts plot saved to {total_loss_svg_path}")
# # plt.close(fig)

# # ------------------ Plot 2: Collisions and Obstacle Loss vs Number of Rollouts ------------------
# fig, ax1 = plt.subplots(figsize=(10, 3))  # Wide but short
# ax1.set_xlabel("Number of Rollouts", fontsize=14)
# ax1.set_ylabel("Collisions [%]", color="tab:red", fontsize=14)
# ax1.plot(rollouts_df["num_rollouts"], rollouts_df["percentage_test_collisions"], label="Collisions [%]", color="tab:red", marker="o")
# ax1.tick_params(axis="y", labelcolor="tab:red", labelsize=12)
# ax1.tick_params(axis="x", labelsize=12)

# # Add a second y-axis for obstacle loss
# ax2 = ax1.twinx()
# ax2.set_ylabel("Obstacle Loss", color="tab:blue", fontsize=14)
# ax2.plot(rollouts_df["num_rollouts"], rollouts_df["test_obst_loss"], label="Obstacle Loss", color="tab:blue", marker="s")
# ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=12)
# # ax1.set_title("Collisions and Obstacle Loss vs Number of Rollouts", fontsize=14)

# # Save the plot as SVG
# collisions_obstacle_loss_svg_path = os.path.join(figures_dir, f"collisions_obstacle_loss_vs_num_rollouts_{current_datetime}.svg")
# plt.tight_layout()
# plt.savefig(collisions_obstacle_loss_svg_path, format="svg")
# print(f"Collisions and Obstacle Loss vs Number of Rollouts plot saved to {collisions_obstacle_loss_svg_path}")
# # plt.close(fig)

# # ------------------ Plot 3: Metric vs Number of Rollouts ------------------
# fig, ax = plt.subplots(figsize=(10, 3))  # Wide but short
# ax.set_xlabel("Number of Rollouts", fontsize=14)
# ax.set_ylabel("Metric (Straight-Line Distance / Distance Covered)", color="tab:purple", fontsize=14)
# ax.plot(rollouts_df["num_rollouts"], rollouts_df["test_metric"], label="Metric", color="tab:purple", marker="o")
# ax.tick_params(axis="y", labelcolor="tab:purple", labelsize=12)
# ax.tick_params(axis="x", labelsize=12)
# # ax.set_title("Metric vs Number of Rollouts", fontsize=14)

# # Save the plot as SVG
# metric_svg_path = os.path.join(figures_dir, f"metric_vs_num_rollouts_{current_datetime}.svg")
# plt.tight_layout()
# plt.savefig(metric_svg_path, format="svg")
# print(f"Metric vs Number of Rollouts plot saved to {metric_svg_path}")
# # plt.close(fig)
# plt.show()









# # ------------------ Plot 2a: Collisions Percentage vs Obstacle Loss ------------------
# fig, ax1 = plt.subplots(figsize=(8, 5))  # Slightly smaller rectangular size
# ax1.set_xlabel("Number of Rollouts", fontsize=14)
# ax1.set_ylabel("Test Collisions [%] (on 500 rollouts)", color="tab:red", fontsize=14)
# ax1.plot(rollouts_df["num_rollouts"], rollouts_df["percentage_test_collisions"], label="Percentage of test collisions", color="tab:red", marker="o")
# ax1.tick_params(axis="y", labelcolor="tab:red", labelsize=12)
# ax1.tick_params(axis="x", labelsize=12)

# # Add a second y-axis for obstacle loss
# ax1_2 = ax1.twinx()
# ax1_2.set_ylabel("Obstacle Loss", color="tab:blue", fontsize=14)
# ax1_2.plot(rollouts_df["num_rollouts"], rollouts_df["test_obst_loss"], label="Obstacle Loss", color="tab:blue", marker="o")
# ax1_2.tick_params(axis="y", labelcolor="tab:blue", labelsize=12)
# # ax1.set_title("Collisions Percentage vs Obstacle Loss", fontsize=16)

# # Save the plot as SVG
# collisions_loss_svg_path = os.path.join(figures_dir, f"collisions_vs_obstacle_loss_{current_datetime}.svg")
# plt.tight_layout()
# plt.savefig(collisions_loss_svg_path, format="svg")
# print(f"Collisions vs Obstacle Loss plot saved to {collisions_loss_svg_path}")

# # ------------------ Plot 2b: Total Loss vs Test Metric ------------------
# fig, ax2 = plt.subplots(figsize=(8, 5))  # Slightly smaller rectangular size
# ax2.set_xlabel("Number of Rollouts", fontsize=14)
# ax2.set_ylabel("Total Loss", color="tab:green", fontsize=14)
# ax2.plot(rollouts_df["num_rollouts"], rollouts_df["test_loss"], label="Total Loss", color="tab:green", marker="o")
# ax2.tick_params(axis="y", labelcolor="tab:green", labelsize=12)
# ax2.tick_params(axis="x", labelsize=12)

# # Add a second y-axis for test metric
# ax2_2 = ax2.twinx()
# ax2_2.set_ylabel("Straight-Line Distance / Distance Covered", color="tab:purple", fontsize=14)
# ax2_2.plot(rollouts_df["num_rollouts"], rollouts_df["test_metric"], label="Straight-Line Distance / Distance Covered", color="tab:purple", marker="o")
# ax2_2.tick_params(axis="y", labelcolor="tab:purple", labelsize=12)
# # ax2.set_title("Total Loss vs Straight-Line Distance / Distance Covered", fontsize=16)

# # Save the plot as SVG
# total_loss_metric_svg_path = os.path.join(figures_dir, f"total_loss_vs_test_metric_{current_datetime}.svg")
# plt.tight_layout()
# plt.savefig(total_loss_metric_svg_path, format="svg")
# print(f"Total Loss vs Test Metric plot saved to {total_loss_metric_svg_path}")

# plt.show()