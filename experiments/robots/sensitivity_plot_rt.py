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
epochs_file = os.path.join(sensitivity_dir, "rt_epochs2.json")

with open(epochs_file, "r") as f:
    epochs_data = json.load(f)

# Convert JSON data to DataFrame
epochs_df = pd.DataFrame(epochs_data)

# Get the current timestamp
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ------------------ Plot 1: rt_epochs vs percentage_test_collisions and test_obst_loss ------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot rt_epochs vs percentage_test_collisions
ax1.set_xlabel("rt_epochs")
ax1.set_ylabel("Collisions [%]", color="tab:red")
ax1.plot(epochs_df["rt_epochs"], epochs_df["percentage_test_collisions"], label="Percentage of Collisions", color="tab:red", marker="o")
ax1.tick_params(axis="y", labelcolor="tab:red")

# Add a second y-axis for test_obst_loss
ax2 = ax1.twinx()
ax2.set_ylabel("Obstacle Loss", color="tab:blue")
ax2.plot(epochs_df["rt_epochs"], epochs_df["test_obst_loss"], label="Obstacle Loss", color="tab:blue", marker="s")
ax2.tick_params(axis="y", labelcolor="tab:blue")

# Add title and grid
plt.title("rt_epochs vs Collisions and Obstacle Loss")
fig.tight_layout()

# Save the plot
collisions_loss_pdf_path = os.path.join(figures_dir, f"rt_epochs_vs_collisions_and_loss_{current_datetime}.pdf")
plt.savefig(collisions_loss_pdf_path)
print(f"rt_epochs vs Collisions and Obstacle Loss plot saved to {collisions_loss_pdf_path}")

# ------------------ Plot 2: rt_epochs vs test_metric ------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Plot rt_epochs vs test_metric
ax.set_xlabel("rt_epochs")
ax.set_ylabel("Test Metric", color="tab:green")
ax.plot(epochs_df["rt_epochs"], epochs_df["test_metric"], label="Test Metric", color="tab:green", marker="o")
ax.tick_params(axis="y", labelcolor="tab:green")

# Add title and grid
plt.title("rt_epochs vs Test Metric")
fig.tight_layout()

# Save the plot
test_metric_pdf_path = os.path.join(figures_dir, f"rt_epochs_vs_test_metric_{current_datetime}.pdf")
plt.savefig(test_metric_pdf_path)
print(f"rt_epochs vs Test Metric plot saved to {test_metric_pdf_path}")

# Show plots
plt.show()