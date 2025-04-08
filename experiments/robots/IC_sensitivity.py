import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from utils.plot_functions import plot_trajectories
from plants import RobotsSystem
from loss_functions import RobotsLoss
from controllers import PerfBoostController
from config import device
from arg_parser import argument_parser, print_args
import json

args = argument_parser()
save_folder = "round_trip_test_results"
os.makedirs(save_folder, exist_ok=True)

# If the --plot argument is provided, skip collision calculations and only plot
if args.plot:
    # Find the latest round_trip_results.json file
    result_files = [f for f in os.listdir(save_folder) if f.startswith("round_trip_results") and f.endswith(".json")]
    if not result_files:
        print("No results file found in the folder.")
        sys.exit(1)
    latest_file = max(result_files, key=lambda f: os.path.getctime(os.path.join(save_folder, f)))
    results_file = os.path.join(save_folder, latest_file)

    # Define the obstacle centers and covariances
    obstacle_centers = args.obstacle_centers
    obstacle_covs = args.obstacle_covs

    # Load the results
    with open(results_file, "r") as f:
        results = json.load(f)

    print(f"Loaded results from {results_file}")
else:
    # Define the path to the trained model
    model_name = "trained_controller_is_8_nl_8_num_rollouts_500.pt"
    trained_model_path = os.path.join("saved_models", model_name)  # Path to the saved model

    # Load the trained model
    checkpoint = torch.load(trained_model_path, map_location=device)
    Q = checkpoint['Q']  # Load Q matrix
    controller_state_dict = {k: v for k, v in checkpoint.items() if k != 'Q'}

    # Infer dim_internal and dim_nl from the checkpoint
    dim_internal = checkpoint['Y'].shape[0]  # Assuming 'X' represents the internal dimension
    dim_nl = checkpoint['D12'].shape[0]        # Assuming 'Y' represents the nonlinear dimension

    # Define the target state for the round trip
    target_state = torch.tensor([2, 2, 0, 0, -2, 2, 0, 0])  # Adjust as needed

    # Define the obstacle centers and covariances
    obstacle_centers = args.obstacle_centers
    obstacle_covs = args.obstacle_covs

    # Define the system and controller
    n_agents = 2  # Adjust based on your setup
    sys = RobotsSystem(
        x_init=None, u_init=None, linear_plant=False, k=1.0, n_agents=n_agents
    ).to(device)

    ctl = PerfBoostController(
        noiseless_forward=sys.noiseless_forward,
        input_init=sys.x_init, output_init=sys.u_init,
        dim_internal=dim_internal, dim_nl=dim_nl,
        initialization_std=0.1, output_amplification=20
    ).to(device)

    ctl.c_ren.load_state_dict(controller_state_dict)

    Q = 95*torch.kron(torch.eye(n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info
    Qs = 1*torch.kron(torch.eye(n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info 
    loss_fn = RobotsLoss(
        Q=Q, Qs=Qs, alpha_u=args.alpha_u, xbar=target_state,
        loss_bound=None, sat_bound=None,
        alpha_col=args.alpha_col, alpha_obst=args.alpha_obst, obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs,
        min_dist=args.min_dist if args.col_av else None,
        n_agents=sys.n_agents if args.col_av else None,
    )

    # Define the rectangle for initial conditions
    x_range = (-4, 4)  # Range for x values
    y_range = (-3, -1)  # Range for y values

    # Generate random initial conditions
    def generate_initial_conditions(num_points=100, x_range=(-2, 2), y_range=(-2, -1)):
        x_values = torch.empty(num_points).uniform_(*x_range)  # Uniformly distributed x values
        y_values = torch.empty(num_points).uniform_(*y_range)  # Uniformly distributed y values
        initial_conditions = []

        for x, y in zip(x_values, y_values):
            # Create a full initial condition tensor (8-dimensional)
            initial_condition = torch.tensor([x, y, 0, 0, -x, y, 0, 0])  # Symmetric positions for two agents
            initial_conditions.append(initial_condition)

        return initial_conditions

    initial_conditions = generate_initial_conditions(num_points=100, x_range=x_range, y_range=y_range)

    # Test round-trip collisions
    results = []
    for idx, x0 in enumerate(initial_conditions):
        # Create the round-trip trajectory
        data_verif = torch.zeros(1, 400, 16)  # Adjust horizon as needed
        data_verif[:, 0:1, :8] = x0
        data_verif[:, 1:200, 8:] = target_state
        data_verif[:, 200:, 8:] = x0

        # Rollout the controller
        with torch.no_grad():
            x_log, _, _ = sys.rollout(controller=ctl, data=data_verif, train=False)

        # Count collisions
        collisions = loss_fn.count_collisions(x_log)
        results.append({"initial_condition": x0.tolist(), "collisions": collisions})

    # Save the results
    results_file = os.path.join(save_folder, "round_trip_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Round-trip test results saved to {results_file}")

x_coords = [result["initial_condition"][0] for result in results]
y_coords = [result["initial_condition"][1] for result in results]
collision_counts = [result["collisions"] for result in results]

# Create the figure and adjust the layout
fig = plt.figure(figsize=(8, 6))  # Adjust the overall figure size
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 0.01], wspace=0.05)  # Define grid with space for the color bar

# Create the main plot (square region)
ax = fig.add_subplot(gs[0])
ax.set_aspect('equal')  # Ensure the plot area is a square

# Plot the obstacles as density maps
    # plot obstacles
if not obstacle_covs is None:
    assert not obstacle_centers is None
    yy, xx = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    zz = xx * 0
    for center, cov in zip(obstacle_centers, obstacle_covs):
        distr = multivariate_normal(
            cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
            mean=center.detach().clone().cpu().numpy().flatten()
        )
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
    z_min, z_max = np.abs(zz).min(), np.abs(zz).max()

    ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')

# Plot the initial conditions as a scatter plot with a heat map
scatter = ax.scatter(x_coords, y_coords, c=collision_counts, cmap="hot", s=100, vmin=0, vmax=50,edgecolor="black", linewidth=0.5) 
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Number of Collisions", fontsize=12)

# Set plot labels and title
ax.set_xlabel("X Coordinate", fontsize=12)
ax.set_ylabel("Y Coordinate", fontsize=12)
ax.set_title("Initial Conditions Heat Map with Obstacles", fontsize=14)

# Adjust plot limits
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# Save the plot
plot_path = os.path.join(save_folder, "initial_conditions_heatmap.pdf")
plt.tight_layout()
plt.savefig(plot_path)
print(f"Heat map saved to {plot_path}")

# Show the plot
plt.show()