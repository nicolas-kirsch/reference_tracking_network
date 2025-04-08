import sys, os, torch
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from arg_parser import argument_parser
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import plot_trajectories
from controllers import PerfBoostController
from loss_functions import RobotsLoss

# ----- parse and set experiment arguments -----
args = argument_parser()
torch.manual_seed(args.random_seed)

# ------------ 1. Dataset ------------
xbar_train = torch.tensor([2, 2, 0, 0, -2., 2 ,0 ,0 ])
xbar_verif2 = torch.tensor([-2, 2, 0, 0, 2., 2 ,0 ,0 ])
xbar_verif3 = torch.tensor([0.5, 4, 0, 0, 1.5, 4 ,0 ,0 ])
x_init = torch.tensor([7, 4, 0, 0, -3., 4 ,0 ,0 ])

obstacle_centers = args.obstacle_centers
obstacle_covs = args.obstacle_covs

x0 = torch.tensor([-2, -2, 0, 0,
                    2, -2, 0, 0,
                    ])

dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, x_bar=xbar_train, x0=x0, std_ini=args.std_init_plant, n_agents=args.n_agents)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
train_data, test_data = train_data.to(device), test_data.to(device)

# data for plots
t_ext = args.horizon * 4
n_agents = args.n_agents

plot_data = test_data[250:350,:,:]
plot_data[:, 0, :4*args.n_agents] = dataset.x0.detach()
plot_data = plot_data.to(device)

# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant ------------
plant_input_init = None     # all zero
plant_state_init = None    # same as xbar
sys = RobotsSystem(
    x_init=plant_state_init,
    u_init=plant_input_init, linear_plant=args.linearize_plant, k=args.spring_const, n_agents=n_agents
).to(device)

# ------------ 3. Controller ------------
ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init, output_init=sys.u_init,
    dim_internal=args.dim_internal, dim_nl=args.dim_nl,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)

# ------------ 4. Loss ------------
Q = 95*torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info
Qs = 1*torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info 
loss_fn = RobotsLoss(
    Q=Q, Qs=Qs, alpha_u=args.alpha_u, xbar=train_data[0,:,8:],
    loss_bound=None, sat_bound=None,
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst, obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=sys.n_agents if args.col_av else None,
)

# Plot closed-loop trajectories before training the controller
x_log, _, u_log = sys.rollout(ctl, plot_data)

plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    xbar=plot_data[0,5,8:], n_agents=sys.n_agents,
    save_folder='.', filename='CL_init.pdf',save=False,
    text="CL - before training", T=t_ext,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)

# data_verif = torch.zeros(3, args.horizon+200, 16)

# data_verif[:, 0:1, :8] = dataset.x0 
# data_verif[0:1, 1:, 8:] = xbar_train
# data_verif[1:2, 1:, 8:] = xbar_verif2
# data_verif[2:3, 1:, 8:] = xbar_verif3

# x_verif, _, u_verif = sys.rollout(ctl, data_verif)

# plot_trajectories(
#     x_verif[0, :, :], # remove extra dim due to batching
#     xbar=xbar_train, n_agents=sys.n_agents,
#     save_folder='.', filename='CL_diag_ref.pdf',save=False,
#     text="CL - before training", T=t_ext, 
#     obstacle_centers=loss_fn.obstacle_centers,
#     obstacle_covs=loss_fn.obstacle_covs
# )

# plot_trajectories(
#     x_verif[1, :, :], # remove extra dim due to batching
#     xbar=xbar_verif2, n_agents=sys.n_agents,
#     save_folder='.', filename='CL_direct_ref.pdf',save=False,
#     text="CL - before training", T=t_ext, 
#     obstacle_centers=loss_fn.obstacle_centers,
#     obstacle_covs=loss_fn.obstacle_covs
# )

# plot_trajectories(
#     x_verif[2, :, :], # remove extra dim due to batching
#     xbar=xbar_verif3, n_agents=sys.n_agents, save=False,
#     save_folder='.', filename='CL_center_ref.pdf',
#     text="CL - before training", T=t_ext, 
#     obstacle_centers=loss_fn.obstacle_centers,
#     obstacle_covs=loss_fn.obstacle_covs
# )