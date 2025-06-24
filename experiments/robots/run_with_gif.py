import sys, os, logging, torch, time, json
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset, Network
from utils.plot_functions import *
from controllers import PerfBoostController
from loss_functions import RobotsLoss
from utils.assistive_functions import WrapLogger
from utils.assistive_functions import compute_distance_metric, calculate_average_distance
import distributed_control 

# torch.autograd.set_detect_anomaly(True)
args = argument_parser()
# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', "final_pres_results")

# save_folder = os.path.join(save_path, 'perf_boost_'+now)
save_folder = os.path.join(
    save_path,
    f"{now}_alpha_formation_{args.alpha_formation}"
)
os.makedirs(save_folder)

logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ----- parse and set experiment arguments -----
# args = argument_parser()
msg = print_args(args)
logger.info(msg)
torch.manual_seed(args.random_seed)

# ------------ 1. Dataset ------------
xbar_direct = torch.tensor([0.75, 6, 0, 0,
                            3.25, 6, 0, 0])

xbar_direct2 = torch.tensor([2, 6, 0, 0,
                             4.5, 6, 0, 0])

xbar_direct3 = torch.tensor([3.25, 6, 0, 0,
                             0.75, 6, 0, 0])

# xbar_direct1 = torch.tensor([-1, 4, 0, 0])
# xbar_direct2 = torch.tensor([0, 4, 0, 0])

xbar_diag = torch.tensor([5, 4, 0, 0])
xbar_center = torch.tensor([0.5, 4, 0, 0])

obstacle_centers = args.obstacle_centers
obstacle_covs = args.obstacle_covs

x0 = torch.tensor([0.75, -2, 0, 0, 
                   3.25, -2, 0, 0])   # x y vx vy
x0_1 = x0[:4]   # x y vx vy
x0_2 = x0[4:8]   # x y vx vy

dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, x_bar=xbar_direct, x0=x0, std_ini=args.std_init_plant, n_agents=args.n_agents)

# divide to train and test

train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500, x_interval=args.x_interval, y_interval=args.y_interval)
train_data, test_data = train_data.to(device), test_data.to(device)

train_data_1 = torch.cat((train_data[:, :, 0:4], train_data[:, :, 8:12]), dim=-1)
train_data_2 = torch.cat((train_data[:, :, 4:8], train_data[:, :, 12:16]), dim=-1)
test_data_1 = torch.cat((test_data[:, :, 0:4], test_data[:, :, 8:12]), dim=-1)
test_data_2 = torch.cat((test_data[:, :, 4:8], test_data[:, :, 12:16]), dim=-1)


# data for plots
# t_ext = args.horizon * 4
t_ext = args.horizon + 200 - 1
n_agents = args.n_agents

plot_data = test_data[250:350,:,:]
plot_data1 = test_data_1[250:350,:,:]
plot_data1[:, 0, :4] = x0_1
plot_data2 = test_data_2[250:350,:,:]
plot_data2[:, 0, :4] = x0_2
plot_data1 = plot_data1.to(device)
plot_data2 = plot_data2.to(device)

# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)


# ------------ 2. Plant ------------
plant_input_init = None     # all zero
plant_state_init = None    # same as 

x_init_1 = torch.zeros((1,4)) 
x_init_1[:, :2] = x_init_1[:, :2] + 1e-6  # Add to x and y

x_init_2 = torch.zeros((1,4))
x_init_2[:, :2] = x_init_2[:, :2] + 5e-6 # Add to x and y
# x_init_2 = x0_2

k_distance = 10
robot1 = RobotsSystem(
    x_init=x_init_1,
    u_init=plant_input_init, linear_plant=args.linearize_plant, k=args.spring_const, n_agents=1, leader=False , distance_to_neighbor=args.distance_agents, k_distance=k_distance
).to(device)

robot2 = RobotsSystem(
    x_init=x_init_2,
    u_init=plant_input_init, linear_plant=args.linearize_plant, k=args.spring_const, n_agents=1, leader=False , distance_to_neighbor=args.distance_agents, k_distance=k_distance
).to(device)



# ------------ 3. Controller ------------
ctl1 = PerfBoostController(
    noiseless_forward=robot1.noiseless_forward,
    input_init=robot1.x_init, output_init=robot1.u_init,
    dim_internal=args.dim_internal, dim_nl=args.dim_nl,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)

ctl2 = PerfBoostController(
    noiseless_forward=robot2.noiseless_forward,
    input_init=robot2.x_init, output_init=robot2.u_init,
    dim_internal=args.dim_internal, dim_nl=args.dim_nl,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)
# Load the trained controllers
logger.info('[INFO] Loading trained controllers...')
# controller_folder = os.path.join(BASE_DIR, 'experiments', 'robots', args.save_path, 'num_rollouts_600', '06_13_10_39_18_alpha_formation_10000.0')

controller_folder = os.path.join(BASE_DIR, 'experiments', 'robots', args.save_path, 'num_rollouts_500', '06_23_11_24_47_alpha_formation_10000')
# ctl1.load_state_dict(torch.load(os.path.join(controller_folder, 'ctl1.pth')))
# ctl2.load_state_dict(torch.load(os.path.join(controller_folder, 'ctl2.pth')))
ctl1 = torch.load(os.path.join(controller_folder, 'ctl1.pth'), weights_only=False)
ctl2 = torch.load(os.path.join(controller_folder, 'ctl2.pth'), weights_only=False)

network_robots = Network([robot1, robot2], [ctl1, ctl2])
# # ------------ 4. Loss ------------
Q = 95*torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info
# Q = 0*torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info
Qs = 1*torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info 
loss_fn = RobotsLoss(
    Q=Q,Qs = Qs, alpha_u=args.alpha_u, xbar=train_data_1[0,:,:],
    loss_bound=None, sat_bound=None,
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,obstacle_centers=obstacle_centers,obstacle_covs=obstacle_covs,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=args.n_agents if args.col_av else None,
    alpha_formation=args.alpha_formation, desired_distance=args.distance_agents
)

# loss_fn_2 = RobotsLoss(
#     Q=Q,Qs = Qs, alpha_u=args.alpha_u, xbar=train_data_2[0,:,:],
#     loss_bound=None, sat_bound=None,
#     alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,obstacle_centers=obstacle_centers,obstacle_covs=obstacle_covs,
#     min_dist=args.min_dist if args.col_av else None,
#     n_agents=robot2.n_agents if args.col_av else None,
# )

# # ------------ 5. Optimizer ------------
# valid_data = train_data      # use the entire train data for validation
# valid_data_1 = train_data_1
# valid_data_2 = train_data_2
# assert not (valid_data is None and args.return_best)
optimizer1 = torch.optim.Adam(ctl1.parameters(), lr=args.lr)
optimizer2 = torch.optim.Adam(ctl2.parameters(), lr=args.lr)
 
# # ------------ 6. Training ------------
# # plot closed-loop trajectories before training the controller
# logger.info('Plotting closed-loop trajectories before training the controller...')

data_verif_1 = torch.zeros(3, args.horizon+200, 8)
data_verif_1[:, 0:1, :4] = \
    x0_1
data_verif_1[0:1, 1:, 4:] = \
    xbar_direct[:4]
data_verif_1[1:2, 1:, 4:] = \
    xbar_direct2[:4]
data_verif_1[2:3, 1:, 4:] = \
    xbar_direct3[:4]

data_verif_2 = torch.zeros(3, args.horizon+200, 8)
data_verif_2[:, 0:1, :4] = \
    x0_2
data_verif_2[0:1, 1:, 4:] = \
    xbar_direct[4:8]
data_verif_2[1:2, 1:, 4:] = \
    xbar_direct2[4:8]
data_verif_2[2:3, 1:, 4:] = \
    xbar_direct3[4:8]




# plot closed-loop trajectories using the trained controller
logger.info('Plotting closed-loop trajectories using the trained controller...')
x_log, u_log, v_log, e_log = network_robots.rollout(
    data_list=[data_verif_1, data_verif_2], device=device)

# Calculate the average distance between the two robots for the first 100 seconds
average_distance_straight = calculate_average_distance(x_log[0, :, :], t=100)
average_distance_diag = calculate_average_distance(x_log[1, :, :], t=100)
average_distance_crossed = calculate_average_distance(x_log[2, :, :], t=100)
logger.info(f"Average distance between robots for the first 100 seconds: {average_distance_straight}")
logger.info(f"Average distance between robots for the first 100 seconds: {average_distance_diag}")
logger.info(f"Average distance between robots for the first 100 seconds: {average_distance_crossed}")


# plot_trajectories(
#     x_log[0, :, :], # remove extra dim due to batching
#     xbar=xbar_direct, n_agents=args.n_agents,
#     save_folder=save_folder, filename='straight_trained.png',
#     text="CL - after training", T=t_ext,
#     obstacle_centers=obstacle_centers,
#     obstacle_covs=obstacle_covs,
# )
# Generate frames for the trajectory
save_trajectory_frames(
    x=x_log[0, :, :], xbar=xbar_direct, n_agents=args.n_agents,
    save_folder=os.path.join(save_folder, 'trajectory_frames_straight'), T=100,
    obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs
)

# Create GIF for the trajectory
create_gif_from_frames(
    frame_folder=os.path.join(save_folder, 'trajectory_frames_straight'),
    gif_filename=os.path.join(save_folder, 'trajectory_straight.gif'),
    duration=0.1
)

# plot_trajectories(
#     x_log[1, :, :], # remove extra dim due to batching
#     xbar=xbar_direct2, n_agents=args.n_agents,
#     save_folder=save_folder, filename='diag_trained.png',
#     text="CL - after training", T=t_ext,
#     obstacle_centers=obstacle_centers,
#     obstacle_covs=obstacle_covs,
# )


# plot_trajectories(
#     x_log[2, :, :], # remove extra dim due to batching
#     xbar=xbar_direct3, n_agents=args.n_agents,
#     save_folder=save_folder, filename='crossed_trained.png',
#     text="CL - after training", T=t_ext,
#     obstacle_centers=obstacle_centers,
#     obstacle_covs=obstacle_covs,
# )
# Generate frames for the trajectory
save_trajectory_frames(
    x=x_log[2, :, :], xbar=xbar_direct3, n_agents=args.n_agents,
    save_folder=os.path.join(save_folder, 'trajectory_frames_crossed'), T=100,
    obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs
)

# Create GIF for the trajectory
create_gif_from_frames(
    frame_folder=os.path.join(save_folder, 'trajectory_frames_crossed'),
    gif_filename=os.path.join(save_folder, 'trajectory_crossed.gif'),
    duration=0.1
)
