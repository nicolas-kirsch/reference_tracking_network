import sys, os, logging, torch, time, json
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController
from loss_functions import RobotsLoss
from utils.assistive_functions import WrapLogger
from utils.assistive_functions import compute_distance_metric


args = argument_parser()
# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', args.save_path)

# save_folder = os.path.join(save_path, 'perf_boost_'+now)
save_folder = os.path.join(
    save_path,
    f"{now}_nl_{args.dim_nl}_int_{args.dim_internal}_rol_{args.num_rollouts}_ep_{args.epochs}_rt_{args.rt_epochs}"
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
# xbar_direct = torch.tensor([5, 4, 0, 0, -1., 4 ,0 ,0 ])
# xbar_diag = torch.tensor([-1, 4, 0, 0, 5., 4 ,0 ,0 ])
# xbar_center = torch.tensor([-2, 4, 0, 0, 6, 4 ,0 ,0 ])
xbar_direct = torch.tensor([-1, 4, 0, 0, 5., 4 ,0 ,0 ])
xbar_diag = torch.tensor([5, 4, 0, 0, -1., 4 ,0 ,0 ])
xbar_center = torch.tensor([0.5, 4, 0, 0, 1.5, 4 ,0 ,0 ])


obstacle_centers = args.obstacle_centers
obstacle_covs = args.obstacle_covs

x0 = torch.tensor([4, 0, 0, 0,   # x y vx vy
                    0, 0, 0, 0,
                    ])
x0_back = torch.tensor([4, 4, 0, 0,   # x y vx vy
                    0, 4, 0, 0,
                    ])

dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, x_bar=xbar_direct, x0=x0, std_ini=args.std_init_plant, n_agents=args.n_agents)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
train_data, test_data = train_data.to(device), test_data.to(device)


# Generate the backward dataset
if args.rt_epochs > 0:
    # Generate the backward dataset
    back_dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, x_bar=xbar_direct, x0=x0_back, std_ini=args.std_init_plant, n_agents=args.n_agents)
    back_train_data, back_test_data = back_dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500, y_interval=(0, 0.1))
    back_train_data, back_test_data = back_train_data.to(device), back_test_data.to(device)

# data for plots
t_ext = args.horizon
n_agents = args.n_agents
# n_agents = 2

plot_data = test_data[250:350,:,:]
plot_data[:, 0, :4*args.n_agents] = dataset.x0.detach()
plot_data = plot_data.to(device)

# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
if args.rt_epochs > 0:
    # batch the backward data
    back_train_dataloader = DataLoader(back_train_data, batch_size=args.batch_size, shuffle=True)
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

ctl_back = PerfBoostController(
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
    Q=Q,Qs = Qs, alpha_u=args.alpha_u, xbar=train_data[0,:,8:],
    loss_bound=None, sat_bound=None,
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,obstacle_centers=obstacle_centers,obstacle_covs=obstacle_covs,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=sys.n_agents if args.col_av else None,
)

# # backward_loss_fn
if args.rt_epochs > 0:
    loss_fn_back = RobotsLoss(
        Q=Q,Qs = Qs, alpha_u=args.alpha_u, xbar=back_train_data[0,:,8:],
        loss_bound=None, sat_bound=None,
        alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,obstacle_centers=obstacle_centers,obstacle_covs=obstacle_covs,
        min_dist=args.min_dist if args.col_av else None,
        n_agents=sys.n_agents if args.col_av else None,
    )

# ------------ 5. Optimizer ------------
valid_data = train_data      # use the entire train data for validation
if args.rt_epochs > 0:
    back_valid_data = back_test_data
assert not (valid_data is None and args.return_best)
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
if args.rt_epochs > 0:
    back_optimizer = torch.optim.Adam(ctl_back.parameters(), lr=args.lr)
 
# ------------ 6. Training ------------
# plot closed-loop trajectories before training the controller
logger.info('Plotting closed-loop trajectories before training the controller...')
x_log, e_log, u_log = sys.rollout(ctl, plot_data)
x_log, e_log, u_log = sys.augmented_rollout(controller=ctl, controller_back=ctl_back, data=plot_data, train=False,
                                                                x_log=x_log, u_log=u_log, e_log=e_log)
plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    xbar=plot_data[0,5,8:], n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_init.png',
    text="CL - before training", T=t_ext,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)


# Define the initial state and target state

data_verif = torch.zeros(4, args.horizon+200, 16)

data_verif[:, 0:1, :8] = \
    dataset.x0 
data_verif[0:1, 1:, 8:] = \
    xbar_direct
data_verif[1:2, 1:, 8:] = \
    xbar_diag
data_verif[2:3, 1:, 8:] = \
    xbar_center
# Set the trajectory to the target state
# data_verif[3:4, 1:data_verif.size(1)//2, 8:] = xbar_direct
# data_verif[3:4, data_verif.size(1)//2:, 8:] = dataset.x0


x_verif, e_verif, u_verif = sys.rollout(ctl, data_verif)
x_verif, _, u_verif = sys.augmented_rollout(controller=ctl, controller_back=ctl_back, data=data_verif, train=False,
                                                                x_log=x_verif, u_log=u_verif, e_log=e_verif)

total_params = sum(p.numel() for p in ctl.parameters())
print(f"Number of parameters: {total_params}")

plot_trajectories(
    x_verif[0, :, :], # remove extra dim due to batching
    xbar=xbar_direct, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_diag_ref.png',
    text="CL - before training", T=t_ext, 
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs, save= not args.no_save_plot
)

plot_trajectories(
    x_verif[1, :, :], # remove extra dim due to batching
    xbar=xbar_diag, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_direct_ref.png',
    text="CL - before training", T=t_ext, 
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs, save= not args.no_save_plot
)

plot_trajectories(
    x_verif[2, :, :], # remove extra dim due to batching
    xbar=xbar_center, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_center_ref.png',
    text="CL - before training", T=t_ext, 
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs, save= not args.no_save_plot
)

logger.info('\n------------ Begin training ------------')
best_valid_loss = 1e6
t = time.time()
for epoch in range(1+args.epochs):
    # print(f"Epoch {epoch}")
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()
        # simulate over horizon steps
        x_log, e_log, u_log= sys.rollout(
            controller=ctl, data=train_data_batch, train=True,
        )
        # loss of this rollout
        loss = loss_fn.forward(x_log, u_log,e_log)[0]
        # take a step
        loss.backward()
        optimizer.step()

    # print info
    if epoch%args.log_epoch == 0:
        msg = 'Epoch: %i --- train loss: %.2f'% (epoch, loss)
        
        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                x_log_valid, e_log_valid, u_log_valid= sys.rollout(
                    controller=ctl, data=valid_data, train=False,
                )
                # loss of the valid data
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid,e_log_valid)[0]
            msg += ' ---||--- validation loss: %.2f' % (loss_valid.item())
            # compare with the best valid loss
            if loss_valid.item()<best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_params_ren = ctl.get_parameters_as_vector()  # record state dict if best on valid
                best_params_mlp = ctl.get_mlp_parameters()
                msg += ' (best so far)'
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % (duration)
        logger.info(msg)
        t = time.time()

# set to best seen during training
if args.return_best:
    ctl.set_parameters_as_vector(best_params_ren)
    ctl.set_mlp_parameters(best_params_mlp)
    

# ------ 7. Save and evaluate the forward model ------
# save
res_dict = ctl.c_ren.state_dict()
# TODO: append args
res_dict['Q'] = Q
filename = os.path.join(save_folder, 'trained_controller'+'.pt')
torch.save(res_dict, filename)
logger.info('[INFO] saved trained model.')

# # Do a forward rollout
# x_log, e_log, u_log = sys.rollout(ctl, plot_data)
# # Get the last state of the rollout and set is as the initial state for the next rollout
# back_x0 = x_log[:, -1, :].detach().clone()


# -------- 8. Backward training --------
if args.rt_epochs > 0:
    # Create a new controller for the backward training and set it to the same parameters as the forward controller
    ctl_back.set_parameters_as_vector(ctl.get_parameters_as_vector())
    ctl_back.set_mlp_parameters(ctl.get_mlp_parameters())
    # Now train the backward controller
    logger.info('\n------------ Begin backward training ------------')
    t = time.time()
    best_valid_loss = 1e6
    for epoch in range(1+int(args.rt_epochs*args.epochs)):
        # print(f"Epoch {epoch}")
        # iterate over all data batches
        for train_data_batch in back_train_dataloader:
            back_optimizer.zero_grad()
            # simulate over horizon steps
            x_log, e_log, u_log= sys.rollout(
                controller=ctl_back, data=train_data_batch, train=True,
            )
            # loss of this rollout
            loss = loss_fn_back.forward(x_log, u_log,e_log)[0]
            # take a step
            loss.backward()
            back_optimizer.step()

        if epoch%args.log_epoch == 0:
            msg = 'Epoch: %i --- train loss: %.2f'% (epoch, loss)

            if args.return_best:
                # rollout the current controller on the valid data
                with torch.no_grad():
                    x_log_valid, e_log_valid, u_log_valid= sys.rollout(
                        controller=ctl_back, data=back_valid_data, train=False,
                    )
                    # loss of the valid data
                    loss_valid = loss_fn_back.forward(x_log_valid, u_log_valid,e_log_valid)[0]
                msg += ' ---||--- validation loss: %.2f' % (loss_valid.item())
                # compare with the best valid loss
                if loss_valid.item()<best_valid_loss:
                    best_valid_loss = loss_valid.item()
                    best_params_ren = ctl_back.get_parameters_as_vector()  # record state dict if best on valid
                    best_params_mlp = ctl_back.get_mlp_parameters()
                    msg += ' (best so far)'
            duration = time.time() - t
            msg += ' ---||--- time: %.0f s' % (duration)
            logger.info(msg)
            t = time.time()

    # set to best seen during training
    if args.return_best:
        ctl_back.set_parameters_as_vector(best_params_ren)
        ctl_back.set_mlp_parameters(best_params_mlp)



# evaluate on the train data
logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % train_data.shape[0])
with torch.no_grad():
    x_log, e_log, u_log = sys.rollout(
        controller=ctl, data=train_data, train=False,
    )   # use the entire train data, not a batch

    # Augment the data with the backward rollout
    x_log, e_log, u_log = sys.augmented_rollout(controller=ctl, controller_back=ctl_back, data=train_data, train=False,
                                                                x_log=x_log, e_log=e_log, u_log=u_log)

    # evaluate losses
    loss = loss_fn.forward(x_log, u_log,e_log)[0]
    msg = 'Loss: %.4f' % (loss)
# count collisions
train_collisions = 0
if args.col_av:
    train_collisions, percentage_train_collisions = loss_fn.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % train_collisions + ' -- Percentage of collisions = %.2f' % percentage_train_collisions
logger.info(msg)

# evaluate on the test data
logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
with torch.no_grad():
    # simulate over horizon steps
    x_log, e_log, u_log = sys.rollout(
        controller=ctl, data=test_data, train=False,
    )
    # Augment the data with the backward rollout
    x_log, e_log, u_log = sys.augmented_rollout(controller=ctl,controller_back=ctl_back, data=test_data, train=False,
                                                                x_log=x_log, e_log=e_log, u_log=u_log)
    # loss
    test_loss, test_obst_loss = loss_fn.forward(x_log, u_log,e_log)[:2]
    test_loss, test_obst_loss = test_loss.item(), test_obst_loss.item()
    test_metrics = compute_distance_metric(x_log, test_data, sys.n_agents, rt=True)
    msg = "Loss: %.4f" % (test_loss)
    if args.alpha_obst:
        msg += " -- Obstacle Loss: %.4f" % (test_obst_loss)
# count collisions
test_collisions = 0
if args.col_av:
    test_collisions, percentage_test_collisions = loss_fn.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % test_collisions + ' -- Percentage of collisions = %.2f' % percentage_test_collisions
if args.alpha_obst:
    obst_col, obst_col_percentage, obst_col_cases = loss_fn.count_obstacle_collisions(x_log, test_data)
    msg += ' -- Number of obstacle collisions = %i' % obst_col + ' -- Percentage of obstacle collisions = %.2f' % obst_col_percentage
logger.info(msg)


if args.record:

    # Define the output directory and file
    results_dir = os.path.join('sensitivity_study')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{args.filename.strip().lower()}.json")

    # Load existing data if the file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
            # Ensure existing_results is a list
            if isinstance(existing_results, dict):
                existing_results = [existing_results]
    else:
        existing_results = []

    # Add the new results to the existing data
    new_results = {
        "dim_internal": int(args.dim_internal),
        "dim_nl": int(args.dim_nl),
        "num_rollouts": int(args.num_rollouts),
        "rt_epochs": float(args.rt_epochs),
        "train_collisions": int(train_collisions),
        "percentage_train_collisions": float(percentage_train_collisions),
        "test_collisions": int(test_collisions),
        "percentage_test_collisions": float(percentage_test_collisions),
        "test_loss": float(test_loss),
        "test_obst_loss": float(test_obst_loss),
        'test_obst_col': int(obst_col),
        'test_obst_col_percentage': float(obst_col_percentage),
        "test_metric": float(test_metrics),  # Ensure test_metrics is a float
        "num_parameters": int(total_params),
    }
    
    existing_results.append(new_results)

    # Save the updated results back to the JSON file
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=4)

        print(f"Results saved to {results_file}")


# plot closed-loop trajectories using the trained controller
logger.info('Plotting closed-loop trajectories using the trained controller...')
x_log, e_log, u_log = sys.rollout(ctl, plot_data)
x_log, e_log, u_log = sys.augmented_rollout(controller=ctl, controller_back=ctl_back, data=plot_data, train=False,
                                                                x_log=x_log, e_log=e_log, u_log=u_log)

plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    xbar=plot_data[0,5,8:], n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_trained.png',
    text="CL - trained controller", T=t_ext, 
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs, save= not args.no_save_plot
)

x_verif, e_log, u_verif = sys.rollout(ctl, data_verif)
x_verif, e_log, u_verif = sys.augmented_rollout(controller=ctl, controller_back=ctl_back, data=data_verif, train=False,
                                                                x_log=x_verif, e_log=e_log, u_log=u_verif)
v_verif = sys.v_log
plot_trajectories(
    x_verif[0, :, :], # remove extra dim due to batching
    xbar=xbar_direct, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_diag_trained.png',
    text="rPB - trained controller", T=t_ext, 
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs, save= not args.no_save_plot
)

plot_trajectories(
    x_verif[1, :, :], # remove extra dim due to batching
    xbar=xbar_diag, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_direct_trained.png',
    text="rPB - trained controller", T=t_ext, 
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs, save= not args.no_save_plot
)

plot_trajectories(
    x_verif[2, :, :], # remove extra dim due to batching
    xbar=xbar_center, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_center_trained.png',
    text="CL - trained controller", T=t_ext, 
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs, save= not args.no_save_plot
)


# x_ref_evol = torch.zeros(1,args.horizon+200,8)
# x_ref_evol[:,:,0:2] = u_verif[0:1,:,0:2]
# x_ref_evol[:,:,4:6] = u_verif[0:1,:,2:4]

# x_ref_evol = x_ref_evol + xbar_direct

# plot_trajectories(
#    x_ref_evol[0,:,:], # remove extra dim due to batching
#     xbar=xbar_direct, n_agents=sys.n_agents,
#     save_folder=save_folder, filename='CL_xbar_evolution.png',
#     text="CL - evolution of the reference", T=t_ext, dots = True,
#     obstacle_centers=loss_fn.obstacle_centers,
#     obstacle_covs=loss_fn.obstacle_covs
# )
# u_verif = u_verif.cpu().detach().numpy()
# # Create a figure with a 2x2 grid of subplots
# fig, axs = plt.subplots(2, 1, figsize=(10, 7))
# axs[0].plot(np.array(range(u_verif.shape[1])), u_verif[2,:,0],label = "dX")
# axs[0].plot(np.array(range(u_verif.shape[1])), u_verif[2,:,1],label = "dY")
# axs[0].set_title("Robot 1")
# axs[0].set_xlabel("Time (s)")
# axs[0].set_ylabel("Delta ref")
# axs[0].legend()
# axs[0].grid()

# axs[1].plot(np.array(range(u_verif.shape[1])), u_verif[2,:,2],label = "dX")
# axs[1].plot(np.array(range(u_verif.shape[1])), u_verif[2,:,3],label = "dY")
# axs[1].set_title("Robot 2")
# axs[1].set_xlabel("Time (s)")
# axs[1].set_ylabel("Delta ref")
# axs[1].legend()
# axs[1].grid()

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

# plt.suptitle(f'Performance boosting offset to the reference over time \n for the diagonal scenario', fontsize=13)
# plt.savefig(os.path.join(save_folder, "U_over_time.png"))
# plt.close()

# v_verif = v_verif.cpu().detach().numpy()
# # Create a figure with a 2x2 grid of subplots
# fig, axs = plt.subplots(2, 1, figsize=(10, 7))
# axs[0].plot(np.array(range(u_verif.shape[1])), v_verif[0,:,0],label = "v_X")
# axs[0].plot(np.array(range(u_verif.shape[1])), v_verif[0,:,1],label = "v_Y")
# axs[0].set_title("Robot 1")
# axs[0].set_xlabel("Time (s)")
# axs[0].set_ylabel("v")
# axs[0].legend()
# axs[0].grid()

# axs[1].plot(np.array(range(u_verif.shape[1])), v_verif[0,:,2],label = "v_X")
# axs[1].plot(np.array(range(u_verif.shape[1])), v_verif[0,:,3],label = "v_Y")
# axs[1].set_title("Robot 2")
# axs[1].set_xlabel("Time (s)")
# axs[1].set_ylabel("v")
# axs[1].legend()
# axs[1].grid()

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

# plt.suptitle(f'Integral variable over time \n for the diagonal scenario', fontsize=13)
# plt.savefig(os.path.join(save_folder, "V_over_time.png"))
# plt.close()


# print(u_verif[0,:,:])
