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
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', args.save_path, 'num_rollouts_'+str(args.num_rollouts))

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

xbar_direct2 = torch.tensor([4, 6, 0, 0,
                             6.5, 6, 0, 0])

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


network_robots = Network([robot1, robot2], [ctl1, ctl2])
# # ------------ 4. Loss ------------
Q = 150*torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info
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
 
# ------------ 6. Training ------------
# plot closed-loop trajectories before training the controller
logger.info('Plotting closed-loop trajectories before training the controller...')

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



# x_log, u_log, v_log = network_robots.rollout(
#     data_list=[plot_data1, plot_data2], device=device)
x_log, u_log, v_log, e_log = network_robots.rollout(
    data_list=[data_verif_1, data_verif_2], device=device)

# Calculate the average distance between the two robots for the first 100 seconds
average_distance_straight = calculate_average_distance(x_log[0, :, :], t=100)
average_distance_diag = calculate_average_distance(x_log[1, :, :], t=100)
average_distance_crossed = calculate_average_distance(x_log[2, :, :], t=100)
logger.info(f"Average distance between robots for the first 100 seconds: {average_distance_straight}")
logger.info(f"Average distance between robots for the first 100 seconds: {average_distance_diag}")
logger.info(f"Average distance between robots for the first 100 seconds: {average_distance_crossed}")



# Calculate the distance between the two robots at the end of the trajectory
# distance = torch.norm(x_log[:, -1, 0:2] - x_log[:, -1, 4:6], dim=-1)
# logger.info(f"Distance between robots at the end of the trajectory: {distance.item()}")


# x_log = x_log.permute(0, 2, 1, 3)  # (batch, T, n_agents, state_dim)
# x_log = x_log.reshape(x_log.shape[0], x_log.shape[1], -1)  # (batch, T, n_agents * state_dim)
plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    xbar=xbar_direct, n_agents=args.n_agents,
    save_folder=save_folder, filename='straight_init.png',
    text="CL - before training", T=t_ext,
    obstacle_centers=obstacle_centers,
    obstacle_covs=obstacle_covs,
)

plot_trajectories(
    x_log[1, :, :], # remove extra dim due to batching
    xbar=xbar_direct2, n_agents=args.n_agents,
    save_folder=save_folder, filename='diag_init.png',
    text="CL - before training", T=t_ext,
    obstacle_centers=obstacle_centers,
    obstacle_covs=obstacle_covs,
)


plot_trajectories(
    x_log[2, :, :], # remove extra dim due to batching
    xbar=xbar_direct3, n_agents=args.n_agents,
    save_folder=save_folder, filename='crossed_init.png',
    text="CL - before training", T=t_ext,
    obstacle_centers=obstacle_centers,
    obstacle_covs=obstacle_covs,
)

logger.info('\n------------ Begin training ------------')
best_valid_loss = 1e9
t = time.time()
plot_at_best_epoch = args.plot_at_best_epoch
for epoch in range(1+args.epochs):
    # print(f"Epoch {epoch}")
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        train_data_batch_robot1 = torch.cat((train_data_batch[:, :, 0:4], train_data_batch[:, :, 8:12]), dim=-1)  # (batch, T, 8)
        train_data_batch_robot2 = torch.cat((train_data_batch[:, :, 4:8], train_data_batch[:, :, 12:16]), dim=-1)  # (batch, T, 8)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # simulate over horizon steps
        x_log, u_log, v_log, e_log = network_robots.rollout(data_list=[train_data_batch_robot1, train_data_batch_robot2], device=device, train=True)

        # loss of this rollout
        loss1 = loss_fn.forward(x_log, u_log,e_log)[0]
        loss_form = loss_fn.forward(x_log, u_log,e_log)[-1]
        # take a step
        loss1.backward()
        
        optimizer1.step()
        optimizer2.step()

    # print info
    if epoch%args.log_epoch == 0:
        msg = 'Epoch: %i --- train loss: %.2f -- form loss: %.2f' % (epoch, loss1, loss_form)

        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                network_robots = Network([robot1, robot2], [ctl1, ctl2])
                x_log, u_log, v_log, e_log = network_robots.rollout(data_list=[train_data_batch_robot1, train_data_batch_robot2], device=device)
                # loss of the valid data
                loss_valid = loss_fn.forward(x_log, u_log,e_log)[0]
            msg += ' ---||--- validation loss: %.2f' % (loss_valid.item())
            # compare with the best valid loss
            if loss_valid.item()<best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_params_ren_1 = ctl1.get_parameters_as_vector()  # record state dict if best on valid
                best_params_ren_2 = ctl2.get_parameters_as_vector()  # record state dict if best on valid   

                best_params_mlp_1 = ctl1.get_mlp_parameters()
                best_params_mlp_2 = ctl2.get_mlp_parameters()
                
                msg += ' (best so far)'
                
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % (duration)
        logger.info(msg)
        t = time.time()

        if plot_at_best_epoch:
            # plot closed-loop trajectories using the best controller
            # logger.info('Plotting closed-loop trajectories using the best controller...')
            x_log, u_log, v_log, e_log = network_robots.rollout(
                data_list=[data_verif_1, data_verif_2], device=device)

            # Calculate the distance between the two robots at the end of the trajectory
            # distance = torch.norm(x_log[:, -1, 0:2] - x_log[:, -1, 4:6], dim=-1)
            # logger.info(f"Distance between robots at the end of the trajectory: {distance.item()}")
            plot_trajectories(
                x_log[0, :, :], # remove extra dim due to batching
                xbar=xbar_direct, n_agents=args.n_agents,
                save_folder=save_folder, filename=f'CL_best_{epoch}.png',
                text=f"CL - best controller - {epoch}", T=t_ext,
                obstacle_centers=obstacle_centers,
                obstacle_covs=obstacle_covs,
            )

# set to best seen during training
# if args.return_best:
#     ctl1.set_parameters_as_vector(best_params_ren_1)
#     ctl2.set_parameters_as_vector(best_params_ren_2)
#     ctl1.set_mlp_parameters(best_params_mlp_1)
#     ctl2.set_mlp_parameters(best_params_mlp_2)

network_robots = Network([robot1, robot2], [ctl1, ctl2])



# # ------ 7. Save and evaluate the trained model ------
# save
torch.save(ctl1, os.path.join(save_folder, 'ctl1.pth'))
torch.save(ctl2, os.path.join(save_folder, 'ctl2.pth'))
logger.info('[INFO] saved trained controllers.')

# res_dict = ctl1.c_ren.state_dict()
# # TODO: append args
# res_dict['Q'] = Q
# filename = os.path.join(save_folder, 'trained_controller'+'.pt')
# torch.save(res_dict, filename)
# logger.info('[INFO] saved trained model.')

# # collision_log_file = os.path.join(save_folder, 'collision_log.csv')
# #
# # collision_log_file = os.path.join('collision_log.csv')
# # with open(collision_log_file, 'a') as f:
#     # f.write("dim_internal,dim_nl,train_collisions,test_collisions\n")

# # evaluate on the train data
# logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % train_data.shape[0])
# with torch.no_grad():
#     x_log, e_log, u_log= sys.rollout(
#         controller=ctl, data=train_data, train=False,
#     )   # use the entire train data, not a batch
#     # evaluate losses
#     loss = loss_fn.forward(x_log, u_log,e_log)[0]
#     msg = 'Loss: %.4f' % (loss)
# # count collisions
# train_collisions = 0
# if args.col_av:
#     train_collisions, percentage_train_collisions = loss_fn.count_collisions(x_log)
#     msg += ' -- Number of collisions = %i' % train_collisions
# logger.info(msg)

# # evaluate on the test data
# logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
# with torch.no_grad():
#     # simulate over horizon steps
#     x_log, e_log, u_log= sys.rollout(
#         controller=ctl, data=test_data, train=False,
#     )
#     test_metrics = compute_distance_metric(x_log, test_data, sys.n_agents)
#     # loss
#     test_loss, test_obst_loss = loss_fn.forward(x_log, u_log,e_log)[:2]
#     test_loss, test_obst_loss = test_loss.item(), test_obst_loss.item()
    
#     msg = "Loss: %.4f" % (test_loss)
#     msg += " -- Distance metric: %.4f" % (test_metrics)
#     if args.alpha_obst:
#         msg += " -- Obstacle Loss: %.4f" % (test_obst_loss)

# # count collisions
# test_collisions = 0
# if args.col_av:
#     test_collisions, percentage_test_collisions = loss_fn.count_collisions(x_log)
#     msg += ' -- Number of collisions = %i' % test_collisions + ' -- Percentage of collisions = %.2f' % percentage_test_collisions
# if args.alpha_obst:
#     obst_col, obst_col_percentage, obst_col_cases = loss_fn.count_obstacle_collisions(x_log, test_data)
#     msg += ' -- Number of obstacle collisions = %i' % obst_col + ' -- Percentage of obstacle collisions = %.2f' % obst_col_percentage
# logger.info(msg)

# # # Evaluate the round-trip trajectory
# # logger.info('Evaluating the round-trip trajectory...')
# # with torch.no_grad():
# #     x_log, e_log, u_log = sys.rollout(
# #         controller=ctl, data=data_verif[3:4, :, :], train=False,
# #     )
# #     rt_loss, rt_obst_loss  = loss_fn.forward(x_log, u_log, e_log)[:2]
# #     rt_loss, rt_obst_loss = rt_loss.item(), rt_obst_loss.item()
# #     msg = 'Round-trip Loss: %.4f' % (rt_loss)
# #     rt_test_collisions = 0
# #     if args.col_av:
# #         rt_test_collisions, percentage_rt_collisions= loss_fn.count_collisions(x_log)
# #         msg += ' -- Number of collisions = %i' % rt_test_collisions + ' -- Percentage of collisions = %.2f' % percentage_rt_collisions
# #     if args.alpha_obst:
# #         msg += " -- Obstacle Loss: %.4f" % (rt_obst_loss)
# #     logger.info(msg)

# # with open(collision_log_file, 'a') as f:
# #     f.write(f"{args.dim_internal},{args.dim_nl},{train_collisions},{test_collisions}\n")
# # Log the results based on the --record argument
# if args.record:

#     # Define the output directory and file
#     results_dir = os.path.join('sensitivity_study')
#     os.makedirs(results_dir, exist_ok=True)
#     results_file = os.path.join(results_dir, f"{args.filename.strip().lower()}.json")

#     # Load existing data if the file exists
#     if os.path.exists(results_file):
#         with open(results_file, 'r') as f:
#             existing_results = json.load(f)
#             # Ensure existing_results is a list
#             if isinstance(existing_results, dict):
#                 existing_results = [existing_results]
#     else:
#         existing_results = []

#     # Add the new results to the existing data
#     new_results = {
#         "dim_internal": int(args.dim_internal),
#         "dim_nl": int(args.dim_nl),
#         "num_rollouts": int(args.num_rollouts),
#         "train_collisions": int(train_collisions),
#         "percentage_train_collisions": float(percentage_train_collisions),
#         "test_collisions": int(test_collisions),
#         "percentage_test_collisions": float(percentage_test_collisions),
#         "test_loss": float(test_loss),
#         "test_obst_loss": float(test_obst_loss),
#         'test_obst_col': int(obst_col),
#         'test_obst_col_percentage': float(obst_col_percentage),
#         "test_metric": float(test_metrics),  # Ensure test_metrics is a float
#         "num_parameters": int(total_params),
#         "epochs": int(args.epochs),
#     }
#     existing_results.append(new_results)

#     # Save the updated results back to the JSON file
#     with open(results_file, 'w') as f:
#         json.dump(existing_results, f, indent=4)

#         print(f"Results saved to {results_file}")


#     # log_dir = os.path.join('sensitivity_study')
#     # os.makedirs(log_dir, exist_ok=True)
#     # # record_type = args.record.strip().lower()
#     # record_type = args.filename.strip().lower()
#     # log_file = os.path.join(log_dir, f'{record_type}.csv')


#     # # Define the column headers
#     # headers = "dim_internal,dim_nl,train_collisions,test_collisions,rt_test_collisions,rt_loss,rt_obst_loss\n" if args.record == 'dim' else \
#     #         "num_rollouts,train_collisions,test_collisions,rt_test_collisions,rt_loss,rt_obst_loss\n"

#     # # Check if the file exists
#     # if not os.path.exists(log_file):
#     #     # Write the headers if the file doesn't exist
#     #     with open(log_file, 'w') as f:
#     #         f.write(headers)

#     # with open(log_file, 'a') as f:
#     #     if args.record == 'dim':
#     #         f.write(f"{args.dim_internal},{args.dim_nl},{train_collisions},{test_collisions},{rt_test_collisions}, {rt_loss}, {rt_obst_loss}\n")
#     #     elif args.record == 'rollouts':
#     #         f.write(f"{args.num_rollouts},{train_collisions},{test_collisions},{rt_test_collisions}, {rt_loss}, {rt_obst_loss}\n")

# # # count collisions
# # if args.col_av:
# #     num_col = loss_fn.count_collisions(x_log)
# #     msg += ' -- Number of collisions = %i' % num_col
# # logger.info(msg)

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

# Calculate the distance between the two robots at the end of the trajectory
# distance = torch.norm(x_log[:, -1, 0:2] - x_log[:, -1, 4:6], dim=-1)
# logger.info(f"Distance between robots at the end of the trajectory: {distance.item()}")


# x_log = x_log.permute(0, 2, 1, 3)  # (batch, T, n_agents, state_dim)
# x_log = x_log.reshape(x_log.shape[0], x_log.shape[1], -1)  # (batch, T, n_agents * state_dim)
plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    xbar=xbar_direct, n_agents=args.n_agents,
    save_folder=save_folder, filename='straight_trained.png',
    text="CL - after training", T=t_ext,
    obstacle_centers=obstacle_centers,
    obstacle_covs=obstacle_covs,
)
plot_trajectories(
    x_log[1, :, :], # remove extra dim due to batching
    xbar=xbar_direct2, n_agents=args.n_agents,
    save_folder=save_folder, filename='diag_trained.png',
    text="CL - after training", T=t_ext,
    obstacle_centers=obstacle_centers,
    obstacle_covs=obstacle_covs,
)


plot_trajectories(
    x_log[2, :, :], # remove extra dim due to batching
    xbar=xbar_direct3, n_agents=args.n_agents,
    save_folder=save_folder, filename='crossed_trained.png',
    text="CL - after training", T=t_ext,
    obstacle_centers=obstacle_centers,
    obstacle_covs=obstacle_covs,
)
# x_verif, _, u_verif = sys.rollout(ctl, data_verif)
# v_verif = sys.v_log
# plot_trajectories(
#     x_verif[0, :, :], # remove extra dim due to batching
#     xbar=xbar_direct, n_agents=sys.n_agents,
#     save_folder=save_folder, filename='CL_diag_trained.png',
#     text="rPB - trained controller", T=t_ext, 
#     obstacle_centers=loss_fn.obstacle_centers,
#     obstacle_covs=loss_fn.obstacle_covs
# )

# plot_trajectories(
#     x_verif[1, :, :], # remove extra dim due to batching
#     xbar=xbar_diag, n_agents=sys.n_agents,
#     save_folder=save_folder, filename='CL_direct_trained.png',
#     text="rPB - trained controller", T=t_ext, 
#     obstacle_centers=loss_fn.obstacle_centers,
#     obstacle_covs=loss_fn.obstacle_covs
# )

# plot_trajectories(
#     x_verif[2, :, :], # remove extra dim due to batching
#     xbar=xbar_center, n_agents=sys.n_agents,
#     save_folder=save_folder, filename='CL_center_trained.png',
#     text="CL - trained controller", T=t_ext, 
#     obstacle_centers=loss_fn.obstacle_centers,
#     obstacle_covs=loss_fn.obstacle_covs
# )


# # # plot the cases which resulted in collisions with the obstacle
# # if args.col_av and args.alpha_obst:
# #     x_log, _, u_log = sys.rollout(ctl, obst_col_cases)
# #     for i in range(2):
        
# #         plot_trajectories(
# #             x_log[i, :, :], # remove extra dim due to batching
# #             xbar=obst_col_cases[i][0,5,8:], n_agents=sys.n_agents,
# #             save_folder=save_folder, filename=f'Collisions/CL_obstacle_collision_{i}.pdf',
# #             text="CL - trained controller", T=t_ext, 
# #             obstacle_centers=loss_fn.obstacle_centers,
# #             obstacle_covs=loss_fn.obstacle_covs
# #         )

# x_ref_evol = torch.zeros(1,args.horizon+200,4)
# x_ref_evol[:,:,0:2] = u_verif[0:1,:,0:2]
# # x_ref_evol[:,:,4:6] = u_verif[0:1,:,2:4]

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


# # print(u_verif[0,:,:])
