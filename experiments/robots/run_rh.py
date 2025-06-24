import sys, os, logging, torch, time, json, copy
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
save_path = os.path.join(BASE_DIR, 'experiments', 'robots','saved_results', args.save_path)

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
xbar_diag = torch.tensor([-1, 4, 0, 0, 5., 4 ,0 ,0 ])
xbar_direct = torch.tensor([5, 4, 0, 0, -1., 4 ,0 ,0 ])
xbar_center = torch.tensor([0.5, 4, 0, 0, 1.5, 4 ,0 ,0 ])

# x0s = [torch.tensor([4, 0, 0, 0,   # x y vx vy
#                     0, 0, 0, 0,
#                     ]),
#         torch.tensor([5, 4, 0, 0,   # x y vx vy
#                     -1, 4, 0, 0,
#                     ])]
# xbars = [torch.tensor([5, 4, 0, 0, 
#                        -1, 4 ,0 ,0 ]),
#          torch.tensor([0, 8, 0, 0, 
#                        4, 8 ,0 ,0 ])]
# training_ranges = [(4, 4.1), (8, 8.1)] # y intervals for the training data


# x0s = [torch.tensor([4, 0, 0, 0,   # x y vx vy
#                     0, 0, 0, 0,
#                     ]),
#         torch.tensor([5, 4, 0, 0,   # x y vx vy
#                     -1, 4, 0, 0,
#                     ])]
# xbars = [torch.tensor([5, 4, 0, 0, 
#                        -1, 4 ,0 ,0 ]),
#          torch.tensor([4, 0, 0, 0, 
#                        0, 0 ,0 ,0 ])]
# training_ranges = [(4, 4.1), (0, 0.1)] # y intervals for the training data


xbars = [torch.tensor([4, 0, 0, 0,   # x y vx vy
                    0, 0, 0, 0,
                    ]),
        torch.tensor([5, 4, 0, 0,   # x y vx vy
                    -1, 4, 0, 0,
                    ])]
x0s = [torch.tensor([5, 4, 0, 0, 
                       -1, 4 ,0 ,0 ]),
         torch.tensor([4, 0, 0, 0, 
                       0, 0 ,0 ,0 ])]
training_ranges = [(0, 0.1), (4, 4.1)] # y intervals for the training data

# Update the obstacle centers and covariances
# obstacle_centers = args.obstacle_centers
# new_obstacle_centers = copy.deepcopy(obstacle_centers)
# for center in new_obstacle_centers:
#     center[0, 1] += 4  
# obstacle_centers += new_obstacle_centers
# obstacle_covs = args.obstacle_covs*2

obstacle_centers = args.obstacle_centers
obstacle_covs = args.obstacle_covs


# data for plots
t_ext = args.horizon
n_agents = args.n_agents

# ------------ 2. Plant ------------
# x0 = torch.tensor([4., 0., 0., 0.,   # x y vx vy
#                     0., 0., 0., 0.,
#                     ])
plant_input_init = None     # all zero
plant_state_init = None    # same as xbar
sys = RobotsSystem(
    x_init=plant_state_init,
    u_init=plant_input_init, linear_plant=args.linearize_plant, k=args.spring_const, n_agents=n_agents
).to(device)

# ------------ 3. Loss ------------
Q = 95*torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info
Qs = 1*torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info 


# ------------ 4. Training ------------
controllers = []
distance_metrics = []
x_verifs = []
u_verifs = []
e_verifs = []
v_verifs = []
test_collisions_list = []
test_collisions_percentage_list = []
test_loss_list = []
test_obst_loss_list = []
test_metrics_list = []
logger.info('\n------------ Begin training ------------')
for i in range(len(x0s)):
    # Initialize the dataset
    dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, x_bar = xbars[i], x0=x0s[i], 
                            std_ini=args.std_init_plant, n_agents=args.n_agents)
    
    # divide to train and test
    train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500, y_interval=training_ranges[i])
    train_data, test_data = train_data.to(device), test_data.to(device)
    
    # batch the data
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_data = train_data      # use the entire train data for validation
    
    # initialize the controller
    ctl = PerfBoostController(
        noiseless_forward=sys.noiseless_forward,
        input_init=sys.x_init, output_init=sys.u_init,
        dim_internal=args.dim_internal, dim_nl=args.dim_nl,
        initialization_std=args.cont_init_std,
        output_amplification=20,
    ).to(device)
    param_vector = torch.cat([p.data.view(-1).cpu() for p in ctl.parameters()])
    logger.info(f"Controller {i} hash: {hash(param_vector.numpy().tobytes())}")
    # Initialize the loss function
    loss_fn = RobotsLoss(
        Q=Q,Qs = Qs, alpha_u=args.alpha_u, xbar=xbars[i],
        loss_bound=None, sat_bound=None,
        alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,obstacle_centers=obstacle_centers,obstacle_covs=obstacle_covs,
        min_dist=args.min_dist if args.col_av else None,
        n_agents=sys.n_agents if args.col_av else None,
    )
    # Initialize the optimizer
    optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
    # Load model parameters from a file 
    if args.model_path and i == 0:
        resolved_path = os.path.abspath(args.model_path)
        logger.info(f"Resolved model path: {resolved_path}")
        # saved_model = torch.load(resolved_path, map_location=device)
        # del saved_model["Q"]
        # ctl.c_ren.load_state_dict(saved_model)  # Load the model parameters into the controller
        # print(ctl.c_ren.state_dict())
        ctl = torch.load(resolved_path, map_location=device, weights_only=False)
        logger.info(f'[INFO] Loaded trained controller {i} as a complete object.')
        logger.info("Model loaded successfully.")
        param_vector = torch.cat([p.data.view(-1).cpu() for p in ctl.parameters()])
        logger.info(f"Controller {i} hash: {hash(param_vector.numpy().tobytes())}")

    else:
        print(args.use_previous_params)
        if args.use_previous_params == 1 and i > 0:
            # Load the previous controller's parameters
            prev_controller = controllers[i-1]
            prev_params = prev_controller.get_parameters_as_vector()
            prev_params_mlp = prev_controller.get_mlp_parameters()
            ctl.set_parameters_as_vector(prev_params)
            ctl.set_mlp_parameters(prev_params_mlp)
            print(f"Controller {i} parameters set to previous controller's parameters.")
            # assert torch.allclose(
            #     torch.as_tensor(ctl.get_parameters_as_vector()),
            #     torch.as_tensor(prev_params),
            #     atol=1e-6
            # ), "Parameter copy failed!"

            # assert torch.allclose(
            #     torch.as_tensor(ctl.get_mlp_parameters()),
            #     torch.as_tensor(prev_params_mlp),
            #     atol=1e-6
            # ), "MLP parameter copy failed!"
            # logger.info("Previous controller parameters loaded successfully.")
        param_vector = torch.cat([p.data.view(-1).cpu() for p in ctl.parameters()])
        logger.info(f"Controller {i} hash: {hash(param_vector.numpy().tobytes())}")

        logger.info(f'\n------------ Training controller {i} ------------')
        total_params = sum(p.numel() for p in ctl.parameters())
        print(f"Number of parameters: {total_params}")
        best_valid_loss = 1e10
        t = time.time()

        if not (0 <= args.rt_epochs <= 1):
            raise ValueError("rt_epochs must be a fraction between 0 and 1.")
        if i == 0:
            epochs = args.epochs
        else:  
            epochs = int(args.rt_epochs*args.epochs)

        for epoch in range(epochs):
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
        # if args.return_best and (args.rt_epochs > 0 or i == 0):
        #     ctl.set_parameters_as_vector(best_params_ren)
        #     ctl.set_mlp_parameters(best_params_mlp)

        # ------  Save and evaluate the forward model ------
        # res_dict = ctl.c_ren.state_dict()
        # print(res_dict)
        # # TODO: append args
        # res_dict['Q'] = Q
        filename = os.path.join(save_folder, f'trained_controller_{i}.pt')
        # torch.save(res_dict, filename)
        torch.save(ctl, filename)
        logger.info(f'[INFO] saved trained model {i}.')
        
    controllers.append(ctl)
    print(f"Controller {i} parameters: {ctl.get_parameters_as_vector()}")

    # evaluate on the test data
    logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
    with torch.no_grad():
        # simulate over horizon steps
        x_log, e_log, u_log= sys.rollout(
            controller=ctl, data=test_data, train=False,
        )
        test_metrics = compute_distance_metric(x_log, test_data, sys.n_agents)
        test_metrics_list.append(float(test_metrics))
        # loss
        test_loss, test_obst_loss = loss_fn.forward(x_log, u_log,e_log)[:2]
        test_loss, test_obst_loss = test_loss.item(), test_obst_loss.item()
        test_loss_list.append(test_loss)
        test_obst_loss_list.append(test_obst_loss)
        
        msg = "Loss: %.4f" % (test_loss)
        msg += " -- Distance metric: %.4f" % (test_metrics)
        if args.alpha_obst:
            msg += " -- Obstacle Loss: %.4f" % (test_obst_loss)

    # count collisions
    test_collisions = 0
    if args.col_av:
        test_collisions, percentage_test_collisions = loss_fn.count_collisions(x_log)
        test_collisions_list.append(test_collisions)
        test_collisions_percentage_list.append(percentage_test_collisions)
        msg += ' -- Number of collisions = %i' % test_collisions + ' -- Percentage of collisions = %.2f' % percentage_test_collisions
    if args.alpha_obst:
        obst_col, obst_col_percentage, obst_col_cases = loss_fn.count_obstacle_collisions(x_log, test_data)
        msg += ' -- Number of obstacle collisions = %i' % obst_col + ' -- Percentage of obstacle collisions = %.2f' % obst_col_percentage
    logger.info(msg)

    # Do a forward rollout 

    data_verif = torch.zeros(1, args.horizon, 16)
    data_verif[:, 0:1, :8] = \
        x0s[i]
    data_verif[0:1, 1:, 8:] = \
        xbars[i]
    x_verif, e_verif, u_verif = sys.rollout(ctl, data_verif)
    v_verif = sys.v_log
    x_verifs.append(x_verif)
    e_verifs.append(e_verif)
    distance_metric = compute_distance_metric(x_verif, data_verif, sys.n_agents)
    distance_metrics.append(distance_metric)
    msg = "Distance metric Forward: %.4f" % (distance_metric)
    logger.info(msg)

    plot_trajectories(
        x_verif[0, :, :], # remove extra dim due to batching
        xbar=xbars[i], n_agents=sys.n_agents,
        save_folder=save_folder, filename=f'CL_trained_{i}.png',
        text="rPB - trained controller", T=t_ext, 
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs
    )

    u_verif = u_verif.cpu().detach().numpy()
    u_verifs.append(u_verif)
    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    axs[0].plot(np.array(range(u_verif.shape[1])), u_verif[0,:,0],label = "dX")
    axs[0].plot(np.array(range(u_verif.shape[1])), u_verif[0,:,1],label = "dY")
    axs[0].set_title("Robot 1")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Delta ref")
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(np.array(range(u_verif.shape[1])), u_verif[0,:,2],label = "dX")
    axs[1].plot(np.array(range(u_verif.shape[1])), u_verif[0,:,3],label = "dY")
    axs[1].set_title("Robot 2")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Delta ref")
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle
    plt.suptitle(f'Performance boosting offset to the reference over time', fontsize=13)
    plt.savefig(os.path.join(save_folder, f"U_over_time_{i}.png"))
    plt.close()

# Concatenate the rollouts
x_verif = torch.cat(x_verifs, dim=1)
plot_trajectories(
    x_verif[0, :, :], # remove extra dim due to batching
    xbar=xbars[i], n_agents=sys.n_agents,
    save_folder=save_folder, filename=f'CL_trained_rt.png',
    text="rPB - trained controller", T=t_ext, 
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)
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
        # "train_collisions": int(train_collisions),
        # "percentage_train_collisions": float(percentage_train_collisions),
        "test_collisions": list(test_collisions_list),
        "percentage_test_collisions": list(test_collisions_percentage_list),
        "test_loss": list(test_loss_list),
        "test_obst_loss": list(test_obst_loss_list),
        'test_obst_col': int(obst_col),
        'test_obst_col_percentage': float(obst_col_percentage),
        "test_metric": list(test_metrics_list),  # Ensure test_metrics is a float
        "num_parameters": int(total_params),
    }
    
    existing_results.append(new_results)

    # Save the updated results back to the JSON file
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=4)

        print(f"Results saved to {results_file}")

