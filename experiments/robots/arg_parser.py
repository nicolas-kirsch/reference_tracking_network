import argparse, math
import torch
from config import device
from obstacles import obstacles  # Import the obstacles
import ast



 
# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Robots minimal experiment.")

    # experiment
    parser.add_argument('--random-seed', type=int, default=5, help='Random seed. Default is 5.')
    parser.add_argument('--col-av', type=bool, default=True, help='Avoid collisions. Default is True.')
    parser.add_argument('--obst-av', type=bool, default=True, help='Avoid obstacles. Default is True.')

    # dataset
    parser.add_argument('--horizon', type=int, default=100, help='Time horizon for the computation. Default is 100.')
    parser.add_argument('--n-agents', type=int, default=2, help='Number of agents. Default is 2.')
    parser.add_argument('--num-rollouts', type=int, default=500, help='Number of rollouts in the training data. Default is 600.')
    parser.add_argument('--std-init-plant', type=float, default=0.2, help='std of the plant initial conditions. Default is 0.2.') # Anto: I changed the default value from 0.2 to 1

    # plant
    parser.add_argument('--spring-const', type=float, default=1.0 , help='Spring constant. Default is 1.0.')
    parser.add_argument('--linearize-plant', type=bool, default=False, help='Linearize plant or not. Default is False.')

    # controller
    parser.add_argument('--cont-init-std', type=float, default=0.1, help='Initialization std for controller params. Default is 0.1.')
    parser.add_argument('--dim-internal', type=int, default=12, help='Dimension of the internal state of the controller. Adjusts the size of the linear part of REN. Default is 8.')
    parser.add_argument('--dim-nl', type=int, default=12, help='size of the non-linear part of REN. Default is 8.')

    # loss
    parser.add_argument('--alpha-u', type=float, default=0.1/400, help='Weight of the loss due to control input "u". Default is 0.1/400.')  #TODO: 400 is output_amplification^2
    parser.add_argument('--alpha-col', type=float, default=3000, help='Weight of the collision avoidance loss. Default is 100 if "col-av" is True, else None.')
    parser.add_argument('--alpha-obst', type=float, default=5e3, help='Weight of the obstacle avoidance loss. Default is 5e3 if "obst-av" is True, else None.')
    parser.add_argument('--min-dist', type=float, default=1.0, help='TODO. Default is 1.0 if "col-av" is True, else None.')  #TODO: add help

    # optimizer
    parser.add_argument('--batch-size', type=int, default=1500, help='Number of forward trajectories of the closed-loop system at each step. Default is 100.')
    parser.add_argument('--epochs', type=int, default=400, help='Total number of epochs for training. Default is 5000 if collision avoidance, else 100.')
    parser.add_argument('--lr', type=float, default=-1, help='Learning rate. Default is 2e-3 if collision avoidance, else 5e-3.')
    parser.add_argument('--log-epoch', type=int, default=-1, help='Frequency of logging in epochs. Default is 0.1 * epochs.')
    parser.add_argument('--return-best', type=bool, default=True, help='Return the best model on the validation data among all logged iterations. The train data can be used instead of validation data. The Default is True.')

    # Obstacles
    parser.add_argument('--obstacle-shape', type=int, default=7, help='Shape of the obstacles. Default is 6 -> Mountain range.')
    parser.add_argument('--obstacle-centers', type=float, default=None, help='Centers of the obstacles. Default is None.')
    parser.add_argument('--obstacle-covs', type=float, default=None, help='Covariances of the obstacles. Default is None.')
    
    # Record
    # Add a parder argument for which data to record it must be a string of either 'dim' or 'rollouts', default is 'none'
    parser.add_argument('--record', type=str, default='none', help='Record the data for the sensitivity study. Default is none.')
    parser.add_argument('--filename', type=str, default='sensitivity_study', help='Filename for the recorded data. Default is sensitivity_study.')
    parser.add_argument('--no-save-plot', action='store_true', help='Disable saving plots. Default is False (plots are saved).')  
      
    # Roudtrip
    parser.add_argument('--rt-epochs', type=float, default=0.0, help='Percentage of epochs to be used for learning to come back to x0, Default is 0.0.')
    parser.add_argument('--use-previous-params', type=bool, default=False, help='Use the best ctrl param of the previous ctrl. Default is False.')

    # Load model
    parser.add_argument('--model-path', type=str, default=None, help='Path to the saved model file')

    # Save results path
    parser.add_argument('--save-path', type=str, default='saved_results_rt', help='Path to save the results. Default is "saved_results".')

    # Training range x_interval tuple
    parser.add_argument('--x-interval', type=ast.literal_eval, default=(-3, 5), help='Interval for x. Default is (-3, 7).')
    parser.add_argument('--y-interval', type=ast.literal_eval, default=(4, 4.1), help='Interval for y. Default is (4, 4.1).')

    # TODO: add the following
    # parser.add_argument('--patience-epoch', type=int, default=None, help='Patience epochs for no progress. Default is None which sets it to 0.2 * total_epochs.')
    # parser.add_argument('--lr-start-factor', type=float, default=1.0, help='Start factor of the linear learning rate scheduler. Default is 1.0.')
    # parser.add_argument('--lr-end-factor', type=float, default=0.01, help='End factor of the linear learning rate scheduler. Default is 0.01.')
    # # save/load args
    # parser.add_argument('--experiment-dir', type=str, default='boards', help='Name tag for the experiments. By default it will be the "boards" folder.')
    # parser.add_argument('--load-model', type=str, default=None, help='If it is not set to None, a pretrained model will be loaded instead of training.')
    # parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to run the computations on, "cpu" or "cuda:0". Default is "cuda:0" if available, otherwise "cpu".')

    args = parser.parse_args()

    # set default values that depend on other args
    if args.batch_size == -1:
        args.batch_size = args.num_rollouts  # use all train data

    if args.epochs == -1 or args.epochs is None:
        args.epochs = 1000 if args.col_av else 50

    if args.lr == -1 or args.lr is None:
        args.lr = 2e-3 if args.col_av else 5e-3

    if args.log_epoch == -1 or args.log_epoch is None:
        args.log_epoch = math.ceil(float(args.epochs)/10)

    # assertions and warning
    if not args.col_av:
        args.alpha_col = None
        args.min_dist = None
    if not args.obst_av:
        args.alpha_obst = None

    # if args.total_epochs < 10000:
    #     print(f'Minimum of 10000 epochs are required for proper training')

    if args.horizon > 100:
        print(f'Long horizons may be unnecessary and pose significant computation')

    # # Obstacles 
    # if args.obstacle_shape == 0:
    #     args.obstacle_centers = [
    #                     torch.tensor([[-0.5, 0]], device=device),
    #                     torch.tensor([[0.5, 0.0]], device=device),
    #                 ]
    #     args.obstacle_covs = [
    #         torch.tensor([[0.1, 0.1]], device=device)
    #     ] * len(args.obstacle_centers)
    # elif args.obstacle_shape == 1:
    #     args.obstacle_centers = [
    #                     torch.tensor([[0.5, 2]], device=device),
    #                     torch.tensor([[1, 2.0]], device=device),
    #                     torch.tensor([[3, 2]], device=device),
    #                     torch.tensor([[3.5, 2.0]], device=device),

    #                 ]
    #     args.obstacle_covs = [
    #         torch.tensor([[0.1, 0.1]], device=device)
    #     ] * len(args.obstacle_centers)
    # elif args.obstacle_shape == 2:
    #     args.obstacle_centers = [
    #     # Upper wall (y = 2.5)
    #     torch.tensor([[-3, -1]], device=device),
    #     torch.tensor([[-2, -0.6]], device=device),
    #     torch.tensor([[-1, 0]], device=device),
    #     torch.tensor([[-2, 0.6]], device=device),
    #     torch.tensor([[-3, 1]], device=device),
    #     torch.tensor([[3, -1]], device=device),
    #     torch.tensor([[2, -0.6]], device=device),
    #     torch.tensor([[1, 0]], device=device),
    #     torch.tensor([[2, 0.6]], device=device),
    #     torch.tensor([[3, 1]], device=device),
    #     ]

    #     # Set the same covariance for each obstacle (small radius)
    #     args.obstacle_covs = [
    #         torch.tensor([[0.1, 0.1]], device=device)
    #     ] * len(args.obstacle_centers)
    # elif args.obstacle_shape == 3:
    #     args.obstacle_centers = [
    #         # Left rectangle (from x = -3 to x = -1, y = -1 to y = 1)
    #         torch.tensor([[-3.0, -1.0]], device=device),
    #         torch.tensor([[-2.5, -1.0]], device=device),
    #         torch.tensor([[-2.0, -1.0]], device=device),
    #         torch.tensor([[-1.5, -1.0]], device=device),
    #         torch.tensor([[-1.0, -1.0]], device=device),

    #         torch.tensor([[-3.0, 1.0]], device=device),
    #         torch.tensor([[-2.5, 1.0]], device=device),
    #         torch.tensor([[-2.0, 1.0]], device=device),
    #         torch.tensor([[-1.5, 1.0]], device=device),
    #         torch.tensor([[-1.0, 1.0]], device=device),

    #         torch.tensor([[-3.0, -0.5]], device=device),
    #         torch.tensor([[-3.0, 0.0]], device=device),
    #         torch.tensor([[-3.0, 0.5]], device=device),

    #         torch.tensor([[-1.0, -0.5]], device=device),
    #         torch.tensor([[-1.0, 0.0]], device=device),
    #         torch.tensor([[-1.0, 0.5]], device=device),

    #         # Right rectangle (from x = 1 to x = 3, y = -1 to y = 1)
    #         torch.tensor([[1.0, -1.0]], device=device),
    #         torch.tensor([[1.5, -1.0]], device=device),
    #         torch.tensor([[2.0, -1.0]], device=device),
    #         torch.tensor([[2.5, -1.0]], device=device),
    #         torch.tensor([[3.0, -1.0]], device=device),

    #         torch.tensor([[1.0, 1.0]], device=device),
    #         torch.tensor([[1.5, 1.0]], device=device),
    #         torch.tensor([[2.0, 1.0]], device=device),
    #         torch.tensor([[2.5, 1.0]], device=device),
    #         torch.tensor([[3.0, 1.0]], device=device),

    #         torch.tensor([[1.0, -0.5]], device=device),
    #         torch.tensor([[1.0, 0.0]], device=device),
    #         torch.tensor([[1.0, 0.5]], device=device),

    #         torch.tensor([[3.0, -0.5]], device=device),
    #         torch.tensor([[3.0, 0.0]], device=device),
    #         torch.tensor([[3.0, 0.5]], device=device),
    #     ]

    #     # Smaller covariance for precise obstacles
    #     args.obstacle_covs = [
    #         torch.tensor([[0.05, 0.05]], device=device)
    #     ] * len(args.obstacle_centers)
    # Obstacles 
    if args.obstacle_shape in obstacles:
        args.obstacle_centers = obstacles[args.obstacle_shape]["centers"]
        args.obstacle_covs = obstacles[args.obstacle_shape]["covs"] * len(args.obstacle_centers)
    else:
        raise ValueError(f"Invalid obstacle shape: {args.obstacle_shape}")

    return args

def print_args(args):
    msg = '\n[INFO] Dataset: n_agents: %i' % args.n_agents + ' -- num_rollouts: %i' % args.num_rollouts
    msg += ' -- std_ini: %.2f' % args.std_init_plant + ' -- time horizon: %i' % args.horizon

    msg += '\n[INFO] Plant: spring constant: %.2f' % args.spring_const + ' -- use linearized plant: ' + str(args.linearize_plant)

    msg += '\n[INFO] Controller: dimension of the internal state: %i' % args.dim_internal
    msg += ' -- dim_nl: %i' % args.dim_nl + ' -- cont_init_std: %.2f'% args.cont_init_std

    msg += '\n[INFO] Loss:  alpha_u: %.6f' % args.alpha_u
    msg += ' -- alpha_col: %.f' % args.alpha_col if args.col_av else ' -- no collision avoidance'
    msg += ' -- alpha_obst: %.1f' % args.alpha_obst if args.obst_av else ' -- no obstacle avoidance'

    msg += '\n[INFO] Optimizer: lr: %.2e' % args.lr
    msg += ' -- batch_size: %i' % args.batch_size + ', -- return best model for validation data among logged epochs:' + str(args.return_best)

    return msg
