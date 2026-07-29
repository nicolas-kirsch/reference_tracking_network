import argparse, math


def argument_parser():
    """
    Self-contained argument parser for run_networked_control_corridor.py.
    Same dataset/plant/controller flags as networked_robots/arg_parser.py,
    but the loss section is replaced with the corridor-experiment flags
    (alpha_track/alpha_speed/alpha_u/alpha_col/alpha_obst/col_av/obst_av/
    col_min_dist) consumed by CorridorRobotsLoss - kept as its own module
    rather than added to the shared arg_parser.py, so run_networked_control.py
    / run_networked_control_windowed.py keep exactly the flags they had
    before (alpha_tracking/alpha_energy, no collision/obstacle knobs).
    """
    parser = argparse.ArgumentParser(
        description="Networked robots - corridor experiment (single-pass crossing with obstacle wall + collision avoidance)."
    )

    parser.add_argument('--random-seed', type=int, default=5, help='Random seed. Default is 5.')

    # dataset
    parser.add_argument('--n-agents', type=int, default=2, help='Number of agents in the network. Default is 2.')
    parser.add_argument('--horizon', type=int, default=100, help='Time horizon for the rollout. Default is 100.')
    parser.add_argument('--num-rollouts', type=int, default=30, help='Number of rollouts in the training set. Default is 30.')
    parser.add_argument('--num-test-samples', type=int, default=50, help='Number of rollouts in the held-out test set. Default is 50.')
    parser.add_argument('--std-init-plant', type=float, default=0.2, help='std of the plant initial conditions. Default is 0.2.')
    parser.add_argument('--min-dist', type=float, default=2.0, help='Minimum pairwise distance enforced when sampling initial positions and targets. Default is 2.0.')

    # plant
    parser.add_argument('--spring-const', type=float, default=1.0, help='Spring constant of the pre-stabilizing base loop. Default is 1.0.')
    parser.add_argument('--linearize-plant', type=bool, default=False, help='Linearize plant or not. Default is False.')

    # controller
    parser.add_argument('--dim-internal', type=int, default=8, help='Dimension of the internal state of each agent\'s REN. Default is 8.')
    parser.add_argument('--dim-nl', type=int, default=8, help='Size of the non-linear part of each agent\'s REN. Default is 8.')
    parser.add_argument('--cont-init-std', type=float, default=0.1, help='Initialization std for controller params. Default is 0.1.')
    parser.add_argument('--central-like', type=bool, default=False, help='Imitate the centralized controller as closely as possible: every agent\'s REN/MLP sees every agent\'s w_hat (and, for the MLP, every agent\'s xbar) instead of just its own + own_position + neighbors\' delayed actions - no u/z interconnection between agents. Default is False (normal networked behavior).')

    # loss
    parser.add_argument('--col-av', type=bool, default=True, help='Avoid collisions between agents. Default is True.')
    parser.add_argument('--obst-av', type=bool, default=True, help='Avoid obstacles. Default is True.')
    parser.add_argument('--alpha-track', type=float, default=100.0, help='Weight of the tracking loss term (expands to alpha_track * kron(eye(n_agents), eye(2))). Default is 100.0.')
    parser.add_argument('--alpha-speed', type=float, default=1.0, help='Weight of the speed penalty term (expands to alpha_speed * kron(eye(n_agents), eye(2))). Default is 1.0.')
    parser.add_argument('--alpha-u', type=float, default=0.1 / 400, help='Weight of the loss due to control input "u". Default is 0.1/400.')  # 400 is output_amplification^2
    parser.add_argument('--alpha-col', type=float, default=100, help='Weight of the collision avoidance loss. Default is 100 if "col-av" is True, else None.')
    parser.add_argument('--alpha-obst', type=float, default=5e3, help='Weight of the obstacle avoidance loss. Default is 5e3 if "obst-av" is True, else None.')
    parser.add_argument('--col-min-dist', type=float, default=1.0, help='Distance below which the collision-avoidance loss penalizes two agents (separate from --min-dist, which only constrains data sampling). Default is 1.0 if "col-av" is True, else None.')

    # training
    parser.add_argument('--epochs', type=int, default=200, help='Total number of training epochs. Default is 200.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate. Default is 2e-3.')
    parser.add_argument('--batch-size', type=int, default=-1, help='Batch size for training. Default is -1, which uses all of num-rollouts as one batch.')
    parser.add_argument('--log-epoch', type=int, default=-1, help='Frequency of logging in epochs. Default is -1, which sets it to ceil(epochs/10).')
    parser.add_argument('--return-best', type=bool, default=True, help='Return the best model on the validation data among all logged epochs. Default is True.')

    args = parser.parse_args()

    if args.batch_size == -1:
        args.batch_size = args.num_rollouts

    if args.log_epoch == -1:
        args.log_epoch = math.ceil(args.epochs / 10)

    if not args.col_av:
        args.alpha_col = None
        args.col_min_dist = None
    if not args.obst_av:
        args.alpha_obst = None

    return args


def print_args(args):
    msg = '\n[INFO] Dataset: n_agents: %i' % args.n_agents + ' -- num_rollouts: %i' % args.num_rollouts
    msg += ' -- num_test_samples: %i' % args.num_test_samples
    msg += ' -- std_ini: %.2f' % args.std_init_plant + ' -- time horizon: %i' % args.horizon
    msg += ' -- min_dist: %.2f' % args.min_dist

    msg += '\n[INFO] Plant: spring constant: %.2f' % args.spring_const + ' -- use linearized plant: ' + str(args.linearize_plant)

    msg += '\n[INFO] Controller: dim_internal: %i' % args.dim_internal
    msg += ' -- dim_nl: %i' % args.dim_nl + ' -- cont_init_std: %.2f' % args.cont_init_std
    msg += ' -- central_like: ' + str(args.central_like)

    msg += '\n[INFO] Loss: alpha_track: %.4g' % args.alpha_track + ' -- alpha_speed: %.4g' % args.alpha_speed
    msg += ' -- alpha_u: %.6f' % args.alpha_u
    msg += ' -- alpha_col: %.4g' % args.alpha_col if args.col_av else ' -- no collision avoidance'
    msg += ' -- alpha_obst: %.4g' % args.alpha_obst if args.obst_av else ' -- no obstacle avoidance'

    msg += '\n[INFO] Training: epochs: %i' % args.epochs + ' -- lr: %.2e' % args.lr
    msg += ' -- batch_size: %i' % args.batch_size + ' -- return_best: ' + str(args.return_best)

    return msg
