import argparse, math


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Networked robots - base control rollout / networked PB controller training."
    )

    parser.add_argument('--random-seed', type=int, default=5, help='Random seed. Default is 5.')

    # dataset
    parser.add_argument('--n-agents', type=int, default=4, help='Number of agents in the network. Default is 4.')
    parser.add_argument('--horizon', type=int, default=100, help='Time horizon for the rollout. Default is 100.')
    parser.add_argument('--num-rollouts', type=int, default=10, help='Number of rollouts in the training set. Default is 10.')
    parser.add_argument('--num-test-samples', type=int, default=50, help='Number of rollouts in the held-out test set. Default is 50.')
    parser.add_argument('--std-init-plant', type=float, default=0.2, help='std of the plant initial conditions. Default is 0.2.')
    parser.add_argument('--min-dist', type=float, default=2.0, help='Minimum pairwise distance enforced when sampling initial positions and targets. Default is 2.0.')
    parser.add_argument('--central-data', action='store_true',
                        help='Sample initial conditions and targets from the same distributions as the '
                             'CENTRALIZED experiment (experiments/robots/run.py) instead of this script\'s, so '
                             'the two runs are comparable. Nominal formation becomes the centralized fixed '
                             'x0=[4,0,0,0, 0,0,0,0] (still perturbed per rollout by --std-init-plant), and '
                             'targets are drawn in x=[-1,5] / y=[4,4.1] with min pairwise distance 2.0; '
                             '--min-dist and the x0/target intervals are ignored. Requires --n-agents 2. '
                             'NOTE: this script\'s loss (tracking + energy) is not the centralized one - for a '
                             'loss-comparable run use run_networked_control_corridor.py.')

    # plant
    parser.add_argument('--spring-const', type=float, default=1.0, help='Spring constant of the pre-stabilizing base loop. Default is 1.0.')
    parser.add_argument('--linearize-plant', type=bool, default=False, help='Linearize plant or not. Default is False.')

    # controller (only used by run_networked_control.py)
    parser.add_argument('--dim-internal', type=int, default=8, help='Dimension of the internal state of each agent\'s REN. Default is 8.')
    parser.add_argument('--dim-nl', type=int, default=8, help='Size of the non-linear part of each agent\'s REN. Default is 8.')
    parser.add_argument('--cont-init-std', type=float, default=0.1, help='Initialization std for controller params. Default is 0.1.')
    parser.add_argument('--window-fraction', type=float, default=0.75, help='Fraction of the horizon during which the windowed-context REN input stays nonzero. Only used by run_networked_control_windowed.py. Default is 0.75.')

    # training (only used by run_networked_control.py)
    parser.add_argument('--epochs', type=int, default=200, help='Total number of training epochs. Default is 200.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate. Default is 2e-3.')
    parser.add_argument('--alpha-tracking', type=float, default=1.0, help='Weight of the tracking loss term. Default is 1.0.')
    parser.add_argument('--alpha-energy', type=float, default=0.0, help='Weight of the energy regularizer on the PB output (dxref). Default is 0.0 (off).')
    parser.add_argument('--batch-size', type=int, default=-1, help='Batch size for training. Default is -1, which uses all of num-rollouts as one batch.')
    parser.add_argument('--log-epoch', type=int, default=-1, help='Frequency of logging in epochs. Default is -1, which sets it to ceil(epochs/10).')
    parser.add_argument('--return-best', type=bool, default=True, help='Return the best model on the validation data among all logged epochs. Default is True.')

    # output
    parser.add_argument('--plot-samples', type=int, default=1, help='Number of rollouts to plot. Keep small when num-rollouts is large. Default is 1.')

    args = parser.parse_args()

    if args.batch_size == -1:
        args.batch_size = args.num_rollouts

    if args.log_epoch == -1:
        args.log_epoch = math.ceil(args.epochs / 10)

    return args


def print_args(args):
    msg = '\n[INFO] Dataset: n_agents: %i' % args.n_agents + ' -- num_rollouts: %i' % args.num_rollouts
    msg += ' -- num_test_samples: %i' % args.num_test_samples
    msg += ' -- std_ini: %.2f' % args.std_init_plant + ' -- time horizon: %i' % args.horizon
    msg += ' -- min_dist: %.2f' % args.min_dist
    if args.central_data:
        msg += ' -- CENTRALIZED x0/target distributions (min_dist / intervals ignored)'

    msg += '\n[INFO] Plant: spring constant: %.2f' % args.spring_const + ' -- use linearized plant: ' + str(args.linearize_plant)

    msg += '\n[INFO] Controller: dim_internal: %i' % args.dim_internal
    msg += ' -- dim_nl: %i' % args.dim_nl + ' -- cont_init_std: %.2f' % args.cont_init_std

    msg += '\n[INFO] Training: epochs: %i' % args.epochs + ' -- lr: %.2e' % args.lr
    msg += ' -- batch_size: %i' % args.batch_size + ' -- return_best: ' + str(args.return_best)
    msg += ' -- alpha_tracking: %.4g' % args.alpha_tracking + ' -- alpha_energy: %.4g' % args.alpha_energy

    return msg
