import sys, os, logging, copy, time, torch
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)

from config import device
from networked_robots.corridor_arg_parser import argument_parser, print_args
from networked_robots.robots_dataset import NetworkedRobotsDataset
from networked_robots.robots_network import RobotsNetwork
from networked_robots.pb_controller_network import PBControllerNetwork
from networked_robots.corridor_robots_loss import CorridorRobotsLoss
from networked_robots.run_networked_control import build_complete_graph_adjacency, xbar_to_flat, format_loss_components
from utils.plot_functions import plot_trajectories, plot_input_norm
from utils.assistive_functions import WrapLogger

"""
Corridor experiment: both robots must cross an obstacle wall in a single
pass to reach a target on the other side, generalizing
experiments/robots/run.py's centralized corridor scenario to the networked
(per-agent factorized) controller. Kept as its own run file - together with
corridor_arg_parser.py and corridor_robots_loss.py - so run_networked_control.py
and run_networked_control_windowed.py keep doing exactly what they did
before (plain tracking + energy loss, no obstacles/collisions).
"""


def build_corridor_obstacles():
    """
    Gaussian-bump obstacle wall at y=2 with a gap around x=0, matching the
    corridor scenario in experiments/robots/run.py (the centralized code's
    actual scenario, not RobotsLoss's generic defaults).
    """
    obstacle_centers = [
        torch.tensor([[-1, 2]], device=device),
        torch.tensor([[1, 2.0]], device=device),
        torch.tensor([[3, 2]], device=device),
        torch.tensor([[5, 2.0]], device=device),
        torch.tensor([[7, 2.0]], device=device),
        torch.tensor([[-3, 2.0]], device=device),
    ]
    obstacle_covs = [torch.tensor([[0.05, 0.05]], device=device)] * len(obstacle_centers)
    return obstacle_centers, obstacle_covs


def build_verification_data(x0, target_y, n_agents, horizon):
    """
    Two deterministic reference scenarios sharing the same nominal
    (noiseless) initial formation x0:
      - 'aligned': every agent's target sits directly above its own start
        (straight crossing, no interaction needed to avoid each other's path).
      - 'crossed': targets are swapped end-to-end (agent i's target is where
        agent n_agents-1-i started), so every agent's path crosses another's
        while passing through the corridor gap.

    Returns:
        - data: tensor (2, horizon, 2*state_dim), row 0 = aligned, row 1 = crossed,
          in the same layout NetworkedRobotsDataset produces (first state_dim
          columns = initial condition at t=0, next state_dim columns =
          reference block, x,y sub-columns per agent held constant from t=1).
        - aligned, crossed: compact target tensors (2*n_agents,), for plotting.
    """
    state_dim = 4 * n_agents
    data = torch.zeros(2, horizon, 2 * state_dim)
    data[:, 0, :state_dim] = x0

    aligned = torch.zeros(2 * n_agents)
    crossed = torch.zeros(2 * n_agents)
    for i in range(n_agents):
        aligned[2 * i:2 * i + 2] = torch.tensor([x0[4 * i].item(), target_y])
        j = n_agents - 1 - i
        crossed[2 * i:2 * i + 2] = torch.tensor([x0[4 * j].item(), target_y])

    for i in range(n_agents):
        data[0, 1:, state_dim + 4 * i: state_dim + 4 * i + 2] = aligned[2 * i:2 * i + 2]
        data[1, 1:, state_dim + 4 * i: state_dim + 4 * i + 2] = crossed[2 * i:2 * i + 2]

    return data, aligned, crossed


def plot_verification_scenarios(net, controller, x0, target_y, args, save_folder, tag, text_prefix,
                                 obstacle_centers=None, obstacle_covs=None):
    """
    Roll out and plot the two canonical corridor scenarios (see
    build_verification_data) under `controller` (and, for comparison, under
    the base loop alone). Returns a dict of mean final-timestep tracking
    error per scenario.
    """
    data, aligned, crossed = build_verification_data(x0, target_y, args.n_agents, args.horizon)
    data = data.to(device)

    x_log, e_log, u_log = net.rollout(data, controller=controller)
    x_log_base, _, _ = net.rollout(data)

    errors = {}
    for idx, (name, xbar_compact) in enumerate([('aligned', aligned), ('crossed', crossed)]):
        plot_trajectories(
            x_log[idx, :, :],
            xbar=xbar_to_flat(xbar_compact, args.n_agents),
            n_agents=args.n_agents,
            save_folder=save_folder,
            filename=f'{tag}_{name}.pdf',
            text=f'{text_prefix} - {name}',
            T=args.horizon,
            x_compare=x_log_base[idx, :, :],
            compare_label='base control only (dxref=0)',
            obstacle_centers=obstacle_centers,
            obstacle_covs=obstacle_covs,
        )
        plot_input_norm(
            u_log[idx, :, :],
            n_agents=args.n_agents,
            save_folder=save_folder,
            filename=f'{tag}_{name}_dxref_norm.pdf',
            text=f'Reference offset $\\delta x_{{ref}}$ norm per robot ({tag} - {name})',
            T=args.horizon,
        )
        errors[name] = e_log[idx, -1, :].abs().mean().item()

    return errors


def main():
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = os.path.join(BASE_DIR, 'networked_robots', 'saved_results')
    save_folder = os.path.join(save_path, 'networked_control_corridor_' + now)
    os.makedirs(save_folder, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_folder, 'log'),
        format='%(asctime)s %(message)s',
        filemode='w',
    )
    logger = logging.getLogger('networked_robots_networked_control_corridor')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)

    args = argument_parser()
    logger.info(print_args(args))
    torch.manual_seed(args.random_seed)

    # ------------ 1. Communication graph ------------
    adjacency = build_complete_graph_adjacency(args.n_agents)

    # ------------ 2. Dataset ------------
    # corridor scenario: agents start below the obstacle wall (y~0) and are
    # given targets above it (y~4), same single-pass crossing as the
    # centralized experiments/robots/run.py scenario (x0=[...,0,...],
    # targets sampled with y in [4, 4.1]), generalized to N agents.
    target_y = 4.0
    obstacle_centers, obstacle_covs = build_corridor_obstacles()
    dataset = NetworkedRobotsDataset(
        random_seed=args.random_seed,
        horizon=args.horizon,
        n_agents=args.n_agents,
        std_ini=args.std_init_plant,
        min_dist=args.min_dist,
        x0_interval_x1=(-1, 5), x0_interval_x2=(-0.2, 0.2),
        target_interval_x1=(-1, 5), target_interval_x2=(target_y - 0.2, target_y + 0.2),
    )
    train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=args.num_test_samples)
    train_data, test_data = train_data.to(device), test_data.to(device)

    # ------------ 3. Plant (network of decoupled per-agent dynamics) ------------
    net = RobotsNetwork(
        n_agents=args.n_agents,
        adjacency=adjacency,
        linear_plant=args.linearize_plant,
        k=args.spring_const,
    ).to(device)

    # ------------ 4. Networked PB controller (random init before training) ------------
    pb_net = PBControllerNetwork(
        net,
        dim_internal=args.dim_internal,
        dim_nl=args.dim_nl,
        initialization_std=args.cont_init_std,
        central_like=args.central_like,
    ).to(device)
    total_params = sum(p.numel() for p in pb_net.parameters())
    logger.info(f'Number of controller parameters: {total_params}')

    # ------------ 4b. Loss ------------
    loss_fn = CorridorRobotsLoss(
        n_agents=args.n_agents,
        alpha_track=args.alpha_track,
        alpha_speed=args.alpha_speed,
        alpha_u=args.alpha_u,
        alpha_col=args.alpha_col,
        alpha_obst=args.alpha_obst,
        min_dist=args.col_min_dist,
        obstacle_centers=obstacle_centers,
        obstacle_covs=obstacle_covs,
    )

    # ------------ 5. Plot before training ------------
    logger.info('Rolling out the network under the UNTRAINED networked PB controller...')
    err_before = plot_verification_scenarios(
        net, pb_net, dataset.x0, target_y, args, save_folder,
        tag='before_training', text_prefix='Networked PB controller (untrained)',
        obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs,
    )
    logger.info(f'[before training] mean final-timestep tracking error: aligned={err_before["aligned"]:.4f}, crossed={err_before["crossed"]:.4f}')

    # ------------ 6. Training ------------
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(pb_net.parameters(), lr=args.lr)
    valid_data = train_data     # use the entire train data for validation, matching the centralized script

    best_valid_loss = float('inf')
    best_state_dict = None

    logger.info('\n------------ Begin training ------------')
    t = time.time()
    for epoch in range(1 + args.epochs):
        for train_data_batch in train_dataloader:
            optimizer.zero_grad()
            x_log, e_log, u_log = net.rollout(train_data_batch, controller=pb_net, train=True)
            loss = loss_fn.forward(x_log, u_log, e_log)
            loss.backward()
            optimizer.step()

        if epoch % args.log_epoch == 0:
            msg = 'Epoch: %i --- train loss: %.4f' % (epoch, loss.item())

            if args.return_best:
                with torch.no_grad():
                    x_log_valid, e_log_valid, u_log_valid = net.rollout(valid_data, controller=pb_net, train=False)
                    loss_valid_parts = loss_fn.components(x_log_valid, u_log_valid, e_log_valid)
                    loss_valid = loss_valid_parts['total']
                msg += ' ---||--- validation loss: %.4f (%s)' % (loss_valid.item(), format_loss_components(loss_valid_parts))
                if loss_valid.item() < best_valid_loss:
                    best_valid_loss = loss_valid.item()
                    best_state_dict = copy.deepcopy(pb_net.state_dict())
                    msg += ' (best so far)'
            duration = time.time() - t
            msg += ' ---||--- time: %.0f s' % duration
            logger.info(msg)
            t = time.time()

    if args.return_best and best_state_dict is not None:
        pb_net.load_state_dict(best_state_dict)

    # ------------ 7. Save trained controller ------------
    checkpoint_path = os.path.join(save_folder, 'trained_controller.pt')
    torch.save(pb_net.state_dict(), checkpoint_path)
    logger.info(f'[INFO] saved trained controller to {checkpoint_path}')

    # ------------ 8. Test-time deployment: reload the saved checkpoint from disk ------------
    # test_data was sampled independently from train_data (fresh initial positions
    # and targets, drawn separately in NetworkedRobotsDataset._generate_data) - this
    # is a genuine deployment check, not a reuse of the in-memory trained object.
    pb_net_deployed = PBControllerNetwork(
        net, dim_internal=args.dim_internal, dim_nl=args.dim_nl,
        initialization_std=args.cont_init_std,
        central_like=args.central_like,
    ).to(device)
    pb_net_deployed.load_state_dict(torch.load(checkpoint_path, map_location=device))

    with torch.no_grad():
        x_log_test, e_log_test, u_log_test = net.rollout(test_data, controller=pb_net_deployed, train=False)
        test_loss = loss_fn.forward(x_log_test, u_log_test, e_log_test).item()
    logger.info(f'Test loss of the reloaded checkpoint (over {test_data.shape[0]} fresh test rollouts): {test_loss:.4f}')
    if args.col_av:
        num_col = loss_fn.count_collisions(x_log_test)
        logger.info(f'Number of collisions on the test set: {num_col:.0f}')

    # ------------ 9. Plot the reloaded checkpoint on the two canonical verification scenarios ------------
    err_test = plot_verification_scenarios(
        net, pb_net_deployed, dataset.x0, target_y, args, save_folder,
        tag='test_deployment', text_prefix='Networked PB controller (trained, reloaded from checkpoint)',
        obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs,
    )
    logger.info(f'[test deployment] mean final-timestep tracking error: aligned={err_test["aligned"]:.4f}, crossed={err_test["crossed"]:.4f}')
    logger.info(f'Saved plots to {save_folder}')


if __name__ == "__main__":
    main()
