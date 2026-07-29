import sys, os, logging, torch
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)

from config import device
from networked_robots.arg_parser import argument_parser, print_args
from networked_robots.robots_dataset import NetworkedRobotsDataset
from networked_robots.robots_network import RobotsNetwork
from utils.plot_functions import plot_trajectories
from utils.assistive_functions import WrapLogger


def build_complete_graph_adjacency(n_agents):
    """
    Explicit adjacency matrix (every agent connected to every other), passed
    directly to RobotsNetwork. Not used by the dynamics yet - reserved for the
    networked controller added in a later pass. No CLI topology flag for this
    pass, per design: the caller constructs the matrix explicitly.
    """
    return torch.ones(n_agents, n_agents) - torch.eye(n_agents)


def xbar_to_flat(xbar_compact, n_agents):
    """plot_trajectories expects xbar indexed the same way as x (stride 4 per agent, only x,y populated)."""
    xbar_flat = torch.zeros(4 * n_agents)
    for i in range(n_agents):
        xbar_flat[4 * i:4 * i + 2] = xbar_compact[2 * i:2 * i + 2]
    return xbar_flat


def main():
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = os.path.join(BASE_DIR, 'networked_robots', 'saved_results')
    save_folder = os.path.join(save_path, 'base_control_' + now)
    os.makedirs(save_folder, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_folder, 'log'),
        format='%(asctime)s %(message)s',
        filemode='w',
    )
    logger = logging.getLogger('networked_robots_base_control')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)

    args = argument_parser()
    logger.info(print_args(args))
    torch.manual_seed(args.random_seed)

    # ------------ 1. Communication graph (explicit adjacency, unused by the dynamics for now) ------------
    adjacency = build_complete_graph_adjacency(args.n_agents)

    # ------------ 2. Dataset ------------
    dataset = NetworkedRobotsDataset(
        random_seed=args.random_seed,
        horizon=args.horizon,
        n_agents=args.n_agents,
        std_ini=args.std_init_plant,
        min_dist=args.min_dist,
    )
    data, _ = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=1)
    data = data.to(device)

    # ------------ 3. Plant (network of decoupled per-agent dynamics) ------------
    net = RobotsNetwork(
        n_agents=args.n_agents,
        adjacency=adjacency,
        linear_plant=args.linearize_plant,
        k=args.spring_const,
    ).to(device)

    # ------------ 4. Rollout under the pre-stabilizing base loop alone ------------
    logger.info('Rolling out the network under the pre-stabilizing base loop (no learned controller)...')
    x_log, e_log, u_log = net.rollout(data)

    xbar_cols = [4 * i + j for i in range(args.n_agents) for j in (0, 1)]
    xbar_full = data[:, :, net.state_dim:2 * net.state_dim][:, :, xbar_cols]

    # ------------ 5. Plot ------------
    num_plots = min(args.plot_samples, x_log.shape[0])
    for r in range(num_plots):
        plot_trajectories(
            x_log[r, :, :],
            xbar=xbar_to_flat(xbar_full[r, -1, :], args.n_agents),
            n_agents=args.n_agents,
            save_folder=save_folder,
            filename=f'base_control_rollout_{r}.pdf',
            text=f'Networked base control - rollout {r}',
            T=args.horizon,
        )

    final_tracking_error = e_log[:, -1, :].abs().mean().item()
    logger.info(f'Mean |tracking error| at final time step (over {data.shape[0]} rollouts): {final_tracking_error:.4f}')
    logger.info(f'Saved plots to {save_folder}')


if __name__ == "__main__":
    main()
