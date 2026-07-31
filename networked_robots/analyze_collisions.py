import sys, os, re, json, argparse
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)

from config import device
from networked_robots.robots_dataset import NetworkedRobotsDataset
from networked_robots.robots_network import RobotsNetwork
from networked_robots.pb_controller_network import PBControllerNetwork
from networked_robots.corridor_robots_loss import CorridorRobotsLoss
from networked_robots.run_networked_control_corridor import build_complete_graph_adjacency, build_corridor_obstacles
from networked_robots.run_networked_control import xbar_to_flat
from utils.plot_functions import plot_trajectories

"""
Post-hoc collision analysis for a trained run_networked_control_corridor.py
checkpoint: deploys it on fresh test rollouts and reports, per trajectory
(not per timestep): whether it ever collides, how long each collision
episode lasts, and whether episodes fall in the first 25% / middle 50% /
last 25% of the horizon. Reconstructs the run's exact architecture/dataset
by parsing the sibling `log` file (falls back to CLI overrides / corridor
defaults for anything not logged, e.g. random_seed and col_min_dist).
"""

TARGET_Y = 4.0    # matches run_networked_control_corridor.py's corridor geometry
X0_INTERVAL_X1, X0_INTERVAL_X2 = (-1, 5), (-0.2, 0.2)
TARGET_INTERVAL_X1, TARGET_INTERVAL_X2 = (-1, 5), (TARGET_Y - 0.2, TARGET_Y + 0.2)


def parse_run_log(log_path):
    text = open(log_path).read()

    def grab(pattern, cast, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    return dict(
        n_agents=grab(r'n_agents:\s*(\d+)', int),
        horizon=grab(r'time horizon:\s*(\d+)', int),
        std_ini=grab(r'std_ini:\s*([\d.]+)', float),
        spring_const=grab(r'spring constant:\s*([\d.]+)', float),
        linearize_plant=grab(r'use linearized plant:\s*(True|False)', lambda s: s == 'True'),
        dim_internal=grab(r'dim_internal:\s*(\d+)', int),
        dim_nl=grab(r'dim_nl:\s*(\d+)', int),
        mode=parse_mode(text),
        position_scale=grab(r'position_scale:\s*([\d.]+)', float, default=5.0),
    )


def parse_mode(text):
    """
    Current logs write 'controller_mode: <base|no_interconnection|central_like>'.
    Older logs (pre mode-selector) wrote 'central_like: <True|False>' - fall
    back to that and map True -> 'central_like', False -> 'base', so old
    checkpoints still reconstruct correctly.
    """
    m = re.search(r'controller_mode:\s*(\w+)', text)
    if m:
        return m.group(1)
    m = re.search(r'central_like:\s*(True|False)', text)
    if m:
        return 'central_like' if m.group(1) == 'True' else 'base'
    return 'base'


def rebuild_dataset(cfg, random_seed):
    return NetworkedRobotsDataset(
        random_seed=random_seed,
        horizon=cfg['horizon'],
        n_agents=cfg['n_agents'],
        std_ini=cfg['std_ini'],
        min_dist=2.0,    # dataset-sampling spacing - run_networked_control_corridor.py's fixed value, not logged
        x0_interval_x1=X0_INTERVAL_X1, x0_interval_x2=X0_INTERVAL_X2,
        target_interval_x1=TARGET_INTERVAL_X1, target_interval_x2=TARGET_INTERVAL_X2,
    )


def rebuild_controller(cfg, checkpoint_path):
    adjacency = build_complete_graph_adjacency(cfg['n_agents'])
    net = RobotsNetwork(
        n_agents=cfg['n_agents'], adjacency=adjacency,
        linear_plant=cfg['linearize_plant'], k=cfg['spring_const'],
    ).to(device)
    pb_net = PBControllerNetwork(
        net, dim_internal=cfg['dim_internal'], dim_nl=cfg['dim_nl'],
        mode=cfg['mode'], position_scale=cfg['position_scale'],
    ).to(device)
    pb_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return net, pb_net


def find_runs(bool_1d):
    """Contiguous True-runs of a 1D bool sequence as (start, end) half-open intervals."""
    runs = []
    t, T = 0, len(bool_1d)
    while t < T:
        if bool_1d[t]:
            start = t
            while t < T and bool_1d[t]:
                t += 1
            runs.append((start, t))
        else:
            t += 1
    return runs


def bucket_of(midpoint, T):
    frac = midpoint / T
    if frac < 0.25:
        return 'first_25pct'
    if frac < 0.75:
        return 'middle_50pct'
    return 'last_25pct'


def main():
    parser = argparse.ArgumentParser(description="Post-hoc collision analysis of a trained corridor checkpoint.")
    parser.add_argument('--run-folder', type=str, required=True, help='saved_results/<run> folder containing log + trained_controller.pt')
    parser.add_argument('--num-test-samples', type=int, default=500)
    parser.add_argument('--random-seed', type=int, default=5, help='Not logged by the run script - defaults to its CLI default.')
    parser.add_argument('--col-min-dist', type=float, default=1.0, help='Collision-loss distance threshold - not logged by the run script, defaults to its CLI default.')
    parser.add_argument('--max-plots', type=int, default=40, help='Cap on how many colliding trajectories to render as PDFs.')
    args = parser.parse_args()

    log_path = os.path.join(args.run_folder, 'log')
    checkpoint_path = os.path.join(args.run_folder, 'trained_controller.pt')
    out_folder = os.path.join(args.run_folder, 'collision_analysis')
    os.makedirs(out_folder, exist_ok=True)

    cfg = parse_run_log(log_path)
    print(f'[INFO] Parsed run config: {cfg}')

    dataset = rebuild_dataset(cfg, args.random_seed)
    _, test_data = dataset.get_data(num_train_samples=1, num_test_samples=args.num_test_samples)
    test_data = test_data.to(device)

    net, pb_net = rebuild_controller(cfg, checkpoint_path)

    with torch.no_grad():
        x_log, e_log, u_log = net.rollout(test_data, controller=pb_net, train=False)

    torch.save({'x_log': x_log, 'e_log': e_log, 'u_log': u_log}, os.path.join(out_folder, 'trajectories.pt'))

    n_agents, T = cfg['n_agents'], x_log.shape[1]
    loss_probe = CorridorRobotsLoss(n_agents=n_agents, min_dist=args.col_min_dist, alpha_col=1.0)
    distance_sq = loss_probe.get_pairwise_distance_sq(x_log.unsqueeze(-1))    # (S, T, n_agents, n_agents)
    col_matrix = (0.0001 < distance_sq) & (distance_sq < args.col_min_dist ** 2)
    total_collision_instances = col_matrix.sum().item() / 2    # sanity check vs. the run's logged "Number of collisions"
    col_bool = col_matrix.any(dim=-1).any(dim=-1)    # (S, T) - any pair colliding at (sample, t)

    S = x_log.shape[0]
    per_traj = []
    bucket_counts = {'first_25pct': 0, 'middle_50pct': 0, 'last_25pct': 0}
    episode_lengths = []

    for s in range(S):
        runs = find_runs(col_bool[s].tolist())
        if not runs:
            continue
        episodes = []
        for start, end in runs:
            length = end - start
            midpoint = (start + end - 1) / 2
            b = bucket_of(midpoint, T)
            bucket_counts[b] += 1
            episode_lengths.append(length)
            episodes.append({'start': start, 'end': end, 'length': length, 'bucket': b})
        per_traj.append({'sample_idx': s, 'num_episodes': len(episodes), 'episodes': episodes})

    num_colliding = len(per_traj)
    num_episodes = len(episode_lengths)

    summary = {
        'checkpoint': checkpoint_path,
        'config': cfg,
        'num_test_samples': S,
        'horizon': T,
        'total_collision_instances_sanity_check': total_collision_instances,
        'num_trajectories_with_collision': num_colliding,
        'fraction_trajectories_with_collision': num_colliding / S,
        'num_collision_episodes': num_episodes,
        'episode_length_stats': {
            'min': min(episode_lengths) if episode_lengths else None,
            'max': max(episode_lengths) if episode_lengths else None,
            'mean': sum(episode_lengths) / len(episode_lengths) if episode_lengths else None,
        },
        'episode_timing_bucket_counts': bucket_counts,
    }

    with open(os.path.join(out_folder, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_folder, 'per_trajectory.json'), 'w') as f:
        json.dump(per_traj, f, indent=2)

    print(json.dumps(summary, indent=2))

    obstacle_centers, obstacle_covs = build_corridor_obstacles()
    xbar_cols = [4 * i + j for i in range(n_agents) for j in (0, 1)]
    xbar_full = test_data[:, :, net.state_dim:2 * net.state_dim][:, :, xbar_cols]

    n_plots = min(len(per_traj), args.max_plots)
    for k in range(n_plots):
        s = per_traj[k]['sample_idx']
        plot_trajectories(
            x_log[s, :, :],
            xbar=xbar_to_flat(xbar_full[s, -1, :], n_agents),
            n_agents=n_agents,
            save_folder=out_folder,
            filename=f'collision_traj_{s}.pdf',
            text=f'Test rollout {s} - {per_traj[k]["num_episodes"]} collision episode(s)',
            T=T,
            obstacle_centers=obstacle_centers,
            obstacle_covs=obstacle_covs,
        )
    if len(per_traj) > args.max_plots:
        print(f'[INFO] {len(per_traj)} colliding trajectories found - plotted the first {args.max_plots} (--max-plots to change).')

    print(f'[INFO] Saved trajectories, summary, per-trajectory detail, and {n_plots} plots to {out_folder}')


if __name__ == "__main__":
    main()
