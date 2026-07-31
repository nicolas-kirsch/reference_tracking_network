import sys, os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import multivariate_normal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)

from config import device
from networked_robots.corridor_robots_loss import CorridorRobotsLoss
from networked_robots.run_networked_control_corridor import build_corridor_obstacles
from networked_robots.analyze_collisions import (
    parse_run_log, rebuild_dataset, rebuild_controller, find_runs,
)
from utils.plot_functions import AGENT_COLORS

"""
Animate the worst colliding test rollouts of a trained
run_networked_control_corridor.py checkpoint, as GIFs.

Complements analyze_collisions.py (which reports collision statistics and
static PDFs) by showing the TIME evolution: each frame draws every agent as a
disc of radius col_min_dist/2, so two discs touching is exactly the condition
the run's reported collision count tests, alongside a live pairwise-distance
trace against the two thresholds that matter and that differ from each other:
  - col_min_dist        - what count_collisions() reports (default 1.0)
  - col_min_dist + 0.2  - where CorridorRobotsLoss.f_loss_ca actually starts
                          penalizing (min_sec_dist), i.e. the only region with
                          a nonzero collision gradient
Rollouts are ranked by the same quantity the reported count sums, so
`--num-gifs k` gives the k rollouts contributing most to that number.
"""


def obstacle_field(obstacle_centers, obstacle_covs, lim=(-3, 7), n=100):
    """Summed Gaussian-bump obstacle density on a grid, as in plot_trajectories."""
    yy, xx = np.meshgrid(np.linspace(*lim, n), np.linspace(*lim, n))
    zz = np.zeros_like(xx)
    for center, cov in zip(obstacle_centers, obstacle_covs):
        distr = multivariate_normal(
            cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
            mean=center.detach().clone().cpu().numpy().flatten(),
        )
        zz += distr.pdf(np.dstack((xx, yy)))
    return xx, yy, zz


def animate_rollout(x, xbar, n_agents, col_min_dist, obstacle_field_data, episodes,
                    out_path, title, fps=12, tail=25):
    """
    Args:
        - x: (T, 4*n_agents) trajectory of one rollout, flat per-agent layout.
        - xbar: (2*n_agents,) that rollout's targets.
        - episodes: list of {'start','end',...} collision intervals, shaded on
          the distance panel.
    """
    T = x.shape[0]
    px = np.stack([x[:, 4 * i] for i in range(n_agents)], axis=1)        # (T, n_agents)
    py = np.stack([x[:, 4 * i + 1] for i in range(n_agents)], axis=1)

    # pairwise distances, min over pairs at each t (n_agents=2 -> just the one pair)
    pairs = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
    dists = np.stack([np.hypot(px[:, i] - px[:, j], py[:, i] - py[:, j]) for i, j in pairs], axis=1)
    dmin = dists.min(axis=1)

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(5.4, 7.2), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.28})

    xx, yy, zz = obstacle_field_data
    ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=np.abs(zz).min(), vmax=np.abs(zz).max(), shading='gouraud')
    ax.set_xlim(-3, 7); ax.set_ylim(-3, 7); ax.set_aspect('equal')
    ax.set_xlabel('$p_x$'); ax.set_ylabel('$p_y$')
    ax.set_title(title, fontsize=9)

    trails, discs, centers = [], [], []
    for i in range(n_agents):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(px[:, i], py[:, i], color=c, lw=0.6, alpha=0.25)            # full path, faded
        ax.plot(px[0, i], py[0, i], color=c, marker='8', ms=5)              # start
        ax.plot(xbar[2 * i], xbar[2 * i + 1], color=c, marker='*', ms=11)   # target
        trails.append(ax.plot([], [], color=c, lw=1.8)[0])
        # radius col_min_dist/2: two discs overlapping <=> the reported collision test fires
        disc = plt.Circle((px[0, i], py[0, i]), col_min_dist / 2, color=c, alpha=0.30, ec=c, lw=1.0)
        ax.add_patch(disc); discs.append(disc)
        centers.append(ax.plot([], [], color=c, marker='o', ms=4)[0])

    axd.plot(np.arange(T), dmin, color='k', lw=1.2)
    axd.axhline(col_min_dist, color='crimson', ls='--', lw=1.0,
                label='$d_{col}=%.2f$ (counted)' % col_min_dist)
    axd.axhline(col_min_dist + 0.2, color='darkorange', ls=':', lw=1.0,
                label='$d_{col}+0.2=%.2f$ (loss active)' % (col_min_dist + 0.2))
    for ep in episodes:
        axd.axvspan(ep['start'], ep['end'] - 1, color='crimson', alpha=0.18, lw=0)
    axd.set_xlim(0, T - 1); axd.set_ylim(0, max(dmin.max() * 1.1, col_min_dist + 0.5))
    axd.set_xlabel('timestep'); axd.set_ylabel('min pairwise dist')
    axd.legend(fontsize=6, loc='upper right')
    cursor = axd.axvline(0, color='tab:blue', lw=1.2)
    readout = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', fontsize=8,
                      bbox=dict(fc='white', ec='none', alpha=0.75))

    def update(t):
        for i in range(n_agents):
            lo = max(0, t - tail)
            trails[i].set_data(px[lo:t + 1, i], py[lo:t + 1, i])
            discs[i].center = (px[t, i], py[t, i])
            centers[i].set_data([px[t, i]], [py[t, i]])
        colliding = dmin[t] < col_min_dist
        for i in range(n_agents):
            discs[i].set_alpha(0.55 if colliding else 0.30)
            discs[i].set_edgecolor('crimson' if colliding else AGENT_COLORS[i % len(AGENT_COLORS)])
            discs[i].set_linewidth(2.0 if colliding else 1.0)
        cursor.set_xdata([t, t])
        readout.set_text('t = %3d\nmin dist = %.3f%s' % (t, dmin[t], '   COLLISION' if colliding else ''))
        return trails + discs + centers + [cursor, readout]

    FuncAnimation(fig, update, frames=T, blit=False).save(
        out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Animate the worst colliding test rollouts of a trained corridor checkpoint as GIFs.")
    parser.add_argument('--run-folder', type=str, required=True,
                        help='saved_results/<run> folder containing log + trained_controller.pt')
    parser.add_argument('--num-test-samples', type=int, default=500)
    parser.add_argument('--random-seed', type=int, default=5,
                        help='Not logged by the run script - defaults to its CLI default.')
    parser.add_argument('--col-min-dist', type=float, default=1.0,
                        help='Collision threshold reported by count_collisions - not logged, defaults to its CLI default.')
    parser.add_argument('--num-gifs', type=int, default=4,
                        help='How many rollouts to animate, worst first. Default 4.')
    parser.add_argument('--fps', type=int, default=12)
    args = parser.parse_args()

    log_path = os.path.join(args.run_folder, 'log')
    checkpoint_path = os.path.join(args.run_folder, 'trained_controller.pt')
    out_folder = os.path.join(args.run_folder, 'collision_gifs')
    os.makedirs(out_folder, exist_ok=True)

    cfg = parse_run_log(log_path)
    print(f'[INFO] Parsed run config: {cfg}')

    dataset = rebuild_dataset(cfg, args.random_seed)
    _, test_data = dataset.get_data(num_train_samples=1, num_test_samples=args.num_test_samples)
    test_data = test_data.to(device)

    net, pb_net = rebuild_controller(cfg, checkpoint_path)
    with torch.no_grad():
        x_log, _, _ = net.rollout(test_data, controller=pb_net, train=False)

    n_agents, T, S = cfg['n_agents'], x_log.shape[1], x_log.shape[0]
    probe = CorridorRobotsLoss(n_agents=n_agents, min_dist=args.col_min_dist, alpha_col=1.0)
    distance_sq = probe.get_pairwise_distance_sq(x_log.unsqueeze(-1))    # (S, T, n_agents, n_agents)
    # exactly count_collisions()'s test, including its 0.0001 lower bound
    col_matrix = (0.0001 < distance_sq) & (distance_sq < args.col_min_dist ** 2)
    per_sample_count = col_matrix.sum(dim=(1, 2, 3)) / 2    # (S,) - this rollout's share of the reported total
    total = per_sample_count.sum().item()
    print('[INFO] reported-style collision count over %d rollouts x %d steps: %.0f '
          '(%.3f per rollout, %.2f%% of agent-pair-timesteps)'
          % (S, T, total, total / S, 100 * total / (S * T * len(
              [1 for i in range(n_agents) for j in range(i + 1, n_agents)]))))

    col_bool = col_matrix.any(dim=-1).any(dim=-1)    # (S, T)
    order = torch.argsort(per_sample_count, descending=True).tolist()
    worst = [s for s in order if per_sample_count[s] > 0][:args.num_gifs]
    if not worst:
        print('[INFO] no colliding rollouts found - nothing to animate.')
        return

    obstacle_centers, obstacle_covs = build_corridor_obstacles()
    field = obstacle_field(obstacle_centers, obstacle_covs)
    xbar_cols = [4 * i + j for i in range(n_agents) for j in (0, 1)]
    xbar_full = test_data[:, :, net.state_dim:2 * net.state_dim][:, :, xbar_cols]

    manifest = []
    for s in worst:
        episodes = [{'start': a, 'end': b, 'length': b - a}
                    for a, b in find_runs(col_bool[s].tolist())]
        n_steps = int(per_sample_count[s].item())
        dmin_traj = distance_sq[s].masked_fill(distance_sq[s] <= 0.0001, float('inf')).min().sqrt().item()
        out_path = os.path.join(out_folder, f'collision_rollout_{s}.gif')
        animate_rollout(
            x_log[s].detach().cpu().numpy(),
            xbar_full[s, -1, :].detach().cpu().numpy(),
            n_agents, args.col_min_dist, field, episodes, out_path,
            title=('rollout %d - %d counted collision-timesteps in %d episode(s), min dist %.3f'
                   % (s, n_steps, len(episodes), dmin_traj)),
            fps=args.fps,
        )
        manifest.append({'sample_idx': s, 'counted_collision_timesteps': n_steps,
                         'num_episodes': len(episodes), 'min_distance': dmin_traj,
                         'episodes': episodes, 'gif': os.path.basename(out_path)})
        print('[INFO] wrote %s  (%d collision-timesteps, %d episode(s), min dist %.3f)'
              % (out_path, n_steps, len(episodes), dmin_traj))

    with open(os.path.join(out_folder, 'manifest.json'), 'w') as f:
        json.dump({'run_folder': args.run_folder, 'config': cfg,
                   'num_test_samples': S, 'horizon': T,
                   'total_reported_style_count': total, 'animated': manifest}, f, indent=2)
    print(f'[INFO] Saved {len(worst)} GIFs + manifest.json to {out_folder}')


if __name__ == "__main__":
    main()
