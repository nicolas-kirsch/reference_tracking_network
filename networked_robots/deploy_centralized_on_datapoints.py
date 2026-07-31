import sys, os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)

from config import device
from plants import RobotsSystem
from controllers import PerfBoostController
from networked_robots.corridor_robots_loss import CorridorRobotsLoss
from networked_robots.run_networked_control_corridor import build_corridor_obstacles
from networked_robots.analyze_collisions import parse_run_log, rebuild_dataset, rebuild_controller, find_runs
from networked_robots.animate_collisions import obstacle_field
from utils.plot_functions import AGENT_COLORS

"""
Deploy a trained CENTRALIZED controller (experiments/robots) on specific test
datapoints taken from a NETWORKED run (networked_robots), and animate the two
side by side.

The two experiments share a data layout exactly, so the datapoints transfer
with no conversion:
    data[:, 0, :4N]                      initial condition (t=0 only)
    data[:, 1:, 4N + 4i : 4N + 4i + 2]   agent i's target, held from t=1
and both rollouts read the reference as columns [8,9,12,13] of the reference
block for N=2. x_log has the identical flat per-agent layout
[x1,y1,vx1,vy1, x2,y2,vx2,vy2] in both, so one animation routine serves both.

The datapoints are selected by index into the networked run's test set, which
is regenerated deterministically from its log (same seed/std/horizon), so
`--rollouts 276 487 ...` refers to exactly the rollouts animate_collisions.py
picked out.

NOTE ON CHECKPOINTS: experiments/robots/run.py originally saved only
ctl.c_ren.state_dict(), which omits the MLP (M2, the bounded context factor)
that multiplies the REN output. Such a checkpoint CANNOT reconstruct the
trained controller. This script refuses it by default; --allow-missing-mlp
proceeds with a randomly initialized MLP purely to demonstrate that the
resulting trajectories are arbitrary, and labels every output as INVALID.
"""


def load_centralized_controller(checkpoint_path, sys_plant, allow_missing_mlp, mlp_seed):
    """
    Returns (controller, cfg_args, mlp_is_random).

    Accepts both the current full-controller format ({'controller': state_dict,
    'Q':..., 'args':...}) and the legacy REN-only format (flat REN keys).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get('args', {})

    torch.manual_seed(mlp_seed)    # only matters when the MLP has to be invented
    ctl = PerfBoostController(
        internal_model=sys_plant.internal_model(),
        input_init=sys_plant.eta_init,
        output_init=sys_plant.u_init,
        dim_internal=ckpt_args.get('dim_internal', 12),
        dim_nl=ckpt_args.get('dim_nl', 12),
        initialization_std=ckpt_args.get('cont_init_std', 0.1),
        output_amplification=ckpt.get('output_amplification', 20),
    ).to(device)

    if 'controller' in ckpt:
        ctl.load_state_dict(ckpt['controller'])
        return ctl, ckpt_args, False

    ren_sd = {k: v for k, v in ckpt.items() if k not in ('Q', 'args')}
    if not any('MLP' in k or 'mlp' in k for k in ren_sd):
        msg = (
            f"\n{checkpoint_path}\ncontains only the REN (M1): keys {sorted(ren_sd)}.\n"
            "The MLP (M2) parameters were never saved, and the controller output is\n"
            "    output = REN(w_hat) * MLP(w_hat, xbar) * output_amplification,\n"
            "so without M2 the trained controller cannot be reconstructed - the\n"
            "elementwise factor multiplying the REN would be arbitrary.\n"
            "Fix: experiments/robots/run.py now saves the full ctl.state_dict();\n"
            "retrain to obtain a deployable checkpoint. Pass --allow-missing-mlp to\n"
            "proceed with a RANDOM MLP anyway (for demonstration only - the\n"
            "resulting trajectories are meaningless)."
        )
        if not allow_missing_mlp:
            raise SystemExit('[ERROR]' + msg)
        print('[WARNING]' + msg)
    ctl.c_ren.load_state_dict(ren_sd)
    return ctl, ckpt_args, True


def animate_side_by_side(x_left, x_right, xbar, n_agents, col_min_dist, field,
                         out_path, label_left, label_right, suptitle, fps=12, tail=25):
    """Two map panels sharing one min-pairwise-distance panel, same datapoint under two controllers."""
    T = x_left.shape[0]

    def unpack(x):
        px = np.stack([x[:, 4 * i] for i in range(n_agents)], axis=1)
        py = np.stack([x[:, 4 * i + 1] for i in range(n_agents)], axis=1)
        pairs = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
        d = np.stack([np.hypot(px[:, i] - px[:, j], py[:, i] - py[:, j]) for i, j in pairs], axis=1)
        return px, py, d.min(axis=1)

    pxl, pyl, dl = unpack(x_left)
    pxr, pyr, dr = unpack(x_right)

    fig = plt.figure(figsize=(9.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.32, wspace=0.18)
    axl, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axd = fig.add_subplot(gs[1, :])
    fig.suptitle(suptitle, fontsize=9)

    xx, yy, zz = field
    artists = []
    for ax, px, py, dmin, label in ((axl, pxl, pyl, dl, label_left), (axr, pxr, pyr, dr, label_right)):
        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=np.abs(zz).min(), vmax=np.abs(zz).max(), shading='gouraud')
        ax.set_xlim(-3, 7); ax.set_ylim(-3, 7); ax.set_aspect('equal')
        ax.set_title(label, fontsize=8)
        ax.set_xlabel('$p_x$'); ax.set_ylabel('$p_y$')
        trails, discs = [], []
        for i in range(n_agents):
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.plot(px[:, i], py[:, i], color=c, lw=0.6, alpha=0.25)
            ax.plot(px[0, i], py[0, i], color=c, marker='8', ms=5)
            ax.plot(xbar[2 * i], xbar[2 * i + 1], color=c, marker='*', ms=11)
            trails.append(ax.plot([], [], color=c, lw=1.8)[0])
            disc = plt.Circle((px[0, i], py[0, i]), col_min_dist / 2, color=c, alpha=0.30, ec=c, lw=1.0)
            ax.add_patch(disc); discs.append(disc)
        readout = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', fontsize=7.5,
                          bbox=dict(fc='white', ec='none', alpha=0.75))
        artists.append((px, py, dmin, trails, discs, readout))

    axd.plot(np.arange(T), dl, color='tab:purple', lw=1.3, label=label_left)
    axd.plot(np.arange(T), dr, color='tab:green', lw=1.3, label=label_right)
    axd.axhline(col_min_dist, color='crimson', ls='--', lw=1.0,
                label='$d_{col}=%.2f$ (counted)' % col_min_dist)
    axd.axhline(col_min_dist + 0.2, color='darkorange', ls=':', lw=1.0,
                label='$d_{col}+0.2$ (loss active)')
    axd.set_xlim(0, T - 1); axd.set_ylim(0, max(dl.max(), dr.max()) * 1.1)
    axd.set_xlabel('timestep'); axd.set_ylabel('min pairwise dist')
    axd.legend(fontsize=6, ncol=4, loc='upper right')
    cursor = axd.axvline(0, color='k', lw=1.0)

    def update(t):
        out = [cursor]
        for px, py, dmin, trails, discs, readout in artists:
            lo = max(0, t - tail)
            colliding = dmin[t] < col_min_dist
            for i in range(n_agents):
                trails[i].set_data(px[lo:t + 1, i], py[lo:t + 1, i])
                discs[i].center = (px[t, i], py[t, i])
                discs[i].set_alpha(0.55 if colliding else 0.30)
                discs[i].set_edgecolor('crimson' if colliding else AGENT_COLORS[i % len(AGENT_COLORS)])
                discs[i].set_linewidth(2.0 if colliding else 1.0)
            readout.set_text('t = %3d\nmin dist = %.3f%s' % (t, dmin[t], '   COLLISION' if colliding else ''))
            out += trails + discs + [readout]
        cursor.set_xdata([t, t])
        return out

    FuncAnimation(fig, update, frames=T, blit=False).save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Deploy a trained centralized controller on datapoints from a networked run; animate side by side.")
    p.add_argument('--networked-run-folder', type=str, required=True,
                   help='networked_robots/saved_results/<run> supplying the datapoints (and the networked comparison).')
    p.add_argument('--centralized-run-folder', type=str, required=True,
                   help='experiments/robots/saved_results/<run> holding the centralized checkpoint.')
    p.add_argument('--rollouts', type=int, nargs='*', default=None,
                   help='Test-set indices to deploy. Default: the ones in the networked run\'s collision_gifs/manifest.json.')
    p.add_argument('--num-test-samples', type=int, default=500)
    p.add_argument('--random-seed', type=int, default=5)
    p.add_argument('--col-min-dist', type=float, default=1.0)
    p.add_argument('--allow-missing-mlp', action='store_true',
                   help='Proceed with a RANDOM MLP if the checkpoint lacks M2. Output is meaningless - demonstration only.')
    p.add_argument('--mlp-seed', type=int, default=0, help='Seed for the invented MLP under --allow-missing-mlp.')
    p.add_argument('--fps', type=int, default=12)
    p.add_argument('--out-folder', type=str, default=None,
                   help="Where to write the GIFs and summary. Default: <centralized-run-folder>/gifs, "
                        "i.e. alongside the deployed checkpoint, since that is the artifact being evaluated.")
    args = p.parse_args()

    # outputs live with the centralized checkpoint being deployed; the networked run
    # that supplied the datapoints is recorded in every filename and in the summary,
    # so several networked runs can be deployed into the same folder without clashing.
    out_folder = args.out_folder or os.path.join(args.centralized_run_folder, 'gifs')
    os.makedirs(out_folder, exist_ok=True)
    # e.g. networked_control_corridor_07_29_15_12_15 -> 07_29_15_12_15
    net_tag = os.path.basename(os.path.normpath(args.networked_run_folder)).replace(
        'networked_control_corridor_', '')

    # ---- 1. the datapoints: regenerate the networked run's exact test set ----
    cfg = parse_run_log(os.path.join(args.networked_run_folder, 'log'))
    dataset = rebuild_dataset(cfg, args.random_seed)
    _, test_data = dataset.get_data(num_train_samples=1, num_test_samples=args.num_test_samples)
    test_data = test_data.to(device)
    n_agents = cfg['n_agents']

    rollouts = args.rollouts
    if not rollouts:
        manifest = os.path.join(args.networked_run_folder, 'collision_gifs', 'manifest.json')
        rollouts = [a['sample_idx'] for a in json.load(open(manifest))['animated']]
        print(f'[INFO] taking rollouts from {manifest}: {rollouts}')
    data = test_data[rollouts]

    # ---- 2. networked side (reference behaviour on these datapoints) ----
    net, pb_net = rebuild_controller(cfg, os.path.join(args.networked_run_folder, 'trained_controller.pt'))
    with torch.no_grad():
        x_net, _, _ = net.rollout(data, controller=pb_net, train=False)

    # ---- 3. centralized side, same datapoints, no conversion ----
    assert n_agents == 2, "the centralized experiment is hardcoded for 2 agents"
    sys_plant = RobotsSystem(
        x_init=None, u_init=None, linear_plant=cfg['linearize_plant'],
        k=cfg['spring_const'], n_agents=n_agents,
    ).to(device)
    ctl, ckpt_args, mlp_is_random = load_centralized_controller(
        os.path.join(args.centralized_run_folder, 'trained_controller.pt'),
        sys_plant, args.allow_missing_mlp, args.mlp_seed,
    )
    print(f'[INFO] centralized checkpoint args: {ckpt_args}')
    with torch.no_grad():
        x_cen, _, _ = sys_plant.rollout(controller=ctl, data=data, train=False)

    # ---- 4. collision stats under each controller, identical test ----
    probe = CorridorRobotsLoss(n_agents=n_agents, min_dist=args.col_min_dist, alpha_col=1.0)

    def stats(x):
        d2 = probe.get_pairwise_distance_sq(x.unsqueeze(-1))
        m = (0.0001 < d2) & (d2 < args.col_min_dist ** 2)
        per = (m.sum(dim=(1, 2, 3)) / 2).tolist()
        dmin = d2.masked_fill(d2 <= 0.0001, float('inf')).amin(dim=(1, 2, 3)).sqrt().tolist()
        colb = m.any(-1).any(-1)
        eps = [find_runs(colb[k].tolist()) for k in range(x.shape[0])]
        return per, dmin, eps

    per_net, dmin_net, eps_net = stats(x_net)
    per_cen, dmin_cen, eps_cen = stats(x_cen)

    tag_cen = 'centralized [INVALID: random M2]' if mlp_is_random else 'centralized'
    print('\n%-8s %-34s %-34s' % ('rollout', 'networked (%s)' % cfg['mode'], tag_cen))
    for k, s in enumerate(rollouts):
        print('%-8d col-steps %-3d  min dist %-6.3f      col-steps %-3d  min dist %-6.3f'
              % (s, per_net[k], dmin_net[k], per_cen[k], dmin_cen[k]))

    # ---- 5. side-by-side GIFs ----
    field = obstacle_field(*build_corridor_obstacles())
    xbar_cols = [4 * i + j for i in range(n_agents) for j in (0, 1)]
    xbar_all = data[:, -1, net.state_dim:2 * net.state_dim][:, xbar_cols]

    suffix = '_INVALID_random_mlp_seed%d' % args.mlp_seed if mlp_is_random else ''
    manifest = []
    for k, s in enumerate(rollouts):
        out_path = os.path.join(out_folder, f'vs_networked_{net_tag}_rollout_{s}{suffix}.gif')
        animate_side_by_side(
            x_net[k].detach().cpu().numpy(), x_cen[k].detach().cpu().numpy(),
            xbar_all[k].detach().cpu().numpy(), n_agents, args.col_min_dist, field, out_path,
            label_left='networked (%s) - %d col-steps, min %.3f' % (cfg['mode'], per_net[k], dmin_net[k]),
            label_right='%s - %d col-steps, min %.3f' % (tag_cen, per_cen[k], dmin_cen[k]),
            suptitle='test rollout %d - identical initial condition and targets' % s
                     + ('\nRIGHT PANEL IS NOT THE TRAINED CONTROLLER: checkpoint lacks M2, MLP seed %d' % args.mlp_seed
                        if mlp_is_random else ''),
            fps=args.fps,
        )
        manifest.append({'sample_idx': s,
                         'networked': {'col_steps': per_net[k], 'min_dist': dmin_net[k],
                                       'episodes': eps_net[k]},
                         'centralized': {'col_steps': per_cen[k], 'min_dist': dmin_cen[k],
                                         'episodes': eps_cen[k], 'mlp_is_random': mlp_is_random},
                         'gif': os.path.basename(out_path)})
        print('[INFO] wrote %s' % out_path)

    with open(os.path.join(out_folder, 'summary_vs_networked_%s%s.json' % (net_tag, suffix)), 'w') as f:
        json.dump({'networked_run': args.networked_run_folder, 'networked_mode': cfg['mode'],
                   'centralized_run': args.centralized_run_folder,
                   'centralized_mlp_is_random': mlp_is_random,
                   'mlp_seed': args.mlp_seed if mlp_is_random else None,
                   'rollouts': rollouts, 'comparison': manifest}, f, indent=2)
    print(f'[INFO] Saved to {out_folder}')


if __name__ == "__main__":
    main()
