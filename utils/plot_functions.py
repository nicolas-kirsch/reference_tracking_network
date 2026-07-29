import torch, os
from scipy.stats import multivariate_normal # TODO: use something compatible with tensors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime
import imageio

# extended past 2 entries so per-agent dotted/solid color pairs stay
# distinguishable for n_agents > 2 (e.g. the networked robots experiments)
AGENT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def plot_trajectories(
    x, xbar, n_agents, save_folder, text="", save=True, filename='', T=100,
    dots=False, circles=False, axis=True, min_dist=1, f=5,
    obstacle_centers=None, obstacle_covs=None, x_compare=None, compare_label='base control only'
):
    """
    Args:
        - x_compare: optional tensor, same shape/layout as x - plotted as a
          dotted line per agent (same color as that agent's solid line), e.g.
          to compare the actual closed-loop trajectory against what the base
          tracking loop alone would produce (dxref=0, no learned controller).
        - compare_label: legend label for the x_compare dotted line.
    """
    filename = 'trajectories.pdf' if filename == '' else filename

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # plot obstacles
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        yy, xx = np.meshgrid(np.linspace(-3, 7, 100), np.linspace(-3, 7, 100))
        zz = xx * 0
        for center, cov in zip(obstacle_centers, obstacle_covs):
            distr = multivariate_normal(
                cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                mean=center.detach().clone().cpu().numpy().flatten()
            )
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()

        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')

    ax.set_title(text)
    colors = AGENT_COLORS
    for i in range(n_agents):
        ax.plot(
            x[:T+1,4*i].detach().cpu(), x[:T+1,4*i+1].detach().cpu(),
            color=colors[i%len(colors)], linewidth=1
        )
        ax.plot(
            x[T:,4*i].detach().cpu(), x[T:,4*i+1].detach().cpu(),
            color='k', linewidth=0.1, linestyle='dotted', dashes=(3, 15)
        )
    if x_compare is not None:
        for i in range(n_agents):
            ax.plot(
                x_compare[:T+1,4*i].detach().cpu(), x_compare[:T+1,4*i+1].detach().cpu(),
                color=colors[i%len(colors)], linewidth=1, linestyle=':'
            )
    for i in range(n_agents):
        ax.plot(
            x[0,4*i].detach().cpu(), x[0,4*i+1].detach().cpu(),
            color=colors[i%len(colors)], marker='8'
        )
        ax.plot(
            xbar[4*i].detach().cpu(), xbar[4*i+1].detach().cpu(),
            color=colors[i%len(colors)], marker='*', markersize=10
        )

    if dots:
        for i in range(n_agents):
            ax.plot(
                x[:T+1,4*i].detach().cpu(), x[:T+1,4*i+1].detach().cpu(),
                color=colors[i%len(colors)], linewidth=1, marker = "x"
            )

    if circles:
        for i in range(n_agents):
            r = min_dist/2
            circle = ax.Circle(
                (x[T, 4*i].detach().cpu(), x[T, 4*i+1].detach().cpu()),
                r, color=colors[i%len(colors)], alpha=0.5, zorder=10
            )
            ax.add_patch(circle)
    if x_compare is not None:
        legend_handles = [
            Line2D([0], [0], color='gray', linestyle='-', label='with learned offset'),
            Line2D([0], [0], color='gray', linestyle=':', label=compare_label),
        ]
        ax.legend(handles=legend_handles, loc='best')
    ax.axes.xaxis.set_visible(axis)
    ax.axes.yaxis.set_visible(axis)
    if save:
        fig.savefig(
            os.path.join(save_folder, filename),
            format='pdf'
        )
        plt.close()
    else:
        plt.show()


def plot_input_norm(u, n_agents, save_folder, text="", save=True, filename='', T=None,
                     symbol=r'\delta x_{\mathrm{ref}}'):
    """
    Plot ||u^(i)_t||_2 per agent over time - a quick visual check of the l_p
    nature of the reference offset dxref (u fed to RobotDynamics IS dxref,
    the offset added to the nominal setpoint xbar - see CLAUDE.md's
    "Reference governor structure"): since dxref is driven by a disturbance
    reconstruction that is itself l_p (e.g. nonzero only at t=0 in the
    nominal case, no process noise afterward), it should decay towards 0
    rather than staying persistently large.

    Args:
        - u: tensor of shape (T_total, in_dim), in_dim = 2*n_agents, layout
          [u1_x,u1_y, u2_x,u2_y, ...] (one rollout, no batch dim).
        - T: number of initial time steps to plot. Defaults to all of u.
        - symbol: LaTeX symbol used for the y-axis label. Defaults to the
          reference-offset dxref, which is what "u" is in this codebase.
    """
    filename = 'u_norm.pdf' if filename == '' else filename
    T = u.shape[0] if T is None else min(T, u.shape[0])
    t = torch.arange(T)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(n_agents):
        u_i = u[:T, 2 * i:2 * i + 2]
        norm_i = torch.norm(u_i, dim=-1).detach().cpu()
        ax.plot(t, norm_i, linewidth=1.5, label=f'robot {i + 1}')

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\|' + symbol + r'^{(i)}(t)\|_2$')
    ax.set_title(text)
    ax.legend()
    ax.grid(True)

    if save:
        fig.savefig(
            os.path.join(save_folder, filename),
            format='pdf'
        )
        plt.close(fig)
    else:
        plt.show()


def plot_traj_vs_time(t_end, n_agents, save_folder, x, u=None, text="", save=True, filename=''):
    filename = filename if filename=='' else filename+'_'
    now = datetime.now()
    formatted_date = now.strftime('%m-%d-%H:%M')
    t = torch.linspace(0,t_end-1, t_end)
    if u is not None:
        p = 3
    else:
        p = 2
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i].detach().cpu())
        plt.plot(t, x[:,4*i+1].detach().cpu())
    plt.xlabel(r'$t$')
    plt.title(r'$x(t)$')
    plt.subplot(1, p, 2)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i+2].detach().cpu())
        plt.plot(t, x[:,4*i+3].detach().cpu())
    plt.xlabel(r'$t$')
    plt.title(r'$v(t)$')
    plt.suptitle(text)
    if p == 3:
        plt.subplot(1, 3, 3)
        for i in range(n_agents):
            plt.plot(t, u[:, 2*i].detach().cpu())
            plt.plot(t, u[:, 2*i+1].detach().cpu())
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig(
            os.path.join(
                save_folder,
                filename+text+'_x_u.pdf'
            ),
            format='pdf'
        )
        plt.close()
    else:
        plt.show()


def save_trajectory_frames(x, xbar, n_agents, save_folder, T=100, interval=1,f=5, obstacle_centers=None, obstacle_covs=None):
    os.makedirs(save_folder, exist_ok=True)

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # plot obstacles
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        yy, xx = np.meshgrid(np.linspace(-3, 7, 100), np.linspace(-3, 7, 100))
        zz = xx * 0
        for center, cov in zip(obstacle_centers, obstacle_covs):
            distr = multivariate_normal(
                cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                mean=center.detach().clone().cpu().numpy().flatten()
            )
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()

        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')



    colors = ['tab:blue', 'tab:orange']
    for t in range(0, T+1, interval):
        print(f'Saving frame {t}/{T}...')
        fig, ax = plt.subplots(figsize=(f,f))
        # plot obstacles
        if not obstacle_covs is None:
            assert not obstacle_centers is None
            yy, xx = np.meshgrid(np.linspace(-3, 7, 100), np.linspace(-3, 7, 100))
            zz = xx * 0
            for center, cov in zip(obstacle_centers, obstacle_covs):
                distr = multivariate_normal(
                    cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                    mean=center.detach().clone().cpu().numpy().flatten()
                )
                for i in range(xx.shape[0]):
                    for j in range(xx.shape[1]):
                        zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
            z_min, z_max = np.abs(zz).min(), np.abs(zz).max()

            ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')
        for i in range(n_agents):
            ax.plot(
                x[:t+1, 4*i].detach().cpu(), x[:t+1, 4*i+1].detach().cpu(),
                color=colors[i%2], linewidth=1
            )
            ax.plot(
                x[t, 4*i].detach().cpu(), x[t, 4*i+1].detach().cpu(),
                color=colors[i%2], marker='o'
            )
            ax.plot(
                x[0, 4*i].detach().cpu(), x[0, 4*i+1].detach().cpu(),
                color=colors[i%2], marker='o', markerfacecolor='none'
            )
            r = 0.5
            circle = plt.Circle(
                (x[t, 4*i].detach().cpu(), x[t, 4*i+1].detach().cpu()),
                r, color=colors[i%2], alpha=0.5, zorder=10
            )
            ax.add_patch(circle)
            ax.plot(
                xbar[4*i].detach().cpu(), xbar[4*i+1].detach().cpu(),
                color=colors[i%2], marker='*', markersize=10
            )
            
        dist = torch.sqrt((x[t, 4*0] - x[t, 4*1])**2 + (x[t, 4*0+1] - x[t, 4*1+1])**2)
        if dist < 1:
            ax.set_title('Collision', color='red')
             
        else:
            ax.set_title(f'Time step: {t}')
        """ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)"""
        
        frame_filename = os.path.join(save_folder, f'frame_{t:03d}.png')
        fig.savefig(frame_filename)
        plt.close(fig)


def create_gif_from_frames(frame_folder, gif_filename, duration=0.1):
    frames = []
    for frame_file in sorted(os.listdir(frame_folder)):
        if frame_file.endswith('.png'):
            frame_path = os.path.join(frame_folder, frame_file)
            frames.append(imageio.imread(frame_path))
    imageio.mimsave(gif_filename, frames, duration=duration)