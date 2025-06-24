import torch, os
from scipy.stats import multivariate_normal # TODO: use something compatible with tensors
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import imageio
from matplotlib.patches import Circle  # Import Circle from matplotlib.patches
import matplotlib.colors as mcolors
from experiments.robots.arg_parser import argument_parser


args = argument_parser()

def plot_trajectories(
    x, xbar, n_agents, save_folder, text="", save=True, filename='', T=100, horizon = 50,
    dots=False, circles=False, axis=True, min_dist=1, f=5,
    obstacle_centers=None, obstacle_covs=None, round_trip=False, xbar2=None, treshold=None, file_type='png', disp_distance=False, disp_time=False
):
    filename = 'trajectories.pdf' if filename == '' else filename

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # plot obstacles
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        yy, xx = np.meshgrid(np.linspace(-3, 7, 100), np.linspace(-3, 7, 100))
        positions = np.stack([xx, yy], axis=-1)  # Shape: (100, 100, 2)
        zz = np.zeros_like(xx)

        for center, cov in zip(obstacle_centers, obstacle_covs):
            distr = multivariate_normal(
                cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                mean=center.detach().clone().cpu().numpy().flatten()
            )
            zz += distr.pdf(positions)  # Vectorized PDF calculation

        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')
        

    ax.set_title(text)
    colors = ['tab:blue', 'tab:orange']
    epsilon = 0.1 

    # Calculate distances between agents *before* the loop
    if n_agents >= 2:
        x_data_0 = x[:T+1, 0:4].detach().cpu().numpy()  # Agent 0 data (x, y)
        x_data_1 = x[:T+1, 4:8].detach().cpu().numpy()  # Agent 1 data (x, y)
        distances = np.sqrt(np.sum((x_data_0[:, :2] - x_data_1[:, :2])**2, axis=1))

        # Create a mask based on the distances
        distance_mask = ((distances >= args.distance_agents - epsilon) & (distances <= args.distance_agents + epsilon)).astype(int) # 1 if good, 0 if bad

    # Plotting loop
    for i in range(n_agents):
        # Extract trajectory data for the current agent
        x_data = x[:T+1, 4*i].detach().cpu().numpy()
        y_data = x[:T+1, 4*i+1].detach().cpu().numpy()

        if n_agents >= 2:
            color_mask = distance_mask
            # color = 'blue' if i == 0 else 'red'
            # Plot each segment of the trajectory with the appropriate color
            
            if T == 0:
                if disp_distance and n_agents >= 2:
                    # Get current distance between agents
                    current_distance = distances[T]
                    ax.text(0.98, 0.02, f'Distance: {current_distance:.2f}', 
                            transform=ax.transAxes,
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            bbox=dict(facecolor='white', alpha=0.7))

                if disp_time:
                    ax.text(0.02, 0.02, f'Time step: {T}',
                            transform=ax.transAxes,
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            bbox=dict(facecolor='white', alpha=0.7))
            
            
            
            for j in range(T):
                if distance_mask[j] == 1:
                    color = 'blue'  # Distance is within the desired range
                else:
                    color = 'red'  # Distance is outside the desired range
                ax.plot(x_data[j:j+2], y_data[j:j+2], color=color, linewidth=1)
                if disp_distance and n_agents >= 2:
                    # Get current distance between agents
                    current_distance = distances[T]
                    ax.text(0.98, 0.02, f'Distance: {current_distance:.2f}', 
                            transform=ax.transAxes,
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            bbox=dict(facecolor='white', alpha=0.7))

                if disp_time:
                    ax.text(0.02, 0.02, f'Time step: {T}',
                            transform=ax.transAxes,
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            bbox=dict(facecolor='white', alpha=0.7))
        else:
            # If there's only one agent, plot the trajectory in a default color
            ax.plot(x_data, y_data, color='green', linewidth=1)

        ax.plot(
            x[T:,4*i].detach().cpu(), x[T:,4*i+1].detach().cpu(),
            color=colors[i%2], linewidth=1, linestyle='dotted', dashes=(5, 5)
        )




    # colors = ['tab:blue', 'tab:orange']
    # for i in range(n_agents):

    #     # Create a colormap for the trajectory
    #     cmap = plt.get_cmap('jet')  # You can choose a different colormap
    #     num_points = T + 1
    #     norm = mcolors.Normalize(vmin=0, vmax=num_points - 1)

    #     # Plot the trajectory with changing colors
    #     x_data = x[:T+1, 4*i].detach().cpu().numpy()
    #     y_data = x[:T+1, 4*i+1].detach().cpu().numpy()

    #     for j in range(num_points - 1):
    #         color = cmap(j/num_points)
    #         ax.plot(x_data[j:j+2], y_data[j:j+2], color=color, linewidth=1)

    #     ax.plot(
    #         x[T:,4*i].detach().cpu(), x[T:,4*i+1].detach().cpu(),
    #         color=colors[i%2], linewidth=1, linestyle='dotted', dashes=(5, 5)
    #     )

        # ax.plot(
        #     x[:T+1,4*i].detach().cpu(), x[:T+1,4*i+1].detach().cpu(),
        #     color=colors[i%2], linewidth=1
        # )
        # ax.plot(
        #     x[T:,4*i].detach().cpu(), x[T:,4*i+1].detach().cpu(),
        #     color=colors[i%2], linewidth=1, linestyle='dotted', dashes=(5, 5)
        # )


        # ax.plot(
        #     x[T:,4*i].detach().cpu(), x[T:,4*i+1].detach().cpu(),
        #     color='k', linewidth=0.1, linestyle='dotted', dashes=(3, 15)
        # )
    for i in range(n_agents):
        ax.plot(
            x[0,4*i].detach().cpu(), x[0,4*i+1].detach().cpu(),
            color=colors[i%2], marker='8'
        )
        ax.plot(
            xbar[4*i].detach().cpu(), xbar[4*i+1].detach().cpu(),
            color=colors[i%2], marker='*', markersize=10
        )
        if round_trip and xbar2 is not None:
            ax.plot(
                xbar2[4*i].detach().cpu(), xbar2[4*i+1].detach().cpu(),
                color=colors[i%3], marker='*', markersize=10
            )

    if dots:
        for i in range(n_agents):
            ax.plot(  
                x[:T+1,4*i].detach().cpu(), x[:T+1,4*i+1].detach().cpu(),
                color=colors[i%2], linewidth=1, marker = "x"
            )

    # if circles:
    #     for i in range(n_agents):
    #         r = min_dist/2
    #         circle = ax.Circle(
    #             (x[T, 4*i].detach().cpu(), x[T, 4*i+1].detach().cpu()),
    #             r, color=colors[i%2], alpha=0.5, zorder=10
    #         )
    #         ax.add_patch(circle)
    if circles:
        for i in range(n_agents):
            r = min_dist / 2
            # Convert tensor data to numpy arrays
            x_coords = x[:T+1, 4*i].detach().cpu().numpy()
            y_coords = x[:T+1, 4*i+1].detach().cpu().numpy()
            
            # Create circle at final position (T)
            circle = Circle(
                (x_coords[-1], y_coords[-1]),  # Use final position as center
                r, 
                color=colors[i % 2], 
                alpha=0.5, 
                zorder=10
            )
            ax.add_patch(circle)
    ax.axes.xaxis.set_visible(axis)
    ax.axes.yaxis.set_visible(axis)
    ax.set_xlim(-3, 7)
    if save:
        fig.savefig(
            os.path.join(save_folder, filename),
            format=file_type, dpi=300, bbox_inches='tight'
        )       
        plt.close()
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

def save_trajectory_frames(x, xbar, n_agents, save_folder, T=100, interval=1,        obstacle_centers=None, obstacle_covs=None, args=None):
    os.makedirs(save_folder, exist_ok=True)
    for t in range(0, T+1, interval):
        filename = f'frame_{t:03d}.png'
        plot_trajectories(
            x=x[:t+1],  # Trajectory up to time t
            xbar=xbar,
            n_agents=n_agents,
            save_folder=save_folder,
            filename=filename,
            T=t,  # Current time step
            obstacle_centers=obstacle_centers,
            obstacle_covs=obstacle_covs,
            save=True,
            axis=False,
            disp_distance=True,
            disp_time=True,  # Uncomment if you want to display time step
            circles=True,

        )

# def save_trajectory_frames(x, xbar, n_agents, save_folder, T=100, interval=1, obstacle_centers=None, obstacle_covs=None):
#     os.makedirs(save_folder, exist_ok=True)
#     colors = ['tab:blue', 'tab:orange']
#     for t in range(0, T+1, interval):
#         fig, ax = plt.subplots(figsize=(5, 5))
#         if obstacle_centers is not None and obstacle_covs is not None:
#             yy, xx = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
#             zz = xx * 0
#             for center, cov in zip(obstacle_centers, obstacle_covs):
#                 distr = multivariate_normal(
#                     cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
#                     mean=center.detach().clone().cpu().numpy().flatten()
#                 )
#                 for i in range(xx.shape[0]):
#                     for j in range(xx.shape[1]):
#                         zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
#             z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
#             ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')

#         for i in range(n_agents):
#             ax.plot(
#                 x[:t+1, 4*i].detach().cpu(), x[:t+1, 4*i+1].detach().cpu(),
#                 color=colors[i%2], linewidth=1
#             )
#             ax.plot(
#                 x[t, 4*i].detach().cpu(), x[t, 4*i+1].detach().cpu(),
#                 color=colors[i%2], marker='o'
#             )
#             ax.plot(
#                 xbar[4*i].detach().cpu(), xbar[4*i+1].detach().cpu(),
#                 color=colors[i%2], marker='*', markersize=10
#             )
#         ax.set_xlim(-5, 5)
#         ax.set_ylim(-5, 5)
#         ax.set_title(f'Time step: {t}')
#         frame_filename = os.path.join(save_folder, f'frame_{t:03d}.png')
#         fig.savefig(frame_filename)
#         plt.close(fig)


def create_gif_from_frames(frame_folder, gif_filename, duration=0.1,loop=0):
    frames = []
    for frame_file in sorted(os.listdir(frame_folder)):
        if frame_file.endswith('.png'):
            frame_path = os.path.join(frame_folder, frame_file)
            frames.append(imageio.imread(frame_path))
    imageio.mimsave(gif_filename, frames, duration=duration,loop=loop)

def plot_obstacles_and_random_points(
    obstacle_centers, obstacle_covs, x_range=(-1, 5), y_range=(4, 5),
    n_points=5, save_folder=None, filename='random_points_plot.png', file_type='png'
):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot obstacles
    if obstacle_centers is not None and obstacle_covs is not None:
        yy, xx = np.meshgrid(np.linspace(-3, 7, 200), np.linspace(-3, 7, 200))
        zz = xx * 0
        for center, cov in zip(obstacle_centers, obstacle_covs):
            distr = multivariate_normal(
                cov=np.diag(cov.flatten()),
                mean=center.flatten()
            )
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
            ax.pcolormesh(xx, yy, zz, cmap='Greys', shading='gouraud')

    # Generate random points
    np.random.seed(42)  # For reproducibility
    x_blue = np.random.uniform(x_range[0], x_range[1], n_points)
    y_blue = np.random.uniform(y_range[0], y_range[1], n_points)
    x_orange = np.random.uniform(x_range[0], x_range[1], n_points)
    y_orange = np.random.uniform(y_range[0], y_range[1], n_points)

    # Plot random points
    ax.scatter(x_blue, y_blue, color='tab:blue', label='Blue Points', s=50, marker='x')
    ax.scatter(x_orange, y_orange, color='tab:orange', label='Orange Points', s=50, marker='x')

    # Add labels and legend
    ax.set_xlim(-3, 7)
    ax.set_ylim(-3, 7)
    

    # Save or show the plot
    if save_folder:
        fig.savefig(
            os.path.join(save_folder, filename),
            format=file_type, dpi=300, bbox_inches='tight'
        )
        plt.close(fig)
    else:
        plt.show()