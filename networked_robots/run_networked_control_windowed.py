import sys, os, logging, copy, time, torch
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)

from config import device
from networked_robots.arg_parser import argument_parser, print_args
from networked_robots.robots_dataset import NetworkedRobotsDataset
from networked_robots.robots_network import RobotsNetwork
from networked_robots.pb_controller_network_windowed import PBControllerNetworkWindowed
from networked_robots.robots_loss import NetworkedRobotsLoss
from networked_robots.run_networked_control import (
    build_complete_graph_adjacency, xbar_to_flat, plot_rollout_diagnostics, format_loss_components,
)
from utils.assistive_functions import WrapLogger

"""
EXPERIMENTAL script - trains PBControllerNetworkWindowed, which deliberately
relaxes CLAUDE.md's Invariant 1 (see RobotPBControllerWindowed's docstring)
in exchange for giving M_p a windowed copy of the context signals, to address
the "w_hat is a single impulse -> REN explodes" training pathology observed
with the guarantee-preserving PBControllerNetwork (run_networked_control.py,
left untouched). Kept as a separate script/output folder so the two never
collide and can diverge freely while this is experimental.
"""


def main():
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = os.path.join(BASE_DIR, 'networked_robots', 'saved_results')
    save_folder = os.path.join(save_path, 'networked_control_windowed_' + now)
    os.makedirs(save_folder, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_folder, 'log'),
        format='%(asctime)s %(message)s',
        filemode='w',
    )
    logger = logging.getLogger('networked_robots_networked_control_windowed')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)

    args = argument_parser()
    logger.info(print_args(args))
    logger.info(f'[INFO] EXPERIMENTAL windowed controller - window_fraction: {args.window_fraction}')
    torch.manual_seed(args.random_seed)

    # ------------ 1. Communication graph ------------
    adjacency = build_complete_graph_adjacency(args.n_agents)

    # ------------ 2. Dataset ------------
    dataset = NetworkedRobotsDataset(
        random_seed=args.random_seed,
        horizon=args.horizon,
        n_agents=args.n_agents,
        std_ini=args.std_init_plant,
        min_dist=args.min_dist,
        central_data=args.central_data,
    )
    train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=args.num_test_samples)
    train_data, test_data = train_data.to(device), test_data.to(device)
    plot_data = test_data[:args.plot_samples]

    # ------------ 3. Plant (network of decoupled per-agent dynamics) ------------
    net = RobotsNetwork(
        n_agents=args.n_agents,
        adjacency=adjacency,
        linear_plant=args.linearize_plant,
        k=args.spring_const,
    ).to(device)

    # ------------ 4. Networked PB controller, windowed variant (random init before training) ------------
    pb_net = PBControllerNetworkWindowed(
        net,
        dim_internal=args.dim_internal,
        dim_nl=args.dim_nl,
        horizon=args.horizon,
        window_fraction=args.window_fraction,
        initialization_std=args.cont_init_std,
    ).to(device)
    total_params = sum(p.numel() for p in pb_net.parameters())
    logger.info(f'Number of controller parameters: {total_params}')

    # ------------ 4b. Loss ------------
    loss_fn = NetworkedRobotsLoss(alpha_tracking=args.alpha_tracking, alpha_energy=args.alpha_energy)

    # ------------ 5. Plot before training ------------
    logger.info('Rolling out the network under the UNTRAINED windowed networked PB controller...')
    err_before = plot_rollout_diagnostics(
        net, pb_net, plot_data, args, save_folder,
        tag='before_training', text_prefix='Windowed networked PB controller (untrained)',
    )
    logger.info(f'[before training] mean final-timestep tracking error (over {plot_data.shape[0]} rollouts): {err_before:.4f}')

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
            _, e_log, u_log = net.rollout(train_data_batch, controller=pb_net, train=True)
            loss = loss_fn.forward(e_log, u_log)
            loss.backward()
            optimizer.step()

        if epoch % args.log_epoch == 0:
            msg = 'Epoch: %i --- train loss: %.4f' % (epoch, loss.item())

            if args.return_best:
                with torch.no_grad():
                    _, e_log_valid, u_log_valid = net.rollout(valid_data, controller=pb_net, train=False)
                    loss_valid_parts = loss_fn.components(e_log_valid, u_log_valid)
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
    checkpoint_path = os.path.join(save_folder, 'trained_controller_windowed.pt')
    torch.save(pb_net.state_dict(), checkpoint_path)
    logger.info(f'[INFO] saved trained controller to {checkpoint_path}')

    # ------------ 8. Test-time deployment: reload the saved checkpoint from disk ------------
    pb_net_deployed = PBControllerNetworkWindowed(
        net, dim_internal=args.dim_internal, dim_nl=args.dim_nl,
        horizon=args.horizon, window_fraction=args.window_fraction,
        initialization_std=args.cont_init_std,
    ).to(device)
    pb_net_deployed.load_state_dict(torch.load(checkpoint_path, map_location=device))

    with torch.no_grad():
        _, e_log_test, u_log_test = net.rollout(test_data, controller=pb_net_deployed, train=False)
        test_loss = loss_fn.forward(e_log_test, u_log_test).item()
    logger.info(f'Test loss of the reloaded checkpoint (over {test_data.shape[0]} fresh test rollouts): {test_loss:.4f}')

    # ------------ 9. Plot the reloaded checkpoint on a held-out test rollout ------------
    err_test = plot_rollout_diagnostics(
        net, pb_net_deployed, plot_data, args, save_folder,
        tag='test_deployment', text_prefix='Windowed networked PB controller (trained, reloaded from checkpoint)',
    )
    logger.info(f'[test deployment] mean final-timestep tracking error (over {plot_data.shape[0]} rollouts): {err_test:.4f}')
    logger.info(f'Saved plots to {save_folder}')


if __name__ == "__main__":
    main()
