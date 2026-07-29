import torch
import torch.nn as nn

from .robot_pb_controller import RobotPBController


class PBControllerNetwork(nn.Module):
    """
    Networked PB controller: N local RobotPBControllers (M_p x M_infty per
    agent), composed per the adjacency graph stored on `robots_network`.

    Each agent's M_p (REN) sees only its own local reconstructed disturbance
    w_hat^(i) - CLAUDE.md's Objective explicitly marks neighbor w_hat's as
    optional for R_p, deferred for now. Each agent's M_infty (MLP)
    additionally sees its neighbors' PREVIOUS-timestep control actions
    z^(Ni), which is what introduces graph coupling at the controller level.

    The one-step communication delay (required for well-posedness - no
    algebraic loop across the network) is enforced here, not inside
    RobotPBController: every agent's forward() this step reads
    `self.last_z` (all agents' z from step t-1); only after every agent in
    the loop has produced its new z is `self.last_z` overwritten. No agent
    ever sees another agent's *current*-timestep output.

    central_like=True switches every agent's controller into the mode
    described in RobotPBController's docstring: M_p/M_infty see every
    agent's w_hat (and, for M_infty, every agent's xbar) concatenated,
    instead of own_position + z_neighbors - imitating the centralized
    controller's single REN/MLP pair as closely as possible while keeping
    per-agent (locally parametrized) networks and per-agent outputs. This
    requires two passes per step (every agent's w_hat must be gathered
    before any agent's REN/MLP can run), so central_like uses a completely
    separate forward() branch; the central_like=False branch below is the
    original code, unmodified.
    """
    def __init__(self, robots_network, dim_internal: int, dim_nl: int,
                 initialization_std: float = 0.5, posdef_tol: float = 0.001,
                 contraction_rate_lb: float = 1.0, output_amplification: float = 20,
                 central_like: bool = False):
        super().__init__()

        self.n_agents = robots_network.n_agents
        self.AGENT_ETA_DIM = robots_network.AGENT_ETA_DIM
        self.AGENT_IN_DIM = robots_network.AGENT_IN_DIM
        self.AGENT_XBAR_DIM = robots_network.AGENT_XBAR_DIM
        self.in_dim = robots_network.in_dim
        self.central_like = central_like

        self.neighbor_lists = [robots_network.neighbors(i) for i in range(self.n_agents)]
        internal_models = robots_network.internal_models()
        eta_init_full = robots_network.eta_init    # (1, eta_dim)

        self.controllers = nn.ModuleList([
            RobotPBController(
                internal_model=internal_models[i],
                eta_init=eta_init_full[:, i * self.AGENT_ETA_DIM:(i + 1) * self.AGENT_ETA_DIM],
                output_init=torch.zeros(1, self.AGENT_IN_DIM),
                n_neighbors=len(self.neighbor_lists[i]),
                dim_internal=dim_internal, dim_nl=dim_nl,
                initialization_std=initialization_std, posdef_tol=posdef_tol,
                contraction_rate_lb=contraction_rate_lb, output_amplification=output_amplification,
                central_like=central_like, total_n_agents=self.n_agents,
            ) for i in range(self.n_agents)
        ])

        z_init = torch.zeros(1, 1, self.in_dim)
        self.register_buffer('z_init', z_init)

    def reset(self):
        for controller in self.controllers:
            controller.reset()
        self.last_z = self.z_init.detach().clone()

    def forward(self, eta: torch.Tensor, xbar: torch.Tensor):
        """
        Args:
            eta: (batch,1,eta_dim) - full network augmented state (RobotsNetwork layout).
            xbar: (batch,1,xbar_dim) - full network reference.

        Returns:
            z: (batch,1,in_dim) - full network control action (u for RobotsNetwork.rollout).
        """
        batch = eta.shape[0]

        if self.central_like:
            # Phase A: every agent reconstructs its own w_hat locally (dynamics
            # are decoupled, so this never needs another agent's state).
            etas_i, w_hats = [], []
            for i, controller in enumerate(self.controllers):
                eta_i = eta[:, :, i * self.AGENT_ETA_DIM:(i + 1) * self.AGENT_ETA_DIM]
                xbar_i = xbar[:, :, i * self.AGENT_XBAR_DIM:(i + 1) * self.AGENT_XBAR_DIM]
                etas_i.append(eta_i)
                w_hats.append(controller.compute_w_hat(eta_i, xbar_i))
            w_hat_all = torch.cat(w_hats, dim=2)

            # Phase B: every agent's REN/MLP sees the FULL joint w_hat and FULL
            # joint xbar - no own_position, no z_neighbors (no u/z interconnection).
            new_zs = []
            for i, controller in enumerate(self.controllers):
                output_i = controller.compute_output_central_like(w_hat_all, xbar)
                controller.commit(etas_i[i], output_i)
                new_zs.append(output_i)

            z = torch.cat(new_zs, dim=-1)
            self.last_z = z
            return z

        last_z = self.last_z
        if last_z.shape[0] != batch:
            last_z = last_z.expand(batch, -1, -1)

        new_zs = []
        for i, controller in enumerate(self.controllers):
            eta_i = eta[:, :, i * self.AGENT_ETA_DIM:(i + 1) * self.AGENT_ETA_DIM]
            xbar_i = xbar[:, :, i * self.AGENT_XBAR_DIM:(i + 1) * self.AGENT_XBAR_DIM]

            neighbor_idx = self.neighbor_lists[i]
            if neighbor_idx:
                cols = [self.AGENT_IN_DIM * j + c for j in neighbor_idx for c in range(self.AGENT_IN_DIM)]
                z_neighbors_i = last_z[:, :, cols]
            else:
                z_neighbors_i = torch.zeros(batch, 1, 0, device=eta.device, dtype=eta.dtype)

            new_zs.append(controller(eta_i, xbar_i, z_neighbors_i))

        z = torch.cat(new_zs, dim=-1)
        self.last_z = z    # commit only after every agent used the PREVIOUS z - enforces the one-step delay
        return z
