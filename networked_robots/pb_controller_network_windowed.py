import torch
import torch.nn as nn

from .robot_pb_controller_windowed import RobotPBControllerWindowed
from .pb_controller_network import PBControllerNetwork


class PBControllerNetworkWindowed(PBControllerNetwork):
    """
    EXPERIMENTAL variant of PBControllerNetwork: builds RobotPBControllerWindowed
    agents instead of RobotPBController - see that class's docstring for the
    relaxed-guarantee trade-off. forward()/reset() are OWN copies of
    PBControllerNetwork's pre-refactor (single self-contained forward() per
    controller) implementation, not inherited: PBControllerNetwork's
    forward()/reset() evolved to match RobotPBController's joint-w_hat
    reconstruction (compute_w_hat/compute_output/commit), which
    RobotPBControllerWindowed does not implement (it keeps its own
    self-contained forward(), by design - see that class's docstring). This
    class is therefore fully decoupled from PBControllerNetwork's internals
    on purpose, not just today.
    """
    def __init__(self, robots_network, dim_internal: int, dim_nl: int, horizon: int,
                 window_fraction: float = 0.75,
                 initialization_std: float = 0.5, posdef_tol: float = 0.001,
                 contraction_rate_lb: float = 1.0, output_amplification: float = 20):
        nn.Module.__init__(self)    # skip PBControllerNetwork.__init__ - builds a different controller class

        self.n_agents = robots_network.n_agents
        self.AGENT_ETA_DIM = robots_network.AGENT_ETA_DIM
        self.AGENT_IN_DIM = robots_network.AGENT_IN_DIM
        self.AGENT_XBAR_DIM = robots_network.AGENT_XBAR_DIM
        self.in_dim = robots_network.in_dim

        self.neighbor_lists = [robots_network.neighbors(i) for i in range(self.n_agents)]
        internal_models = robots_network.internal_models()
        eta_init_full = robots_network.eta_init    # (1, eta_dim)

        self.controllers = nn.ModuleList([
            RobotPBControllerWindowed(
                internal_model=internal_models[i],
                eta_init=eta_init_full[:, i * self.AGENT_ETA_DIM:(i + 1) * self.AGENT_ETA_DIM],
                output_init=torch.zeros(1, self.AGENT_IN_DIM),
                n_neighbors=len(self.neighbor_lists[i]),
                dim_internal=dim_internal, dim_nl=dim_nl,
                horizon=horizon, window_fraction=window_fraction,
                initialization_std=initialization_std, posdef_tol=posdef_tol,
                contraction_rate_lb=contraction_rate_lb, output_amplification=output_amplification,
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
