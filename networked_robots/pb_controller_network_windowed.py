import torch
import torch.nn as nn

from .robot_pb_controller_windowed import RobotPBControllerWindowed
from .pb_controller_network import PBControllerNetwork


class PBControllerNetworkWindowed(PBControllerNetwork):
    """
    EXPERIMENTAL variant of PBControllerNetwork: builds RobotPBControllerWindowed
    agents instead of RobotPBController - see that class's docstring for the
    relaxed-guarantee trade-off. forward()/reset() are inherited unchanged
    from PBControllerNetwork, since they're already generic over whatever
    controller class populates self.controllers.
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
