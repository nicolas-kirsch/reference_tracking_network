import torch
import torch.nn as nn

from .robot_pb_controller import RobotPBController


class PBControllerNetwork(nn.Module):
    """
    Networked PB controller: ONE rPB disturbance-reconstruction step for the
    whole network, followed by N interconnected M operators (RobotPBController
    instances) composed per the adjacency graph stored on `robots_network` -
    "one rPB, N interconnected M's", not N interconnected mini-rPBs each
    redoing their own piece of the reconstruction.

    The reconstruction (_reconstruct()) computes w_hat = eta - F_hat(last_eta,
    last_output, xbar) once per step. It necessarily loops per agent
    internally (self.internal_models, one RobotDynamics per agent) because
    the underlying plant dynamics are decoupled - exactly mirroring how
    RobotsNetwork.forward() itself loops per agent internally while
    presenting a single joint dynamics step - but this is organizationally
    ONE step of the network's controller, not N separate reconstructions:
    this class, not RobotPBController, owns the internal models and all the
    eta/output/t recurrent bookkeeping the reconstruction needs.

    Every agent's M_p (REN) sees the resulting joint w_hat - CLAUDE.md's
    Objective explicitly allows this ("local w_hat and optionally neighbors'
    w_hat - communicated, still exogenous, so R_p stays feedforward, no
    loop"). Each agent's M_infty (MLP) additionally sees its neighbors'
    CURRENT positions and targets (undelayed - sensed state and given
    reference, not another controller's output, so no algebraic-loop
    concern) and, in 'base' mode only, neighbors' PREVIOUS-timestep control
    actions z^(Ni) (delayed - this IS another controller's output), which is
    what introduces graph coupling at the M-operator level (see `mode` below).

    The one-step communication delay for z (required for well-posedness -
    no algebraic loop across the network) is enforced here: forward() this
    step reads `self.last_z` (all agents' z from step t-1); only after every
    agent has produced its new z is `self.last_z` overwritten. No agent ever
    sees another agent's *current*-timestep output. Neighbor positions/xbars
    (and w_hat) carry no such restriction and are computed/read fresh every
    step.

    `mode` selects every agent's M_infty input composition - see
    RobotPBController's docstring for the full 3-way breakdown ('base',
    'no_interconnection', 'central_like'). The reconstruction and M_p's
    input (the joint w_hat) are unaffected by mode - only M_infty's
    composition differs.

    position_scale is forwarded to every RobotPBController - see its
    docstring for the fixed-constant input normalization this controls.
    """
    def __init__(self, robots_network, dim_internal: int, dim_nl: int,
                 initialization_std: float = 0.5, posdef_tol: float = 0.001,
                 contraction_rate_lb: float = 1.0, output_amplification: float = 20,
                 mode: str = 'base', position_scale: float = 5.0,
                 imc_tol: float = 1e-4):
        super().__init__()

        self.n_agents = robots_network.n_agents
        self.AGENT_ETA_DIM = robots_network.AGENT_ETA_DIM
        self.AGENT_STATE_DIM = robots_network.AGENT_STATE_DIM
        self.AGENT_IN_DIM = robots_network.AGENT_IN_DIM
        self.AGENT_XBAR_DIM = robots_network.AGENT_XBAR_DIM
        self.AGENT_POSITION_DIM = 2    # (px, py) - the first 2 entries of each agent's eta block
        self.in_dim = robots_network.in_dim
        self.mode = mode
        self.imc_tol = imc_tol

        self.neighbor_lists = [robots_network.neighbors(i) for i in range(self.n_agents)]
        # THE network's own reconstruction mechanism - one RobotDynamics per agent
        # (decoupled dynamics), owned here rather than by the M operators.
        self.internal_models = robots_network.internal_models()

        self.controllers = nn.ModuleList([
            RobotPBController(
                state_dim=self.AGENT_STATE_DIM, in_dim=self.AGENT_IN_DIM,
                xbar_dim=self.AGENT_XBAR_DIM, position_dim=self.AGENT_POSITION_DIM,
                n_neighbors=len(self.neighbor_lists[i]),
                dim_internal=dim_internal, dim_nl=dim_nl,
                initialization_std=initialization_std, posdef_tol=posdef_tol,
                contraction_rate_lb=contraction_rate_lb, output_amplification=output_amplification,
                mode=mode, total_n_agents=self.n_agents, position_scale=position_scale,
            ) for i in range(self.n_agents)
        ])

        self.register_buffer('eta_init', robots_network.eta_init.detach().clone().unsqueeze(1))    # (1,1,eta_dim)
        self.register_buffer('output_init', torch.zeros(1, 1, self.in_dim))
        self.register_buffer('z_init', torch.zeros(1, 1, self.in_dim))

    def reset(self):
        for controller in self.controllers:
            controller.reset()
        self.t = 0
        self.last_eta = self.eta_init.detach().clone()
        self.last_output = self.output_init.detach().clone()
        self.last_w_hat_v = None    # v-components of the last reconstruction, all agents concatenated (IMC check)
        self.last_z = self.z_init.detach().clone()

    def _reconstruct(self, eta: torch.Tensor, xbar: torch.Tensor):
        """
        THE network's single rPB disturbance-reconstruction step for this
        timestep: w_hat^(i) = eta^(i) - F_hat^(i)(last_eta^(i), last_output^(i),
        xbar^(i)) for every agent i, looped internally only because the
        dynamics are decoupled (see class docstring), then concatenated.

        Returns:
            w_hats: list of each agent's own w_hat (state part), for
                M_infty's local term. Each (batch,1,AGENT_STATE_DIM).
            w_hat_all: every agent's w_hat concatenated, for every M_p.
                (batch,1,AGENT_STATE_DIM*n_agents).
        """
        w_hats, w_hat_vs = [], []
        for i, model in enumerate(self.internal_models):
            eta_i = eta[:, :, i * self.AGENT_ETA_DIM:(i + 1) * self.AGENT_ETA_DIM]
            xbar_i = xbar[:, :, i * self.AGENT_XBAR_DIM:(i + 1) * self.AGENT_XBAR_DIM]
            last_eta_i = self.last_eta[:, :, i * self.AGENT_ETA_DIM:(i + 1) * self.AGENT_ETA_DIM]
            last_output_i = self.last_output[:, :, i * self.AGENT_IN_DIM:(i + 1) * self.AGENT_IN_DIM]

            eta_noiseless = model.noiseless_forward(t=self.t, eta=last_eta_i, u=last_output_i, xbar=xbar_i)
            w_hat_i = eta_i - eta_noiseless
            w_hats.append(w_hat_i[:, :, :self.AGENT_STATE_DIM])
            w_hat_vs.append(w_hat_i[:, :, self.AGENT_STATE_DIM:])

        self.last_w_hat_v = torch.cat(w_hat_vs, dim=2)
        w_hat_all = torch.cat(w_hats, dim=2)
        return w_hats, w_hat_all

    def forward(self, eta: torch.Tensor, xbar: torch.Tensor):
        """
        Args:
            eta: (batch,1,eta_dim) - full network augmented state (RobotsNetwork layout).
            xbar: (batch,1,xbar_dim) - full network reference.

        Returns:
            z: (batch,1,in_dim) - full network control action (u for RobotsNetwork.rollout).
        """
        batch = eta.shape[0]
        w_hats, w_hat_all = self._reconstruct(eta, xbar)

        new_zs = []
        if self.mode == 'central_like':
            for controller in self.controllers:
                new_zs.append(controller.compute_output_central_like(w_hat_all, xbar))
        elif self.mode == 'no_interconnection':
            for i, controller in enumerate(self.controllers):
                eta_i = eta[:, :, i * self.AGENT_ETA_DIM:(i + 1) * self.AGENT_ETA_DIM]
                xbar_i = xbar[:, :, i * self.AGENT_XBAR_DIM:(i + 1) * self.AGENT_XBAR_DIM]
                own_position_i = eta_i[:, :, :self.AGENT_POSITION_DIM]

                neighbor_idx = self.neighbor_lists[i]
                if neighbor_idx:
                    pos_cols = [self.AGENT_ETA_DIM * j + c for j in neighbor_idx for c in range(self.AGENT_POSITION_DIM)]
                    neighbor_positions_i = eta[:, :, pos_cols]
                    xbar_cols = [self.AGENT_XBAR_DIM * j + c for j in neighbor_idx for c in range(self.AGENT_XBAR_DIM)]
                    neighbor_xbars_i = xbar[:, :, xbar_cols]
                else:
                    neighbor_positions_i = torch.zeros(batch, 1, 0, device=eta.device, dtype=eta.dtype)
                    neighbor_xbars_i = torch.zeros(batch, 1, 0, device=eta.device, dtype=eta.dtype)

                output_i = controller.compute_output_no_interconnection(
                    own_position=own_position_i, xbar_t=xbar_i, w_hat_own=w_hats[i], w_hat_all=w_hat_all,
                    neighbor_positions=neighbor_positions_i, neighbor_xbars=neighbor_xbars_i,
                )
                new_zs.append(output_i)
        else:
            assert self.mode == 'base', f"unknown mode '{self.mode}'"
            last_z = self.last_z
            if last_z.shape[0] != batch:
                last_z = last_z.expand(batch, -1, -1)

            for i, controller in enumerate(self.controllers):
                eta_i = eta[:, :, i * self.AGENT_ETA_DIM:(i + 1) * self.AGENT_ETA_DIM]
                xbar_i = xbar[:, :, i * self.AGENT_XBAR_DIM:(i + 1) * self.AGENT_XBAR_DIM]
                own_position_i = eta_i[:, :, :self.AGENT_POSITION_DIM]

                neighbor_idx = self.neighbor_lists[i]
                if neighbor_idx:
                    pos_cols = [self.AGENT_ETA_DIM * j + c for j in neighbor_idx for c in range(self.AGENT_POSITION_DIM)]
                    neighbor_positions_i = eta[:, :, pos_cols]
                    xbar_cols = [self.AGENT_XBAR_DIM * j + c for j in neighbor_idx for c in range(self.AGENT_XBAR_DIM)]
                    neighbor_xbars_i = xbar[:, :, xbar_cols]
                    cols = [self.AGENT_IN_DIM * j + c for j in neighbor_idx for c in range(self.AGENT_IN_DIM)]
                    z_neighbors_i = last_z[:, :, cols]
                else:
                    neighbor_positions_i = torch.zeros(batch, 1, 0, device=eta.device, dtype=eta.dtype)
                    neighbor_xbars_i = torch.zeros(batch, 1, 0, device=eta.device, dtype=eta.dtype)
                    z_neighbors_i = torch.zeros(batch, 1, 0, device=eta.device, dtype=eta.dtype)

                output_i = controller.compute_output(
                    own_position=own_position_i, xbar_t=xbar_i, w_hat_own=w_hats[i], w_hat_all=w_hat_all,
                    neighbor_positions=neighbor_positions_i, neighbor_xbars=neighbor_xbars_i,
                    z_neighbors=z_neighbors_i,
                )
                new_zs.append(output_i)

        z = torch.cat(new_zs, dim=-1)
        self.last_eta, self.last_output, self.last_z = eta, z, z    # commit only after every agent used the PREVIOUS z
        self.t += 1
        return z

    def check_imc_exactness(self, tol=None):
        """
        Assert that the v-components of the last reconstruction are ~0
        across every agent, as required by exact IMC reconstruction in the
        nominal (no model-mismatch) case (process noise only ever enters the
        plant state, never the integrator state).
        """
        tol = self.imc_tol if tol is None else tol
        assert self.last_w_hat_v is not None, "no forward() call yet to check"
        max_dev = torch.max(torch.abs(self.last_w_hat_v)).item()
        assert max_dev < tol, f"IMC reconstruction violated: max |w_hat_v| = {max_dev} >= {tol}"
        return max_dev
