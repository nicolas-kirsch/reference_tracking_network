import torch

from .robot_dynamics import RobotDynamics


class RobotsNetwork(torch.nn.Module):
    """
    N-agent network: composes N independent RobotDynamics instances (one per
    agent). Physically decoupled for now - each agent's update depends only
    on its own local state/input/reference - but carries an explicit
    adjacency matrix describing which agents may exchange information, so a
    future networked PB controller (see CLAUDE.md's "Objective") can route
    neighbor signals through it without requiring changes to this class's
    per-agent update logic.

    eta layout convention (per-agent contiguous blocks, chosen so slicing per
    agent for the ModuleList loop is trivial):
        eta = [x_1(4), v_1(2), x_2(4), v_2(2), ..., x_N(4), v_N(2)]
    u, xbar, and w (process noise) are already contiguous per-agent in the
    existing data convention:
        u = [u_1(2), ..., u_N(2)], xbar = [xbar_1(2), ..., xbar_N(2)],
        w = [w_1(4), ..., w_N(4)]
    """
    AGENT_ETA_DIM = 6
    AGENT_STATE_DIM = 4
    AGENT_V_DIM = 2
    AGENT_IN_DIM = 2
    AGENT_XBAR_DIM = 2

    def __init__(self, n_agents, adjacency, linear_plant, k: float = 1.0,
                 mass: float = 1.0, b: float = 1.0, h: float = 0.05, x_init=None):
        super().__init__()

        self.n_agents = n_agents
        self.linear_plant = linear_plant

        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        assert adjacency.shape == (n_agents, n_agents), (
            f"adjacency must be ({n_agents}, {n_agents}), got {tuple(adjacency.shape)}"
        )
        self.register_buffer('adjacency', adjacency)

        # identical nominal per-agent model, matching the current example
        # where every robot shares the same k/mass/b/h
        self.agents = torch.nn.ModuleList([
            RobotDynamics(linear_plant=linear_plant, k=k, mass=mass, b=b, h=h)
            for _ in range(n_agents)
        ])

        self.state_dim = self.AGENT_STATE_DIM * n_agents
        self.in_dim = self.AGENT_IN_DIM * n_agents
        self.v_dim = self.AGENT_V_DIM * n_agents
        self.eta_dim = self.AGENT_ETA_DIM * n_agents
        self.xbar_dim = self.AGENT_XBAR_DIM * n_agents

        x_init = torch.zeros((1, self.state_dim)) if x_init is None else x_init.reshape(1, -1)
        self.register_buffer('x_init', x_init)
        assert self.x_init.shape[1] == self.state_dim

        u_init = torch.zeros(1, self.in_dim)
        self.register_buffer('u_init', u_init)

        eta_init = self._x_v_to_eta(x_init.unsqueeze(1), torch.zeros(1, 1, self.v_dim)).squeeze(1)
        self.register_buffer('eta_init', eta_init)

    def neighbors(self, i):
        """
        Return the neighbor indices of agent i per the adjacency matrix.
        Not used by the dynamics update yet - reserved for the networked
        controller (neighbor ŵ / z routing per CLAUDE.md's Objective).
        """
        return torch.nonzero(self.adjacency[i]).flatten().tolist()

    def internal_models(self):
        """
        Build a fresh, independent RobotDynamics per agent, nominally
        identical to self.agents (mirrors RobotsSystem.internal_model() in
        the centralized code). Intended to be handed to a PBControllerNetwork
        for IMC disturbance reconstruction.
        """
        return torch.nn.ModuleList([
            RobotDynamics(linear_plant=agent.linear_plant, k=agent.k, mass=agent.mass, b=agent.b, h=agent.h)
            for agent in self.agents
        ])

    def _x_v_to_eta(self, x, v):
        """Interleave flat per-agent x (state_dim) and v (v_dim) blocks into the eta layout."""
        *lead, _ = x.shape
        x = x.view(*lead, self.n_agents, self.AGENT_STATE_DIM)
        v = v.view(*lead, self.n_agents, self.AGENT_V_DIM)
        eta = torch.cat((x, v), dim=-1)    # (..., n_agents, 6)
        return eta.reshape(*lead, self.eta_dim)

    def eta_to_x(self, eta):
        """Flatten eta's x-parts across all agents into [x1,y1,vx1,vy1, x2,y2,vx2,vy2, ...]."""
        *lead, _ = eta.shape
        eta = eta.view(*lead, self.n_agents, self.AGENT_ETA_DIM)
        x = eta[..., :self.AGENT_STATE_DIM]
        return x.reshape(*lead, self.state_dim)

    def eta_to_v(self, eta):
        """Flatten eta's v-parts across all agents into [v1x,v1y, v2x,v2y, ...]."""
        *lead, _ = eta.shape
        eta = eta.view(*lead, self.n_agents, self.AGENT_ETA_DIM)
        v = eta[..., self.AGENT_STATE_DIM:]
        return v.reshape(*lead, self.v_dim)

    def noiseless_forward(self, t, eta: torch.Tensor, u: torch.Tensor, xbar: torch.Tensor):
        eta = eta.view(-1, 1, self.eta_dim)
        u = u.view(-1, 1, self.in_dim)
        xbar = xbar.view(-1, 1, self.xbar_dim)

        next_parts = []
        for i, agent in enumerate(self.agents):
            eta_i = eta[:, :, i * self.AGENT_ETA_DIM:(i + 1) * self.AGENT_ETA_DIM]
            u_i = u[:, :, i * self.AGENT_IN_DIM:(i + 1) * self.AGENT_IN_DIM]
            xbar_i = xbar[:, :, i * self.AGENT_XBAR_DIM:(i + 1) * self.AGENT_XBAR_DIM]
            next_parts.append(agent.noiseless_forward(t, eta_i, u_i, xbar_i))
        return torch.cat(next_parts, dim=-1)

    def forward(self, t, eta, u, w, xbar):
        eta = eta.view(-1, 1, self.eta_dim)
        u = u.view(-1, 1, self.in_dim)
        xbar = xbar.view(-1, 1, self.xbar_dim)
        w = w.view(-1, 1, self.state_dim)

        next_parts = []
        for i, agent in enumerate(self.agents):
            eta_i = eta[:, :, i * self.AGENT_ETA_DIM:(i + 1) * self.AGENT_ETA_DIM]
            u_i = u[:, :, i * self.AGENT_IN_DIM:(i + 1) * self.AGENT_IN_DIM]
            xbar_i = xbar[:, :, i * self.AGENT_XBAR_DIM:(i + 1) * self.AGENT_XBAR_DIM]
            w_i = w[:, :, i * self.AGENT_STATE_DIM:(i + 1) * self.AGENT_STATE_DIM]
            next_parts.append(agent.forward(t, eta_i, u_i, w_i, xbar_i))
        return torch.cat(next_parts, dim=-1)

    def rollout(self, data, controller=None, train=False):
        """
        Roll out the network. When `controller` is None, the reference offset
        u is fixed at zero for the whole rollout (base loop alone, no learned
        controller). When given a PBControllerNetwork, u = controller(eta,
        xbar_t) is used each step instead - same eta = plant(...); u =
        controller(eta, xbar_t) pattern as the centralized RobotsSystem.rollout.

        Args:
            - data: tensor of shape (batch_size, T, 8*n_agents). First
              4*n_agents columns are the process noise / initial-state
              perturbation (only t=0 populated); next 4*n_agents columns are
              the reference block (only the x,y sub-columns per agent
              populated, held constant from t=1 onward).

        Returns:
            - x_log of shape (batch_size, T, state_dim), flat per-agent layout
              [x1,y1,vx1,vy1, x2,y2,vx2,vy2, ...] (matches plot_trajectories/
              RobotsLoss's expected layout).
            - e_log of shape (batch_size, T, xbar_dim): tracking error
              xbar - [px,py] per agent.
            - u_log of shape (batch_size, T, in_dim): control action per
              agent, [u1_x,u1_y, u2_x,u2_y, ...] (all zero when controller
              is None).
        """
        if controller is not None:
            controller.reset()

        eta = self.eta_init.detach().clone().repeat(data.shape[0], 1, 1)
        u = self.u_init.detach().clone().repeat(data.shape[0], 1, 1)

        w = data[:, :, :self.state_dim]
        xbar_cols = [4 * i + j for i in range(self.n_agents) for j in (0, 1)]
        xbar = data[:, :, self.state_dim:2 * self.state_dim][:, :, xbar_cols]

        x_position_cols = [4 * i + j for i in range(self.n_agents) for j in (0, 1)]

        for t in range(data.shape[1]):
            eta = self.forward(t=t, eta=eta, u=u, w=w[:, t:t + 1, :], xbar=xbar[:, t:t + 1, :])
            if controller is not None:
                u = controller(eta, xbar[:, t:t + 1, :])
            x = self.eta_to_x(eta)
            e = xbar[:, t:t + 1, :] - x[:, :, x_position_cols]

            if t == 0:
                x_log, e_log, u_log, v_log = x, e, u, self.eta_to_v(eta)
            else:
                x_log = torch.cat((x_log, x), 1)
                e_log = torch.cat((e_log, e), 1)
                u_log = torch.cat((u_log, u), 1)
                v_log = torch.cat((v_log, self.eta_to_v(eta)), 1)

        if controller is not None:
            controller.reset()

        self.v_log = v_log.detach()
        if not train:
            x_log, u_log = x_log.detach(), u_log.detach()
        # e_log is never detached - the loss needs its gradient regardless of train
        return x_log, e_log, u_log
