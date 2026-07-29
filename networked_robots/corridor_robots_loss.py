import torch

from config import device


class CorridorRobotsLoss:
    """
    Networked-robots corridor-experiment loss. Mirrors
    loss_functions.robots_loss.RobotsLoss (tracking + speed + control effort
    + collision avoidance + Gaussian obstacle avoidance), generalized to N
    agents via the per-agent stride (state_dim_per_agent = 4) already used by
    RobotsNetwork's x_log layout [x1,y1,vx1,vy1, x2,y2,vx2,vy2, ...].
    Companion to run_networked_control_corridor.py - the plain
    NetworkedRobotsLoss in robots_loss.py (tracking + energy only) is left
    untouched for run_networked_control.py / run_networked_control_windowed.py.

    Unlike the centralized RobotsLoss (which takes Q, Qs as full matrices
    built once in the run script), the tracking/speed weights here are plain
    scalars (alpha_track, alpha_speed) expanded internally to
    alpha * kron(eye(n_agents), eye(2)) - every loss weight is then a single
    CLI-exposable float, consistent with alpha_u/alpha_col/alpha_obst.
    NOTE: the centralized RobotsLoss.forward computes its speed term with
    self.Q instead of self.Qs (Qs is stored but never read) - here
    alpha_speed genuinely scales the speed term, independent of alpha_track.
    """

    def __init__(
        self, n_agents, alpha_track=100.0, alpha_speed=1.0, alpha_u=0.1 / 400,
        alpha_col=None, alpha_obst=None, min_dist=1.0,
        obstacle_centers=None, obstacle_covs=None,
        loss_bound=None, sat_bound=None,
    ):
        self.n_agents = n_agents
        self.alpha_u = alpha_u
        self.alpha_col, self.alpha_obst, self.min_dist = alpha_col, alpha_obst, min_dist
        self.loss_bound, self.sat_bound = loss_bound, sat_bound
        assert (self.alpha_col is None and self.min_dist is None) or not (self.alpha_col is None or self.min_dist is None)

        eye2 = torch.kron(torch.eye(n_agents), torch.eye(2)).to(device)
        self.Q = alpha_track * eye2
        self.Qs = alpha_speed * eye2

        # obstacles (same corridor-wall convention as the centralized default)
        if obstacle_centers is None:
            self.obstacle_centers = [
                torch.tensor([[-2.5, 0]], device=device),
                torch.tensor([[2.5, 0.0]], device=device),
                torch.tensor([[-1.5, 0.0]], device=device),
                torch.tensor([[1.5, 0.0]], device=device),
            ]
        else:
            self.obstacle_centers = obstacle_centers
        if obstacle_covs is None:
            self.obstacle_covs = [torch.tensor([[0.2, 0.2]], device=device)] * len(self.obstacle_centers)
        else:
            self.obstacle_covs = obstacle_covs

        self.mask = torch.logical_not(torch.eye(self.n_agents, device=device))    # shape = (n_agents, n_agents)

    def forward(self, xs, us, es):
        """
        Args:
            - xs: tensor of shape (S, T, state_dim) - RobotsNetwork.rollout's x_log.
            - us: tensor of shape (S, T, in_dim) - RobotsNetwork.rollout's u_log.
            - es: tensor of shape (S, T, xbar_dim) - RobotsNetwork.rollout's e_log.

        Return:
            - loss of shape (1, 1).
        """
        return self.components(xs, us, es)['total']

    def components(self, xs, us, es):
        """
        Same inputs as forward(), but returns the weighted terms making up
        the loss individually, so callers can log the breakdown instead of
        just the combined scalar.

        Returns:
            dict with keys 'tracking', 'speed', 'u', 'collision', 'obstacle',
            'total' (each a scalar tensor); the first five sum to 'total'.
        """
        x_batch = xs.reshape(*xs.shape, 1)
        u_batch = us.reshape(*us.shape, 1)
        e_batch = es.reshape(*es.shape, 1)

        speed = self._get_speed(x_batch)

        xTQx = torch.matmul(torch.matmul(e_batch.transpose(-1, -2), self.Q), e_batch)     # (S, T, 1, 1)
        loss_x = torch.sum(xTQx, 1) / e_batch.shape[1]

        sTQs = torch.matmul(torch.matmul(speed.transpose(-1, -2), self.Q), speed)        # (S, T, 1, 1)
        loss_speed = torch.sum(sTQs, 1) / speed.shape[1]

        uTRu = self.alpha_u * torch.matmul(u_batch.transpose(-1, -2), u_batch)            # (S, T, 1, 1)
        loss_u = torch.sum(uTRu, 1) / u_batch.shape[1]

        if self.alpha_col is None:
            loss_ca = torch.zeros_like(loss_x)
        else:
            loss_ca = self.alpha_col * self.f_loss_ca(x_batch)

        if self.alpha_obst is None:
            loss_obst = torch.zeros_like(loss_x)
        else:
            loss_obst = self.alpha_obst * self.f_loss_obst(x_batch)

        loss_val = loss_x + loss_u + loss_ca + loss_obst + loss_speed
        if self.sat_bound is not None:
            loss_val = torch.tanh(loss_val / self.sat_bound)
        if self.loss_bound is not None:
            loss_val = self.loss_bound * loss_val

        total = torch.sum(loss_val, 0) / xs.shape[0]
        parts = {
            'tracking': torch.sum(loss_x, 0) / xs.shape[0],
            'speed': torch.sum(loss_speed, 0) / xs.shape[0],
            'u': torch.sum(loss_u, 0) / xs.shape[0],
            'collision': torch.sum(loss_ca, 0) / xs.shape[0],
            'obstacle': torch.sum(loss_obst, 0) / xs.shape[0],
            'total': total,
        }
        return parts

    def _get_speed(self, x_batched):
        """
        Extract every agent's (vx, vy) and concatenate into a single
        (S, T, 2*n_agents, 1) vector aligned with self.Qs, generalizing the
        centralized code's hardcoded [2,3,6,7] indices to N agents.
        """
        state_dim_per_agent = x_batched.shape[2] // self.n_agents
        vel_idx = [state_dim_per_agent * i + j for i in range(self.n_agents) for j in (2, 3)]
        return x_batched[:, :, vel_idx, :]

    def f_loss_obst(self, x_batched):
        """
        Obstacle avoidance loss.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - obstacle avoidance loss of shape (S, 1, 1).
        """
        qx = x_batched[:, :, 0::4, :]   # x of all agents. shape = (S, T, n_agents, 1)
        qy = x_batched[:, :, 1::4, :]   # y of all agents. shape = (S, T, n_agents, 1)
        q = torch.cat((qx, qy), dim=-1).view(x_batched.shape[0], x_batched.shape[1], 1, -1).squeeze(dim=2)    # (S, T, 2*n_agents)
        for ind, (center, cov) in enumerate(zip(self.obstacle_centers, self.obstacle_covs)):
            if ind == 0:
                loss_obst = normpdf(q, mu=center, cov=cov)   # shape = (S, T)
            else:
                loss_obst += normpdf(q, mu=center, cov=cov)
        loss_obst = loss_obst.sum(1) / loss_obst.shape[1]    # average over time steps. shape = (S,)
        return loss_obst.reshape(-1, 1, 1)

    def f_loss_ca(self, x_batch):
        """
        Collision avoidance loss.

        Args:
            - x_batch: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - collision avoidance loss of shape (S, 1, 1).
        """
        min_sec_dist = self.min_dist + 0.2
        distance_sq = self.get_pairwise_distance_sq(x_batch)    # (S, T, n_agents, n_agents)
        loss_ca = (1 / (distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2)) * self.mask).sum((-1, -2)) / 2   # (S, T)
        loss_ca = loss_ca.sum(1) / loss_ca.shape[1]
        return loss_ca.reshape(-1, 1, 1)

    def count_collisions(self, x_batch):
        """
        Count the number of collisions between agents.

        Args:
            - x_batch: tensor of shape (S, T, state_dim) or (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - number of collisions between agents.
        """
        if len(x_batch.shape) == 3:
            x_batch = x_batch.reshape(*x_batch.shape, 1)
        distance_sq = self.get_pairwise_distance_sq(x_batch)
        col_matrix = (0.0001 < distance_sq) * (distance_sq < self.min_dist ** 2)
        n_coll = col_matrix.sum().item()
        return n_coll / 2    # each collision is counted twice

    def get_pairwise_distance_sq(self, x_batch):
        """
        Squared distance between pairwise agents.

        Args:
            - x_batch: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - matrix of shape (S, T, n_agents, n_agents) of squared pairwise distances.
        """
        state_dim_per_agent = int(x_batch.shape[2] / self.n_agents)
        x_agents = x_batch[:, :, 0::state_dim_per_agent, :]    # (S, T, n_agents, 1)
        y_agents = x_batch[:, :, 1::state_dim_per_agent, :]    # (S, T, n_agents, 1)
        deltaqx = x_agents.repeat(1, 1, 1, self.n_agents) - x_agents.repeat(1, 1, 1, self.n_agents).transpose(-2, -1)
        deltaqy = y_agents.repeat(1, 1, 1, self.n_agents) - y_agents.repeat(1, 1, 1, self.n_agents).transpose(-2, -1)
        return deltaqx ** 2 + deltaqy ** 2


def normpdf(q, mu, cov):
    """
    PDF of normal distribution with mean "mu" and covariance "cov", evaluated
    per-agent and summed - identical to loss_functions.robots_loss.normpdf.

    Args:
        - q: shape (S, T, 2*n_agents), [x1,y1,x2,y2,...].
        - mu: shape (1, 2).
        - cov: shape (1, 2) (diagonal of the covariance matrix).
    """
    d = 2
    mu = mu.view(1, d)
    cov = cov.view(1, d)
    qs = torch.split(q, 2, dim=-1)
    for ind, qi in enumerate(qs):
        den = (2 * torch.pi) ** (0.5 * d) * torch.sqrt(torch.prod(cov))
        nom = torch.exp((-0.5 * (qi - mu) ** 2 / cov).sum(-1))
        if ind == 0:
            out = nom / den
        else:
            out += nom / den
    return out
