import torch

from plants import CostumDataset


def generate_positions_with_min_distance(n_points, interval_x1=(-3, 3), interval_x2=(2, 3),
                                          min_distance=1.0, max_tries=20000):
    """
    Vectorized rejection sampling for n_points 2D positions such that all
    pairwise distances exceed min_distance. Generalizes
    plants/robots/robots_dataset.py's generate_vector_with_min_distance (which
    hardcodes exactly 2 points) to N points, reusing the pairwise-squared-
    distance/mask pattern from loss_functions/robots_loss.py::get_pairwise_distance_sq.

    Resamples all N points together each try (simple and robust for the small
    n_points / short interval ranges used here) rather than only the
    violating points.

    Returns:
        flat tensor of shape (2*n_points,): [x1,y1, x2,y2, ...]
    """
    mask = ~torch.eye(n_points, dtype=torch.bool)
    for _ in range(max_tries):
        x1 = torch.empty(n_points).uniform_(interval_x1[0], interval_x1[1])
        x2 = torch.empty(n_points).uniform_(interval_x2[0], interval_x2[1])
        pts = torch.stack((x1, x2), dim=-1)    # (n_points, 2)

        deltaqx = pts[:, 0].unsqueeze(0) - pts[:, 0].unsqueeze(1)
        deltaqy = pts[:, 1].unsqueeze(0) - pts[:, 1].unsqueeze(1)
        distance_sq = deltaqx ** 2 + deltaqy ** 2    # (n_points, n_points)

        if not mask.any() or torch.all(distance_sq[mask] >= min_distance ** 2):
            return pts.flatten()    # (2*n_points,), [x1,y1,x2,y2,...]

    raise RuntimeError(
        f"could not place {n_points} points with pairwise min distance {min_distance} "
        f"in x1={interval_x1}, x2={interval_x2} within {max_tries} tries "
        f"(widen the intervals, lower min_distance, or reduce n_points)"
    )


class NetworkedRobotsDataset(CostumDataset):
    """
    N-agent generalization of plants/robots/robots_dataset.py::RobotsDataset.
    A fixed nominal initial formation (generated once, with pairwise min
    distance) is perturbed per rollout by std_ini Gaussian noise, matching
    the centralized dataset's convention; targets are randomly resampled
    (with pairwise min distance) per rollout, also matching the centralized
    convention.
    """
    def __init__(self, random_seed, horizon, n_agents, std_ini=0.2, min_dist=2.0,
                 x0_interval_x1=(-2, 8), x0_interval_x2=(-2, 2),
                 target_interval_x1=(-2, 8), target_interval_x2=(2, 6)):
        exp_name = 'networked_robots'
        file_name = f'data_T{horizon}_stdini{std_ini}_agents{n_agents}_RS{random_seed}.pkl'
        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.n_agents = n_agents
        self.std_ini = std_ini
        self.min_dist = min_dist
        self.target_interval_x1 = target_interval_x1
        self.target_interval_x2 = target_interval_x2

        # nominal fixed initial formation, generated once - parity with the
        # centralized dataset's fixed x0 + per-rollout std_ini Gaussian noise
        x0_positions = generate_positions_with_min_distance(
            n_agents, interval_x1=x0_interval_x1, interval_x2=x0_interval_x2, min_distance=min_dist,
        )
        self.x0 = torch.zeros(4 * n_agents)
        for i in range(n_agents):
            self.x0[4 * i:4 * i + 2] = x0_positions[2 * i:2 * i + 2]

    # ---- data generation ----
    def _generate_data(self, num_samples):
        state_dim_x0 = 4 * self.n_agents
        state_dim_ref = 4 * self.n_agents
        state_dim = state_dim_x0 + state_dim_ref

        data = torch.zeros(num_samples, self.horizon, state_dim)

        for rollout_num in range(num_samples):
            targets = generate_positions_with_min_distance(
                self.n_agents, interval_x1=self.target_interval_x1,
                interval_x2=self.target_interval_x2, min_distance=self.min_dist,
            )
            data[rollout_num, 0, :state_dim_x0] = self.x0 + self.std_ini * torch.randn(self.x0.shape)
            for i in range(self.n_agents):
                data[rollout_num, 1:, state_dim_x0 + 4 * i: state_dim_x0 + 4 * i + 2] = targets[2 * i:2 * i + 2]

        assert data.shape[0] == num_samples
        return data
