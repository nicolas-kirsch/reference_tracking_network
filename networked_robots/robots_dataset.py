import torch

from plants import CostumDataset

# Sampling configuration of the CENTRALIZED experiment, mirrored here so a
# networked run can draw from the same distributions (NetworkedRobotsDataset's
# central_data flag). Sources: plants/robots/robots_dataset.py::RobotsDataset.x0
# for the formation, and its _generate_data call site for the target rectangle
# and minimum pairwise distance (which override that method's own signature
# defaults - read the call site, not the signature).
CENTRAL_X0 = torch.tensor([4., 0., 0., 0.,      # agent 1 at (4, 0), zero velocity
                            0., 0., 0., 0.])     # agent 2 at (0, 0), zero velocity
CENTRAL_TARGET_INTERVAL_X1 = (-1, 5)
CENTRAL_TARGET_INTERVAL_X2 = (4, 4.1)
CENTRAL_MIN_DIST = 2.0


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

    `central_data=True` swaps this class's own sampling configuration for the
    centralized experiment's (see __init__), so a networked run draws its
    rollouts from the SAME distributions as experiments/robots/run.py.
    """
    def __init__(self, random_seed, horizon, n_agents, std_ini=0.2, min_dist=2.0,
                 x0_interval_x1=(-2, 8), x0_interval_x2=(-2, 2),
                 target_interval_x1=(-2, 8), target_interval_x2=(2, 6),
                 central_data=False):
        """
        Args:
            central_data: when True, sample from the same initial-condition and
                target distributions as the centralized experiment
                (plants/robots/robots_dataset.py::RobotsDataset as configured
                by experiments/robots/run.py), so a networked run is comparable
                to it. Requires n_agents == 2. Individual rollouts are NOT the
                same points as a centralized run's - only the distributions
                they are drawn from match. Concretely this overrides:
                  - the nominal initial formation -> CENTRAL_X0, the
                    centralized dataset's fixed x0 (agent 1 at (4,0), agent 2
                    at (0,0)), still perturbed per rollout by the same
                    std_ini Gaussian noise. The x0_interval_* arguments are
                    ignored, since the centralized x0 is a fixed formation
                    rather than a sampled one.
                  - the target rectangle -> x in [-1,5], y in [4,4.1], and the
                    minimum pairwise target distance -> 2.0, overriding
                    target_interval_* and min_dist.

                The target *distribution* then matches exactly, not just the
                box: for 2 agents, this class's joint rejection sampling
                (generate_positions_with_min_distance) and the centralized
                sequential vec1/vec2 rejection both yield a uniform draw on
                the rectangle squared conditioned on ||p1 - p2|| >= min_dist.
                The RNG is consumed in a different order, which is why the
                realized points differ.
        """
        exp_name = 'networked_robots'
        file_name = f'data_T{horizon}_stdini{std_ini}_agents{n_agents}_RS{random_seed}.pkl'
        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        if central_data:
            assert n_agents == 2, (
                f"central_data=True matches the centralized 2-agent experiment's sampling "
                f"(CENTRAL_X0 has shape (8,)), so it is only defined for n_agents == 2, got {n_agents}"
            )
            min_dist = CENTRAL_MIN_DIST
            target_interval_x1, target_interval_x2 = CENTRAL_TARGET_INTERVAL_X1, CENTRAL_TARGET_INTERVAL_X2

        self.n_agents = n_agents
        self.std_ini = std_ini
        self.min_dist = min_dist
        self.target_interval_x1 = target_interval_x1
        self.target_interval_x2 = target_interval_x2
        self.central_data = central_data

        # nominal fixed initial formation, generated once - parity with the
        # centralized dataset's fixed x0 + per-rollout std_ini Gaussian noise
        if central_data:
            self.x0 = CENTRAL_X0.detach().clone()
        else:
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
