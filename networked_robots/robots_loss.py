class NetworkedRobotsLoss:
    """
    Networked-robots training loss: tracking error plus an energy
    regularizer on the PB output (the reference offset dxref added to the
    base loop's setpoint), each independently scaled. Mirrors the role of
    loss_functions.robots_loss.RobotsLoss in the centralized code - a single
    loss object instantiated once in the run script and called (via
    .forward()) at every train/valid/test rollout.
    """

    def __init__(self, alpha_tracking=1.0, alpha_energy=0.0):
        self.alpha_tracking = alpha_tracking
        self.alpha_energy = alpha_energy

    def forward(self, e_log, u_log):
        """
        Args:
            e_log: tensor of shape (batch, T, xbar_dim) - per-agent tracking
                error, concatenated (RobotsNetwork.rollout's e_log layout).
            u_log: tensor of shape (batch, T, in_dim) - per-agent PB output,
                concatenated (RobotsNetwork.rollout's u_log layout).

        Returns:
            scalar tensor.
        """
        return self.components(e_log, u_log)['total']

    def components(self, e_log, u_log):
        """
        Same inputs as forward(), but returns the weighted terms making up
        the loss individually, so callers can log the breakdown instead of
        just the combined scalar.

        Returns:
            dict with keys 'tracking', 'energy', 'total' (each a scalar
            tensor); 'tracking' + 'energy' == 'total'.
        """
        tracking = self.alpha_tracking * self.tracking_loss(e_log)
        energy = self.alpha_energy * self.energy_loss(u_log)
        return {'tracking': tracking, 'energy': energy, 'total': tracking + energy}

    @staticmethod
    def tracking_loss(e_log):
        """
        Per timestep, sum every agent's squared distance to its target
        (squared-norm over the full vector already sums every agent's
        contribution, so no explicit per-agent loop is needed); time-average
        over the horizon; batch-average over rollouts.
        """
        per_step = (e_log ** 2).sum(dim=-1)      # (batch, T) - sum over agents
        per_rollout = per_step.mean(dim=-1)      # (batch,) - time-average
        return per_rollout.mean()                # scalar - batch-average

    @staticmethod
    def energy_loss(u_log):
        """
        Per timestep, sum every agent's squared PB-output norm; time-average
        over the horizon; batch-average over rollouts. Penalizing this
        discourages large-magnitude transients in dxref (an unconstrained
        IMC free parameter can otherwise spike early in the rollout without
        this term pulling it down).
        """
        per_step = (u_log ** 2).sum(dim=-1)      # (batch, T) - sum over agents
        per_rollout = per_step.mean(dim=-1)      # (batch,) - time-average
        return per_rollout.mean()                # scalar - batch-average
