import torch
import numpy as np
from config import device


def to_tensor(x):
    return torch.from_numpy(x).contiguous().float().to(device) if isinstance(x, np.ndarray) else x

def compute_distance_metric(x_log, data, n_agents):
    """
    Compute the average ratio of bird flight distance to actual distance traveled
    for all agents over all rollouts in the batch.

    Args:
        x_log (torch.Tensor): The trajectory of shape (batch_size, Time, state_dim).
        data (torch.Tensor): The data of shape (batch_size, Time, state_dim), where
                             the first time step is x0 and the last is x_bar.
        n_agents (int): The number of agents.

    Returns:
        float: The average metric (bird flight distance / actual distance traveled)
               over all rollouts in the batch.
    """
    batch_size, T, state_dim = x_log.shape
    metrics = []

    for batch in range(batch_size):
        batch_metrics = []
        for agent in range(n_agents):
            # Extract x and y positions for the agent
            x_positions = x_log[batch, :, 4 * agent].cpu().numpy()  # Shape: (Time,)
            y_positions = x_log[batch, :, 4 * agent + 1].cpu().numpy()  # Shape: (Time,)

            # Extract x0 and x_bar for the agent
            x0_x = data[batch, 0, 4 * agent].cpu().numpy()  # Initial x position
            x0_y = data[batch, 0, 4 * agent + 1].cpu().numpy()  # Initial y position
            x_bar_x = data[batch, -1, 4 * agent].cpu().numpy()  # Target x position
            x_bar_y = data[batch, -1, 4 * agent + 1].cpu().numpy()  # Target y position

            # Compute actual distance traveled
            actual_distance = sum(
                ((x_positions[t+1] - x_positions[t])**2 + (y_positions[t+1] - y_positions[t])**2)**0.5
                for t in range(len(x_positions) - 1)
            )

            # Compute bird flight distance
            bird_flight_distance = ((x_bar_x - x0_x)**2 + (x_bar_y - x0_y)**2)**0.5

            # Compute the metric for this agent
            if actual_distance > 0:
                metric = bird_flight_distance / actual_distance
            else:
                metric = float('inf')  # Handle the case where actual distance is zero

            batch_metrics.append(metric)

        # Compute the average metric for this batch
        metrics.append(sum(batch_metrics) / len(batch_metrics))

    # Return the average metric over all batches
    return sum(metrics) / len(metrics)

class WrapLogger():
    def __init__(self, logger, verbose=True):
        self.can_log = (logger is not None)
        self.logger = logger
        self.verbose = verbose

    def info(self, msg):
        if self.can_log:
            self.logger.info(msg)
        if self.verbose:
            print(msg)

    def close(self):
        if not self.can_log:
            return
        while len(self.logger.handlers):
            h = self.logger.handlers[0]
            h.close()
            self.logger.removeHandler(h)
