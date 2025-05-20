import torch
from config import device

def distributed_rollout(sys1, sys2, ctl1, ctl2, data1, data2, train=False, device=device):
    """
    Distributed rollout for two robots with explicit communication.
    Args:
        sys1, sys2: RobotsSystem instances (n_agents=1)
        ctl1, ctl2: PerfBoostController instances (n_agents=1)
        data1, data2: (batch, T, state_dim+ref_dim) input data for each robot
        train: bool, if True, do not detach logs
        device: torch.device
    Returns:
        x1_log, x2_log, u1_log, u2_log, e1_log, e2_log
    """
    ctl1.reset()
    ctl2.reset()

    batch_size = data1.shape[0]
    T = data1.shape[1]

    # Initial states, controls, integrals
    x1 = sys1.x_init.detach().clone().repeat(batch_size, 1, 1).to(device)
    x2 = sys2.x_init.detach().clone().repeat(batch_size, 1, 1).to(device)
    u1 = sys1.u_init.detach().clone().repeat(batch_size, 1, 1).to(device)
    u2 = sys2.u_init.detach().clone().repeat(batch_size, 1, 1).to(device)
    v1 = torch.zeros_like(u1)
    v2 = torch.zeros_like(u2)

    # Process noise (if any)
    w1 = data1[:, :, :4]
    w2 = data2[:, :, :4]
    xbar1 = data1[:, :, 4:]
    xbar2 = data2[:, :, 4:]

    # Logs
    x1_log, x2_log = [], []
    u1_log, u2_log = [], []
    v1_log, v2_log = [], []
    e1_log, e2_log = [], []

    for t in range(T):
        # Each robot gets the other's position
        p1 = x1[..., :2]  # (batch, 1, 2)
        p2 = x2[..., :2]

        # Controller computes control using neighbor's position
        u1 = ctl1(x1, v1, xbar1[:, t:t+1, :2], neighbor_pos=p2)
        u2 = ctl2(x2, v2, xbar2[:, t:t+1, :2], neighbor_pos=p1)

        # Step each system
        x1, v1 = sys1.forward(t, x1, v1, u1, w=w1[:, t:t+1, :], xbar=xbar1[:, t:t+1, :])
        x2, v2 = sys2.forward(t, x2, v2, u2, w=w2[:, t:t+1, :], xbar=xbar2[:, t:t+1, :])

        # Log
        x1_log.append(x1)
        x2_log.append(x2)
        u1_log.append(u1)
        u2_log.append(u2)
        v1_log.append(v1)
        v2_log.append(v2)
        e1_log.append(xbar1[:, t:t+1, :2] - x1[:, :, :2])
        e2_log.append(xbar2[:, t:t+1, :2] - x2[:, :, :2])

    # Stack logs
    x1_log = torch.cat(x1_log, dim=1)
    x2_log = torch.cat(x2_log, dim=1)
    u1_log = torch.cat(u1_log, dim=1)
    u2_log = torch.cat(u2_log, dim=1)
    v1_log = torch.cat(v1_log, dim=1)
    v2_log = torch.cat(v2_log, dim=1)
    e1_log = torch.cat(e1_log, dim=1)
    e2_log = torch.cat(e2_log, dim=1)

    ctl1.reset()
    ctl2.reset()
    if not train:
        x1_log, u1_log, x2_log, u2_log = x1_log.detach(), u1_log.detach(), x2_log.detach(), u2_log.detach()

    return x1_log, x2_log, u1_log, u2_log, e1_log, e2_log