import numpy as np
from scipy.signal import place_poles

import torch
import torch.nn.functional as F


class RobotDynamics(torch.nn.Module):
    """
    Pure closed-loop dynamics of a single robot's pre-stabilized base tracking
    loop (state feedback K_p + integral action K_i), acting on its own local
    augmented state eta = (x, v): x is the robot's own state (px, py, vx, vy),
    v is its own integrator state (2D, integral of the px,py tracking error).

    This is the single-agent building block composed by RobotsNetwork's
    ModuleList. It is physically decoupled: it never sees another agent's
    state. Mirrors plants/robots/robots_sys.py's RobotsDynamics, but for
    exactly one agent (no kron block-diagonal replication needed) - this is
    literally the pre-kron single-agent block already computed inside that
    file's RobotsSystem.__init__.
    """
    def __init__(self, linear_plant: bool, k: float = 1.0, mass: float = 1.0, b: float = 1.0, h: float = 0.05):
        super().__init__()

        self.linear_plant = linear_plant
        self.state_dim = 4
        self.in_dim = 2
        self.v_dim = 2
        self.eta_dim = self.state_dim + self.v_dim

        self.h = h
        self.mass = mass
        self.k = k
        self.b = b
        # kept as 0.0 (not None) when linear, so mask_tanh below stays well-defined
        # even though it's unused in the linear_plant branch of noiseless_forward.
        self.b2 = 0.0 if linear_plant else 0.1
        m = self.mass

        # A and A2 matrices for a single agent
        A_ = np.eye(4)
        A2 = np.block([[np.zeros((2, 2)), np.eye(2)],
                    [np.diag([-k/m, -k/m]), np.diag([-b/m, -b/m])]])
        A = A_ + h * A2

        # B matrix
        B_np = np.array([[0, 0], [0, 0], [h, 0], [0, h]])

        # Output matrix C tracking (px, py)
        C = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        # Augmented system (including integral states)
        A_aug = np.block([[A, np.zeros((4, 2))], [-C, np.eye(2)]])
        B_aug = np.vstack([B_np, np.zeros((2, 2))])

        B = torch.tensor([[0, 0],
                           [0., 0],
                           [1 / m, 0],
                           [0, 1 / m]]) * h
        self.register_buffer('B', B)

        A_lin = torch.eye(4) + h * torch.cat((
            torch.cat((torch.zeros(2, 2), torch.eye(2)), dim=1),
            torch.cat((
                torch.diag(torch.tensor([-self.k / self.mass, -self.k / self.mass])),
                torch.diag(torch.tensor([-self.b / self.mass, -self.b / self.mass])),
            ), dim=1),
        ), dim=0)
        self.register_buffer('A_lin', A_lin)

        # desired poles for the augmented (state + integrator) system - same as
        # the centralized model's per-agent pole placement
        desired_poles = np.exp(np.array([-3, -3.05, -4, -4.05, -2, -2.05]) * h)
        place_obj = place_poles(A_aug, B_aug, desired_poles)
        K_aug = place_obj.gain_matrix

        K_p = torch.tensor(K_aug[:, :4].astype(np.float32))
        K_i = torch.tensor(K_aug[:, 4:].astype(np.float32))
        self.register_buffer('K_p', K_p)
        self.register_buffer('K_i', K_i)

        mask_tanh = torch.tensor([[0, 0],
                                   [0., 0],
                                   [-self.b2 / m, 0],
                                   [0, -self.b2 / m]]) * h
        self.register_buffer('mask_tanh', mask_tanh)

    def noiseless_forward(self, t, eta: torch.Tensor, u: torch.Tensor, xbar: torch.Tensor):
        """
        Pure closed-loop transition of the noise-free single-agent dynamics.

        Args:
            - eta (torch.Tensor): augmented state (x, v) at t. shape = (batch_size, 1, eta_dim=6)
            - u (torch.Tensor): reference offset dxref at t. shape = (batch_size, 1, in_dim=2)
            - xbar (torch.Tensor): nominal reference (target px, py) at t. shape = (batch_size, 1, 2)

        Returns:
            eta at t+1 (single tensor, no process noise added).
        """
        eta = eta.view(-1, 1, self.eta_dim)
        x = eta[:, :, :self.state_dim]
        v = eta[:, :, self.state_dim:]
        dxref = u.view(-1, 1, self.in_dim)

        e = (xbar + dxref) - x[:, :, [0, 1]]

        # integrator is updated before computing u_base (uses v_plus, not v)
        v_plus = v + e

        u_base = -F.linear(x, self.K_p) - F.linear(v_plus, self.K_i)

        if self.linear_plant:
            f = F.linear(x, self.A_lin) + F.linear(u_base, self.B)
        else:
            tanh_q = torch.tanh(x[:, :, [2, 3]])
            f = F.linear(x, self.A_lin) + F.linear(tanh_q, self.mask_tanh) + F.linear(u_base, self.B)

        return torch.cat((f, v_plus), dim=-1)    # shape = (batch_size, 1, eta_dim)

    def forward(self, t, eta, u, w, xbar):
        """
        Forward of this agent's dynamics with process noise.

        Args:
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim=4).
              Only ever enters the plant state x, never the integrator state v.
        """
        eta_next = self.noiseless_forward(t, eta, u, xbar)

        w = w.view(-1, 1, self.state_dim)
        w_padded = torch.cat((w, torch.zeros(w.shape[0], 1, self.v_dim, device=w.device, dtype=w.dtype)), dim=-1)

        return eta_next + w_padded
