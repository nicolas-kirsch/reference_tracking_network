import numpy as np
from scipy.linalg import block_diag
from scipy.signal import place_poles

import torch
import torch.nn.functional as F


class RobotsDynamics(torch.nn.Module):
    """
    Pure closed-loop dynamics of the base tracking loop (plant + pre-stabilizing
    state feedback K_p / integral action K_i), acting on the augmented state
    eta = (x, v) where x is the plant state and v is the integrator state.

    This module owns its own copy of the model parameters, so it can be
    instantiated independently for the plant itself and for a controller's
    internal model (IMC reconstruction), enabling model-mismatch experiments.
    """
    def __init__(self, n_agents, state_dim, in_dim, v_dim,
                 A_lin, B, K_p, K_i, mask_tanh, mask, linear_plant, mass, b2):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.in_dim = in_dim
        self.v_dim = v_dim
        self.eta_dim = state_dim + v_dim
        self.linear_plant = linear_plant
        self.mass = mass
        self.b2 = b2

        self.register_buffer('A_lin', A_lin.detach().clone())
        self.register_buffer('B', B.detach().clone())
        self.register_buffer('K_p', K_p.detach().clone())
        self.register_buffer('K_i', K_i.detach().clone())
        self.register_buffer('mask_tanh', mask_tanh.detach().clone())
        self.register_buffer('mask', mask.detach().clone())

    def noiseless_forward(self, t, eta: torch.Tensor, u: torch.Tensor, xbar: torch.Tensor):
        """
        Pure closed-loop transition of the noise-free dynamics.

        Args:
            - eta (torch.Tensor): augmented state (x, v) at t. shape = (batch_size, 1, eta_dim)
            - u (torch.Tensor): reference offset dxref at t. shape = (batch_size, 1, in_dim)
            - xbar (torch.Tensor): nominal reference at t. shape = (batch_size, 1, in_dim)

        Returns:
            eta at t+1 (single tensor, no process noise added).
        """
        eta = eta.view(-1, 1, self.eta_dim)
        x = eta[:, :, :self.state_dim]
        v = eta[:, :, self.state_dim:]
        dxref = u.view(-1, 1, self.in_dim)

        e = (xbar + dxref) - x[:, :, [0, 1, 4, 5]]

        # integrator is updated before computing u_base (uses v_plus, not v)
        v_plus = v + e

        u_base = -F.linear(x, self.K_p) - F.linear(v_plus, self.K_i)

        if self.linear_plant:
            f = F.linear(x, self.A_lin) + F.linear(u_base, self.B)
        else:
            tanh_q = torch.tanh(x[:, :, [2, 3, 6, 7]])
            f = F.linear(x, self.A_lin) + F.linear(tanh_q, self.mask_tanh) + F.linear(u_base, self.B)

        return torch.cat((f, v_plus), dim=-1)    # shape = (batch_size, 1, eta_dim)


class RobotsSystem(torch.nn.Module):
    def __init__(self, linear_plant: bool, x_init=None, u_init=None, k: float=1.0, n_agents: int = 2):
        """

        Args:
            xbar: concatenated nominal initial point of all agents
            linear_plant: if True, a linearized model of the system is used.
                             O.w., the model is non-lineardue to the dependence of friction on the speed.
            x_init: concatenated initial point of all agents. Defaults to xbar when None.
            u_init: initial input to the plant. Defaults to zero when None.
            k (float): gain of the pre-stabilizing controller (acts as a spring constant).
        """
        super().__init__()

        self.linear_plant = linear_plant



        # check dimensions
        self.n_agents = n_agents
        self.state_dim = 4*self.n_agents
        self.in_dim = 2*self.n_agents
        self.v_dim = 2*self.n_agents
        self.eta_dim = self.state_dim + self.v_dim

        # initial state
        x_init = torch.zeros((1,self.state_dim)) if x_init is None else x_init.reshape(1, -1) # shape = (1, state_dim)
        self.register_buffer('x_init', x_init)
        u_init = torch.zeros(1, int(self.x_init.shape[1]/2)) if u_init is None else u_init.reshape(1, -1)   # shape = (1, in_dim)
        self.register_buffer('u_init', u_init)
        eta_init = torch.cat((self.x_init, torch.zeros(1, self.v_dim)), dim=-1)   # shape = (1, eta_dim)
        self.register_buffer('eta_init', eta_init)

        assert self.x_init.shape[1] == self.state_dim
        assert self.u_init.shape[1] == self.in_dim

        self.h = 0.05
        self.mass = 1.0
        self.k = k
        self.b = 1.0
        self.b2 = None if self.linear_plant else 0.1
        m = self.mass

        # Define A and A2 matrices
        A_ = np.eye(4)
        A2 = np.block([[np.zeros((2, 2)), np.eye(2)],
                    [np.diag([-k/m, -k/m]), np.diag([-self.b/m, -self.b/m])]])

        A = A_ + self.h * A2  # Compute A matrix

        # Define B matrix
        B = np.array([[0, 0], [0, 0], [self.h, 0], [0, self.h]])


        # Define output matrix C to track x_0, x_1, x_4, x_5
        C = np.array([[1, 0, 0, 0],   # Track x_0 for agent 1
                    [0, 1, 0, 0]])  # Track x_5 for agent 2

        # Augmented system (including integral states)
        A_aug = np.block([[A, np.zeros((4, 2))], [-C, np.eye(2)]])
        B_aug = np.vstack([B, np.zeros((2, 2))])


        B = torch.kron(torch.eye(self.n_agents),
                       torch.tensor([[0, 0],
                                     [0., 0],
                                     [1/m, 0],
                                     [0, 1/m]])
                       ) * self.h
        """k_i = 0
        self.K_i = torch.kron(torch.eye(self.n_agents),
                       torch.tensor([[0, 0],
                                     [0., 0],
                                     [k_i, 0],
                                     [0, k_i]])
                       ) * self.h"""
        self.register_buffer('B', B)

        _A1 = torch.eye(4*self.n_agents)
        _A2 = torch.cat((torch.cat((torch.zeros(2,2),
                                torch.eye(2)
                                ), dim=1),
                        torch.cat((torch.diag(torch.tensor([-self.k/self.mass, -self.k/self.mass])),
                                   torch.diag(torch.tensor([-self.b/self.mass, -self.b/self.mass]))
                                   ), dim=1),
                        ), dim=0)
        _A2 = torch.kron(torch.eye(self.n_agents), _A2)
        A_lin = _A1 + self.h * _A2



        self.register_buffer('A_lin', A_lin)



        # Choose desired poles for the augmented system
        desired_poles = np.exp(np.array([-3, -3.05, -4, -4.05, -2, -2.05]) * self.h)
        #desired_poles = np.exp(np.array([-3, -3.05, -5, -5.05, -7, -7.05]) * self.h)

        # Compute the state feedback gains (K) and integral gains (K_I) using pole placement
        # Using scipy's place_poles function
        place_obj = place_poles(A_aug, B_aug, desired_poles)
        K_aug = place_obj.gain_matrix

        # Extract state feedback gains and integral gains
        K_p = K_aug[:, :4].astype(np.float32)  # State feedback gains
        K_i = K_aug[:, 4:].astype(np.float32)  # Integral gains
        K_p = torch.kron(torch.eye(self.n_agents), torch.tensor(K_p))
        K_i = torch.kron(torch.eye(self.n_agents), torch.tensor(K_i))

        self.register_buffer('K_p', K_p)
        self.register_buffer('K_i', K_i)



        # Output the results
        """print("State feedback gains (K_p):")
        print(self.K_p)
        print("\nIntegral gains (K_i):")
        print(self.K_i)"""

        mask = torch.tensor([[0, 0], [1, 1]]).repeat(self.n_agents, 1)

        mask_tanh = torch.kron(torch.eye(self.n_agents),
                       torch.tensor([[0, 0],
                                     [0., 0],
                                     [-self.b2/m, 0],
                                     [0, -self.b2/m]])
                       ) * self.h
        self.register_buffer('mask', mask)
        self.register_buffer('mask_tanh', mask_tanh)

        # dynamics module (owns its own copies of the parameters above, so that
        # an independent instance can be handed to a controller as its internal model)
        self.dynamics = RobotsDynamics(
            n_agents=self.n_agents, state_dim=self.state_dim, in_dim=self.in_dim, v_dim=self.v_dim,
            A_lin=self.A_lin, B=self.B, K_p=self.K_p, K_i=self.K_i,
            mask_tanh=self.mask_tanh, mask=self.mask,
            linear_plant=self.linear_plant, mass=self.mass, b2=self.b2,
        )

    def internal_model(self):
        """
        Build a fresh, independent RobotsDynamics instance with parameters
        identical to this plant's (nominal case). Intended to be handed to a
        controller as its internal model for IMC disturbance reconstruction.
        """
        return RobotsDynamics(
            n_agents=self.n_agents, state_dim=self.state_dim, in_dim=self.in_dim, v_dim=self.v_dim,
            A_lin=self.A_lin, B=self.B, K_p=self.K_p, K_i=self.K_i,
            mask_tanh=self.mask_tanh, mask=self.mask,
            linear_plant=self.linear_plant, mass=self.mass, b2=self.b2,
        )

    def noiseless_forward(self, t, eta: torch.Tensor, u: torch.Tensor, xbar: torch.Tensor):
        """
        forward of the plant without the process noise.

        Args:
            - eta (torch.Tensor): plant's augmented state (x, v) at t. shape = (batch_size, 1, eta_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next augmented state of the noise-free dynamics.
        """
        return self.dynamics.noiseless_forward(t, eta, u, xbar)

    def forward(self, t, eta, u, w, xbar):
        """
        forward of the plant with the process noise.

        Args:
            - eta (torch.Tensor): plant's augmented state (x, v) at t. shape = (batch_size, 1, eta_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            next augmented state. Process noise only enters the plant state x,
            never the integrator state v.
        """
        eta_next = self.noiseless_forward(t, eta, u, xbar)

        w = w.view(-1, 1, self.state_dim)
        w_padded = torch.cat((w, torch.zeros(w.shape[0], 1, self.v_dim, device=w.device, dtype=w.dtype)), dim=-1)

        return eta_next + w_padded
    # simulation
    def rollout(self, controller, data, train=False):
        """
        rollout REN for rollouts of the process noise

        Args:
            - data: sequence of disturbance samples of shape
                (batch_size, T, state_dim).

        Rerurn:
            - x_log of shape = (batch_size, T, state_dim)
            - u_log of shape = (batch_size, T, in_dim)
        """

        # init

        controller.reset()

        eta = self.eta_init.detach().clone().repeat(data.shape[0], 1, 1)
        u = self.u_init.detach().clone().repeat(data.shape[0], 1, 1)
        w = data[:,:,:8]
        xbar=data[:,:,[8, 9, 12, 13]]

        # Simulate
        for t in range(data.shape[1]):

            #eta_k+1 = f(eta_k,u_k, w_k, xbar_k)
            eta = self.forward(t=t, eta=eta, u=u, w=w[:, t:t+1, :], xbar=xbar[:, t:t+1, :])    # shape = (batch_size, 1, eta_dim)

            #u_k = c(eta_k,xbar_k)
            u = controller(eta, xbar[:, t:t+1, :])                                       # shape = (batch_size, 1, in_dim)

            x = eta[:, :, :self.state_dim]

            if t == 0:
                x_log, u_log, eta_log = x, u, eta
                e_log = xbar[:, t:t+1, :] - x[:,:,[0,1,4,5]]


            else:
                x_log = torch.cat((x_log, x), 1)
                u_log = torch.cat((u_log, u), 1)
                eta_log = torch.cat((eta_log, eta), 1)
                e_log = torch.cat((e_log, xbar[:, t:t+1, :] - x[:,:,[0,1,4,5]]), 1)

        controller.reset()
        if not train:
            x_log, u_log = x_log.detach(), u_log.detach()

        self.v_log = eta_log[:, :, self.state_dim:].detach()
        return x_log, e_log, u_log
