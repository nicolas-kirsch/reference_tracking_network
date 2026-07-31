import torch
import torch.nn as nn

from config import device
from controllers import ContractiveREN
from controllers.MLP import MLP

from .robot_pb_controller import RobotPBController


class RobotPBControllerWindowed(RobotPBController):
    """
    EXPERIMENTAL variant of RobotPBController: also feeds a WINDOWED copy of
    the context signals (xbar, own position, neighbors' z - already fed to
    M_infty) into M_p (the REN), using a FIXED time window
    T_t = 1{t < T_theta}, T_theta = round(window_fraction * horizon).
    Since T_t*context has finite support, it is trivially in every l_p space,
    so this technically satisfies "M_p only sees l_p inputs" - but see the
    warning below.

    WARNING - this deliberately relaxes CLAUDE.md's Invariant 1. The
    windowed signal's l_p-NORM scales with T_theta and with the reference's
    magnitude, so M_p's output is no longer guaranteed bounded independent
    of xbar - the property the M1/M2 (R_p/R_infty) split exists to provide
    ("no gain condition... works for any xref"). This class is kept
    deliberately separate from RobotPBController (which stays
    guarantee-preserving, w_hat-only REN input) - NOT a drop-in replacement.

    M_infty keeps receiving the same raw (unwindowed) context it already
    did - only the new M_p pathway is windowed. IMC disturbance
    reconstruction itself is unchanged, so check_imc_exactness() (inherited)
    should still hold exactly.
    """
    def __init__(
        self, internal_model, eta_init: torch.Tensor, output_init: torch.Tensor, n_neighbors: int,
        dim_internal: int, dim_nl: int, horizon: int, window_fraction: float = 0.75,
        initialization_std: float = 0.5,
        posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0,
        ren_internal_state_init=None,
        output_amplification: float = 20,
        imc_tol: float = 1e-4,
    ):
        nn.Module.__init__(self)    # skip RobotPBController.__init__ - c_ren/MLP dims differ here

        self.output_amplification = output_amplification
        self.imc_tol = imc_tol

        self.internal_model = internal_model
        self.state_dim = internal_model.state_dim    # 4
        self.in_dim = internal_model.in_dim           # 2
        self.xbar_dim = 2
        self.position_dim = 2
        self.n_neighbors = n_neighbors

        self.horizon = horizon
        self.window_fraction = window_fraction
        self.T_theta = round(window_fraction * horizon)

        self.eta_init = eta_init.reshape(1, -1)
        self.output_init = output_init.reshape(1, -1)

        self.context_dim = self.xbar_dim + self.position_dim + self.in_dim * n_neighbors

        self.c_ren = ContractiveREN(
            dim_in=self.state_dim + self.context_dim, dim_out=self.in_dim, dim_internal=dim_internal,
            dim_nl=dim_nl, initialization_std=initialization_std,
            internal_state_init=ren_internal_state_init,
            posdef_tol=posdef_tol, contraction_rate_lb=contraction_rate_lb,
        ).to(device)

        mlp_in_dim = self.state_dim + self.context_dim
        self.MLP = MLP(dim_in=mlp_in_dim, dim_out=self.in_dim)

        self.reset()

    def reset(self):
        """
        Own copy of the pre-refactor RobotPBController.reset() body - kept
        self-contained (not inherited) since RobotPBController no longer
        owns eta/output/t bookkeeping (that moved to PBControllerNetwork's
        joint reconstruction step; this class keeps the old per-controller
        pattern deliberately, see class docstring).
        """
        self.t = 0
        self.last_eta = self.eta_init.detach().clone()
        self.last_output = self.output_init.detach().clone()
        self.last_w_hat_v = None    # v-components of the last reconstructed disturbance (for IMC checks)

        self.c_ren.x = self.c_ren.init_x

    def forward(self, eta_t: torch.Tensor, xbar_t: torch.Tensor, z_neighbors: torch.Tensor):
        """
        Args:
            eta_t: this agent's measured augmented state (x, v), (batch,1,6).
            xbar_t: this agent's current reference, (batch,1,2).
            z_neighbors: this agent's neighbors' z from the PREVIOUS timestep,
                already delayed by the caller. (batch,1,2*n_neighbors); width
                0 when n_neighbors=0.

        Returns:
            z_t (torch.Tensor): this agent's control action, (batch,1,2).
        """
        eta_noiseless = self.internal_model.noiseless_forward(
            t=self.t,
            eta=self.last_eta,
            u=self.last_output,
            xbar=xbar_t,
        )

        w_hat = eta_t - eta_noiseless
        w_hat_x = w_hat[:, :, :self.state_dim]
        self.last_w_hat_v = w_hat[:, :, self.state_dim:]

        own_position = eta_t[:, :, :self.position_dim]
        context = torch.cat((xbar_t, own_position, z_neighbors), dim=2)

        window = 1.0 if self.t < self.T_theta else 0.0
        ren_input = torch.cat((w_hat_x, context * window), dim=2)
        output_REN = self.c_ren.forward(ren_input)

        mlp_input = torch.cat((w_hat_x, context), dim=2)    # M_infty: raw, unwindowed context
        output_MLP = self.MLP.forward(mlp_input)

        output = output_REN * output_MLP * self.output_amplification

        self.last_eta, self.last_output = eta_t, output
        self.t += 1
        return output
