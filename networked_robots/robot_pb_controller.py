import torch
import torch.nn as nn

from config import device
from controllers import ContractiveREN
from controllers.MLP import MLP


class RobotPBController(nn.Module):
    """
    Single agent's local factorized controller:
        z = M_p(w_hat) * M_infty(w_hat, xbar, own_position, z_neighbors).
    Mirrors controllers/PB_controller.py::PerfBoostController (same IMC
    disturbance reconstruction + REN*MLP factorization), generalized with two
    extra M_infty inputs: this agent's own current (px,py) position as
    context, and neighbors' control actions from the PREVIOUS timestep
    (one-step delay enforced by the caller, PBControllerNetwork - this class
    only ever receives an already-delayed z_neighbors and does not manage
    the delay itself). Feeding raw position into M_infty is fine per the
    theory - M_infty takes "arbitrary context (references, anything)" and is
    unconditionally bounded regardless of its input; only M_p (the REN) is
    restricted to exogenous l_p signals. For now only the agent's own
    position is used as context (not neighbors' positions).

    Unlike the centralized controller, the IMC reconstruction here uses the
    CURRENT xbar (already available as a forward() argument) rather than a
    lagged copy - so w_hat exactly equals the true injected disturbance in
    the nominal case (no reference-change transient).

    central_like=True switches this agent's M_p/M_infty to imitate what the
    centralized controllers/PB_controller.py::PerfBoostController's single
    REN/MLP pair actually sees, as closely as possible while keeping N
    separate per-agent (locally parametrized) networks: M_p sees every
    agent's w_hat concatenated (still only exogenous w_hat signals - own_position
    and z_neighbors are dropped entirely, so there is no u/z interconnection
    between agents, matching the centralized controller which never routes
    one agent's output into another's computation), and M_infty sees every
    agent's w_hat and every agent's xbar concatenated (in place of just its
    own + own_position + z_neighbors). REN/MLP dims widen accordingly at
    construction time (see __init__) - this only changes which methods
    PBControllerNetwork calls (compute_w_hat/compute_output_central_like/commit
    instead of forward()), so the default central_like=False path is
    untouched: forward() itself is unmodified.
    """
    def __init__(
        self, internal_model, eta_init: torch.Tensor, output_init: torch.Tensor, n_neighbors: int,
        dim_internal: int, dim_nl: int,
        initialization_std: float = 0.5,
        posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0,
        ren_internal_state_init=None,
        output_amplification: float = 20,
        imc_tol: float = 1e-4,
        central_like: bool = False, total_n_agents: int = 1,
    ):
        super().__init__()

        self.output_amplification = output_amplification
        self.imc_tol = imc_tol

        self.internal_model = internal_model
        self.state_dim = internal_model.state_dim    # 4
        self.in_dim = internal_model.in_dim           # 2
        self.xbar_dim = 2
        self.position_dim = 2
        self.n_neighbors = n_neighbors
        self.central_like = central_like
        self.total_n_agents = total_n_agents

        self.eta_init = eta_init.reshape(1, -1)
        self.output_init = output_init.reshape(1, -1)

        if central_like:
            ren_dim_in = self.state_dim * total_n_agents
            mlp_in_dim = self.state_dim * total_n_agents + self.xbar_dim * total_n_agents
        else:
            ren_dim_in = self.state_dim
            mlp_in_dim = self.state_dim + self.xbar_dim + self.position_dim + self.in_dim * n_neighbors

        self.c_ren = ContractiveREN(
            dim_in=ren_dim_in, dim_out=self.in_dim, dim_internal=dim_internal,
            dim_nl=dim_nl, initialization_std=initialization_std,
            internal_state_init=ren_internal_state_init,
            posdef_tol=posdef_tol, contraction_rate_lb=contraction_rate_lb,
        ).to(device)

        self.MLP = MLP(dim_in=mlp_in_dim, dim_out=self.in_dim)

        self.reset()

    def reset(self):
        """Set time to 0 and reset to initial state."""
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
            xbar=xbar_t,    # current xbar, not a lagged copy
        )

        w_hat = eta_t - eta_noiseless
        w_hat_x = w_hat[:, :, :self.state_dim]
        self.last_w_hat_v = w_hat[:, :, self.state_dim:]

        own_position = eta_t[:, :, :self.position_dim]    # this agent's own (px,py) - context, not an l_p signal

        output_REN = self.c_ren.forward(w_hat_x)
        mlp_input = torch.cat((w_hat_x, xbar_t, own_position, z_neighbors), dim=2)
        output_MLP = self.MLP.forward(mlp_input)

        output = output_REN * output_MLP * self.output_amplification

        self.last_eta, self.last_output = eta_t, output
        self.t += 1
        return output

    def compute_w_hat(self, eta_t: torch.Tensor, xbar_t: torch.Tensor):
        """
        IMC disturbance reconstruction only (no REN/MLP/state commit) - used
        by PBControllerNetwork's central_like path, which must gather every
        agent's w_hat before any agent's REN/MLP can run. Dynamics are
        decoupled, so this local step is identical regardless of
        central_like - only the REN/MLP inputs (computed downstream) differ.
        """
        eta_noiseless = self.internal_model.noiseless_forward(
            t=self.t, eta=self.last_eta, u=self.last_output, xbar=xbar_t,
        )
        w_hat = eta_t - eta_noiseless
        w_hat_x = w_hat[:, :, :self.state_dim]
        self.last_w_hat_v = w_hat[:, :, self.state_dim:]
        return w_hat_x

    def compute_output_central_like(self, w_hat_all: torch.Tensor, xbar_all: torch.Tensor):
        """
        central_like REN/MLP step: both operators see every agent's w_hat and
        every agent's reference - no own_position, no z_neighbors, i.e. no
        u/z interconnection between agents, mirroring the centralized
        controller's single REN(w_hat) * MLP(w_hat, xbar) exactly, just
        replicated per-agent with local weights and a local output slice.
        """
        output_REN = self.c_ren.forward(w_hat_all)
        mlp_input = torch.cat((w_hat_all, xbar_all), dim=2)
        output_MLP = self.MLP.forward(mlp_input)
        return output_REN * output_MLP * self.output_amplification

    def commit(self, eta_t: torch.Tensor, output: torch.Tensor):
        """Advance this agent's recurrent state once its output for this step is known."""
        self.last_eta, self.last_output = eta_t, output
        self.t += 1

    def check_imc_exactness(self, tol=None):
        """
        Assert that the v-components of the last reconstructed disturbance
        are ~0, as required by exact IMC reconstruction in the nominal
        (no model-mismatch) case. Should hold unconditionally here (unlike
        the centralized controller), since the reconstruction uses the
        current xbar rather than a lagged copy.
        """
        tol = self.imc_tol if tol is None else tol
        assert self.last_w_hat_v is not None, "no forward() call yet to check"
        max_dev = torch.max(torch.abs(self.last_w_hat_v)).item()
        assert max_dev < tol, f"IMC reconstruction violated: max |w_hat_v| = {max_dev} >= {tol}"
        return max_dev
