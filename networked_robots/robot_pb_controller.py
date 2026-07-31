import torch
import torch.nn as nn

from config import device
from controllers import ContractiveREN
from controllers.MLP import MLP


class RobotPBController(nn.Module):
    """
    Single agent's M operator: z = M_p(w_hat_all) * M_infty(w_hat_own, xbar,
    own_position, neighbor_positions, neighbor_xbars, z_neighbors) - one of
    the N interconnected M operators that make up the network's rPB
    controller. This class owns no disturbance-reconstruction state (no
    internal_model, no eta/output/t bookkeeping): the IMC reconstruction is
    PBControllerNetwork's job, done ONCE for the whole network (see its
    docstring) - "one rPB [reconstruction], N interconnected M's", not N
    separate mini-rPBs each redoing their own piece of it. This class's only
    recurrent state is its own REN's internal state (reset() below).

    M_p sees the joint w_hat (every agent's local reconstruction,
    concatenated) - CLAUDE.md's Objective explicitly allows this ("local
    w_hat and optionally neighbors' w_hat - communicated, still exogenous,
    so R_p stays feedforward, no loop"). M_infty additionally sees this
    agent's own current (px,py) position, its neighbors' CURRENT (px,py)
    positions and targets, and neighbors' control actions from the PREVIOUS
    timestep (one-step delay enforced by the caller, PBControllerNetwork).
    Feeding raw position/xbar into M_infty is fine per the theory - M_infty
    takes "arbitrary context (references, anything)" and is unconditionally
    bounded regardless of its input; only M_p (the REN) is restricted to
    exogenous w_hat-type signals, which the joint w_hat still is.

    `mode` selects one of three M_infty input compositions (M_p, the REN,
    always sees the joint w_hat regardless of mode):
      - 'base' (default): M_infty sees its own w_hat + own xbar + neighbors'
        xbar + own position + neighbors' positions + neighbors' control
        actions from the PREVIOUS timestep (z_neighbors, one-step delayed -
        this is the actual u/z interconnection between agents).
      - 'no_interconnection': identical to 'base' minus z_neighbors - states
        and context (positions, references) are still shared across agents,
        but no agent ever sees another agent's output, isolating the effect
        of the u/z interconnection itself from state/context sharing.
      - 'central_like': M_infty sees every agent's w_hat and every agent's
        xbar concatenated, in place of the 'base' inputs above - no
        positions, no z_neighbors, no u/z interconnection at all - imitating
        what the centralized controllers/PB_controller.py::PerfBoostController's
        single REN/MLP pair actually sees, as closely as possible while
        keeping N separate per-agent (locally parametrized) networks.
    MLP dim varies accordingly at construction time; M_p's input is
    unaffected by mode (already the joint w_hat in all three cases).

    M_infty's inputs are fixed-scale-normalized before concatenation, so no
    one channel dominates purely from unit choice (this is pure input
    conditioning - M_infty's bounded-output guarantee is unconditional
    regardless of input scale, so this changes nothing about the invariants):
      - z_neighbors ('base' mode only) is divided by output_amplification,
        undoing the one deliberate gain in the architecture (output =
        REN*MLP*amplification, so z_neighbors is O(amplification) even for
        near-random early weights, while every other M_infty input is O(1) -
        left unscaled, it dominates the concatenated input and can push
        M_infty into a saturated/uninformative regime right from the start
        of training).
      - xbar_t/own_position/neighbor_positions/neighbor_xbars (and, in
        central_like mode, xbar_all) are divided by position_scale, a
        characteristic length of the scenario (e.g. the corridor's spatial
        extent), so raw coordinates land in O(1) rather than O(position_scale).
    """
    def __init__(
        self, state_dim: int, in_dim: int, xbar_dim: int, position_dim: int, n_neighbors: int,
        dim_internal: int, dim_nl: int,
        initialization_std: float = 0.5,
        posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0,
        ren_internal_state_init=None,
        output_amplification: float = 20,
        mode: str = 'base', total_n_agents: int = 1,
        position_scale: float = 5.0,
    ):
        super().__init__()

        self.output_amplification = output_amplification
        self.position_scale = position_scale

        self.state_dim = state_dim        # 4 - this agent's own (px,py,vx,vy)
        self.in_dim = in_dim              # 2
        self.xbar_dim = xbar_dim          # 2
        self.position_dim = position_dim  # 2
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.total_n_agents = total_n_agents

        # M_p (REN) always sees the joint w_hat (every agent's, concatenated).
        ren_dim_in = self.state_dim * total_n_agents

        if mode == 'central_like':
            mlp_in_dim = self.state_dim * total_n_agents + self.xbar_dim * total_n_agents
        elif mode == 'no_interconnection':
            mlp_in_dim = (
                self.state_dim                               # own w_hat only (M_infty keeps its local term)
                + self.xbar_dim * (1 + n_neighbors)          # own xbar + neighbors' xbars
                + self.position_dim * (1 + n_neighbors)      # own position + neighbors' positions
            )
        else:
            assert mode == 'base', f"unknown mode '{mode}'"
            mlp_in_dim = (
                self.state_dim                               # own w_hat only (M_infty keeps its local term)
                + self.xbar_dim * (1 + n_neighbors)          # own xbar + neighbors' xbars
                + self.position_dim * (1 + n_neighbors)      # own position + neighbors' positions
                + self.in_dim * n_neighbors                  # neighbors' delayed z
            )

        self.c_ren = ContractiveREN(
            dim_in=ren_dim_in, dim_out=self.in_dim, dim_internal=dim_internal,
            dim_nl=dim_nl, initialization_std=initialization_std,
            internal_state_init=ren_internal_state_init,
            posdef_tol=posdef_tol, contraction_rate_lb=contraction_rate_lb,
        ).to(device)

        self.MLP = MLP(dim_in=mlp_in_dim, dim_out=self.in_dim)

        self.reset()

    def reset(self):
        """This operator's only recurrent state is the REN's internal state."""
        self.c_ren.x = self.c_ren.init_x

    def compute_output(self, own_position: torch.Tensor, xbar_t: torch.Tensor,
                        w_hat_own: torch.Tensor, w_hat_all: torch.Tensor,
                        neighbor_positions: torch.Tensor, neighbor_xbars: torch.Tensor,
                        z_neighbors: torch.Tensor):
        """
        'base' mode M_p/M_infty step, called once PBControllerNetwork
        has gathered every agent's w_hat into w_hat_all.

        Args:
            own_position: this agent's current (px,py). (batch,1,2).
            xbar_t: this agent's current reference. (batch,1,2).
            w_hat_own: this agent's own w_hat, used as M_infty's local
                disturbance term. (batch,1,state_dim).
            w_hat_all: every agent's w_hat concatenated, used as M_p's input.
                (batch,1,state_dim*total_n_agents).
            neighbor_positions: neighbors' CURRENT (px,py), concatenated,
                undelayed (sensed state, not another controller's output).
                (batch,1,2*n_neighbors); width 0 when n_neighbors=0.
            neighbor_xbars: neighbors' CURRENT reference, concatenated,
                undelayed (given data, not another controller's output).
                (batch,1,2*n_neighbors); width 0 when n_neighbors=0.
            z_neighbors: this agent's neighbors' z from the PREVIOUS timestep,
                already delayed by the caller. (batch,1,2*n_neighbors); width
                0 when n_neighbors=0.

        Returns:
            z_t (torch.Tensor): this agent's control action, (batch,1,2).
        """
        output_REN = self.c_ren.forward(w_hat_all)
        mlp_input = torch.cat((
            w_hat_own,
            xbar_t / self.position_scale,
            own_position / self.position_scale,
            neighbor_positions / self.position_scale,
            neighbor_xbars / self.position_scale,
            z_neighbors / self.output_amplification,
        ), dim=2)
        output_MLP = self.MLP.forward(mlp_input)

        return output_REN * output_MLP * self.output_amplification

    def compute_output_no_interconnection(self, own_position: torch.Tensor, xbar_t: torch.Tensor,
                                           w_hat_own: torch.Tensor, w_hat_all: torch.Tensor,
                                           neighbor_positions: torch.Tensor, neighbor_xbars: torch.Tensor):
        """
        'no_interconnection' mode M_p/M_infty step: identical to
        compute_output() (states and context - positions, references - are
        still shared across agents) but with no z_neighbors term, i.e. no
        u/z interconnection between agents at all - isolates the effect of
        the delayed-action coupling from state/context sharing.

        Args: same as compute_output() minus z_neighbors.

        Returns:
            z_t (torch.Tensor): this agent's control action, (batch,1,2).
        """
        output_REN = self.c_ren.forward(w_hat_all)
        mlp_input = torch.cat((
            w_hat_own,
            xbar_t / self.position_scale,
            own_position / self.position_scale,
            neighbor_positions / self.position_scale,
            neighbor_xbars / self.position_scale,
        ), dim=2)
        output_MLP = self.MLP.forward(mlp_input)

        return output_REN * output_MLP * self.output_amplification

    def compute_output_central_like(self, w_hat_all: torch.Tensor, xbar_all: torch.Tensor):
        """
        'central_like' mode M_p/M_infty step: both operators see every
        agent's w_hat and every agent's reference - no own_position, no
        z_neighbors, i.e. no u/z interconnection between agents, mirroring
        the centralized controller's single REN(w_hat) * MLP(w_hat, xbar)
        exactly, just replicated per-agent with local weights and a local
        output slice.
        """
        output_REN = self.c_ren.forward(w_hat_all)
        mlp_input = torch.cat((w_hat_all, xbar_all / self.position_scale), dim=2)
        output_MLP = self.MLP.forward(mlp_input)
        return output_REN * output_MLP * self.output_amplification
