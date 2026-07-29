# CLAUDE.md

## What this repo is

PyTorch research code for **rPB (reference Performance Boosting)** — the implementation behind
"Boosting the transient performance of reference tracking controllers with neural networks"
(Kirsch, Massai, Ferrari-Trecate, CDC 2025). It currently implements the **centralized** case
on a two-robot reference-tracking task. The end goal of this repo is the **networked extension**
(see "Objective" below).

## Theory background (essential to not break guarantees)

- **PB framework**: for an ℓp-stable base system, *all and only* stability-preserving controllers
  are parametrized via an IMC scheme: reconstruct the disturbance ŵ = η − F(η, u, xref) using an
  internal model of the dynamics, then set u = M(ŵ) for a free ℓp-stable operator M. Performance
  is optimized by unconstrained training over parametrized ℓp operators (contractive RENs).
- **rPB extension**: M may also take the reference xref (not an ℓp signal). ℓp-ness of u is
  preserved via the **factorization** (Remark 1 of the paper):
      M(ŵ, xref) = M1(ŵ) ∗ M2(ŵ, xref)   (elementwise product)
  where M1 ∈ Lp (a contractive REN) and M2 has **uniformly bounded output** (MLP whose last
  hidden layer is sigmoid). |output_t| ≤ |M1(ŵ)_t|·bound, so u ∈ ℓp for any xref.
- **Reference governor structure**: here M's output is not a raw input but an offset `dxref`
  added to the setpoint of a pre-stabilized base tracking loop (pole-placed state feedback K_p
  + integral action K_i with integrator state v). Since dxref ∈ ℓp it vanishes asymptotically,
  so steady-state tracking of the base loop is preserved.

**Invariants that must never be violated by any refactor or extension:**
1. The ℓp factor (REN) never receives non-ℓp inputs (no xref, no raw neighbor outputs into it).
   Only ŵ-type signals (exogenous, ℓp) may enter it.
2. The context factor (MLP) must have uniformly bounded output regardless of input
   (bounded activation before the final fixed linear layer).
3. The IMC reconstruction must be exact in the nominal case: ŵ equals the true injected w
   (v-components of ŵ are 0 by construction since noise only enters plant states).
4. Stability follows from structure, not from trained weights — no stability-related constraint
   may depend on parameter values.

## Code map

- `experiments/robots/run.py` — main entrypoint: dataset → plant/controller/loss → training by
  backprop through closed-loop rollouts → validation on held-out references → plots/checkpoints
  in `experiments/robots/saved_results/`.
- `plants/robots/robots_sys.py` — `RobotsSystem`: two 2D point-mass robots (kron-structured,
  optional nonlinear friction via tanh), with the base tracking controller (K_p pole placement
  + K_i integral action, integrator state v) baked into `noiseless_forward`. Also `rollout()`.
- `controllers/PB_controller.py` — `PerfBoostController`: IMC disturbance reconstruction +
  `output = REN(ŵ) * MLP(ŵ, xbar) * output_amplification`. This is exactly M1 ∗ M2.
- `controllers/contractive_ren.py` — free parametrization of contractive (ℓp-stable) RENs.
- `controllers/MLP.py` — the bounded context factor M2.
- `plants/robots/robots_dataset.py` — synthetic rollouts: random initial states + random target
  positions (min inter-agent distance), so training generalizes across references.
- `loss_functions/robots_loss.py` — tracking + control effort + collision + obstacle penalties.
- `utils/` — logging, plotting.

## Conventions

- Signal shapes: `(batch_size, 1, dim)` per timestep; logs `(batch_size, T, dim)`.
- Data tensor layout: `data[:, :, :8]` = process noise w (nonzero only at t=0 = initial state
  perturbation), `data[:, :, 8:]` = reference xbar; positions of agent i are state indices
  `[4i, 4i+1]`, velocities `[4i+2, 4i+3]`; tracked outputs are indices `[0, 1, 4, 5]`.
- Integrator update ordering in the base loop: v is updated **before** computing
  u_base = −K_p x − K_i v⁺ (uses v⁺, not v). Preserve this ordering — the pole placement
  was done for this convention.

## Known issues / current state

- **Tangled integrator state**: the base-controller state v lives both in the plant's rollout
  loop and in the controller (`last_vg`), synced by hand. The controller already recomputes the
  correct v inside `noiseless_forward` and discards it. Planned fix: explicit augmented state
  η = (x, v), `noiseless_forward: (η, u, xbar) ↦ η⁺` pure with single return value.
- **Controller holds the plant's bound method** as its internal model. Should be an independent
  module instance (enables model-mismatch experiments per Thm. 2 of the paper).
- **Hardcoded 2-agent dimensions** throughout (`zeros(1,4)`, indices `[0,1,4,5]`, `[:, :, :8]`).
- `setup.py` is placeholder; README references a nonexistent `requirements.txt`.
- `output_amplification=20` and a commented-out clamp are tuning leftovers.

## Objective: networked rPB

Extend rPB to N interconnected subsystems on a communication graph G (adjacency only),
**without** the L2-gain/LMI machinery of Saccani et al. (arXiv:2404.02820: mixing matrices
Mvz/Mvw/Muz, dissipativity LMI, Gershgorin-based gain allocation η(b)).

Design: each agent i runs a local factorized controller
    z⁽ⁱ⁾ = R_p⁽ⁱ⁾(ŵ^(Ni)) ∗ R_∞⁽ⁱ⁾(ŵ⁽ⁱ⁾, s⁽ⁱ⁾, z^(Ni))
with u⁽ⁱ⁾ = z⁽ⁱ⁾, where:
- R_p⁽ⁱ⁾ ∈ Lp (contractive REN) sees only exogenous ℓp signals: local ŵ⁽ⁱ⁾ and optionally
  neighbors' ŵ⁽ʲ⁾ (communicated; still exogenous ⇒ the R_p layer stays feedforward, no loop).
- R_∞⁽ⁱ⁾ is uniformly bounded (sigmoid) and takes arbitrary context s⁽ⁱ⁾ (references, anything)
  plus neighbors' outputs z⁽ʲ⁾, j ∈ Nᵢ, with a **one-step communication delay** (well-posedness:
  no algebraic loop across the network).
- ŵ⁽ⁱ⁾ = η⁽ⁱ⁾ − F̂⁽ⁱ⁾(η^(Ni), u⁽ⁱ⁾, xbar⁽ⁱ⁾) reconstructed locally from neighbor states.

Why stability is free: |z⁽ⁱ⁾_t| ≤ |R_p⁽ⁱ⁾(ŵ)_t|·β⁽ⁱ⁾, so every feedback path between agents
passes through the bounded factor and cannot amplify ℓp norms. w ∈ ℓp ⇒ u ∈ ℓp for any s,
any graph, any weights — no gain condition, no LMI, no dimension bookkeeping.

Open theory items (paper-grade, handle with care):
- Robust (model-mismatch) version: the product breaks global incremental gain; expect a
  local/bounded-disturbance-set result or extra Lipschitz structure on R_∞.

## Workflow notes

- Run: `python experiments/robots/run.py` (see `arg_parser.py` for flags; `--epochs 50
  --num-rollouts 10 --horizon 50` for a quick smoke run).
- Any refactor of plant/controller must come with an equivalence check against the current
  implementation (same seed ⇒ same trajectories) and an assert that ŵ reconstruction is exact.
