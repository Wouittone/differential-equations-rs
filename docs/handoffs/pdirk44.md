# PDIRK44 implementation handoff

## Scope and upstream identity

`Pdirk44`/`PDIRK44` implements the fixed-step fourth-order parallel
diagonally implicit Runge--Kutta method exported by `OrdinaryDiffEqPDIRK` at
revision `211142263781255a9aa2f910f6760b9f18ec29c8`.

The constructor and order metadata come from
`lib/OrdinaryDiffEqPDIRK/src/algorithms.jl`. The stage recurrence and exact
rational tableau come from `pdirk_perform_step.jl` and `pdirk_caches.jl`:

- diagonal factors `(1/2, 2/3)`;
- stage abscissae `(1/2, 2/3, 1/2, 1/3)`;
- second-wave bases `u - 5k11/2 + 5k12/2` and
  `u - 5k11/3 + 4k12/3`;
- final update `u - k11 - k21 + 3k12/2 + 3k22/2`.

## Rust behavior

Each increment solves `k = h*f(base + gamma*k, t + c*h)` by Newton iteration.
Dense analytic problem Jacobians are used when supplied; otherwise the kernel
builds a forward finite-difference Jacobian. Singular systems, non-finite
derivatives, and nonlinear failure use the crate's shared `SolveError` values,
and all RHS/Jacobian/factorization/linear-solve counters are recorded.

The pinned Julia method can evaluate the two stages in each wave concurrently.
The Rust kernel deliberately evaluates them sequentially: this preserves the
method and deterministic statistics without nesting a per-step thread policy.
Parallel independent solves are available through `solve_ensemble`.

## Verification

- Rust integration tests cover fourth-order convergence, vector and
  nonautonomous backward integration, analytic Jacobians, solver statistics,
  unsupported adaptive mode, and non-finite RHS failure.
- `examples/pdirk44_compliance.rs` and `tests/julia/pdirk44.jl` compare the
  same fixed-step scalar endpoint against the pinned Julia constructor.
- The Julia compliance project pins `OrdinaryDiffEqPDIRK` directly to the
  reference monorepo revision.

The focused Rust suite, repository-wide Rust tests, Clippy with warnings
denied, formatting, diff checks, pinned Julia environment validation, and the
matched Rust/Julia fixture passed during integration.

## Remaining related work

PDIRK44 has no adaptive estimator upstream. Method-specific dense
interpolation is tracked by the repository-wide dense-output roadmap item; the
current PDIRK44 solve uses the shared accepted-endpoint fallback until that
family interpolation is defined.
