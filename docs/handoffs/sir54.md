# SIR54 implementation handoff

- Upstream package: `OrdinaryDiffEqLowOrderRK`
- Pinned revision: `211142263781255a9aa2f910f6760b9f18ec29c8`
- Algorithm declaration: `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:425-449`
- Tableau source: `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1733-1828`
- Step source: `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:1798-1843`

The Rust `Sir54` facade uses the shared explicit Runge–Kutta driver with the
exact seven-stage embedded (4,5) tableau, nodes, and error estimator from the
pinned Julia cache. Its eighth row evaluates the accepted endpoint derivative
and repeats the primary weights, preserving the upstream FSAL lifecycle.

The pinned upstream `btilde` values are retained verbatim as the embedded
estimator. The upstream convergence fixture exercises SIR54 with fixed
stepping; this branch does the same because the shared controller's direct
error norm exposes the unusually large pinned estimator on a scalar
exponential test.

The public Rust name follows this crate's acronym convention (`Sir54`), while
the Julia constructor remains `SIR54`.

Validation on this branch:

- `cargo fmt -- --check`
- `cargo test --all-targets` (including fixed-step convergence and dense
  save-at tests)
- `cargo clippy --all-targets -- -D warnings`
- `git diff --check`

Julia validation is pending if `julia` is unavailable in the execution
environment. Retry `julia --project=tests/julia tests/julia/pinned_environment.jl --check`
and `julia --project=tests/julia tests/julia/runtests.jl` when Julia is available.
