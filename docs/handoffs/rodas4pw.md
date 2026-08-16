# Rodas4PW handoff

Summary: Added the native regular-ODE Rodas4PW Rosenbrock-W algorithm using
the pinned nine-stage tableau and the existing allocation-free Rosenbrock
driver.

Files changed:

- `src/rosenbrock_extended.rs`: public marker, exact pinned A/C/nodes/d/b and
  final-stage embedded estimator, driver registration, lifecycle tests.
- `src/lib.rs`: public `Rodas4PW` export.
- `examples/rosenbrock_extended_compliance.rs`: fixed/adaptive endpoint row.
- `tests/julia/rosenbrock_extended.jl`: Julia reference fixture row.
- `tests/rodas4pw_allocations.rs`: callback-free step allocation regression.

Public APIs added: `differential_equations::Rodas4PW`.

Upstream source and revision: `lib/OrdinaryDiffEqRosenbrockTableaus/src/
rosenbrock_tableaus.jl`, `Rodas4PWTableau`, revision
`211142263781255a9aa2f910f6760b9f18ec29c8`; algorithm metadata is in
`lib/OrdinaryDiffEqRosenbrock/src/alg_utils.jl` and `algorithms.jl`.

Rust tests: full `cargo test --all-targets` passed (111 unit tests plus all
integration targets); focused tests cover fourth-order fixed-step convergence,
adaptive stiff integration, analytic Jacobian use, callbacks, `save_at`,
backward integration, and callback-free allocation invariance.

Julia tests: unavailable in this environment because PowerShell cannot resolve
the `julia` executable. Exact retry command:

```text
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl
```

Commands run: `cargo fmt --all`; `cargo fmt -- --check` (passed after the
compliance import was formatted); `cargo test --all-targets` (passed); focused
Rodas4PW convergence test (passed); `cargo clippy --all-targets -- -D
warnings` (passed); `git diff --check` (passed); compliance example (passed).

Numerical differences: The shared recorder supplies trajectory sampling rather
than OrdinaryDiffEq's stiff-specific dense interpolation. The pinned regular
ODE tableau uses `btilde = [0,0,0,0,0,0,0,0,1]` for adaptive error control.

Allocation/performance impact: Uses the existing reusable Rosenbrock workspace
and factorization; no per-stage heap allocation was introduced.

Known limitations: Julia compliance remains pending the missing Julia runtime;
DAE-only residual behavior, wrappers, and stiff-specific dense interpolation
remain outside this regular-ODE port.

Follow-up dependencies: Coordinator should regenerate the regular-ODE
inventory and update the overnight status counts after merging.

Recommended next task: Port the next missing pinned Rosenbrock/Rodas
constructor, preserving the shared driver interface.
