# Rodas6P handoff

Rodas6P is implemented in `src/rosenbrock_extended.rs` and exported from the
algorithm namespace as `differential_equations::algorithms::rosenbrock::Rodas6P`. The regular ODE tableau is
the pinned `Rodas6PTableau` from OrdinaryDiffEq revision
`211142263781255a9aa2f910f6760b9f18ec29c8`, with 19 stages, `gamma = 0.26`,
and the upstream primary/embedded weights (`btilde` is the sixteenth stage).
The shared Rosenbrock-W kernel provides finite-difference or user-supplied
Jacobians, time derivatives, LU solves, adaptive control, fixed/backward
stepping, callbacks, and `save_at` handling. DAE/SDE paths and the upstream
stiff-aware dense interpolant are intentionally excluded from this regular
ODE parity port.

Validation on this branch:

- `cargo fmt -- --check`
- `cargo test --all-targets`
- `cargo clippy --all-targets -- -D warnings`
- `git diff --check`
- `cargo run --quiet --release --example rosenbrock_extended_compliance`
- `tests/rodas6p_allocations.rs` confirms callback-free allocations are step-invariant.

Julia parity fixture updates are in `tests/julia/rosenbrock_extended.jl`. Julia
was not installed in the validation environment (`Get-Command julia` returned
no executable); retry with the repository's configured Julia executable using
the test fixture once Julia is available.
