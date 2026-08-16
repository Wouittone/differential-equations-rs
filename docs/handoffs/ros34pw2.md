# ROS34PW2 handoff

Status: implemented, pending coordinator integration.

The public `Ros34Pw2` constructor is exported from `src/lib.rs` and wired to
the shared native Rosenbrock driver in `src/rosenbrock_extended.rs`. The
tableau is copied exactly from `ROS34PW2RodasTableau` in
`lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl` at pinned
revision `211142263781255a9aa2f910f6760b9f18ec29c8`:

- `gamma = 0.435866521508459`, four stages, no stiff-aware dense matrix `H`.
- `A`, `C`, `c`, `d`, `b`, and `btilde` are preserved as the upstream
  Float64 literals.
- The stage/update path follows the generic `RodasTableau` branch in
  `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl`, specifically
  the `perform_step!(integrator, cache::RosenbrockCache, ...)` implementation
  (stages 1–4, explicit `b` update, and adaptive `btilde` estimate).

Focused Rust coverage is in `tests/ros34pw2.rs`: fixed-step convergence on the
upstream method's advertised `(3)4` behavior (the nonstiff linear fixture
converges at order ≈ 3), adaptive stiff decay with an analytic Jacobian,
backward callback/save-at lifecycle, and public-constructor compilation.

Julia compliance should be retried with Julia available from the repository's
Julia project, running the pinned OrdinaryDiffEq Rosenbrock tableau fixture for
`ROS34PW2`; this environment has no `julia` executable on PATH.
