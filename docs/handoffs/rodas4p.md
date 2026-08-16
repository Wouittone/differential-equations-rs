# Rodas4P parity handoff

Rodas4P is implemented as a public native regular-ODE Rosenbrock method in
`src/rosenbrock_extended.rs` and exported from `src/lib.rs`. It reuses the
shared Rodas kernel, so fixed and adaptive scheduling, backward integration,
callbacks, `save_at`, analytic or finite-difference Jacobians, and workspace
allocation are covered by the same lifecycle as the other extended methods.

The tableau is copied from `Rodas4PTableau` in
`lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl` at pinned
OrdinaryDiffEq revision `211142263781255a9aa2f910f6760b9f18ec29c8`:

- six stages, `gamma = 0.25`;
- primary weights are the sixth row of `RODAS4PA` plus the final unit weight;
- embedded error weights are `[0, 0, 0, 0, 0, 1]`;
- nodes and time-derivative weights are `RODAS4Pc` and `RODAS4Pd`.

The compliance fixture is `examples/rosenbrock_extended_compliance.rs` and
prints `rodas4p_adaptive` and `rodas4p_fixed` records.

Verification performed:

- `cargo fmt -- --check`
- focused fixed-order, adaptive stiff, backward, callback/save-at/Jacobian tests
- Julia parity could not be run because `julia` is not on `PATH`; retry with
  `JULIA-PATH=<path-to-julia>`. The upstream fixture is
  `lib/OrdinaryDiffEqRosenbrockTableaus/test/ode_rosenbrock_tests.jl`.
