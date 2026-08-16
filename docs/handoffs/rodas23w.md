# Rodas23W parity handoff

## Upstream references

Pinned OrdinaryDiffEq revision: `211142263781255a9aa2f910f6760b9f18ec29c8`.

- `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl:36` lists `Rodas23W` as an adaptive Rosenbrock-W algorithm; `:Rodas23W` is also included in the shared Rodas perform-step dispatch at line 459.
- `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl:179-232` defines `Rodas23WRodasTableau`: five stages, `gamma = 1/3`, a second-order primary solution with the named 2/3 pair, and the embedded defect `btilde = [0, 0, 0, 1, -1]`.
- `lib/OrdinaryDiffEqRosenbrock/src/alg_utils.jl:10` retains `alg_order(Rodas23W) = 3` for adaptive-controller metadata, while the pinned regular-ODE endpoint fixture at `lib/OrdinaryDiffEqRosenbrock/test/ode_rosenbrock_tests.jl:111-132` explicitly measures order 2.
- `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl:432-557` is the regular ODE combined-cache stage/update path. The Rust implementation maps its `A`, `C`, `c`, `d`, `b`, and `btilde` fields into the allocation-free shared `perform_rodas` kernel.
- `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_caches.jl:411,436,487,516` identifies the method as a W-method and records its three-row stiff interpolation matrix `H`. `H` is intentionally not used for regular ODE output here; the crate's common recorder handles `save_at` trajectory samples.

## Rust implementation

`src/rosenbrock_extended.rs` adds the public `Rodas23W` constructor and exact Float64 tableau, registers it with the shared adaptive Rosenbrock kernel, and keeps the Jacobian factorization reusable between accepted steps. `src/lib.rs` re-exports the constructor.

Coverage tests exercise adaptive stiff integration, fixed-step convergence, backward integration, analytic Jacobian callbacks, discrete callback invalidation, and `save_at` recording. The existing shared Rosenbrock workspace is preallocated for eight stages, so this five-stage method has no per-step stage allocation.

## Pinned order limitation

The Rust fixed-step endpoint ratio is approximately `3.9773` for steps `0.1`
and `0.05`, i.e. regular-ODE order 2. This is pinned upstream behavior, not a
tableau-port error: the upstream regular-ODE convergence fixture explicitly
asserts order 2. The implementation retains the upstream `alg_order = 3`
metadata for adaptive controller semantics while testing endpoint order against
the pinned fixture.

Julia is not installed in the verification environment; retry the upstream comparison with `julia --project=.tmp-upstream` and the Rodas23W cases in `lib/OrdinaryDiffEqRosenbrock/test/ode_rosenbrock_tests.jl:111-132` when Julia is available.
