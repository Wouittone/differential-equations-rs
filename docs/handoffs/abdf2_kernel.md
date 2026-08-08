# ABDF2 regular-ODE parity wave

## Scope and upstream identity

This wave ports the pinned `OrdinaryDiffEqBDF.ABDF2` regular initial-value
algorithm at OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. Exact source references used:

- `lib/OrdinaryDiffEqBDF/src/algorithms.jl:47-61,63-180` (ABDF2 declaration,
  fixed-leading-coefficient adaptive order-2 constructor and cache shape).
- `lib/OrdinaryDiffEqBDF/src/bdf_perform_step.jl:1-11` (history
  initialization), `:13-29` (implicit-Euler startup), and `:30-93`
  (variable-step coefficients and nonlinear residual).
- `lib/OrdinaryDiffEqBDF/src/bdf_perform_step.jl:103-181` (mutable history
  update contract).

The implementation excludes `DABDF2`/residual DAE setup, variable-order
`QNDF`/`FBDF`, split/IMEX variants, singular mass matrices, and wrappers. It
supports regular identity-mass ODEs only.

## Numerical design

`Abdf2Kernel` uses the shared frozen `StepKernel` lifecycle. The first accepted
step is implicit Euler. For later steps, with `rho = h_n/h_{n-1}`, the
residual is:

```
y_{n+1} - (1 + rho^2/3)y_n + (rho^2/3)y_{n-1}
  - h_n[(2/3)f(t_{n+1}, y_{n+1}) - (rho - 1)/3 f(t_n, y_n)] = 0.
```

Analytic Jacobians are used directly; otherwise a checked finite-difference
Jacobian is formed. Newton failures, singular factors, and non-finite
derivatives follow the shared recoverable adaptive-attempt policy. Callback
effects reset the two-step history and restart with implicit Euler.

Adaptive error control uses the accepted candidate's predictor defect against
the current-state Euler predictor and the shared order-2 proportional
controller. Adaptive step/rejection counts can therefore differ from Julia's
controller while endpoints remain within parity tolerances.

## Validation evidence

`tests/abdf2.rs` covers fixed-step order, adaptive stiff/nonautonomous
integration and rejection, backward integration and callback reset, analytic
versus finite-difference Jacobians, non-finite and singular failures, and a
long fixed run with bounded workspace shape. `examples/abdf2_compliance.rs`
emits fixed and adaptive endpoints for Julia comparison.

`tests/julia/abdf2.jl` compares the same problem with pinned
`OrdinaryDiffEqBDF.ABDF2`. The coordinator must retain the pinned
`OrdinaryDiffEqBDF` Project/Manifest entry when merging this wave.

Rust and Julia gates passed on this isolated branch after the dependency is
present:

```
cargo fmt -- --check
cargo test --all-targets                 # 96 Rust tests including 6 ABDF2 tests
cargo clippy --all-targets -- -D warnings
git diff --check
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl  # 206 assertions
```

