# Rodas4 parity audit

## Result

The regular native ODE `Rodas4` port is already parity-complete on the audit
branch. No duplicate public type, tableau, export, fixture, or implementation
was added by this audit.

## Pinned upstream

The audit target is OrdinaryDiffEq revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl:48` identifies `Rodas4` and
  describes it as a fourth-order A-stable, stiffly stable method.
- `lib/OrdinaryDiffEqRosenbrock/src/alg_utils.jl:29` assigns primary order 4;
  `:63` marks it non-FSAL.
- `lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl:77-110`
  defines `RODAS4A`, `RODAS4C`, `RODAS4c`, `RODAS4d`, `RODAS4H`, and
  `Rodas4Tableau`.
- `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl:729-862`
  defines the shared regular-ODE `RosenbrockCache` stage/update/error path.

The Rust constants at `src/rosenbrock_extended.rs:734-830` match the pinned
`gamma = 0.25`, six-stage `A` and `C` arrays, nodes
`[0, 0.386, 0.21, 0.63, 1, 1]`, time-derivative weights
`[0.25, -0.1043, 0.1035, -0.0362, 0, 0]`, solution weights
`[1.221224509226641, 6.019134481288629, 12.53708332932087,
-0.687886036105895, 1, 1]`, and embedded weights `[0, 0, 0, 0, 0, 1]`.
The upstream `H` interpolation matrix is intentionally not represented by the
crate's regular-ODE recorder; this is the existing documented limitation for
the Rosenbrock family and does not alter endpoint or fixed-step integration.

## Native registration and coverage

`Rodas4` is publicly exported from `src/lib.rs:65`, registered as an adaptive
algorithm at `src/rosenbrock_extended.rs:1170`, and dispatched through the
shared six-stage `perform_rodas` kernel at `:1235` and `:1524`. The kernel
reuses one `I - gamma*h*J` factorization for all stages, applies the pinned
time-derivative and stage matrices, forms the primary update and btilde error,
and supports analytic or finite-difference Jacobians, callbacks, backward
spans, adaptive/fixed scheduling, and endpoint/save-at recording.

Existing evidence includes:

- fixed-step fourth-order convergence and adaptive stiff nonautonomous solves
  in `src/rosenbrock_extended.rs` tests;
- analytic-vs-finite-difference Jacobian agreement and callback/save-at tests;
- callback-free allocation invariance in `tests/rosenbrock_driver.rs`;
- `Rodas4` adaptive and fixed rows in
  `examples/rosenbrock_extended_compliance.rs` and
  `tests/julia/rosenbrock_extended.jl`.

The scope is regular ODE only. DAE, SDE, AMF, and wrapper variants are not
claimed by this audit.

## Verification

The focused Rust Rosenbrock tests pass: 12 passed. The full required Rust
target suite also passes: 106 library tests plus all integration and example
targets. The compliance example emits the Rodas4 rows
`rodas4_adaptive,5.40302305990256637e-1` and
`rodas4_fixed,2.71828182843933241e0`.

The required gates passed from this worktree:

```text
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check
cargo run --quiet --release --example rosenbrock_extended_compliance
```

The Julia gate was not run because `Get-Command julia` returned no command.
Retry with the exact `JULIA-PATH` (the pinned Julia executable path supplied by
the coordinator environment), using `julia --project=. tests/julia/runtests.jl`.

Status: audit complete; implementation already present; no solver changes
required.
