# Rodas5 parity handoff

## Summary

Implemented the native regular-ODE `Rodas5` solver from pinned
OrdinaryDiffEqRosenbrockTableaus revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. The eight-stage, fifth-order
stiffly accurate method uses the shared allocation-free Rosenbrock/Rodas
kernel.

## Upstream source and revision

- `lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl:23-73`
  defines `RODAS5A`, `RODAS5C`, `RODAS5c`, `RODAS5d`, `RODAS5H`, and
  `Rodas5Tableau`.
- `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl:75-82` identifies `Rodas5`
  as a fifth-order adaptive Rosenbrock method.
- `lib/OrdinaryDiffEqRosenbrock/src/alg_utils.jl:34,59` records order five
  and non-FSAL behavior.
- `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl:729-862`
  provides the regular ODE stage/update/error path used by the Rust driver.

The Rust tableau uses `gamma = 0.19`, the exact eight-stage `A` and `C`
matrices, nodes `[0, 0.38, 0.3878509998321533, 0.483971893787384,
0.457047700881958, 1, 1, 1]`, time weights
`[0.19, -0.18230792253337146, -0.3192318321868749, 0.3449828624725343,
-0.37741756439208984, 0, 0, 0]`, primary weights equal to the final stage,
and embedded weights `[0, 0, 0, 0, 0, 0, 0, 1]`. The upstream `H` matrix is
not needed by the crate's regular-ODE recorder; `save_at` uses shared output
recording as for the other Rosenbrock methods.

## Files changed and public API

- `src/rosenbrock_extended.rs`: public `Rodas5`, pinned tableau, driver
  registration.
- `src/lib.rs`: public re-export.
- `tests/rodas5.rs`: fixed/adaptive, backward, analytic-Jacobian, callback,
  and `save_at` coverage.
- `tests/rodas5_allocations.rs`: callback-free allocation invariance.
- `examples/rosenbrock_extended_compliance.rs` and
  `tests/julia/rosenbrock_extended.jl`: compliance row.

## Verification

The focused tests pass (3 tests total), including allocation invariance. The
full required Rust gates passed in this worktree:

```text
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check
cargo run --quiet --release --example rosenbrock_extended_compliance
```

The compliance example produced:

```text
rodas5_adaptive,5.40302305521310755e-1
rodas5_fixed,2.71828182845907396e0
```

Julia was unavailable (`julia` is not recognized). Retry the pinned
environment and full Julia suites when the exact executable is available:

```powershell
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl
```

## Numerical differences, performance, and limitations

The Rust driver shares the existing finite-difference or analytic Jacobian,
LU factorization, adaptive controller, callback, backward-time, and endpoint
recording semantics. No per-step stage allocation is introduced; the
allocation test confirms step-count invariance. As with the existing native
Rosenbrock family, this port claims regular first-order ODE behavior only and
does not claim DAE residual, SDE, wrapper, or method-specific stiff dense-H
interpolation behavior.

Recommended next task: port the next independent native Rosenbrock/Rodas
tableau and add its aggregate compliance row.
