# FRK65 handoff

## Scope

Implemented the regular fitted six-stage-order explicit `FRK65` algorithm
from OrdinaryDiffEqLowOrderRK at pinned revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. Exact upstream references are
`lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:256-279`,
`lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:830-1008`, and
`lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:897-1090`.

## Implementation

`Frk65` is exported from `src/lib.rs` and owns a nine-stage FSAL kernel. The
Butcher coefficients, nodes, embedded estimator, and frequency-fitted rational
weights are copied from the pinned `FRK65ConstantCache`; `Frk65::new(omega)`
selects the phase estimate and `Default` uses the upstream `omega = 0.0`.
Adaptive error is measured as the main update minus the upstream embedded
`utilde` residual. Accepted-step Hermite dense output, backward integration,
callbacks, and fixed/adaptive modes use the shared driver lifecycle.

The low-order compliance example and Julia fixture include a fixed-step
`frk65` endpoint case. Focused Rust tests cover sixth-order convergence and
adaptive endpoint accuracy; allocation-sensitive work uses persistent kernel
buffers (no per-step stage allocations).

## Validation

Passing in the isolated worktree:

- `cargo fmt -- --check`
- `cargo test --all-targets` (97 library tests plus integrations/examples)
- `cargo clippy --all-targets -- -D warnings`
- `git diff --check`

Julia pinned/full suites are pending because no `julia` executable is installed
or available on `PATH`. Retry `julia --project=tests/julia
tests/julia/pinned_environment.jl --check` followed by
`julia --project=tests/julia tests/julia/runtests.jl` after Julia is installed
and the pinned test environment is restored.

Implementation commit: `4720489ab7959052af405e3bff2e64a8a5eab469`.
