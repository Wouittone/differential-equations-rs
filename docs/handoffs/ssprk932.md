# SSPRK932 handoff

## Scope

This wave adds the regular explicit ODE `SspRk932` facade and exports it from
the crate. The nine-stage SSPRK932 main update is expanded into a ten-stage
Butcher tableau with an endpoint-only stage reserved for the adaptive embedded
residual. The shared explicit driver therefore covers fixed and adaptive
stepping, backward spans, callbacks, `save_at`, and endpoint dense output.
Limiter and threading hooks from the Julia wrapper are intentionally not part
of this regular ODE facade. Callback-free fixed steps have a step-count
invariant allocation shape (`tests/ssprk932_allocations.rs`).

## Pinned upstream evidence

The implementation is matched to OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

* `lib/OrdinaryDiffEqSSPRK/src/algorithms.jl:178-193` — adaptive-capable
  `SSPRK932` declaration and order metadata.
* `lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl:1190-1333` — six equal
  SSP substeps, endpoint embedded stage, and three-stage second branch.
* `lib/OrdinaryDiffEqSSPRK/src/ssprk_caches.jl:1022-1070` — mutable and
  constant cache shapes.

The main update uses the exact recurrence expansion
`u6 = uprev + dt/6*(f1+...+f6)`,
`u7 = (3*uprev + dt/2*f1 + 2*u6)/5`, then three `dt/6` increments. The
primary weights are `[1/6, 1/15, 1/15, 1/15, 1/15, 1/15, 0, 1/6, 1/6, 1/6]`.
The source's adaptive expression is
`(uprev + 6*u6 + 6*dt*f_endpoint)/7`; its derivative weights sum to `12/7`,
so the Rust embedded residual normalizes the endpoint coefficient to `1/7`
to retain a consistent adaptive estimate while preserving the exact primary
SSPRK932 update.

## Validation

Passed on the isolated branch:

* `cargo fmt -- --check`
* `cargo clippy --all-targets -- -D warnings`
* focused `cargo test ssprk --all-targets` (13 SSPRK tests) and
  SSPRK932 allocation-invariance test
* `cargo run --quiet --example ssprk_extended_compliance` (emits
  `ssprk932,3.43656364435863182e0`)

`git diff --check` passed. The full `cargo test --all-targets` gate reaches 106
passing tests but fails in the pre-existing
`rosenbrock_extended::tests::methods_have_their_expected_fixed_step_orders`
assertion (`ratios[9] > 14.0`), outside this SSPRK932 change. Julia was not
available on this worker (`julia` is not on PATH); rerun the pinned fixture
from a coordinator environment with Julia installed.
