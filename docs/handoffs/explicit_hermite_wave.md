# Explicit RK accepted-step Hermite wave

## Scope

This wave wires the accepted-step Hermite recorder service into the generic
explicit Runge–Kutta kernel, exercising RK4 as the representative simple
family. No other solver family or public option was changed.

## Implementation

- `StepKernel::record_dense_step` is an optional lifecycle hook. Kernels that
  do not provide accepted dense data retain the endpoint fallback.
- The shared driver calls the hook only when `save_at` is configured, after
  callback root truncation/effect preparation and before forcing a callback
  right-limit state. Rejected attempts never reach the hook.
- Explicit RK uses its accepted left-stage derivative and evaluates the
  pre-effect right endpoint into existing workspace scratch. A borrowed
  Hermite segment samples `save_at` without a per-step allocation.
- Endpoint-only solves keep their prior lifecycle and allocation behavior.

## Tests and gates

`tests/explicit_dense.rs` covers RK4 cubic forward/backward asymmetric samples,
exact save-at endpoints, and rejected adaptive explicit attempts. Existing
allocation invariance tests remain green; the dense path uses stack-borrowed
segment metadata and workspace scratch rather than allocating per step.

The complete Rust gates and pinned Julia environment/full compliance suite were
run on this branch and passed.

## Limitations

Continuous callback root localization still uses the existing problem-local
linear probe because `src/problem.rs` is outside this bounded wave. Retained
post-solve dense queries and method-specific polynomial interpolants remain
future Phase 6 work. The endpoint RHS evaluation is intentionally scoped to
`save_at` solves and is not performed for endpoint-only output.
