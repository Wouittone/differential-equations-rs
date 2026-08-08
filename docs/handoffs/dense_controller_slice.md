# Dense-output/controller service slice

## Scope

This bounded Phase 6 wave adds the accepted-step Hermite service and protects
controller history across callback discontinuities. It intentionally does not
change solver-family modules, public options, or the continuous-callback
interpolator; those require each family to expose its accepted stage/derivative
data in a later wave.

## Changes

- `HermiteSegment` now validates finite endpoint data, handles forward and
  backward intervals, returns exact endpoint states, and rejects out-of-domain
  or wrong-dimension queries.
- `TrajectoryRecorder::record_step_dense` samples `save_at` targets through a
  supplied accepted-step segment using the recorder's existing scratch buffer.
  The existing endpoint-linear `record_step` remains the compatibility path
  until kernels provide a segment.
- The shared adaptive driver resets controller error history before recording
  an accepted step whose callback mutated the endpoint state. The current
  accepted error is then installed as the new history value.

## Validation

Inline tests cover backward Hermite endpoint exactness, dimension/domain and
finite-data checks, cubic midpoint/asymmetric save-at values, and controller
history reset semantics. The full Rust and pinned Julia gates were run before
the commit; no solver endpoint or allocation behavior changed because existing
kernels continue to use the compatibility recorder path.

## Limitations and follow-up

No solver kernel currently supplies `record_step_dense`; wiring method-specific
stage data, continuous roots, retained post-solve segments, and public dense
queries belongs to the subsequent per-family dense-output waves. No controller
kind/options were added here.
