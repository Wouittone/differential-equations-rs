# Reusable stepping contract

The additive public `stepping` module provides `ExplicitRungeKuttaStepper<'a>`
and `RknStepper<'a>`. Each borrows its validated tableau for `'a`, owns fixed
workspace buffers, and owns the last accepted time and state. Construction copies
initial slices once. Attempts, acceptance, rejection, and same-shape reset reuse
storage. No trajectory is retained.

`ExplicitRungeKuttaStepper::new(&tableau, time, state)` creates a stepper.
`attempt(step, &mut rhs)` accepts a signed step and a mutable closure
`FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>`. It returns
`Result<StepView<'_>, StepError<E>>`; the borrowed view exposes candidate,
component error, optional second embedded error, stage derivatives, step interval,
and cumulative statistics. Errors preserve the original `E`. A zero step produces
an unchanged candidate without evaluating the RHS. A nonzero step that cannot
advance floating-point time is rejected. `attempt_to(endpoint, proposal, rhs)`
clamps the proposal to the endpoint, preserving direction.

An attempt never commits state. Exactly one successful candidate can be pending;
call `accept()` or `reject()` before attempting again. Failed RHS evaluation
leaves accepted state unchanged and no candidate pending. A borrowed result cannot
outlive the next mutable operation. `state()`, `time()`, `statistics()`,
`reset(time,state)`, and `invalidate_derivative()` expose lifecycle operations.
`inject_derivative(slice)` supplies the derivative at the accepted state;
`current_derivative()` returns a valid cached derivative when available. Rejection
retains the accepted-state derivative; acceptance retains the final stage only
for a validated FSAL tableau. Caller parameter/control changes require explicit
invalidation. State-changing callbacks call reset. Host interruption simply stops
calling attempt and retains the accepted state. Cached derivatives include every
term supplied by the closure, including control.

RKN uses `FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>` and separate
position/velocity candidate and embedded-error slices; stages are accelerations.
A policy explicitly distinguishes velocity-independent formulas from legacy
frozen-velocity compatibility. Velocity-dependent stages require a supplied
velocity-stage tableau. Stage access lets hosts apply exact custom error weights
without recapturing RHS calls. Norms consume old/candidate/error slices and are
independent of stepping. Controllers likewise consume a scalar error and own
history plus the actual next proposal. Existing solve APIs/defaults are unchanged.

A continuation owns the complete stepper and controller, avoiding unchecked
method/state token interchange; moving it preserves caches and proposal exactly.
No implicit cloning, serialization, recording, callbacks, or allocations occur
inside attempt/accept/reject. Retained output and dense-export allocations are
separate explicit operations.
