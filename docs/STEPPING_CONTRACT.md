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

## Storage and allocation accounting

Workspace construction and coefficient parsing are setup costs. For dimension
`n`, stages `s`, and auxiliary dimension `m`, the current persistent storage is:

| Workspace | f64 elements (excluding coefficients) | Other storage |
|---|---:|---|
| Owned explicit RK | `(s + 6) n` | small scalar state |
| Borrowed explicit RK | `(s + 5) n` | caller owns `n` state elements |
| Owned RKN | `(s + 9) n` | small scalar state |
| Borrowed RKN | `(s + 7) n` | caller owns `2n` state elements |
| Owned Rosenbrock | `2 n² + (2s + 9) n` | `n` LU pivot indices |
| Borrowed Rosenbrock | `2 n² + (2s + 8) n` | caller owns `n` state elements; `n` pivots |
| Mixed RKN | owned RKN plus `(s + 4) m` | small scalar state |

Each vector allocates once during construction. Dimension overflow is checked;
changing shape requires constructing a new workspace. Reset requires the existing
shape and never resizes. Coefficients are borrowed from a validated tableau and
are not copied into the workspace. Fixed arrays coerce to the borrowed RK/RKN/Rosenbrock
constructors (`from_buffer`/`from_buffers`). Acceptance copies the candidate into
the accepted-state buffer; it never constructs a solution/trajectory object.

The allocation tests instrument accepted and rejected attempts, numerical
Rosenbrock differentiation, mixed STM propagation and reset after setup. These
paths perform zero allocator calls. RHS, derivative hooks, norms and observers
are user code and may allocate themselves; this is outside the kernel guarantee.
Recording output, building owned dense segments, cloning a tableau, serializing
results, formatting errors, and resizing caller output collections are explicit
additional costs. The output-free driver itself retains no trajectory and
`copy_state_into` requires a correctly sized destination (it does not resize).

## Continuation and cache validity

`Continuation<S>` moves the entire workspace and controller, retaining method
identity, dimensions, accepted state, cached derivatives, accepted-error history
and `next_step`. The last interval is not a substitute for the next proposal.
Moving a continuation never allocates and cannot accidentally apply a cache to a
different tableau. It is an in-memory ownership token, not a serialization format.

Every cached derivative includes all terms supplied by the RHS, including control.
Rejection preserves the derivative at the unchanged accepted state. RK acceptance
retains the final stage only for an FSAL method; RKN and Rosenbrock do not assume
FSAL. After parameter/control changes, call `invalidate_derivative`; after state
changes, call `reset`. After a discontinuity, also reset controller history.
Hooks may have observable side effects. If an evaluation fails after earlier
stages succeeded, the accepted-state derivative may remain cached; invalidate it
before retrying if the failed application operation changed the mathematical RHS.

## Rosenbrock conventions and finite differences

`RosenbrockStepView::solved_stages` contains scaled solved increments, whereas
`stage_derivatives` contains unscaled physical RHS evaluations. Their counters
are separate. The Jacobian is row-major `J[row*n+column] = ∂f_row/∂y_column`.
The time hook returns the explicit partial `∂f/∂t` while holding `y` fixed, never
the total derivative `∂f/∂t + J f`. Analytic hooks preserve arbitrary application
errors and avoid numerical probe calls. The matrix factorization is shared by
all stages of an attempt; differentiation is reused after rejection.

Numerical time differentiation now uses two direction-following probes within
the attempted interval, a physical time scale independent of the absolute epoch,
and optional smooth-domain bounds. Three-point one-sided differentiation is
quadratic-exact and uses actual representable offsets. If only one probe is
representable, it falls back to a first difference. This adds one RHS call versus
the previous one-probe policy, in exchange for removing the epoch-sized probe
error. Analytic partials remove both probe calls. Large-epoch stage-time rounding
still exists because `f64` cannot represent arbitrary sub-ULP times; shifting the
independent variable to a local epoch is useful when that resolution matters.


## Representable intervals at large epochs

All public reusable kernels first compute the finite endpoint `t_end = t + h`,
then use `h_effective = t_end - t` in their numerical formulas. This aligns the
advanced state with the returned clock, especially when the requested `h` is not
representable relative to a large epoch. A nonzero proposal that rounds back to
`t` fails with `TimeResolution`. The effective interval may differ from the
proposal by floating-point rounding; external controllers must assess
`view.end_time - view.start_time`. RK/RKN/Rosenbrock views expose both times, and
mixed views expose them in their physical partition. The built-in adaptive and
root drivers already use the effective interval. Initial-step estimation uses
scaled RMS norms with a log-domain fallback to avoid spurious overflow.
