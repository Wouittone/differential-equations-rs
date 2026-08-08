# Phase 6 dense-output and controller audit

## Summary

The pinned upstream does not treat dense output as a recorder feature. It is an
accepted-step service shared by `saveat`, continuous callbacks, and solution
queries. A method either supplies a special interpolant from its saved stages,
possibly materializing extra stages on demand, or participates in the core
Hermite fallback. The local crate currently routes both `save_at` and root
location through endpoint-linear interpolation and does not retain segments in
`Solution`; the newly landed crate-private `HermiteSegment` is a sound storage
primitive, but is not wired into the driver.

The pinned upstream controller is likewise stateful and centralized. PI is the
generic default, I is selected for VCABM and several BDF methods, PID and
predictive controllers are supported, and accepted/rejected-step history is
owned by a per-solve controller cache. The local shared driver implements only
a stateless proportional factor with a one-step rejection cap. It has no
`dtmin`, `tstops`, controller selection, or distinct proposed-versus-clipped
step state.

Phase 6 should therefore land in two vertical slices: (1) one accepted-step
dense service consumed by save-at and continuous roots, then retained segments
and public queries; (2) a controller state object plus step-bound/tstop
scheduling. Method-specific interpolants should then replace Hermite in the
family order in the overnight plan. Merely changing the recorder to cubic
Hermite would leave event roots, lazy stages, stiff methods, and post-solve
queries inconsistent.

## Audit basis

- Local base: `831296cacab35a8734a313626cfbae5482ff5884`.
- Upstream checkout: `D:/Source/_review/OrdinaryDiffEq.jl`.
- Verified upstream revision: `211142263781255a9aa2f910f6760b9f18ec29c8`
  (detached and clean during the audit).
- Scope: native regular initial-value ODE behavior only. Singular mass
  matrices/DAE initialization, SDE/DDE/BVP/PDE behavior, and external wrappers
  are not design targets.

## Current local behavior and concrete gaps

| Area | Current local source | Observed behavior | Phase 6 gap |
|---|---|---|---|
| Dense seam | `src/solution.rs:22-95` | `DenseSegment` and owning cubic `HermiteSegment`, with checked dimensions/time | Not connected to kernels, recorder, callbacks, or `Solution` |
| Save-at | `src/solution.rs:97-189` | `TrajectoryRecorder::record_step` linearly blends accepted endpoints | Must call the accepted step's method-specific service; no extra RHS evaluation unless the method requires a lazy stage |
| Continuous roots | `src/problem.rs:305-425` | 52 bisections, each using the file-local linear interpolator | Must evaluate the exact same accepted-step segment as save-at/query; event state must not depend on `save_at` settings |
| Solution query | `src/solution.rs:191-250` | Stores only sampled times/states/stats | Needs retained step segments, interval lookup, checked query API, and a discontinuity convention |
| Dense metadata | `src/coefficients.rs:48-60,251-267` | Schema distinguishes generic Hermite, free-stage polynomial, and lazy-extra-stage polynomial | Generated metadata is not consumed by a runtime interpolator |
| Shared driver | `src/integrator.rs:171-331` | Calls callbacks and recorder after an accepted trial; clips only final endpoint | Needs prepare/evaluate/freeze dense lifecycle before callbacks, then tstop-aware scheduling |
| Controller | `src/integrator.rs:8-51,333-341` | Immutable proportional constants; zero error grows by `max_factor`; nonfinite error shrinks by `min_factor` | No PI/PID/predictive cache, history, deadband, first-step bound, or controller choice |
| Step contract | `src/integrator.rs:102-169` | `StepEstimate` exposes only error norm; kernel has attempt/accept/reject | Needs dense-step preparation/evaluation and explicit controller-order metadata without exposing family caches publicly |
| Options | `src/solver.rs:16-56,178-186` | Tolerances, initial/max step, adaptive, save and ordered `save_at` | No `dense`, `dtmin`, `tstops`, controller selection, or post-solve interpolation policy |
| Event discontinuity | `src/solution.rs:111-125` and `src/integrator.rs:273-311` | Equal-time records overwrite; callbacks may mutate candidate before commit | Cannot retain both left and right limits at an event; a side/duplicate-time policy is required before public queries |

`src/rosenbrock_extended.rs:1-7` explicitly records that stiff dense
interpolants were deferred. The same functional gap exists for TRBDF2,
multistep, second-order, and partitioned methods even where their step kernels
are already shared.

## Verified pinned-upstream dense-output behavior

### Core lifecycle

1. The core `ode_addsteps!` contract populates the derivative/stage vector and
   supports method-specific extra-stage work; current-integrator interpolation
   calls it before dispatching to the method interpolant
   (`lib/OrdinaryDiffEqCore/src/dense/generic_dense.jl:102-224`).
2. Post-solve evaluation maps absolute time to normalized `Theta`, uses linear
   interpolation only when stage data are unavailable, and otherwise performs
   lazy addsteps plus the method dispatch
   (`lib/OrdinaryDiffEqCore/src/dense/generic_dense.jl:705-726`,
   `lib/OrdinaryDiffEqCore/src/dense/generic_dense.jl:795-808`).
3. Solve initialization derives `dense` and `calck` from `save_everystep`,
   callbacks, and `saveat`; it also carries `dtmin`, `dtmax`, `tstops`, and a
   controller choice (`lib/OrdinaryDiffEqCore/src/solve.jl:130-175`).
4. Saving points crossed by an accepted step computes `Theta` and invokes
   `interp_at_saveat`; for regular ODEs that path deliberately performs
   polynomial interpolation even if retained post-solve dense output is off
   (`lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl:307-400`).
5. `tstops` are direction-normalized and include the final endpoint
   (`lib/OrdinaryDiffEqCore/src/solve.jl:978-996`). Before a step, upstream
   preserves the proposed `dt`, clips the attempted `dt` to the stop, and later
   restores the proposal so an artificial clip does not poison controller
   history (`lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl:225-303`,
   `lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl:580-604`).

These are upstream observations. The Rust interfaces below are local design
inferences intended to preserve those semantics without copying Julia's cache
hierarchy.

### Family map

| Local family | Pinned-upstream evidence | Required local representation | Current gap |
|---|---|---|---|
| Tsit5 | Special interpolation dispatch and stage-polynomial coefficients are in `lib/OrdinaryDiffEqTsit5/src/interpolants.jl:1-113` | Free-stage polynomial over accepted-step stages; no endpoint RHS solely for interpolation | All sampling/root work is linear |
| Generic explicit RK | Core supports saved-stage interpolation/addsteps (`lib/OrdinaryDiffEqCore/src/dense/generic_dense.jl:102-224`); the generic RK tests require tableau dense coefficients | Schema-driven polynomial when `DenseRecord` supplies it, otherwise checked Hermite | Runtime ignores `DenseRecord` |
| Owren-Zen 3/4/5 | Dedicated interpolants begin at `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:304-541`, `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:541-865`, and `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:865-1328`; addsteps for higher variants are in `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_addsteps.jl:284-404` | Method rows plus any required end/extra derivative, reusable in-step scratch | Only metadata comments exist locally |
| BS5 | Dedicated interpolant in `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:1329-1475`; lazy/non-lazy extra-stage path begins at `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_addsteps.jl:611-700` | Lazy extra-stage polynomial; cache invalidated on rejection/callback mutation | Extra interpolation stages are explicitly outside the local shared kernel |
| Vern6/7/8/9 | Four distinct high-order interpolants at `lib/OrdinaryDiffEqVerner/src/interpolants.jl:24-183`, `lib/OrdinaryDiffEqVerner/src/interpolants.jl:183-379`, `lib/OrdinaryDiffEqVerner/src/interpolants.jl:379-668`, and `lib/OrdinaryDiffEqVerner/src/interpolants.jl:668-810` | Per-method dense coefficient rows and lazy-stage materialization; do not flatten to cubic Hermite | Vern kernels expose endpoint stepping only |
| SSP explicit | Several low-order methods share explicit special interpolants in `lib/OrdinaryDiffEqSSPRK/src/interpolants.jl:1-198` | Family polynomial where upstream supplies one; otherwise Hermite fallback | Linear sampling for every SSP method |
| Rosenbrock23/32 | Dedicated stiff interpolation from Rosenbrock stage vectors in `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_interpolants.jl:1-178` | Stiff stage-polynomial segment; preserve the method's stage scaling | Locally documented as omitted |
| Rodas/Rosenbrock extended | Combined-cache dense formulae and derivatives are in `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_interpolants.jl:178-697` | Coefficient-driven stiff polynomial using accepted `k` data; no generic endpoint blend | Locally documented as omitted |
| TRBDF2 and SDIRK | SDIRK retains two endpoint derivatives for the core path (`lib/OrdinaryDiffEqSDIRK/src/sdirk_perform_step.jl:1-15`), which then participates in generic dense dispatch | Cubic Hermite first; a special segment only where pinned evidence identifies one | Local TRBDF2 recorder remains linear |
| Fixed/variable Adams | Upstream retains two derivatives for generic dense interpolation, including adaptive variants (`lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_bashforth_moulton_perform_step.jl:14-37`, `lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_bashforth_moulton_perform_step.jl:518-586`) | Endpoint Hermite segment initially; retain independent segment data because rolling history is overwritten | No retained segment; linear save-at/root |
| Symplectic/partitioned | Upstream step caches retain two derivative entries (`lib/OrdinaryDiffEqSymplecticRK/src/symplectic_perform_step.jl:1-45`) and use the generic dense machinery | Representation-aware Hermite over the flattened public state, with position/velocity derivatives captured before caches advance | Local second-order-specific interpolation does not flow through the shared recorder/root/query service |

The ranges above establish dispatch/data shape, not a claim that every member
has the same formal interpolation order. Formal order and extra-stage counts
must come from the generated dense metadata and one per-algorithm compliance
fixture.

## Verified pinned-upstream controller behavior

- Controller configuration and mutable cache are separate; the generic
  lifecycle defines `stepsize_controller!`, acceptance, accepted-step update,
  and rejected-step update (`lib/OrdinaryDiffEqCore/src/integrators/controllers.jl:1-76`,
  `lib/OrdinaryDiffEqCore/src/integrators/controllers.jl:206-250`).
- The generic algorithm default is PI, with order-scaled gains and safety
  defaults in `lib/OrdinaryDiffEqCore/src/alg_utils.jl:708-791`. Generic bounds
  include `qmin = 0.2`, `qmax = 10`, a large first-accepted-step upper bound,
  and an implicit-method steady band
  (`lib/OrdinaryDiffEqCore/src/alg_utils.jl:378-420`,
  `lib/OrdinaryDiffEqCore/src/alg_utils.jl:820-857`).
- The I controller has explicit zero-error handling and a steady-factor band
  (`lib/OrdinaryDiffEqCore/src/integrators/controllers.jl:600-688`). PI owns
  prior error/factor state and updates it only at the controller lifecycle
  points (`lib/OrdinaryDiffEqCore/src/integrators/controllers.jl:705-843`). PID
  owns two-step history and clamps zero errors before its formula
  (`lib/OrdinaryDiffEqCore/src/integrators/controllers.jl:912-1065`).
- The predictive/Gustafsson controller incorporates nonlinear iteration count
  and prior accepted/rejected data
  (`lib/OrdinaryDiffEqCore/src/integrators/controllers.jl:1170-1237`). This is
  relevant to future SDIRK/BDF work, but should not be invented for current
  explicit kernels before nonlinear-solver state is available.
- VCABM overrides the generic PI default with I
  (`lib/OrdinaryDiffEqAdamsBashforthMoulton/src/alg_utils.jl:1-17`), and DP5
  overrides PI gains (`lib/OrdinaryDiffEqLowOrderRK/src/alg_utils.jl:31-42`).
  These family exceptions must live in coefficient/algorithm metadata rather
  than conditional branches in the driver.
- Accepted proposals are clamped to `dtmax` and time-dependent `dtmin`, while
  rejection and nonlinear-failure paths run through the same bounds/tstop
  logic (`lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl:75-123`,
  `lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl:881-906`).

### Controller parity map

| Behavior | Local status | Required action |
|---|---|---|
| Initial step | Kernel estimate plus option override | Preserve, but feed a controller-owned `proposed_dt` and clamp against direction, `dtmin`, `dtmax`, next tstop |
| Proportional/I | Stateless proportional formula | Introduce stateful `I` variant; keep current formula as a temporary explicit compatibility mode only |
| PI default | Missing | Make generic adaptive default after Julia fixtures lock endpoint/step-count tolerances |
| PID | Missing | Add as an opt-in controller after PI; clamp error history away from zero |
| Predictive | Missing | Defer until nonlinear iteration counts are standardized across implicit kernels |
| Rejection | Boolean plus one accepted-step cap | Controller owns rejection state and history update; kernel rejection only invalidates method caches |
| Failure shrink | Fixed factor in config | Keep centralized and bounded; nonlinear failure may supply iterations/status later |
| Deadband | Missing | Apply controller-specific steady interval to avoid step-size chatter |
| First accepted step | Same `max_factor` as later steps | Separate first-step growth bound |
| `dtmin`/`dtmax` | Only `max_step` | Add validated signed-magnitude bounds and exact failure when tolerance cannot be met at `dtmin` |
| `tstops` | Missing | Direction-normalized ordered schedule; distinguish proposed from clipped attempted dt |
| Zero/nonfinite error | Zero grows maximally; nonfinite shrinks minimally | Retain finite guards, but update PI/PID history with bounded sentinel values, never NaN/zero division |
| Statistics | Driver owns accepted/rejected/evaluation counts | Keep ownership in driver; controller state must not increment solver stats |

## Implementation-ready local interfaces

The following signatures are a recommendation, not upstream Rust API:

```rust
pub(crate) trait CurrentStepDense<F, P> {
    fn prepare_dense_step(
        &mut self,
        problem: &mut F,
        params: &mut P,
        t0: f64,
        dt: f64,
        u0: &[f64],
        u1: &[f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>;

    fn interpolate_dense_step(
        &self,
        time: f64,
        output: &mut [f64],
    ) -> Result<(), SolveError>;

    fn freeze_dense_step(
        &self,
        valid_end: f64,
    ) -> Result<OwnedDenseSegment, SolveError>;

    fn invalidate_dense_step(&mut self);
}

pub(crate) enum OwnedDenseSegment {
    Hermite(HermiteSegment),
    Polynomial(PolynomialSegment),
}

pub(crate) struct ControllerState {
    kind: ControllerKind,
    proposed_dt: f64,
    accepted_steps: usize,
    rejected_last_attempt: bool,
    history: ControllerHistory,
}

pub(crate) enum ControllerHistory {
    I,
    Pi { previous_error: f64, previous_factor: f64 },
    Pid { error_n1: f64, error_n2: f64 },
    Predictive { previous_dt: f64, previous_error: f64 },
}
```

Invariants:

- `prepare_dense_step` is called only after a numerically accepted trial and
  before any callback effect mutates the endpoint. It may perform lazy RHS
  evaluations and must count them in the driver-owned `SolverStats`.
- `interpolate_dense_step` is allocation-free, dimension-checked, valid only on
  the current accepted interval, and is the sole source for save-at and
  continuous root evaluation.
- `freeze_dense_step` allocates only when retained post-solve queries are
  requested. It owns all coefficient/state data needed after family caches are
  reused. `valid_end` permits an event to truncate the domain without
  renormalizing a polynomial whose coefficients use the original attempted
  `dt`.
- A rejected step and a callback endpoint mutation invalidate prepared/lazy
  dense data. Endpoint-only discrete callbacks force-save the right-limit state
  but do not rewrite the left-limit polynomial.
- `PolynomialSegment` stores `t0`, interpolation `dt`, valid domain, dimension,
  polynomial order, and contiguous row-major coefficients. Construction checks
  finite ordered bounds and exact coefficient dimensions. Evaluation uses a
  caller-provided output slice and Horner form.
- Do not put boxed trait objects or per-evaluation `Vec` creation on the hot
  path. A small crate-private enum is sufficient for the dense `f64` backend.
- Controller configuration belongs to `SolveOptions`/algorithm metadata;
  history belongs to one `ControllerState` per solve. The controller reads the
  error estimate/order and returns proposals; only the driver applies direction,
  endpoint/tstop clipping, `dtmin`/`dtmax`, and updates stats.

Recommended accepted-step order:

1. Kernel completes trial and error estimate.
2. Controller decides acceptance; rejected trials invalidate dense state.
3. Kernel prepares current-step dense data.
4. Continuous callbacks locate the earliest root with that interpolant. If a
   root truncates the step, record/interpolate only through the root.
5. Recorder samples all crossed `save_at` points from the same interpolant.
6. Freeze the left-limit segment only if retained queries are enabled.
7. Apply callback effect, invalidate family FSAL/dense caches as required, and
   force-save the right-limit state under the chosen discontinuity policy.
8. Commit kernel/controller accepted-step state and compute the next proposed
   `dt`; clip only the attempted copy to the next tstop/end time.

## Tests and benchmarks required for Phase 6

Dense correctness:

- For every public algorithm, compare midpoint and two asymmetric in-step
  queries against pinned Julia, in forward and backward integration.
- Assert endpoint exactness, output-dimension errors, out-of-domain errors, and
  derivative/stage cache invalidation after rejection and callback mutation.
- Verify `save_at`, continuous root location, and post-solve query return the
  same state at an identical interior time.
- Add targeted fixtures for Tsit5, OwrenZen3/4/5, BS5 lazy extra stages,
  Vern6-9, Rosenbrock23/32, Rodas4/5P, TRBDF2, fixed/variable Adams, and each
  second-order representation.
- Cover an event that truncates a step and a discrete callback at the same time;
  explicitly test left/right limits and duplicate-time lookup behavior.
- Count RHS calls so free interpolants remain free and lazy-stage interpolants
  pay their upstream-required cost once per accepted step, not once per query.

Controller correctness:

- Unit-test I/PI/PID factor formulas, zero/tiny/nonfinite error, deadband,
  first-step growth, rejection history, backward time, and `dtmin` exhaustion.
- Compliance fixtures should compare accepted/rejected counts and a bounded
  sequence of attempted/proposed steps for representative Tsit5, Vern, DP5,
  VCABM, Rosenbrock, and TRBDF2 problems. Do not require bitwise Julia equality.
- Test multiple `tstops`, duplicate stops, stops near the current time, final
  endpoint, backward ordering, and restoration of the pre-clip proposal.

Performance:

- Benchmark save-at-heavy and root-heavy solves before/after each slice.
- Assert zero allocations per `interpolate_dense_step` after warm-up and no new
  accepted-step allocation when retained dense output is disabled.
- Benchmark retained polynomial segments against the current sampled solution;
  report bytes per accepted step by method/order.
- Benchmark controller-only overhead on a cheap scalar RHS. Do not add a large
  dependency to implement Horner evaluation, interval lookup, or four scalar
  controller formulas.

## Deferred dependencies and limitations

- Public `Solution::at` naming, extrapolation policy, and event left/right lookup
  require the Phase 5 solution/problem representation decision. The internal
  accepted-step service does not need to wait.
- Partitioned/second-order freezing needs the canonical flattened-state mapping
  from representation work; save-at/root use can first consume an in-step
  representation-aware evaluator.
- Predictive controller parity waits for a common nonlinear-solver outcome and
  iteration-count contract. BDF order-selection controllers wait for the BDF
  family implementation.
- Singular mass matrices and DAE residual initialization remain excluded. A
  nonsingular mass-matrix ODE may use the same dense/controller interfaces once
  its state derivative is defined consistently.
- This audit makes no sparse/GPU/distributed design claim and recommends no new
  dependency.

## Bounded first implementation task

Implement `CurrentStepDense` for the shared explicit-RK kernel and Tsit5 only,
using generated free-stage polynomial metadata where present and the existing
checked `HermiteSegment` otherwise. Route `TrajectoryRecorder::record_step` and
`locate_root` through that service, retain no segments, and make no public API
change. Add equivalence tests proving save-at and root interpolation agree and
allocation/RHS-count tests proving no hot-path allocation or unrequired extra
evaluation. This task is bounded to `src/integrator.rs`, `src/solution.rs`,
`src/problem.rs`, the shared explicit kernel/coefficient consumer, and focused
Rust tests. Controller/tstop work should be a separate task.

## Handoff

- **Files changed:** `docs/handoffs/dense_controller_audit.md` only.
- **Public APIs added:** none.
- **Upstream source and revision:** `D:/Source/_review/OrdinaryDiffEq.jl` at
  `211142263781255a9aa2f910f6760b9f18ec29c8`; all cited paths and ranges were
  resolved against that checkout.
- **Rust tests:** none required; report-only audit.
- **Julia tests:** none required; no Julia source/environment changed.
- **Commands run:** source-of-truth document reads; local `rg` call-site audit;
  upstream `git rev-parse`/status; extensive upstream `rg`; line-range resolver;
  `git diff --check`; final `git status --short --branch`.
- **Numerical differences:** none introduced. The report identifies current
  endpoint-linear sampling/root location versus pinned method-specific dense
  interpolation, and proportional versus stateful PI/default behavior.
- **Allocation/performance impact:** none introduced. Proposed in-step
  interpolation is allocation-free; allocation occurs only when freezing
  retained segments.
- **Known limitations:** formal dense order/extra-stage counts must be validated
  per generated record and fixture; public query/discontinuity API is deferred;
  predictive/BDF controller state is deferred as described above.
- **Follow-up dependencies:** coefficient runtime consumer; solution query and
  representation decisions; common nonlinear iteration reporting for
  predictive control.
- **Recommended next task:** the bounded shared-explicit/Tsit5 accepted-step
  dense service above, followed independently by `ControllerState` plus
  `dtmin`/`tstops` scheduling.
