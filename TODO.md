# TODO

Prioritized on 2026-09-20 after the Brahe and SatKit migrations to published
`differential-equations-rs = "=1.0.0"`. Planning baseline: main
`31decb9cbb1edfedf882d65280305f77b6b9cfb9` (also version 1.0.0).
Unchecked items are planned work, not implemented features or measured gains.

## Evidence and priorities

- [Migration API audit: 21 findings](https://github.com/Wouittone/satkit/blob/4b99262fa294f36d2cef468c35d1c59c78cfeef2/API_FRICTION.md)
- [Brahe results and reproduction](https://github.com/Wouittone/brahe/blob/338818fcb2161fac3ee6dd09e62c55ef42210561/RESULTS.md)
- [SatKit results and reproduction](https://github.com/Wouittone/satkit/blob/4b99262fa294f36d2cef468c35d1c59c78cfeef2/RESULTS.md)

P0 means a prerequisite for trustworthy replacement or for the rest of this plan;
P1 removes measured integration/performance barriers; P2 adds broader capability.
All are in scope; lower priority does not mean omitted. Dependencies determine
execution order within a priority. Acceptance criteria below are targets, not
claims that the current implementation satisfies them.

The explicit RK/RKN migrations replace stage kernels but retain host controllers,
error formulas, and interpolation. Repeated one-step whole solves add overhead;
these results do not establish intrinsic whole-solve performance. Brahe method
geometric-mean time ratios are 0.995–1.014 (effectively neutral), with peak live
heap up 2.5–9.3%. RKF78 allocation traffic falls about 20.5%, while other methods
increase about 8–35%. SatKit explicit methods are 3.7–13.7% slower on average,
with allocation traffic up 40–88%; the worst tight-tolerance LEO STM case is
about 71% slower. SatKit RODAS4 is 18.5% slower on average and has 1.8–21.3 times
the analytic orbital endpoint error at the same nominal tolerances. Its driver
and control policy differ, so this is not an equal-achieved-accuracy comparison.
Peak heap, allocation traffic, and process resident memory are distinct metrics.

Already present in 1.0.0: fallible RHS traits, RKN resource methods, Rosenbrock
and Rodas4, analytic Jacobians, events/time stops/requested saves, high-order
dense output, allocation-free `try_interpolate_into`, and parallel ensembles.
The tasks below extend interoperability and control of these capabilities.
Native solves already reuse internal workspace; existing allocation tests cover
step-count scaling. Owned array conversion also already works. The missing
capabilities are public reuse and borrowed/fixed storage without copies, not
these existing facilities. Retaining an unbounded trajectory necessarily uses
additional storage and is outside the zero-allocation stepping target.

## Prioritized implementation backlog

Each task has one primary owner in the schedule below. Sub-bullets inherit its
priority and form its completion checklist; all must be addressed to close it.

- [ ] **T01 · P0 · Public reusable stepping contract** — Owner A; no dependency.
  - Expose an integrator/workspace with attempt, accept, reject, reset/restart,
    signed step sizes, endpoint handling, and explicit state/lifetime invariants.
    Do not expose unstable private kernel details accidentally.
  - Return a borrowed method-aware step result containing candidate state,
    component embedded errors, accessible stages or a dense segment, and RHS,
    Jacobian, factorization, acceptance/rejection statistics. Distinguish solved
    Rosenbrock stages from RHS evaluations.
  - Accept when external host controllers can drive RK and RKN without a fresh
    solve or RHS-stage capture, including rejection, backward/zero steps,
    callback interruption, and error propagation tests. T12 extends this to
    Rosenbrock. Publish the contract first so other agents can work against it.

- [ ] **T02 · P0 · Error norms and tolerance policies** — Owner B; depends on T01 contract.
  - Add componentwise absolute/relative tolerances, masks, physical/STM/sensitivity
    block policies, maximum and RMS norms, and a custom norm hook.
  - Validate dimensions and invalid/nonfinite tolerance inputs; define scales,
    zero scales, masked components, and behavior when no components contribute.
  - Accept when Brahe maximum-norm and SatKit RMS acceptance decisions can be
    reproduced on shared candidate/error vectors, and independent analytic
    state/STM tests pass at both benchmark tolerance levels.

- [ ] **T03 · P0 · Public controllers and minimum-step policy** — Owner A; depends on T01, T02 interface.
  - Expose PI/PID coefficients, safety, growth/shrink/rejection limits, initial
    step selection and minimum-step behavior through configuration or a trait.
  - Export/restore controller history and the actual next proposed step; specify
    reset rules and distinguish failure at minimum step from forced acceptance.
  - Accept when replay tests reproduce both hosts' accepted/rejected step
    sequences under matching policies without changing existing defaults.

- [ ] **T04 · P1 · Allocation-free reuse and output control** — Owner A; depends on T01.
  - Add `solve_into` or equivalent caller-buffer reuse, reusable first-/second-order
    workspaces, and no-trajectory/final-state-only output. `Endpoints` currently
    still stores two states. Avoid duplicate candidate/solution construction.
  - Specify output-buffer capacity and resizing behavior. Preserve final-state
    access, events and requested output times; do not suppress requested output
    silently. Prefer an additive output-policy API if extending `SaveMode` breaks
    exhaustive matches.
  - Accept with zero steady-state allocations for fixed-size RK/RKN accepted and
    rejected attempts after setup, or a documented method-specific exception;
    measure construction separately and publish all remaining allocation sources.

- [ ] **T05 · P1 · Mutable and fallible closure ergonomics** — Owner B; no dependency.
  - Add direct `FnMut` RHS support and fallible closure constructors for both
    first- and second-order problems, retaining the stateless fast path.
  - Accept when mutable force evaluation and stage observation need no `RefCell`
    adapter, and ordinary `Result` closures work without a named trait wrapper.
    Verify callback/RHS mutation order and early termination; no NaN sentinels.

- [ ] **T06 · P1 · Preserve application errors without side channels** — Owner B; depends on T05.
  - Design typed user errors or an error-source wrapper preserving original
    payloads and cause chains, including errors in RHS/Jacobian/callback paths.
  - Keep the existing `SolveError` API compatible where possible via an additive
    wrapper/new entry point; a required breaking change needs a major release.
  - Accept with downstream examples recovering original Brahe-like errors,
    distinguishing solver/invariant failures, and no out-of-band error storage.
    Check exhaustive public error-enum matches and document migration impact.

- [ ] **T07 · P1 · State storage and layout adapters** — Owner C (A owns kernel buffers); depends on T01 contract.
  - Support borrowed slices, const-sized arrays and reusable buffers before
    attempting a general scalar/storage abstraction. Evaluate generic scalar
    support with compile-time, runtime and maintenance evidence; record the
    decision and implement only the justified scope.
  - Document row-major ndarray versus column-major nalgebra/numeris indexing;
    offer optional adapters or strided views with explicit flatten/unflatten.
  - Accept with nonsymmetric rectangular STM/sensitivity round trips, static and
    dynamic shape tests, and conversion/allocation benchmarks; never assume raw
    matrix slices have the same layout.

- [ ] **T08 · P1 · Typed tableaus and useful resource diagnostics** — Owner C; no dependency.
  - Construct validated RK/RKN methods from borrowed/owned coefficient arrays;
    remove the forced `&'static LazyTableau` lifetime for per-instance methods.
    Retain compile-time JSON resources as an optional authoring path.
  - Supply sensible metadata defaults where appropriate and identify the precise
    missing/invalid name, description, kind or coefficient instead of a generic
    `TableauResource(JsonSyntax)` error. Missing description caused the observed
    failure; `$schema` was not the missing field.
  - Accept with exact host-tableau construction without JSON parsing or global
    caches, invalid-dimension/coefficient diagnostics, and coefficient/embedded
    formula/dense-output equivalence checks rather than method-name matching.

- [ ] **T09 · P1 · Portable dense output and explicit interpolation quality** — Owner B; depends on T01 contract.
  - Expose a host-usable interpolation/segment interface, validated segment
    export/import, and optional versioned serde storage for solutions/segments.
  - Identify method-specific versus linear interpolation and allow callers to
    require method-quality dense output, rejecting an unavailable quality level.
  - Accept with serialization round trips preserving interpolated values,
    malformed/version-mismatch rejection, forward/backward boundary tests, and
    analytic interior-point error tests. Preserve allocation-free interpolation.
  - Replace linear saved-time/segment searches with indexed or cursor lookup
    where applicable (`src/solution/api.rs`). Benchmark scaling with trajectory
    length while preserving last-saved-state precedence at repeated callback
    times, segment-boundary semantics and backward trajectories. This is a
    source-observed optimization opportunity, not a measured benchmark regression.

- [ ] **T10 · P1 · Borrowed callbacks and thread-safe problems** — Owner B; depends on T05, T06 API design.
  - Remove unnecessary `'static` restrictions on callbacks, finalizers and
    Jacobians; provide lifetime-aware and `Send`/`Sync` variants or generic
    storage without forcing thread bounds on all single-threaded users.
  - Accept compile tests for borrowed captures and moving eligible problems to
    Rayon workers, plus rejection of unsafe sharing. Document today's valid
    parameter-context and construct-inside-worker workarounds and their costs.

- [ ] **T11 · P1 · FSAL and resumable continuation** — Owner A; depends on T01, T03.
  - Permit initial derivative injection/final derivative access and return a
    continuation token with workspace, controller history and next step proposal.
    Do not substitute the last accepted interval for the next proposal.
  - Define cache ownership and invalidation after control, parameters, events,
    discontinuities and state changes; reject incompatible method/state tokens.
  - Accept with split-versus-continuous solves, exact force-evaluation counts,
    rejected steps, final short intervals, backward propagation and restart tests.

- [ ] **T12 · P0 · Rosenbrock observability and orbital accuracy investigation** — Owner A (D owns independent accuracy reproduction); investigation starts immediately; implementation depends on T01–T03, T10 hooks.
  - Expose solved stages, embedded error, factorization statistics and analytic
    Jacobian/partial-time-derivative hooks with borrowed lifetimes; document matrix,
    scaling and partial-versus-total derivative conventions.
  - Isolate norm, controller, step selection, coefficients, roundoff, and time
    differentiation contributions to the observed RODAS4 orbital discrepancies.
    Run order/convergence and work-versus-achieved-error studies before changing
    defaults. Explain every remaining discrepancy; do not relax tests to match it.
  - Investigate time-origin sensitivity: `rosenbrock_extended/steps.rs` currently
    probes at `t + sqrt(EPSILON) * max(abs(t), 1)`, approximately 15 seconds away
    at epoch 1e9 and 14,901 seconds at epoch 1e12. This is a source-identified risk,
    not a demonstrated cause of the orbital results. Add analytic time partials
    and a domain-/scale-aware finite-difference policy; test shifted-epoch forms
    of the same analytic problem, backward integration and discontinuity bounds.
  - Preserve the correct explicit time term: numeris 0.6.0 declares `GAMMA_SUM`
    but omits it from its stage loop. On `y'=-1000(y-cos(t))-sin(t)`, the crate
    achieved error 8.59e-8 in 50 accepted steps at tolerance 1e-6 versus upstream
    1.03e-6 in 15,985; at 1e-9 upstream exceeded 100,000 attempts. Analytic
    solutions, not upstream alone, are the correctness oracle.
  - Accept with analytic autonomous/nonautonomous forward/backward tests,
    verified convergence, and a published equal-accuracy orbital comparison;
    classify any remaining slowdown without claiming a universal speedup.

- [ ] **T13 · P1 · RKN mixed variational systems** — Owner A; depends on T01, T02, T07 interfaces.
  - Add partitioned first-/second-order augmented states or a stage observer that
    updates STM/sensitivity blocks without recomputing primary force stages.
  - Specify velocity-dependent acceleration stages, auxiliary error control and
    coupling semantics; retain compatibility choices as explicit policies.
  - Accept with analytic/finite-difference STM checks, drag-dependent acceleration,
    forward/backward propagation, shape tests and matched RHS evaluation counts.

- [ ] **T14 · P0 · Reproducible performance and accuracy gates** — Owner D; no dependency for baseline; final comparison depends on T01–T13, T15 and the T16 adapter-code milestone.
  - Keep the pinned migration results/raw samples; add native whole-arc, compatibility
    stepper and upstream configurations as separate experiments. Cover LEO, GEO,
    eccentric orbits, two-body/J2/full forces, STM/sensitivities, loose/tight
    tolerances, backward propagation, events and dense interpolation where valid.
  - Profile setup, conversion, stage construction, rejected steps, recording and
    retained output separately. Measure allocations/bytes per attempt and accepted
    step, peak live heap, retained heap and process memory separately; label stack,
    native allocations, warm caches and other exclusions. Add process-memory
    measurements on a supported platform instead of relabeling allocator bytes.
  - Preserve fair force models, initial states, output retention, compiler/features,
    warm-up and hardware. Use independent analytic/reference solutions and plot
    work versus achieved error as well as equal-tolerance timings. Include setup
    and cold/warm results separately; use black-box outputs and record uncertainty.
  - Use at least the existing 3 rounds × 9 samples per backend/case separately for
    timing and instrumented memory; retain variability and raw artifacts. Never
    run competing benchmarks or builds on the measurement machine concurrently.
  - Target removal of bridge regressions without accuracy/output regressions;
    agree noise-aware thresholds from baseline variance before judging results.
    Report unresolved regressions rather than tuning tolerances to hide them.

## Gauss–Jackson (existing proposal, retained)

**T15 · P2 · Owner C.** Literature/independent prototype can begin early;
production integration depends on T01, T07, T09 and T11. This is required to
complete the SatKit method replacement, but is not on the existing-RK fast path.

- [ ] Add a Gauss–Jackson integrator, starting with fixed-step eighth order (`GaussJackson8`), for smooth, long-duration second-order problems such as orbital propagation.
  - Support acceleration depending on time, position, and velocity, including atmospheric drag. Provide reusable acceleration history, accurate startup/refinement, configurable corrector convergence, and explicit failure reporting.
  - Define behavior for backward integration, short intervals, final partial steps, interpolation, and history restart after discontinuities or state-changing callbacks.
  - Base the mathematics on [Berry & Healy, Implementation of Gauss-Jackson Integration for Orbit Propagation (2004)](https://drum.lib.umd.edu/handle/1903/2202). This is an addition beyond the existing SciML ports: no Gauss–Jackson implementation was found in the pinned OrdinaryDiffEq.jl reference.
  - Use [SatKit](https://github.com/ssmichael1/satkit) as the Rust compatibility reference. Evaluate [NASA JEOD](https://github.com/nasa/jeod) and [GROOPS](https://groops-devs.github.io/groops/html/orbitPropagatorType.html) as independent numerical references. The [Python/MATLAB/C++ GJ8 implementation](https://github.com/lorcan2440/Gauss-Jackson-Integrator) is an additional comparison candidate, with limited adoption evidence. Check source licenses before incorporating code.
  - Validate convergence order, startup accuracy, velocity-dependent forces, backward propagation, and long-arc error against analytic solutions and independent implementations. Benchmark runtime, force evaluations, and memory against SatKit GJ8, VCABM, and appropriate high-order RK/RKN methods at matched achieved accuracy, including startup cost.

## Integration and release work

- [ ] **T16 · P1 · Complete both downstream integrations** — Owner D; depends on the relevant T01–T13 APIs; Gauss–Jackson completion depends on T15.
  - Split acceptance into an adapter-code milestone (buildable integrations and
    passing functional tests) and final closure (T14 measurements and reports).
    T14 consumes the code milestone, not T16's final closure; avoid a circular
    dependency between integration and benchmark acceptance.
  - Replace per-attempt whole-solve adapters and captured-RHS reconstruction with
    persistent steppers; remove retained host control/error/interpolation code
    only after parity is demonstrated. Keep exact host coefficients and document
    unavoidable floating-point ordering or interface differences.
  - Cover static/dynamic states, STM/sensitivities, fallible forces, serialized
    output, events, controlled propagation and thread usage. Implement the real
    new Gauss–Jackson path rather than relabeling another solver or wrapping the
    original host implementation. Retain immutable upstream and migration baselines.
  - Run relevant Rust/Python tests in both forks and the complete benchmark matrix;
    record changed interfaces, known inherited failures and achieved accuracy.

- [ ] **T17 · P0 · Compatibility design and release gates** — Owner D; design starts immediately; final gate depends on T01–T16.
  - Freeze additive API designs on day 1: ownership/lifetimes, result/error types,
    output policy, dense representation, controller state and thread guarantees.
    Avoid blindly adding variants to exhaustively matched enums or bounds to
    existing traits. Preserve current defaults; place unavoidable breaking
    changes behind a major-version plan with migration notes.
  - Follow [RELEASING.md](docs/RELEASING.md), existing CI, MSRV 1.85, default and
    no-default features, downstream renamed-dependency consumers, rustdoc,
    clippy, formatting, resource validation, dependency/license checks and
    semver checks for both public library crates. Run pinned SciML compliance
    and comparison suites for changed kernels/coefficients/reference fixtures.
  - Add focused external-consumer tests for new APIs and examples showing each
    friction item resolved. Document performance limitations and unsupported
    combinations. Coordinate versions of the three workspace crates; validate
    package contents. This plan does not authorize publishing a release.

## Host and experiment follow-ups (not crate defects)

Owner D coordinates these separate fork changes and fresh baselines. Their
priorities apply within this plan; upstream reports or PRs require a separate
request. Keep compatibility reproductions separate from corrected-host results.
Implementation handoffs: A handles H03 with T13 and H04 with T12; B handles the
bounded H01/H02/H05 fork fixes after its foundation APIs; D implements H06 and
coordinates independent validation of all six. Review the size of those fixes
on day 1 and use contingency/re-estimation if they threaten the core milestones.

- [ ] **H01 · P1 · Brahe fixed-step configuration.** Honor the configured fixed
  step or reject/document conflicting `initial_step`; current `fixed_step(30)`
  propagated with 60 seconds. Test precedence explicitly. Set both intended
  step controls in future benchmarks and retain the recorded 60-second baseline.
- [ ] **H02 · P0 · Brahe controlled FSAL.** Stop adding control twice when the
  cached final derivative already contains it. Define physical-versus-controlled
  cache contents and invalidate on changing control; test repeated constant
  control against an analytic solution and time-varying controls separately.
- [ ] **H03 · P1 · Brahe velocity-dependent RKN stages.** Validate/correct the
  inherited use of the initial velocity at all stages using drag and analytic
  velocity-dependent problems. Preserve a documented legacy mode only if needed;
  do not imply gravity-only benchmark validation covers this case.
- [ ] **H04 · P0 · numeris/SatKit Rosenbrock time term.** Track the omitted
  explicit time derivative and preserve the analytic counterexample in T12.
  Validate a separate corrected reference; never remove the crate's correct
  term to make upstream parity look better.
- [ ] **H05 · P1 · Brahe offline SPICE fixture restoration.** Make coverage
  fixtures restore cached `de440s` reliably under an empty redirected cache.
  The instrumented gate had one inherited cleanup failure; rerun the complete
  gate after repair and distinguish it from the passing ordinary Rust tests.
- [ ] **H06 · P1 · Benchmark force/reference consistency.** Assert the intended
  GM, tides, EOP, space weather, output retention and fixed-step settings before
  timing. Preserve fixture URLs/hashes and fail early when official data is
  unavailable; do not mix missing-data skips with passing validation. Retain
  the corrected EGM96/tides assumptions from the SatKit two-body experiment.

## Audit coverage

`I01`–`I21` identify the rows of the linked API audit in their original order.
This crosswalk keeps every discovery traceable even when implementation shares
one API or one test suite.

| Audit finding | Resolution task(s) |
|---|---|
| I01 Whole-solve-only public entry point | T01 |
| I02 Private stages/component errors | T01, T12 |
| I03 Per-solve allocation/setup | T04, T14 |
| I04 No output-free solve | T04 |
| I05 Scalar tolerances/fixed norm | T02 |
| I06 Private controller policies/state | T03 |
| I07 Mutable RHS closure burden | T05 |
| I08 Fallible closure constructor burden | T05 |
| I09 Lost application error payload | T06, T17 |
| I10 Owned f64 storage/conversions | T07 |
| I11 Matrix layout mismatch | T07 |
| I12 Static JSON resource lifetime | T08 |
| I13 Metadata/parser diagnostics | T08 |
| I14 Dense serialization/interoperability | T09 |
| I15 Silent interpolation-quality downgrade | T09 |
| I16 Borrowing and thread bounds | T10 |
| I17 FSAL injection/extraction | T11 |
| I18 Missing continuation/next proposal | T11 |
| I19 Mixed RKN variational blocks | T13 |
| I20 Rosenbrock stage/time-partial access | T12 |
| I21 Missing Gauss–Jackson family | T15 |

Additional discoveries are tracked in T09 (lookup complexity), T12 (epoch-scale
finite differences and measured orbital accuracy), T14 (time/memory and fairness),
T16–T17 (downstream/release contracts), and H01–H06 (host/fixture issues).

## Fast parallel resolution schedule

Planning estimate: **18–25 working days from implementation kickoff**, plus
**3–5 working days contingency** for numerical/API surprises. This is an estimate,
not a deadline or an assertion that implementation has started. Re-estimate after
the first-day contract review and the first measured persistent-stepper result.
Use four agents total: three implementers plus an integration/verification agent.
No implementation agents or recurring jobs are launched by this TODO update.

| Agent | Primary ownership and files |
|---|---|
| A — solver lifecycle | T01/T03/T04/T11/T12/T13; integrator driver/kernel/controller and RK/RKN/Rosenbrock internals; kernel buffers in T07 |
| B — public problem/output API | T02/T05/T06/T09/T10; problem/function/callback and solution modules; tolerance interface in coordination with A |
| C — resources and new method | T07/T08/T15; tableau-core/macros/resource constructors, layout adapters, Gauss–Jackson module and independent references |
| D — integration and verification | T14/T16/T17/H01–H06; shared public exports/manifests, downstream forks, independent oracles, CI/release review and exclusive benchmark execution |

Use separate branches/worktrees for each agent. D owns shared exports/manifests
and merges small reviewed increments; other agents submit agreed interface
changes rather than editing those files concurrently. A and B agree error-vector,
controller and lifetime contracts before coding dependent modules. File ownership
may transfer at a milestone with an explicit handoff; it is not concurrent editing.

| Wave | Working days (estimated) | A | B | C | D / exit gate |
|---|---:|---|---|---|---|
| Contract + independent reproduction | 1 | T01/T03 contract, rejected-step invariants | T02/T05/T06/T10/T09 API contracts | T08 design; T15 references/oracles | T17 semver decisions; T12 analytic/orbital baseline; T14 provenance; H02/H04 triage |
| Foundations | 2–6 | T01/T04 plus T03 and T07 kernel buffers | T02 then T05/T06 | T08/T07 layouts; start T15 prototype | Merge foundations; T14 allocation smoke and first stepper comparison; H01/H05/H06 fixtures |
| Interoperability | 7–12 | T11 then T12 hooks/stages; H04 corrected reference | T10 then T09 representation/quality/serde; bounded H01/H02/H05 fixes | T15 startup, history, predictor/corrector | T16 explicit-RK adapters; validate separate host fixes; T12 equal-accuracy diagnosis |
| Complete method coverage | 13–17 | T13/H03, Rosenbrock and restart integration | Finish T09 lookup optimization and thread/borrow/error consumer tests; close host fixes | T15 velocity dependence, backward/endpoints, interpolation, convergence | T16 RKN/STM/Rodas integration; independent host validation; regression suite |
| Acceptance + optimization | 18–25 | Remove remaining measured stepping costs; resolve numerical findings | Address profiling/compatibility findings; docs | T15 independent cross-checks and SatKit Gauss–Jackson integration | T14 full serial benchmarks; T16 downstream tests; T17 final compatibility/package review |

The existing-RK milestone targets day 6: externally usable persistent stepping,
compatible tolerance/controller policies and measured allocation costs. Gauss–Jackson
starts early because its numerical work is independent and long-running; it must
not delay that milestone. Its production merge waits for stable workspace/history
and dense-output contracts. T12 investigation starts before those contracts settle;
hook implementation and downstream acceptance wait for their actual dependencies.

Critical paths are T01 → T03/T04 → T11/T12/T13 → T16 adapter code → T14 →
T16 closure/T17, and the parallel
T15 numerical-validation path joining T09/T11 before final integration. T05/T06 →
T10 can also delay borrowed Rosenbrock integration. D helps write independent tests
and examples when not merging or measuring; transfer bounded modules to an idle
agent instead of adding speculative agents against the same shared APIs.

During measurements, D reserves the benchmark machine and all other agents stop
builds/tests on that machine; work on documentation/review or another machine can
continue. Record immutable candidate commits. Do not sacrifice accuracy gates,
sample repetition or independent references to shorten the calendar.

The estimate covers the focused f64, borrowed/fixed-buffer API and an additive
user-error preservation design. T07's general-scalar feasibility decision is
included. If that decision warrants a pervasive generic-scalar/storage rewrite,
or T06 needs pervasive generic error types, budget a separate **10–20 working-day
follow-on**, re-estimated after a prototype; keep it tracked rather than declaring
it solved by the focused API. No task closes merely because its estimate expires.
All host follow-ups close only with evidence in the affected fork or an explicit
documented external dependency; no upstream issue or PR is opened automatically.
