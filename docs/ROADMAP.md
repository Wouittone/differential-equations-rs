# Solver improvement roadmap

Tracking issue: https://github.com/Wouittone/differential-equations-rs/issues/33

Project: https://github.com/users/Wouittone/projects/3

Baseline: crate 1.0.0, commit 31decb9c. The complete acceptance criteria live in the linked issues. Versions are release targets, not claims of completion; all three workspace crates advance together. Preserve defaults and existing public enum exhaustiveness. Any unavoidable break moves the affected wave to 2.0.0 with migration notes. No crate publication is part of this work.

## v1.1.0 — Reusable stepping foundations

- [x] [T01: Public reusable stepping contract](https://github.com/Wouittone/differential-equations-rs/issues/16)
- [x] [T02: Error norms and tolerance policies](https://github.com/Wouittone/differential-equations-rs/issues/17)
- [x] [T03: Public controllers and minimum-step policy](https://github.com/Wouittone/differential-equations-rs/issues/18)
- [x] [T04: Allocation-free reuse and output control](https://github.com/Wouittone/differential-equations-rs/issues/19)
- [x] [T05: Mutable and fallible closure ergonomics](https://github.com/Wouittone/differential-equations-rs/issues/20)
- [x] [T06: Preserve application errors without side channels](https://github.com/Wouittone/differential-equations-rs/issues/21)
- [x] [T07: State storage and layout adapters](https://github.com/Wouittone/differential-equations-rs/issues/22)
- [x] [T08: Typed tableaus and useful resource diagnostics](https://github.com/Wouittone/differential-equations-rs/issues/23)

## v1.2.0 — Interoperability and continuation

- [x] [T09: Portable dense output and explicit interpolation quality](https://github.com/Wouittone/differential-equations-rs/issues/24)
- [x] [T10: Borrowed callbacks and thread-safe problems](https://github.com/Wouittone/differential-equations-rs/issues/25)
- [x] [T11: FSAL and resumable continuation](https://github.com/Wouittone/differential-equations-rs/issues/26)
- [ ] [H01: Brahe fixed-step configuration](https://github.com/Wouittone/differential-equations-rs/issues/10)
- [ ] [H02: Brahe controlled FSAL](https://github.com/Wouittone/differential-equations-rs/issues/11)
- [ ] [H05: Brahe offline SPICE fixture restoration](https://github.com/Wouittone/differential-equations-rs/issues/14)

## v1.3.0 — Rosenbrock accuracy and mixed variational systems

- [x] [T12: Rosenbrock observability and orbital accuracy investigation](https://github.com/Wouittone/differential-equations-rs/issues/27)
- [x] [T13: RKN mixed variational systems](https://github.com/Wouittone/differential-equations-rs/issues/28)
- [ ] [H03: Brahe velocity-dependent RKN stages](https://github.com/Wouittone/differential-equations-rs/issues/12)
- [ ] [H04: numeris/SatKit Rosenbrock time term](https://github.com/Wouittone/differential-equations-rs/issues/13)

## v1.4.0 — Gauss-Jackson and downstream integration

- [x] [T15: Gauss-Jackson eighth-order integrator](https://github.com/Wouittone/differential-equations-rs/issues/30)
- [x] [T16: Complete both downstream integrations](https://github.com/Wouittone/differential-equations-rs/issues/31)

## v1.4.1 — Measured optimization and final acceptance

- [x] [T14: Reproducible performance and accuracy gates](https://github.com/Wouittone/differential-equations-rs/issues/29)
- [x] [T17: Compatibility design and release gates](https://github.com/Wouittone/differential-equations-rs/issues/32)
- [ ] [H06: Benchmark force/reference consistency](https://github.com/Wouittone/differential-equations-rs/issues/15)

## External solver replacement evaluations

Forked replacement experiments broaden the acceptance evidence beyond SatKit and
Brahe's original migration. These experiments compare native solver paths with
the merged `1.4.0` backend at commit `a03385a3e427f583573db62b2a716099104ae000`.

- [External `eqsolver` replacement benchmark](https://github.com/Wouittone/differential-equations-rs/issues/37):
  adaptive Tsit5 improves the adaptive comparison surface but is 11x–19x slower
  than tiny fixed-step RK4 cases; keep fixed-step and adaptive paths distinct.
- [External `diffeq` backend tradeoffs](https://github.com/Wouittone/differential-equations-rs/issues/36):
  replacement Tsit5 beats the native explicit Ode4 case in the sampled workload,
  while replacement Rodas5P trails the native stiff baseline; state/callback
  compatibility remains the main adapter barrier.
- [Brahe solver replacement evaluation](https://github.com/Wouittone/differential-equations-rs/issues/38):
  68 orbital/STM cases are performance-neutral (geometric mean -0.5% to +1.4%);
  endpoint differences are zero for RK cases and at most 0.000216 m for RKN1210.
- [SatKit solver replacement evaluation](https://github.com/Wouittone/differential-equations-rs/issues/40):
  representative six-hour LEO RKV98 is 1.376x slower with the replacement, with
  micrometre-scale position and nanometre-per-second velocity deltas.
- [Portable solver diagnostics](https://github.com/Wouittone/differential-equations-rs/issues/41)
  is the primary API follow-up: downstream comparisons need stable RHS/Jacobian
  and accepted/rejected-step counters.
- [Gauss-Jackson work-versus-error benchmark](https://github.com/Wouittone/differential-equations-rs/issues/42)
  tracks the remaining solver-specific performance gap.

## Scope clarification

The user deferred upstream issue fixes (H01–H06). They remain tracked and open, but are excluded from the current implementation waves. Work focuses on this library; final downstream migration and benchmark evaluation remain the acceptance experiment after crate work.

## Dependencies and execution

Contract review precedes implementation. A owns persistent kernels/controller/output/continuation/Rosenbrock/mixed RKN; B owns norms, closures/errors, borrowing and dense output; C owns typed resources, layout adapters and Gauss–Jackson; D coordinates exports/manifests, independent verification, downstream migrations and exclusive benchmarks. Use isolated worktrees and reviewed commits. T01 establishes the step contract; T02/T03/T04/T07 build on it; T11 enables T12/T13 and Gauss–Jackson history. T05/T06 precede T10. T09/T11 precede production T15. T16 adapter-code completion precedes T14 measurements; T16 final closure consumes those measurements, avoiding a cycle. T17 starts with compatibility design and closes at final acceptance. H01–H06 remain separately attributable host fixes.

Run targeted code/tests for each change. Do not repeat full Julia comparisons or full linting on every step. After implementation and buildable migrations, stop competing builds, retain candidate commits, run at least 3 rounds × 9 samples separately for timing and memory, preserve raw data, and report equal-tolerance and achieved-accuracy results with variance. Run consolidated release gates once at the end; numerical changes require the relevant pinned compliance coverage.

## Baseline evidence and audit traceability

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
