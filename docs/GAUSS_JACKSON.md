# Gauss–Jackson 8 implementation and validation

`GaussJackson8` is a persistent fixed-step solver for acceleration depending on
time, position and velocity. It owns reusable nine-sample acceleration history,
physical-unit first/second sums, and startup/corrector scratch buffers. Its
fallible FnMut force interface preserves the original error value. Accepted state,
history and the last dense interval survive failed attempts unchanged; work
counters include failed force calls. Clone is a continuation snapshot. Restart
invalidates history after state or force/parameter changes.

The steady-state mathematics follows [Berry and Healy (2004)](https://api.drum.lib.umd.edu/server/api/core/bitstreams/86fad2f4-fb18-4778-9bd7-55e46994d965/content).
The generator `scripts/gauss_jackson_coefficients.py` derives exact rational
coefficients from the inverse-logarithm series and expands backward differences
into ordinates. The position recurrence uses a double sum; velocity uses summed
Adams, with force evaluations using both corrected position and velocity.
No external implementation is wrapped or relabeled as this solver.

Startup uses eight one-sided, tenth-order extrapolated-midpoint steps with
convergence refinement. No initial evaluation before the supplied epoch is
required. Short arcs remain explicitly counted startup work. Final partial
intervals use the same high-order extrapolation and clear the fixed-grid history;
continuation after a tail restarts history. This costs more than a mature GJ step.
Changing direction requires restart. Callbacks may inspect each returned step;
state-changing callbacks must restart before continuing. The solver does not
silently step across a caller-known discontinuity.

The last interval has quintic Hermite position interpolation and its analytic
velocity derivative, reusing caller buffers. This is deliberately documented as
lower-order than the eighth-order endpoint method. It is not an eighth-order
continuous extension. Degree-major coefficients can be exported for portable
dense segments. Invalid/out-of-range queries fail. Exact endpoints return the
accepted states rather than reevaluating a rounded polynomial.

## Focused numerical evidence

The module tests cover harmonic long arcs; circular two-body energy/phase;
time-polynomial accelerations; damped, velocity-dependent dynamics forward and
backward; short endpoints; startup force-domain bounds; restart and exact cloned
continuation; explicit startup/corrector convergence failures; original user
errors; dense interpolation and dense preservation after failure.

For q''=-q on [0,20], max position/velocity errors in one focused run were
1.190e-6 at h=0.4, 5.340e-10 at h=0.2 and 8.7e-14 at h=0.1. This demonstrates at
least eighth-order convergence before roundoff on that case, not a universal
superconvergence claim. Startup and endpoint behavior are tested separately.
The circular two-body test spans 100 normalized time units; the harmonic long
arc spans 200. Broader orbital timing/work-versus-error benchmarks remain part
of final acceptance and have not been run by this module's implementation tests.

An independent [Python implementation](https://github.com/lorcan2440/Gauss-Jackson-Integrator/tree/b011576364d77fb606fae6f02415265c33215421)
was run at pinned commit b011576364d77fb606fae6f02415265c33215421. Its BSD-2-Clause
license was checked before use. It is downloaded only into ignored target data;
no reference code is included in the crate. `scripts/gauss_jackson_reference.py`
checks the source SHA-256 and reproduces `tests/fixtures/gauss_jackson_reference.json`.
Harmonic h=0.1 and damped h=0.1 endpoints agree within 1e-12. Its centered startup
differs from this implementation's one-sided startup, and coarse-step errors
therefore differ; analytic solutions are the accuracy oracle.

SatKit at 4b99262fa294f36d2cef468c35d1c59c78cfeef2 was inspected as a compatibility
reference under its MIT/Apache licenses. Its coefficient conventions agree with
the generated steady-state arrays; its runtime benchmark comparison is deferred.
[NASA JEOD](https://github.com/nasa/jeod/blob/main/LICENSE) uses NOSA 1.3 and
[GROOPS](https://github.com/groops-devs/groops/blob/main/LICENSE) uses GPLv3.
They were evaluated as reference candidates; neither source is incorporated nor
claimed to have been executed. GROOPS documents the same method family, but its
application build/runtime is not a lightweight independent unit-test dependency.
