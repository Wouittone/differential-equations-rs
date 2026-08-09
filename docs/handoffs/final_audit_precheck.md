# Final-audit precheck (report-only)

Audit date: 2026-08-09  
Integrated base audited: `721825b` (`docs: record Vern9 generated wave`)  
Pinned upstream: SciML/OrdinaryDiffEq.jl
`211142263781255a9aa2f910f6760b9f18ec29c8`

This is a precheck, not a parity sign-off. The inventory and feature evidence
show substantial progress, but the final parity gate is not close: 275
in-scope public names remain missing and several cross-family runtime features
are still deliberately incomplete.

## Scope and inventory evidence

The strict generator check was run against the pinned checkout:

```text
pwsh -NoProfile -File scripts/generate_ode_inventory.ps1
  -UpstreamPath 'D:\Source\_review\OrdinaryDiffEq.jl' -Check
Verified byte-stable inventory artifacts in ...\docs
Verified 349 source references at 211142263781255a9aa2f910f6760b9f18ec29c8
{"public_solver_names":349,"included_names":345,
 "included_canonical_or_composite":333,"included_aliases":12,
 "excluded_names":4,"implemented_and_julia_tested":70,
 "missing_included_names":275}
```

The generated CSV/JSON and Markdown summary agree. The 70 detected public
names are:

```text
AB3 AB4 AB5 ABM32 ABM43 ABM54 VCAB3 VCAB4 VCAB5 VCABM3 VCABM4 VCABM5
ABDF2 MEBDF2 QNDF1 QNDF2 Alshina2 Alshina3 BS3 BS5 DP5 Euler Heun Midpoint
OwrenZen3 OwrenZen4 OwrenZen5 Ralston Ralston4 RK4 RKM CarpenterKennedy2N54
DGLDDRK73_C DGLDDRK84_C DGLDDRK84_F NDBLSRK124 NDBLSRK134 NDBLSRK144 ORK256
SHLDDRK64 Rodas4 Rodas5P Rosenbrock23 Rosenbrock32 ImplicitEuler
ImplicitMidpoint SDIRK2 Trapezoid TRBDF2 SSPRK104 SSPRK22 SSPRK33 SSPRK43
SSPRK53 SSPRK53_2N1 SSPRK53_2N2 SSPRK53_H SSPRK54 SSPRK63 SSPRK73 SSPRK83
LeapfrogDriftKickDrift SymplecticEuler VelocityVerlet VerletLeapfrog Tsit5
Vern6 Vern7 Vern8 Vern9
```

Remaining missing counts by family (all are still in scope) are:

| Family | Missing |
| --- | ---: |
| Adams multistep | 1 |
| approximate-matrix-factorization wrapper | 1 |
| automatic/default composite | 2 |
| BDF and IMEX multistep | 11 |
| explicit Runge-Kutta | 1 |
| exponential Runge-Kutta | 17 |
| extrapolation | 7 |
| fully implicit Runge-Kutta | 5 |
| high-order explicit Runge-Kutta | 12 |
| IMEX multistep | 2 |
| linear and Lie-group methods | 18 |
| low-order explicit Runge-Kutta | 13 |
| low-storage explicit Runge-Kutta | 35 |
| multirate and MRI-GARK | 9 |
| Nordsieck variable-order multistep | 4 |
| parallel diagonally implicit Runge-Kutta | 1 |
| parallel explicit Runge-Kutta | 1 |
| QPRK explicit Runge-Kutta | 1 |
| Rosenbrock and Rosenbrock-W | 36 |
| Runge-Kutta interval prediction | 1 |
| Runge-Kutta-Nystrom | 17 |
| SDIRK, ESDIRK, and additive IMEX RK | 34 |
| second-order structural dynamics | 2 |
| SIMD explicit Runge-Kutta | 3 |
| stabilized explicit Runge-Kutta | 13 |
| stabilized implicit Runge-Kutta | 1 |
| strong-stability-preserving Runge-Kutta | 9 |
| symplectic and partitioned Runge-Kutta | 14 |
| Taylor series | 3 |
| user-tableau explicit Runge-Kutta | 1 |

The four explicit exclusions remain correctly justified: `DABDF2`, `DFBDF`,
and `DImplicitEuler` are residual-form DAE algorithms; `FunctionMap` is a
discrete map rather than a continuous IVP solver. Package-level SDE, DDE,
BVP, PDE, steady-state, and external-wrapper exports are outside the native
regular-ODE inventory by `docs/UPSTREAM_SCOPE.md`.

## Runtime parity gaps that block final sign-off

* `docs/FEATURE_COVERAGE.md` still records linear interpolation for accepted
  `save_at` samples and continuous-root localization except for the explicit
  RK Hermite slice. Method-specific retained segments, public dense queries,
  and callback/root use of the same segment are not complete. The generic
  Hermite hook is therefore a sound foundation, not dense-output parity.
* The controller remains a stateless proportional policy. Pinned OrdinaryDiffEq
  uses stateful PI by default, I for VCABM/BDF exceptions, PID/predictive
  variants, `dtmin`/`dtmax`, `tstops`, deadbands, and a distinct proposed versus
  clipped step. These are documented in `docs/handoffs/dense_controller_audit.md`
  with pinned Core references (`controllers.jl:1-76,206-250,600-843,912-1065`
  and `integrator_utils.jl:225-303,580-604,881-906`).
* QNDF1's fixed endpoint differs from pinned Julia by approximately `4.95e-5`
  on the stiff tracking fixture and uses `rtol=2e-4`; QNDF2 fixtures use
  relaxed low-order tolerances (`3e-4` fixed and `8e-4` adaptive in the wave
  handoff). These are explicit representation differences, not evidence of
  exact BDF parity, and require final review before any claim of completion.
* SDIRK2 and ABDF2 document adaptive controller-count divergence because the
  local driver uses proportional control while upstream uses method-specific
  PI/I history. MEBDF2 is fixed-step only; QNDF1/2, ABDF2, and MEBDF2 are
  identity-mass regular-ODE slices and do not cover singular mass, DAE residual,
  split/IMEX, or variable-order behavior.
* Verner6/7/8/9 and the explicit dense slice preserve endpoint behavior through
  the generic kernel. Upstream Verner dense interpolants and lazy extra stages
  are not represented locally; the handoff explicitly limits the claim to
  endpoint stepping and generic Hermite sampling.

## Allocation and performance evidence

`benchmarks/RESULTS.md` is the available directional benchmark (2026-07-30,
25 configurations, 50 measured solves after warm-up, endpoint-only storage).
Rust/Julia geometric-mean ratios are approximately `2.50x` runtime and
`0.48x` allocated bytes for explicit methods, and `0.51x` runtime and
`0.086x` allocated bytes for the four implicit methods. The report cautions
that the counting allocator adds timing overhead, adaptive controllers take
different step sequences, bytes are cumulative traffic rather than peak live
memory, and the implicit path rebuilds finite-difference Jacobians/factors.
This is harness evidence only; final Phase 8 still needs separated timing and
allocation runs, sampling variance, peak-live/process-RSS measurements, and
work-precision curves for newly added families.

Existing Rust allocation tests assert step-count-invariant workspace shape for
the shared explicit/implicit/Adams paths and the new BDF/Verner slices. They do
not establish zero-allocation dense sampling for every family or prove no
fixed-step/callback-free regression across all 70 names.

## Recommended next bounded wave

Continue Phase 7 with one SSPRK constructor that can reuse the frozen explicit
kernel, beginning with `SSPRK432` and its matched fixture. Pinned source
evidence is:

* `lib/OrdinaryDiffEqSSPRK/src/algorithms.jl:105-119` — public adaptive-capable
  four-stage, third-order declaration;
* `lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl:851-946` — constant/cache
  perform-step formulas and embedded residual;
* `lib/OrdinaryDiffEqSSPRK/src/interpolants.jl:26-198` — shared SSPRK dense
  interpolant dispatch.

This wave must decide explicitly whether to support the upstream embedded
adaptive estimate and limiter options or to defer the constructor; endpoint-only
fixed stepping is insufficient for a final public-name claim. The remaining
SSPRK names (`SSPRK932`, `SSPRKMSVS32`, `SSPRKMSVS43`, `KYKSSPRK42`,
`KYK2014DGSSPRK_3S2`, `pRRK22`, `pRRK33`, `pRRK54`) should follow as separate
bounded cards because their cache/limiter forms differ.

## Required final gate commands

Before final-audit sign-off, rerun from the integrated coordinator checkout:

```text
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl
pwsh -NoProfile -File scripts/generate_ode_inventory.ps1
  -UpstreamPath 'D:\Source\_review\OrdinaryDiffEq.jl' -Check
pwsh -NoProfile -File benchmarks/run.ps1 -Repetitions <n>
```

The inventory command must be repeated after every public algorithm addition.
No environmental blocker was observed in this precheck: the pinned checkout
was present and strict inventory verification passed. The parity blocker is
the documented in-scope implementation/feature gap; retry the final audit only
after the missing-name count reaches zero and the dense/controller and
representation caveats above have matched tests or explicit scope records.

