# FIRK and extrapolation family handoff

## Summary

Implemented the five pinned `OrdinaryDiffEqFIRK` public algorithms and the
seven pinned `OrdinaryDiffEqExtrapolation` public algorithms as native Rust
kernels. The FIRK implementation generates collocation tableaus from Radau
right-endpoint or Gauss--Legendre nodes and solves the coupled stage equations
with dense Newton. The extrapolation implementation evaluates the pinned base
discretizations and performs polynomial extrapolation on the same inverse-step
or inverse-square nodes.

## Files changed

- `src/firk.rs`
- `src/extrapolation.rs`
- `src/lib.rs`
- `src/compatibility.rs`
- `tests/firk_extrapolation.rs`
- `examples/firk_extrapolation_compliance.rs`
- `tests/julia/firk_extrapolation.jl`
- `tests/julia/Project.toml`
- `tests/julia/Manifest.toml`
- `tests/julia/runtests.jl`

## Public APIs added

- `algorithms::implicit::fully_implicit::{RadauIIA3, RadauIIA5, RadauIIA9,
  AdaptiveRadau, GaussLegendre}`
- `algorithms::extrapolation::{AitkenNeville,
  ExtrapolationMidpointDeuflhard, ExtrapolationMidpointHairerWanner,
  ImplicitEulerExtrapolation, ImplicitDeuflhardExtrapolation,
  ImplicitHairerWannerExtrapolation, ImplicitEulerBarycentricExtrapolation,
  ExtrapolationSequence}`

## Upstream source and revision

Authority: SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`.

Primary files:

- `lib/OrdinaryDiffEqFIRK/src/algorithms.jl`
- `lib/OrdinaryDiffEqFIRK/src/firk_tableaus.jl`
- `lib/OrdinaryDiffEqFIRK/src/firk_perform_step.jl`
- `lib/OrdinaryDiffEqFIRK/src/firk_interpolants.jl`
- `lib/OrdinaryDiffEqExtrapolation/src/algorithms.jl`
- `lib/OrdinaryDiffEqExtrapolation/src/extrapolation_caches.jl`
- `lib/OrdinaryDiffEqExtrapolation/src/extrapolation_perform_step.jl`
- `lib/OrdinaryDiffEqExtrapolation/src/controllers.jl`

The Julia manifest pins `OrdinaryDiffEqFIRK` tree
`1c80f6b93e9f78636e1c9b64e0df3d82e384cc47` and
`OrdinaryDiffEqExtrapolation` tree
`66b811d207e365f060c1087b8a2a57c39c3465db` from that revision.

## Rust tests

- Generated Radau/Gauss collocation moment checks through seven Radau stages.
- Fixed-step stiff decay for Radau IIA 3/5/9.
- Adaptive tolerance checks for `AdaptiveRadau` and `GaussLegendre`.
- All twelve public algorithms on a matched scalar reference problem.
- Implicit nonlinear/Jacobian/factorization/solve statistics.
- Backward integration.
- Continuous callback localization and termination through accepted dense
  segments.
- Fixed-step explicit and stiff extrapolation coverage.

## Julia tests

`tests/julia/firk_extrapolation.jl` runs the Rust release compliance example
and compares all twelve fixed-step endpoints with constructors from the two
pinned Julia subpackages. Focused result: 13/13 assertions passed.

## Numerical differences

- OrdinaryDiffEq's fixed Radau methods transform the coupled Newton system into
  real and complex eigen-basis blocks. Rust factors the mathematically
  equivalent full real block system.
- Rust uses Richardson step doubling for adaptive FIRK error estimates. This
  matches the pinned Gauss--Legendre design; pinned Radau uses specialized
  embedded estimators.
- `AdaptiveRadau` uses the same generated Radau collocation family and valid
  order range, with a local adjacent-order window. Its order-selection cost
  heuristic is intentionally simpler than the pinned work model.
- Extrapolation is accumulated with the Neville recurrence. Pinned Deuflhard
  and Hairer--Wanner implementations use precomputed barycentric weights in
  several paths; both evaluate the same extrapolation polynomial.

## Allocation/performance impact

The coupled FIRK Newton path is correctness-first and allocates full
`(stages * dimension)^2` storage. The extrapolation path retains workspace but
currently allocates individual raw and extrapolated vectors per level. Neither
family currently parallelizes internal independent discretizations; independent
problem solves can use the crate's Rayon ensemble API.

## Known limitations

- Regular identity-mass ODEs only; the crate's singular mass-matrix/DAE model
  is not yet available to these kernels.
- FIRK stage transforms and extrapolation internal level parallelism remain
  performance follow-ups, not formula gaps.

### Dense lifecycle audit

| Family | `save_at` | continuous callbacks | opt-in retained interpolation |
|---|---|---|---|
| Radau IIA 3/5/9, `AdaptiveRadau`, `GaussLegendre` | genuine collocation polynomial; adaptive attempts use the two accepted half-step collocation pieces | genuine collocation polynomial | gap: no owning dynamic collocation segment in shared `Solution` |
| All seven extrapolation algorithms | generic cubic Hermite from accepted endpoint RHS values | generic cubic Hermite | gap: no owning Hermite segment in shared `Solution` |

Generic Hermite is intentionally not described as method-specific
extrapolation interpolation. The current owning `Solution` dense-segment
representation accepts only static explicit-RK coefficient tables. Adding an
owning dense-segment enum (or equivalent trait object) is a shared dense-output
follow-up and was not changed in this family branch.

## Follow-up dependencies

- A general owning dynamic dense-segment representation would allow retained
  post-solve FIRK/extrapolation interpolation.
- A matrix-free or block-transform linear solve layer would remove the dense
  coupled FIRK scaling cost.

## Recommended next task

Add allocation baselines and block/eigen-basis FIRK linear solves after the
remaining public algorithm inventory reaches zero.
