# BDF/SDIRK feasibility audit

## Summary

The pinned SciML/OrdinaryDiffEq.jl checkout is at revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. The regular-IVP BDF surface in
scope contains `ABDF2`, `FBDF`, `MEBDF2`, `QNDF`, `QNDF1`, `QNDF2`, `SBDF`, and
the configured `SBDF2/3/4` aliases. The DAE-only `DImplicitEuler`, `DABDF2`,
and `DFBDF` paths remain excluded. The SDIRK package contributes 35 regular
ODE names in the inventory; split/IMEX variants need a separate split problem
representation and are not a first target.

The smallest useful next family implementation is **SDIRK2** (regular,
non-split, adaptive two-stage ESDIRK). It is a better first SDIRK target than
the high-stage Kvaerno/KenCarp methods and avoids BDF's variable-history and
order-selection machinery. For the BDF wave, implement **ABDF2** only after
the shared nonlinear/Jacobian/mass seams are ready; its pinned implementation
is a compact fixed-leading-coefficient two-step method, but it still requires
startup implicit Euler, history rescaling, nonlinear solve integration, and
adaptive controller hooks. `QNDF1` is a possible lower-order fallback, not the
recommended parity anchor, because it still carries Nordsieck/backward-
difference rescaling and κ/error-control semantics.

## Upstream evidence (exact source references)

### BDF algorithms and lifecycle

- Algorithm declarations and constructors are in
  `lib/OrdinaryDiffEqBDF/src/algorithms.jl:63-180` (`ABDF2`, `SBDF`),
  `:255-374` (`QNDF1`, `QNDF2`), `:375-503` (`QNDF`, `MEBDF2`), and
  `:504-735` (`FBDF` and DAE variants). `ABDF2` is documented as adaptive
  order 2 with fixed leading coefficient (`algorithms.jl:47-61`).
- ABDF2 initialization evaluates the starting derivative and seeds a two-entry
  interpolation history (`bdf_perform_step.jl:1-11`). Its first accepted step
  delegates to an implicit-Euler cache (`:13-29`), so a native port must retain
  a startup branch before BDF2 history is valid.
- The regular ABDF2 step computes the variable-step ratio and coefficients
  `β₀=2/3`, `β₁=-(ρ-1)/3`, `α₁=1+ρ²/3`, `α₂=-ρ²/3`, forms the implicit RHS,
  invokes `nlsolve!`, and evaluates the new derivative (`:30-93`). The mutable
  path repeats the same contract with preallocated history (`:103-181`).
- ABDF2's non-identity mass path explicitly applies `M` to the history
  combination before the nonlinear solve (`:49-60` and `:140-151`). This is
  regular nonsingular mass behavior; singular/DAE initialization is outside
  this audit.
- QNDF1 is first-order but still rescales backward differences when
  `dt/dtₙ₋₁ != 1`, builds a κ-adjusted implicit relation, and computes adaptive
  error estimates (`bdf_perform_step.jl:354-440`). QNDF2 adds two-step history,
  two previous step sizes, and startup branches (`:528-640`). These are not
  smaller architectural targets than ABDF2 despite their lower order.
- Variable-order QNDF stores `D`, `prevD`, `R`, and `U`, reinterpolates history
  whenever step size/order changes, updates differences after convergence, and
  tests neighboring-order error estimates (`bdf_perform_step.jl:754-899`).
  FBDF has the analogous variable-order path with finite-difference weights
  and a larger dense-output/history surface (`:1152-1347` and `:1364-1502`).
- Shared history helpers are `reinterpolate_history!` and `update_D!` in
  `lib/OrdinaryDiffEqBDF/src/bdf_utils.jl:52-65` and `:114-125`; the
  variable-step transformation is `calc_R!` in `:84-107`. Error/controller
  policy lives in `controllers.jl:59-121` and `:256-314` (post-Newton and
  rejection/order decisions).

### SDIRK/ESDIRK algorithms and lifecycle

- Regular algorithm declarations are in
  `lib/OrdinaryDiffEqSDIRK/src/algorithms.jl:63-95` (`ImplicitEuler`),
  `:203-230` (`TRBDF2`), `:256-282` (`SDIRK2`), and `:304-331`
  (`SDIRK22`). Higher-stage Kvaerno/KenCarp declarations begin at
  `:408` and `:454`.
- `SDIRK2` is a two-stage, second-order adaptive algorithm with the same
  `nlsolve`, `linsolve`, predictor, smooth-estimate, and step-limiter knobs as
  the existing implicit family (`algorithms.jl:233-282`). Its tableau dispatch
  is `ESDIRKIMEXTableau(::SDIRK2, ...)` in
  `src/imex_tableaus.jl:2284-2291`, with coefficients constructed by
  `SDIRK2ESDIRKIMEXTableau` at `:2511-2610`.
- The common non-split stage loop is
  `generic_imex_perform_step.jl:52-1329` (in-place) and `:1329-2400`
  (out-of-place). It obtains `γ = Ai[s,s]`, predicts each stage, calls
  `nlsolve!`, and may reuse `W` at stage 2 (`:67-72`, `:122-128`, `:216-223`).
  This is a direct fit for the repository's frozen first-order driver plus a
  tableau-backed multi-stage kernel.
- The compact trapezoid special path is at
  `generic_imex_perform_step.jl:2407-2568`; it fixes `γ=1/2`, applies the mass
  matrix to the previous state when non-identity (`:2516-2559`), and performs
  one nonlinear stage. It is useful as a regression reference, but `SDIRK2`
  remains the smallest new adaptive SDIRK algorithm with an embedded estimate.
- Stage cache construction and the diagonal coefficient used for W are in
  `sdirk_caches.jl:90-143`; mass-matrix algebraic-variable detection is
  intentionally coupled to DAE setup at `:188-192` and must not be copied into
  the regular-ODE port.
- The pinned package exports all regular names from
  `OrdinaryDiffEqSDIRK/src/OrdinaryDiffEqSDIRK.jl:59-63`; IMEX/split names in
  the same export list require `SplitODEFunction` and should be queued after a
  split-problem representation exists.

## Feasibility and dependency assessment

1. **SDIRK2 first (recommended next SDIRK task).** Reuse the existing checked
   dense Jacobian/LU and controller seams, add one generated two-stage tableau,
   and implement a `StepKernel` that stores stage states/derivatives and one
   embedded error vector. Required tests: scalar stiff decay, nonautonomous
   RHS, backward integration, analytic/finite-difference Jacobian parity,
   callback invalidation, adaptive rejection, and pinned Julia convergence.
2. **ABDF2 second (recommended BDF anchor).** Add a two-state history and
   previous-step-size cache, an implicit-Euler startup branch, the coefficients
   from `bdf_perform_step.jl:30-47`, and a fixed leading-coefficient adaptive
   controller. First restrict to identity mass and regular ODEs; add constant
   dense nonsingular mass only after `MassOperator` application/solve tests.
   Do not port `SplitODEFunction`, DAE residual initialization, or variable-order
   FBDF/QNDF in this slice.
3. **Defer variable-order QNDF/FBDF.** They require Nordsieck/history
   reinterpolation, order selection, dense output based on history, and
   failure/discontinuity state machines; these are independent architecture
   phases rather than coefficient-only ports.
4. **Defer KenCarp/Kvaerno/IMEX.** Their generic stage loops are reusable, but
   each requires high-stage tableau records, method-specific dense estimates,
   and (for additive names) a split RHS representation. Porting them before
   SDIRK2 would duplicate the same nonlinear-stage policy.

## Numerical and scope caveats

- Upstream uses `M/(dt*γ)-J` in its nonlinear linear system while the Rust
  implementation currently uses the algebraically scaled `I-(γ*dt)J` form.
  A mass-enabled port must preserve one convention consistently in both W and
  the nonlinear RHS; copying only the upstream matrix formula is incorrect.
- `DImplicitEuler`, `DABDF2`, and `DFBDF` are residual-form DAE algorithms and
  remain excluded even though their source shares BDF caches.
- Sparse/Krylov linear solvers, autodiff backends, split/IMEX problems, varying
  mass matrices, and singular mass matrices are not prerequisites for the first
  regular SDIRK2 or ABDF2 implementation.

## Recommended next task

Spawn an isolated `sdirk2-kernel` agent after the current linear/operator and
coefficient-schema gates are merged. Allow only the new tableau record, a
dedicated `src/sdirk.rs`, tests, and a handoff. Keep `src/lib.rs`, manifests,
inventory, shared driver, and status docs coordinator-owned. Follow with an
isolated `abdf2-kernel` agent that may reuse the same driver but must not touch
variable-order BDF or DAE modules.

## Validation

- Upstream checkout verified at the target revision with `git -C
  D:/Source/_review/OrdinaryDiffEq.jl rev-parse HEAD`.
- Source references above were read directly from that detached checkout.
- This report changes no Rust or test source and therefore has no numerical or
  allocation impact.
