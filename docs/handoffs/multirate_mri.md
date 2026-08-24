# Multirate and MRI-GARK parity wave

## Scope

Implemented the nine exact names from the pinned `OrdinaryDiffEqMultirate`
inventory as native split-problem methods:

- `MIS`, `MRAB`, `MREEF`
- `MRIGARKERK22a`, `MRIGARKERK22b`, `MRIGARKERK33a`, `MRIGARKERK45a`
- `MRIGARKESDIRK34a`, `MRIGARKIRK21a`

The Rust family is available from `algorithms::multirate` and through the
aggregate `algorithms` namespace. The explicit half of `SplitOdeProblem` is
the fast RHS and the implicit half is the slow RHS, matching Julia's
`SplitFunction(fast, slow)` convention.

## Authority and formulas

Authority: SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`, specifically:

- `lib/OrdinaryDiffEqMultirate/src/algorithms.jl`
- `lib/OrdinaryDiffEqMultirate/src/multirate_tableaus.jl`
- `lib/OrdinaryDiffEqMultirate/src/multirate_perform_step.jl`
- `lib/OrdinaryDiffEqMultirate/src/multirate_caches.jl`

The MRI-GARK `delta-c`, `W0`, `W1`, embedded weights, implicit diagonal
weights, and inner RK orders are direct ports. `MIS` uses the pinned
Knoth--Wolke alpha/beta/gamma construction. `MREEF` implements harmonic and
Romberg sequences with the upstream reverse Neville update. `MRAB` preserves
the upstream detail that AB history is bootstrapped anew inside each macro
step rather than carried between macro steps.

## Lifecycle and statistics

- Fixed and adaptive integration are available for all constructors. As in
  upstream, adaptive `MRAB(k = 1)` has no lower-order estimator and returns
  `AdaptiveStepUnsupported`.
- Forward and backward spans, endpoint/every-step saving, ordered `save_at`,
  initial/discrete callbacks, and continuous callbacks are supported.
- Continuous roots and save-at values use an owning/borrowed cubic Hermite
  segment built from the combined fast-plus-slow endpoint derivatives.
- `retain_dense_output = true` retains those owning Hermite segments for
  `Solution::interpolate`. This is **generic Hermite lifecycle interpolation**,
  not a method-specific MRI continuous extension or collocation polynomial.
- Callback-truncated retained segments are bounded at the localized root, and
  callback state changes invalidate cached endpoint derivatives.
- `rhs_evaluations` is deterministic and counts both fast and slow RHS calls
  in one total. The current public stats schema has no separate `nf`/`nf2`
  fields. Implicit MRI methods additionally report Jacobian evaluations,
  nonlinear iterations, factorizations, and linear solves.

## Exact limitations

- State and time scalars remain the crate's `f64`/`Vec<f64>` representation.
- The fast inner integrator is the tableau-prescribed explicit RK2/RK3/RK4
  method with `m` uniform microsteps; a user-replaceable inner solver is not
  exposed.
- Implicit slow stages use dense Newton with an optional analytic
  `with_implicit_jacobian` callback and finite differences otherwise.
- `MREEF` currently validates orders 2 through 10. The pinned default is
  order 4; order 1 has no embedded estimate in this driver.
- There is no method-specific dense-output formula upstream for this family in
  the pinned package. A future dense-output wave can add one without changing
  the present generic Hermite contract.

## Compliance assets

- Rust behavioral tests: `tests/multirate_mri.rs`
- Rust/Julia endpoint fixture: `examples/multirate_compliance.rs`
- Pinned Julia comparison: `tests/julia/multirate_mri.jl`
- `tests/julia/Project.toml` and `Manifest.toml` pin
  `OrdinaryDiffEqMultirate` to the authority revision and monorepo subdirectory.

Focused Julia parity compares all nine constructors at the same fixed macro
step and microstep count. The fixture passes all ten assertions (inventory key
set plus nine endpoints).
