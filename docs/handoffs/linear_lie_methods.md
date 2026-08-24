# Exact linear and Lie-group methods handoff

## Scope and upstream

This wave ports all 18 names exported by `OrdinaryDiffEqLinear` at
SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`: `CayleyEuler`, `CG2`, `CG3`,
`CG4a`, `LieEuler`, `LieRK4`, `LinearExponential`, `MagnusAdapt4`,
`MagnusGauss4`, `MagnusGL4`, `MagnusGL6`, `MagnusGL8`, `MagnusLeapfrog`,
`MagnusMidpoint`, `MagnusNC6`, `MagnusNC8`, `RKMK2`, and `RKMK4`.

The formulas come from `lib/OrdinaryDiffEqLinear/src/linear_perform_step.jl`
and the public orders from `alg_utils.jl`. The Julia fixture pins the
same repository revision and subdirectory in `Manifest.toml`.

## Rust APIs and representation

- `LinearOperatorProblem<O, P>` represents dense row-major
  `u' = A(u,p,t)u` problems.
- `LieGroupProblem<O, P>` has checked vector exponential-action and square
  matrix conjugation representations.
- `solve_linear_operator` and `solve_lie_group` run those typed problems.
- `LinearOperatorAlgorithm` and `LieGroupAlgorithm` expose family order.
- Vector algorithms also implement `OdeAlgorithm`. On that compatibility path
  an analytic Jacobian is interpreted as the linear generator; without one,
  a dense finite-difference operator is formed. A state-dependent typed
  operator should use `LinearOperatorProblem`, because the Jacobian of
  `A(u)u` is not generally `A(u)`.

All matrix functions use the existing correctness-first dense `f64`
scaling/Taylor/squaring backend. Matrices are row-major and dimensions must be
small enough for dense `n²` storage. Krylov actions and sparse/GPU operators
are intentionally deferred.

## Formula inventory

- Lie Euler, exponential midpoint, constant linear exponential, and the
  two-step Magnus leapfrog use their direct exponential actions.
- RKMK2/RKMK4 use the pinned truncated dexp/commutator formulas.
- LieRK4 uses the pinned two-exponential final composition.
- CG2, CG3, and CG4a retain the upstream ordered exponential compositions and
  stage abscissae.
- MagnusAdapt4 retains all six operator stages, `Q1` through `Q6`, embedded
  exponential estimate, commutators, and adaptive residual.
- Gauss/GL/NC Magnus methods retain their pinned quadrature nodes and the
  fourth-, sixth-, and eighth-order commutator expansions. In the pinned
  source, `MagnusGauss4` and `MagnusGL4` have opposite commutator signs; the
  Rust schemes deliberately remain distinct.
- Cayley Euler builds `(I-hL/2)⁻¹(I+hL/2)` by pivoted dense LU and applies
  `Y <- V Y Vᵀ`, preserving similarity invariants for skew generators.

## Lifecycle audit

The shared integration driver provides fixed/adaptive validation, forward and
backward time direction, accepted/rejected accounting, maximum-step handling,
save modes, `save_at`, callback invocation accounting, and terminal errors.
Only `MagnusAdapt4` advertises adaptive control, matching upstream; the other
methods reject adaptive options.

Ordinary `OdeProblem` use supports discrete and continuous callbacks. Typed
operator/Lie problems currently do not carry callbacks. `save_at` works via the
shared endpoint-linear compatibility recorder. Continuous callback roots also
use endpoint-linear localization. Enabling retained dense output does not add
a method-specific exponential segment: `Solution::interpolate` keeps its
documented linear fallback. These are explicit dense-lifecycle gaps for the
later cross-family dense-output audit, not claimed exponential extensions.

## Coverage

Rust tests invoke every constructor and cover constant scalar/vector exactness,
time-dependent quadrature behavior, adaptive rejection, backward integration,
Cayley trace/determinant preservation, invalid representations, non-finite
operators, statistics, fixed/adaptive failures, save-at, interpolation fallback,
and discrete/continuous callbacks. The Julia fixture invokes all 18 pinned
constructors and compares matched constant-generator endpoints, including the
Cayley matrix path.
