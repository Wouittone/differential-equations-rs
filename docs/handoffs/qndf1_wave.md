# QNDF1 regular-ODE wave

This wave implements the pinned OrdinaryDiffEqBDF `QNDF1` algorithm for
regular identity-mass initial-value ODEs. It is fixed order one: variable-order
QNDF, higher-order QNDF2+, FBDF/DFBDF, residual DAEs, singular mass matrices,
and split/IMEX algorithms are excluded.

## Pinned source evidence

The pinned checkout is `D:/Source/_review/OrdinaryDiffEq.jl` at
`211142263781255a9aa2f910f6760b9f18ec29c8`.

- `lib/OrdinaryDiffEqBDF/src/algorithms.jl:231-285` defines QNDF1 and its
  default `kappa = -37//200`, identity extrapolant, and adaptive algorithm.
- `lib/OrdinaryDiffEqBDF/src/bdf_caches.jl:184-280` defines the order-one
  backward-difference cache and one-step history.
- `lib/OrdinaryDiffEqBDF/src/bdf_perform_step.jl:352-526` defines startup,
  variable-step history reinterpolation, the NDF residual coefficients,
  initial guess, error estimator, FSAL derivative, and acceptance state.
- `lib/OrdinaryDiffEqBDF/src/alg_utils.jl:7-49` and
  `lib/OrdinaryDiffEqBDF/src/controllers.jl:56-69` establish order one,
  qsteady bounds, and IController defaults. The Rust controller uses the
  equivalent exponent/order and gamma safety policy within the frozen driver.

## Rust representation

`src/qndf1.rs` owns one previous accepted state, a reinterpolated order-one
backward difference, NDF residual/Newton workspace, and cached dense LU. The
shared `StepKernel` driver owns callback, save, termination, rejection, and
step-size lifecycle. Identity mass is represented by the existing dense
`I - scale*J` convention; no DAE residual or mass-matrix behavior is claimed.

## Validation

`tests/qndf1.rs` checks first-order fixed-step convergence, adaptive stiff
decay, backward callback integration, nonfinite RHS, and singular Jacobian
handling. `examples/qndf1_compliance.rs` and `tests/julia/qndf1.jl` provide a
matched pinned-Julia endpoint fixture. Full Rust and Julia gates are required
before merge. On the pinned stiff tracking fixture with `dt = 0.01`, the
Rust fixed endpoint differs from Julia by `4.95e-5` (both are order-one and
the exact solution is `cos(1)`); the fixture records a `2e-4` relative
tolerance for this low-order nonlinear-solve representation difference.
