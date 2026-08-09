# QNDF2 regular-ODE wave

This wave adds fixed/adaptive order-two QNDF for regular identity-mass ODEs.
The pinned source is OrdinaryDiffEqBDF at
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `src/algorithms.jl:289-345` defines QNDF2 and default `kappa = -1//9`;
- `src/bdf_caches.jl:283-388` defines the two-history, two-difference cache;
- `src/bdf_perform_step.jl:528-754` defines startup, order-two history
  reinterpolation, NDF residual coefficients, error estimator, and acceptance;
- `src/bdf_utils.jl:1-32` defines the `R*U` history transform for changed
  step ratios.

The Rust implementation uses the frozen `StepKernel` lifecycle, preallocated
order-two history/Newton/LU workspace, and regular identity-mass `I-scale*J`
systems. DAE residuals, singular mass, split/IMEX, and variable-order paths
are excluded. Pinned Julia endpoint fixtures use relaxed low-order tolerances
to account for the local Newton/LU representation while checking convergence
and exact qualitative behavior.
