# Specialized operator solvers

This wave ports the remaining operator-specific OrdinaryDiffEq families at the
pinned upstream revision.

- `AMF(Rosenbrock23)` applies the Rosenbrock-W stages through an ordered product
  of approximate Jacobian factors. The structured API accepts exact Jacobian
  and factor-update callbacks; ordinary `OdeProblem` use remains available with
  a single dense factor.
- `RKIP` evolves semilinear problems in the interaction picture with the pinned
  Verner tableau, cached matrix exponentials, cache-grid step snapping, adaptive
  error estimation, and cache reuse across solves.
- `IRKC` preserves the implicit/explicit split, the pinned 50-stage Chebyshev
  recurrence, Newton solves for the implicit component, spectral-radius
  estimation or override, and the upstream error estimator.

Rust unit tests cover factor ordering, callback lifecycle, RKIP cache recycling,
adaptive and backward integration, IRKC eigenvalue handling, and failure paths.
`tests/julia/specialized_operator_methods.jl` compares endpoints with the pinned
Julia packages. The IRKC comparison uses `dt = 0.001`: its documented first-order
fixed-step behavior is then in the converged regime, unlike the deliberately
coarse stability-oriented `dt = 0.01` case.
