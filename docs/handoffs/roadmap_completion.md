# ODE roadmap completion

The pinned OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8` now has complete in-scope
algorithm-name coverage: all 345 included public names have a Rust numerical
implementation and a detected Julia compliance fixture. Four public names are
excluded by the documented ODE scope.

The dense-output lifecycle is also complete across the implemented drivers:

- native continuous extensions are retained for explicit RK, SSP,
  Rosenbrock/Rodas, DPRKN6, FIRK collocation, extrapolation, and Taylor
  families when pinned upstream data supplies them;
- the shared first-order driver uses cubic Hermite segments for other kernels;
- typed AMF, RKIP, IRKC, linear-operator, and Lie-group adapters provide their
  real derivatives instead of the placeholder RHS used for driver plumbing;
- IMEX multistep and multirate paths use total split derivatives;
- second-order, structural, and symplectic solutions retain partition-aware
  segments with cubic-Hermite positions and linear velocities.

`save_at`, continuous roots where callbacks are supported, forward/backward
queries, exact endpoints, and callback discontinuity ownership share these
accepted-step segments. Methods whose pinned upstream dispatch is generic are
described as generic Hermite output, not as method-specific high-order output.

Independent solve cases and ensemble sweeps are available through the
Rayon-backed execution policy. Algorithm types live under the public
`algorithms` hierarchy; the final SIMD and Taylor families were not added as
new crate-root algorithm reexports.
