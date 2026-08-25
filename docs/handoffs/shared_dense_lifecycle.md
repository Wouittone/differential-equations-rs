# Shared first-order dense lifecycle

The integration driver now supplies an owning cubic-Hermite segment whenever a
`StepKernel` does not expose a method-specific continuous extension. The
fallback is activated only for continuous callbacks, `save_at`, or retained
dense output. It caches the accepted endpoint derivative for the next segment,
shares one prepared segment between root localization and sampling, and bounds
segments at callback discontinuities.

Methods with native extensions opt out explicitly. This includes explicit RK,
SSP, Rosenbrock/Rodas, FIRK, and extrapolation kernels. FIRK now retains its
dynamic collocation polynomial, including the two half-step stage sets used by
adaptive attempts; extrapolation retains its accepted Hermite segment instead
of falling back to endpoint-linear post-solve queries.

The regression suite covers representative implicit, multistep, stabilized,
and low-storage kernels through the shared fallback, native FIRK and
extrapolation retention, forward queries, exact endpoints, and callback
discontinuity ownership.

The completion audit extended the same contract to typed adapters and the
standalone partitioned drivers. AMF, RKIP, IRKC, linear-operator, and Lie-group
kernels override the dense-derivative hook so placeholder driver problems
cannot produce zero slopes. IMEX multistep uses the total explicit-plus-
implicit derivative. `SecondOrderSolution` and `SymplecticSolution` retain
partition-aware segments and expose post-solve interpolation queries.
