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
