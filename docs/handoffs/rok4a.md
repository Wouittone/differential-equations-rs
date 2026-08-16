# ROK4a parity handoff

Implemented the regular initial-value `Rok4a` Rosenbrock algorithm from the
pinned OrdinaryDiffEq revision `211142263781255a9aa2f910f6760b9f18ec29c8`.

The native implementation uses the exact four-stage `ROK4aRodasTableau` from
`lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl` and the
shared adaptive/fixed-step Rosenbrock driver. The primary fourth-order method
and its third-order embedded estimator are exposed as
`differential_equations::Rok4a`.

Coverage includes fixed and adaptive regular ODE integration, backward
integration, the compliance example endpoint, and the pinned Julia fixture.
The Julia executable is unavailable in this environment, so Julia parity
execution remains subject to the documented `JULIA-PATH-20260809` retry.

Excluded behavior: SDE, DDE, BVP, PDE, steady-state, DAE-only residual, and
external wrapper paths.
