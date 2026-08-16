# ROS3PR handoff

Added the native regular-ODE `Ros3Pr` algorithm, corresponding to upstream
`ROS3PR` at OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`.

The implementation uses the existing Rosenbrock/Rodas shared driver and dense
LU/Jacobian cache. Its tableau is the exact upstream
`ROS3PRRodasTableau`: three stages, order three, adaptive embedded estimate,
and `gamma = 0.788675134594813`. The method supports numeric or supplied
Jacobians, nonautonomous right-hand sides, callbacks, requested save points,
and backward integration through the shared lifecycle. As for the other
Rosenbrock methods in this crate, regular ODE dense sampling is provided by
the shared recorder; upstream DAE-only residual behavior is out of scope.

The public Rust spelling is `Ros3Pr`; inventory matching normalizes this to
the upstream `ROS3PR` name.

Focused coverage adds the stiff nonautonomous adaptive endpoint, fixed-step
third-order convergence, backward solve, callback/save-at behavior, and the
Julia compliance row in `tests/julia/rosenbrock_extended.jl`.
