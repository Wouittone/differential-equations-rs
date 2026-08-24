# Exponential Runge--Kutta family

The `algorithms::exponential` namespace contains the dense `f64` exponential
Runge--Kutta and exponential Rosenbrock implementations corresponding to
`OrdinaryDiffEqExponentialRK` at revision
`211142263781255a9aa2f910f6760b9f18ec29c8`.

Use `SemilinearOdeProblem` and `solve_exponential` when the constant split
`u' = A u + g(u,p,t)` is known. The operator is a finite row-major
`dimension × dimension` matrix. This route preserves `A` exactly and is the
preferred interface for Lawson, Nørsett/ETD, ETDRK, and Hochbruck--Ostermann
methods. The same algorithm values also implement `OdeAlgorithm`; on a regular
`OdeProblem` they use the supplied Jacobian, or a finite-difference Jacobian,
as the step-local linearization required by exponential Rosenbrock formulas.

The dense backend evaluates exponential and phi actions through an augmented
matrix exponential with norm scaling, a converged Taylor series, and repeated
squaring. It is intended for small and medium static systems. Krylov and
matrix-free actions are not yet implemented; no ordinary explicit RK fallback
is used.

`Exprb32` and `Exprb43` provide their published embedded estimators and support
adaptive or fixed stepping. The remaining methods are fixed-step algorithms
and therefore require `SolveOptions::with_adaptive(false)` plus an initial
step. `ETD1` is an exact type and value-constructor alias of `NorsettEuler`.
`ETD2` uses ETD1 startup followed by its two-step phi quadrature.

The implementation follows `exponential_rk_caches.jl` and
`exponential_rk_perform_step.jl` in the pinned upstream package. Tests cover
all constructors, scalar and vector matrix functions, exact linear evolution,
semilinear nonlinear convergence, nonautonomous and backward integration,
adaptive/fixed contracts, statistics, aliases, and non-finite failures. The
Julia fixture invokes every upstream constructor and numerically compares the
methods on the same analytic linear problem, using the required split
representation for upstream's split-only ETD2.
