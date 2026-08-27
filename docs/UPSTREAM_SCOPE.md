# Upstream scope

The numerical reference is SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`, retained as the
`reference/OrdinaryDiffEq.jl` Git submodule. Initialize it with
`git submodule update --init --recursive` after cloning this repository.

## Included

Native OrdinaryDiffEq algorithms for ordinary differential equations:

- general first-order explicit and implicit ODE methods;
- split and IMEX ODE methods;
- stiff ODE methods;
- second-order, Runge–Kutta–Nyström, and symplectic ODE methods;
- exponential, stabilized, extrapolation, multirate, and multistep ODE methods;
- automatic stiffness detection and switching between native ODE methods.

## Excluded

- stochastic ODEs and stochastic DAEs;
- random ODEs;
- delay differential equations;
- differential-algebraic-equation-only algorithms and features;
- boundary-value, partial differential, and steady-state solvers;
- wrappers around external C, Fortran, Python, R, or MATLAB solvers.

Some native OrdinaryDiffEq algorithms can solve both ODEs and mass-matrix
DAEs. Their ordinary ODE behavior is in scope; DAE-specific initialization,
residual forms, and singular mass matrices are not.

## Compliance rule

Every public Rust algorithm must have a Julia `Test` case using the
corresponding OrdinaryDiffEq algorithm. Matching an analytic solution alone is
not sufficient. Comparisons use identical problems and tolerances or identical
fixed step sizes. Differences in controller step sequences are permitted when
the resulting trajectory satisfies the declared numerical tolerance.
