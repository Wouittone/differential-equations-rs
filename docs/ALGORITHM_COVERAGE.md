# ODE algorithm coverage

Coverage is measured against SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. A Rust type is counted only when
it has a Julia differential-compliance test; sharing a kernel with another
method is not enough.

## Implemented and benchmarked

The current crate exposes 25 ODE algorithms:

- adaptive explicit: `Tsit5`, `Midpoint`, `Heun`, `Ralston`, `BS3`, `DP5`,
  `Alshina2`, `Alshina3`, and `SSPRK43`;
- fixed explicit: `Euler`, `RK4`, `RKM`, `Ralston4`, `SSPRK22`, and `SSPRK33`;
- fixed multistep: `AB3`, `AB4`, `AB5`, `ABM32`, `ABM43`, and `ABM54`;
- fixed implicit: `ImplicitEuler`, `ImplicitMidpoint`, and `Trapezoid`;
- adaptive linearly implicit: `Rosenbrock23`.

All 25 appear in the matched Rust/Julia benchmark matrix. This is algorithm
name coverage, not full feature parity: dense output, event handling,
arbitrary scalar types, limiters, and every upstream controller option remain
separate work.

## Remaining ODE work

The port is not yet at OrdinaryDiffEq ODE algorithm parity. Major remaining
groups include:

- the rest of the low-, medium-, and high-order explicit Runge–Kutta catalog,
  including Owren–Zen, BS5, Vern, Feagin, TanYam, and stabilized methods;
- variable-step and variable-order Adams methods;
- the remaining Rosenbrock, Rosenbrock–W, Rodas, SDIRK, ESDIRK, TRBDF2,
  KenCarp, FIRK, BDF, and QNDF methods;
- IMEX, split, partitioned, and multirate ODE methods;
- Runge–Kutta–Nyström, second-order, and symplectic ODE methods;
- extrapolation, exponential, and automatic stiffness-switching methods.

Some of those groups require new problem representations before their kernels
can be ported faithfully: split right-hand sides, partitioned and second-order
state, Jacobian sparsity/coloring metadata, mass matrices, and composite
algorithms. Dense analytic state-Jacobian callbacks are already supported by
the current implicit and Rosenbrock kernels. DAE-only behavior remains out of
scope even when an in-scope ODE algorithm also supports mass matrices upstream.

Benchmark results must therefore be described as covering every *currently
implemented* Rust solver, not every OrdinaryDiffEq ODE solver.
