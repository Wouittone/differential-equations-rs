# ODE algorithm coverage

Coverage is measured against SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. A Rust type is counted only when
it has a Julia differential-compliance test; sharing a kernel with another
method is not enough.

## Current status

The generated pinned-revision inventory currently detects **132 of 345**
in-scope public ODE names as both implemented and Julia-tested. The complete
per-name and per-family ledger is generated in
[`ODE_PARITY_INVENTORY.md`](ODE_PARITY_INVENTORY.md), with JSON and CSV forms
alongside it.

Coverage now includes automatic/default composite facades and the generic
user-tableau alias in addition to low/high-order and low-storage explicit
Runge–Kutta,
fixed and variable-step Adams, SSPRK, fixed implicit, TRBDF2, Rosenbrock/Rodas,
and an initial `q' = v` symplectic family. The original 25 methods appear in
the matched benchmark matrix; newly added methods have differential-compliance
tests but have not yet been added to that benchmark.

This is algorithm-name coverage, not full feature parity. Basic discrete and
scalar continuous callbacks plus `save_at` sampling are shared by implemented
first-order methods; dense high-order output, arbitrary scalar types, limiters,
and every upstream controller and callback option remain separate work. See
[`FEATURE_COVERAGE.md`](FEATURE_COVERAGE.md).

## Remaining ODE work

The port is not yet at OrdinaryDiffEq ODE algorithm parity. Major remaining
groups include:

- the rest of the low-, medium-, high-order, low-storage, and stabilized
  explicit Runge–Kutta catalog, including Feagin and TanYam methods;
- variable-order Adams and Nordsieck methods;
- the remaining Rosenbrock, Rosenbrock–W, Rodas, SDIRK, ESDIRK, KenCarp,
  FIRK, BDF, and QNDF methods;
- IMEX, split, partitioned, and multirate ODE methods;
- Runge–Kutta–Nyström, second-order, and symplectic ODE methods;
- extrapolation, exponential, and automatic stiffness-switching methods. The
  public automatic/default facades are present, but runtime stiffness
  switching remains deferred and is documented in their handoff.

Some of those groups require new problem representations before their kernels
can be ported faithfully: split right-hand sides, partitioned and second-order
state, Jacobian sparsity/coloring metadata, mass matrices, and composite
algorithms. Dense analytic state-Jacobian callbacks are already supported by
the current implicit and Rosenbrock kernels. DAE-only behavior remains out of
scope even when an in-scope ODE algorithm also supports mass matrices upstream.

Benchmark results must therefore be described as covering the original
25-solver benchmark slice, not every currently implemented Rust solver or every
OrdinaryDiffEq ODE solver.
