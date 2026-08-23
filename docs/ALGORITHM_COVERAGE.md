# ODE algorithm coverage

Coverage is measured against SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. A Rust type is counted only when
it has a Julia differential-compliance test; sharing a kernel with another
method is not enough.

## Current status

The generated pinned-revision inventory currently detects **264 of 345**
in-scope public ODE names as both implemented and Julia-tested, with **81**
in-scope names still missing a Rust implementation. The complete per-name and
per-family ledger is generated in
[`ODE_PARITY_INVENTORY.md`](ODE_PARITY_INVENTORY.md), with JSON and CSV forms
alongside it.

Coverage includes automatic/default composite facades and the generic
user-tableau alias in addition to low/high-order and low-storage explicit
Runge-Kutta, fixed and variable-step Adams, SSPRK, fixed implicit, TRBDF2,
SDIRK/ESDIRK, Rosenbrock/Rodas, variable-order BDF/QNDF, stabilized explicit,
parallel explicit RK, split Euler, RKN/Nyström, and symplectic/partitioned
families. A representative **31-algorithm** slice appears in
the matched benchmark matrix; the benchmark is not intended to cover every
implemented method.

This is algorithm-name coverage, not full feature parity. Basic discrete and
scalar continuous callbacks plus `save_at` sampling are shared by implemented
first-order methods. Tsit5 now provides retained high-order dense segments and
method-specific root localization, and DPRKN6 provides its pinned dense
extension; other method families, arbitrary scalar types, limiters, and every
upstream controller and callback option remain separate work. See
[`FEATURE_COVERAGE.md`](FEATURE_COVERAGE.md).

## Remaining ODE work

The port is not yet at OrdinaryDiffEq ODE algorithm parity. The 81 remaining
in-scope public names are concentrated in these groups:

- split/IMEX multistep methods, VCABM, and Nordsieck/JVODE variants;
- exponential and linear/Lie-group methods, plus multirate/MRI-GARK;
- extrapolation, fully implicit Radau/Gauss, Taylor, and SIMD RK methods;
- implicit second-order structural methods, PDIRK44, RKIP, and IRKC;
- the native AMF wrapper. Aliases do not require a second numerical kernel.

The public automatic/default facades are present and use a nonstiff-first
full-restart fallback after selected numerical failures. True in-flight
stiffness detection and switching remains deferred and is documented in their
handoffs.

Some of those groups require additional typed infrastructure before their
kernels can be ported faithfully: semilinear operators and exponential action,
fully implicit stage systems, multirate scheduling, Jacobian sparsity/coloring
metadata, and richer mass-matrix behavior. Typed split, second-order, and
partitioned representations now exist. Dense analytic state-Jacobian callbacks
are already supported by the current implicit and Rosenbrock kernels. DAE-only
behavior remains out of scope even when an in-scope ODE algorithm also supports
mass matrices upstream.

Benchmark results therefore cover a representative 31-algorithm slice, not
every currently implemented Rust solver or every OrdinaryDiffEq ODE solver.
