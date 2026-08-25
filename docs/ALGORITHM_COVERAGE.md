# ODE algorithm coverage

Coverage is measured against SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. A Rust type is counted only when
it has a Julia differential-compliance test; sharing a kernel with another
method is not enough.

## Current status

The generated pinned-revision inventory detects **345 of 345** in-scope public
ODE names as both implemented and Julia-tested, with **zero** included names
missing a Rust implementation or matched compliance fixture. The complete
per-name and per-family ledger is generated in
[`ODE_PARITY_INVENTORY.md`](ODE_PARITY_INVENTORY.md), with JSON and CSV forms
alongside it.

Coverage includes automatic/default composite facades and the generic
user-tableau alias in addition to every included family in the pinned ledger:
explicit, implicit, split/IMEX, multistep, exponential, linear/Lie,
multirate, second-order, symplectic, stabilized, SIMD RK, Taylor, AMF, RKIP,
IRKC, and structural methods. A representative **31-algorithm** slice appears in
the matched benchmark matrix; the benchmark is not intended to cover every
implemented method.

This is algorithm-name coverage, not full feature parity. Basic discrete and
scalar continuous callbacks plus `save_at` sampling are shared by implemented
first-order methods. Tsit5, DP5, BS5, Owren--Zennaro 3/4/5, and Verner 6/7/8/9
provide retained method-specific dense segments and root localization, while
DPRKN6 provides its pinned second-order dense extension. Implemented SSP methods
match their pinned dense dispatch: SSPRK22/33/43/432 use the special quadratic
extension and the remainder use retained generic Hermite segments. Other method
families, arbitrary scalar types, limiters, and every upstream controller and
callback option remain separate work. See
[`FEATURE_COVERAGE.md`](FEATURE_COVERAGE.md).

## Scope boundary

Algorithm-name parity is complete for the pinned ODE scope. The public
automatic/default facades use a nonstiff-first full-restart fallback after
selected numerical failures; true in-flight stiffness switching remains a
documented feature distinction. DAE-only behavior, arbitrary scalar/container
types, and the full Julia integrator interface remain outside this algorithm
ledger and are tracked separately in [`FEATURE_COVERAGE.md`](FEATURE_COVERAGE.md).

Benchmark results therefore cover a representative 31-algorithm slice, not
every currently implemented Rust solver or every OrdinaryDiffEq ODE solver.
