# Generated Vern7 coefficient wave

The public `Vern7` constructor remains a facade over the shared explicit
Runge--Kutta kernel. Its endpoint tableau is now represented by deterministic
generated records, preserving the pinned OrdinaryDiffEq Verner coefficients.

Pinned source: OrdinaryDiffEqVerner revision
`211142263781255a9aa2f910f6760b9f18ec29c8`, specifically
`lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl` (Vern7 tableau block,
lines 585-1369 in the pinned source) and the corresponding Vern7 perform-step
path in `verner_rk_perform_step.jl`.

The generated fixture retains all ten endpoint nodes, lower-triangular stage
rows, primary weights, and embedded error weights. OrdinaryDiffEq's additional
stages used by its method-specific lazy dense interpolant are intentionally
outside this endpoint-only generic kernel; endpoint fixed/adaptive stepping,
backward integration, callbacks, and allocation invariance are covered by the
new Rust tests. Existing `tests/julia/verner.jl` already compares Vern7 fixed
and adaptive endpoint results against the pinned Julia implementation, so no
duplicate fixture was needed.
