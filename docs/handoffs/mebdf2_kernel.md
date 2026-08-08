# MEBDF2 fixed regular-ODE parity wave

This wave ports the pinned `OrdinaryDiffEqBDF.MEBDF2` construction at
revision `211142263781255a9aa2f910f6760b9f18ec29c8` for regular identity-mass
ODEs only. The exact upstream references are:

- `lib/OrdinaryDiffEqBDF/src/algorithms.jl:414-453` (MEBDF2 declaration,
  fixed-step `OrdinaryDiffEqNewtonAlgorithm` constructor and constant cache).
- `lib/OrdinaryDiffEqBDF/src/bdf_perform_step.jl:1045-1098` (three correction
  stages and final derivative) and `:1100-1148` (mutable equivalent).
- `lib/OrdinaryDiffEqBDF/src/bdf_caches.jl:551-602` (three stage buffers and
  nonlinear cache).

For a step of size `h`, the implementation follows the pinned sequence of
backward-Euler corrections: solve `z₁` about `uₙ` at `t+h`, solve `z₂` about
`uₙ+z₁` at `t+2h`, form
`tmp₂ = 0.5uₙ + (uₙ+z₁) - 0.5(uₙ+z₁+z₂)`, then solve the third correction
about `tmp₂` at `t+h`. The final state is `tmp₂ + z₃`. Each correction uses
checked Newton/Jacobian/LU behavior and preallocated factors after the first
checked factorization. The algorithm is fixed-step only, so adaptive options
return `AdaptiveStepUnsupported`.

The scope excludes DAE residual initialization, non-identity/singular mass
matrices, split/IMEX paths, wrappers, and adaptive/variable-order behavior.

Rust tests cover second-order convergence, stiff nonautonomous and backward
integration, callback/Jacobian safety, malformed derivative handling, and
fixed-step configuration validation. `examples/mebdf2_compliance.rs` matches
the Julia endpoint exactly for the pinned stiff test (`dt=0.01`, 100 accepted
steps). `tests/julia/mebdf2.jl` compares against `OrdinaryDiffEqBDF.MEBDF2`.
The coordinator must retain the pinned BDF Project/Manifest dependency.

