# pRRK33 regular-ODE wave

Implemented the native fixed-step `Prrk33`/`pRRK33` algorithm from pinned
OrdinaryDiffEqSSPRK revision `211142263781255a9aa2f910f6760b9f18ec29c8`.
The coefficient and stage semantics follow `lib/OrdinaryDiffEqSSPRK/src/algorithms.jl`
(`pRRK33` declaration), `ssprk_caches.jl` (SSPRK(3,3) Shu--Osher coefficients),
and `ssprk_perform_step.jl` (`_prrk33_coeffs` and `perform_step!`, lines
1682--1781 in the pinned source).

The Rust kernel applies the exact \(\psi_1,\psi_2,\psi_3\) rescaling, modified
abscissae, and three RHS stage evaluations. `kappa = 0` reduces to SSPRK33;
nonzero `kappa` enables the pinned parametric relaxation. It is fixed-step
only, preserves shared callback/backward/save-at behavior, and performs no
per-step allocations after initialization.

Validation in this worktree:

- `cargo fmt --all`
- `cargo test --all-targets` (97 unit tests plus integration/example targets)
- `cargo clippy --all-targets -- -D warnings`
- `git diff --check`
- `cargo run --quiet --example ssprk_compliance` (pRRK33 endpoint agrees with SSPRK33)

Julia validation was attempted but Julia is not installed in this environment;
retry `julia --project=tests/julia tests/julia/pinned_environment.jl --check`
and `julia --project=tests/julia tests/julia/runtests.jl` on a Julia-enabled host.
