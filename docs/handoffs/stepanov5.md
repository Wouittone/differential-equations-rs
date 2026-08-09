# Stepanov5 implementation handoff

- Upstream package: `OrdinaryDiffEqLowOrderRK`
- Pinned revision: `211142263781255a9aa2f910f6760b9f18ec29c8`
- Algorithm declaration: `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:400-423`
- Tableau source: `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1648-1729`
- Step source: `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:1657-1712`

The Rust `Stepanov5` facade uses the shared explicit Runge–Kutta driver with
the exact seven-stage embedded (4,5) tableau, endpoint FSAL row, nodes, and
error estimator from the pinned Julia cache. The final row is the primary
weights with a zero endpoint weight, so accepted steps reuse the endpoint
derivative unless a callback invalidates the cache.

Validation on this branch:

- `cargo fmt -- --check`
- `cargo test --all-targets` (all tests passed, including Stepanov5 forward,
  backward, fixed-order, adaptive, callback, save-at, and allocation tests)
- `cargo clippy --all-targets -- -D warnings`
- `git diff --check`

Julia validation is pending because `julia` is not installed in the execution
environment. Retry `julia --project=tests/julia tests/julia/pinned_environment.jl --check`
and `julia --project=tests/julia tests/julia/runtests.jl` when Julia is available.
