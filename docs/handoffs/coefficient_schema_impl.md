# Phase 4 coefficient schema foundation

`src/coefficients.rs` now contains crate-private, tagged metadata records for
explicit Butcher, Rosenbrock, Shu–Osher, low-storage, multistep,
symplectic/partitioned, and dense interpolation coefficients. Scalars retain
their canonical rational, decimal-string, or allow-listed symbolic spelling;
the validator checks dimensions, triangular structure, finite values, dense
row shape, and required provenance without parsing files at runtime.

Inline fixtures validate RK4-shaped explicit data and AB3-shaped multistep
metadata with generic Hermite dense output. `src/generated_coefficients.rs`
contains deterministic compile-time RK4, AB3, and VelocityVerlet constants;
`scripts/generate_coefficients.ps1 -Check` verifies the canonical manifest.
The generic explicit RK4 facade now consumes the generated stage times, rows,
and weights; all other solver families remain on hand-written constants until
their dedicated migration waves.

Validation:

```text
cargo fmt -- --check: pass
cargo test --all-targets: pass (82 unit/integration tests plus examples)
cargo clippy --all-targets -- -D warnings: pass
git diff --check: pass
```

Pinned Julia checks remain required after the pending implicit linear-caller
wave. No public algorithm names changed, so the inventory schema is unchanged.
