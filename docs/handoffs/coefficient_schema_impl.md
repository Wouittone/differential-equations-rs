# Phase 4 coefficient schema foundation

`src/coefficients.rs` now contains crate-private, tagged metadata records for
explicit Butcher, Rosenbrock, Shu–Osher, low-storage, multistep,
symplectic/partitioned, and dense interpolation coefficients. Scalars retain
their canonical rational, decimal-string, or allow-listed symbolic spelling;
the validator checks dimensions, triangular structure, finite values, dense
row shape, and required provenance without parsing files at runtime.

Inline fixtures validate RK4-shaped explicit data and AB3-shaped multistep
metadata with generic Hermite dense output. `src/generated_coefficients.rs` is
the canonical f64 source for the migrated compile-time fixtures. Its
`coefficient-method` records generate `docs/coefficients_manifest.txt` in a
stable order, and the manifest records a SHA-256 of the complete source after
newline normalization. This keeps the sensitive coefficient literals in one
place while making source/manifest drift platform-independent and detectable.

Run `scripts/generate_coefficients.ps1` after changing the canonical source.
`scripts/generate_coefficients.ps1 -Check` validates record syntax, rejects
duplicate methods, and verifies the checked-in manifest byte-for-byte. The
generic explicit RK4 facade consumes the canonical stage times, rows, and
weights; other solver families remain on hand-written constants until their
dedicated migration waves.

Canonical decimal precision exceptions are scoped to pinned coefficient
catalogues: the generated, high-order, SDIRK, and symplectic modules plus
low-storage associated constants. Individual symplectic constants may also need
an approximate-constant exception to preserve the upstream bit pattern.
Integration kernels remain subject to the normal Clippy precision checks except
where a catalogue and its kernel still share a legacy module.

Validation:

```text
cargo fmt -- --check: pass
cargo test --all-targets: pass (82 unit/integration tests plus examples)
cargo clippy --all-targets -- -D warnings: pass
git diff --check: pass
```

Pinned Julia checks remain required after the pending implicit linear-caller
wave. No public algorithm names changed, so the inventory schema is unchanged.
