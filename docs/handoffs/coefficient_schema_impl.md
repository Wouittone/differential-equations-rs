# Phase 4 coefficient schema foundation

`src/coefficients.rs` now contains crate-private, tagged metadata records for
explicit Butcher, Rosenbrock, Shu–Osher, low-storage, multistep,
symplectic/partitioned, and dense interpolation coefficients. Scalars retain
their canonical rational, decimal-string, or allow-listed symbolic spelling;
the validator checks dimensions, triangular structure, finite values, dense
row shape, and required provenance without parsing files at runtime.

Inline fixtures validate explicit and multistep metadata with generic Hermite
dense output. `src/generated_coefficients.rs` remains the legacy canonical f64
source for fixtures not yet migrated to declarative resources. Resource-backed
methods under `tableaux/` are parsed, validated, and compiled directly by
`define_explicit_rk_from_file!`. The source and resource records generate
`docs/coefficients_manifest.txt` in a stable order, with SHA-256 values after
newline normalization. This keeps each coefficient literal in one canonical
place while making source/manifest drift platform-independent and detectable.

Run `scripts/generate_coefficients.ps1` after changing a canonical source or
tableau resource.
`scripts/generate_coefficients.ps1 -Check` validates record syntax, rejects
duplicate methods, and verifies the checked-in manifest byte-for-byte. The
generic explicit RK4 facade is defined from `tableaux/explicit/rk4.toml`;
other solver families remain on generated or hand-written constants until
their dedicated migration waves.

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
