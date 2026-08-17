# Automatic/default composite and ExplicitRK wave

## Summary

Added the eight remaining public names in this bounded slice:
`AutoTsit5`, `AutoVern6`, `AutoVern7`, `AutoVern8`, `AutoVern9`,
`DefaultODEAlgorithm`, `DefaultImplicitODEAlgorithm`, and `ExplicitRK`.

The automatic facades retain their configured stiff branch and delegate the
regular ODE solve to the native nonstiff component (`Tsit5` or `Vern6`--`9`).
The default nonstiff and stiff facades delegate to `Tsit5` and `Rodas5P`,
respectively. `ExplicitRK` is an API-compatible alias over the existing
generic `ExplicitRungeKutta` tableau marker.

## Upstream source and revision

SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`.

Relevant upstream constructors are in `OrdinaryDiffEqTsit5`,
`OrdinaryDiffEqVerner`, `OrdinaryDiffEqDefault`, and
`OrdinaryDiffEqExplicitRK`.

## Files changed

- `src/composites.rs`
- `src/explicit_rk.rs`
- `src/lib.rs`
- `tests/composites.rs`
- `examples/composites_compliance.rs`
- `tests/julia/composites.jl`
- `tests/julia/runtests.jl`
- regenerated `docs/ode_algorithm_inventory.{json,csv}` and
  `docs/ODE_PARITY_INVENTORY.md`

## Validation

Focused checks passed:

```text
cargo fmt --all
cargo test --test composites                 (2 passed)
cargo test --lib composites::tests            (1 passed)
cargo run --quiet --example composites_compliance
inventory regeneration                         (132 implemented, 213 missing)
git diff --check
```

The pinned Julia executable is not available in the coordinator environment;
the new `tests/julia/composites.jl` fixture must be run when Julia is restored.
No full-project validation was run for this bounded wave.

## Numerical differences and limitations

The Rust facades intentionally do not implement OrdinaryDiffEq's runtime
stiffness detection and switching yet. The configured component is preserved
for API compatibility and future switching state. `ExplicitRK` reuses the
existing marker-based Rust tableau API; it does not add Julia's runtime
tableau object representation.

## Recommended next task

Continue with the next dependency-ready regular-ODE family, preferably a
coefficient-driven explicit or low-storage family that can reuse the frozen
driver. Re-run this wave's Julia fixture and the full family/global gates only
after the selected family is complete and Julia is available.
