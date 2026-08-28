# Contributing

Contributions are welcome. Open an issue before a large API or numerical-method
change so design and validation expectations can be agreed early.

## Development baseline

- Use Rust 1.85 or newer; CI verifies the declared MSRV separately from stable.
- Keep `Cargo.lock` committed and use `--locked` for release-oriented checks.
- Preserve unrelated working-tree changes and keep commits focused and
  semantically named.

Before submitting a change, run:

```console
cargo +1.85 fmt --all -- --check
cargo +1.85 clippy --workspace --all-targets --all-features -- -D warnings
cargo +1.85 test --workspace --all-features
cargo deny check
pwsh ./scripts/check_package_policy.ps1
```

## Numerical methods and tableaus

Fixed Runge--Kutta coefficients belong in a JSON resource below `tableaux/`,
not in generated Rust or solver-module constants. Use exact-style expressions
for rational or radical values when practical. The shared parser and procedural
macro must reject malformed data at compile time and preserve typed runtime
errors.

Add convergence, failure-path, callback/output, and allocation tests in
proportion to the method's behavior. When porting SciML data, record the pinned
upstream revision and retain its license notice in `THIRD_PARTY_NOTICES.md`.

## Compatibility and documentation

Public API changes require rustdoc, an entry under `CHANGELOG.md`'s Unreleased
section, and an update to the applicable coverage document. Keep examples
compilable and avoid promising feature parity beyond the tested surface.
