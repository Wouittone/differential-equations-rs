# Rodas42 handoff

Implemented the native regular-ODE `Rodas42` algorithm from pinned
OrdinaryDiffEq.jl revision `211142263781255a9aa2f910f6760b9f18ec29c8`.

## Source mapping

- Tableau constants are transcribed from
  `lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl`,
  `RODAS42A`, `RODAS42C`, `RODAS42c`, `RODAS42d`, and `RODAS42H` (the
  `Rodas42Tableau` definition around lines 125--171).
- The shared regular-ODE Rosenbrock driver supplies fixed/adaptive stepping,
  numerical or supplied Jacobians, callbacks, `save_at`, backward integration,
  and accepted-segment recording. DAE residual behavior and wrappers are not
  included.
- Public constructor: `differential_equations::algorithms::rosenbrock::Rodas42`.

## Verification

- `cargo fmt -- --check`: pass.
- `cargo test --all-targets`: pass (110 unit tests plus all integration tests,
  including `tests/rodas42.rs`).
- `cargo clippy --all-targets -- -D warnings`: pass.
- `git diff --check`: pass.
- `cargo run --quiet --example rosenbrock_extended_compliance`: pass; the
  Rodas42 fixed endpoint is `2.71828182845909527e0` and adaptive endpoint is
  `5.40302306948189370e-1`.
- Julia checks are blocked by the environment: PowerShell reports
  `julia : The term 'julia' is not recognized ...`. Retry when the pinned Julia
  executable is installed and available on `PATH`.
