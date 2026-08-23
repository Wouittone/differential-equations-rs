# Rodas5Pr parity handoff

## Scope

This wave adds the regular-ODE `Rodas5Pr` constructor and export. The method
uses the exact pinned `Rodas5PTableau` and its eight-stage `RODAS5PA`/
`RODAS5PC` coefficients. It also ports the pinned residual-control branch from
`OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl`: when adaptive
`EEst < 1`, the three `RODAS5PH` rows construct the midpoint state and an
additional residual estimate is taken as the maximum of the embedded and
residual estimates. The residual check is allocation-free and is restricted to
regular ODEs; DAE/SDE/wrapper behavior is intentionally out of scope.

Pinned source: `211142263781255a9aa2f910f6760b9f18ec29c8`

- Tableau: `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl`,
  `Rodas5PTableau` lines 51--71 (`RODAS5PA`, `RODAS5PC`, `RODAS5Pc`,
  `RODAS5Pd`, `RODAS5PH`).
- Algorithm metadata: `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl`,
  lines 95--100 (`Rodas5Pr`, order 5, adaptive).
- Step implementation: `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl`,
  lines 530--551 and 816--837 (additional residual control).
- Tableau dispatch: `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_caches.jl`,
  lines 426--427 (`Rodas5P` and `Rodas5Pr` share `Rodas5PTableau`).

## Rust surface

- Public type: `differential_equations::algorithms::rosenbrock::Rodas5Pr`.
- Native implementation: `src/rosenbrock_extended.rs`.
- Focused test: `rodas5pr_matches_rodas5p_on_regular_ode_paths` covers fixed
  tableau identity, adaptive stiff integration, and RHS activity.

## Verification

Passed:

```text
cargo fmt
cargo test --lib rosenbrock_extended::tests::rodas5pr_matches_rodas5p_on_regular_ode_paths
git diff --check
```

The pre-existing `methods_have_their_expected_fixed_step_orders` test still
fails for its existing `Rodas23W` ratio assertion (`ratios[13] > 7.0`); this
failure is unrelated to Rodas5Pr. Full required commands remain to be run by
the integrator after cherry-pick:

```text
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check
```

Julia compliance is pending because `julia` is not on PATH in this worker
(`JULIA-PATH-20260816`). Retry the pinned compliance fixture with the Julia
executable available; no Julia output is claimed by this handoff.
