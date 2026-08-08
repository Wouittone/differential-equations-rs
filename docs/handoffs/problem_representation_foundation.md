# Phase 5 problem-representation foundation

`SplitOdeProblem` now keeps explicit and implicit RHS components, shared
parameters, dimensions, and time span in a typed split/IMEX representation.
`MassMatrixOdeProblem` models a regular ODE with a finite constant dense mass
matrix and rejects empty states, dimension mismatches, non-finite entries, and
overflow; singular DAE residual behavior is not represented.

Both types expose evaluation/accessor methods without routing through the
first-order solver until a dedicated split/mass kernel wave is complete.
Tests cover component evaluation and representation validation.

Validation:

```text
cargo fmt -- --check: pass
cargo test --all-targets: pass (86 unit/integration tests plus examples)
cargo clippy --all-targets -- -D warnings: pass
git diff --check: pass
```

The pinned Julia compliance suite is unchanged by this representation-only
slice and will be rerun with the first split/mass algorithm implementation.
