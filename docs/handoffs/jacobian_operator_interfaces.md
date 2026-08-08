# Phase 3 Jacobian/operator interface slice

The checked linear layer now includes crate-private `LinearOperator`,
`IdentityOperator`, and finite dense `DenseOperator` seams with dimension and
coefficient validation. `JacobianProvider` adapts the existing analytic
Jacobian callback without changing `OdeProblem`'s public API and reports
whether an analytic callback is available.

Tests cover analytic callback selection, identity and dense matrix-vector
application, and dimension errors. Existing solver callers remain unchanged in
this bounded slice; the implicit caller migration is tracked separately.

Validation on the coordinator branch:

```text
cargo fmt -- --check: pass
cargo test --all-targets: pass (84 unit/integration tests plus examples)
cargo clippy --all-targets -- -D warnings: pass
git diff --check: pass
```

Pinned Julia and inventory checks were already green immediately before this
crate-private-only change and must be rerun in the integrated Phase 3 gate.
