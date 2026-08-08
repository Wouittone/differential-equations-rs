# Linear factorization statistics

`SolverStats` now reports `linear_factorizations` alongside Jacobian
evaluations and linear solves. Implicit, TRBDF2, Rosenbrock23/32, and Rodas
factorization paths increment the counter exactly once per dense factorization
build; cached solves do not inflate it. Existing numerical behavior and
allocation tests remain unchanged.

Validation:

```text
cargo fmt -- --check: pass
cargo test --all-targets: pass (87 unit/integration tests plus examples)
cargo clippy --all-targets -- -D warnings: pass
git diff --check: pass
```
