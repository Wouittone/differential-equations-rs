# Controller metadata foundation

`ControllerConfig` now carries an optional integral-history exponent and a
checked `step_factor_with_history` helper for future PI/PID policies. The
default proportional controller remains bit-for-bit unchanged (`exponent =
0`); no existing solver changes controller behavior yet. Tests cover default
equivalence, history use, and missing-history fallback.

Validation:

```text
cargo fmt -- --check: pass
cargo test --all-targets: pass (88 unit/integration tests plus examples)
cargo clippy --all-targets -- -D warnings: pass
git diff --check: pass
```
