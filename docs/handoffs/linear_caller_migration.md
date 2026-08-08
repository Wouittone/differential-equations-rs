# Implicit linear caller migration

This wave migrates the first Jacobian factorization and Newton solve in
`src/implicit.rs` to the checked `StateLayout`/`DenseLu` interface introduced
in `a1bb8fa`. The layout validates the state and matrix dimensions before the
factorization and the first correction solve uses the revision-tagged dense
factorization. Existing preallocated LU storage remains available for later
chord-Newton refreshes, so repeated refreshes stay allocation-free and retain
the established reuse policy and singular-system error.

The compliance endpoints are byte-for-byte unchanged for
`cargo run --example implicit_compliance`:

```
implicit_euler,4.11435264507034812e-2,3.69711212329119410e-1
implicit_midpoint,4.09151729242362219e-2,3.67876375476220874e-1
trapezoid,4.09151729242362219e-2,3.67876375476220874e-1
```

Regression coverage in `tests/linear_caller_migration.rs` checks all three
implicit method endpoints and confirms callback-free solve allocations are
invariant between one and one thousand fixed steps. Existing implicit callback
termination and Jacobian reuse tests continue to pass.

Validation commands run on this branch:

* `cargo fmt -- --check`
* `cargo test --all-targets`
* `cargo clippy --all-targets -- -D warnings`
* `git diff --check`
* `julia --project=tests/julia tests/julia/pinned_environment.jl --check`
* `julia --project=tests/julia tests/julia/runtests.jl`

No public names or inventory sources changed.
