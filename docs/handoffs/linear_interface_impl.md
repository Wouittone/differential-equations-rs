# Phase 3 checked linear interface implementation

The first Phase 3 slice adds crate-private checked dense state/matrix views and
an owning `DenseLu` factorization cache in `src/linear.rs`. The existing flat
slice `factorize`/`solve_factorized` functions remain unchanged for current
solver callers, so this slice has no numerical or allocation impact on public
solves.

`StateLayout` rejects zero dimensions, checked multiplication overflow, and
length mismatches. `DenseLu::factorize` validates matrix length and finite
coefficients, preserves the existing absolute `f64::EPSILON` pivot rule, and
records an explicit caller-supplied revision. `DenseLu::solve` validates the
right-hand-side length before reusing the immutable factors.

Inline tests cover wrong-length views, row-pivoted solves, revision retention,
non-finite coefficients, and singular matrices. Existing implicit and
Rosenbrock callers are intentionally not migrated in this bounded slice; the
next Phase 3 wave must replace one caller and prove unchanged compliance and
allocation behavior before broader migration.

Validation on the coordinator branch:

```text
cargo fmt -- --check: pass
cargo test --all-targets: pass (80 unit/integration tests plus examples)
cargo clippy --all-targets -- -D warnings: pass
git diff --check: pass
julia --project=tests/julia tests/julia/pinned_environment.jl --check: pass
julia --project=tests/julia tests/julia/runtests.jl: pass (202/202)
```

No public API or inventory entry changed.
