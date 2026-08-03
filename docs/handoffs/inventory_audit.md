# Inventory audit handoff

Summary:

- Regenerated the exact-revision native regular-IVP ODE inventory as schema version 2.
- Audited all 39 `OrdinaryDiffEq*` subpackages: 33 solver packages and 6 support-only packages.
- Resolved and checked source references for all 349 solver constructor rows.
- Corrected the prior false positive `OrdinaryDiffEqSDIRK.Predictor`, which is a nonlinear-stage predictor enum namespace, not a solver constructor.
- Classified 345 in-scope names, 4 explicitly excluded names, 12 aliases/configured constructors, 65 Rust-and-Julia-tested names, and 280 missing in-scope names.
- Added explicit fixed/adaptive, Jacobian, linear-solver, dense-output, controller, Rust, Julia, alias, and exclusion fields.
- Added a non-mutating `-Check` mode that regenerates into an isolated temporary directory, compares SHA-256 hashes, and validates every source reference and required schema field.

Files changed:

- `scripts/generate_ode_inventory.ps1`
- `docs/ode_algorithm_inventory.json`
- `docs/ode_algorithm_inventory.csv`
- `docs/ODE_PARITY_INVENTORY.md`
- `docs/ALGORITHM_COVERAGE.md`
- `docs/handoffs/inventory_audit.md`

Public APIs added:

- None. The inventory script adds a `-Check` command-line switch.

Upstream source and revision:

- Repository: `https://github.com/SciML/OrdinaryDiffEq.jl`
- Revision: `211142263781255a9aa2f910f6760b9f18ec29c8`
- Reusable clean local checkout: `D:/Source/_review/OrdinaryDiffEq.jl`
- Checkout state: detached HEAD at the exact revision with no working-tree changes.

Rust tests:

- `cargo fmt -- --check`: passed.
- `cargo test --all-targets`: passed, 70 tests total (54 library, 10 callback/saving integration, 6 second-order integration; example targets also built and ran).
- `cargo clippy --all-targets -- -D warnings`: passed.

Julia tests:

- `julia --project=tests/julia tests/julia/pinned_environment.jl --check`: passed; 13 OrdinaryDiffEq packages verified at the exact revision.
- `julia --project=tests/julia tests/julia/runtests.jl`: passed; all 15 reported compliance testsets and 202 assertions passed.
- The coordinator-provisioned ignored `tests/julia/Manifest.toml` was used only for validation and was not edited or committed.

Commands run:

- `./scripts/generate_ode_inventory.ps1 -UpstreamPath 'D:\Source\_review\OrdinaryDiffEq.jl'`
- `./scripts/generate_ode_inventory.ps1 -UpstreamPath 'D:\Source\_review\OrdinaryDiffEq.jl' -Check` (twice; byte-identical both times)
- `cargo fmt -- --check`
- `cargo test --all-targets`
- `cargo clippy --all-targets -- -D warnings`
- `git diff --check`
- `julia --project=tests/julia tests/julia/pinned_environment.jl --check`
- `julia --project=tests/julia tests/julia/runtests.jl`

Numerical differences:

- None. No Rust algorithm or runtime code changed.

Allocation/performance impact:

- None in solver code. Inventory generation remains an offline documentation operation and takes about 35 seconds on this checkout.

Known limitations:

- The public parity surface is deliberately the exported constructor surface of classified native OrdinaryDiffEq solver subpackages; internal unexported experimental types are not inventory targets.
- Rust and Julia status detection remains name/import based. It records whether matched compliance is detected, not the depth or quality of each numerical fixture.
- Dense-output requirements identify packages/families that require method-specific parity, but exact interpolant polynomial/order remains a per-solver port audit.
- Generated Julia constructor families cite the exact tuple entry that generates the concrete type; direct constructors and aliases cite their definition line.
- `AMF` remains included because it is a native OrdinaryDiffEq wrapper around native Rosenbrock-W methods, not an external solver wrapper.
- `Tsit5DA` remains in scope because its upstream documentation states that it reduces to an explicit Runge-Kutta method for pure ODEs; its DAE-only behavior is not in scope.

Follow-up dependencies:

- The coordinator should merge this commit before using the inventory counts in later family task cards.
- Regenerate and run `-Check` after every public Rust algorithm-name change.
- Family agents should use the explicit representation and requirement fields to select architecture dependencies before implementation.

Recommended next task:

- Use the 280 missing in-scope rows to seed dependency-ordered solver-family task cards after the shared integrator and representation gates pass, starting with remaining explicit/high-order RK constructors.
