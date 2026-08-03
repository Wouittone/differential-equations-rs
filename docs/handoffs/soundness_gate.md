# Soundness gate handoff

Summary:
- Audited the current Rust regular-ODE implementation for unsafe code, panic-prone public inputs, unchecked tableau assumptions, and existing test coverage.
- The crate forbids unsafe code and the audit found no `unsafe`, `todo!`, or `unimplemented!` use in Rust source/tests.
- Fixed two reachable panics for public custom `ButcherTableau` implementations: an empty FSAL tableau underflowed while validating `stage_count - 1`, and a one-stage adaptive tableau indexed nonexistent second-stage storage during initial-step estimation.
- Reused the existing candidate buffer for the initial-step trial derivative. This removes the hidden two-stage storage assumption without adding allocations or changing built-in solver behavior.
- All six execution gates pass after the coordinator provisioned the existing ignored pinned `tests/julia/Manifest.toml` into this isolated worktree solely for validation. The manifest was neither edited nor staged.

Files changed:
- `src/explicit_rk.rs`
- `docs/handoffs/soundness_gate.md`

Public APIs added:
- None.

Upstream source and revision:
- Target reviewed against the repository's pinned SciML/OrdinaryDiffEq.jl revision `211142263781255a9aa2f910f6760b9f18ec29c8`.
- The changes are local validation/workspace soundness fixes and do not alter upstream coefficients or algorithm mappings.

Rust tests:
- Extended `explicit_rk::tests::supports_custom_tableaus_and_rejects_malformed_ones` with an empty FSAL tableau regression and a one-stage adaptive-tableau regression.
- Full result: 54 library unit tests, 10 callback/saving integration tests, 6 second-order integration tests, and all example test targets passed.

Julia tests:
- The pinned-environment check verified 13 OrdinaryDiffEq packages at revision `211142263781255a9aa2f910f6760b9f18ec29c8`.
- The full compliance suite passed all 202 assertions across its 15 test sets.
- No Julia files were edited.

Commands run:
- PASS: `cargo fmt -- --check`
- PASS: `cargo test --all-targets`
- PASS: `cargo clippy --all-targets -- -D warnings`
- PASS: `git diff --check`
- PASS: `julia --project=tests/julia tests/julia/pinned_environment.jl --check`
- PASS: `julia --project=tests/julia tests/julia/runtests.jl`
- Additional audit: `rg -n "unsafe|todo!|unimplemented!" src tests -g '*.rs'` found only `#![forbid(unsafe_code)]`.

Numerical differences:
- None expected for built-in methods. Initial-step estimation performs the same RHS evaluation and floating-point operations; only the destination scratch buffer changed.
- Previously panicking one-stage public adaptive tableaus can now execute according to their supplied coefficients.

Allocation/performance impact:
- No new allocation and no hot-loop change. Initial-step estimation reuses the already allocated candidate vector.

Known limitations:
- The pinned manifest is intentionally gitignored and coordinator-owned. Fresh worktrees need it provisioned before running the Julia checks; this does not affect the passing result in the validated worktree.

Follow-up dependencies:
- None. The Phase 1 soundness execution gate is passing.

Recommended next task:
- Begin the shared first-order integrator-driver phase using this commit as the soundness baseline.
