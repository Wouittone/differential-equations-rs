# Phase 2 lifecycle audit

Date: 2026-08-08
Branch: `codex/phase2-lifecycle-audit`
Base: `e747d0b` (Adams/variable-Adams shared-driver migration)
Scope: native regular first-order ODE algorithms. Second-order, SDE/DDE/BVP/PDE,
steady-state, DAE-only residual behavior, and external wrappers are excluded.

## Result

**PASS — recommend closing Phase 2 shared first-order driver migration.**

The repository has one first-order numerical lifecycle loop, in
`src/integrator.rs:228`. Every in-scope first-order family implements the
`StepKernel` contract and is dispatched through that loop. No in-scope module
contains another `while`/`loop` integration lifecycle. Callback application,
accepted-segment recording, termination, and kernel accept/reject hooks are
centralized in the driver.

## Source evidence

The lifecycle contract is declared in `src/integrator.rs:120-169`; the shared
entry point starts at `src/integrator.rs:171`. Driver-owned lifecycle behavior
is visible at:

- initial callbacks and recorder setup: `src/integrator.rs:201-210`;
- attempt/rejection accounting and recoverable failure handling:
  `src/integrator.rs:228-267`;
- accepted-step callback localization and recording: `src/integrator.rs:273-301`;
- kernel accept hook and controller update: `src/integrator.rs:303-319`;
- numerical rejection hook: `src/integrator.rs:321-327`.

The in-scope numerical kernels are:

| Family | `StepKernel` implementation | Driver dispatch evidence |
| --- | --- | --- |
| Explicit RK (including generic/custom, low-order, SSP, Tsit5, Verner) | `src/explicit_rk.rs:757-886` | `src/explicit_rk.rs:731`; named facades call the shared `ExplicitRungeKutta` at `src/explicit_rk.rs:440`, `src/explicit_rk.rs:614`, `src/ssprk_extended.rs:43`, `src/tsit5.rs:80`, `src/verner.rs:558` |
| Fixed implicit Euler family | `src/implicit.rs:102-192` | `src/implicit.rs:32` |
| Low-storage RK | `src/low_storage_rk.rs:381-472` | `src/low_storage_rk.rs:351` |
| Rosenbrock23/32/Rodas4/Rodas5P | `src/rosenbrock.rs:88-188` and `src/rosenbrock_extended.rs:462-557` | `src/rosenbrock.rs:66`; `src/rosenbrock_extended.rs:347` |
| TRBDF2 | `src/trbdf2.rs:100-200` | `src/trbdf2.rs:38` |
| Fixed Adams | `src/adams.rs:174-302` | `src/adams.rs:91` |
| Variable-coefficient Adams | `src/variable_adams.rs:161-298` | `src/variable_adams.rs:56` |

The only additional `StepKernel` implementation is the test-only mock at
`src/integrator.rs:407-492`; it exercises driver lifecycle invariants and is
not an algorithm family.

## Loop and exception audit

The command

```text
rg -n "\\bwhile\\b|\\bloop\\s*\\{" src\\adams.rs src\\explicit_rk.rs src\\implicit.rs src\\low_storage_rk.rs src\\rosenbrock.rs src\\rosenbrock_extended.rs src\\trbdf2.rs src\\ssprk_extended.rs src\\tsit5.rs src\\verner.rs src\\variable_adams.rs
```

returns no matches in any in-scope solver module. Across `src`, the only
first-order lifecycle loop is `src/integrator.rs:228`. The other matches are
intentional exceptions:

- `src/second_order.rs:440` is the standalone fixed-step partitioned
  second-order solver, explicitly outside regular first-order Phase 2 scope.
- `src/second_order.rs:945` and `src/solution.rs:77` are recorder `save_at`
  traversal loops, not integration lifecycles.

Numerical `for` loops in kernels are stage/Newton/linear-algebra/history work;
they do not own time progression, callbacks, saving, termination, or controller
policy.

## Gate results

All required gates passed in this clean audit worktree:

| Gate | Result |
| --- | --- |
| `cargo fmt -- --check` | pass |
| `cargo test --all-targets` | pass — 77 unit tests plus all integration/example targets; no failures |
| `cargo clippy --all-targets -- -D warnings` | pass |
| `git diff --check` | pass |
| `julia --project=tests/julia tests/julia/pinned_environment.jl --check` | pass — all 13 OrdinaryDiffEq packages at pinned revision |
| `julia --project=tests/julia tests/julia/runtests.jl` | pass — all compliance groups (202 total checks) |

No blocker was observed; `docs/OVERNIGHT_BLOCKERS.md` does not require an
entry for this wave.

