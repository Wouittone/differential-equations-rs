# Prompt for an unattended Codex run

Copy the prompt below into a fresh Codex task opened at the repository root.

```text
Work continuously on the regular initial-value ODE parity program in this
repository without waiting for external input. Use sub-agents extensively for
bounded independent tasks, and follow docs/OVERNIGHT_EXECUTION_PLAN.md and
docs/AGENT_OPERATING_RULES.md as the source of truth.

Target native regular ODE algorithms from
SciML/OrdinaryDiffEq.jl revision
211142263781255a9aa2f910f6760b9f18ec29c8. Exclude SDEs, DDEs, BVPs, PDEs,
steady-state solvers, DAE-only residual behavior, and external wrappers.

First run a simplifier/soundness gate. Inspect the current worktree, preserve
unrelated changes, run the existing Rust and pinned Julia suites, and fix only
concrete correctness, allocation, or documentation issues. Do not start new
solver ports until that gate passes.

Then execute the phases in docs/OVERNIGHT_EXECUTION_PLAN.md: exact inventory;
shared first-order integrator driver; internal vector/matrix, Jacobian, operator,
and linear-solver interfaces; declarative coefficient schema and compile-time
Rust generation; split/IMEX, second-order, partitioned, and nonsingular
mass-matrix problem representations; method-specific dense output, time stops,
callback root finding, and controller parity; solver-family migration; and the
final Rust, pinned-Julia, inventory, performance, and allocation audit.

Use separate branches or worktrees for nontrivial sub-agent tasks. Give every
agent a task card containing objective, upstream reference, allowed files,
forbidden files, required tests, commands, limitations, and definition of done.
Coordinator-owned files include src/lib.rs, tests/julia/runtests.jl, project
manifests, coverage docs, and overnight status/blocker files unless a task
explicitly grants ownership.

Do not ask me questions. If an agent is blocked, require a minimal reproducer,
record the blocker in docs/OVERNIGHT_BLOCKERS.md, and continue independent work.
Never weaken tolerances, delete failing tests, or claim unsupported dense
output/controller/problem-representation parity. Preserve upstream provenance
and document intentional differences.

After each agent finishes, inspect its handoff, run targeted verification,
merge only passing work, update docs/OVERNIGHT_STATUS.md, regenerate the
inventory when public algorithms change, and spawn the next dependency-ready
task. Continue until final parity criteria are met or a genuine environmental
blocker is documented with an exact retry condition.

At minimum, maintain these gates:

cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl

Every public algorithm must have Rust tests and a matched Julia compliance
case covering applicable convergence, forward/backward integration, fixed or
adaptive behavior, callbacks, saving, Jacobians, failures, and statistics.

Leave the repository passing, with docs/OVERNIGHT_STATUS.md and
docs/OVERNIGHT_BLOCKERS.md updated, even if some independent work remains.
```
