# Agent operating rules

Sub-agents are expected and should be used for bounded work. The coordinator
owns architecture, shared-file integration, merge order, and final
verification.

## Coordinator responsibilities

The coordinator must maintain the dependency graph and task queue; own
`src/lib.rs`, aggregate Julia test includes, manifests, coverage documents, and
status/blocker files; assign non-overlapping files; merge only passing branches;
regenerate the inventory after public algorithm changes; run complete
verification after architecture waves; and continue independent work when an
agent is blocked.

## Sub-agent responsibilities

Every sub-agent must inspect the repository before editing, use a separate
branch or worktree for nontrivial tasks, use only granted files, avoid
coordinator-owned files, validate against the pinned upstream commit, add Rust
and Julia tests for public algorithms, preserve or document numerical
differences, run formatting/tests/Clippy, and return a reproducible handoff.

## Task-card format

Every task should contain:

```text
Task:
Objective:
Upstream revision:
Allowed files:
Forbidden files:
Dependencies already satisfied:
Required implementation:
Required Rust tests:
Required Julia tests:
Required commands:
Known limitations:
Definition of done:
Handoff report fields:
```

The handoff report must include:

```text
Summary:
Files changed:
Public APIs added:
Upstream source and revision:
Rust tests:
Julia tests:
Commands run:
Numerical differences:
Allocation/performance impact:
Known limitations:
Follow-up dependencies:
Recommended next task:
```

## Recommended roles

Use sub-agents for soundness review, the shared integrator driver,
vector/matrix interfaces, coefficient schema/generation, problem
representations, dense output/controllers, individual solver-family ports,
Julia compliance, independent wave review, and benchmark/allocation review.

Family agents may spawn child agents for separate coefficient audits,
implementations, or fixtures when the child has a bounded deliverable and does
not touch shared files. The parent owns child integration and reporting.

## Parallelism rules

Safe parallel work includes separate solver modules, upstream source audits,
independent fixtures, inventory generation, and performance measurements.

Do not run multiple agents editing the same driver interface, coefficient
schema, shared callback implementation, `src/lib.rs`,
`tests/julia/runtests.jl`, or project manifest. Freeze shared interfaces before
launching dependent family ports.

## Blocker policy

Agents must not wait for user input. When blocked, create a minimal reproducer,
identify the exact missing dependency or upstream behavior, record it in
`docs/OVERNIGHT_BLOCKERS.md`, return control to the coordinator, and continue
independent tasks. If a test fails, classify it as an implementation defect,
upstream mismatch, known upstream broken test, or unsupported feature. Never
silently reclassify a failure as success.

## Unattended queue behavior

When an agent finishes, the coordinator automatically reads its handoff, runs
targeted verification, merges or rejects the branch, updates
`docs/OVERNIGHT_STATUS.md`, regenerates the inventory when public names change,
and spawns the next dependency-ready task. Stop only at final parity, a
documented hard environmental blocker, or an explicit safety condition.

