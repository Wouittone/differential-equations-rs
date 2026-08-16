# Overnight blockers

Record blockers here instead of waiting for user input.

Each entry must include:

```text
ID:
Date/time:
Agent:
Phase:
Severity: blocking | major | minor
Reproducer:
Expected behavior:
Observed behavior:
Upstream reference:
Likely cause:
Independent work started:
Proposed resolution:
Retry condition:
Status:
```

## Open blockers

ID: JULIA-PATH-20260809
Date/time: 2026-08-09T15:12:00+02:00
Agent: /root
Phase: Phase 8 compliance gate
Severity: major
Reproducer: `julia --project=tests/julia tests/julia/pinned_environment.jl --check`
Expected behavior: Julia verifies the tracked OrdinaryDiffEq packages at the pinned revision.
Observed behavior: PowerShell reports `julia: The term 'julia' is not recognized`; no executable is on PATH.
Upstream reference: SciML/OrdinaryDiffEq.jl revision `211142263781255a9aa2f910f6760b9f18ec29c8`.
Likely cause: coordinator environment currently lacks the Julia executable; previous isolated waves with Julia available passed their full suites.
Independent work started: Rust gates, inventory regeneration, SSPRK implementation, and report-only final audit continued; pRRK22 completed all six gates in its worker.
Proposed resolution: restore the pinned Julia 1.12.6 executable/toolchain on PATH and rerun both mandated Julia commands from the integrated checkout.
Retry condition: `Get-Command julia` resolves an executable, then rerun the reproducer and `julia --project=tests/julia tests/julia/runtests.jl`.
Status: open; no source or package blocker identified.

Latest verification: 2026-08-16T19:07:00+02:00 coordinator reran the pinned
reproducer; PowerShell still reports that `julia` is not recognized. Rust
gates and inventory regeneration continue independently, and the exact retry
condition above is unchanged.
