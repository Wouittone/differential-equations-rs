# Fixed-step Runge--Kutta--Nystrom subset

This wave ports four non-adaptive methods from OrdinaryDiffEqRKN at revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `Nystrom4`, whose acceleration may depend on velocity;
- `Nystrom4VelocityIndependent` and `Nystrom5VelocityIndependent`, whose
  acceleration must ignore velocity;
- `Rkn4`, which upstream documents as fourth order for linear inhomogeneous
  problems and second order in general.

The exact tableaus come from
`lib/OrdinaryDiffEqRKN/src/rkn_tableaus.jl`. The reusable Rust kernel implements
the generic position and velocity stage formulas used by the pinned upstream
`rkn_perform_step.jl`; the Julia fixture checks endpoints against the same pin.

The embedded velocity-independent group (`DPRKN12`, `DPRKN4`, `DPRKN5`,
`DPRKN6FM`, `DPRKN8`, `ERKN4`, `ERKN5`, and `ERKN7`) is now covered by
`rkn_adaptive.md`. The same follow-up also completes specialized `DPRKN6`,
velocity-dependent `FineRKN4`/`FineRKN5`, and history-based `IRKN3`/`IRKN4`.
Their tableaus and constant caches can be extracted reproducibly with:

```powershell
rtk proxy powershell -NoProfile -File scripts/extract_rkn_coefficients.ps1 `
    -UpstreamPath D:\Source\_review\OrdinaryDiffEq.jl
```

The script refuses any checkout whose `HEAD` differs from the pinned revision.
