# Rodas5Pe parity handoff

Implemented native regular-ODE `Rodas5Pe` parity against OrdinaryDiffEq at
revision `211142263781255a9aa2f910f6760b9f18ec29c8`.

The upstream `Rodas5PeTableau` reuses `RODAS5PA`, `RODAS5PC`, `RODAS5Pc`,
`RODAS5Pd`, `RODAS5PH`, and primary weights from `Rodas5PTableau`; only the
embedded weights differ. The Rust method therefore reuses the existing
eight-stage `Rodas5P` primary tableau and adds the exact pinned embedded
weights:

```text
[ 0.2606326497975715, -0.005158627295444251, 1.3038988631109731,
  1.235000722062074, -0.7931985603795049, -1.005448461135913,
 -0.18044626132120234, 0.17051519239113755 ]
```

The regular Rosenbrock driver covers fixed and adaptive stepping, reverse-time
integration, user Jacobians, callbacks, `save_at`, and preallocated stage
workspace. Focused integration and allocation tests are in
`tests/rodas5pe.rs` and `tests/rodas5pe_allocations.rs`.

The small compliance executable in `examples/rodas5pe_compliance.rs` prints
fixed and adaptive endpoint values for `u'=u`, `u(0)=1`, over `[0,1]`.

Julia was unavailable on this worker. Retry with the exact pinned compliance
commands after `JULIA-PATH` resolves to a real Julia executable:

```powershell
& $env:JULIA_PATH --project=tests/julia tests/julia/pinned_environment.jl --check
& $env:JULIA_PATH --project=tests/julia tests/julia/runtests.jl
```
