# differential-equations-rs

An experimental Rust port of the ODE solvers in Julia's
[DifferentialEquations.jl ecosystem](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/).

## Goals

This project exists first and foremost as a proof of concept to answer a
specific performance question:

> Can Rust ODE solvers reach performance comparable to Julia's
> DifferentialEquations.jl while retaining Rust's substantially lower memory
> usage observed in earlier `orskit` experiments?

The working goals are:

- port representative high-value OrdinaryDiffEq.jl algorithms faithfully,
  beginning with `Tsit5` for non-stiff systems;
- compare numerical results directly with Julia and use Julia solver outputs
  as the compliance oracle;
- measure runtime, allocations, and peak memory on equivalent workloads;
- design the hot path around reusable caller-owned or solver-owned buffers;
- preserve a small, idiomatic Rust API without hiding allocation costs;
- record discrepancies instead of silently weakening tolerances.

The solver target is native OrdinaryDiffEq.jl **ODE algorithm parity**.
SDEs, RODEs, DDEs, DAEs, and external solver wrappers are explicitly out of
scope. Events, dense interpolation, arbitrary numeric types, and sensitivities
are separate API features and are not implied by algorithm parity.

## Method

Each algorithm is developed in three layers:

1. implement and unit-test the numerical kernel in Rust;
2. generate reference results with the corresponding Julia algorithm and
   compare endpoints, saved trajectories, and solver statistics;
3. benchmark matched Rust and Julia workloads for elapsed time and peak memory.

Julia tests use an isolated project under `tests/julia`. Rust tests remain
usable without Julia; cross-language compliance tests are explicit so normal
`cargo test` runs stay fast and deterministic.

Run both test layers with:

```console
cargo test
julia --project=tests/julia tests/julia/runtests.jl
```

## Roadmap

- [x] Define the ODE problem, solver options, solution, and statistics API.
- [x] Implement adaptive `Tsit5` with reusable stage storage.
- [x] Validate `Tsit5` against OrdinaryDiffEq.jl on scalar and vector problems.
- [x] Implement the shared explicit Runge–Kutta kernel plus Euler, midpoint,
      Heun, Ralston, RK4, BS3, and DP5.
- [x] Implement fixed-step Adams–Bashforth methods AB3, AB4, and AB5.
- [x] Establish dense Newton/Jacobian/linear-solve infrastructure and fixed
      Implicit Euler, Implicit Midpoint, and Trapezoid methods.
- [x] Implement SSPRK22, SSPRK33, and adaptive SSPRK43.
- [ ] Port all remaining native OrdinaryDiffEq.jl ODE algorithm families.
- [ ] Establish matched runtime and peak-memory benchmarks.
- [ ] Add dense output and save-at behavior.
- [ ] Select a stiff solver based on benchmark coverage, likely
      `Rosenbrock23` or `Rodas5P`.

## Status

Pre-alpha. The crate is not yet suitable for scientific or production use.
