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

The initial target is not feature parity with the full SciML ecosystem.
Events, dense interpolation, automatic stiffness switching, arbitrary numeric
types, DAEs, sensitivities, and the long tail of solver algorithms will be
added only after the core performance and memory hypothesis has evidence.

## Method

Each algorithm is developed in three layers:

1. implement and unit-test the numerical kernel in Rust;
2. generate reference results with the corresponding Julia algorithm and
   compare endpoints, saved trajectories, and solver statistics;
3. benchmark matched Rust and Julia workloads for elapsed time and peak memory.

Julia tests use an isolated project under `tests/julia`. Rust tests remain
usable without Julia; cross-language compliance tests are explicit so normal
`cargo test` runs stay fast and deterministic.

## Roadmap

- [ ] Define the ODE problem, solver options, solution, and statistics API.
- [ ] Implement adaptive `Tsit5` with reusable stage storage.
- [ ] Validate `Tsit5` against OrdinaryDiffEq.jl on scalar and vector problems.
- [ ] Establish matched runtime and peak-memory benchmarks.
- [ ] Add dense output and save-at behavior.
- [ ] Select a stiff solver based on benchmark coverage, likely
      `Rosenbrock23` or `Rodas5P`.

## Status

Pre-alpha. The crate is not yet suitable for scientific or production use.
