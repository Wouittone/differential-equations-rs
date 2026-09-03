# Benchmarks

All benchmark sources live under `benches/`.

## Regression suite

`solver_performance` tracks representative explicit, stiff, dense-output, and
sequential/parallel ensemble paths with stable Criterion-compatible IDs:

```console
cargo bench --locked --bench solver_performance
```

CI builds this target with `cargo codspeed build` and always executes a smoke
pass. Uploads are enabled only after the repository is connected to CodSpeed
and its `CODSPEED_ENABLED` repository variable is set to `true`. Fork and
Dependabot pull requests remain smoke-only because GitHub does not grant them
an upload-capable OIDC token.

To compare a local branch against a saved baseline:

```console
cargo bench --bench solver_performance -- --save-baseline main
cargo bench --bench solver_performance -- --baseline main
```

The `hybrid_workspace/tsit5da_matrix/{128,256,1024}` cases measure short
Tsit5DA ODE solves with increasing matrix-state sizes, including workspace
construction but excluding problem construction and initial tableau parsing.
This specialization uses explicit stages and should not allocate dense
Jacobian or factorization buffers. Run these cases alone with:

```console
cargo bench --locked --bench solver_performance -- hybrid_workspace
```

The `rosenbrock_resource_allocations` integration test also checks allocated
bytes scale linearly when vector and non-square matrix states double in size,
with fixed/adaptive stepping and dense output enabled/disabled. It checks the
same decay ODE with scalar states and verifies numerical results and shapes.
This deterministic allocation check complements the timing benchmarks:

```console
cargo test --locked --test rosenbrock_resource_allocations
```

## Matched Rust/Julia matrix

The matched 31-algorithm sources are in `benches/comparison`. Run timing and
allocation measurements separately so allocation instrumentation cannot skew
the timing lane:

```console
cargo bench --locked --bench comparison_matrix -- --repetitions 20
cargo bench --locked --features allocation-metrics --bench comparison_matrix -- --repetitions 20
julia --startup-file=no --project=tests/julia benches/comparison/julia_matrix.jl --repetitions 20 --mode timing
julia --startup-file=no --project=tests/julia benches/comparison/julia_matrix.jl --repetitions 20 --mode allocation
```

Each command writes CSV to standard output. Compare rows with matching
algorithm names, dimensions, tolerances, and solver modes. Benchmark results
are machine- and revision-specific artifacts and are deliberately not checked
into the repository.
