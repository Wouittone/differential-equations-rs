# Benchmarks

All benchmark sources live under `benches/`.

## Regression suite

`solver_performance` tracks representative explicit, stiff, dense-output, and
sequential/parallel ensemble paths with stable Criterion-compatible IDs:

```console
cargo bench --locked --bench solver_performance
```

CI builds this target with `cargo codspeed build` and uploads measurements on
trusted branches and pull requests. Dependabot pull requests still build the
target, but skip upload because GitHub does not grant them an OIDC token.

To compare a local branch against a saved baseline:

```console
cargo bench --bench solver_performance -- --save-baseline main
cargo bench --bench solver_performance -- --baseline main
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
