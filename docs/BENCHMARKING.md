# Performance regression benchmarks

The `solver_performance` suite tracks representative hot paths with stable
benchmark IDs:

- adaptive Tsit5 on the Lorenz system;
- Rodas5P on a stiff tracking problem with an analytic Jacobian;
- retained Tsit5 dense-output queries; and
- the same 64-case ensemble executed sequentially and with Rayon.

Problem definitions, solver options, dense-query grids, and Rayon's one-time
thread-pool initialization are prepared outside timed iterations. Per-case
problem construction remains timed for ensemble benchmarks because it is part
of the public ensemble API's execution contract.

Run the Criterion-compatible suite locally with:

```console
cargo bench --bench solver_performance
```

To compare a branch against a local baseline:

```console
git switch main
cargo bench --bench solver_performance -- --save-baseline main
git switch -
cargo bench --bench solver_performance -- --baseline main
```

CI builds the same target with `cargo codspeed build` and runs it through
CodSpeed's deterministic simulation mode. `workflow_dispatch` is enabled so a
maintainer can populate or rebuild the baseline. Benchmark IDs should only be
renamed deliberately because CodSpeed uses them to associate historical
measurements.
