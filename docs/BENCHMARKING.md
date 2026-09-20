# Benchmarks

All benchmark sources live under `benches/`.

## Regression suite

`solver_performance` tracks representative explicit, stiff, automatic,
dense-output, and sequential/parallel ensemble paths with stable
Criterion-compatible IDs:

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

The automatic-switch regression group uses scalar problems. Its fixed-step
cases keep matched work sequences so adaptive-controller divergence does not
obscure dispatch and handoff costs; separate adaptive cases cover the real
controller path. Its stable IDs are:

- `automatic_switching/no_switch/explicit_tsit5`: the explicit baseline;
- `automatic_switching/no_switch/auto_tsit5`: the same Tsit5 work plus the
  detector and automatic dispatch, configured never to switch;
- `automatic_switching/one_switch/auto_tsit5_rodas5p`: one deterministic
  accepted-state handoff from Tsit5 to Rodas5P; and
- `automatic_switching/one_switch/stiff_rodas5p`: the stiff-only baseline on
  the same problem and step sequence;
- `automatic_switching/adaptive_no_switch/explicit_tsit5` and
  `automatic_switching/adaptive_no_switch/auto_tsit5`: matched adaptive
  explicit and detector paths; and
- `automatic_switching/two_switch/auto_tsit5_rodas5p`: a complete
  explicit-to-stiff-to-explicit cycle with both caches re-entered.

Each automatic workload is probed before measurement and fails immediately if
it no longer performs the expected number of switches. Run only this group with:

```console
cargo bench --locked --bench solver_performance -- automatic_switching
```

The `hybrid_workspace/tsit5da_matrix/{128,256,1024}` cases measure short
Tsit5DA ODE solves with increasing matrix-state sizes, including workspace
construction but excluding problem construction and initial tableau parsing.
This specialization uses explicit stages and should not allocate dense
Jacobian or factorization buffers. Run these cases alone with:

```console
cargo bench --locked --bench solver_performance -- hybrid_workspace
```

## Exact RKV98 stage-kernel diagnostics

`rkv98` imports the exact 16-core/5-lazy-dense-stage Numeris fixture used by
the host-coefficient tests. It compares a reusable `ExplicitRungeKuttaStepper`
loop with the reusable adaptive driver on the same 64 accepted intervals and
prints attempts, RHS evaluations, accepted/rejected steps, core stage count,
and lazy dense-stage count before timing either path:

```console
cargo bench --locked --bench rkv98
```

The diagnostic intentionally keeps dense-stage evaluations at zero: the
output-free stepping API exposes core stage derivatives but does not evaluate
the five optional lazy interpolation stages. Their count is reported to make
that limitation explicit rather than attributing interpolation work to the
stage kernel. This isolates stage-kernel time from controller/driver overhead
without registering an unvalidated production RKV98 solver or changing
tolerances and trajectory semantics.

The `rosenbrock_resource_allocations` integration test also checks allocated
bytes scale linearly when vector and non-square matrix states double in size,
with fixed/adaptive stepping and dense output enabled/disabled. It checks the
same decay ODE with scalar states and verifies numerical results and shapes.
This deterministic allocation check complements the timing benchmarks:

```console
cargo test --locked --test rosenbrock_resource_allocations
```

### Reusable downstream lifecycle

Downstream adapters should construct the tableau, stepper, controller, and
caller-owned state once, outside the host's repeated solve loop. Reuse the
stepper with `reset` and the controller with `reset`, then drive each arc with
`integrate_rk`; keep requested outputs in caller-owned storage. This separates
solver-intrinsic stepping from setup and avoids rebuilding scratch workspaces.

The reusable lifecycle timing lane reports those costs separately:

```console
cargo bench --locked --bench reusable_lifecycle
```

Its allocation regression runs the same reusable path over short and long
arcs, after warm-up, and requires zero allocation calls and bytes in both
steady-state measurements. A longer arc therefore cannot hide per-step
workspace growth:

```console
cargo test --locked --test reusable_lifecycle_allocations
```

## Matched Rust/Julia matrix

The repository checkout contains the matched 31-algorithm sources in
`benches/comparison`; the Julia runner and verifier are repository-only because
they also require the pinned submodule and `tests/julia` environment. The crate
archive retains the Cargo benchmark target but not that repository
infrastructure. Run timing and allocation measurements separately so allocation
instrumentation cannot skew the timing lane:

```console
cargo bench --locked --bench comparison_matrix -- --repetitions 20 > rust-timing.csv
julia --startup-file=no --project=tests/julia benches/comparison/julia_matrix.jl --repetitions 20 --mode timing > julia-timing.csv
julia --startup-file=no benches/comparison/verify_matrix.jl --rust rust-timing.csv --julia julia-timing.csv

cargo bench --locked --features allocation-metrics --bench comparison_matrix -- --repetitions 20 > rust-allocation.csv
julia --startup-file=no --project=tests/julia benches/comparison/julia_matrix.jl --repetitions 20 --mode allocation > julia-allocation.csv
julia --startup-file=no benches/comparison/verify_matrix.jl --rust rust-allocation.csv --julia julia-allocation.csv
```

The verifier requires the complete 31-algorithm matrix, exact matched
dimensions, finite positive timing and right-hand-side work measurements, and
valid allocation measurements when present. Endpoint checksums must agree with
relative tolerance `2e-7` or absolute tolerance `5e-8`; these defaults can be
overridden with `--checksum-rtol` and `--checksum-atol`. The absolute threshold
covers the stiff cases whose expected endpoints are close to zero, while the
relative threshold covers order-one states.

Timing and right-hand-side ratios are printed for investigation but are not CI
failure thresholds. Shared-runner noise makes cross-language speed gates
unstable, and the implementations account for finite-difference and Jacobian
work differently. CodSpeed remains the regression gate for Rust timing. CI
runs this matched-matrix certification with one repetition and one Julia
thread as a correctness smoke test; release comparisons should use the 20
repetitions shown above on an otherwise idle machine.

Benchmark commands write CSV to standard output. Results are machine- and
revision-specific artifacts and are deliberately not checked into the
repository.
