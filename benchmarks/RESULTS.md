# Preliminary benchmark results

Measured on 2026-07-30 with:

- AMD Ryzen 7 3800X, 8 cores / 16 threads;
- 32 GiB RAM;
- Rust 1.97.0, release profile;
- Julia 1.12.6;
- fifty measured solves after one warm-up solve;
- endpoint-only solution storage.

The non-stiff workload is a 128-component independent linear decay system on
`t ∈ [0, 2]`. The implicit workload is an 8-component stiff decay system on
`t ∈ [0, 1]`. Fixed methods use `dt = 0.01`; adaptive methods use
`abstol = reltol = 1e-7`.

Ratios are Rust divided by Julia, so values below one favor Rust.

| Algorithm | Rust µs | Julia µs | Time ratio | Rust KiB | Julia KiB | Allocation ratio |
|---|---:|---:|---:|---:|---:|---:|
| Tsit5 | 16.3 | 9.0 | 1.82 | 12.0 | 23.1 | 0.52 |
| Midpoint | 860.6 | 436.0 | 1.97 | 7.1 | 16.4 | 0.43 |
| Heun | 854.1 | 486.2 | 1.76 | 7.1 | 16.4 | 0.43 |
| Ralston | 954.8 | 475.9 | 2.01 | 7.1 | 16.4 | 0.43 |
| BS3 | 57.1 | 22.5 | 2.54 | 9.1 | 19.9 | 0.46 |
| DP5 | 26.0 | 6.7 | 3.86 | 12.2 | 19.9 | 0.61 |
| Euler | 41.8 | 40.5 | 1.03 | 6.1 | 12.1 | 0.50 |
| RK4 | 286.2 | 66.1 | 4.33 | 9.1 | 16.4 | 0.56 |
| AB3 | 85.9 | 40.0 | 2.15 | 12.2 | 15.4 | 0.79 |
| AB4 | 118.0 | 62.0 | 1.90 | 13.2 | 19.8 | 0.67 |
| AB5 | 121.6 | 73.9 | 1.65 | 14.3 | 19.8 | 0.72 |
| SSPRK22 | 108.2 | 47.9 | 2.26 | 7.1 | 9.9 | 0.72 |
| SSPRK33 | 173.5 | 53.9 | 3.22 | 8.1 | 9.9 | 0.82 |
| SSPRK43 | 74.2 | 27.6 | 2.69 | 9.1 | 16.5 | 0.55 |
| Implicit Euler | 59.2 | 154.1 | 0.38 | 1.2 | 14.1 | 0.09 |
| Implicit Midpoint | 68.9 | 167.1 | 0.41 | 1.2 | 14.1 | 0.09 |
| Trapezoid | 60.6 | 146.1 | 0.42 | 1.2 | 14.5 | 0.08 |
| Rosenbrock23 | 314.0 | 524.9 | 0.60 | 2.0 | 23.2 | 0.09 |

Across the explicit methods, the geometric-mean Rust/Julia ratios are 2.22×
for runtime and 0.57× for allocated bytes. Across the four implicit methods,
they are 0.45× for runtime and 0.086× for allocated bytes.

These results are directional, not publication-quality:

- the Rust allocation instrumentation uses an atomic-counting global
  allocator and therefore adds runtime overhead;
- a single 50-solve batch is enough to validate the harness, not to
  characterize run-to-run variance;
- adaptive controllers take different step sequences, visible in the RHS
  evaluation counts;
- allocated bytes are cumulative allocation traffic, not peak resident memory;
- the initial Rust implicit implementation recomputes a finite-difference
  Jacobian and factorization every step.

The next benchmark iteration must separate timing and allocation
instrumentation, increase sampling, add peak-live and process-RSS measurement,
and compare work-precision curves rather than a single tolerance.
