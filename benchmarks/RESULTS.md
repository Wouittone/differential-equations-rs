# Preliminary benchmark results

Measured on 2026-07-30 with:

- AMD Ryzen 7 3800X, 8 cores / 16 threads;
- 32 GiB RAM;
- Rust 1.97.0, release profile;
- Julia 1.12.6;
- five measured solves after one warm-up solve;
- endpoint-only solution storage.

The non-stiff workload is a 128-component independent linear decay system on
`t ∈ [0, 2]`. The implicit workload is an 8-component stiff decay system on
`t ∈ [0, 1]`. Fixed methods use `dt = 0.01`; adaptive methods use
`abstol = reltol = 1e-7`.

Ratios are Rust divided by Julia, so values below one favor Rust.

| Algorithm | Rust µs | Julia µs | Time ratio | Rust KiB | Julia KiB | Allocation ratio |
|---|---:|---:|---:|---:|---:|---:|
| Tsit5 | 18.3 | 19.7 | 0.93 | 12.0 | 23.1 | 0.52 |
| Midpoint | 764.6 | 604.5 | 1.26 | 7.1 | 16.4 | 0.43 |
| Heun | 824.1 | 438.4 | 1.88 | 7.1 | 16.4 | 0.43 |
| Ralston | 870.9 | 581.5 | 1.50 | 7.1 | 16.4 | 0.43 |
| BS3 | 54.0 | 26.2 | 2.06 | 9.1 | 19.9 | 0.46 |
| DP5 | 21.4 | 12.8 | 1.68 | 12.2 | 19.9 | 0.61 |
| Euler | 43.0 | 37.8 | 1.14 | 6.1 | 12.1 | 0.50 |
| RK4 | 243.5 | 66.0 | 3.69 | 9.1 | 16.4 | 0.56 |
| AB3 | 79.2 | 39.8 | 1.99 | 12.2 | 15.4 | 0.79 |
| AB4 | 96.0 | 51.1 | 1.88 | 13.2 | 19.8 | 0.67 |
| AB5 | 112.8 | 67.0 | 1.68 | 14.3 | 19.8 | 0.72 |
| SSPRK22 | 96.6 | 41.6 | 2.32 | 7.1 | 9.9 | 0.72 |
| SSPRK33 | 161.5 | 53.8 | 3.00 | 8.1 | 9.9 | 0.82 |
| SSPRK43 | 68.9 | 29.6 | 2.33 | 9.1 | 16.5 | 0.55 |
| Implicit Euler | 58.3 | 143.2 | 0.41 | 1.2 | 14.1 | 0.09 |
| Implicit Midpoint | 59.5 | 185.6 | 0.32 | 1.2 | 14.1 | 0.09 |
| Trapezoid | 89.3 | 165.5 | 0.54 | 1.2 | 14.5 | 0.08 |

Across the explicit methods, the geometric-mean Rust/Julia ratios are 1.83×
for runtime and 0.57× for allocated bytes. Across the three implicit methods,
they are 0.41× for runtime and 0.086× for allocated bytes.

These results are directional, not publication-quality:

- the Rust allocation instrumentation uses an atomic-counting global
  allocator and therefore adds runtime overhead;
- five repetitions are enough to validate the harness, not to characterize
  variance;
- adaptive controllers take different step sequences, visible in the RHS
  evaluation counts;
- allocated bytes are cumulative allocation traffic, not peak resident memory;
- the initial Rust implicit implementation recomputes a finite-difference
  Jacobian and factorization every step.

The next benchmark iteration must separate timing and allocation
instrumentation, increase sampling, add peak-live and process-RSS measurement,
and compare work-precision curves rather than a single tolerance.
