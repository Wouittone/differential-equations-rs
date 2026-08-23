# Matched benchmark results

## 2026-08-23 local stiff-candidate slice (31-algorithm matrix)

The matched Rust/Julia matrix now covers 31 algorithms. The table below records
the seven stiff candidates from a local 20-repetition run after warm-up, using
the matrix's 8-component stiff linear-decay problem, endpoint-only storage, and
`abstol = reltol = 1e-7`. Timing and allocation measurements were collected in
separate runs. Ratios are Rust divided by Julia, so values below one favor Rust.

| Algorithm    | Rust ns | Julia ns | Time ratio | Rust bytes | Julia bytes | Allocation ratio |
| ------------ | ------: | -------: | ---------: | ---------: | ----------: | ---------------: |
| Rosenbrock23 |  277205 |   535705 |      0.517 |       2064 |       23712 |            0.087 |
| TRBDF2       |  390775 |   146210 |      2.673 |       1616 |       11120 |            0.145 |
| Kvaerno5     |  197700 |   214235 |      0.923 |     116744 |       13552 |            8.615 |
| KenCarp5     |  132005 |   171750 |      0.769 |      83592 |       15968 |            5.235 |
| Rodas4P      |  135610 |   242560 |      0.559 |       3088 |       16128 |            0.191 |
| Rodas5P      |   78980 |   145140 |      0.544 |       3088 |       14656 |            0.211 |
| Rodas5Pr     |  242055 |   139460 |      1.736 |       3088 |       14656 |            0.211 |

For the crate's current regular-ODE scope, **Rodas5P is the selected default
stiff solver**. It is the fastest Rust candidate in this matched slice, retains
low allocation traffic, and already backs `DefaultImplicitODEAlgorithm`. This
is a scoped default, not a claim that Rodas5P is universally best.

Limitations of this selection:

- this is one small diagonal stiff problem at one tolerance, not a
  work-precision study;
- the candidates may take different accepted/rejected step sequences;
- the problem does not supply an analytic Jacobian, so this slice exercises
  finite-difference Jacobian paths;
- twenty solves provide a useful local comparison but not robust
  cross-machine statistics;
- allocated bytes are cumulative allocation traffic, not peak live memory or
  process RSS;
- nonlinear, sparse, large-dimensional, event-heavy, and application-specific
  workloads may favor another method.

The next stiff-selection benchmark wave should add work-precision curves,
analytic-versus-finite-difference Jacobian lanes, representative nonlinear
problems such as Robertson and Van der Pol, and process-RSS measurements.

## 2026-07-30 preliminary 25-solver baseline

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

| Algorithm         | Rust µs | Julia µs | Time ratio | Rust KiB | Julia KiB | Allocation ratio |
| ----------------- | ------: | -------: | ---------: | -------: | --------: | ---------------: |
| Tsit5             |    20.4 |      8.9 |       2.29 |     12.0 |      23.1 |             0.52 |
| Midpoint          |   889.5 |    442.0 |       2.01 |      7.1 |      16.4 |             0.43 |
| Heun              |   814.3 |    436.6 |       1.87 |      7.1 |      16.4 |             0.43 |
| Ralston           |   869.8 |    437.4 |       1.99 |      7.1 |      16.4 |             0.43 |
| BS3               |    73.2 |     25.0 |       2.92 |      9.1 |      19.9 |             0.46 |
| DP5               |    24.2 |      7.5 |       3.24 |     12.2 |      19.9 |             0.61 |
| Euler             |    55.4 |     33.5 |       1.65 |      6.1 |      12.1 |             0.50 |
| RK4               |   251.2 |     66.6 |       3.77 |      9.1 |      16.4 |             0.56 |
| RKM               |   471.3 |    115.6 |       4.08 |     11.2 |      20.1 |             0.56 |
| Ralston4          |   257.6 |     66.7 |       3.86 |      9.1 |      15.6 |             0.59 |
| Alshina2          |   107.0 |     69.1 |       1.55 |      7.1 |      14.2 |             0.50 |
| Alshina3          |   180.8 |     57.5 |       3.14 |      8.1 |      15.4 |             0.53 |
| AB3               |    91.0 |     47.7 |       1.91 |     13.2 |      15.4 |             0.86 |
| AB4               |   100.6 |     51.6 |       1.95 |     14.2 |      19.8 |             0.72 |
| AB5               |   118.3 |     52.2 |       2.27 |     15.3 |      19.8 |             0.77 |
| ABM32             |   157.1 |     79.5 |       1.98 |     13.2 |     248.6 |             0.05 |
| ABM43             |   269.1 |    104.3 |       2.58 |     14.2 |      44.7 |             0.32 |
| ABM54             |   260.1 |    104.1 |       2.50 |     15.3 |      45.7 |             0.33 |
| SSPRK22           |   109.6 |     44.2 |       2.48 |      7.1 |       9.9 |             0.72 |
| SSPRK33           |   195.8 |     57.6 |       3.40 |      8.1 |       9.9 |             0.82 |
| SSPRK43           |   114.7 |     34.9 |       3.28 |      9.1 |      16.5 |             0.55 |
| Implicit Euler    |   103.7 |    135.0 |       0.77 |      1.2 |      14.1 |             0.09 |
| Implicit Midpoint |    62.1 |    177.7 |       0.35 |      1.2 |      14.1 |             0.09 |
| Trapezoid         |    64.8 |    150.2 |       0.43 |      1.2 |      14.5 |             0.08 |
| Rosenbrock23      |   306.0 |    540.9 |       0.57 |      2.0 |      23.2 |             0.09 |

Across the explicit methods, the geometric-mean Rust/Julia ratios are 2.50×
for runtime and 0.48× for allocated bytes. Across the four implicit methods,
they are 0.51× for runtime and 0.086× for allocated bytes.

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
