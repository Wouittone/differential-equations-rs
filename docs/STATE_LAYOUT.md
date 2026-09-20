# State storage and scalar scope

`MatrixView` borrows `&[f64]`; dynamic Vec buffers, fixed `[f64; N]` arrays,
and host matrix slices use the same allocation-free view. `flatten_into`
accepts a reusable mutable slice and validates its length before writes.
Row-major indexing is `r * columns + c`; nalgebra/numeris column-major indexing
is `c * rows + r`. Never reinterpret a nonsymmetric matrix slice without an
explicit order. Padded views and transposes retain strides; `to_rows` supports
fixed rectangular arrays.

The standalone `benches/layout.rs` measures a 6x7 sensitivity matrix conversion
with black-box input/output and counts global allocator calls. On the development
Windows host (rustc optimized build), 1,000,000 conversions took 11,283,200 ns
and made zero allocations. This is a smoke measurement, not a stable speedup or
cross-host benchmark. Repeat it with `rustc -O --edition 2024 benches/layout.rs`.

General scalar parameterization is deferred, rather than silently narrowing a
claimed generic interface. At this baseline 143 source files mention f64; the
public tableau coefficients, state math, dense polynomials, Jacobian/factorization
paths and tolerances require coordinated changes for a second scalar type.
The borrowed layout implementation adds no generic scalar monomorphizations and
has no third-party matrix dependency. Its only generic operations use const
array lengths. The zero-allocation conversion smoke test gives no evidence that
scalar genericity would improve the observed bridge regressions; those concern
workspace reuse and layout copies. A pervasive generic rewrite would need
f32/extended-precision accuracy oracles, independent compile-time/binary-size
measurements, and separate compatibility review. No such benefit is established
by the current f64 downstream integrations. The supported scope is therefore
f64 with borrowed/fixed/reusable storage; this decision is not a claim that a
generic-scalar prototype or comparison has been completed.
