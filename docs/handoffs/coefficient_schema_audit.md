# Phase 4 coefficient-schema and code-generation audit

## Scope and provenance

This audit is against `OrdinaryDiffEq.jl` commit
`211142263781255a9aa2f910f6760b9f18ec29c8`, checked out at
`D:/Source/_review/OrdinaryDiffEq.jl`. The resolver below checked that every
path cited in this report exists in that checkout and that the cited line range
is within the file. The audit covers native regular ODE method data only. SDE,
DDE, BVP, PDE, steady-state, DAE-only residual, and external-wrapper records
are intentionally not included.

The upstream implementation has two representations that Phase 4 must keep
separate: immutable coefficient data (a tableau/recurrence) and mutable solve
state (cache/history/work vectors). Coefficients must be generated at compile
time; YAML/JSON/TOML (or any other declarative source) must never be parsed on
the solve path.

## Upstream source map (all paths resolved)

| family/data | pinned upstream source and lines | observations that constrain the schema |
| --- | --- | --- |
| Common explicit/implicit tableau metadata | `lib/DiffEqBase/src/tableaus.jl:6-38,41-58` | `ExplicitRKTableau` stores `A,c,alpha,b,d,stages,order,adaptiveorder,fsal,stability_size,B_interp`; `ImplicitRKTableau` has the analogous `A,c,alpha,b,stages,order,adaptiveorder`. Empty `d` and optional `B_interp` are meaningful values, not absent records. |
| Classic/low-order explicit coefficients | `lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_low_order.jl:4-219`; `lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_classic.jl:1-366`; `lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl:1-823`; `lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl:1-2023`; `lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl:1-1087`; `lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order8_9.jl:1-2073` | Constructors pass dense rectangular `A` plus `c`, high/low weights and optional dense data. The generated schema must preserve exact decimal/rational values and triangular zeroes instead of inferring them from runtime matrices. |
| Generic explicit cache and stage execution | `lib/OrdinaryDiffEqExplicitRK/src/explicit_rk_caches.jl:1-78`; `lib/OrdinaryDiffEqExplicitRK/src/explicit_rk_perform_step.jl:1-106,166-316` | Runtime cache stores stage vectors and optional interpolation matrix. FSAL initialization and endpoint derivative retention are observable lifecycle behavior. |
| Generic RK dense records | `lib/OrdinaryDiffEqExplicitRK/src/explicit_rk_perform_step.jl:317-471`; `lib/OrdinaryDiffEqExplicitRK/src/interpolants.jl:1-129`; `lib/OrdinaryDiffEqExplicitRK/src/algorithms.jl:47-90,130-139,178-186,241-249` | `B_interp` is a stage-by-polynomial coefficient matrix. Its rows are evaluated as polynomials in `Theta`; missing `B_interp` falls back to Hermite. Tsit5/DP5-style records must not be reduced to endpoint values. |
| Owren-Zennaro, BS5 and lazy dense stages | `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:63-512,799-971,1039-1205`; `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:138-724` | OwrenZen3/4/5 and BS5 have explicit, embedded and interpolation weights. BS5 stores extra stages only when dense output is requested (the source comments at lines 799-847 are an explicit lazy-stage contract). |
| Verner family | `lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl:2-4408`; `lib/OrdinaryDiffEqVerner/src/verner_caches.jl:1-353`; `lib/OrdinaryDiffEqVerner/src/verner_rk_perform_step.jl:1-1628`; `lib/OrdinaryDiffEqVerner/src/interp_func.jl:1-55` | Vern6/7/8/9 records contain large named coefficient sets, extra dense-only stages, and dedicated interpolation coefficient structs (`Vern6InterpolationCoefficients` at 200-370, then analogous Vern7/8/9 blocks). The `lazy` cache flag controls whether extra stages are evaluated. |
| SSP/Shu-Osher | `lib/OrdinaryDiffEqSSPRK/src/ssprk_caches.jl:1-1425`; `lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl:1-1723`; `lib/OrdinaryDiffEqSSPRK/src/interp_func.jl:1-14` | SSPRK22/33 use compact hard-coded recurrences; higher SSP methods have named scalar coefficients and several 2N/H forms. The schema needs a tagged Shu-Osher/SSP recurrence, not a generic Butcher matrix. Dense mode is explicitly “2nd order free SSP interpolation”; otherwise linear. |
| Low-storage RK | `lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_caches.jl:1-3369`; `lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_perform_step.jl:1-1081` | Families include Williamson 2N, 2C, 3S/3Sp/3SpFSAL and 2R/3R/4R/5R. Coefficients are stored as tuples/scalars in family-specific constant caches; recurrence shape and register count are part of the method identity. |
| Fixed/variable Adams and ABM | `lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_bashforth_moulton_caches.jl:1-1098`; `lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_bashforth_moulton_perform_step.jl:1-1562`; `lib/OrdinaryDiffEqAdamsBashforthMoulton/src/algorithms.jl:1-186` | Fixed AB/ABM records carry order-specific history weights. VCAB/VCABM records carry order, unequal-step `dts`, divided-difference/phi buffers and RK starter metadata; their unequal-step coefficients are derived at runtime from history and must not be emitted as a fixed tableau. |
| Rosenbrock/Rodas coefficients | `lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl:1-886`; `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl:1-342` | `RodasTableau` (line 11) groups A/C/gamma/c/d/H/b/bhat and stage/order metadata. H may be empty for methods that use generic Hermite. The local file additionally defines Rosenbrock23/32 and Tsit5DA with explicit A, linear-implicit C, gamma, b/bhat, c, d and H (291-368). |
| Rosenbrock caches, stages and dense dispatch | `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_caches.jl:51-178`; `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl:1-1018`; `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_interpolants.jl:1-213,545-697` | Jacobian/factorization reuse state is mutable and separate from tableau data. Dense dispatch selects H-polynomial paths or a generic Hermite path; lines 180-187 document the endpoint derivative convention and the residual/DAE caveat that is out of regular-ODE scope. |
| Symplectic composition | `lib/OrdinaryDiffEqSymplecticRK/src/symplectic_tableaus.jl:1-563`; `lib/OrdinaryDiffEqSymplecticRK/src/symplectic_caches.jl:1-182`; `lib/OrdinaryDiffEqSymplecticRK/src/symplectic_perform_step.jl:1-366` | `SymplecticTableau` is a pair of composition vectors `a,b`; constructors provide PseudoVerlet, McAte, Ruth, CandyRoz, CalvoSanz, Yoshida, KahanLi and SofSpa data. The Hamiltonian state/cache is distinct from first-order ODE state. |
| Partitioned and quadratic partitioned RK | `lib/OrdinaryDiffEqPRK/src/prk_caches.jl:1-92`; `lib/OrdinaryDiffEqPRK/src/prk_perform_step.jl:1-107`; `lib/OrdinaryDiffEqQPRK/src/qprk_tableaus.jl:1-235`; `lib/OrdinaryDiffEqQPRK/src/qprk_caches.jl:1-76`; `lib/OrdinaryDiffEqQPRK/src/qprk_perform_step.jl:1-238` | KuttaPRK2p5 uses a dedicated coefficient cache with `c2..c5_6`; QPRK98 has a named tableau with `d`, `w`, `b` entries and a 16-stage cache. These must be tagged partitioned records, not flattened explicit RK. |

## Proposed declarative schema

Use a versioned, canonical source (Rust-side schema definitions plus a small
generator input; no runtime parser) with one tagged record per method:

```text
MethodRecord {
  name, family, order, embedded_order: Option<u16>, fsal: bool,
  stage_times: Vec<Scalar>, coefficients: FamilyCoefficients,
  dense: Option<DenseRecord>, provenance: Provenance, caveats: Vec<String>
}
```

`Scalar` is an explicit tagged value: `Rational(num,den)`, canonical decimal
string, symbolic constant (a closed allow-list such as `sqrt(2)`), or exact
IEEE hexadecimal float. The generator emits typed `const` arrays/tuples with
the smallest stable shape needed by each family. Do not round-trip through
binary `f64` while generating.

Family payloads:

* `ExplicitButcher { a_lower: rows, b, b_embedded, stability_size }`, with an
  optional `dense` payload. `a_lower` is emitted as row tuples so strict
  lower-triangular structure is visible at compile time.
* `Rosenbrock { a, c_matrix, gamma, c, d, b, b_embedded, h_matrix }`, retaining
  an explicitly empty H and the method's stage count.
* `ShuOsher { alpha, beta, stages, positivity_domain }`; compact SSP/2N forms
  use a recurrence variant with fixed register count and named scalars.
* `LowStorage { variant, a, b, c, register_count, fsal }`, where `variant`
  distinguishes Williamson2N, 2C, 3S/3Sp/3SpFSAL and 2R/3R/4R/5R.
* `Multistep { kind, order, history_weights, corrector_weights,
  starter: Option<MethodRef>, variable_step: bool }`. VCAB/VCABM keep
  variable-step coefficient construction as a runtime history operation.
* `Symplectic { a, b, composition_order }`, `Partitioned { aq, ap, bq, bp,
  stage_times }`, and `QuadraticPartitioned { named_entries }`.

`DenseRecord` has three explicit variants: `GenericHermite { order: 3 }`,
`FreeStagePolynomial { b_interp: rows, polynomial_order }`, and
`LazyExtraStagePolynomial { base_stage_count, extra_stages, coefficients,
polynomial_order }`. Rosenbrock H matrices are represented by
`FreeStagePolynomial` with a method tag and separate derivative semantics;
Verner interpolation structs map to `LazyExtraStagePolynomial`; SSP's free
interpolant maps to a dedicated `FreeStagePolynomial` variant. Adams dense
history is deferred until the dense-output phase defines a segment interface.

`Provenance` always contains upstream package, normalized path, pinned commit,
and source line. `caveats` records empty H, lazy stages, FSAL, method-specific
controller gains, and unsupported DAE residual behavior.

## Generator validation gate

Before emitting Rust, validate deterministically:

1. Method names are unique; family tags and provenance paths are non-empty;
   every scalar is finite after materialization and every symbolic constant is
   from the allow-list.
2. Stage/time/weight dimensions agree. Explicit `A` is strictly lower
   triangular. Embedded vectors have the same stage count. FSAL requires the
   final node at one and the final-stage row to equal the primary weights for
   all preceding stages (within exact scalar equality).
3. Rosenbrock matrices are square and stage-sized; `gamma` is present and
   nonzero; `d`, `b` and embedded vectors have matching stage length; H is
   either empty or rectangular with one column per stage.
4. Shu-Osher rows sum to one and positivity claims have non-negative alpha and
   beta entries. Low-storage tuple lengths/register counts agree with the
   selected recurrence variant and FSAL is only allowed for variants with an
   endpoint derivative register.
5. Fixed multistep weights match order/history length. Variable-step records
   declare runtime coefficient construction and a valid starter method.
6. Symplectic composition satisfies
   `b_i*a_ij + b_j*a_ji == b_i*b_j` for every pair represented by a full RK
   tableau; partitioned records satisfy the corresponding cross-condition.
7. Dense records have polynomial rows with consistent degree, endpoint value
   and derivative constraints, and lazy-stage references point to declared
   extra stages. Empty dense data is distinct from a zero matrix.
8. Records are sorted by `(family,name)`, numeric spellings are canonical,
   generated Rust is `rustfmt`-stable, and a second generation is byte-identical.

## Migration order and bounded work units

1. Land schema/generator and validation tests without changing solver runtime;
   generate a manifest for the already-tested generic explicit tableaus and
   compare all coefficient bytes against the pinned constructors.
2. Migrate classic/low-order/high-order explicit and embedded records, then
   generic dense `B_interp`/Hermite metadata. Keep lazy extra stages disabled
   unless dense output is requested.
3. Migrate Verner and Owren/BS5, including their interpolation coefficient
   records and lazy-stage flags; add byte-level generated-vs-upstream fixtures.
4. Migrate Rosenbrock/Rodas and Tsit5DA, preserving empty-H fallback and
   separate Jacobian/cache state. Add H-matrix endpoint and derivative tests.
5. Migrate SSP/Shu-Osher and all low-storage recurrences; validate positivity,
   register counts, and FSAL metadata before changing kernels.
6. Migrate fixed Adams/ABM, then variable Adams metadata (runtime unequal-step
   construction remains in the kernel). Add starter/history shape tests.
7. Migrate symplectic and partitioned/QPRK records as separate representations;
   do not route Hamiltonian or partitioned callbacks through first-order RK
   records.
8. Only after all generated records are byte-stable should family kernels be
   switched from hand-written constants. Every wave reruns the inventory and
   the repository gates; dense/controller parity remains a later phase.

## Read-only resolver check

The report was checked with a PowerShell resolver that extracted every
backticked `lib/...:line-range` citation, verified the file under
`D:/Source/_review/OrdinaryDiffEq.jl`, and checked both range endpoints against
the file's line count. `git diff --check` is clean in this worktree.
