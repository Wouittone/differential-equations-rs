# Tableau migration coverage

The resource migration covers every fixed canonical Runge--Kutta tableau that
is currently stored by the crate: **56 explicit** and **39 implicit** methods.
Every JSON file below `tableaux/` is referenced by a compile-validating macro;
there are no orphaned tableau resources and no production fixed Butcher
tableaus encoded as Rust coefficient constants.

"Covered" means that `A`, `b`, `c`, embedded error weights, dense-output rows,
lazy dense stages, and supported fitted weights are deserialized by the shared
Serde parser. The macro validates the same source at compile time, embeds it
with `include_str!`, and leaves runtime materialization behind a `LazyLock`.

## Deliberately outside the canonical resource format

The following cases are not fixed `A`/`b`/`c` Butcher tableaus and are therefore
not represented by the explicit/implicit JSON migration:

- `Anas5` has stage coefficients that are analytic functions of `w * dt` and
  `tan(w * dt)`. Its fixed and dynamic formula still lives in the specialized
  kernel. `Frk65`, by contrast, is covered: its fixed tableau and rational
  fitted weights share one typed resource.
- `Prrk22`, `Prrk33`, and `Prrk54` alter their relaxation coefficients from
  `kappa * dt` for each attempted step.
- Every algorithm in `solvers::explicit::low_storage_rk` uses a low-storage
  recurrence (2N, 2C, 3S, alternating 2N, or register-plus-history form), not
  a canonical Butcher matrix. Those recurrence resources remain separate from
  the tableau schema.
- `SspRkMsvs32`/`SSPRKMSVS32` and `SspRkMsvs43`/`SSPRKMSVS43` are multistep
  Shu--Osher methods with accepted-state history.
- `SplitEuler` acts on a split operator and is not a one-tableau RK method.
- `AdaptiveRadau` and `GaussLegendre` generate collocation nodes and integration
  matrices at runtime from the selected stage count. `RadauIIA3`, `RadauIIA5`,
  and `RadauIIA9` currently use the same generator so the FIRK kernel can share
  one collocation and dense-output implementation. They contain no stored Rust
  coefficient arrays, but they are generated rather than deserialized.
- Test-only `ButcherTableau` implementations in `explicit::general` deliberately
  remain in Rust to exercise downstream custom-tableau compatibility and invalid
  input handling. They are not production solvers.

## Partial additive-tableau coverage

The `Ars*`, `Bhr553`, `Cfnlirk3`, `ImexSsp*`, and `KenCarp*` resources encode
the implicit projection used by the current regular-ODE SDIRK compatibility
surface. A future split/additive API must extend the schema to represent the
paired explicit tableau and coupling semantics; the present files must not be
mistaken for complete additive Runge--Kutta pairs.

## Other solver families

The explicit/implicit Butcher schema intentionally does not cover algorithms
owned by these modules:

- `automatic`: solver-selection and fallback composites;
- `exponential`: exponential RK, Lawson, Magnus, and Krylov recurrences;
- `extrapolation`: midpoint/implicit-Euler extrapolation sequences;
- `linear`: linear-operator and splitting methods;
- `multirate`: MRI/GARK coupling data;
- `multistep`: Adams, BDF/QNDF, Nordsieck, IMEX multistep, MEBDF2, and TRBDF2;
- `rosenbrock`: Rosenbrock/W/RODAS and approximate-matrix-factorization data;
- `second_order`: RKN/IRKN/Newmark and symplectic composition coefficients;
- `stabilized`: Chebyshev/stabilized explicit and implicit recurrences;
- `taylor`: Taylor-series and derivative-order configuration.

Those families need their own typed schemas if their non-Butcher coefficient
data is migrated later. Treating it as a canonical RK tableau would erase
method semantics and make validation weaker, not stronger.
