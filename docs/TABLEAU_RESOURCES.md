# Tableau resources

Resource-backed method data lives below `src/tableau/resources` as JSON.
Explicit and implicit Runge--Kutta resources use canonical Butcher matrices;
symplectic compositions use paired drift/kick vectors. Specialized families
also use this tree for typed method data. Migration is not yet complete:
legacy embedded coefficients remain in some multirate, multistep,
exponential, and Rosenbrock implementations.

Resources use a FracturedJson-style layout: object fields have stable ordering,
scalar arrays stay on one line, and every matrix row occupies one line. This is
an authoring convention, not a build requirement. The `tableau_resources`
integration test validates JSON syntax and rejects obsolete schema-version
fields. The Runge--Kutta schema is
[`schema.json`](../src/tableau/resources/schema.json).

The shared parser uses Serde and `serde_json`. String coefficients may contain
numeric expressions parsed by `exmex`; accepted expressions are limited to
numeric literals, parentheses, `+`, `-`, `*`, `/`, and `sqrt(...)`.

## Defining a downstream method

```json
{
  "name"           : "FileHeun",
  "description"    : "An adaptive Heun method defined by a resource.",
  "kind"           : "explicit-runge-kutta",
  "order"          : 2,
  "embedded_order" : 1,
  "A"              : [
    [0, 0],
    [1, 0]
  ],
  "b"              : ["1/2", "1/2"],
  "c"              : [0, 1],
  "error"          : ["-1/2", "1/2"]
}
```

The format has no schema-version field while the crate is pre-1.0. Define the
solver with a path relative to the downstream package manifest:

```rust
use differential_equations::{
    OdeProblem, SolveOptions, define_explicit_rk_from_file, solve,
};

define_explicit_rk_from_file!(pub FileHeun, "resources/file_heun.json");

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let problem = OdeProblem::new(
    |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
    [1.0],
    (0.0, 1.0),
    (),
);
let solution = solve(&problem, FileHeun, &SolveOptions::default())?;
assert!(solution.last_state()[0].is_finite());
# Ok(())
# }
```

The procedural macro validates the document during compilation. Invalid JSON,
unknown fields, name or dimension mismatches, non-finite expressions,
non-triangular explicit matrices, malformed estimators or dense extensions,
and inconsistent FSAL metadata fail the build with a resource-path diagnostic.

The expansion embeds the original text with `include_str!` and materializes the
tableau behind a `LazyLock` only when that method is first used. Runtime parsing
errors are typed values rather than panics. `FileHeun.tableau()` exposes the
materialized tableau for inspection.

`A` must be square, and an explicit method must be strictly lower triangular.
`error`, `second_error`, and dense rows use the same stage ordering as `b`.
Alternatively, provide the embedded formula's weights as `b_hat`, together
with `embedded_order`. The parser validates their length, finite values, and
sum of one, then materializes `error = b - b_hat`. Do not provide both `error`
and `b_hat`. Derived errors and coefficient sums are checked for overflow;
`second_error` can accompany either representation.
`lazy_dense_stages` can add sparse stages used only by continuous output.
Parametric primary weights can use `fitted_weights`, whose numerator and
denominator vectors are stored in ascending powers of the solver's fit
variable.

If the dependency is renamed, pass its local crate path:

```rust,ignore
define_explicit_rk_from_file!(
    pub FileHeun,
    "resources/file_heun.json",
    crate = diffeq,
);
```

## Interaction-picture methods

RKIP uses the canonical explicit Runge--Kutta resource
[`explicit/rkip.json`](../src/tableau/resources/explicit/rkip.json), including
the Verner pair's primary and embedded weights. No separate interaction-picture
coefficient parser or Rust coefficient arrays are needed. `algorithm.tableau()`
returns the shared, lazily parsed tableau; constructing an RKIP algorithm does
not materialize it. Cache slots and repeated-node sharing use the resource's
stage nodes rather than a second coefficient list.

The exponential cache can snap proposed steps to its configured grid. After
rejection, however, the controller's smaller step takes precedence, even below
the lower cache bound. This prevents rounding a retry back to the same rejected
step indefinitely. Cache clamping is a reuse preference, not an accuracy limit.

## Symplectic compositions

Every built-in named drift/kick composition uses its own lazily parsed JSON
file under `src/tableau/resources/symplectic`. The representation follows the
upstream `SymplecticTableau`: each stage first drifts the position by `b[i]`,
then kicks the velocity by `a[i]`. Negative coefficients are allowed. The
parser shares the same numeric-expression machinery as Runge--Kutta resources.

```json
{
  "name"        : "FileDriftKick",
  "description" : "Second-order drift/kick composition.",
  "kind"        : "symplectic-composition",
  "order"       : 2,
  "a"           : [1, 0],
  "b"           : ["1/2", "1/2"]
}
```

Define a downstream method with one call:

```rust,ignore
use differential_equations::tableau::define_symplectic_from_file;

define_symplectic_from_file!(pub FileDriftKick, "resources/file_drift_kick.json");
```

The result implements `SymplecticAlgorithm` and works with `solve_symplectic`.
The optional `crate = local_name` argument supports renamed dependencies.
Compile-time validation rejects unknown fields, name mismatches, invalid
expressions, non-finite coefficients, unequal or empty stage counts, and sums
of `a` or `b` inconsistent with one. Higher-order conditions are not proved by
this structural validation. The editor schema is
[`symplectic-schema.json`](../src/tableau/resources/symplectic-schema.json).

`Method::tableau()` now returns `Result<&'static SymplecticTableau, TableauError>`.
Use `a()` and `b()` instead of public fields; metadata includes `name()`,
`description()`, `order()`, and `stages()`. Raw `SymplecticTableau::new` has
been replaced by `parse_symplectic_tableau`, so solver inputs are validated.
Each method caches its own result and does not parse other methods' files.
The expansion embeds source text, never Rust coefficient arrays.

## Packaging

Resource files are compile inputs and must be included in a published package:

```toml
[package]
include = [
    "/src/**/*.rs",
    "/src/tableau/resources/**/*.json",
    "/README.md",
]
```

Use `cargo package --list` to inspect the archive. Cargo tracks every
`include_str!` input, so editing a resource invalidates its dependent build.
