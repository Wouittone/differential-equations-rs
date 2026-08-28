# Tableau resources

All built-in compile-time method data lives below `src/tableau/resources` as
JSON. Explicit and implicit Runge--Kutta resources use canonical Butcher
matrices; specialized families use the same resource tree for their typed
method data instead of maintaining a separate coefficient directory or parser.

Resources use a FracturedJson-style layout: object fields have stable ordering,
scalar arrays stay on one line, and every matrix row occupies one line. The
`tableau_resources` integration test validates JSON syntax, rejects obsolete
schema-version fields, and enforces this layout. The Runge--Kutta schema is
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
