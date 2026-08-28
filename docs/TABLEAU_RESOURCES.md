# Tableau resources

Runge--Kutta methods are stored as JSON resources containing the canonical
Butcher matrix `A`, weights `b`, and nodes `c`. JSON is used because its array
and object model maps directly to vectors, matrices, sparse dense-output
stages, and future typed extensions. Exact coefficients remain readable as
strings such as `"-2197/4104"` or `"(3 - sqrt(3)) / 6"`.

The machine-readable contract is [`tableau.schema.json`](tableau.schema.json).
Resource files use a FracturedJson-style layout: stable object ordering,
single-line vectors, and one compact line per matrix row. Long rows intentionally
remain wide because these are machine-oriented resource files.

The shared parser uses Serde/`serde_json` for the document model and maintained
`exmex` for exact-style arithmetic expressions. Plain decimal and scientific
notation literals are parsed directly as `f64`; expression strings are limited
to numeric literals, parentheses, `+`, `-`, `*`, `/`, and `sqrt(...)`. Variables,
constants such as `pi`, and other functions are rejected.

## Defining a method

```json
{
  "name": "FileHeun",
  "description": "An adaptive Heun method defined by a resource.",
  "kind": "explicit-runge-kutta",
  "order": 2,
  "embedded_order": 1,
  "A": [[0, 0], [1, 0]],
  "b": ["1/2", "1/2"],
  "c": [0, 1],
  "error": ["-1/2", "1/2"]
}
```

The resource has no schema-version field. The crate is pre-1.0, so format
changes are handled directly instead of preserving artificial version layers.

Define the solver with a package-relative path:

```rust
use differential_equations::{
    OdeProblem, SolveOptions, define_explicit_rk_from_file, solve,
};

define_explicit_rk_from_file!(pub FileHeun, "tableaux/file_heun.json");

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let problem = OdeProblem::new(
    |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
    vec![1.0],
    (0.0, 1.0),
    (),
);
let solution = solve(&problem, FileHeun, &SolveOptions::default())?;
# Ok(())
# }
```

The procedural macro reads and validates the JSON during compilation using
the same parser as the runtime loader. Invalid JSON, unknown fields, name or
dimension mismatches, non-finite expressions, non-triangular explicit
matrices, invalid primary weights, malformed estimators or dense extensions,
and inconsistent FSAL metadata therefore fail the build with a resource-path
diagnostic.

The expansion does not generate coefficient arrays. It embeds the original
text with `include_str!` and defines a `LazyLock<Result<...>>`; only a method
that is actually used is parsed and allocated at runtime. Runtime parsing
errors remain typed values rather than panics. `FileHeun.tableau()` exposes the
validated materialized tableau for inspection.

`A` must be a full square matrix. An explicit method must be strictly lower
triangular. `error`, `second_error`, and every dense row use the same stage
ordering as `b`. `lazy_dense_stages` may append sparse stages used only by a
continuous extension. The parser accepts JSON numbers and string expressions
using parentheses, `+`, `-`, `*`, `/`, and `sqrt(...)`.

Parametric methods whose primary weights are rational polynomials can add
`fitted_weights`. Each entry names a stage and gives numerator and denominator
coefficient vectors in ascending powers of the solver-defined fit variable.
Validation requires unique in-range stages, finite coefficients, a non-zero
denominator at zero, and agreement with the corresponding zero-fit `b` value.
FRK65 uses this extension so its fixed tableau and runtime-fitted weights remain
one resource rather than separate coefficient primitives.

See [`TABLEAU_MIGRATION_COVERAGE.md`](TABLEAU_MIGRATION_COVERAGE.md) for the
exact boundary between canonical resource-backed tableaus, runtime-generated or
parametric methods, and non-Butcher solver families.

If the dependency is renamed, pass its local path:

```rust,ignore
define_explicit_rk_from_file!(
    pub FileHeun,
    "tableaux/file_heun.json",
    crate = diffeq,
);
```

## Packaging

Resource files are compile inputs and must be included in published packages:

```toml
[package]
include = [
    "/src/**/*.rs",
    "/tableaux/**/*.json",
    "/README.md",
]
```

Run `cargo package --list` and `cargo publish --dry-run` before release. Cargo
tracks each macro's `include_str!`, so editing a resource invalidates the
corresponding build automatically.
