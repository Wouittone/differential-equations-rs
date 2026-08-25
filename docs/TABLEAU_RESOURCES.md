# Compile-time tableau resources

Explicit Runge--Kutta methods can be defined in TOML and compiled into the
same static kernel used by built-in methods. The TOML file is read and
validated by a procedural macro while the crate is compiled. A solve performs
no parsing, allocation, file I/O, or dynamic dispatch because the macro
expands directly to static `ButcherTableau` arrays and a zero-sized algorithm
type. No generated Rust source is checked into the repository.

## Defining a method

Place a resource anywhere under the consuming package. Paths passed to the
macro are relative to that package's `CARGO_MANIFEST_DIR`:

```toml
schema_version = 1
name = "FileHeun"
description = "An adaptive Heun method defined by a TOML resource."
order = 2
embedded_order = 1
fsal = false

nodes = ["0", "1"]
coefficients = [[], ["1"]]
weights = ["1/2", "1/2"]
error_weights = ["-1/2", "1/2"]
```

Then define and use the algorithm:

```rust
use differential_equations::{
    OdeProblem, SolveOptions, define_explicit_rk_from_file, solve,
};

define_explicit_rk_from_file!(pub FileHeun, "tableaux/file_heun.toml");

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

The generated type is a normal `OdeAlgorithm`, so it works with callbacks,
save-at, retained dense output, and Rayon-backed ensemble APIs supported by the
shared explicit driver.

## Schema

Required fields are `schema_version`, `name`, `description`, `order`, `fsal`,
`nodes`, `coefficients`, and `weights`. Optional fields are `embedded_order`,
`error_estimator`, `error_weights`, `second_error_weights`, and
`dense_coefficients`.

Coefficient values may be finite TOML integers/floats or strings containing a
decimal or exact integer ratio such as `"-2197/4104"`. String ratios preserve
the human-auditable source representation; the macro materializes the final
round-trip-safe `f64` literals in generated code.

Compilation fails with a targeted diagnostic when:

- the schema contains unknown fields or an unsupported version;
- a name, order, embedded estimator, or dimension is inconsistent;
- an explicit row is not strictly lower triangular or does not sum to its node;
- primary weights do not sum to one or error weights do not sum to zero;
- coefficients are invalid or non-finite;
- FSAL metadata does not match the final node, row, and weight;
- dense coefficient rows are missing or malformed.

`error_weights` defaults to an embedded-difference estimator and must sum to
zero. Specialized methods whose local error is a direct weighted stage
combination declare `error_estimator = "stage-combination"` explicitly.

Because the expansion contains `include_str!` for the resource, Cargo tracks
the file and recompiles the method when it changes.

## Built-in coefficient banks

Large built-in coefficient banks follow the same resource-first design. Their
typed TOML files live under `coefficients/` and are loaded with
`define_coefficients_from_file!` inside the owning solver family. The schema
supports finite `f64`, `usize`, `i32`, and Boolean scalars; slices; fixed
arrays and matrices; ragged row sets; and lazy dense-output stages. Numeric
strings may use Rust-style `+`, `-`, `*`, and `/` expressions, which keeps
exact ratios auditable.

These resources are the source of truth. There are no generated coefficient
modules to regenerate or synchronize, and compile-time validation rejects
unknown fields, duplicate constant names, invalid types, malformed matrices,
non-finite values, and unsupported expressions.

The generated code uses the normal `differential_equations` crate name by
default. If the dependency is renamed, pass its local path explicitly:

```rust,ignore
define_explicit_rk_from_file!(
    pub FileHeun,
    "tableaux/file_heun.toml",
    crate = diffeq,
);
```

## Repository-owned methods

The canonical low-order built-ins—Euler, midpoint, Heun, Ralston, Alshina2,
and RK4—are resource-backed under `tableaux/explicit`. Family-specific banks
for explicit, second-order, and stabilized methods live under `coefficients/`.
Changing either kind of resource automatically invalidates Cargo's build
because each macro expansion tracks its source with `include_str!`.
