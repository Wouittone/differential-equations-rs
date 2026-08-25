# Compile-time tableau resources

Explicit Runge--Kutta methods can be defined in TOML and compiled into the
same static kernel used by built-in methods. The TOML file is read and
validated by a procedural macro while the crate is compiled. A solve performs
no parsing, allocation, file I/O, or dynamic dispatch because the macro emits
ordinary static `ButcherTableau` arrays and a zero-sized algorithm type.

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
and RK4—are resource-backed under `tableaux/explicit`.
`scripts/generate_coefficients.ps1` hashes every resource into
`docs/coefficients_manifest.txt`; run it after adding or changing a
repository-owned resource, and use `-Check` in CI. Remaining large generated
coefficient banks can be migrated family by family without changing their
solver APIs.
