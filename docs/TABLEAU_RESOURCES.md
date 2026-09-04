# Tableau resources

Resource-backed method data lives below `src/tableau/resources` as JSON.
Explicit and implicit Runge--Kutta resources use canonical Butcher matrices;
symplectic compositions use paired drift/kick vectors. Specialized families
also use this tree for typed method data, including canonical linear multistep
formulas for fixed-step Adams and MRAB, and per-method Rosenbrock tableaus.
Migration is not yet complete:
legacy embedded coefficients remain in some multistep and exponential
implementations.

Resources use a FracturedJson-style layout: object fields have stable ordering,
scalar arrays stay on one line, and every matrix row occupies one line. This is
an authoring convention, not a build requirement. The `tableau_resources`
integration test validates JSON syntax and rejects obsolete schema-version
fields. The Runge--Kutta schema is
[`schema.json`](../src/tableau/resources/schema.json).

The shared parser uses Serde and `serde_json`. String coefficients may contain
numeric expressions parsed by `exmex`; accepted expressions are limited to
numeric literals, parentheses, `+`, `-`, `*`, `/`, and `sqrt(...)`.
JSON numeric tokens use `serde_json`'s accurate float-roundtrip parsing so
decimal numbers and equivalent decimal strings produce the same `f64` bits.

## MRI-GARK resources

MRI-GARK coupling data lives in independent files under
`src/tableau/resources/mri`. Each resource contains `dc`, the square causal
slow-forcing matrices `W0` and `W1`, the implicit-slow diagonal `gamma`, the
outer and fast-inner orders, and an optional paired `embedded0`/`embedded1`
estimator. The parser requires `dc` to sum to one and rejects mismatched,
non-finite, non-causal, or half-specified embedded data.

Use `tableau::define_mri_tableau_from_file!` to compile-time validate and embed
one resource. It creates a `LazyMriTableau`, so the coefficients are parsed
only when that method is inspected or solved. The six built-in MRI-GARK
algorithms expose `.tableau()` and each owns an independent lazy resource.
The Knoth--Wolke `MIS` method uses the companion `MisTableau` representation
for its `alpha`, `beta`, and `gamma` coupling matrices and its `d`, `c`, and
`c_tilde` vectors. `define_mis_tableau_from_file!` applies the same compile-time
validation and lazy-loading policy. No multirate method coefficients remain in
the Rust solver module.

## Rosenbrock resources

The general Rosenbrock/Rodas methods use independent files under
`src/tableau/resources/rosenbrock`. Their `kind` is `rosenbrock`, with the
fields `name`, `description`, `order`, `gamma`, `A`, `C`, `c`, `d`, `b`,
optional `btilde`, and optional `H`. The representation follows SciML's
[RodasTableau convention](https://github.com/SciML/OrdinaryDiffEq.jl/blob/211142263781255a9aa2f910f6760b9f18ec29c8/lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl).
`A` and `C` are square and strictly lower triangular; unused trailing entries
of upstream rectangular `C` matrices are padded with zeros.

For stage increments `k[i]`, the numerical kernel uses:

```text
(I - h*gamma*J) k[i] = h*gamma*(f(u + sum(A[i,j]*k[j]), t + c[i]*h)
                               + h*d[i]*f_t + sum(C[i,j]*k[j])/h)
u_next = u + sum(b[i]*k[i])
error  = sum(btilde[i]*k[i])
```

Stage sums use only `j < i`. These weights are not ordinary Butcher weights:
in particular, `b` need not sum to one. Omit `btilde` (or use `null`) for a
fixed-step formula; an all-zero embedded estimator is rejected. `H` holds
two through four stiff dense-output correction rows, each with one entry per
stage. Empty or omitted `H` selects the Hermite fallback. With
`theta = (t - t_start)/h`, the correction to linear endpoint interpolation is
`theta*(1-theta)*(H[0]*k + theta*(H[1]*k + ...))`.

Use `tableau::define_rosenbrock_tableau_from_file!` to define a validated lazy
static for a specialized kernel:

```rust,ignore
use differential_equations::tableau::{define_rosenbrock_tableau_from_file, load_tableau};

define_rosenbrock_tableau_from_file!(pub ROS2, "Ros2", "resources/ros2.json");
let tableau = load_tableau(&ROS2)?;
```

This macro defines data, not an `OdeAlgorithm`. It shares the core scalar
expression parser and lazy-static expansion with the other tableau families;
it emits no Rust coefficient arrays. Malformed fields, dimensions, triangular
structure, non-finite expressions, or dense rows fail compilation. These are
structural checks, not a proof of classical order or stability.

The `Rosenbrock23/32` resource uses a dedicated representation of the shared
three-stage, low-storage W-method pair. Its `state`, `derivative`, `stage`, and
`post_solve` matrices describe the native stage equations; the resource also
holds both solution formulas, the direct error weights, and one polynomial row
for each of the two stages used by dense output. `Rosenbrock23`,
`Rosenbrock32`, and `AMF<Rosenbrock23>` all
consume this one lazy resource. Use
`tableau::define_rosenbrock_pair_tableau_from_file!` for this specialized
representation; it validates exactly three causal stages and the pair's
required dimensions at compile time.

Built-in methods expose `.tableau()` for inspection. Solving materializes only
the selected method; repeat access is allocation-free. `Rodas5Pr` shares
`Rodas5P`'s parsed coefficients and adds residual control. Controller policies
are kept separate from classical method order, including the existing
step-doubling estimator for `Ros34Pw1a`. `Scholz4_7`'s resource records its
upstream classical order of three while preserving the existing controller.

### Hybrid explicit/implicit tableaus

`Tsit5DA` and its `HybridExplicitImplicitRK` alias share a single lazy resource
with `kind: "hybrid-explicit-implicit"`. It uses the same fields and parser as
Rosenbrock resources, but `C` contains the raw lower-triangular Gamma matrix
including its common `gamma` diagonal. `RosenbrockTableau::kind()` distinguishes
these conventions; the driver rejects a resource of the wrong kind.
Hybrid solution weights must sum to one and error weights to zero. The same
compile-time checks cover finite values, dimensions, and dense-output rows.

The ordinary ODE specialization evaluates explicit stages
`k[i] = h*f(u + sum(A[i,j]*k[j]), t + c[i]*h)`, without requesting a Jacobian
or performing linear solves. Gamma and `d` are retained as source metadata for
the upstream algebraic specialization; this does **not** add DAE support.
`H` remains in the upstream correction basis, with no duplicate polynomial
coefficient representation or separate dense-output coefficient bank.

The pinned source specifies nodes `c` and time-derivative weights `d`
independently. They are not silently recomputed from row sums. In particular,
zero-based `c[7]` is `0.9999990000000002` while its `A` row sums to `1`;
`d[4]` is `0.2843274226367331` while its Gamma row sums to approximately
`0.2843327428458331`. The Gamma row sum at index 6 also differs from `d[6]`
by approximately `4e-10`. These pinned-source discrepancies are preserved
for reproducibility and need a mathematical audit before stronger order or
DAE claims. The resource validator proves structural validity, not those claims.

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
`fsal` requires a zero first stage row at `c = 0` and a final stage at `c = 1`
whose row equals `b`. This also describes stiffly accurate implicit methods
with an explicit first stage, such as TR-BDF2. Explicit FSAL methods additionally
require the final primary weight to be zero.
`error`, `second_error`, and dense rows use the same stage ordering as `b`.
Alternatively, provide the embedded formula's weights as `b_hat`, together
with `embedded_order`. The parser validates their length, finite values, and
sum of one, then materializes `error = b - b_hat`. Do not provide both `error`
and `b_hat`. Derived errors and coefficient sums are checked for overflow;
`second_error` can accompany either representation.
`embedded_order` describes the companion formula: it must be lower than the
primary order for explicit resources, while implicit resources can use a
higher-order companion (TR-BDF2 uses orders two and three).
`lazy_dense_stages` can add sparse stages used only by continuous output.
Parametric primary weights can use `fitted_weights`, whose numerator and
denominator vectors are stored in ascending powers of the solver's fit
variable.

Implicit tableaux can include `stage_predictors`, one row per ordinary stage.
Row `i` contains initial-guess weights on derivatives from stages `0..i` (not
including stage `i` itself). Rows can omit trailing zero weights; an empty row
selects the numerical driver's default guess. Nonempty rows must sum to one
and can contain negative weights. The shared parser checks finite values,
row count, and causal stage references at compile time and first use.
`tableau.stage_predictor(i)` returns the row, or `None` for a default guess or
an out-of-range index. This is solver metadata, not an extra Butcher matrix;
a driver must explicitly support it to use custom predictors.

If the dependency is renamed, pass its local crate path:

```rust,ignore
define_explicit_rk_from_file!(
    pub FileHeun,
    "resources/file_heun.json",
    crate = diffeq,
);
```

## TR-BDF2

TR-BDF2 is represented as an implicit Runge--Kutta tableau in
[`implicit/trbdf2.json`](../src/tableau/resources/implicit/trbdf2.json), including
its Newton stage predictors. `Trbdf2.tableau()` exposes the same lazily loaded
data used by the solver, including its FSAL derivative reuse. The specialized
kernel retains its smoothed error estimate and stiffly accurate final stage;
it does not generate or maintain a
second set of coefficients.

The resource preserves the direct error weights from
[SciML's TR-BDF2 definition](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqSDIRK/src/sdirk_tableaus.jl):
these use **`b_hat - b`**, unlike the `b - b_hat` convention derived when a
resource supplies `b_hat` directly. Either sign gives the same error norm.
The companion is third order, while the accepted solution remains second
order. The Newton predictors only provide initial guesses; they do not
change the method's Butcher coefficients.

## Linear multistep formulas

Constant-step formulas use coefficients ordered from newest to oldest:

`sum(alpha[j] * y[n+1-j]) = h * sum(beta[j] * f[n+1-j])`.

For example, Adams--Bashforth of order two is:

```json
{
  "name"        : "FileAB2",
  "description" : "Second-order Adams-Bashforth formula.",
  "kind"        : "linear-multistep",
  "order"       : 2,
  "alpha"       : [1, -1, 0],
  "beta"        : [0, "3/2", "-1/2"]
}
```

Define a shared lazy tableau without generating Rust coefficients:

```rust,ignore
use differential_equations::tableau::{define_multistep_tableau_from_file, load_tableau};

define_multistep_tableau_from_file!(pub FORMULA, "FileAB2", "resources/ab2.json");
let tableau = load_tableau(&FORMULA)?;
assert_eq!(tableau.order(), 2);
```

The optional `crate = local_name` argument supports renamed dependencies.
This defines a tableau, not a generic multistep solver: a numerical driver must
support the formula's structure. `LinearMultistepTableau` exposes `alpha()`,
`beta()`, `steps()`, `order()`, and `is_explicit()`. The leading `alpha` coefficient
must be nonzero; arrays must have equal lengths of at least two. Validation
checks polynomial order conditions through the declared order, using normalized
coefficients only for the checks. It preserves the resource's coefficient bits
and does not prove zero-stability or the stability region. The editor schema is
[`multistep-schema.json`](../src/tableau/resources/multistep-schema.json).

Fixed-step Adams methods and MRAB share individual resources under
`src/tableau/resources/multistep`: `Ab3.tableau()` and
`MRAB::new(3, 8).tableau()` refer to the same parsed formula. ABM methods expose
the corrector through `tableau()` and the shared explicit predictor through
`predictor_tableau()`. MRAB loads lower-order startup formulas only as needed;
fixed-step Adams startup reuses the ordinary Ralston or RK4 tableau. The
upstream ABM32/ABM43 repeating-startup predictor behavior remains unchanged.

### Variable-step two-step formulas

`VariableMultistepTableau` represents canonical two-step formulas whose
coefficients depend on `rho = current_step / previous_step`. Each `alpha` and
`beta` entry is either an array of polynomial coefficients in ascending powers
of `rho`, or an object containing numerator and denominator coefficient arrays.
The same representation stores derivative-defect weights and their scale.
A nested fixed `startup` formula covers the first step without a second,
untyped coefficient bank.

`define_variable_multistep_tableau_from_file!` validates the startup and the
declared nonuniform-grid order conditions at compile time, then embeds only the
JSON source for lazy loading. Runtime evaluation rejects nonpositive ratios,
poles, and overflow. `Abdf2.tableau()` exposes the built-in resource; its solver
evaluates the canonical `alpha`, `beta`, and defect formulas directly, without
generated Rust constants. The editor schema is
[`variable-multistep-schema.json`](../src/tableau/resources/variable-multistep-schema.json).

### Backward differentiation and NDF

The same parser and macro accept `"kind": "backward-differentiation"` with
canonical BDF `alpha`/`beta` arrays and an `ndf_kappa` scalar. These resources
describe the **BDF base formula**, with its accompanying NDF modifier, rather
than a second bank of precomputed NDF coefficients. For example:

```json
{
  "name"        : "FileBDF2",
  "description" : "BDF2 base formula with its NDF modifier.",
  "kind"        : "backward-differentiation",
  "order"       : 2,
  "alpha"       : ["3/2", -2, "1/2"],
  "beta"        : [1, 0, 0],
  "ndf_kappa"   : "-1/9"
}
```

BDF drivers use the base arrays unchanged. NDF subtracts
`ndf_kappa * alpha[0] * difference^(order+1)(y)` from the left-hand side,
where `difference` is the backward difference on the constant-step history.
The quasi-constant-step drivers reinterpolate that history when steps change.
Their harmonic weight is the BDF leading coefficient `alpha[0]`, not a
separate coefficient table. The shared error factor is
`ndf_kappa * alpha[0] + 1/(order+1)` (with zero kappa for BDF).

Validation additionally requires exactly `order` steps, `beta = [1, 0, ...]`,
a positive leading coefficient, and positive finite NDF leading/error factors.
The modifier is required for this resource kind and forbidden for ordinary
`linear-multistep` resources. As with other multistep data, these checks do not
prove zero-stability or the stability region.

`Qndf.tableau(order)`, `Qbdf.tableau(order)`, and `Fbdf.tableau(order)` share
the same per-order resource for orders one through five. Fixed-order
`Qndf1`, `Qbdf1`, `Qndf2`, and `Qbdf2` expose `tableau()` without an order
argument. `ndf_kappa()` returns `Some(kappa)` on these base tableaux and `None`
on ordinary Adams resources. Merely inspecting an order does not initialize
the other orders; adaptive QNDF/QBDF also load adjacent orders as needed for
their error comparisons. These resource definitions do not change the
existing startup, history, or order-selection algorithms.

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
