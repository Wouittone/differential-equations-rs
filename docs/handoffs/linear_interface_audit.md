# Phase 3 dense linear-interface audit

## Summary

The current dense `f64` implementation can be migrated without a numerical or
allocation regression by retaining flat `Vec<f64>` storage and partial-pivoted
LU, but making five currently implicit contracts explicit:

1. checked contiguous state and dense row-major views;
2. one reusable analytic-or-finite-difference Jacobian provider;
3. a matrix-free `LinearOperator` seam for analytic/finite-difference Jv;
4. an owning `DenseLu` factorization cache with explicit validity/revision;
5. an updatable, nonsingular mass-operator contract whose first implementations
   are identity and constant dense matrices.

This is a local design recommendation, not a claim that the Rust port should
copy SciML's full operator or `LinearSolve.jl` type hierarchy. No new dependency
is justified for the first backend. The pinned upstream uses concrete matrices,
matrix-free operators, and cached linear solvers behind a substantially more
general abstraction; the useful parity point for this repository is the
separation of responsibilities and cache invalidation, not the size of that
abstraction.

Audit basis:

- local branch base: `0ff7012dbc5739d4e19042ee05d54124932f045b`;
- upstream checkout: `D:/Source/_review/OrdinaryDiffEq.jl`;
- verified upstream `HEAD`: `211142263781255a9aa2f910f6760b9f18ec29c8`;
- in scope: regular initial-value ODE behavior, including nonsingular mass
  matrices;
- excluded: singular mass matrices, DAE residual initialization, sparse/GPU/
  distributed implementations, and external wrappers.

## Current local implementation

### Shared LU

`src/linear.rs:3-40` performs in-place row-major LU with partial pivoting and
stores one pivot row per column. `src/linear.rs:43-67` applies those pivots and
does forward/back substitution in place. It allocates nothing during
factorization or solution.

The hidden preconditions are not checked: matrix length must be `n*n`, pivot
and right-hand-side lengths must be `n`, and all inputs must use the same
dimension. A short slice panics. The singular test is an absolute
`pivot_magnitude <= f64::EPSILON` (`src/linear.rs:18-20`), so the first migration
must preserve that exact rule and record any later scale-aware change as a
separate numerical change with tests.

The factorization and right-hand side have no types distinguishing an
unfactorized matrix, a factorized matrix, and an arbitrary vector. The solve
function cannot report invalid/stale state and all statistics are incremented
manually by callers.

### Jacobian and factorization callers

- `OdeProblem` stores an optional boxed analytic callback at
  `src/problem.rs:11-20`; `with_jacobian` documents a complete row-major
  overwrite at `src/problem.rs:40-50`; `evaluate_jacobian` returns only a
  presence Boolean at `src/problem.rs:126-132`. The public `Vec<f64>` API can
  remain unchanged while an internal provider adapts this callback.
- Implicit Euler/Midpoint/Trapezoid own finite-difference scratch, one combined
  matrix/factor buffer, pivots, and `factorization_scale` in
  `src/implicit.rs:52-80`. `newton_step` decides reuse by exact equality of the
  derivative scale (`src/implicit.rs:216-220`), solves at
  `src/implicit.rs:255-268`, and forces a refresh after one chord-Newton
  correction (`src/implicit.rs:273-275`). `build_factorization` duplicates
  analytic/finite-difference selection, forms `I-scale*J`, records one Jacobian,
  factors, and marks the scale valid at `src/implicit.rs:325-382`. Accepted
  callback effects invalidate it at `src/implicit.rs:172`.
- TRBDF2 owns the same scratch plus a combined matrix/factor buffer and Boolean
  validity at `src/trbdf2.rs:45-80`. Each stage clears validity
  (`src/trbdf2.rs:298-310`), builds and solves inside Newton
  (`src/trbdf2.rs:343-353`), and reuses the final valid factorization for the
  default smoothed error estimate (`src/trbdf2.rs:271-287`). Its duplicated
  Jacobian/assembly/factorization path is `src/trbdf2.rs:367-423`.
- Rosenbrock23 has separate Jacobian and factor buffers but raw pivots and a
  Boolean differentiation flag (`src/rosenbrock.rs:15-53`). It reuses one
  factorization for three stage/error solves (`src/rosenbrock.rs:205-305`) and
  duplicates analytic/finite-difference Jacobian plus time differentiation at
  `src/rosenbrock.rs:308-365`. Accepted steps/callback processing invalidate
  differentiation at `src/rosenbrock.rs:153`.
- Rosenbrock32/Rodas own the analogous buffers at
  `src/rosenbrock_extended.rs:398-432`. Rosenbrock32 reuses one factorization
  for three solves (`src/rosenbrock_extended.rs:562-648`); Rodas reuses it for
  every stage (`src/rosenbrock_extended.rs:668-730`). The shared local
  `prepare_factorization` forms `I-gamma*h*J` and factors it at
  `src/rosenbrock_extended.rs:750-778`; differentiation is duplicated at
  `src/rosenbrock_extended.rs:780-828`; accepted state changes invalidate it at
  `src/rosenbrock_extended.rs:520`.

These are all production factorization/solve call sites found by
`rg "factorize\\(|solve_factorized\\(" src --glob '*.rs'` at the audited base.
The remaining calls are the unit test in `src/linear.rs:73-83`.

## Verified pinned-upstream behavior

The following statements describe upstream behavior at the pinned revision;
they are not local design inference.

### Jacobian selection and statistics

- `calc_J` labels the source as user-provided, autodiff, or finite-difference,
  selects `f.jac` when present and otherwise calls the configured Jacobian
  machinery, then increments `njacs` exactly once
  (`lib/OrdinaryDiffEqDifferentiation/src/derivative_utils.jl:304-354`). The
  in-place form follows the same selection and increments once at
  `derivative_utils.jl:366-417`.
- A matrix-free `JVPCache` stores the prepared Jv action and current `u,p,t`
  (`lib/OrdinaryDiffEqDifferentiation/src/operators.jl:19-26`), exposes square
  dimensions (`operators.jl:44`), applies through `mul!`
  (`operators.jl:56-61`), uses a supplied Jv when available or a prepared
  pushforward otherwise (`operators.jl:65-81`), and explicitly updates its
  point (`operators.jl:84-88`). It advertises that it cannot be concretized
  (`operators.jl:28-34`).

### Concrete matrices versus operators

- `build_J_W` is the allocation/representation decision point
  (`lib/OrdinaryDiffEqDifferentiation/src/derivative_utils.jl:1177-1189`). It
  accepts operator prototypes (`derivative_utils.jl:1204-1220`), uses concrete
  `jac_prototype` storage for a factorization solver
  (`derivative_utils.jl:1221-1233`), selects matrix-free Jv when the chosen
  linear solver does not need a concrete matrix
  (`derivative_utils.jl:1234-1245`), and can retain both a concrete Jacobian and
  Jv/operator path for preconditioning (`derivative_utils.jl:1246-1280`). Dense
  arrays are the ordinary fallback (`derivative_utils.jl:1281-1322`).
- The upstream source itself still carries TODOs to make Jacobians and mass
  matrices uniformly lazy operators (`derivative_utils.jl:1190-1193`). This is
  evidence against reproducing an oversized hierarchy in the Rust port now.

### Iteration-matrix assembly and mass matrices

- For a regular mass-matrix ODE, upstream forms
  `W = M/dtgamma - J`. It checks matrix axes and mass-matrix axes before the
  operation (`derivative_utils.jl:599-613`), special-cases identity/uniform
  scaling (`derivative_utils.jl:614-626`), and has a tight dense loop for an
  explicit matrix mass (`derivative_utils.jl:638-659`).
- `calc_W!` updates an operator's coefficients/gamma or rebuilds a concrete W,
  with the concrete regular-ODE path calling `jacobian2W!`
  (`derivative_utils.jl:804-844`). It records W construction separately from a
  Jacobian evaluation (`derivative_utils.jl:846-858`).
- Rosenbrock stage code applies the mass matrix to stage increments when it is
  not identity (`lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl:80-88`
  and `:194-202`). General Rodas stages likewise branch on identity before
  forming later stage right-hand sides (`rosenbrock_perform_step.jl:730-785`).
- BDF methods apply the mass matrix to history/predictor terms, for example in
  the in-place path at `lib/OrdinaryDiffEqBDF/src/bdf_perform_step.jl:140-153`,
  and recover the final derivative from the converged nonlinear relation using
  `M*u` at `bdf_perform_step.jl:1490-1502`. These uses require `apply`, not only
  dense W assembly.

The Rust port currently uses the algebraically scaled convention
`I-(gamma*h)J`, not upstream's `I/(gamma*h)-J`. Keeping the Rust convention is a
local numerical-preservation decision. With a mass matrix it becomes
`M-(gamma*h)J`; each family must also scale its right-hand side consistently.
Copying upstream's W formula without copying the corresponding stage scaling
would be incorrect.

### Reuse, invalidation, and cached solves

- Upstream distinguishes “new Jacobian” from “new W”. For Newton paths a linear
  RHS has constant J but W must be rebuilt after sufficient `gamma*dt` drift;
  nonadaptive paths rebuild; nonlinear convergence/failure, step-size change,
  derivative discontinuity, first stage, and varying mass matrix enter the
  decision (`derivative_utils.jl:519-578`). In particular, a varying mass matrix
  forces W refresh even when J is reusable (`derivative_utils.jl:546-578`).
- The newer nonlinear-solver reuse path likewise sets `new_jac` only on first
  use/forced refresh/divergence and independently compares the W scale
  (`lib/OrdinaryDiffEqNonlinearSolve/src/newton.jl:118-136`). A concrete cached
  J is used to rebuild W without a new Jacobian evaluation
  (`newton.jl:175-205`).
- The cached linear problem aliases A and b at construction
  (`lib/OrdinaryDiffEqNonlinearSolve/src/utils.jl:509-515`). During Newton, a
  new W is passed only when refactorization is required; later solves update b/u
  but pass no A (`newton.jl:435-445`). Linear failures are checked and `nsolve`
  is recorded centrally (`newton.jl:447-454`).
- `dolinsolve` mutates/reinitializes the cached linear solve and calls `solve!`
  (`lib/OrdinaryDiffEqDifferentiation/src/linsolve_utils.jl:20-38`). It also owns
  the extra RHS accounting needed by finite-difference matrix-free iterations
  (`linsolve_utils.jl:39-50`). `issuccess_W` separately tests factorization
  success (`linsolve_utils.jl:1-10`).
- Rosenbrock caches build J/W once and initialize an aliased linear solve once
  (`lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_caches.jl:243-260`). A step
  passes W only when it is new and otherwise solves with the cached
  factorization (`rosenbrock_perform_step.jl:55-73`, `:169-187`, and
  `:751-766`). Rosenbrock differentiation updates the time derivative only
  when J changes (`derivative_utils.jl:935-964`).
- Upstream's W-method reuse policy explicitly rebuilds W from a cached J when
  `dtgamma` changes and treats that O(n²) assembly/factorization as cheaper than
  reevaluating J (`derivative_utils.jl:20-40` and `:1007-1025`). Strict
  Rosenbrock behavior and mass-matrix behavior are more conservative
  (`derivative_utils.jl:42-100`).
- TRBDF2 exposes `linsolve`, nonlinear solve, `smooth_est`, autodiff, and
  `concrete_jac` as distinct configuration axes
  (`lib/OrdinaryDiffEqSDIRK/src/algorithms.jl:181-227`). This does not imply that
  all axes must become public in Rust Phase 3; it does show why Jacobian choice,
  factorization storage, and solver-family logic must not be one Boolean.

## Proposed crate-private Rust interfaces

The signatures below are implementation-ready targets. Names may change during
review, but the ownership and invariants should not.

### Errors, layout, and checked views

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinearError {
    EmptyDimension,
    DimensionOverflow { rows: usize, columns: usize },
    LengthMismatch { expected: usize, actual: usize },
    NonSquare { rows: usize, columns: usize },
    NonFiniteCoefficient,
    Singular,
    Unfactorized,
    DenseRepresentationRequired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StateLayout {
    dimension: usize,
}

impl StateLayout {
    pub(crate) fn new(dimension: usize) -> Result<Self, LinearError>;
    pub(crate) fn dimension(self) -> usize;
    pub(crate) fn matrix_len(self) -> Result<usize, LinearError>;
    pub(crate) fn state<'a>(self, data: &'a [f64])
        -> Result<StateRef<'a>, LinearError>;
    pub(crate) fn state_mut<'a>(self, data: &'a mut [f64])
        -> Result<StateMut<'a>, LinearError>;
    pub(crate) fn matrix<'a>(self, data: &'a [f64])
        -> Result<DenseMatrixRef<'a>, LinearError>;
    pub(crate) fn matrix_mut<'a>(self, data: &'a mut [f64])
        -> Result<DenseMatrixMut<'a>, LinearError>;
}

#[derive(Clone, Copy)]
pub(crate) struct StateRef<'a> {
    layout: StateLayout,
    data: &'a [f64],
}

pub(crate) struct StateMut<'a> {
    layout: StateLayout,
    data: &'a mut [f64],
}

#[derive(Clone, Copy)]
pub(crate) struct DenseMatrixRef<'a> {
    rows: usize,
    columns: usize,
    data: &'a [f64],
}

pub(crate) struct DenseMatrixMut<'a> {
    rows: usize,
    columns: usize,
    data: &'a mut [f64],
}

impl<'a> DenseMatrixRef<'a> {
    pub(crate) fn from_row_major(
        data: &'a [f64], rows: usize, columns: usize,
    ) -> Result<Self, LinearError>;
    pub(crate) fn rows(self) -> usize;
    pub(crate) fn columns(self) -> usize;
    pub(crate) fn row(self, row: usize) -> Option<&'a [f64]>;
    pub(crate) fn as_slice(self) -> &'a [f64];
}

impl<'a> DenseMatrixMut<'a> {
    pub(crate) fn from_row_major(
        data: &'a mut [f64], rows: usize, columns: usize,
    ) -> Result<Self, LinearError>;
    pub(crate) fn as_ref(&self) -> DenseMatrixRef<'_>;
    pub(crate) fn row_mut(&mut self, row: usize) -> Option<&mut [f64]>;
    pub(crate) fn as_mut_slice(&mut self) -> &mut [f64];
}
```

Construction checks `rows.checked_mul(columns)` and exact slice length. All
matrix data is row-major; there is no stride, transpose, shape, or generic
scalar abstraction in Phase 3. `StateRef` is `Copy`; mutable views are not.
Views borrow caller-owned storage and never allocate. Hot loops take one
checked row slice and iterate it; they do not call a checked `(row,column)`
getter per coefficient.

`LinearError::Singular` maps to the existing
`SolveError::SingularLinearSystem`. A nonfinite analytic/finite-difference
Jacobian initially maps to the existing `NonFiniteDerivative` to preserve
public behavior. Dimension errors should become a distinct solver-boundary
error before the mass-matrix representation is public; they must never be
misreported as singular. Internal constructors still return `LinearError` so a
programming error cannot turn into slice panic.

### Dense LU and factorization state

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MatrixRevision(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SystemMatrixKey {
    pub(crate) jacobian: MatrixRevision,
    pub(crate) mass_matrix: MatrixRevision,
    pub(crate) scale_bits: u64,
}

pub(crate) struct DenseLu {
    layout: StateLayout,
    factors: Vec<f64>,
    pivots: Vec<usize>,
    valid: bool,
    revision: MatrixRevision,
}

impl DenseLu {
    pub(crate) fn new(layout: StateLayout) -> Result<Self, LinearError>;
    pub(crate) fn dimension(&self) -> usize;
    pub(crate) fn invalidate(&mut self);
    pub(crate) fn is_valid(&self) -> bool;
    pub(crate) fn factors_mut(&mut self) -> DenseMatrixMut<'_>;
    pub(crate) fn load(&mut self, matrix: DenseMatrixRef<'_>)
        -> Result<(), LinearError>;
    pub(crate) fn factorize(&mut self, stats: &mut SolverStats)
        -> Result<MatrixRevision, LinearError>;
    pub(crate) fn solve_in_place(
        &self, right_hand_side: StateMut<'_>, stats: &mut SolverStats,
    ) -> Result<(), LinearError>;
}
```

`new` is the only allocation: `n*n` factors and `n` pivots. Calling
`factors_mut` or `load` invalidates the old factorization before returning.
`factorize` starts invalid, succeeds atomically with respect to the validity
flag, and increments a wrapping nonzero revision only on success. A failed
factorization remains invalid. `solve_in_place` checks layout and validity and
does not mutate the factorization, so multiple right-hand sides reuse it.

For the first migration, the elimination order, pivot tie rule, row swaps, and
absolute `f64::EPSILON` singular cutoff remain byte-for-byte equivalent to
`src/linear.rs`. This protects current numerical behavior. A later benchmark/
accuracy task may replace the cutoff, but not as incidental refactoring.

Add `linear_factorizations` to `SolverStats` in the integration wave. Stats
ownership is centralized:

- the shared RHS evaluator increments `rhs_evaluations`;
- the Jacobian provider increments `jacobian_evaluations` once per completed
  fill and routes finite-difference RHS calls through that evaluator;
- `DenseLu::factorize` increments `linear_factorizations` once per successful
  factorization;
- `DenseLu::solve_in_place` increments `linear_solves` once per successful
  solve;
- kernels retain ownership of nonlinear iterations and accepted/rejected
  steps.

Callers must remove their existing manual increments when adopting these
helpers. This is the only way to avoid double counts as the same factorization
is reused by stages and smooth estimators.

### Jacobian providers

```rust
#[derive(Clone, Copy)]
pub(crate) struct JacobianPoint<'a> {
    pub(crate) state: StateRef<'a>,
    pub(crate) base_derivative: StateRef<'a>,
    pub(crate) time: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JacobianSource {
    Analytic,
    ForwardFiniteDifference,
}

pub(crate) trait DenseJacobianProvider<F, P>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn layout(&self) -> StateLayout;
    fn revision(&self) -> MatrixRevision;
    fn invalidate(&mut self);
    fn fill(
        &mut self,
        problem: &OdeProblem<F, P>,
        point: JacobianPoint<'_>,
        output: DenseMatrixMut<'_>,
        stats: &mut SolverStats,
    ) -> Result<JacobianSource, SolveError>;
}

pub(crate) struct OdeJacobianProvider {
    layout: StateLayout,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    revision: MatrixRevision,
}
```

`OdeJacobianProvider` is a static, crate-private adapter around today's
`OdeProblem::evaluate_jacobian`: use the analytic callback if present,
otherwise perform the current forward column difference with
`sqrt(EPSILON)*max(abs(u[j]),1)`. It owns the two scratch vectors now repeated
in every solver. `fill` checks output layout, requires the provided base
derivative to be at the same point, verifies all output coefficients are
finite, increments its revision and Jacobian statistic only after a complete
fill, and allocates nothing after construction.

The base-derivative point is an invariant the type cannot fully prove. The
kernel owns it and must invalidate/re-evaluate after callback effects, resize,
or accepted state change. A later analytic Jacobian public representation can
implement the same internal trait without changing kernels. Rosenbrock time
differentiation remains a separate cached operation tied to the Jacobian
revision; it must not be counted as a second Jacobian.

### Linear operators and Jv

```rust
pub(crate) trait LinearOperator {
    fn layout(&self) -> StateLayout;
    fn revision(&self) -> MatrixRevision;
    fn apply(
        &mut self,
        output: StateMut<'_>,
        input: StateRef<'_>,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>;
    fn dense(&self) -> Option<DenseMatrixRef<'_>> {
        None
    }
}

pub(crate) struct DenseMatrixOperator<'a> {
    matrix: DenseMatrixRef<'a>,
    revision: MatrixRevision,
}

pub(crate) struct FiniteDifferenceJvp<'a, F, P> {
    problem: &'a OdeProblem<F, P>,
    point: JacobianPoint<'a>,
    perturbed_state: &'a mut [f64],
    perturbed_derivative: &'a mut [f64],
    revision: MatrixRevision,
}

pub(crate) struct ShiftedOperator<'a, M, J> {
    mass: &'a mut M,
    jacobian: &'a mut J,
    scale: f64,
    scratch: &'a mut [f64],
    revision: MatrixRevision,
}
```

`FiniteDifferenceJvp` binds the problem and one valid base point for the
duration of a linear solve. `apply` computes a directional perturbation into
preallocated scratch and routes the RHS call through shared accounting. An
analytic Jv adapter can have the same shape later. `ShiftedOperator` applies
`M*x-scale*J*x`, supporting a future Krylov backend without requiring a dense
J or W. These types use generic static dispatch; no trait object is required in
the numerical hot path.

Do not implement GMRES, preconditioners, sparse storage, or a public
linear-solver selector in the first task. The interface is justified now
because matrix-free Jv is required by later in-scope methods, but a concrete
iterative backend should arrive with its solver-family tests and benchmark.

If `jvp_evaluations` is added to `SolverStats`, `LinearOperator::apply` owns that
increment for Jacobian actions. Finite-difference Jv additionally increments
RHS work through the RHS evaluator. Dense matrix and mass applications do not
increment Jv or linear-solve counts.

### Nonsingular mass-matrix operators

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OperatorUpdate {
    Unchanged,
    ValuesChanged(MatrixRevision),
}

pub(crate) trait NonsingularMassMatrix<P>: LinearOperator {
    fn update(
        &mut self,
        state: StateRef<'_>,
        parameters: &P,
        time: f64,
    ) -> Result<OperatorUpdate, SolveError>;

    fn solve_in_place(
        &mut self,
        right_hand_side: StateMut<'_>,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>;
}

pub(crate) struct IdentityMass {
    layout: StateLayout,
}

pub(crate) struct DenseConstantMass {
    matrix: Vec<f64>,
    solve: DenseLu,
    revision: MatrixRevision,
}

pub(crate) fn assemble_identity_minus_scaled_jacobian(
    output: DenseMatrixMut<'_>,
    jacobian: DenseMatrixRef<'_>,
    scale: f64,
) -> Result<(), LinearError>;

pub(crate) fn assemble_dense_mass_minus_scaled_jacobian(
    output: DenseMatrixMut<'_>,
    mass: DenseMatrixRef<'_>,
    jacobian: DenseMatrixRef<'_>,
    scale: f64,
) -> Result<(), LinearError>;
```

The trait's “nonsingular” promise is established by construction. Identity is
zero-cost; constant dense mass validates shape/finiteness and factorizes once.
`solve_in_place` supplies `M^-1*f` where an algorithm needs a physical rate;
`apply` supplies `M*x` for BDF/Rosenbrock histories and stage right-hand sides;
`dense` permits the first direct backend to assemble `M-scale*J`. An operator
whose `dense()` is `None` is rejected by `DenseLu` with
`DenseRepresentationRequired`; it becomes usable only with a future iterative
solver.

`update` is present for a future state/time-dependent nonsingular operator. A
reported value change advances its revision and invalidates any W key. The
first public Phase 5 representation should expose only identity and constant
dense mass unless a varying-mass algorithm and compliance test are ready.
Singular matrices are rejected and remain out of scope; this trait must not be
reused to imply DAE residual support.

### Cache invariants

Each kernel retains the policy decision and stores a `SystemMatrixKey`. Shared
linear code supplies mechanisms, not solver policy.

1. A Jacobian fill advances `jacobian` revision.
2. A mass `ValuesChanged` result advances `mass_matrix` revision.
3. The exact system scale is keyed by `scale.to_bits()`, matching current exact
   `f64` comparison rather than adding an unreviewed tolerance.
4. Any mutable access to LU factors invalidates the factorization.
5. A key is installed only after assembly and factorization both succeed.
6. Callback effects, resize, algorithm switch, or a kernel-declared derivative
   discontinuity invalidate the Jacobian point and therefore W.
7. Rejection does not automatically invalidate a Jacobian at the unchanged
   state, but a changed step scale invalidates W. Family policy may still force
   a fresh J after convergence failure.
8. A terminating callback performs no update, Jacobian, W assembly,
   factorization, or solve afterward, consistent with the shared-driver gate.

The policy deliberately permits J reuse while rebuilding/factorizing W. That
is the upstream distinction needed by SDIRK/BDF/Rosenbrock and avoids encoding
two independent facts in `factorization_valid`.

## Current caller migration map

| Current location | Current responsibility | Proposed replacement | Required preservation |
|---|---|---|---|
| `src/linear.rs:3-67` | raw LU functions | `StateLayout`, dense views, `DenseLu` | exact elimination/pivot/singularity rule initially; zero hot-path allocations |
| `src/problem.rs:126-132` | optional analytic callback probe | called only by `OdeJacobianProvider` adapter | public `with_jacobian` and row-major `Vec<f64>` API unchanged |
| `src/implicit.rs:52-80` | per-kernel FD scratch, matrix, pivots, scale | `OdeJacobianProvider`, `DenseLu`, `Option<SystemMatrixKey>` | allocate once in workspace |
| `src/implicit.rs:216-275` | chord-Newton reuse and solves | kernel chooses refresh; `DenseLu::solve_in_place` owns solve count | one stale correction before forced refresh; callback invalidation |
| `src/implicit.rs:325-382` | J selection, `I-scale*J`, factorization | provider fills `DenseLu::factors_mut`; assembly helper transforms it; factorize | same perturbation, finite checks, exact scale key |
| `src/trbdf2.rs:45-80` | FD scratch plus Boolean LU validity | provider, `DenseLu`, key | no per-stage allocation |
| `src/trbdf2.rs:271-287` | smooth-estimator solve | solve through the last valid `DenseLu` | rebuild at final candidate only when key is absent/stale |
| `src/trbdf2.rs:298-423` | Newton J/W rebuild and solve | separate J revision from W key; provider/assembly/LU | current rebuild frequency during Newton until separately reviewed |
| `src/rosenbrock.rs:15-53` | separate J/W buffers and Boolean differentiation validity | provider, cached time derivative revision, `DenseLu` | same allocated vector count or fewer |
| `src/rosenbrock.rs:205-305` | form W and solve three RHS values | assemble into `DenseLu`, factor once, solve three times | stage arithmetic/order unchanged |
| `src/rosenbrock.rs:308-365` | J and time finite differences | provider for J; small Rosenbrock-only time derivative cache | same perturbations and counts |
| `src/rosenbrock_extended.rs:398-432` | shared Rosenbrock/Rodas raw caches | same provider/LU/cache structure | max-stage buffer unchanged |
| `src/rosenbrock_extended.rs:562-730` | three/many solves against one W | one `DenseLu` factorization, repeated `solve_in_place` | tableau arithmetic and RHS scaling unchanged |
| `src/rosenbrock_extended.rs:750-828` | duplicated J/time derivative/W build | provider plus identity assembly and LU | `gamma*h` remains part of W key |

Migration should be one solver module at a time after the standalone linear
types pass. Do not edit all four kernels in one change: statistics ownership
and invalidation are easier to review with focused numerical/allocation tests.

## What must wait for Phase 5 problem representations

- A public mass-matrix constructor/API, dimension validation at problem
  construction, and algorithm compatibility errors.
- Public analytic Jv and user-selectable linear-solver configuration. The
  private operator seam can land now; the public callback shape depends on the
  general problem representation.
- Split/IMEX Jacobian ownership (full J versus implicit-component J).
- State/time/parameter-dependent mass operators and their update callbacks.
- Selection between dense direct and iterative solvers for operator-only mass
  matrices.
- Any singular mass matrix, algebraic-variable detection, consistent
  initialization, or DAE residual Jacobian. Those remain explicitly excluded.
- Non-`Vec<f64>` state shapes. `StateLayout` describes one contiguous dimension
  only and must not be presented as general array support.

The identity/dense `NonsingularMassMatrix` implementations and assembly helpers
can be implemented privately before Phase 5, but no current solver should
claim mass-matrix support until residual, predictor, derivative, dense-output,
and compliance behavior are all wired for that representation.

## Required Phase 3 tests and benchmarks

### Linear unit tests

- `StateLayout` rejects zero, multiplication overflow, short, and long slices.
- Dense views prove row-major rows for rectangular matrices and reject length
  mismatch; no production indexing panic is used for a dimension error.
- LU retains the existing row-exchange case, handles 1x1 and multiple RHS
  solves, and reports unfactorized, nonfinite, and singular inputs.
- Mutable factor access invalidates LU; failure leaves it invalid; successful
  refactorization advances revision.
- Regression vectors cover pivot ties and a small-magnitude matrix around the
  current absolute epsilon cutoff so a later policy change is visible.
- Identity/dense `M-scale*J` assembly checks dimensions, diagonal/off-diagonal
  values, negative/backward steps, and nonfinite scale.

### Jacobian/operator tests

- Analytic and finite-difference providers agree on scalar, dense coupled, and
  nonautonomous systems.
- Analytic callback nonfinite output and finite-difference nonfinite RHS return
  the existing error without validating the new revision/key.
- Exact statistics: analytic fill is one Jacobian/zero extra RHS; n-dimensional
  forward difference is one Jacobian/n extra RHS when the base derivative is
  supplied.
- Dense Jv equals dense matrix multiplication; finite-difference Jv agrees
  within its perturbation accuracy and records its RHS work.
- `ShiftedOperator` agrees with explicitly assembled identity and dense-mass
  matrices.
- Operator/mass revision changes make an old `SystemMatrixKey` unequal.

### Per-family migration regression tests

- Existing endpoint, analytic-Jacobian, and reuse tests stay unchanged.
- Add exact factorization/solve-count assertions for one accepted fixed step:
  three Rosenbrock23 solves per factorization, three Rosenbrock32 solves per
  factorization, Rodas stages per factorization, and TRBDF2's smoothed estimate
  reusing the last valid W.
- Exercise accepted callback mutation, rejected step with changed h, backward
  integration, nonlinear retry, analytic versus finite-difference J, singular
  W, and nonfinite J.
- Use `stats_alloc` to assert zero allocations after workspace construction for
  LU factorization/solve, Jacobian fill, and one fixed solver step without
  saving/callback growth.
- Run all existing Julia compliance fixtures after each kernel migration; no
  new Julia fixture is required for private interfaces, but mass-matrix public
  support later requires matched pinned-Julia fixtures.

### Benchmark gate before any dependency

Benchmark the current raw LU against `DenseLu` at dimensions 1, 4, 16, 64, and
128 for factor-once/solve-once and factor-once/solve-eight workloads. Record
time, allocations, and peak live bytes. Also benchmark finite-difference dense
J assembly and a representative fixed stiff step. Acceptance is no allocation
increase in hot paths and no unexplained runtime regression above measurement
noise. Only after these results should a small or large algebra crate be
evaluated, using the same benchmark and binary-size/compile-time data.

## Bounded first implementation task

**Task:** Land checked dense views and owning LU state without migrating a
solver.

**Allowed files:** `src/linear.rs` and its inline unit tests only.

**Forbidden files:** public API modules, solver modules, Cargo manifests,
Julia, inventory, coverage, status, blocker, and benchmark files.

**Implementation:** Add `LinearError`, `StateLayout`, state/matrix views,
`MatrixRevision`, and `DenseLu`. Preserve the existing free functions
temporarily as thin legacy paths or leave callers untouched; prove the new
implementation's output against them. Do not add a dependency.

**Tests:** all linear unit tests listed above that do not require mass or a
problem. Include zero-allocation factor/repeated-solve coverage with the
existing dev dependency if it can remain inside the allowed file.

**Commands:**

```powershell
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check
```

**Definition of done:** checked dimensions cannot panic through the public
crate-private entry points; numerical LU results and singular cutoff match the
old functions; mutation/failure invalidates factor state; repeat solves
allocate zero; all required commands pass. The next task migrates exactly one
kernel (Implicit Euler/Midpoint/Trapezoid is the smallest) and adds centralized
statistics.

## Handoff report

**Summary:** Completed a report-only pinned-upstream and current-code audit for
Phase 3. Recommended a dense-first, dependency-free set of crate-private views,
Jacobian/operator interfaces, reusable LU state, mass-operator extension, cache
keys, ownership rules, migration order, tests, benchmarks, and a bounded first
implementation task.

**Files changed:** `docs/handoffs/linear_interface_audit.md` only.

**Public APIs added:** None.

**Upstream source and revision:** `D:/Source/_review/OrdinaryDiffEq.jl` at
`211142263781255a9aa2f910f6760b9f18ec29c8`; exact source/line evidence is cited
above.

**Rust tests:** None required for the report-only task.

**Julia tests:** None required for the report-only task.

**Commands run:** Upstream `git rev-parse HEAD` and commit verification; local
and upstream `rg` call-site/source audits; exact numbered source extraction;
`git diff --check`; final `git status`.

**Numerical differences:** None; documentation only. The recommended first
migration deliberately preserves the current LU pivoting and absolute singular
cutoff. Upstream/local W scaling differences are identified explicitly.

**Allocation/performance impact:** None; documentation only. Proposed hot-path
operations allocate only during workspace/provider/LU construction and require
allocation/regression benchmarks before a dependency or backend replacement.

**Known limitations:** Dense contiguous `f64`, row-major storage, and direct LU
are the only first backend. Public problem representations, analytic Jv,
iterative/sparse/GPU/distributed solvers, and varying mass matrices are
deferred. Singular mass matrices and DAE-only behavior remain excluded.

**Follow-up dependencies:** Standalone linear primitives; one-by-one current
kernel migrations; Phase 5 problem representation for public nonsingular mass
matrices/Jv selection; performance evidence before any dependency.

**Recommended next task:** Implement only the bounded `src/linear.rs` task
above, then migrate `src/implicit.rs` as the first consumer with exact
statistics and allocation assertions.
