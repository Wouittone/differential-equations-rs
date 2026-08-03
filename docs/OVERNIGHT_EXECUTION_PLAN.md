# Overnight regular-ODE parity execution plan

This is the durable implementation plan for unattended work on
`differential-equations-rs`.

The target is native regular initial-value ODE parity with
SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. It includes first-order, stiff,
split/IMEX, multistep, second-order, partitioned, symplectic, exponential,
extrapolation, multirate, and automatic ODE algorithms. It excludes SDEs,
DDEs, BVPs, PDEs, steady-state solvers, DAE-only residual behavior, and
external wrappers.

## Execution gates

1. Soundness audit and exact upstream inventory.
2. Shared first-order integrator driver.
3. Internal vector/matrix and linear-solver interfaces.
4. Declarative coefficient schema and Rust code generation.
5. General ODE problem representations.
6. Dense output, time stops, callbacks, and controller parity.
7. Solver-family migration and new-family ports.
8. Continuous Julia compliance, performance checks, inventory regeneration,
   and final audit.

Do not begin a dependent phase until the preceding phase has passing tests and
a documented handoff. Inventory generation and upstream source auditing may
run in parallel with the first architecture phase.

## Global execution rules

- Do not ask the user for clarification during unattended execution.
- Preserve unrelated working-tree changes.
- Never weaken a tolerance or delete a failing test to make progress appear
  successful.
- If blocked, record the exact blocker and continue independent work.
- Use separate branches or worktrees for nontrivial agent tasks.
- Every public algorithm requires a matched Julia compliance test.
- Do not claim a feature when only endpoint stepping has been implemented.
- Do not introduce runtime YAML parsing or dynamic dispatch in numerical hot
  paths.
- Keep generated files and pinned-upstream artifacts reproducible.
- Shared files such as `src/lib.rs`, `tests/julia/runtests.jl`, manifests,
  coverage documents, and status documents are coordinator-owned unless a task
  explicitly grants ownership.

## Phase 1: soundness and inventory

The first agent is a simplifier/soundness gate. It inspects the worktree,
preserves intended behavior, removes avoidable complexity, and runs:

```powershell
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl
```

The inventory agent regenerates the machine-readable regular-ODE inventory from
the exact upstream revision and records each constructor's name, alias status,
package, source path/line, family, problem representation, fixed/adaptive
behavior, Jacobian/linear-solver/dense-output/controller requirements, Rust
status, Julia status, and exclusion rationale. All source references must
resolve and repeated generation must be byte-stable.

## Phase 2: shared first-order integrator driver

Add an internal module such as `src/integrator.rs`. The driver owns current and
candidate buffers, time direction, endpoint clipping, attempt counting,
accepted/rejected step accounting, callback processing, event localization,
`SaveMode`, `save_at`, termination, common errors, controller invocation, and
solution assembly.

Numerical kernels own stages, derivatives, Jacobians, factorizations, multistep
histories, estimators, and cache policy. Use static dispatch with an internal
kernel interface equivalent to:

```rust
pub(crate) trait StepKernel<F, P>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities;
    fn initialize(&mut self, problem: &OdeProblem<F, P>, state: &[f64],
        time: f64, stats: &mut SolverStats) -> Result<(), SolveError>;
    fn estimate_initial_step(&mut self, problem: &OdeProblem<F, P>, state: &[f64],
        time: f64, direction: f64, maximum_step: f64, options: &SolveOptions,
        stats: &mut SolverStats) -> Result<f64, SolveError>;
    fn attempt_step(&mut self, problem: &OdeProblem<F, P>, state: &[f64],
        time: f64, step: f64, candidate: &mut [f64], options: &SolveOptions,
        stats: &mut SolverStats) -> Result<StepEstimate, SolveError>;
    fn accept_step(&mut self, problem: &OdeProblem<F, P>, previous_state: &[f64],
        state: &[f64], time: f64, accepted_step: f64, callback_applied: bool,
        stats: &mut SolverStats) -> Result<(), SolveError>;
    fn reject_step(&mut self);
}
```

The lifecycle is: apply initial callbacks; initialize the kernel; select a
step; attempt it; reject without callbacks or saving if its error exceeds one;
locate continuous events on the pre-effect candidate; apply callback effects;
record the pre-effect trajectory; force-save post-effect callback state; return
immediately for termination; swap the accepted state; call the kernel's accept
hook; and update the proposed step. A terminating callback must not evaluate an
RHS, Jacobian, factorization, or multistep history afterward.

Migrate fixed explicit methods first, then adaptive explicit RK, one-step
implicit methods, Rosenbrock/Rodas, fixed Adams, and variable Adams. Add
mock-kernel tests for rejection, backward integration, callbacks, termination,
`save_at`, underflow, max steps, and allocation behavior before migration.

## Phase 3: internal vector/matrix interfaces

Keep the public `Vec<f64>` API initially. Add narrow internal abstractions for
contiguous states and layouts, dense row-major matrix views, Jacobian providers,
linear operators, reusable linear solvers, and nonsingular mass-matrix
operators.

The first backend remains the current dense `f64` slice/LU implementation. Do
not introduce a large linear-algebra dependency without a benchmark proving it
helps. Support dense direct solves, factorization reuse, finite differences,
analytic Jacobians, Jacobian-vector products where needed, singular-system
errors, and solver statistics. Existing implicit and Rosenbrock methods must
use these interfaces with no numerical or allocation regression.

## Phase 4: coefficient schema and code generation

Store coefficient data declaratively, but never parse YAML during a solve.
Generate compile-time Rust constants.

Use family-specific tagged records for explicit/embedded Butcher tableaus,
Rosenbrock/Rodas tableaus, Shu-Osher forms, low-storage recurrences, multistep
coefficients, symplectic/partitioned coefficients, and dense interpolation.

Every record includes method name, family, order, embedded orders, FSAL status,
stage times, coefficients, dense data if available, upstream package/path,
commit, and caveats. Represent values as rationals, decimal strings, symbolic
constants, or explicit hexadecimal floats where required. Validate dimensions,
triangular structure, FSAL constraints, finiteness, estimator lengths, stage
times, and available order conditions before emitting Rust.

Migrate existing tableaus first, then Verner, Rosenbrock/Rodas, SSP,
low-storage, and multistep families. Generated output must be deterministic and
formatted.

## Phase 5: general problem representations

Add separate typed representations rather than flattening every problem:

- `OdeProblem` for first-order equations;
- split/IMEX problems with explicit and implicit RHS components;
- specialized second-order problems with separate position and velocity;
- general partitioned/dynamical problems;
- nonsingular mass-matrix ODEs.

Each representation defines dimensions, parameters, callbacks, saving,
Jacobian/operator metadata, forward/backward semantics, and compatibility with
the shared driver. DAE-only residual initialization and singular-mass behavior
remain excluded.

## Phase 6: dense output and controller parity

Add method-specific dense segments rather than relying permanently on linear
interpolation:

```rust
pub(crate) trait DenseSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]);
}
```

Implement interpolation first for explicit RK, then Verner/Owren-Zen,
Rosenbrock/Rodas/TRBDF2, multistep methods, and second-order/partitioned
methods. Use dense segments for `save_at`, continuous root finding, and
solution queries.

Centralize controller behavior while retaining per-method metadata: initial
step estimation, safety/minimum/maximum growth, rejection limiting, PI/PID
behavior, step history, `dtmin`, `dtmax`, time stops, and zero-error or
steady-state behavior. Preserve existing behavior during extraction; add new
parity features with separate tests.

## Phase 7: solver-family expansion

After the architecture gates pass, continue waves in this order:

1. Remaining explicit and high-order RK.
2. Rosenbrock, Rodas, SDIRK, ESDIRK, and KenCarp.
3. BDF, QNDF, and remaining variable-order multistep.
4. IMEX and split methods.
5. Multirate and MRI-GARK.
6. RKN and Nyström.
7. Remaining symplectic and partitioned methods.
8. Exponential and linear/operator methods.
9. Extrapolation.
10. Stabilized methods.
11. Automatic/default composite and stiffness-switching methods.

Each family agent documents required representations, coefficient schemas,
dense interpolation, controller behavior, cache invalidation, Julia package,
compliance fixture, and unsupported upstream options.

## Phase 8: compliance, performance, and final audit

Every algorithm receives applicable tests for scalar/vector systems,
nonautonomous equations, forward/backward integration, fixed/adaptive modes,
convergence order, stiffness, Jacobian modes, callbacks, `save_at`, dense
output, failures, and statistics.

Measure runtime, allocation traffic, peak live memory, RHS calls, Jacobian
calls, linear solves, nonlinear iterations, accepted steps, and rejected steps.

Final parity requires no missing in-scope inventory entries; every public
algorithm implemented or mapped to a tested alias; Julia compliance for every
public algorithm; all supported problem representations; method-specific dense
output and controller behavior; passing Rust, Clippy, formatting, pinned-Julia,
inventory, and diff checks; documented exclusions; and no unexplained
fixed-step or callback-free allocation regressions.
