# Release process

The workspace contains three versioned crates: `tableau-core`,
`tableau-macros`, and the main `differential-equations-rs` package (whose Rust
library target remains `differential_equations`). Internal
dependencies use exact versions, so publication order is mandatory.

The main manifest currently contains `publish = false`. Removing that lock
requires an explicit reviewed release change.

## Roadmap to 1.0

The existing foundation includes the Rust 1.85 MSRV, Rust 2024 edition, dual
MIT/Apache-2.0 licensing, supply-chain checks, hierarchical solver modules,
Rayon-backed ordered ensembles, and shape-aware ndarray entry points for
scalar, vector, and matrix states. The repository also has compile-time
validated lazy JSON tableaus, allocation tests, pinned Julia comparisons, and
a CodSpeed-compatible Criterion regression harness.

Every item in this section is a 1.0 release blocker.

The resource model, automatic switching, module decomposition, public API
freeze, and local numerical certification are complete. The remaining release
evidence is a green remote CI run from the reviewed commits. A checkbox
represents committed repository state, not work in progress. Close one only
when its implementation, focused regression tests, public documentation, and
applicable local gates land together; record follow-up work as another
unchecked item rather than weakening its completion criteria.

### Finish typed tableau resources

All coefficient data must use typed, method-oriented JSON resources validated
during macro expansion, embedded with `include_str!`, and parsed behind an
independent `LazyLock` only when its method is used. Generated Rust arrays and
generic named-constant banks are not acceptable final representations.

- [x] Complete the second-order RKN/IRKN migration, including fixed and
  adaptive RKN methods, dense extensions, IRKN history data, shared RKN startup
  tableaus, independently lazy method resources, and fallible `.tableau()`
  inspection.
- [x] Model every low-storage RK recurrence explicitly. This includes the 2N,
  2C, 3S, alternating-register, and register-pipeline layouts; each resource
  must validate the invariants of its actual recurrence.
- [x] Add the upstream embedded estimators and genuine adaptive/FSAL lifecycle
  for the RDPK 3S-plus and CKLL register-pipeline families. Preserve their
  method-specific controller metadata without reintroducing Rust constants.
- [x] Replace the combined ROCK/SERK/ESERK bank with typed stabilized-method
  resources that do not parse or retain the entire catalogue for one selected
  method and degree.
- [x] Remove `define_tableau_data_from_file!`, every legacy `coefficient_data`
  module, and their obsolete combined resource files after the final consumer
  is migrated.
- [x] Remove the public const-tableau marker API and the crate-root explicit-RK
  macro alias; downstream methods use validated JSON resources through the
  `tableau` namespace.
- [x] Keep resource JSON consistently styled with stable field ordering,
  compact scalar vectors, and one matrix row per line without adding a
  formatting build step.
- [x] Keep the schemas and tableau resource guide synchronized with the actual
  migration status.

### Implement genuine in-flight automatic switching

Automatic algorithms switch from non-stiff to stiff integration at the current
accepted state. A fallback that restarts from the initial condition after the
explicit algorithm fails is prohibited.

- [x] Retain the accepted time and state, proposed step, direction, pending time
  stops, callbacks, saving and dense-output state, and cumulative statistics
  continuously in the shared driver across every branch handoff. Reinitialize
  method-specific derivative caches at that accepted state and deliberately
  reset controller error history between methods.
- [x] Separate driver-owned integration state from algorithm caches so a stiff
  cache can be initialized lazily at the switching point.
- [x] Preserve accepted evaluations, callback effects, saved points, mutable
  parameter side effects, exact stops, termination, interpolation, and error
  semantics without replaying the initial interval.
- [x] Add a documented stiffness detector that can switch before fatal explicit
  failure using stability-limited steps, rejection history, and available
  derivative or Jacobian estimates.
- [x] Add hysteresis and minimum-residence rules to avoid repeated switching.
  Non-stiff-to-stiff handoff is mandatory; controlled stiff-to-non-stiff
  handoff should be supported by automatic pairs that can provide it.
- [x] Make detector thresholds inspectable and configurable with conservative
  defaults, and expose switch counts and active-algorithm information in solver
  statistics.
- [x] Return typed errors for incompatible handoff pairs; never silently fall
  back to a full restart.
- [x] Apply the shared switching mechanism to `AutoDP5` and the other automatic
  composites.

Verification must cover a non-stiff problem that never switches, a problem
that becomes stiff, an initially stiff problem, and stiffness that later
disappears. Instrumented tests must prove that no accepted interval, callback,
or side effect is replayed. The matrix must also cover forward and backward
integration, rejected steps, time stops, dense output, callback termination,
mutable parameters, and scalar/vector/matrix ndarray states. Compare equivalent
cases with pinned Julia SciML behavior and benchmark detector overhead, switch
cost, allocations, and both explicit-only and stiff-only baselines.

### Finish code and API hardening

- [x] Split oversized files along stable responsibilities, especially the
  second-order API and drivers, extended Rosenbrock and stabilized methods,
  and the shared problem, solution, and integration cores.
- [x] Keep algorithms under `solvers::<family>` while driver, workspace,
  controller, and resource implementation details remain private unless they
  are deliberate extension surfaces.
- [x] Standardize constructors, options, typed errors, tableau inspection,
  solutions, and interpolation across solver families.
- [x] Complete rustdoc examples and links for every public type and supported
  extension point.
- [x] Add downstream smoke crates for default and no-default features, a
  renamed dependency, a file-defined tableau, and scalar/vector/matrix states.
- [x] Audit coefficient provenance and precision, method orders, dense output,
  estimators, stability policies, and controllers against the pinned upstream
  implementation.
- [x] Review the complete public API as a 1.0 compatibility commitment. Once
  that review is committed, create the immutable `api-freeze-v1` tag and use
  it as the pre-1.0 `cargo semver-checks --baseline-rev` baseline.

### Planning estimate

| Milestone | Estimated focused effort |
| --- | ---: |
| Remaining tableau migrations and legacy-loader removal | Completed |
| In-flight automatic switching and verification | Completed |
| Module decomposition and public API hardening | Completed |
| Numerical audit, downstream testing, and release preparation | Completed locally; remote CI pending |

The method-by-method numerical audit, downstream tests, documentation, and
local release gates are complete. A credible 1.0 release still requires a
green remote CI run from the reviewed commits, an explicit decision to remove
the publication lock and beta wording, and publication of the internal crates
in dependency order.

The pre-1.0 implementation namespaces named `general` and
`second_order::function` are no longer public. Import their reachable API from
the owning family instead: for example, use `solvers::explicit::Rk4` rather
than `solvers::explicit::general::Rk4`.

The pre-1.0 second-order solution contract is also normalized. Symplectic
interpolation now returns `(velocity, position)`, matching every other
second-order API. Shape-preserving `interpolate_array` now returns `Option`,
matching `interpolate`; use `try_interpolate_array` when an
`InterpolationError` is required. `SymplecticSolution::stats` exposes the full
`SolverStats`, while `rhs_evaluations` remains a convenience accessor.

Tableau inspection for `Qndf`, `Qbdf`, `Fbdf`, their fixed-order variants,
and `Mrab` now returns `tableau::TableauAccessError` instead of `SolveError`.
Downstream code can match `UnsupportedOrder` separately from embedded-resource
or family-invariant failures. Solver execution continues to report those
conditions through the corresponding `SolveError` variants.

Algorithm construction now rejects context-free invalid configuration instead
of silently normalizing it or deferring it to an unrelated `SolveError`.
Extrapolation `new` functions, `Anas5::new`, `Frk65::new`, and the three pRRK
constructors return `Result<_, ConfigurationError>`. `JVODE::with_biases`,
`JVODE::with_step_factors`, and `IRKC::with_eigenvalue_estimate` are likewise
fallible. Taylor order constructors, adaptive Radau order windows,
Gauss--Legendre stage counts, SBDF orders, and the configurable multirate
families now follow the same contract. Their configured values remain
inspectable through accessors.

All public solution interpolation paths reject non-finite output consistently
and use overflow-resistant linear fallback for finite extreme values. Invalid
saved-state indices, including `usize::MAX`, return `None` without panicking.
Direct calls to public `solve_validated` methods now retain ndarray scalar,
vector, or matrix shape just like the higher-level `solve` entry point.

Downstream implementations of `OdeAlgorithm` and `SecondOrderOdeAlgorithm`
can evaluate a problem through checked public methods and construct validated
saved trajectories with `Solution::from_saved` or
`SecondOrderSolution::from_saved`. Malformed custom results now produce
`SolutionConstructionError` through the corresponding solve error rather than
relying on private constructors or unchecked shape metadata. Custom drivers
that do not implement callback lifecycle behavior can inspect
`has_callbacks()` for both problem kinds and return the typed
`CallbacksUnsupported` error instead of silently dropping user policies. The
second-order drivers and workspaces are split into private fixed-step, RKN,
structural, and shared lifecycle modules without changing the public import
paths.

### Definition of done

Version 1.0 is ready only when all roadmap items above are complete, automatic
composites switch in flight without restart or replay, no solver consumes a
generic coefficient bank, supported state shapes behave consistently,
benchmark changes are understood, remote CI is green, and the packaged API and
documentation match tested behavior.

Before changing the package version, audit this definition against the current
tree and CI results item by item. Absence of a known failure is not completion:
each claim needs direct evidence from source, tests, package contents, benchmark
results, or remote CI as appropriate.

## Prepare

1. Update all three package versions and their exact internal dependency
   requirements.
2. Confirm the working tree is clean and the full CI matrix is green.
3. Run the latest-stable and Rust 1.85 formatting, lint, test, documentation,
   supply-chain, and package checks.
4. Run Julia compliance and the matched comparison benchmarks when numerical
   kernels change.
5. Review user-visible and breaking changes. For 1.0, remove beta wording and
   run `cargo semver-checks` against the `api-freeze-v1` tag. After this project
   has published its own prerelease, also compare against that registry release.

### API compatibility baseline

The renamed `differential-equations-rs` package has no published compatibility
baseline yet. Do not compare it with the unrelated package previously published
as `differential-equations`.

The reviewed API is committed and the immutable annotated `api-freeze-v1` tag
identifies that commit. Subsequent pre-1.0 changes must run
`cargo semver-checks` for each workspace package with that tag supplied through
`--baseline-rev`. Once a project-owned prerelease of each package exists on
crates.io, use the corresponding registry release as an additional baseline.
No project-owned registry release exists yet.

The Cargo-native package gate used by CI is:

```console
cargo package --locked --no-verify -p differential-equations-tableau-core
cargo package --locked --list -p differential-equations-tableau-macros
cargo package --locked --list -p differential-equations-rs
```

The two dependent crates can only perform registry-backed archive verification
after their exact internal versions have been published. Until then, their
Cargo-selected file lists are the deterministic package-content gate.

## Publish

1. Publish `differential-equations-tableau-core` with `--locked`.
2. Wait for that exact version to appear in the crates.io index, then publish
   `differential-equations-tableau-macros`.
3. Wait for the macro version, remove the publication lock from the main
   `differential-equations-rs` package in the reviewed release commit, and run
   its registry-backed dry run.
4. Publish the main crate, create a signed `v<version>` tag and GitHub release,
   and verify crates.io, docs.rs, and fresh downstream default/no-default builds.

Stop on any failure. Do not weaken exact internal versions or reuse a published
version number.
