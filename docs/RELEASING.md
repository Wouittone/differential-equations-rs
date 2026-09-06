# Release process

The workspace contains three versioned crates: `tableau-core`,
`tableau-macros`, and the main `differential-equations` crate. Internal
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

Work through the sections in dependency order: finish the resource model and
remove the legacy loader first, implement in-flight automatic switching on the
resulting stable solver boundaries, then complete module decomposition and the
public-API review. A checkbox represents committed repository state, not work
in progress. Close one only when its implementation, focused regression tests,
public documentation, and applicable CI gates land together; record follow-up
work as another unchecked item rather than weakening its completion criteria.

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
- [ ] Add the upstream embedded estimators and genuine adaptive/FSAL lifecycle
  for the RDPK 3S-plus and CKLL register-pipeline families. Preserve their
  method-specific controller metadata without reintroducing Rust constants.
- [ ] Replace the combined ROCK/SERK/ESERK bank with typed stabilized-method
  resources that do not parse or retain the entire catalogue for one selected
  method and degree.
- [ ] Remove `define_tableau_data_from_file!`, every legacy `coefficient_data`
  module, and their obsolete combined resource files after the final consumer
  is migrated.
- [ ] Keep resource JSON consistently styled with stable field ordering,
  compact scalar vectors, and one matrix row per line without adding a
  formatting build step.
- [ ] Keep the schemas and tableau resource guide synchronized with the actual
  migration status.

### Implement genuine in-flight automatic switching

Automatic algorithms must switch from non-stiff to stiff integration at the
current accepted state. The existing behavior that restarts from the initial
condition after the explicit algorithm fails is not sufficient for 1.0.

- [ ] Introduce an internal handoff state for the accepted time and state,
  derivative, proposed step, direction, controller context, pending time stops,
  callbacks, saving and dense-output state, and cumulative statistics.
- [ ] Separate driver-owned integration state from algorithm caches so a stiff
  cache can be initialized lazily at the switching point.
- [ ] Preserve accepted evaluations, callback effects, saved points, mutable
  parameter side effects, exact stops, termination, interpolation, and error
  semantics without replaying the initial interval.
- [ ] Add a documented stiffness detector that can switch before fatal explicit
  failure using stability-limited steps, rejection history, and available
  derivative or Jacobian estimates.
- [ ] Add hysteresis and minimum-residence rules to avoid repeated switching.
  Non-stiff-to-stiff handoff is mandatory; controlled stiff-to-non-stiff
  handoff should be supported by automatic pairs that can provide it.
- [ ] Make detector thresholds inspectable and configurable with conservative
  defaults, and expose switch counts and active-algorithm information in solver
  statistics.
- [ ] Return typed errors for incompatible handoff pairs; never silently fall
  back to a full restart.
- [ ] Apply the shared switching mechanism to `AutoDP5` and the other automatic
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

- [ ] Split oversized files along stable responsibilities, especially the
  second-order API and drivers, extended Rosenbrock methods, low-storage RK,
  and stabilized methods.
- [ ] Keep algorithms under `solvers::<family>` while driver, workspace,
  controller, and resource implementation details remain private unless they
  are deliberate extension surfaces.
- [ ] Standardize constructors, options, typed errors, tableau inspection,
  solutions, and interpolation across solver families.
- [ ] Complete rustdoc examples and links for every public type and supported
  extension point.
- [ ] Add downstream smoke crates for default and no-default features, a
  renamed dependency, a file-defined tableau, and scalar/vector/matrix states.
- [ ] Audit coefficient provenance and precision, method orders, dense output,
  estimators, stability policies, and controllers against the pinned upstream
  implementation.
- [ ] Review the complete public API as a 1.0 compatibility commitment and run
  `cargo semver-checks` against the latest published prerelease when available.

### Planning estimate

| Milestone | Estimated focused effort |
| --- | ---: |
| Remaining tableau migrations and legacy-loader removal | 8–13 development days |
| In-flight automatic switching and verification | 5–10 development days |
| Module decomposition and public API hardening | 5–10 development days |
| Numerical audit, downstream testing, and release preparation | 5–10 development days |

A credible 1.0 release is approximately four to seven focused weeks away. The
stabilized resource migration, automatic-switching architecture, and numerical
audit carry the most uncertainty. Parallel review can reduce calendar time,
but every integrated release gate below must still pass.

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
   run `cargo semver-checks` against the latest published release.

The Cargo-native package gate used by CI is:

```console
cargo package --locked --no-verify -p differential-equations-tableau-core
cargo package --locked --list -p differential-equations-tableau-macros
cargo package --locked --list -p differential-equations
```

The two dependent crates can only perform registry-backed archive verification
after their exact internal versions have been published. Until then, their
Cargo-selected file lists are the deterministic package-content gate.

## Publish

1. Publish `differential-equations-tableau-core` with `--locked`.
2. Wait for that exact version to appear in the crates.io index, then publish
   `differential-equations-tableau-macros`.
3. Wait for the macro version, remove the main crate's publication lock in the
   reviewed release commit, and run its registry-backed dry run.
4. Publish the main crate, create a signed `v<version>` tag and GitHub release,
   and verify crates.io, docs.rs, and fresh downstream default/no-default builds.

Stop on any failure. Do not weaken exact internal versions or reuse a published
version number.
