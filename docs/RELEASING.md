# Release process

The workspace contains three versioned crates: `tableau-core`,
`tableau-macros`, and the main `differential-equations` crate. Internal
dependencies use exact versions, so publication order is mandatory.

The main manifest currently contains `publish = false`. Removing that lock
requires an explicit reviewed release change.

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
