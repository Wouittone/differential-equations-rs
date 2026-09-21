# Release process

The workspace publishes three crates:

1. `differential-equations-tableau-core`
2. `differential-equations-tableau-macros`
3. `differential-equations-rs`

The macro crate depends on the core crate, and the main crate depends on both.
Those internal requirements are exact, so releases must use one shared version
and be published in that order.

## Versioning and compatibility

Published releases follow semantic versioning. Patch and minor releases must
remain compatible with the latest published stable API; intentional breaking
changes require a new major version and migration notes.

The immutable `v<version>` Git tag for the latest release is the compatibility
baseline for `cargo-semver-checks`. The procedural-macro crate has no Rust
library surface that tool can inspect, so its compatibility is covered by the
downstream compile tests.

## Validate the release commit

Run the same Rust gates used by CI with the locked dependency graph:

```console
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets --all-features
cargo test --locked --workspace --all-targets --no-default-features
cargo test --locked --workspace --doc --all-features
cargo doc --locked --workspace --no-deps --all-features
cargo deny check
```

Run `cargo-semver-checks` against the latest stable tag for both library
crates. When numerical kernels, coefficients, or Julia reference fixtures have
changed, also run the pinned SciML compliance suite and the matched comparison
matrix described in [BENCHMARKING.md](BENCHMARKING.md).

Inspect the exact package contents before publishing:

```console
cargo package --locked --list -p differential-equations-tableau-core
cargo package --locked --list -p differential-equations-tableau-macros
cargo package --locked --list -p differential-equations-rs
```

The archives must include both license files and only the source, resources,
examples, benchmarks, and documentation required by downstream users.

## Publish

Publish from a clean `main` checkout at the release commit:

```console
cargo publish --locked -p differential-equations-tableau-core
cargo publish --locked -p differential-equations-tableau-macros
cargo publish --locked -p differential-equations-rs
```

Wait for each dependency version to appear in the crates.io index before
publishing the crate that depends on it. Stop on any failure: a published
version is immutable and must never be reused.

## Tag and announce

After all three crates are available from crates.io:

1. Create the signed `v<version>` tag on the exact published commit and push it.
2. Create the GitHub release from that tag with concise user-visible changes
   and any migration notes.
3. Verify the crates.io pages, docs.rs builds, license metadata, and a fresh
   downstream build using both default and no-default features.

Keep release notes focused on observable API, behavior, correctness, and
performance changes. Repository maintenance details belong in the commit
history unless they affect users.
