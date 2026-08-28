# Release process

The workspace contains two crates that share a version. The main crate pins
`differential-equations-tableau-macros` exactly, so publication order is
mandatory.

## Intentional publication lock

The main `Cargo.toml` currently contains `publish = false`. This is the final
release lock: ordinary development and automated checks must leave it in place.
Removing it requires an explicit, reviewed release change after every item
below passes. A release is not ready while the lock remains.

## Prepare

1. Confirm the working tree is clean and CI is green on the intended commit.
2. Choose the version and update it in both package manifests and in the exact
   proc-macro dependency requirement.
3. Prepare tagged-release notes covering user-visible changes, migration
   guidance, and every breaking path.
4. For 1.0, remove prerelease/beta wording from the manifests, crate-level
   documentation, README, and release notes. Confirm GitHub private
   vulnerability reporting is enabled.
5. Verify latest stable Rust, the MSRV, both feature modes, Linux/Windows/macOS
   CI, doctests, missing public documentation, docs.rs warnings, license files,
   notices, `cargo-deny`, and the extracted-package build.
6. Run the full Rust suite, pinned Julia compliance suite, and the agreed
   benchmark comparison for a release that changes numerical kernels. Build
   the lightweight regression target with
   `cargo bench --locked --bench solver_performance --no-run` for every release.
7. Run `cargo semver-checks` against the previous published version once such
   a baseline exists. Review every allowed break explicitly.

From a source checkout, the package gate is:

```console
pwsh ./scripts/check_package_policy.ps1
```

The policy script verifies the proc-macro archive, checks both curated file
lists, and compiles an isolated copy of the root files selected by Cargo against
the extracted proc-macro archive. Before the matching proc-macro version exists
on crates.io, Cargo cannot assemble and verify the final root archive; that
ordinary registry-backed verification must wait for the first publication
step.

## Publish

1. In the reviewed release commit only, remove the main crate's
   `publish = false` lock.
2. Run `cargo publish --locked --dry-run -p differential-equations-tableau-macros`.
3. Publish `differential-equations-tableau-macros` with `--locked`.
4. Wait until the exact version is visible in the crates.io index.
5. Run `cargo publish --locked --dry-run -p differential-equations` so verification
   resolves the published macro crate rather than the workspace path.
6. Publish `differential-equations` with `--locked`.
7. Create a signed `v<version>` tag and GitHub release from the same commit,
   using the changelog section as release notes.
8. Confirm both crates render correctly on crates.io and docs.rs, and test a
   fresh downstream project with default and no-default features.

If any publication or verification step fails, stop. Do not publish the main
crate with a different macro version or weaken the exact dependency to bypass
the release order. If a published archive is unusable or contains a security
issue, stop further publication and yank the affected version; never reuse a
published version number.
