# Software supply-chain policy

This policy applies to Rust dependencies, build tooling, and GitHub Actions
used by this repository.

## Dependency requirements

- `Cargo.lock` is committed and CI consumes it with `--locked`.
- Dependencies must come from crates.io. Git dependencies, alternate
  registries, and wildcard version requirements are denied unless this policy
  is amended in the same reviewed change.
- Dependency licenses must satisfy the allowlist in `deny.toml`. The current
  allowlist is MIT, Apache-2.0, and Unicode-3.0.
- Known RustSec advisories, yanked packages, duplicate crate versions, unknown
  registries, and unknown Git sources fail the supply-chain CI job.
- Direct dependencies should be minimal, actively maintained, and used through
  the narrowest practical feature set.

## Review and updates

Dependabot opens weekly grouped updates for Cargo dependencies and GitHub
Actions. Every update must pass the normal tests and `cargo deny check` before
merge. Major updates require review of release notes, MSRV impact, enabled
features, transitive dependency changes, and license changes.

GitHub Actions must be pinned to a full commit SHA. The adjacent version
comment records the reviewed release or moving ref. Dependabot is responsible
for proposing later SHAs.

## Exceptions

Exceptions must be narrow and recorded in `deny.toml` with the affected crate
or advisory ID, a concrete reason, and a removal condition. An advisory may be
temporarily ignored only when exposure has been assessed and no safe upgrade
exists. Exception changes require the same review as source-code changes.

## Local verification

Install the cargo-deny release used by CI and run:

```console
cargo deny check
cargo check --locked --all-targets --no-default-features
cargo test --locked --all-targets --all-features
pwsh ./scripts/check_package_policy.ps1
```

The package-policy script verifies the workspace SPDX expressions, the copied
proc-macro license texts, the license and notice files shipped in each crate,
and the exclusion of development-only trees from the main package. This policy
complements code review and testing; it does not make dependency or licensing
decisions automatic.
