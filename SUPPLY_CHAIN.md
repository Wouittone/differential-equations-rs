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
- Direct dependencies must be actively maintained, license-compatible, MSRV
  compatible, minimal, and used through the narrowest practical feature set.
  New parser/evaluator dependencies also require tests for the exact grammar
  and numeric semantics used by coefficient resources.

## Review and updates

Dependabot opens weekly grouped updates for Cargo dependencies and GitHub
Actions. Every update must pass the normal tests and `cargo deny check` before
merge. Major updates require review of release notes, MSRV impact, enabled
features, transitive dependency changes, maintenance activity, and licenses.

GitHub Actions must be pinned to a full commit SHA. The adjacent version
comment records the reviewed release or moving ref. Dependabot is responsible
for proposing later SHAs.

## Exceptions

Exceptions must be narrow and recorded in `deny.toml` with the affected crate
or advisory ID, a concrete reason, and a removal condition. An advisory may be
temporarily ignored only when exposure has been assessed and no safe upgrade
exists. Exception changes require the same review as source-code changes.

## Local verification

Install the `cargo-deny` release used by CI and run:

```console
cargo deny check
cargo +1.85 check --locked --all-targets --no-default-features
cargo +1.85 test --locked --workspace --all-features
pwsh ./scripts/check_package_policy.ps1
```

The package-policy script verifies workspace SPDX expressions, copied license
texts, required notices and documentation, extracted-package builds, and the
exclusion of development-only trees from published archives. This policy
complements code review and testing; it does not make dependency or licensing
decisions automatic.
