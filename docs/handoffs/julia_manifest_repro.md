# Julia manifest reproducibility handoff

## Decision

Track `tests/julia/Manifest.toml` and stop ignoring it. The compliance project
uses repository sources for OrdinaryDiffEq monorepo packages, so the manifest
is the durable record of each package's repository revision and subdirectory.
Without the tracked manifest, `pinned_environment.jl --check` necessarily fails
in a fresh clone or worktree before it can inspect any pins.

## Artifact audit

The tracked artifact was generated with Julia 1.12.6, uses manifest format 2.0,
and is stored as Git blob `9dcd7f644b3b66f80cbb901c485fdd61749c5edf`.
This blob identifier remains stable when a checkout's Git configuration converts
line endings between LF and CRLF. The manifest contains 143 resolved dependency
entries. Its `project_hash` is
`3cd13161f691aa9045c7d3d58d109f8b51b331d3`, which matches the hash calculated
from the current `tests/julia/Project.toml` by Julia's package manager.

The audit found:

- no `path` dependency entries or machine-local path strings;
- no mutable repository revisions;
- no repository sources other than the 13 expected `OrdinaryDiffEq*` packages;
- exactly 13 `OrdinaryDiffEq*` packages, all sourced from
  `https://github.com/SciML/OrdinaryDiffEq.jl.git` at revision
  `211142263781255a9aa2f910f6760b9f18ec29c8` and the matching
  `lib/<package-name>` subdirectory;
- a 40-character `git-tree-sha1` for every repository-sourced package; and
- no credential-, token-, or secret-like content.

Registered transitive dependencies remain locked by their normal versions and
tree hashes. No dependency has a Julia `pinned = true` marker.

## Verification

From a clean checkout, run:

```powershell
julia --project=tests/julia tests/julia/pinned_environment.jl --check
git diff --check
```

The first command is read-only and does not require package instantiation. Full
Julia compliance tests still require the locked dependencies to be available
or instantiated in the active Julia depot.
