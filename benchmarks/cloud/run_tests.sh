#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out_dir=${1:-$root/benchmarks/cloud/results}
mkdir -p "$out_dir"

{
    printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_revision\t%s\n' "$(git -C "$root" rev-parse HEAD)"
    printf 'hostname\t%s\n' "$(hostname)"
    cargo test --locked --release
    julia --startup-file=no --project="$root/tests/julia" "$root/tests/julia/pinned_environment.jl" --check
    julia --startup-file=no --project="$root/tests/julia" "$root/tests/julia/runtests.jl"
    printf 'status\tpass\n'
} 2>&1 | tee "$out_dir/correctness-tests.log"
