#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential curl git jq numactl procps python3 time util-linux

if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rust_toolchain=${RUST_TOOLCHAIN:-1.97.0}
rustup toolchain install "$rust_toolchain" --profile minimal
rustup default "$rust_toolchain"

julia_version=${JULIA_VERSION:-1.12.6}
julia_major_minor=${julia_version%.*}
julia_archive="julia-${julia_version}-linux-x86_64.tar.gz"
julia_url="https://julialang-s3.julialang.org/bin/linux/x64/${julia_major_minor}/${julia_archive}"
if ! command -v julia >/dev/null 2>&1 || [[ "$(julia --startup-file=no -e 'print(VERSION)')" != "$julia_version" ]]; then
    work_dir=$(mktemp -d)
    trap 'rm -rf "$work_dir"' EXIT
    curl --fail --location --retry 3 --output "$work_dir/$julia_archive" "$julia_url"
    curl --fail --location --retry 3 --output "$work_dir/$julia_archive.sha256" "$julia_url.sha256"
    (cd "$work_dir" && sha256sum -c "$julia_archive.sha256")
    sudo rm -rf "/opt/julia-${julia_version}"
    sudo tar -xzf "$work_dir/$julia_archive" -C /opt
    sudo ln -sfn "/opt/julia-${julia_version}" /opt/julia
    sudo ln -sfn /opt/julia/bin/julia /usr/local/bin/julia
fi

if [[ -d /sys/devices/system/cpu/cpu0/cpufreq ]]; then
    sudo bash -c 'for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$governor" 2>/dev/null || true; done'
fi

cd "${REPO_ROOT:-$PWD}"
git submodule update --init --recursive
cargo --version
julia --version
julia --startup-file=no --project=tests/julia tests/julia/pinned_environment.jl --check
julia --startup-file=no --project=tests/julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
