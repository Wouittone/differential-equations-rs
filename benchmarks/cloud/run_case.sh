#!/usr/bin/env bash
set -euo pipefail

language=""
mode="timing"
algorithm=""
repetitions=50
samples=5
cpu=2
interval_ms=10
drop_caches=1
out_dir=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --language) language=$2; shift 2 ;;
        --mode) mode=$2; shift 2 ;;
        --algorithm) algorithm=$2; shift 2 ;;
        --repetitions) repetitions=$2; shift 2 ;;
        --samples) samples=$2; shift 2 ;;
        --cpu) cpu=$2; shift 2 ;;
        --interval-ms) interval_ms=$2; shift 2 ;;
        --drop-caches) drop_caches=$2; shift 2 ;;
        --out-dir) out_dir=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$language" || -z "$algorithm" ]]; then
    echo "--language and --algorithm are required" >&2
    exit 2
fi
if [[ "$language" != rust && "$language" != julia ]]; then
    echo "language must be rust or julia" >&2
    exit 2
fi
if [[ "$mode" != timing && "$mode" != allocation ]]; then
    echo "mode must be timing or allocation" >&2
    exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out_dir=${out_dir:-$root/benchmarks/cloud/results}
mkdir -p "$out_dir"
safe_algorithm=${algorithm//[^A-Za-z0-9_.-]/_}
case_id="${language}_${mode}_${safe_algorithm}"

printf 'case_id\tlanguage\tmode\talgorithm\trepetitions\tsamples\tgit_revision\tgit_dirty\thostname\tstarted_utc\n' > "$out_dir/$case_id.meta.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$language" "$mode" "$algorithm" "$repetitions" "$samples" \
    "$(git rev-parse HEAD)" "$(if git diff --quiet; then echo false; else echo true; fi)" \
    "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_dir/$case_id.meta.tsv"
{
    printf 'uname\t%s\n' "$(uname -a)"
    printf 'rustc\t%s\n' "$(rustc --version 2>/dev/null || true)"
    printf 'julia\t%s\n' "$(julia --startup-file=no --version 2>/dev/null || true)"
    printf 'cpu_model\t%s\n' "$(awk -F: '/^model name/ {gsub(/^ +/, "", $2); print $2; exit}' /proc/cpuinfo)"
    printf 'cpu_governor\t%s\n' "$(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor 2>/dev/null || true)"
} > "$out_dir/$case_id.host.tsv"

drop_page_cache() {
    if [[ "$drop_caches" == 1 ]]; then
        sync
        echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    fi
}

run_command=()
if [[ "$language" == rust ]]; then
    if [[ "$mode" == allocation ]]; then
        cargo build --quiet --locked --release --features allocation-metrics --example benchmark_matrix
        cp target/release/examples/benchmark_matrix "$out_dir/$case_id.driver"
    else
        cargo build --quiet --locked --release --example benchmark_matrix
        cp target/release/examples/benchmark_matrix "$out_dir/$case_id.driver"
    fi
    run_command=("$out_dir/$case_id.driver" --repetitions "$repetitions" --algorithm "$algorithm")
else
    julia --startup-file=no --project=tests/julia -e 'using Pkg; Pkg.instantiate()'
    run_command=(julia --startup-file=no --project=tests/julia benchmarks/julia_matrix.jl
        --repetitions "$repetitions" --algorithm "$algorithm" --mode "$mode")
fi

for sample in $(seq -w 1 "$samples"); do
    drop_page_cache
    bash "$root/benchmarks/cloud/measure.sh" "$out_dir" "${case_id}_sample${sample}" "$cpu" "$interval_ms" "${run_command[@]}"
done

rm -f "$out_dir/$case_id.driver"
python3 "$root/benchmarks/cloud/aggregate.py" "$out_dir"
