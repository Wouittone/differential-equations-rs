#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "usage: measure.sh OUT_DIR CASE_ID CPU SAMPLE_INTERVAL_MS COMMAND [ARGS...]" >&2
    exit 2
fi

out_dir=$1
case_id=$2
cpu=$3
interval_ms=$4
shift 4
mkdir -p "$out_dir"

stdout_path="$out_dir/$case_id.stdout"
stderr_path="$out_dir/$case_id.stderr"
time_path="$out_dir/$case_id.time"
metrics_path="$out_dir/$case_id.metrics.tsv"
command_path="$out_dir/$case_id.command"
printf '%q ' "$@" > "$command_path"
printf '\n' >> "$command_path"

if command -v taskset >/dev/null 2>&1; then
    command_prefix=(taskset --cpu-list "$cpu")
else
    command_prefix=()
fi

(
    exec /usr/bin/time -f '%e\t%U\t%S\t%M\t%x' -o "$time_path" \
        "${command_prefix[@]}" "$@"
) >"$stdout_path" 2>"$stderr_path" &
pid=$!

rss_sum=0
rss_peak=0
rss_samples=0
sample_seconds="0.$(printf '%03d' "$interval_ms")"
while kill -0 "$pid" 2>/dev/null; do
    rss=0
    if [[ -r "/proc/$pid/status" ]]; then
        rss=$(awk '/^VmRSS:/ {print $2; exit}' "/proc/$pid/status")
        rss=${rss:-0}
    fi
    rss_sum=$((rss_sum + rss))
    if (( rss > rss_peak )); then
        rss_peak=$rss
    fi
    ((rss_samples += 1))
    sleep "$sample_seconds"
done

set +e
wait "$pid"
exit_code=$?
set -e

if [[ -r "/proc/$pid/status" ]]; then
    rss=$(awk '/^VmRSS:/ {print $2; exit}' "/proc/$pid/status")
    rss=${rss:-0}
    if (( rss > rss_peak )); then
        rss_peak=$rss
    fi
    ((rss_samples += 1))
    rss_sum=$((rss_sum + rss))
fi

wall_seconds=NaN
user_seconds=NaN
sys_seconds=NaN
max_rss_kib=NaN
if [[ -s "$time_path" ]]; then
    read -r wall_seconds user_seconds sys_seconds max_rss_kib _ < "$time_path"
fi

avg_rss=0
if (( rss_samples > 0 )); then
    avg_rss=$((rss_sum / rss_samples))
fi

printf 'case_id\texit_code\twall_seconds\tuser_seconds\tsys_seconds\tmax_rss_kib\tsampled_peak_rss_kib\taverage_rss_kib\trss_samples\tsample_interval_ms\thostname\n' > "$metrics_path"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$exit_code" "$wall_seconds" "$user_seconds" "$sys_seconds" \
    "$max_rss_kib" "$rss_peak" "$avg_rss" "$rss_samples" "$interval_ms" "$(hostname)" >> "$metrics_path"

exit "$exit_code"
