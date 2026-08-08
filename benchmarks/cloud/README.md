# Cloud benchmark harness

This harness runs the currently implemented Rust and Julia benchmark lanes on
Google Compute Engine. It deliberately uses one disposable VM per
algorithm/mode pair, so independent cases can run concurrently without
cross-case CPU, allocator, or page-cache interference. By default the manifest
contains 100 benchmark VMs (25 algorithms × 2 languages × timing/allocation)
plus one correctness-test VM.

No cloud command is run by the repository scripts until you invoke them.

## Prerequisites

Install the Google Cloud CLI, PowerShell 7, and Python 3 locally, authenticate,
select a project, and push the benchmark commit to a reachable Git ref:

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
git push origin YOUR_BENCHMARK_REF
```

The default VM is an `n2-standard-4` (4 vCPU, 16 GiB) in
`europe-west4-a`, using Ubuntu 24.04. This is a moderate-cost general-purpose
VM with a stable CPU shape. Add `-Spot` to trade interruption risk for lower
cost. The launcher never requests GPUs or persistent disks beyond the boot
disk.

## Generate and run the matrix

First generate the local manifest. This performs no cloud operation:

```powershell
pwsh benchmarks/cloud/gcloud-matrix.ps1 `
  -Action Manifest `
  -Project YOUR_PROJECT_ID
```

Run all cases, with at most eight VMs active at once. Each VM installs pinned
Julia 1.12.6, Rust 1.97.0, and dependencies. The dedicated correctness VM
runs the full test suites; benchmark VMs build and run only their assigned
case, then download its result archive before being deleted:

```powershell
pwsh benchmarks/cloud/gcloud-matrix.ps1 `
  -Action Run `
  -Project YOUR_PROJECT_ID `
  -Ref YOUR_BENCHMARK_REF `
  -Zone europe-west4-a `
  -MachineType n2-standard-4 `
  -MaxParallel 8 `
  -Samples 5 `
  -Repetitions 50
```

For a cheaper smoke run, select individual IDs from
`benchmarks/cloud/results/manifest.tsv`:

```powershell
pwsh benchmarks/cloud/gcloud-matrix.ps1 `
  -Action Run -Project YOUR_PROJECT_ID -Ref YOUR_BENCHMARK_REF `
  -RunId rust_timing_tsit5,julia_timing_tsit5 -Samples 3 -Repetitions 20 -Spot
```

`-KeepVms` preserves VMs for inspection and prevents automatic deletion. If a
run is interrupted, collect archives later with `-Action Collect`; stop or
delete retained instances with `-Action Stop` or
`-Action Delete -ConfirmDelete`.

The launcher collects everything under the local
`benchmarks/cloud/results/` directory. The important artifacts are:

- `results.tsv` and `results.jsonl`: one merged record per measured sample;
- `*.metrics.tsv`: wall/user/system time, `/usr/bin/time` maximum RSS,
  sampled peak RSS, sampled average RSS, and sampling details;
- `*.stdout` and `*.stderr`: untouched engine output;
- `correctness-tests.log`: the full Rust and Julia test logs;
- `manifest.tsv`: the exact case list used for the run.

## What is measured

The Rust driver has separate timing and `allocation-metrics` builds. Timing
runs do not install the atomic counting allocator; allocation runs report
cumulative allocated bytes and allocation count from `stats_alloc`. Julia uses
separate timing (`@elapsed`) and allocation (`@timed`) modes. Both languages
perform a warm-up solve before measured repetitions.

The external wrapper samples `/proc/<pid>/status` every 10 ms and also records
GNU `time` maximum RSS. Sampled average RSS is therefore an interval-sampled
estimate, while sampled peak and `time` maximum RSS are independent peak
measurements. Compilation, dependency installation, warm-up, result copying,
and cache cleanup are outside the measured command.

Before each sample the VM attempts to set the CPU governor to `performance`,
pins the process to CPU 2 with `taskset`, and (by default) executes
`sync; echo 3 > /proc/sys/vm/drop_caches`. The latter requires the VM's normal
passwordless `sudo`; pass `--drop-caches 0` directly to `run_case.sh` if a
kernel image does not permit it. The benchmark metadata records the git
revision, hostname, VM-side tool versions, and dirty-tree state.

The current harness covers the 25 solver configurations already implemented
by `examples/benchmark_matrix.rs` and `benchmarks/julia_matrix.jl`. Future
language adapters can reuse `measure.sh` and emit the same CSV header; the
collector will preserve their engine-specific columns automatically.

Rust is compiled with `-C target-cpu=native` on each VM, and the exact
compiler, CPU model, governor, kernel, and git revision are retained with the
case artifacts so results are only compared across matched environments.

## Explicit single-VM CLI flow

The PowerShell dispatcher is just a parallel wrapper around these ordinary
`gcloud` operations. For a manual smoke run, use one VM and one case:

```powershell
gcloud compute instances create ode-bench-smoke `
  --project YOUR_PROJECT_ID --zone europe-west4-a `
  --machine-type n2-standard-4 `
  --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud `
  --boot-disk-size 50GB --boot-disk-type pd-balanced

gcloud compute ssh ode-bench-smoke --project YOUR_PROJECT_ID --zone europe-west4-a `
  --command "git clone --depth 1 --branch YOUR_BENCHMARK_REF https://github.com/Wouittone/differential-equations-rs.git /tmp/ode-benchmark; cd /tmp/ode-benchmark; export REPO_ROOT=/tmp/ode-benchmark; bash benchmarks/cloud/bootstrap_vm.sh; bash benchmarks/cloud/run_case.sh --language rust --mode timing --algorithm Tsit5 --samples 5 --repetitions 50"

gcloud compute scp --recurse `
  ode-bench-smoke:/tmp/ode-benchmark/benchmarks/cloud/results `
  .\benchmarks\cloud\results\smoke `
  --project YOUR_PROJECT_ID --zone europe-west4-a

gcloud compute instances delete ode-bench-smoke `
  --project YOUR_PROJECT_ID --zone europe-west4-a --quiet
```
