[CmdletBinding()]
param(
    [ValidateSet('Manifest', 'Run', 'Collect', 'Stop', 'Delete')]
    [string]$Action = 'Manifest',
    [string]$Project = '',
    [string]$Zone = 'europe-west4-a',
    [string]$MachineType = 'n2-standard-4',
    [string]$NamePrefix = 'ode-bench',
    [string]$RepositoryUrl = 'https://github.com/Wouittone/differential-equations-rs.git',
    [string]$Ref = 'main',
    [string]$LocalResults = (Join-Path $PSScriptRoot 'results'),
    [int]$Samples = 5,
    [int]$Repetitions = 50,
    [int]$MaxParallel = 8,
    [int]$Cpu = 2,
    [int]$IntervalMs = 10,
    [string]$RustToolchain = '1.97.0',
    [string]$JuliaVersion = '1.12.6',
    [switch]$Spot,
    [switch]$KeepVms,
    [switch]$SkipCorrectnessTests,
    [switch]$ConfirmDelete,
    [string[]]$RunId
)

$ErrorActionPreference = 'Stop'
$cloudRoot = $PSScriptRoot
$algorithmPath = Join-Path $cloudRoot 'algorithms.txt'
New-Item -ItemType Directory -Force -Path $LocalResults | Out-Null

if ($Action -ne 'Manifest' -and [string]::IsNullOrWhiteSpace($Project)) {
    $Project = (& gcloud config get-value project 2>$null).Trim()
}
if ($Action -ne 'Manifest' -and ([string]::IsNullOrWhiteSpace($Project) -or $Project -eq '(unset)')) {
    throw 'Set -Project or configure a gcloud project before running this script.'
}

$algorithms = Get-Content -LiteralPath $algorithmPath |
    Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') } |
    ForEach-Object { $_.Trim() }
$manifest = foreach ($language in @('rust', 'julia')) {
    foreach ($mode in @('timing', 'allocation')) {
        foreach ($algorithm in $algorithms) {
            $safe = ($algorithm -replace '[^A-Za-z0-9_.-]', '_').ToLowerInvariant()
            [pscustomobject]@{
                run_id = "${language}_${mode}_${safe}"
                language = $language
                mode = $mode
                algorithm = $algorithm
            }
        }
    }
}
if (-not $SkipCorrectnessTests) {
    $manifest += [pscustomobject]@{
        run_id = 'correctness_tests'
        language = 'tests'
        mode = 'tests'
        algorithm = 'all'
    }
}

$manifestPath = Join-Path $LocalResults 'manifest.tsv'
$manifest | Export-Csv -LiteralPath $manifestPath -Delimiter "`t" -NoTypeInformation
Write-Host "Manifest: $manifestPath ($($manifest.Count) runs)"

if ($Action -eq 'Manifest') {
    return
}

$selected = if ($RunId) {
    $wanted = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $RunId | ForEach-Object { [void]$wanted.Add($_) }
    @($manifest | Where-Object { $wanted.Contains($_.run_id) })
} else {
    @($manifest)
}
if (-not $selected) {
    throw 'No matching runs. Use -RunId with an ID from the manifest.'
}

function Get-InstanceName([object]$row) {
    $name = "$NamePrefix-$($row.run_id)".ToLowerInvariant() -replace '[^a-z0-9-]', '-'
    if ($name.Length -gt 63) { $name = $name.Substring(0, 63).TrimEnd('-') }
    return $name
}

$cloudJobScript = {
    param($row, $project, $zone, $machineType, $namePrefix, $repositoryUrl, $ref,
        $localResults, $samples, $repetitions, $cpu, $intervalMs, $juliaVersion, $rustToolchain,
        $spot, $keepVms)

    $instance = "$namePrefix-$($row.run_id)".ToLowerInvariant() -replace '[^a-z0-9-]', '-'
    if ($instance.Length -gt 63) { $instance = $instance.Substring(0, 63).TrimEnd('-') }
    $caseDir = Join-Path $localResults $row.run_id
    New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
    $spotArgs = @()
    if ($spot) { $spotArgs = @('--provisioning-model=SPOT', '--instance-termination-action=DELETE') }
    $createArgs = @(
        'compute', 'instances', 'create', $instance, '--project', $project, '--zone', $zone,
        '--machine-type', $machineType, '--image-family', 'ubuntu-2404-lts-amd64',
        '--image-project', 'ubuntu-os-cloud', '--boot-disk-size', '50GB',
        '--boot-disk-type', 'pd-balanced', '--quiet'
    ) + $spotArgs
    $repoDir = "/tmp/ode-benchmark-$($row.run_id)"
    $remote = "set -euo pipefail; sudo apt-get update; sudo apt-get install -y --no-install-recommends git; rm -rf '$repoDir'; git clone --depth 1 --branch '$ref' '$repositoryUrl' '$repoDir'; cd '$repoDir'; export REPO_ROOT='$repoDir'; export JULIA_VERSION='$juliaVersion'; export RUST_TOOLCHAIN='$rustToolchain'; export RUSTFLAGS='-C target-cpu=native'; bash benchmarks/cloud/bootstrap_vm.sh;"
    if ($row.language -eq 'tests') {
        $remote += " bash benchmarks/cloud/run_tests.sh benchmarks/cloud/results;"
    } else {
        $remote += " bash benchmarks/cloud/run_case.sh --language '$($row.language)' --mode '$($row.mode)' --algorithm '$($row.algorithm)' --repetitions $repetitions --samples $samples --cpu $cpu --interval-ms $intervalMs --drop-caches 1 --out-dir benchmarks/cloud/results;"
    }
    $archive = "/tmp/$($row.run_id).tgz"
    $remote += " tar -czf '$archive' -C '$repoDir/benchmarks/cloud/results' ."

    try {
        & gcloud @createArgs
        if ($LASTEXITCODE -ne 0) { throw "instance creation failed: $instance" }
        & gcloud compute ssh $instance --project $project --zone $zone --quiet --command $remote
        if ($LASTEXITCODE -ne 0) { throw "remote run failed: $instance" }
        & gcloud compute scp "$instance`:$archive" (Join-Path $caseDir 'results.tgz') --project $project --zone $zone --quiet
        if ($LASTEXITCODE -ne 0) { throw "result collection failed: $instance" }
        tar -xzf (Join-Path $caseDir 'results.tgz') -C $caseDir
        [pscustomobject]@{ run_id = $row.run_id; instance = $instance; status = 'pass' } |
            Export-Csv -LiteralPath (Join-Path $caseDir 'cloud-status.csv') -NoTypeInformation
    } catch {
        $_ | Out-String | Set-Content -LiteralPath (Join-Path $caseDir 'cloud-error.txt')
        throw
    } finally {
        if (-not $keepVms) {
            & gcloud compute instances delete $instance --project $project --zone $zone --quiet 2>$null
        }
    }
}

if ($Action -eq 'Run') {
    $pending = [System.Collections.Generic.Queue[object]]::new()
    $selected | ForEach-Object { $pending.Enqueue($_) }
    $active = [System.Collections.ArrayList]::new()
    while ($pending.Count -gt 0 -or $active.Count -gt 0) {
        while ($pending.Count -gt 0 -and $active.Count -lt $MaxParallel) {
            $row = $pending.Dequeue()
            $job = Start-Job -ScriptBlock $cloudJobScript -ArgumentList @(
                $row, $Project, $Zone, $MachineType, $NamePrefix, $RepositoryUrl, $Ref,
                $LocalResults, $Samples, $Repetitions, $Cpu, $IntervalMs, $JuliaVersion, $RustToolchain,
                [bool]$Spot, [bool]$KeepVms
            )
            [void]$active.Add([pscustomobject]@{ Job = $job; RunId = $row.run_id })
            Write-Host "Started $($row.run_id) as job $($job.Id)"
        }
        foreach ($entry in @($active.ToArray())) {
            if ($entry.Job.State -in @('Completed', 'Failed', 'Stopped')) {
                Receive-Job $entry.Job -ErrorAction SilentlyContinue
                if ($entry.Job.State -ne 'Completed') { Write-Warning "$($entry.RunId) ended in $($entry.Job.State)" }
                Remove-Job $entry.Job -Force
                [void]$active.Remove($entry)
            }
        }
        if ($active.Count -gt 0) { Start-Sleep -Seconds 2 }
    }
    python (Join-Path $cloudRoot 'aggregate.py') $LocalResults
    return
}

foreach ($row in $selected) {
    $instance = Get-InstanceName $row
    if ($Action -eq 'Collect') {
        $caseDir = Join-Path $LocalResults $row.run_id
        New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
        & gcloud compute scp "$instance`:/tmp/$($row.run_id).tgz" (Join-Path $caseDir 'results.tgz') --project $Project --zone $Zone --quiet
        tar -xzf (Join-Path $caseDir 'results.tgz') -C $caseDir
    } elseif ($Action -eq 'Stop') {
        & gcloud compute instances stop $instance --project $Project --zone $Zone --quiet
    } elseif ($Action -eq 'Delete') {
        if (-not $ConfirmDelete) { throw 'Pass -ConfirmDelete to delete benchmark VMs.' }
        & gcloud compute instances delete $instance --project $Project --zone $Zone --quiet
    }
}
