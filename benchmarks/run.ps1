param(
    [ValidateRange(1, 10000)]
    [int]$Repetitions = 20
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$results = Join-Path $PSScriptRoot 'results'
New-Item -ItemType Directory -Force -Path $results | Out-Null

$rustTimingOutput = & cargo run --quiet --release --manifest-path (Join-Path $repository 'Cargo.toml') --example benchmark_matrix -- $Repetitions
if ($LASTEXITCODE -ne 0) {
    throw 'Rust timing benchmark failed'
}
$rustAllocationOutput = & cargo run --quiet --release --features allocation-metrics --manifest-path (Join-Path $repository 'Cargo.toml') --example benchmark_matrix -- $Repetitions
if ($LASTEXITCODE -ne 0) {
    throw 'Rust allocation benchmark failed'
}
$rustTimingPath = Join-Path $results 'rust-timing.csv'
$rustAllocationPath = Join-Path $results 'rust-allocation.csv'
$rustPath = Join-Path $results 'rust.csv'
$rustTimingOutput | Set-Content -LiteralPath $rustTimingPath
$rustAllocationOutput | Set-Content -LiteralPath $rustAllocationPath
$rustTiming = Import-Csv -LiteralPath $rustTimingPath
$rustAllocation = Import-Csv -LiteralPath $rustAllocationPath
$rust = foreach ($timingRow in $rustTiming) {
    $allocationRow = $rustAllocation | Where-Object algorithm -EQ $timingRow.algorithm
    [pscustomobject]@{
        language = $timingRow.language
        algorithm = $timingRow.algorithm
        dimension = [int]$timingRow.dimension
        nanoseconds_per_solve = [double]$timingRow.nanoseconds_per_solve
        bytes_allocated_per_solve = [double]$allocationRow.bytes_allocated_per_solve
        allocations_per_solve = [double]$allocationRow.allocations_per_solve
        rhs_evaluations_per_solve = [double]$timingRow.rhs_evaluations_per_solve
        checksum = [double]$timingRow.checksum
    }
}
$rust | Export-Csv -LiteralPath $rustPath -NoTypeInformation

$juliaProject = Join-Path $repository 'tests/julia'
$juliaTimingOutput = & julia --startup-file=no "--project=$juliaProject" (Join-Path $PSScriptRoot 'julia_matrix.jl') --repetitions $Repetitions --mode timing
if ($LASTEXITCODE -ne 0) {
    throw 'Julia timing benchmark failed'
}
$juliaAllocationOutput = & julia --startup-file=no "--project=$juliaProject" (Join-Path $PSScriptRoot 'julia_matrix.jl') --repetitions $Repetitions --mode allocation
if ($LASTEXITCODE -ne 0) {
    throw 'Julia allocation benchmark failed'
}
$juliaTimingPath = Join-Path $results 'julia-timing.csv'
$juliaAllocationPath = Join-Path $results 'julia-allocation.csv'
$juliaPath = Join-Path $results 'julia.csv'
$juliaTimingOutput | Set-Content -LiteralPath $juliaTimingPath
$juliaAllocationOutput | Set-Content -LiteralPath $juliaAllocationPath
$juliaTiming = Import-Csv -LiteralPath $juliaTimingPath
$juliaAllocation = Import-Csv -LiteralPath $juliaAllocationPath
$julia = foreach ($timingRow in $juliaTiming) {
    $allocationRow = $juliaAllocation | Where-Object algorithm -EQ $timingRow.algorithm
    [pscustomobject]@{
        language = $timingRow.language
        algorithm = $timingRow.algorithm
        dimension = [int]$timingRow.dimension
        nanoseconds_per_solve = [double]$timingRow.nanoseconds_per_solve
        bytes_allocated_per_solve = [double]$allocationRow.bytes_allocated_per_solve
        allocations_per_solve = [double]$allocationRow.allocations_per_solve
        rhs_evaluations_per_solve = [double]$timingRow.rhs_evaluations_per_solve
        checksum = [double]$timingRow.checksum
    }
}
$julia | Export-Csv -LiteralPath $juliaPath -NoTypeInformation

$invariant = [Globalization.CultureInfo]::InvariantCulture
$comparison = foreach ($rustRow in $rust) {
    $juliaRow = $julia | Where-Object algorithm -EQ $rustRow.algorithm
    $timeRatio = [double]$rustRow.nanoseconds_per_solve / [double]$juliaRow.nanoseconds_per_solve
    $byteRatio = [double]$rustRow.bytes_allocated_per_solve / [double]$juliaRow.bytes_allocated_per_solve
    [pscustomobject]@{
        algorithm = $rustRow.algorithm
        dimension = [int]$rustRow.dimension
        rust_ns = [double]$rustRow.nanoseconds_per_solve
        julia_ns = [double]$juliaRow.nanoseconds_per_solve
        rust_over_julia_time = $timeRatio.ToString('F3', $invariant)
        rust_bytes = [double]$rustRow.bytes_allocated_per_solve
        julia_bytes = [double]$juliaRow.bytes_allocated_per_solve
        rust_over_julia_bytes = $byteRatio.ToString('F3', $invariant)
        rust_rhs = [double]$rustRow.rhs_evaluations_per_solve
        julia_rhs = [double]$juliaRow.rhs_evaluations_per_solve
    }
}

$comparisonPath = Join-Path $results 'comparison.csv'
$comparison | Export-Csv -LiteralPath $comparisonPath -NoTypeInformation
$comparison | Format-Table algorithm, dimension, rust_ns, julia_ns, rust_over_julia_time, rust_bytes, julia_bytes, rust_over_julia_bytes, rust_rhs, julia_rhs -AutoSize
