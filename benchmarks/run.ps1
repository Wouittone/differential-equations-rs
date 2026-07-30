param(
    [ValidateRange(1, 10000)]
    [int]$Repetitions = 20
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$results = Join-Path $PSScriptRoot 'results'
New-Item -ItemType Directory -Force -Path $results | Out-Null

$rustOutput = & cargo run --quiet --release --manifest-path (Join-Path $repository 'Cargo.toml') --example benchmark_matrix -- $Repetitions
if ($LASTEXITCODE -ne 0) {
    throw 'Rust benchmark failed'
}
$rustPath = Join-Path $results 'rust.csv'
$rustOutput | Set-Content -LiteralPath $rustPath

$juliaProject = Join-Path $repository 'tests/julia'
$juliaOutput = & julia --startup-file=no "--project=$juliaProject" (Join-Path $PSScriptRoot 'julia_matrix.jl') $Repetitions
if ($LASTEXITCODE -ne 0) {
    throw 'Julia benchmark failed'
}
$juliaPath = Join-Path $results 'julia.csv'
$juliaOutput | Set-Content -LiteralPath $juliaPath

$rust = Import-Csv -LiteralPath $rustPath
$julia = Import-Csv -LiteralPath $juliaPath
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
