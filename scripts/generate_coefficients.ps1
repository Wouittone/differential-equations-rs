[CmdletBinding()]
param([switch]$Check)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root 'docs/coefficients_manifest.txt'
$content = @"
schema-version=1
source-revision=211142263781255a9aa2f910f6760b9f18ec29c8
method=AB3|family=multistep|order=3|variable-step=false
method=RK4|family=explicit|order=4|embedded-order=none
method=VelocityVerlet|family=symplectic|order=2|embedded-order=none
"@ -replace "`r`n", "`n"

if ($Check) {
    if (-not (Test-Path -LiteralPath $output)) { throw "Missing generated coefficient manifest: $output" }
    $actual = [IO.File]::ReadAllText($output) -replace "`r`n", "`n"
    if ($actual -cne $content) { throw "Coefficient manifest is stale; rerun generate_coefficients.ps1" }
    Write-Output "Verified deterministic coefficient manifest: $output"
    exit 0
}

[IO.File]::WriteAllText($output, $content, [Text.UTF8Encoding]::new($false))
Write-Output "Wrote deterministic coefficient manifest: $output"
