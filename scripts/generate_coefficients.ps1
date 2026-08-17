[CmdletBinding()]
param([switch]$Check)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$canonicalSource = Join-Path $root 'src/generated_coefficients.rs'
$output = Join-Path $root 'docs/coefficients_manifest.txt'
$sourceRevision = '211142263781255a9aa2f910f6760b9f18ec29c8'
$utf8NoBom = [Text.UTF8Encoding]::new($false)

function Normalize-Newlines([string]$Text) {
    return ($Text -replace "`r`n", "`n") -replace "`r", "`n"
}

function Get-NormalizedSha256([string]$Text) {
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $utf8NoBom.GetBytes((Normalize-Newlines $Text))
        $hash = $sha256.ComputeHash($bytes)
        return ([BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
}

if (-not (Test-Path -LiteralPath $canonicalSource -PathType Leaf)) {
    throw "Missing canonical coefficient source: $canonicalSource"
}

$source = Normalize-Newlines ([IO.File]::ReadAllText($canonicalSource))
$methodPattern = '(?m)^// coefficient-method: (?<record>[^\r\n]+)$'
$methods = @(
    [regex]::Matches($source, $methodPattern) |
        ForEach-Object { $_.Groups['record'].Value } |
        Sort-Object
)

if ($methods.Count -eq 0) {
    throw "No coefficient method records found in $canonicalSource"
}

$methodNames = foreach ($method in $methods) {
    if ($method -notmatch '^method=(?<name>[A-Za-z][A-Za-z0-9_]*)\|family=[a-z][a-z0-9-]*\|order=[1-9][0-9]*(?:\|[a-z][a-z0-9-]*=[A-Za-z0-9-]+)*$') {
        throw "Invalid coefficient method record: $method"
    }
    $Matches['name']
}
$duplicates = @($methodNames | Group-Object | Where-Object Count -gt 1)
if ($duplicates.Count -ne 0) {
    throw "Duplicate coefficient method records: $($duplicates.Name -join ', ')"
}

$sourceHash = Get-NormalizedSha256 $source
$contentLines = @(
    'schema-version=2'
    "source-revision=$sourceRevision"
    'canonical-source=src/generated_coefficients.rs'
    "canonical-source-sha256=$sourceHash"
) + $methods
$content = ($contentLines -join "`n") + "`n"

if ($Check) {
    if (-not (Test-Path -LiteralPath $output)) { throw "Missing generated coefficient manifest: $output" }
    $actual = Normalize-Newlines ([IO.File]::ReadAllText($output))
    if ($actual -cne $content) { throw "Coefficient manifest is stale; rerun generate_coefficients.ps1" }
    Write-Output "Verified canonical coefficient source and deterministic manifest: $output"
    exit 0
}

[IO.File]::WriteAllText($output, $content, $utf8NoBom)
Write-Output "Wrote deterministic coefficient manifest: $output"
