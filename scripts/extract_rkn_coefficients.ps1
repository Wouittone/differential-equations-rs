param(
    [Parameter(Mandatory = $true)]
    [string]$UpstreamPath,
    [string[]]$Algorithm = @(
        'DPRKN12', 'DPRKN4', 'DPRKN5', 'DPRKN6', 'DPRKN6FM', 'DPRKN8',
        'ERKN4', 'ERKN5', 'ERKN7', 'FineRKN4', 'FineRKN5', 'IRKN3',
        'IRKN4', 'Nystrom4', 'Nystrom4VelocityIndependent',
        'Nystrom5VelocityIndependent', 'RKN4'
    )
)

$ErrorActionPreference = 'Stop'
$expectedRevision = '211142263781255a9aa2f910f6760b9f18ec29c8'
$actualRevision = (& git -C $UpstreamPath rev-parse HEAD).Trim()
if ($actualRevision -ne $expectedRevision) {
    throw "OrdinaryDiffEq checkout is $actualRevision; expected $expectedRevision"
}

$source = Join-Path $UpstreamPath 'lib/OrdinaryDiffEqRKN/src/rkn_tableaus.jl'
$lines = Get-Content -LiteralPath $source
$declaration = '^(function|struct)\s+(?<name>[A-Za-z0-9]+)'
$starts = for ($index = 0; $index -lt $lines.Count; $index++) {
    if ($lines[$index] -match $declaration) {
        [pscustomobject]@{ Index = $index; Name = $Matches.name }
    }
}

foreach ($name in $Algorithm) {
    $matches = @($starts | Where-Object {
        $_.Name -eq "${name}Tableau" -or $_.Name -eq "${name}ConstantCache"
    })
    if ($matches.Count -eq 0) {
        throw "No tableau or constant-cache declaration found for $name"
    }
    foreach ($match in $matches) {
        $next = $starts | Where-Object { $_.Index -gt $match.Index } | Select-Object -First 1
        $end = if ($null -eq $next) { $lines.Count - 1 } else { $next.Index - 1 }
        Write-Output "# $name -- ${source}:$($match.Index + 1)-$($end + 1)"
        $lines[$match.Index..$end]
    }
}
