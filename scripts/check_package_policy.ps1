[CmdletBinding()]
param(
    [switch]$AllowDirty
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-CargoPackageList {
    param(
        [string]$Package
    )

    $arguments = @("package", "--locked", "--list", "-p", $Package)
    if ($AllowDirty) {
        $arguments += "--allow-dirty"
    }

    $files = & cargo @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "cargo package --list failed for $Package"
    }

    return @($files | ForEach-Object { $_ -replace "\\", "/" })
}

function Assert-PackageContains {
    param(
        [string[]]$Files,
        [string]$Expected,
        [string]$Package
    )

    if ($Expected -notin $Files) {
        throw "$Package package is missing $Expected"
    }
}

$repository = Split-Path -Parent $PSScriptRoot
Push-Location $repository

try {
    $metadataJson = & cargo metadata --locked --no-deps --format-version 1
    if ($LASTEXITCODE -ne 0) {
        throw "cargo metadata failed"
    }

    $metadata = $metadataJson | ConvertFrom-Json
    $packageNames = @(
        "differential-equations",
        "differential-equations-tableau-macros"
    )

    foreach ($packageName in $packageNames) {
        $package = $metadata.packages | Where-Object { $_.name -eq $packageName }
        if ($null -eq $package) {
            throw "workspace package $packageName was not found"
        }
        if ($package.license -ne "MIT OR Apache-2.0") {
            throw "$packageName must use the MIT OR Apache-2.0 SPDX expression"
        }
    }

    foreach ($license in @("LICENSE-APACHE", "LICENSE-MIT")) {
        $rootLicense = [System.IO.File]::ReadAllText((Join-Path $repository $license))
        $macroLicense = [System.IO.File]::ReadAllText(
            (Join-Path $repository "tableau-macros/$license")
        )
        if ($rootLicense -cne $macroLicense) {
            throw "tableau-macros/$license differs from the repository copy"
        }
    }

    $rootFiles = Invoke-CargoPackageList "differential-equations"
    foreach ($required in @("LICENSE-APACHE", "LICENSE-MIT", "THIRD_PARTY_NOTICES.md")) {
        Assert-PackageContains $rootFiles $required "differential-equations"
    }

    $forbiddenPrefixes = @("benchmarks/", "docs/", "examples/", "scripts/", "tests/")
    foreach ($file in $rootFiles) {
        foreach ($prefix in $forbiddenPrefixes) {
            if ($file.StartsWith($prefix, [System.StringComparison]::Ordinal)) {
                throw "differential-equations package unexpectedly contains $file"
            }
        }
    }

    $macroFiles = Invoke-CargoPackageList "differential-equations-tableau-macros"
    foreach ($required in @("LICENSE-APACHE", "LICENSE-MIT")) {
        Assert-PackageContains $macroFiles $required "differential-equations-tableau-macros"
    }

    Write-Host "Package policy checks passed."
}
finally {
    Pop-Location
}
