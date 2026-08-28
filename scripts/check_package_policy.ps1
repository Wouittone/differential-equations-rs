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

function Assert-PackageMarkdownLinksResolve {
    param(
        [string]$Repository,
        [string[]]$Files,
        [string]$Package
    )

    foreach ($file in $Files) {
        if (-not $file.EndsWith(".md", [System.StringComparison]::OrdinalIgnoreCase)) {
            continue
        }

        $markdownPath = Join-Path $Repository (
            $file -replace "/", [System.IO.Path]::DirectorySeparatorChar
        )
        $content = [System.IO.File]::ReadAllText($markdownPath)
        $links = [System.Text.RegularExpressions.Regex]::Matches(
            $content,
            '\[[^\]]+\]\((?<target>[^)\s]+)'
        )
        foreach ($link in $links) {
            $target = $link.Groups["target"].Value.Trim("<", ">")
            if (
                $target.StartsWith("#", [System.StringComparison]::Ordinal) -or
                $target.StartsWith("https://", [System.StringComparison]::OrdinalIgnoreCase) -or
                $target.StartsWith("http://", [System.StringComparison]::OrdinalIgnoreCase) -or
                $target.StartsWith("mailto:", [System.StringComparison]::OrdinalIgnoreCase)
            ) {
                continue
            }

            $targetWithoutFragment = $target.Split("#", 2)[0]
            if ([string]::IsNullOrWhiteSpace($targetWithoutFragment)) {
                continue
            }
            $resolved = [System.IO.Path]::GetFullPath((Join-Path (
                [System.IO.Path]::GetDirectoryName($markdownPath)
            ) ($targetWithoutFragment -replace "/", [System.IO.Path]::DirectorySeparatorChar)))
            $relative = [System.IO.Path]::GetRelativePath($Repository, $resolved) -replace "\\", "/"
            if ($relative.StartsWith("../", [System.StringComparison]::Ordinal)) {
                throw "$file links outside the $Package package: $target"
            }
            if ($relative -notin $Files) {
                throw "$file links to a file omitted from the $Package package: $target"
            }
        }
    }
}

function Invoke-CargoPackage {
    param(
        [string]$Package
    )

    $arguments = @("package", "--locked", "-p", $Package)
    if ($AllowDirty) {
        $arguments += "--allow-dirty"
    }

    & cargo @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "cargo package failed for $Package"
    }
}

function Copy-PackageFiles {
    param(
        [string]$SourceRoot,
        [string]$DestinationRoot,
        [string[]]$Files,
        [switch]$SkipLock
    )

    [System.IO.Directory]::CreateDirectory($DestinationRoot) | Out-Null
    foreach ($file in $Files) {
        if (
            $file -in @(".cargo_vcs_info.json", "Cargo.toml.orig") -or
            ($SkipLock -and $file -eq "Cargo.lock")
        ) {
            continue
        }
        $source = Join-Path $SourceRoot (
            $file -replace "/", [System.IO.Path]::DirectorySeparatorChar
        )
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Cargo selected a package file that cannot be staged: $source"
        }
        $destination = Join-Path $DestinationRoot (
            $file -replace "/", [System.IO.Path]::DirectorySeparatorChar
        )
        [System.IO.Directory]::CreateDirectory(
            [System.IO.Path]::GetDirectoryName($destination)
        ) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
    }
}

function Test-StagedMainCrate {
    param(
        [string]$Repository,
        [string[]]$RootFiles,
        [string[]]$MacroFiles,
        [string[]]$CoreFiles,
        [object]$MacroPackage,
        [object]$CorePackage
    )

    $temporaryRoot = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::GetTempPath()
    )
    $harness = Join-Path $temporaryRoot (
        "differential-equations-package-" + [System.Guid]::NewGuid().ToString("N")
    )
    $harness = [System.IO.Path]::GetFullPath($harness)
    if (-not $harness.StartsWith($temporaryRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "package harness escaped the temporary directory"
    }

    [System.IO.Directory]::CreateDirectory($harness) | Out-Null
    try {
        $macroDirectoryName = "$($MacroPackage.name)-$($MacroPackage.version)"
        $coreDirectoryName = "$($CorePackage.name)-$($CorePackage.version)"
        $macroDirectory = Join-Path $harness $macroDirectoryName
        $coreDirectory = Join-Path $harness $coreDirectoryName
        Copy-PackageFiles `
            (Join-Path $Repository "tableau-macros") `
            $macroDirectory `
            $MacroFiles `
            -SkipLock
        Copy-PackageFiles `
            (Join-Path $Repository "tableau-core") `
            $coreDirectory `
            $CoreFiles `
            -SkipLock

        $macroManifestPath = Join-Path $macroDirectory "Cargo.toml"
        $macroManifest = [System.IO.File]::ReadAllText($macroManifestPath)
        $macroManifest = $macroManifest.Replace(
            'path = "../tableau-core"',
            "path = `"../$coreDirectoryName`""
        )
        [System.IO.File]::WriteAllText($macroManifestPath, $macroManifest)

        $rootDirectory = Join-Path $harness "staged-main"
        Copy-PackageFiles $Repository $rootDirectory $RootFiles

        $manifestPath = Join-Path $rootDirectory "Cargo.toml"
        $manifest = [System.IO.File]::ReadAllText($manifestPath)
        $manifest = [System.Text.RegularExpressions.Regex]::Replace(
            $manifest,
            '(?ms)\r?\n\[workspace\]\r?\n.*?(?=\r?\n\[)',
            "`n"
        )
        $manifest = $manifest.Replace(
            'path = "tableau-macros"',
            "path = `"../$macroDirectoryName`""
        )
        $manifest = $manifest.Replace(
            'path = "tableau-core"',
            "path = `"../$coreDirectoryName`""
        )
        [System.IO.File]::WriteAllText($manifestPath, $manifest)

        Push-Location $rootDirectory
        try {
            foreach ($featureArguments in @(
                @("--all-features"),
                @("--no-default-features")
            )) {
                $arguments = @("check", "--locked", "--all-targets") + $featureArguments
                & cargo @arguments
                if ($LASTEXITCODE -ne 0) {
                    throw "extracted package check failed: cargo $($arguments -join ' ')"
                }
            }
        }
        finally {
            Pop-Location
        }
    }
    finally {
        if (Test-Path -LiteralPath $harness) {
            Remove-Item -LiteralPath $harness -Recurse -Force
        }
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
        "differential-equations-tableau-core",
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
        $rootLicense = [System.IO.File]::ReadAllText(
            (Join-Path $repository $license)
        ).TrimEnd("`r", "`n")
        $macroLicense = [System.IO.File]::ReadAllText(
            (Join-Path $repository "tableau-macros/$license")
        ).TrimEnd("`r", "`n")
        $coreLicense = [System.IO.File]::ReadAllText(
            (Join-Path $repository "tableau-core/$license")
        ).TrimEnd("`r", "`n")
        if ($rootLicense -cne $macroLicense) {
            throw "tableau-macros/$license differs from the repository copy"
        }
        if ($rootLicense -cne $coreLicense) {
            throw "tableau-core/$license differs from the repository copy"
        }
    }

    $rootFiles = Invoke-CargoPackageList "differential-equations"
    foreach ($required in @(
        "LICENSE-APACHE",
        "LICENSE-MIT",
        "THIRD_PARTY_NOTICES.md",
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "SUPPLY_CHAIN.md",
        "docs/ALGORITHM_COVERAGE.md",
        "docs/BENCHMARKING.md",
        "docs/FEATURE_COVERAGE.md",
        "docs/ODE_PARITY_INVENTORY.md",
        "docs/RELEASING.md",
        "docs/TABLEAU_RESOURCES.md",
        "docs/TABLEAU_MIGRATION_COVERAGE.md",
        "docs/UPSTREAM_SCOPE.md",
        "benches/solver_performance.rs",
        "examples/quickstart.rs",
        "examples/tableau_from_file.rs",
        "examples/resources/file_heun.json"
    )) {
        Assert-PackageContains $rootFiles $required "differential-equations"
    }

    $forbiddenPrefixes = @("benchmarks/", "docs/handoffs/", "scripts/", "tests/")
    foreach ($file in $rootFiles) {
        foreach ($prefix in $forbiddenPrefixes) {
            if ($file.StartsWith($prefix, [System.StringComparison]::Ordinal)) {
                throw "differential-equations package unexpectedly contains $file"
            }
        }
    }
    foreach ($forbidden in @(
        "docs/ode_algorithm_inventory.csv",
        "docs/ode_algorithm_inventory.json"
    )) {
        if ($forbidden -in $rootFiles) {
            throw "differential-equations package unexpectedly contains $forbidden"
        }
    }
    Assert-PackageMarkdownLinksResolve `
        $repository `
        $rootFiles `
        "differential-equations"

    $macroFiles = Invoke-CargoPackageList "differential-equations-tableau-macros"
    foreach ($required in @("LICENSE-APACHE", "LICENSE-MIT", "README.md")) {
        Assert-PackageContains $macroFiles $required "differential-equations-tableau-macros"
    }
    Assert-PackageMarkdownLinksResolve `
        (Join-Path $repository "tableau-macros") `
        $macroFiles `
        "differential-equations-tableau-macros"

    $coreFiles = Invoke-CargoPackageList "differential-equations-tableau-core"
    foreach ($required in @("LICENSE-APACHE", "LICENSE-MIT", "README.md")) {
        Assert-PackageContains $coreFiles $required "differential-equations-tableau-core"
    }
    Assert-PackageMarkdownLinksResolve `
        (Join-Path $repository "tableau-core") `
        $coreFiles `
        "differential-equations-tableau-core"

    $macroPackage = $metadata.packages |
        Where-Object { $_.name -eq "differential-equations-tableau-macros" }
    $corePackage = $metadata.packages |
        Where-Object { $_.name -eq "differential-equations-tableau-core" }

    # Cargo cannot assemble downstream archives until the exact internal crate
    # versions exist on crates.io. Package the leaf crate, then stage exactly
    # the files selected by `cargo package --list`, point that isolated graph
    # at the staged internal crates, and compile both feature modes.
    Invoke-CargoPackage "differential-equations-tableau-core"
    Test-StagedMainCrate `
        $repository `
        $rootFiles `
        $macroFiles `
        $coreFiles `
        $macroPackage `
        $corePackage

    Write-Host "Package policy checks passed."
}
finally {
    Pop-Location
}
