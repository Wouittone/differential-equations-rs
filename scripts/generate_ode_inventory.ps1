[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $UpstreamPath,

    [string] $RepositoryPath = (Split-Path -Parent $PSScriptRoot),

    [string] $OutputDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedRevision = '211142263781255a9aa2f910f6760b9f18ec29c8'
$resolvedUpstream = (Resolve-Path -LiteralPath $UpstreamPath).Path
$resolvedRepository = (Resolve-Path -LiteralPath $RepositoryPath).Path
$actualRevision = (& git -C $resolvedUpstream rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualRevision -ne $expectedRevision) {
    throw "Expected OrdinaryDiffEq.jl revision $expectedRevision, found $actualRevision"
}

# Only packages with public solver constructors belong here. Support-only packages
# (Core, Differentiation, NonlinearSolve, and the tableau packages) are deliberately
# absent. Adding a new upstream solver package is therefore an explicit scope choice.
$packageMetadata = [ordered]@{
    OrdinaryDiffEqAdamsBashforthMoulton = @{
        family = 'Adams multistep'; problem = 'ODEProblem'
        features = @('first-order state', 'multistep history', 'adaptive controller for VC variants')
    }
    OrdinaryDiffEqAMF = @{
        family = 'approximate-matrix-factorization wrapper'; problem = 'ODEProblem with structured ODEFunction'
        features = @('Rosenbrock-W inner method', 'Jacobian operator factors', 'structured linear solve')
        include = @('AMF')
    }
    OrdinaryDiffEqBDF = @{
        family = 'BDF and IMEX multistep'; problem = 'ODEProblem'
        features = @('first-order state', 'nonlinear solve', 'Jacobian and linear solve', 'multistep history')
    }
    OrdinaryDiffEqDefault = @{
        family = 'automatic/default composite'; problem = 'ODEProblem'
        features = @('composite algorithms', 'automatic stiffness detection', 'native explicit and stiff methods')
    }
    OrdinaryDiffEqExplicitRK = @{
        family = 'user-tableau explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'user-supplied Butcher tableau', 'adaptive controller')
    }
    OrdinaryDiffEqExponentialRK = @{
        family = 'exponential Runge-Kutta'; problem = 'ODEProblem or SplitODEProblem'
        features = @('linear/operator part', 'matrix exponential or Krylov action', 'semilinear split')
    }
    OrdinaryDiffEqExtrapolation = @{
        family = 'extrapolation'; problem = 'ODEProblem'
        features = @('repeated subintegration', 'adaptive order/step controller', 'nonlinear solve for implicit variants')
    }
    OrdinaryDiffEqFeagin = @{
        family = 'high-order explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'adaptive controller', 'high-order dense interpolation')
    }
    OrdinaryDiffEqFIRK = @{
        family = 'fully implicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('coupled-stage nonlinear solve', 'Jacobian and linear solve', 'adaptive controller')
    }
    OrdinaryDiffEqFunctionMap = @{
        family = 'discrete map'; problem = 'DiscreteProblem/operator map'
        features = @('fixed-step discrete update')
    }
    OrdinaryDiffEqHighOrderRK = @{
        family = 'high-order explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'adaptive controller', 'high-order dense interpolation')
    }
    OrdinaryDiffEqIMEXMultistep = @{
        family = 'IMEX multistep'; problem = 'SplitODEProblem'
        features = @('split right-hand side', 'multistep history', 'implicit linear/nonlinear solve')
    }
    OrdinaryDiffEqLinear = @{
        family = 'linear and Lie-group methods'; problem = 'ODEProblem with linear/operator ODEFunction'
        features = @('linear/operator action', 'matrix functions or Lie-group operations')
    }
    OrdinaryDiffEqLowOrderRK = @{
        family = 'low-order explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'adaptive controller where applicable', 'method-specific interpolation')
    }
    OrdinaryDiffEqLowStorageRK = @{
        family = 'low-storage explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'low-storage workspaces', 'stage/step limiters')
    }
    OrdinaryDiffEqMultirate = @{
        family = 'multirate and MRI-GARK'; problem = 'SplitODEProblem'
        features = @('split/partitioned right-hand side', 'nested fast integrator', 'multirate controller')
    }
    OrdinaryDiffEqNewmark = @{
        family = 'second-order structural dynamics'; problem = 'SecondOrderODEProblem'
        features = @('second-order state', 'mass matrix', 'nonlinear solve')
    }
    OrdinaryDiffEqNordsieck = @{
        family = 'Nordsieck variable-order multistep'; problem = 'ODEProblem'
        features = @('Nordsieck history', 'variable order/step controller', 'nonlinear solve for BDF variant')
    }
    OrdinaryDiffEqPDIRK = @{
        family = 'parallel diagonally implicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('parallel implicit stages', 'nonlinear solve', 'Jacobian and linear solve')
    }
    OrdinaryDiffEqPRK = @{
        family = 'parallel explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'parallel stages', 'adaptive controller')
    }
    OrdinaryDiffEqQPRK = @{
        family = 'QPRK explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'parallel stages', 'adaptive controller')
    }
    OrdinaryDiffEqRKIP = @{
        family = 'Runge-Kutta interval prediction'; problem = 'ODEProblem'
        features = @('first-order state', 'interval prediction', 'adaptive controller')
    }
    OrdinaryDiffEqRKN = @{
        family = 'Runge-Kutta-Nystrom'; problem = 'SecondOrderODEProblem or DynamicalODEProblem'
        features = @('second-order/partitioned state', 'velocity-dependence handling', 'dense interpolation')
    }
    OrdinaryDiffEqRosenbrock = @{
        family = 'Rosenbrock and Rosenbrock-W'; problem = 'ODEProblem'
        features = @('Jacobian and time derivative', 'linear solve', 'adaptive controller')
    }
    OrdinaryDiffEqSDIRK = @{
        family = 'SDIRK, ESDIRK, and additive IMEX RK'; problem = 'ODEProblem'
        features = @('nonlinear solve', 'Jacobian and linear solve', 'adaptive controller')
    }
    OrdinaryDiffEqSIMDRK = @{
        family = 'SIMD explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'SIMD-specialized workspace')
    }
    OrdinaryDiffEqSSPRK = @{
        family = 'strong-stability-preserving Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'stage/step limiters', 'adaptive controller where applicable')
    }
    OrdinaryDiffEqStabilizedIRK = @{
        family = 'stabilized implicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('nonlinear solve', 'Jacobian and linear solve', 'stability-degree selection')
    }
    OrdinaryDiffEqStabilizedRK = @{
        family = 'stabilized explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('spectral radius estimation', 'variable-stage stabilized recurrence', 'adaptive controller')
    }
    OrdinaryDiffEqSymplecticRK = @{
        family = 'symplectic and partitioned Runge-Kutta'; problem = 'DynamicalODEProblem'
        features = @('partitioned position/momentum state', 'fixed-step geometric update')
    }
    OrdinaryDiffEqTaylorSeries = @{
        family = 'Taylor series'; problem = 'ODEProblem'
        features = @('higher time derivatives or Taylor-mode AD', 'adaptive order/step controller')
    }
    OrdinaryDiffEqTsit5 = @{
        family = 'explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'adaptive controller', 'method-specific dense interpolation')
    }
    OrdinaryDiffEqVerner = @{
        family = 'high-order explicit Runge-Kutta'; problem = 'ODEProblem'
        features = @('first-order state', 'adaptive controller', 'high-order dense interpolation')
    }
}

$nonSolverPackages = [ordered]@{
    OrdinaryDiffEqCore = 'Shared integrator abstractions and internal composite types; no public numerical method constructors.'
    OrdinaryDiffEqDifferentiation = 'Jacobian, time-derivative, and differentiation support used by solver packages.'
    OrdinaryDiffEqExplicitTableaus = 'Butcher-tableau data and constructors, not solver algorithms.'
    OrdinaryDiffEqImplicitTableaus = 'Implicit tableau data and constructors, not solver algorithms.'
    OrdinaryDiffEqNonlinearSolve = 'Nonlinear-solver and DAE-initialization support, not time-integration algorithms.'
    OrdinaryDiffEqRosenbrockTableaus = 'Rosenbrock tableau data and constructors, not solver algorithms.'
}

$knownPackageNames = @($packageMetadata.Keys) + @($nonSolverPackages.Keys)
$upstreamPackageNames = @(Get-ChildItem -LiteralPath (Join-Path $resolvedUpstream 'lib') -Directory -Filter 'OrdinaryDiffEq*' |
    Select-Object -ExpandProperty Name)
$unclassifiedPackages = @($upstreamPackageNames | Where-Object { $_ -notin $knownPackageNames })
$missingPackages = @($knownPackageNames | Where-Object { $_ -notin $upstreamPackageNames })
if ($unclassifiedPackages.Count -gt 0 -or $missingPackages.Count -gt 0) {
    throw "OrdinaryDiffEq package classification drift. Unclassified: $($unclassifiedPackages -join ', '); missing: $($missingPackages -join ', ')"
}

$exactAliases = @{
    ETD1 = 'NorsettEuler'
}

# Upstream documentation explicitly calls these aliases even though they are
# configured constructor functions rather than Julia `const` type aliases.
$configuredAliases = @{
    QBDF1 = 'QNDF1(kappa=0)'
    QBDF2 = 'QNDF2(kappa=0)'
    QBDF = 'QNDF(kappa=(0,0,0,0,0))'
}

$compositeConstructors = @(
    'AMF', 'AutoDP5', 'AutoTsit5', 'AutoVern6', 'AutoVern7', 'AutoVern8', 'AutoVern9',
    'DefaultImplicitODEAlgorithm', 'DefaultODEAlgorithm'
)

$splitOverrides = @(
    'ARS222', 'ARS232', 'ARS343', 'ARS443', 'BHR553', 'CNAB2', 'CNLF2',
    'IMEXEuler', 'IMEXEulerARK', 'IMEXSSP222', 'IMEXSSP2322', 'IMEXSSP3332',
    'IMEXSSP3433', 'KenCarp3', 'KenCarp4', 'KenCarp47', 'KenCarp5', 'KenCarp58',
    'MEBDF2', 'SBDF', 'SBDF2', 'SBDF3', 'SBDF4', 'SplitEuler'
)

$excludedAlgorithms = @{
    DABDF2 = 'DAE residual-form algorithm; DAE-only behavior is outside regular ODE scope.'
    DFBDF = 'DAE residual-form algorithm; DAE-only behavior is outside regular ODE scope.'
    DImplicitEuler = 'DAE residual-form algorithm; DAE-only behavior is outside regular ODE scope.'
    FunctionMap = 'Discrete dynamical-system map, not a continuous initial-value ODE solver.'
}

function Get-ExportRecords {
    param([string] $ModulePath, [string] $Package)

    $lines = @(Get-Content -LiteralPath $ModulePath)
    $records = [System.Collections.Generic.List[object]]::new()
    for ($index = 0; $index -lt $lines.Count; $index++) {
        if (-not $lines[$index].StartsWith('export ')) { continue }
        $startLine = $index + 1
        $statement = $lines[$index].Substring(7)
        while ($statement.TrimEnd().EndsWith(',') -and ($index + 1) -lt $lines.Count) {
            $index++
            $statement += ' ' + $lines[$index].Trim()
        }
        foreach ($match in [regex]::Matches($statement, '(?<![\w!])([A-Za-z_][A-Za-z0-9_!]*)(?![\w!])')) {
            $records.Add([pscustomobject]@{
                name = $match.Groups[1].Value
                package = $Package
                export_source = $ModulePath
                export_line = $startLine
            })
        }
    }
    return $records
}

function Find-Definition {
    param([string] $PackagePath, [string] $Name, [string] $FallbackPath, [int] $FallbackLine)

    $escapedName = [regex]::Escape($Name)
    $patterns = @(
        "(?:mutable\s+)?struct\s+$escapedName(?:\{|\s|<)",
        "const\s+$escapedName\s*=",
        "^\s*$escapedName\s*\([^)]*\)\s*=",
        "^\s*function\s+$escapedName\s*\(",
        ":$escapedName(?:,|\s|\))"
    )
    $files = @(Get-ChildItem -LiteralPath (Join-Path $PackagePath 'src') -Recurse -File -Filter '*.jl' | Sort-Object FullName)
    foreach ($pattern in $patterns) {
        foreach ($file in $files) {
            $lines = @(Get-Content -LiteralPath $file.FullName)
            for ($index = 0; $index -lt $lines.Count; $index++) {
                if ($lines[$index] -match $pattern) {
                    return [pscustomobject]@{ path = $file.FullName; line = $index + 1 }
                }
            }
        }
    }
    return [pscustomobject]@{ path = $FallbackPath; line = $FallbackLine }
}

function Convert-ToRelativeUnixPath {
    param([string] $BasePath, [string] $Path)
    return ([System.IO.Path]::GetRelativePath($BasePath, $Path) -replace '\\', '/')
}

function Normalize-AlgorithmName {
    param([string] $Name)
    # Rust spells identifier fragments such as `_2N` as the idiomatic `TwoN`.
    return (($Name -replace '_', '').ToUpperInvariant() -replace 'TWON', '2N')
}

function Get-RustPublicNames {
    param([string] $RepoPath)
    $libPath = Join-Path $RepoPath 'src/lib.rs'
    $text = Get-Content -LiteralPath $libPath -Raw
    $names = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($match in [regex]::Matches($text, '(?ms)^pub use\s+[^;]+;')) {
        $statement = $match.Value
        if ($statement -match '(?s)\{(?<body>.*?)\}') {
            foreach ($identifier in [regex]::Matches($Matches.body, '\b[A-Z][A-Za-z0-9_]*\b')) {
                [void] $names.Add($identifier.Value)
            }
        } elseif ($statement -match '::(?<name>[A-Z][A-Za-z0-9_]*)\s*;') {
            [void] $names.Add($Matches.name)
        }
    }
    return $names
}

function Get-JuliaComplianceNames {
    param([string] $RepoPath)
    $names = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $RepoPath 'tests/julia') -File -Filter '*.jl') {
        $text = Get-Content -LiteralPath $file.FullName -Raw
        foreach ($match in [regex]::Matches($text, '(?ms)^using\s+OrdinaryDiffEq[A-Za-z0-9_]*\s*:\s*(?<body>.*?)(?=^\S|\z)')) {
            foreach ($identifier in [regex]::Matches($match.Groups['body'].Value, '\b[A-Z][A-Za-z0-9_]*\b')) {
                [void] $names.Add($identifier.Value)
            }
        }
    }
    return $names
}

$rustPublicNames = Get-RustPublicNames -RepoPath $resolvedRepository
$juliaComplianceNames = Get-JuliaComplianceNames -RepoPath $resolvedRepository
$rustByNormalizedName = @{}
foreach ($name in $rustPublicNames) { $rustByNormalizedName[(Normalize-AlgorithmName $name)] = $name }
$juliaByNormalizedName = @{}
foreach ($name in $juliaComplianceNames) { $juliaByNormalizedName[(Normalize-AlgorithmName $name)] = $name }

$inventory = [System.Collections.Generic.List[object]]::new()
$seenPublicNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
foreach ($packageEntry in $packageMetadata.GetEnumerator()) {
    $package = $packageEntry.Key
    $metadata = $packageEntry.Value
    $packagePath = Join-Path $resolvedUpstream "lib/$package"
    $modulePath = Join-Path $packagePath "src/$package.jl"
    if (-not (Test-Path -LiteralPath $modulePath)) {
        throw "Missing upstream module entry point: $modulePath"
    }
    $exports = Get-ExportRecords -ModulePath $modulePath -Package $package
    if ($metadata.ContainsKey('include')) {
        $includeNames = [System.Collections.Generic.HashSet[string]]::new([string[]] $metadata.include)
        $exports = @($exports | Where-Object { $includeNames.Contains($_.name) })
    }
    foreach ($export in $exports) {
        # Upstream currently repeats several SDIRK exports; public names are unique.
        if (-not $seenPublicNames.Add($export.name)) { continue }

        $definition = Find-Definition -PackagePath $packagePath -Name $export.name `
            -FallbackPath $export.export_source -FallbackLine $export.export_line
        $scope = if ($excludedAlgorithms.ContainsKey($export.name)) { 'excluded' } else { 'included' }
        $problem = $metadata.problem
        $features = @($metadata.features)
        if ($splitOverrides -contains $export.name) {
            $problem = 'ODEProblem or SplitODEProblem'
            $features = @($features + @('split explicit/implicit right-hand side')) | Select-Object -Unique
        }
        if ($package -eq 'OrdinaryDiffEqMultirate') {
            $problem = 'SplitODEProblem'
        }

        $normalized = Normalize-AlgorithmName $export.name
        $rustName = if ($rustByNormalizedName.ContainsKey($normalized)) { $rustByNormalizedName[$normalized] } else { $null }
        $hasCompliance = $juliaByNormalizedName.ContainsKey($normalized)
        $rustStatus = if ($scope -eq 'excluded') {
            'not-applicable'
        } elseif ($null -ne $rustName -and $hasCompliance) {
            'implemented-and-julia-tested'
        } elseif ($null -ne $rustName) {
            'implemented-without-detected-julia-test'
        } else {
            'missing'
        }

        $kind = if ($exactAliases.ContainsKey($export.name)) {
            'exact-alias'
        } elseif ($configuredAliases.ContainsKey($export.name)) {
            'configured-alias'
        } elseif ($compositeConstructors -contains $export.name) {
            'composite-constructor'
        } else {
            'algorithm'
        }

        $inventory.Add([pscustomobject][ordered]@{
            name = $export.name
            kind = $kind
            alias_of = if ($exactAliases.ContainsKey($export.name)) {
                $exactAliases[$export.name]
            } elseif ($configuredAliases.ContainsKey($export.name)) {
                $configuredAliases[$export.name]
            } else {
                $null
            }
            upstream_package = $package
            family = $metadata.family
            scope = $scope
            exclusion_reason = if ($excludedAlgorithms.ContainsKey($export.name)) { $excludedAlgorithms[$export.name] } else { $null }
            problem_representation = $problem
            required_features = @($features)
            rust_status = $rustStatus
            rust_name = $rustName
            julia_compliance_detected = $hasCompliance
            upstream_source = Convert-ToRelativeUnixPath -BasePath $resolvedUpstream -Path $definition.path
            upstream_line = $definition.line
        })
    }
}

$inventory = @($inventory | Sort-Object upstream_package, name)
$implemented = @($inventory | Where-Object rust_status -eq 'implemented-and-julia-tested')
$included = @($inventory | Where-Object scope -eq 'included')
$excluded = @($inventory | Where-Object scope -eq 'excluded')
$aliasEntries = @($included | Where-Object kind -like '*-alias')
$missing = @($included | Where-Object rust_status -eq 'missing')
$canonicalIncluded = @($included | Where-Object kind -notlike '*-alias')

if ($implemented.Count -lt 25) {
    throw "Expected to detect at least the baseline 25 Julia-tested Rust algorithms, found $($implemented.Count): $($implemented.name -join ', ')"
}

$summary = [pscustomobject][ordered]@{
    schema_version = 1
    upstream_repository = 'https://github.com/SciML/OrdinaryDiffEq.jl'
    upstream_revision = $actualRevision
    scope_document = 'docs/UPSTREAM_SCOPE.md'
    counts = [pscustomobject][ordered]@{
        public_solver_names = $inventory.Count
        included_names = $included.Count
        included_canonical_or_composite = $canonicalIncluded.Count
        included_aliases = $aliasEntries.Count
        excluded_names = $excluded.Count
        implemented_and_julia_tested = $implemented.Count
        missing_included_names = $missing.Count
    }
    uncertainties = @(
        'The inventory treats package exports as the public algorithm surface; internal, unexported experimental types are not parity targets.',
        'ETD1 is the only exact exported type alias found at the pinned revision. QBDF/QBDF1/QBDF2 are configured aliases as stated by their upstream documentation. Auto* and Default* names are composite constructors.',
        'AMF is counted as a native wrapper constructor because it dispatches back into OrdinaryDiffEq with native Rosenbrock-W methods.',
        'Rust status is detected by normalized public Rust type name plus a corresponding OrdinaryDiffEq import in tests/julia; numerical test quality remains a review concern.',
        'Implemented status measures public algorithm-name coverage only. It does not establish parity for every upstream problem representation or shared feature; consult problem_representation, required_features, and FEATURE_COVERAGE.md separately.',
        'Nonsingular mass-matrix behavior of dual ODE/DAE methods is included, while residual-form DAE constructors and singular-mass-matrix behavior are excluded.'
    )
    non_solver_packages = @($nonSolverPackages.GetEnumerator() | ForEach-Object {
        [pscustomobject][ordered]@{ package = $_.Key; rationale = $_.Value }
    })
    algorithms = $inventory
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$jsonPath = Join-Path $OutputDirectory 'ode_algorithm_inventory.json'
$csvPath = Join-Path $OutputDirectory 'ode_algorithm_inventory.csv'
$markdownPath = Join-Path $OutputDirectory 'ODE_PARITY_INVENTORY.md'

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$inventory | Select-Object name, kind, alias_of, upstream_package, family, scope, exclusion_reason,
    problem_representation, @{ n = 'required_features'; e = { $_.required_features -join '; ' } },
    rust_status, rust_name, julia_compliance_detected, upstream_source, upstream_line |
    Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding utf8

$familyRows = $included | Group-Object family | Sort-Object Name | ForEach-Object {
    $familyEntries = @($_.Group)
    $familyImplemented = @($familyEntries | Where-Object rust_status -eq 'implemented-and-julia-tested').Count
    "| $($_.Name) | $($familyEntries.Count) | $familyImplemented | $($familyEntries.Count - $familyImplemented) |"
}
$excludedRows = $excluded | Sort-Object name | ForEach-Object {
    "| ``$($_.name)`` | $($_.upstream_package) | $($_.exclusion_reason) |"
}
$aliasRows = $aliasEntries | Sort-Object name | ForEach-Object {
    "| ``$($_.name)`` | $($_.kind) | ``$($_.alias_of)`` | $($_.upstream_package) |"
}
$uncertaintyRows = $summary.uncertainties | ForEach-Object { "- $_" }
$nonSolverRows = $summary.non_solver_packages | ForEach-Object {
    "| ``$($_.package)`` | $($_.rationale) |"
}
$missingSections = $missing | Group-Object family | Sort-Object Name | ForEach-Object {
    $entries = @($_.Group | Sort-Object name | ForEach-Object {
        "- ``$($_.name)`` — $($_.upstream_package); $($_.problem_representation)"
    })
    "### $($_.Name) ($($_.Count))`n`n$($entries -join "`n")"
}

$markdown = @"
# Regular ODE parity inventory

This is a generated summary of native solver exports at SciML/OrdinaryDiffEq.jl
revision ``$actualRevision`` under the scope in
[`UPSTREAM_SCOPE.md`](UPSTREAM_SCOPE.md). Regenerate it with:

``````powershell
./scripts/generate_ode_inventory.ps1 -UpstreamPath <path-to-OrdinaryDiffEq.jl>
``````

The full machine-readable records, including upstream definition paths and line
numbers, problem representations, required shared features, aliases, exclusions,
and current Rust status, are in [`ode_algorithm_inventory.json`](ode_algorithm_inventory.json)
and [`ode_algorithm_inventory.csv`](ode_algorithm_inventory.csv).

## Totals

- Public solver names inspected: **$($summary.counts.public_solver_names)**.
- In-scope regular ODE names: **$($summary.counts.included_names)**
  ($($summary.counts.included_canonical_or_composite) canonical/composite constructors and
  $($summary.counts.included_aliases) public aliases).
- Implemented and detected in matched Julia tests: **$($summary.counts.implemented_and_julia_tested)**.
- Missing in-scope public names: **$($summary.counts.missing_included_names)**.
- Explicitly excluded public names: **$($summary.counts.excluded_names)**.

Aliases are public parity obligations but do not require a second numerical kernel.

## Family status

| Family | In scope | Implemented + Julia-tested | Missing names |
| --- | ---: | ---: | ---: |
$($familyRows -join "`n")

## Remaining solver names by family

This is the implementation handoff list. Each entry remains in scope and lacks
either a public Rust implementation or a detected matched Julia compliance
case. Required features and exact upstream source locations are available in
the JSON/CSV records.

$($missingSections -join "`n`n")

## Aliases

| Public name | Kind | Canonical target | Package |
| --- | --- | --- | --- |
$($aliasRows -join "`n")

## Explicit exclusions

| Public name | Package | Rationale |
| --- | --- | --- |
$($excludedRows -join "`n")

Package-level exclusions from [`UPSTREAM_SCOPE.md`](UPSTREAM_SCOPE.md), such as
DelayDiffEq, StochasticDiffEq, external wrappers, BVP, PDE, and steady-state
solvers, are not expanded into per-algorithm rows because they are not part of
the OrdinaryDiffEq native ODE solver export surface.

## Classified support-only subpackages

| Package | Why it has no solver rows |
| --- | --- |
$($nonSolverRows -join "`n")

## Interpretation notes and uncertainties

$($uncertaintyRows -join "`n")
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "Wrote $jsonPath"
Write-Output "Wrote $csvPath"
Write-Output "Wrote $markdownPath"
Write-Output ($summary.counts | ConvertTo-Json -Compress)
