[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Contains {
    param(
        [object[]] $Values,
        [string] $Expected,
        [string] $Message
    )
    if ($Expected -notin $Values) { throw $Message }
}

function Assert-NotContains {
    param(
        [object[]] $Values,
        [string] $Unexpected,
        [string] $Message
    )
    if ($Unexpected -in $Values) { throw $Message }
}

$fixtureRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("ode-inventory-detection-" + [guid]::NewGuid())
$sourceRoot = Join-Path $fixtureRoot 'src'
$juliaRoot = Join-Path $fixtureRoot 'tests/julia'
$generator = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../scripts/generate_ode_inventory.ps1')).Path
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

try {
    [void] [System.IO.Directory]::CreateDirectory($sourceRoot)
    [void] [System.IO.Directory]::CreateDirectory($juliaRoot)

    [System.IO.File]::WriteAllText((Join-Path $sourceRoot 'lib.rs'), @'
mod compatibility;
mod native;

pub use compatibility::algorithms;
pub use compatibility::{CollidingMethod, FacadeOnly};
'@, $utf8NoBom)

    [System.IO.File]::WriteAllText((Join-Path $sourceRoot 'native.rs'), @'
pub struct NativeMethod;
impl OdeAlgorithm for NativeMethod {}

macro_rules! algorithm {
    ($name:ident) => {
        pub struct $name;
        impl OdeAlgorithm for $name {}
    };
}

algorithm!(MacroMethod);
define_explicit_rk_from_file!(pub FileMethod, "tableaux/file_method.toml");

pub struct CollidingMethod;
impl OdeAlgorithm for CollidingMethod {}

pub struct SimpleNestedMethod;
impl OdeAlgorithm for SimpleNestedMethod {}

pub type NativeAlias = NativeMethod;
'@, $utf8NoBom)

    [System.IO.File]::WriteAllText((Join-Path $sourceRoot 'compatibility.rs'), @'
pub mod algorithms {
    pub use crate::native::{FileMethod, MacroMethod, NativeAlias, NativeMethod};

    pub mod nested {
        pub use crate::native::SimpleNestedMethod;
    }
}

pub type FacadeOnly = NativeMethod;
pub type CollidingMethod = NativeMethod;
'@, $utf8NoBom)

    [System.IO.File]::WriteAllText((Join-Path $juliaRoot 'integrity.jl'), @'
using OrdinaryDiffEqIntegrity:
    NativeMethod,
    ImportedOnly

tested = NativeMethod()
'@, $utf8NoBom)

    $detected = & $generator -DetectionOnly -RepositoryPath $fixtureRoot | ConvertFrom-Json

    Assert-Contains $detected.rust_public_names 'FacadeOnly' 'A public facade export was not detected.'
    Assert-Contains $detected.rust_public_names 'CollidingMethod' 'The facade collision was not detected as public.'
    Assert-Contains $detected.rust_public_names 'NativeMethod' 'A namespaced algorithm export was not detected.'
    Assert-Contains $detected.rust_public_names 'SimpleNestedMethod' 'A simple nested algorithm export was not detected.'
    Assert-NotContains $detected.rust_native_public_names 'FacadeOnly' 'A compatibility facade was treated as a native export.'
    Assert-NotContains $detected.rust_native_public_names 'CollidingMethod' 'A compatibility name collision was treated as a native export.'
    Assert-Contains $detected.rust_native_public_names 'NativeMethod' 'A namespaced algorithm export was treated as a facade.'
    Assert-Contains $detected.rust_native_public_names 'SimpleNestedMethod' 'A simple nested algorithm export was treated as a facade.'

    Assert-Contains $detected.rust_algorithm_implementation_names 'NativeMethod' 'A direct trait implementation was not detected.'
    Assert-Contains $detected.rust_algorithm_implementation_names 'MacroMethod' 'A macro-generated trait implementation was not detected.'
    Assert-Contains $detected.rust_algorithm_implementation_names 'FileMethod' 'A file-backed proc-macro implementation was not detected.'
    Assert-Contains $detected.rust_algorithm_implementation_names 'NativeAlias' 'A legitimate alias of an implementation was not detected.'
    Assert-Contains $detected.rust_algorithm_implementation_names 'SimpleNestedMethod' 'A simple nested implementation was not detected.'
    Assert-NotContains $detected.rust_algorithm_implementation_names 'FacadeOnly' 'A compatibility alias was treated as an implementation.'

    foreach ($name in @('NativeMethod', 'MacroMethod', 'FileMethod', 'NativeAlias', 'SimpleNestedMethod')) {
        Assert-Contains $detected.rust_implemented_public_names $name "$name was not recognized as implemented and public."
    }
    Assert-NotContains $detected.rust_implemented_public_names 'CollidingMethod' 'A private implementation was matched to a facade with the same name.'

    Assert-Contains $detected.julia_compliance_names 'NativeMethod' 'An invoked Julia constructor was not detected.'
    Assert-NotContains $detected.julia_compliance_names 'ImportedOnly' 'A Julia import without a constructor call was treated as compliance.'

    Write-Output 'Inventory detection integrity checks passed.'
} finally {
    $resolvedTemp = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    $resolvedFixture = [System.IO.Path]::GetFullPath($fixtureRoot)
    if (-not $resolvedFixture.StartsWith($resolvedTemp, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove fixture outside the temporary directory: $resolvedFixture"
    }
    if ([System.IO.Directory]::Exists($resolvedFixture)) {
        [System.IO.Directory]::Delete($resolvedFixture, $true)
    }
}
