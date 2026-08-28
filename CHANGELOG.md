# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Compile-validated, lazily materialized JSON resources for explicit and
  implicit Runge--Kutta tableaus.
- Scalar problem inputs and ordered parallel ensemble solving through Rayon.
- Criterion-compatible performance regression benchmarks and package-policy
  checks.

### Changed

- Raised the minimum supported Rust version to 1.85 and adopted Rust 2024.
- Organized algorithms below hierarchical solver-family modules.
- Replaced handwritten tableau-expression parsing with maintained `exmex`.

### Removed

- SIMD-specific solver types whose lane-shaped API duplicated scalar/vector
  solvers and constrained user state representation.

## [0.1.0-beta.1] - Unreleased

Initial beta release candidate. This version is not yet API-stable.

[Unreleased]: https://github.com/Wouittone/differential-equations-rs/commits/main
[0.1.0-beta.1]: https://github.com/Wouittone/differential-equations-rs/releases/tag/v0.1.0-beta.1
