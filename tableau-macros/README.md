# differential-equations-tableau-macros

Procedural macros used by the
[`differential-equations`](https://crates.io/crates/differential-equations)
crate to validate JSON tableau resources at compile time and define lazy
resource-backed solvers without generating Rust coefficient arrays.

Most users should depend on `differential-equations` and use the macros
re-exported from `differential_equations::tableau`, such as
`tableau::define_explicit_rk_from_file!` and
`tableau::define_symplectic_from_file!`, rather than depend on this
implementation crate directly. JSON resources are the canonical extension
format: macros validate them during compilation, embed their source text, and
parse each method lazily on first use without generating Rust coefficient
arrays. `tableau::define_multistep_tableau_from_file!` defines canonical linear
multistep data, `tableau::define_rosenbrock_tableau_from_file!` provides
Rosenbrock data for specialized kernels, and `tableau::define_rkn_from_file!`
creates a second-order solver. RKN and IRKN tableau-only macros support
specialized kernels. ROCK2, ROCK4, SERK2, and ESERK degree-specific resources
have corresponding compile-validating lazy tableau macros. The schemas,
publishing requirements, and examples are documented in the main crate's
[tableau resource guide](https://docs.rs/differential-equations/latest/differential_equations/tableau/index.html).

This implementation crate is versioned and released in lockstep with the main
crate.

## License

Licensed under either Apache-2.0 or MIT, at your option.
