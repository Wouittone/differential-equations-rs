# differential-equations-tableau-macros

Procedural macros used by the
[`differential-equations`](https://crates.io/crates/differential-equations)
crate to validate JSON tableau resources at compile time and define lazy
resource-backed solvers without generating Rust coefficient arrays.

Most users should depend on `differential-equations` and use its re-exported
`define_explicit_rk_from_file!` or `define_symplectic_from_file!` macro for
solver definitions, or `define_multistep_tableau_from_file!` for canonical
linear multistep data. `define_rosenbrock_tableau_from_file!` provides canonical
Rosenbrock data for specialized kernels. The schemas, publishing requirements, and examples are
documented in the main crate's
[tableau resource guide](https://docs.rs/differential-equations/latest/differential_equations/tableau/index.html).

This implementation crate is versioned and released in lockstep with the main
crate.

## License

Licensed under either Apache-2.0 or MIT, at your option.
