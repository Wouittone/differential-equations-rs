# differential-equations-tableau-core

Shared Serde parser and validator for the canonical JSON Runge--Kutta tableau
resources used by `differential-equations` and its procedural macros.

Most users should depend on `differential-equations`, which exposes the public
tableau API and compile-validating definition macro. This crate is published
separately so compile-time and runtime validation execute the same code.

Licensed under either Apache-2.0 or MIT, at your option.
