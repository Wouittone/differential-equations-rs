# Downstream smoke crates

These standalone packages exercise the library through the same dependency
boundary as an external application. They are intentionally excluded from the
root workspace so workspace feature unification cannot hide packaging or
feature-boundary regressions.

- `default-features` checks the default Rayon-backed ensemble API.
- `no-default-renamed` disables default features, renames the dependency,
  defines a solver from a JSON resource owned by the consumer, and solves the
  same ODE with scalar, vector, and matrix ndarray states.

Each package has a committed lockfile and is checked and tested by CI with the
Rust 1.85 minimum supported toolchain.
