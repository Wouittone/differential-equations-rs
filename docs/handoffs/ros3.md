# ROS3 handoff

This wave adds the native `Ros3` algorithm, matching the pinned
`ROS3RodasTableau` from `OrdinaryDiffEqRosenbrockTableaus` at revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- three stages, `gamma = 0.435866521508459`;
- L-stable third-order primary weights;
- embedded second-order error weights;
- nonautonomous time nodes and `d` weights copied at full pinned precision.

`Ros3` reuses the shared Rosenbrock factorization, finite-difference or
analytic-Jacobian path, adaptive controller, callback invalidation, backward
integration, and save-at behavior already used by the extended Rosenbrock
family. The compliance example emits the adaptive ROS3 endpoint and the Julia
fixture compares it against pinned `ROS3()`.

Validation on the isolated branch:

```text
cargo fmt --all
cargo test --all-targets                 # pass (104 unit tests and targets)
cargo clippy --all-targets -- -D warnings # pass
git diff --check                          # pass
```

The Julia executable is not available on this worker's PATH; retry
`julia --project=tests/julia tests/julia/runtests.jl` when the documented Julia
environment is available.
