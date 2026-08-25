# SIMD Runge--Kutta family

`MER5v2`, `MER6v2`, and `RK6v4` use the complete pinned tableaus generated
mechanically from `OrdinaryDiffEqSIMDRK/src/caches.jl`. The Julia implementation
packs independent stages into two- or four-lane SIMD values; Rust evaluates the
same independent rows over contiguous `f64` state slices so callers do not need
a packed-vector state type.

The methods retain their 14-, 15-, and 22-stage structures, fifth-/sixth-order
weights, embedded estimators, FSAL endpoint reuse, adaptive control, shared
callbacks, backward integration, and owning Hermite fallback. Tests verify
design-order refinement, distinct stage metadata, adaptivity, dense event
localization, and backward integration. A pinned Julia fixture compares all
three fixed-step endpoints.
