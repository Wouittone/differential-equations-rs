# Nordsieck variable-order wave

This wave implements the four regular-ODE constructors exported by the pinned
`OrdinaryDiffEqNordsieck` package: `AN5`, `JVODE`, `JVODE_Adams`, and
`JVODE_BDF`. The aliases construct the corresponding genuine `JVODE` equation
family; they are not substitute solver names.

## Pinned sources

The implementation follows revision
`211142263781255a9aa2f910f6760b9f18ec29c8`, principally:

- `lib/OrdinaryDiffEqNordsieck/src/algorithms.jl`
- `lib/OrdinaryDiffEqNordsieck/src/nordsieck_caches.jl`
- `lib/OrdinaryDiffEqNordsieck/src/nordsieck_utils.jl`
- `lib/OrdinaryDiffEqNordsieck/src/nordsieck_perform_step.jl`
- `lib/OrdinaryDiffEqNordsieck/src/controllers.jl`

The Rust kernel owns the scaled derivative history, Pascal prediction,
variable-step rescaling, Adams or BDF correction, order-change transforms, and
the method-owned next-step ratio. `AN5` uses the fifth-order fixed-leading-
coefficient Adams family; `JVODE` supports Adams orders 1--12 and BDF orders
1--5. The BDF corrector uses an analytic Jacobian when supplied and otherwise
uses a dense finite-difference Jacobian.

## Verification and lifecycle

Rust tests cover fifth-order refinement, configured-alias identity, adaptive
order switching, analytic versus finite-difference Jacobian accounting,
backward integration, continuous callbacks, `save_at`, and retained dense
queries. The pinned Julia fixture invokes every constructor on the same fixed-
step scalar problem. A small step is intentional: pinned Julia's fixed-step
JVODE startup is visibly first-order over the initial history fill, so the
fixture compares the converged method rather than weakening its tolerance.

Accepted steps retain bounded cubic-Hermite segments made from their endpoint
derivatives. That is the generic upstream value-interpolation fallback, not a
claim of a separate high-order Nordsieck dense polynomial.

## Deliberate limits

Residual-form DAEs, singular mass matrices, arbitrary array element types,
user-selected nonlinear/linear solvers, and Julia's complete controller keyword
surface remain outside the crate's current regular dense-`f64` problem model.
The history and Newton workspaces are reused; owning dense segments allocate
only when retention is explicitly requested.
