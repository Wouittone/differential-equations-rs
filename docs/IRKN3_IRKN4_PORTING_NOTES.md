# IRKN3/IRKN4 porting notes

Source: `OrdinaryDiffEqRKN` at revision
`211142263781255a9aa2f910f6760b9f18ec29c8`, specifically
`rkn_tableaus.jl`, `rkn_caches.jl`, and the in-place kernels in
`rkn_perform_step.jl`. The out-of-place kernels are broken upstream and are not a
reliable specification.

These are fixed-step, velocity-independent, two-step formulas. They require the
previous two positions and velocities, two endpoint accelerations, and the prior
value of every internal stage. A step-size change invalidates that history.

## Bootstrap and history lifecycle

1. At initialization, evaluate the acceleration `A0 = f(v0, y0, t0)` and verify
   that the acceleration is velocity-independent.
2. Advance the first interval with the fourth-order, three-acceleration
   `Nystrom4VelocityIndependent` formula used upstream:
   `K2 = f(v0, y0 + h*v0/2 + h^2*A0/8, t0+h/2)`,
   `K3 = f(v0, y0 + h*v0 + h^2*K2/2, t0+h)`,
   `y1 = y0 + h*v0 + h^2*(A0 + 2*K2)/6`, and
   `v1 = v0 + h*(A0 + 4*K2 + K3)/6`.
3. Evaluate the new endpoint acceleration `A1 = f(v1, y1, t1)`. Seed each
   internal-stage history exactly as the pinned in-place cache does. For IRKN3,
   seed `H0 = f(v0, y0, t0+c1*h)` and
   `G0 = f(v0, y0 + h*(c1*v0 + h*a21*H0), t0+c1*h)`. IRKN4 instead seeds its
   `G0` with `A1`, and also seeds
   `J0 = f(v0, y0 + h*(c2*v0 + h*a32*H0), t0+c1*h)`. The final IRKN4 time is
   intentionally `t0+c1*h`, matching the pinned kernel's line 269.
4. The cache slot subsequently used as `Aold` is initially `H0`, not `A0`.
   After the first steady-state step it becomes the prior endpoint acceleration:
   shift `Aold <- A`, `A <- f(v_next, y_next, t+h)`, and shift every old internal
   stage to the just-computed stage. A discontinuity requires discarding this
   history and bootstrapping again.

## Steady-state IRKN3 formula

With endpoint acceleration `A`, the cached history value `Aold`, and retained
internal stage `Gold` (on the first such step, `Aold` is the seeded `H0`):

```text
G = f(v_n, y_n + h*(c1*v_n + h*a21*Aold), t_n+c1*h)
v_(n+1) = v_n + h*(b1*A + bbar1*Aold + b2*(G-Gold))
y_(n+1) = y_n + h*(bconst1*v_n + bconst2*v_(n-1))
                  + h^2*bbar2*(G-Gold)
```

## Steady-state IRKN4 formula

With the same endpoint/history convention and retained stages `Gold` and
`Jold`:

```text
G = f(v_n, y_n + h*(c1*v_n + h*a21*A), t_n+c1*h)
J = f(v_n, y_n + h*(c2*v_n + h*a32*G), t_n+c2*h)
v_(n+1) = v_n + h*(b1*A + bbar1*Aold
                    + b2*(G-Gold) + b3*(J-Jold))
y_(n+1) = y_n + h*(bconst1*v_n + bconst2*v_(n-1))
                  + h^2*(bbar2*(G-Gold) + bbar3*(J-Jold))
```

The generated coefficient file retains `bbar1` even though it contributes only
to the velocity equation. Regenerate or verify it with:

```powershell
julia scripts/generate_irkn_coefficients.jl `
  D:\Source\_review\OrdinaryDiffEq.jl src\irkn_coefficients.rs --check
```
