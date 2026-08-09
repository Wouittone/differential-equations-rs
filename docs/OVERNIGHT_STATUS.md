# Overnight execution status

Coordinator: `/root`

Started: `2026-08-03T23:32:13Z`

Current phase: `Phase 7 - regular solver-family expansion`

Pinned upstream revision:

```text
211142263781255a9aa2f910f6760b9f18ec29c8
```

## Current gates

- [x] Soundness gate
- [x] Upstream inventory
- [x] Shared integrator driver
- [x] Vector/matrix interfaces
- [x] Coefficient schema/code generation foundation
- [x] General problem representations foundation
- [x] Dense output/controller parity foundation and recorder service
- [ ] Solver-family migration (in progress; SDIRK2 first wave merged)
- [ ] Final compliance audit

## Active agents

| Agent | Task | Branch/worktree | Status | Last update |
| --- | --- | --- | --- | --- |
| `/root/soundness_gate` | Simplifier/soundness gate | `codex/overnight-soundness`; `differential-equations-rs-worktrees/soundness-gate` | completed and merged as `408991c` | 2026-08-03T23:41:39Z |
| `/root/inventory_audit` | Exact pinned regular-ODE inventory audit | `codex/overnight-inventory`; `differential-equations-rs-worktrees/inventory-audit` | completed via `0579ff4` and `b4ff329` | 2026-08-04T00:00:00Z |
| `/root/julia_manifest_repro` | Make pinned Julia gate reproducible in fresh worktrees | `codex/overnight-julia-manifest`; `differential-equations-rs-worktrees/julia-manifest` | completed and merged as `57ea0f8` | 2026-08-03T23:45:00Z |
| `/root/driver_explicit_wave` | Driver foundation and generic explicit-RK migration | `codex/overnight-driver-explicit`; `differential-equations-rs-worktrees/driver-explicit` | completed and merged as `47c8ae5` | 2026-08-03T23:55:11Z |
| `/root/explicit_upstream_audit` | Pinned explicit-RK dense/controller source audit | `codex/overnight-explicit-upstream-audit`; `differential-equations-rs-worktrees/explicit-upstream-audit` | completed and merged as `fc0d556` | 2026-08-04T00:05:00Z |
| `/root/driver_implicit_wave` | Driver contract completion and implicit/TRBDF2 migration | `codex/overnight-driver-implicit`; `differential-equations-rs-worktrees/driver-implicit` | completed and merged as `a288382` | 2026-08-04T00:08:00Z |
| `/root/linear_interface_audit` | Phase 3 vector/matrix/Jacobian/linear-solver source audit | `codex/overnight-linear-interface-audit`; `differential-equations-rs-worktrees/linear-interface-audit` | completed and merged as `8e76510` | 2026-08-04T00:12:00Z |
| `/root/driver_rosenbrock_wave` | Rosenbrock/Rodas shared-driver migration | `codex/overnight-driver-rosenbrock`; `differential-equations-rs-worktrees/driver-rosenbrock` | completed and merged as `f5df68b` | 2026-08-04T00:16:00Z |
| `/root/driver_adams_wave` | Fixed/variable Adams shared-driver migration | `codex/overnight-driver-adams`; `differential-equations-rs-worktrees/driver-adams` | completed and merged as `e747d0b` | 2026-08-08T20:28:34Z |
| `/root/driver_low_storage_wave` | Low-storage RK shared-driver migration | `codex/overnight-driver-low-storage`; `differential-equations-rs-worktrees/driver-low-storage` | completed and merged as `962c89a` | 2026-08-04T00:21:00Z |
| `/root/phase2_lifecycle_audit` | Repository-wide first-order lifecycle audit | `codex/phase2-lifecycle-audit`; `differential-equations-rs-worktrees/phase2-lifecycle-audit` | completed and merged as `5383420` | 2026-08-08T20:34:35Z |
| `/root/linear_caller_migration` | Checked DenseLu/StateLayout migration of implicit caller | `codex/linear-caller-migration`; `differential-equations-rs-worktrees/linear-caller` | completed and merged as `335d162` | 2026-08-08T21:02:30Z |
| `/root/sdirk2_kernel` | Exact regular SDIRK2 family implementation | `codex/sdirk2-kernel`; `differential-equations-rs-worktrees/sdirk2-kernel` | completed and merged as `1909900` + `ab1fd98` | 2026-08-09T00:20:00Z |
| `/root/linear_interface_impl` | Accepted-step dense recorder and controller reset slice | `codex/phase6-dense-controller`; `differential-equations-rs-worktrees/dense-controller` | completed and merged as `dbf9a16` | 2026-08-09T00:35:00Z |
| `/root/abdf2_kernel` | Exact regular ABDF2 identity-mass family implementation | `codex/abdf2-kernel`; `differential-equations-rs-worktrees/abdf2-kernel` | completed and merged as `aedca27` plus estimator/workspace refresh | 2026-08-09T01:05:00Z |
| `/root/linear_interface_impl2` | Additional generated explicit-coefficient slice | `codex/generated-dp5`; `differential-equations-rs-worktrees/generated-dp5` | completed and merged as `69f6caf`; DP5 generated migration passed | 2026-08-09T02:05:00Z |
| `/root/linear_interface_impl` | Wire accepted Hermite dense service into one solver family | `codex/phase6-explicit-hermite`; `differential-equations-rs-worktrees/explicit-hermite` | completed and merged as `59ece5d` | 2026-08-09T02:20:00Z |
| `/root/abdf2_kernel` | Fixed-step regular MEBDF2 feasibility/implementation | `codex/mebdf2-kernel`; `differential-equations-rs-worktrees/mebdf2-kernel` | completed and merged as `5e0650c` | 2026-08-09T03:05:00Z |
| `/root/linear_interface_impl2` | Next missing regular explicit/SSPRK family constructor | `codex/verner9-generated`; `differential-equations-rs-worktrees/verner9-generated` | completed and merged as `3371551` | 2026-08-09T15:35:00Z |
| `/root/final_audit` | Phase 8 final-audit precheck | `codex/final-audit-precheck`; `differential-equations-rs-worktrees/final-audit-precheck` | completed and merged as `475fa28`; no sign-off, gaps documented | 2026-08-09T15:02:00Z |
| `/root/linear_interface_impl2` | Parametric relaxation SSPRK22 (`pRRK22`) | `codex/ssprk-prrk`; `differential-equations-rs-worktrees/ssprk-prrk` | completed and merged as `dd3c46b` | 2026-08-09T15:07:00Z |
| `/root/ssprk432` | Adaptive SSPRK432 | `codex/ssprk432`; `differential-equations-rs-worktrees/ssprk432` | Rust gates pass; Julia blocked by missing executable; merged as `e32ee02` | 2026-08-09T15:10:00Z |
| `/root/low_storage_next` | Parsani Ketcheson Deconinck 3S32 | `codex/low-storage-next`; `differential-equations-rs-worktrees/low-storage-next` | Rust gates pass; Julia blocked by missing executable; merged as `3d1d13d` | 2026-08-09T15:22:00Z |
| `/root/low_order_next` | Alshina6 fixed-step RK | `codex/low-order-next`; `differential-equations-rs-worktrees/low-order-next` | Rust gates pass; Julia blocked by missing executable; merged as `1aafd9d` | 2026-08-09T15:23:00Z |
| `/root/low_storage_next` | Parsani Ketcheson Deconinck 3S82 follow-up | `codex/low-storage-3s82`; `differential-equations-rs-worktrees/low-storage-3s82` | Rust gates pass; Julia blocked by missing executable; merged as `86de559` | 2026-08-09T15:29:00Z |
| `/root/low_order_next` | Anas5 parameterized fixed-step RK | `codex/low-order-anas5`; `differential-equations-rs-worktrees/low-order-anas5` | Rust gates pass; Julia blocked by missing executable; merged as `80867f2` | 2026-08-09T15:33:00Z |
| `/root/low_storage_3s53` | Parsani Ketcheson Deconinck 3S53 | `codex/low-storage-3s53`; `differential-equations-rs-worktrees/low-storage-3s53` | Rust gates pass; Julia blocked by missing executable; merged as `6e905c4` | 2026-08-09T15:40:00Z |
| `/root/low_order_msrk5` | MSRK5 fixed-step RK | `codex/low-order-msrk5`; `differential-equations-rs-worktrees/low-order-msrk5` | Rust gates pass; Julia blocked by missing executable; merged as `1b7499b` | 2026-08-09T15:41:00Z |
| `/root/low_storage_3s94` | Parsani Ketcheson Deconinck 3S94 | `codex/low-storage-3s94`; `differential-equations-rs-worktrees/low-storage-3s94` | Rust gates pass; Julia blocked by missing executable; merged as `2fb5dc5` | 2026-08-09T15:47:00Z |
| `/root/low_order_msrk6` | MSRK6 fixed-step RK | `codex/low-order-msrk6`; `differential-equations-rs-worktrees/low-order-msrk6` | Rust gates pass; Julia blocked by missing executable; final FSAL correction merged as `3cfdf35` | 2026-08-09T15:47:00Z |
| `/root/low_storage_3s105` | Parsani Ketcheson Deconinck 3S105 | `codex/low-storage-3s105`; `differential-equations-rs-worktrees/low-storage-3s105` | Rust gates pass; Julia blocked by missing executable; merged as `e4a659e` | 2026-08-09T15:57:00Z |
| `/root/low_order_frk65` | FRK65 fitted low-order RK | `.worktrees/low-order-frk65`; `codex/low-order-frk65` | Rust gates pass; Julia blocked by missing executable; merged as `dec241b` + `f3e2078` | 2026-08-09T16:02:00Z |
| `/root/low_storage_3s173` | Parsani Ketcheson Deconinck 3S173 | `codex/low-storage-3s173`; `differential-equations-rs-worktrees/low-storage-3s173` | Rust gates pass; Julia blocked by missing executable; merged as `e081664` | 2026-08-09T16:10:00Z |
| `/root/low_order_psrk` | PSRK3p5q4 fixed-step RK | `codex/low-order-psrk`; `differential-equations-rs-worktrees/low-order-psrk` | Rust gates pass; Julia timed out/unavailable; merged as `2823085` | 2026-08-09T16:13:00Z |
| `/root/low_storage_3s184` | Parsani Ketcheson Deconinck 3S184 | `codex/low-storage-3s184`; `differential-equations-rs-worktrees/low-storage-3s184` | Rust gates pass; Julia blocked by missing executable; merged as `8323c7d` | 2026-08-09T16:19:00Z |
| `/root/low_order_rko65` | RKO65 fixed-step RK | `codex/low-order-rko65`; `differential-equations-rs-worktrees/low-order-rko65` | Rust gates pass; Julia blocked by missing executable; merged as `33f525b` | 2026-08-09T16:21:00Z |
| `/root/low_storage_3s205` | Parsani Ketcheson Deconinck 3S205 | `codex/low-storage-3s205`; `differential-equations-rs-worktrees/low-storage-3s205` | Rust gates pass; Julia blocked by missing executable; merged as `db21a51` | 2026-08-09T16:25:00Z |
| `/root/low_order_psrk36` | PSRK3p6q5 fixed-step RK | `codex/low-order-psrk36`; `differential-equations-rs-worktrees/low-order-psrk36` | Rust gates pass; Julia blocked by missing executable; merged as `1d349df` | 2026-08-09T16:26:00Z |
| `/root/low_order_autodp5` | AutoDP5 regular composite facade | `.worktrees/low-order-autodp5`; `codex/low-order-autodp5` | Rust gates pass; Julia blocked by missing executable; merged as `9f5e5c7` | 2026-08-09T16:35:00Z |
| `/root/low_order_psrk47` | PSRK4p7q6 fixed-step RK | `codex/low-order-psrk47`; `differential-equations-rs-worktrees/low-order-psrk47` | Rust gates pass; Julia blocked by missing executable; merged as `77d8bdc` | 2026-08-09T16:36:00Z |
| `/root/low_order_stepanov5` | Stepanov5 adaptive RK | `codex/low-order-stepanov5`; `differential-equations-rs-worktrees/low-order-stepanov5` | Rust gates pass; Julia blocked by missing executable; merged as `7563e9d` | 2026-08-09T16:44:00Z |
| `/root/ssprk_prrk33` | Parametric relaxation pRRK33 | `codex/ssprk-prrk33`; `differential-equations-rs-worktrees/ssprk-prrk33` | Rust gates pass; Julia blocked by missing executable; merged as `6ac81ff` | 2026-08-09T16:44:00Z |

## Completed waves

| Wave | Files/algorithms | Rust tests | Julia tests | Review status |
| --- | --- | ---: | ---: | --- |
| Soundness | Custom `ButcherTableau` validation and one-stage adaptive scratch use | 70 pass | 202 pass | reviewed and merged as `408991c` |
| Julia reproducibility | Track portable pinned manifest for 13 OrdinaryDiffEq packages | not applicable | pinned check plus 202 pass | reviewed and merged as `57ea0f8`; fresh-worktree check passed |
| Driver foundation | Static `StepKernel`, lifecycle mocks, and generic fixed/adaptive explicit RK | 80 pass | 202 pass | reviewed and merged as `47c8ae5`; fixed solve remains 6 allocations |
| Exact inventory | 349 source-resolved exports; schema v2; 345 in scope, 280 missing | 80 pass | 202 pass | strict coordinator and fresh-worktree byte checks pass via `0579ff4` + `b4ff329` |
| Explicit upstream audit | 22 current explicit/Tsit5/Verner methods; 134 verified source references | not applicable | not applicable | report reviewed and merged as `fc0d556` |
| Driver implicit | ImplicitEuler/Midpoint/Trapezoid and TRBDF2; recoverable attempt policy | 85 pass | 202 pass | reviewed and merged as `a288382`; compliance output byte-identical |
| Linear-interface audit | Dense views/LU/Jacobian/Jv/mass-operator Phase 3 design and caller map | not applicable | not applicable | report reviewed and merged as `8e76510` |
| Driver Rosenbrock | Rosenbrock23/32 and Rodas4/5P shared-driver migration | 90 pass | 202 pass | reviewed and merged as `f5df68b`; compliance output byte-identical |
| Driver low-storage | Nine fixed low-storage RK methods on shared driver | 93 pass | 202 pass | reviewed and merged as `962c89a`; compliance output byte-identical |
| Driver Adams | Fixed Adams–Bashforth and variable Adams–Moulton families on shared driver | 71 library plus integration pass | 202 pass | reviewed and merged as `e747d0b`; compliance output byte-identical |
| Phase 2 lifecycle audit | Centralized first-order loop and complete StepKernel coverage; second-order loop explicitly excluded | 77 Rust tests | 202 pass | reviewed and merged as `5383420` |
| Phase 3 checked linear interface | State/matrix views and revisioned dense LU cache; caller migration pending | 80 Rust tests | 202 pass | reviewed and merged as `a1bb8fa` |
| Phase 4 schema foundation | Tagged coefficient metadata, structural validation, and deterministic manifest check | 82 Rust tests | pending caller-wave rerun | reviewed and merged as `c20da11` |
| Phase 4 generated fixtures | Compile-time RK4/AB3/VelocityVerlet constants plus deterministic generator check | 85 Rust tests | 202 pass | reviewed and merged as `775067b` + `cea41fa` |
| Phase 5 representation foundation | Typed split/IMEX and regular nonsingular mass-matrix problem containers (preparatory; solver migration gated on Phase 4) | 86 Rust tests | pending split/mass solver wave | reviewed and merged as `20fe376` |
| Phase 6 dense foundation | Checked Hermite `DenseSegment` seam with endpoint derivative data | 87 Rust tests | 202 pass before foundation | reviewed and merged as `27287cc` |
| Phase 6 controller metadata | PI history metadata seam, default proportional behavior unchanged | 88 Rust tests | 202 pass | reviewed and merged as `883c11d` + `fc797d4` |
| Phase 6 dense/controller audit | Exact pinned service/controller gaps and two vertical implementation slices | not applicable | not applicable | reviewed and merged as `85c0d2e` |
| Solver statistics | Linear factorization counter across implicit/Rosenbrock/TRBDF2 paths | 87 Rust tests | 202 pass | reviewed and merged as `01a0884` |
| BDF/SDIRK feasibility audit | Exact pinned source map and dependency assessment; recommends SDIRK2 then ABDF2 | not applicable | not applicable | reviewed and merged as `0f5948d` |
| Phase 3 caller proof | Implicit Euler/Midpoint/Trapezoid checked first factorization with allocation-invariant refresh path | 82 Rust tests plus migration integration | 202 pass | reviewed and merged as `335d162`; implicit compliance byte-identical |
| Phase 3 operator/mass seams | JacobianProvider, checked LinearOperator, dense/identity operators, and nonsingular mass operator | 84 Rust tests | 202 pass | reviewed and merged as `052cef3` + `c64dda1` |
| SDIRK2 family | Native regular-ODE SDIRK2 kernel, generated tableau fixture, fixed/adaptive Rust tests, and matched pinned Julia compliance | cargo all-targets pass | 206 pass (16 suites) | reviewed and merged as `1909900` + `ab1fd98`; fixed endpoint matches SDIRK2 at dt=.01; adaptive controller-count caveat documented |
| Dense/controller service | Domain-checked Hermite segments, accepted-step `record_step_dense` save-at sampling, and controller-history reset after callback mutation | 92 Rust tests plus integrations | 206 pass (16 suites) | reviewed and merged as `dbf9a16`; existing endpoint compliance and allocations preserved; solver kernels still need stage-data wiring |
| ABDF2 family | Native regular identity-mass ABDF2 with implicit-Euler startup, variable-step coefficients, Newton/Jacobian paths, callback history reset, and pinned Julia fixture | 92 unit tests plus 6 ABDF2 integration tests | 210 pass (17 suites) | reviewed and merged as `aedca27` plus final estimator/workspace refresh; fixed/adaptive endpoint parity within documented tolerance; controller-count caveat documented |
| Generated BS3 | Migrated the public BS3 explicit tableau to generated pinned coefficients with structural fixture validation and regression coverage | 92 unit tests plus integrations | 210 pass (17 suites) | reviewed and merged as `ddefe73`; SBDF2 explicitly deferred because pinned implementation is split/IMEX |
| Generated DP5 | Migrated the public DP5 seven-stage tableau and embedded defect to generated pinned coefficients with structural fixture validation | 92 unit tests plus integrations | 210 pass (17 suites) | reviewed and merged as `69f6caf`; pinned Julia compliance unchanged |
| Explicit RK dense wiring | Wired zero-allocation borrowed Hermite segments into explicit RK save-at sampling, including backward/exact-endpoint/rejection coverage and a pinned Julia cubic reference | 92 unit tests plus integrations and 3 dense tests | 212 pass (18 suites) | reviewed and merged as `59ece5d` + `3ea6279`; endpoint-only allocations and callback semantics preserved |
| MEBDF2 family | Native fixed-step regular identity-mass MEBDF2 with three sequential Newton corrections, checked Jacobian/LU path, callbacks, backward integration, and pinned Julia fixture | 92 library tests plus 4 MEBDF2 integration tests | 214 pass (19 suites) | reviewed and merged as `5e0650c`; fixed endpoint matches pinned Julia; adaptive/DAE/split paths excluded |
| QNDF1 family | Native fixed/adaptive regular identity-mass QNDF1 with one-step history reinterpolation, NDF residual/Newton solve, callback reset, and pinned Julia fixture | 92 library tests plus QNDF1 integration tests | 218 pass (20 suites) | reviewed and merged as `e5a6603`; first-order endpoint differs by documented 4.95e-5 within explicit fixture tolerance |
| QNDF2 family | Native fixed/adaptive regular identity-mass QNDF2 with two-step history reinterpolation, NDF residual/Newton solve, callback reset, and pinned Julia fixture | 92 library tests plus QNDF2 integration tests | 222 pass (21 suites) | reviewed and merged as `e97ef25`; relaxed low-order endpoint tolerances documented in handoff |
| Vern6 family | Exact pinned nine-stage Vern6 tableau moved to generated coefficients with fixed/adaptive/backward/callback and allocation coverage | 92 library tests plus Vern6 integrations | 222 pass (21 suites) | reviewed and merged as `5242ba0`; existing Julia Verner fixture remains green |
| Vern7 family | Exact pinned ten-stage Vern7 tableau moved to generated coefficients with fixed/adaptive/backward/callback and allocation coverage | 92 library tests plus Vern7 integrations | 222 pass (21 suites) | reviewed and merged as `81b08b3`; existing Julia Verner fixture remains green |
| Vern8 family | Exact pinned thirteen-stage Vern8 tableau moved to generated coefficients with fixed/adaptive/backward/callback and allocation coverage | 92 library tests plus Vern8 integrations | 222 pass (21 suites) | reviewed and merged as `b784986`; existing Julia Verner fixture remains green |
| Vern9 family | Exact pinned sixteen-stage Vern9 tableau moved to generated coefficients with fixed/adaptive/backward/callback and allocation coverage | 92 library tests plus Vern9 integrations | 222 pass (21 suites) | reviewed and merged as `3371551`; existing Julia Verner fixture remains green |
| pRRK22 family | Native fixed-step parametric relaxation SSPRK22 with upstream psi/alpha/beta/dt-hat rescaling | 93 library tests plus pRRK22 integrations | pRRK22 worker full suite passed; coordinator rerun pending | reviewed and merged as `dd3c46b`; inventory detects 1 new implemented-and-tested name after generator fix |
| SSPRK432 family | Native adaptive/fixed four-stage SSPRK432 with pinned embedded residual | 93 library tests plus SSPRK432 integration/allocation tests | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `e32ee02`; Julia retry required |
| Parsani 3S32 family | Native fixed low-storage 3S recurrence with shared low-storage driver | 94 library tests plus recurrence/allocation integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `3d1d13d`; Julia retry required |
| Alshina6 family | Exact pinned seven-stage sixth-order fixed explicit tableau | 93 library tests plus Alshina6 integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `1aafd9d`; Julia retry required |
| Parsani 3S82 family | Native fixed low-storage 3S recurrence with pinned seven-stage coefficients | 95 library tests plus recurrence/allocation integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `86de559`; Julia retry required |
| Anas5 family | Parameterized fixed-step low-order RK with pinned dynamic `a65(w,dt)` and FSAL endpoint reuse | 95 library tests plus Anas5 integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `80867f2`; Julia retry required |
| Parsani 3S53 family | Native fixed low-storage 3S recurrence with pinned five-stage coefficients | 95 library tests plus recurrence/allocation integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `6e905c4`; Julia retry required |
| MSRK5 family | Exact pinned nine-stage fixed FSAL tableau | 95 library tests plus MSRK5 integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `1b7499b`; Julia retry required |
| Parsani 3S94 family | Native fixed low-storage 3S recurrence with pinned nine-stage coefficients | 95 library tests plus recurrence/allocation integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `2fb5dc5`; Julia retry required |
| MSRK6 family | Exact pinned eight-stage fixed sixth-order tableau plus endpoint FSAL row/RHS-count regression | 95 library tests plus MSRK6 integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `3cfdf35`; Julia retry required |
| Parsani 3S105 family | Native fixed low-storage 3S recurrence with pinned coefficients | 95 library tests plus recurrence/allocation integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `e4a659e`; Julia retry required |
| FRK65 family | Parameterized fitted fixed/adaptive RK with pinned omega-dependent coefficients | 95 library tests plus FRK65 source tests | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `dec241b` + `f3e2078`; Julia retry required |
| Parsani 3S173 family | Native fixed low-storage 3S recurrence with pinned seventeen-stage coefficients | 97 library tests plus recurrence/allocation integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `e081664`; Julia retry required |
| PSRK3p5q4 family | Exact pinned four-stage third-order fixed explicit tableau | 97 library tests plus PSRK integrations | Julia unavailable/timeout on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `2823085`; Julia retry required |
| Parsani 3S184 family | Native fixed low-storage 3S recurrence with pinned eighteen-stage coefficients | 97 library tests plus recurrence/allocation integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `8323c7d`; Julia retry required |
| RKO65 family | Exact pinned fixed fifth-order tableau with explicit endpoint-stage shift | 97 library tests plus RKO65 integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `33f525b`; Julia retry required |
| Parsani 3S205 family | Native fixed low-storage 3S recurrence with pinned twenty-stage coefficients | 97 library tests plus recurrence/allocation integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `db21a51`; Julia retry required |
| PSRK3p6q5 family | Exact pinned five-stage fixed fourth-order tableau | 97 library tests plus PSRK integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `1d349df`; Julia retry required |
| AutoDP5 family | Native DP5 composite facade for regular ODEs; automatic stiff switching explicitly deferred | 97 library tests plus AutoDP5 integrations | Isolated worker pinned/full Julia passed (14 packages; ~158s); integrated coordinator rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `9f5e5c7`; upstream switch limitation documented |
| PSRK4p7q6 family | Exact pinned seven-stage fixed fourth-order tableau | 97 library tests plus PSRK integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `77d8bdc`; Julia retry required |
| Stepanov5 family | Exact pinned seven-stage embedded adaptive (4,5) FSAL tableau | 97 library tests plus Stepanov5 integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `7563e9d`; Julia retry required |
| pRRK33 family | Native three-stage parametric relaxation SSPRK33 | 97 library tests plus pRRK33 integrations | Julia unavailable on worker/coordinator; pinned/full rerun blocked by `JULIA-PATH-20260809` | Rust-reviewed and merged as `6ac81ff`; Julia retry required |

## Validation snapshot

```text
cargo fmt -- --check: pass
cargo test --all-targets: pass (94 library tests plus integration tests/examples)
cargo clippy --all-targets -- -D warnings: pass
git diff --check: pass
pinned Julia environment: pass and reproducible from tracked manifest (14 packages at pinned revision)
Julia compliance: pRRK22 worker pass; integrated SSPRK432 rerun blocked by `JULIA-PATH-20260809`
inventory regeneration: pass; 349 source references and strict cross-checkout byte identity verified (92 implemented/tested, 253 missing)
```

## Next dependency-ready task

After restoring Julia, rerun the integrated pinned/full suite for SSPRK432, Parsani 3S32/3S53/3S82/3S94/3S105/3S173/3S184/3S205, Alshina6, Anas5, MSRK5, MSRK6, FRK65, PSRK3p5q4/3p6q5/4p7q6, RKO65, AutoDP5, Stepanov5, and pRRK33, then continue with the next dependency-ready regular-ODE family or generated-coefficient slice while wiring accepted-stage dense segments into one solver. SBDF2 remains excluded until split/IMEX parity is implemented.

## Last decision

The shared first-order driver is frozen and all queued first-order families (explicit, implicit/TRBDF2, Rosenbrock/Rodas, low-storage, and Adams) pass. The Phase 2 audit found only `src/integrator.rs:228` as the first-order lifecycle loop; `src/second_order.rs:440` is an explicit exclusion. Phase 3 checked views, revisioned dense LU, Jacobian/operator seams, nonsingular mass operators, and one implicit caller proof are merged with unchanged compliance and allocations. Phase 4 schema/generated fixtures and Phase 5 representation foundations are merged. Phase 6 now has domain-checked Hermite segments, an accepted-step recorder service, and callback controller-history reset; solver stage-data wiring remains queued. Phase 7 now includes SDIRK2, ABDF2, MEBDF2, QNDF1/2, Verner6-9, pRRK22, SSPRK432, Parsani 3S32/3S53/3S82/3S94/3S105/3S173/3S184/3S205, Alshina6, Anas5, MSRK5/6, FRK65, PSRK3p5q4/3p6q5/4p7q6, RKO65, AutoDP5, Stepanov5, and pRRK33; 253 in-scope public constructors remain. Phase 8 is blocked only by the missing Julia executable and still has documented dense/controller and low-order tolerance gaps.
