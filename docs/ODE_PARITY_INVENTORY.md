# Regular ODE parity inventory

This is a generated summary of native solver exports at SciML/OrdinaryDiffEq.jl
revision `211142263781255a9aa2f910f6760b9f18ec29c8` under the scope in
[UPSTREAM_SCOPE.md](UPSTREAM_SCOPE.md). Regenerate it with:

```powershell
./scripts/generate_ode_inventory.ps1 -UpstreamPath <path-to-OrdinaryDiffEq.jl>
```

The full machine-readable records, including upstream definition paths and line
numbers, problem representations, fixed/adaptive behavior, Jacobian, linear-solver,
dense-output and controller requirements, aliases, exclusions, and current Rust and
Julia status, are in [ode_algorithm_inventory.json](ode_algorithm_inventory.json)
and [ode_algorithm_inventory.csv](ode_algorithm_inventory.csv).

## Totals

- Public solver names inspected: **349**.
- In-scope regular ODE names: **345**
  (333 canonical/composite constructors and
  12 public aliases).
- Implemented and detected in matched Julia tests: **84**.
- Missing in-scope public names: **261**.
- Explicitly excluded public names: **4**.

Aliases are public parity obligations but do not require a second numerical kernel.

## Family status

| Family | In scope | Implemented + Julia-tested | Missing names |
| --- | ---: | ---: | ---: |
| Adams multistep | 13 | 12 | 1 |
| approximate-matrix-factorization wrapper | 1 | 0 | 1 |
| automatic/default composite | 2 | 0 | 2 |
| BDF and IMEX multistep | 15 | 4 | 11 |
| explicit Runge-Kutta | 2 | 1 | 1 |
| exponential Runge-Kutta | 17 | 0 | 17 |
| extrapolation | 7 | 0 | 7 |
| fully implicit Runge-Kutta | 5 | 0 | 5 |
| high-order explicit Runge-Kutta | 16 | 4 | 12 |
| IMEX multistep | 2 | 0 | 2 |
| linear and Lie-group methods | 18 | 0 | 18 |
| low-order explicit Runge-Kutta | 28 | 21 | 7 |
| low-storage explicit Runge-Kutta | 44 | 15 | 29 |
| multirate and MRI-GARK | 9 | 0 | 9 |
| Nordsieck variable-order multistep | 4 | 0 | 4 |
| parallel diagonally implicit Runge-Kutta | 1 | 0 | 1 |
| parallel explicit Runge-Kutta | 1 | 0 | 1 |
| QPRK explicit Runge-Kutta | 1 | 0 | 1 |
| Rosenbrock and Rosenbrock-W | 40 | 4 | 36 |
| Runge-Kutta interval prediction | 1 | 0 | 1 |
| Runge-Kutta-Nystrom | 17 | 0 | 17 |
| SDIRK, ESDIRK, and additive IMEX RK | 39 | 5 | 34 |
| second-order structural dynamics | 2 | 0 | 2 |
| SIMD explicit Runge-Kutta | 3 | 0 | 3 |
| stabilized explicit Runge-Kutta | 13 | 0 | 13 |
| stabilized implicit Runge-Kutta | 1 | 0 | 1 |
| strong-stability-preserving Runge-Kutta | 21 | 14 | 7 |
| symplectic and partitioned Runge-Kutta | 18 | 4 | 14 |
| Taylor series | 3 | 0 | 3 |
| user-tableau explicit Runge-Kutta | 1 | 0 | 1 |

## Remaining solver names by family

This is the implementation handoff list. Each entry remains in scope and lacks
either a public Rust implementation or a detected matched Julia compliance
case. Required features and exact upstream source locations are available in
the JSON/CSV records.

### Adams multistep (1)

- `VCABM` — OrdinaryDiffEqAdamsBashforthMoulton; ODEProblem

### approximate-matrix-factorization wrapper (1)

- `AMF` — OrdinaryDiffEqAMF; ODEProblem with structured ODEFunction

### automatic/default composite (2)

- `DefaultImplicitODEAlgorithm` — OrdinaryDiffEqDefault; ODEProblem
- `DefaultODEAlgorithm` — OrdinaryDiffEqDefault; ODEProblem

### BDF and IMEX multistep (11)

- `FBDF` — OrdinaryDiffEqBDF; ODEProblem
- `IMEXEuler` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem
- `IMEXEulerARK` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem
- `QBDF` — OrdinaryDiffEqBDF; ODEProblem
- `QBDF1` — OrdinaryDiffEqBDF; ODEProblem
- `QBDF2` — OrdinaryDiffEqBDF; ODEProblem
- `QNDF` — OrdinaryDiffEqBDF; ODEProblem
- `SBDF` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem
- `SBDF2` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem
- `SBDF3` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem
- `SBDF4` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem

### explicit Runge-Kutta (1)

- `AutoTsit5` — OrdinaryDiffEqTsit5; ODEProblem

### exponential Runge-Kutta (17)

- `EPIRK4s3A` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `EPIRK4s3B` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `EPIRK5P1` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `EPIRK5P2` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `EPIRK5s3` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `ETD1` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `ETD2` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `ETDRK2` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `ETDRK3` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `ETDRK4` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `Exp4` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `Exprb32` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `Exprb43` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `EXPRB53s3` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `HochOst4` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `LawsonEuler` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `NorsettEuler` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem

### extrapolation (7)

- `AitkenNeville` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ExtrapolationMidpointDeuflhard` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ExtrapolationMidpointHairerWanner` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ImplicitDeuflhardExtrapolation` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ImplicitEulerBarycentricExtrapolation` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ImplicitEulerExtrapolation` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ImplicitHairerWannerExtrapolation` — OrdinaryDiffEqExtrapolation; ODEProblem

### fully implicit Runge-Kutta (5)

- `AdaptiveRadau` — OrdinaryDiffEqFIRK; ODEProblem
- `GaussLegendre` — OrdinaryDiffEqFIRK; ODEProblem
- `RadauIIA3` — OrdinaryDiffEqFIRK; ODEProblem
- `RadauIIA5` — OrdinaryDiffEqFIRK; ODEProblem
- `RadauIIA9` — OrdinaryDiffEqFIRK; ODEProblem

### high-order explicit Runge-Kutta (12)

- `AutoVern6` — OrdinaryDiffEqVerner; ODEProblem
- `AutoVern7` — OrdinaryDiffEqVerner; ODEProblem
- `AutoVern8` — OrdinaryDiffEqVerner; ODEProblem
- `AutoVern9` — OrdinaryDiffEqVerner; ODEProblem
- `DP8` — OrdinaryDiffEqHighOrderRK; ODEProblem
- `Feagin10` — OrdinaryDiffEqFeagin; ODEProblem
- `Feagin12` — OrdinaryDiffEqFeagin; ODEProblem
- `Feagin14` — OrdinaryDiffEqFeagin; ODEProblem
- `PFRK87` — OrdinaryDiffEqHighOrderRK; ODEProblem
- `RKV76IIa` — OrdinaryDiffEqVerner; ODEProblem
- `TanYam7` — OrdinaryDiffEqHighOrderRK; ODEProblem
- `TsitPap8` — OrdinaryDiffEqHighOrderRK; ODEProblem

### IMEX multistep (2)

- `CNAB2` — OrdinaryDiffEqIMEXMultistep; ODEProblem or SplitODEProblem
- `CNLF2` — OrdinaryDiffEqIMEXMultistep; ODEProblem or SplitODEProblem

### linear and Lie-group methods (18)

- `CayleyEuler` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `CG2` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `CG3` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `CG4a` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `LieEuler` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `LieRK4` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `LinearExponential` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusAdapt4` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusGauss4` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusGL4` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusGL6` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusGL8` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusLeapfrog` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusMidpoint` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusNC6` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusNC8` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `RKMK2` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `RKMK4` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction

### low-order explicit Runge-Kutta (7)

- `AutoDP5` — OrdinaryDiffEqLowOrderRK; ODEProblem
- `PSRK3p6q5` — OrdinaryDiffEqLowOrderRK; ODEProblem
- `PSRK4p7q6` — OrdinaryDiffEqLowOrderRK; ODEProblem
- `RKO65` — OrdinaryDiffEqLowOrderRK; ODEProblem
- `SIR54` — OrdinaryDiffEqLowOrderRK; ODEProblem
- `SplitEuler` — OrdinaryDiffEqLowOrderRK; ODEProblem or SplitODEProblem
- `Stepanov5` — OrdinaryDiffEqLowOrderRK; ODEProblem

### low-storage explicit Runge-Kutta (29)

- `CFRLDDRK64` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK43_2` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK54_3C` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK54_3C_3R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK54_3M_3R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK54_3M_4R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK54_3N_3R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK54_3N_4R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK65_4M_4R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK75_4M_5R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK85_4C_3R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK85_4FM_4R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK85_4M_3R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK85_4P_3R` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK95_4C` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK95_4M` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `CKLLSRK95_4S` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `ParsaniKetchesonDeconinck3S184` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `ParsaniKetchesonDeconinck3S205` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `RDPK3Sp35` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `RDPK3Sp49` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `RDPK3Sp510` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `RDPK3SpFSAL35` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `RDPK3SpFSAL49` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `RDPK3SpFSAL510` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `RK46NL` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `SHLDDRK_2N` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `SHLDDRK52` — OrdinaryDiffEqLowStorageRK; ODEProblem
- `TSLDDRK74` — OrdinaryDiffEqLowStorageRK; ODEProblem

### multirate and MRI-GARK (9)

- `MIS` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRAB` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MREEF` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKERK22a` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKERK22b` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKERK33a` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKERK45a` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKESDIRK34a` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKIRK21a` — OrdinaryDiffEqMultirate; SplitODEProblem

### Nordsieck variable-order multistep (4)

- `AN5` — OrdinaryDiffEqNordsieck; ODEProblem
- `JVODE` — OrdinaryDiffEqNordsieck; ODEProblem
- `JVODE_Adams` — OrdinaryDiffEqNordsieck; ODEProblem
- `JVODE_BDF` — OrdinaryDiffEqNordsieck; ODEProblem

### parallel diagonally implicit Runge-Kutta (1)

- `PDIRK44` — OrdinaryDiffEqPDIRK; ODEProblem

### parallel explicit Runge-Kutta (1)

- `KuttaPRK2p5` — OrdinaryDiffEqPRK; ODEProblem

### QPRK explicit Runge-Kutta (1)

- `QPRK98` — OrdinaryDiffEqQPRK; ODEProblem

### Rosenbrock and Rosenbrock-W (36)

- `GRK4A` — OrdinaryDiffEqRosenbrock; ODEProblem
- `GRK4T` — OrdinaryDiffEqRosenbrock; ODEProblem
- `HybridExplicitImplicitRK` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas23W` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas3` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas3d` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas3P` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas42` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas4P` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas4P2` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas4PW` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas5` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas5Pe` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas5Pr` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Rodas6P` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROK4a` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS2` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS2PR` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS2S` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS3` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS34PRw` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS34PW1a` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS34PW1b` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS34PW2` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS34PW3` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS3P` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS3PR` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS3PRL` — OrdinaryDiffEqRosenbrock; ODEProblem
- `ROS3PRL2` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Ros4LStab` — OrdinaryDiffEqRosenbrock; ODEProblem
- `RosenbrockW6S4OS` — OrdinaryDiffEqRosenbrock; ODEProblem
- `RosShamp4` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Scholz4_7` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Tsit5DA` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Veldd4` — OrdinaryDiffEqRosenbrock; ODEProblem
- `Velds4` — OrdinaryDiffEqRosenbrock; ODEProblem

### Runge-Kutta interval prediction (1)

- `RKIP` — OrdinaryDiffEqRKIP; ODEProblem

### Runge-Kutta-Nystrom (17)

- `DPRKN12` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `DPRKN4` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `DPRKN5` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `DPRKN6` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `DPRKN6FM` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `DPRKN8` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `ERKN4` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `ERKN5` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `ERKN7` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `FineRKN4` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `FineRKN5` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `IRKN3` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `IRKN4` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `Nystrom4` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `Nystrom4VelocityIndependent` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `Nystrom5VelocityIndependent` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem
- `RKN4` — OrdinaryDiffEqRKN; SecondOrderODEProblem or DynamicalODEProblem

### SDIRK, ESDIRK, and additive IMEX RK (34)

- `ARS222` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `ARS232` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `ARS343` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `ARS443` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `BHR553` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `Cash4` — OrdinaryDiffEqSDIRK; ODEProblem
- `CFNLIRK3` — OrdinaryDiffEqSDIRK; ODEProblem
- `ESDIRK325L2SA` — OrdinaryDiffEqSDIRK; ODEProblem
- `ESDIRK436L2SA2` — OrdinaryDiffEqSDIRK; ODEProblem
- `ESDIRK437L2SA` — OrdinaryDiffEqSDIRK; ODEProblem
- `ESDIRK547L2SA2` — OrdinaryDiffEqSDIRK; ODEProblem
- `ESDIRK54I8L2SA` — OrdinaryDiffEqSDIRK; ODEProblem
- `ESDIRK659L2SA` — OrdinaryDiffEqSDIRK; ODEProblem
- `Hairer4` — OrdinaryDiffEqSDIRK; ODEProblem
- `Hairer42` — OrdinaryDiffEqSDIRK; ODEProblem
- `IMEXSSP222` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `IMEXSSP2322` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `IMEXSSP3332` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `IMEXSSP3433` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `KenCarp3` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `KenCarp4` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `KenCarp47` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `KenCarp5` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `KenCarp58` — OrdinaryDiffEqSDIRK; ODEProblem or SplitODEProblem
- `Kvaerno3` — OrdinaryDiffEqSDIRK; ODEProblem
- `Kvaerno4` — OrdinaryDiffEqSDIRK; ODEProblem
- `Kvaerno5` — OrdinaryDiffEqSDIRK; ODEProblem
- `SDIRK22` — OrdinaryDiffEqSDIRK; ODEProblem
- `SFSDIRK4` — OrdinaryDiffEqSDIRK; ODEProblem
- `SFSDIRK5` — OrdinaryDiffEqSDIRK; ODEProblem
- `SFSDIRK6` — OrdinaryDiffEqSDIRK; ODEProblem
- `SFSDIRK7` — OrdinaryDiffEqSDIRK; ODEProblem
- `SFSDIRK8` — OrdinaryDiffEqSDIRK; ODEProblem
- `SSPSDIRK2` — OrdinaryDiffEqSDIRK; ODEProblem

### second-order structural dynamics (2)

- `GeneralizedAlpha` — OrdinaryDiffEqNewmark; SecondOrderODEProblem
- `NewmarkBeta` — OrdinaryDiffEqNewmark; SecondOrderODEProblem

### SIMD explicit Runge-Kutta (3)

- `MER5v2` — OrdinaryDiffEqSIMDRK; ODEProblem
- `MER6v2` — OrdinaryDiffEqSIMDRK; ODEProblem
- `RK6v4` — OrdinaryDiffEqSIMDRK; ODEProblem

### stabilized explicit Runge-Kutta (13)

- `ESERK4` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `ESERK5` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `RKC` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `RKG1` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `RKG2` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `RKL1` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `RKL2` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `RKMC2` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `ROCK2` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `ROCK4` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `SERK2` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `TSRKC2` — OrdinaryDiffEqStabilizedRK; ODEProblem
- `TSRKC3` — OrdinaryDiffEqStabilizedRK; ODEProblem

### stabilized implicit Runge-Kutta (1)

- `IRKC` — OrdinaryDiffEqStabilizedIRK; ODEProblem

### strong-stability-preserving Runge-Kutta (7)

- `KYK2014DGSSPRK_3S2` — OrdinaryDiffEqSSPRK; ODEProblem
- `KYKSSPRK42` — OrdinaryDiffEqSSPRK; ODEProblem
- `pRRK33` — OrdinaryDiffEqSSPRK; ODEProblem
- `pRRK54` — OrdinaryDiffEqSSPRK; ODEProblem
- `SSPRK932` — OrdinaryDiffEqSSPRK; ODEProblem
- `SSPRKMSVS32` — OrdinaryDiffEqSSPRK; ODEProblem
- `SSPRKMSVS43` — OrdinaryDiffEqSSPRK; ODEProblem

### symplectic and partitioned Runge-Kutta (14)

- `CalvoSanz4` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `CandyRoz4` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `KahanLi6` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `KahanLi8` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `McAte2` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `McAte3` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `McAte4` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `McAte42` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `McAte5` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `McAte8` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `PseudoVerletLeapfrog` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `Ruth3` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `SofSpa10` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem
- `Yoshida6` — OrdinaryDiffEqSymplecticRK; DynamicalODEProblem

### Taylor series (3)

- `ExplicitTaylor` — OrdinaryDiffEqTaylorSeries; ODEProblem
- `ExplicitTaylor2` — OrdinaryDiffEqTaylorSeries; ODEProblem
- `ExplicitTaylorAdaptiveOrder` — OrdinaryDiffEqTaylorSeries; ODEProblem

### user-tableau explicit Runge-Kutta (1)

- `ExplicitRK` — OrdinaryDiffEqExplicitRK; ODEProblem

## Aliases

| Public name | Kind | Canonical target | Package |
| --- | --- | --- | --- |
| `ETD1` | exact-alias | `NorsettEuler` | OrdinaryDiffEqExponentialRK |
| `IMEXEuler` | configured-alias | `SBDF(order=1)` | OrdinaryDiffEqBDF |
| `IMEXEulerARK` | configured-alias | `SBDF(order=1, ark=true)` | OrdinaryDiffEqBDF |
| `JVODE_Adams` | configured-alias | `JVODE(:Adams)` | OrdinaryDiffEqNordsieck |
| `JVODE_BDF` | configured-alias | `JVODE(:BDF)` | OrdinaryDiffEqNordsieck |
| `QBDF` | configured-alias | `QNDF(kappa=(0,0,0,0,0))` | OrdinaryDiffEqBDF |
| `QBDF1` | configured-alias | `QNDF1(kappa=0)` | OrdinaryDiffEqBDF |
| `QBDF2` | configured-alias | `QNDF2(kappa=0)` | OrdinaryDiffEqBDF |
| `SBDF2` | configured-alias | `SBDF(order=2)` | OrdinaryDiffEqBDF |
| `SBDF3` | configured-alias | `SBDF(order=3)` | OrdinaryDiffEqBDF |
| `SBDF4` | configured-alias | `SBDF(order=4)` | OrdinaryDiffEqBDF |
| `Tsit5DA` | configured-alias | `HybridExplicitImplicitRK(Tsit5DATableau, order=5)` | OrdinaryDiffEqRosenbrock |

## Explicit exclusions

| Public name | Package | Rationale |
| --- | --- | --- |
| `DABDF2` | OrdinaryDiffEqBDF | DAE residual-form algorithm; DAE-only behavior is outside regular ODE scope. |
| `DFBDF` | OrdinaryDiffEqBDF | DAE residual-form algorithm; DAE-only behavior is outside regular ODE scope. |
| `DImplicitEuler` | OrdinaryDiffEqBDF | DAE residual-form algorithm; DAE-only behavior is outside regular ODE scope. |
| `FunctionMap` | OrdinaryDiffEqFunctionMap | Discrete dynamical-system map, not a continuous initial-value ODE solver. |

Package-level exclusions from [UPSTREAM_SCOPE.md](UPSTREAM_SCOPE.md), such as
DelayDiffEq, StochasticDiffEq, external wrappers, BVP, PDE, and steady-state
solvers, are not expanded into per-algorithm rows because they are not part of
the OrdinaryDiffEq native ODE solver export surface.

## Classified support-only subpackages

| Package | Why it has no solver rows |
| --- | --- |
| `OrdinaryDiffEqCore` | Shared integrator abstractions and internal composite types; no public numerical method constructors. |
| `OrdinaryDiffEqDifferentiation` | Jacobian, time-derivative, and differentiation support used by solver packages. |
| `OrdinaryDiffEqExplicitTableaus` | Butcher-tableau data and constructors, not solver algorithms. |
| `OrdinaryDiffEqImplicitTableaus` | Implicit tableau data and constructors, not solver algorithms. |
| `OrdinaryDiffEqNonlinearSolve` | Nonlinear-solver and DAE-initialization support, not time-integration algorithms. |
| `OrdinaryDiffEqRosenbrockTableaus` | Rosenbrock tableau data and constructors, not solver algorithms. |

## Audited non-algorithm exports

| Export | Package | Why it has no solver row |
| --- | --- | --- |
| `Predictor` | `OrdinaryDiffEqSDIRK` | Nonlinear-stage predictor enum namespace, not an ODE algorithm constructor. |

## Interpretation notes and uncertainties

- The inventory treats package exports as the public algorithm surface; internal, unexported experimental types are not parity targets.
- ETD1 is the only exact exported type alias found at the pinned revision. Named functions that return a configured canonical algorithm are recorded as configured aliases. Auto* and Default* names are composite constructors.
- AMF is counted as a native wrapper constructor because it dispatches back into OrdinaryDiffEq with native Rosenbrock-W methods.
- Rust status is detected by normalized public Rust type name plus a corresponding OrdinaryDiffEq import in tests/julia; numerical test quality remains a review concern.
- Implemented status measures public algorithm-name coverage only. It does not establish parity for every upstream problem representation or shared feature; consult problem_representation, required_features, and FEATURE_COVERAGE.md separately.
- Nonsingular mass-matrix behavior of dual ODE/DAE methods is included, while residual-form DAE constructors and singular-mass-matrix behavior are excluded.
