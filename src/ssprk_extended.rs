//! Strong-stability-preserving Runge--Kutta methods, including fixed-step
//! families and the adaptive SSPRK432 embedded pair.
//!
//! The coefficients below are algebraically equivalent Butcher forms of the
//! Shu--Osher implementations in OrdinaryDiffEqSSPRK.  Keeping them in the
//! shared explicit RK engine gives these methods the same output and callback
//! handling as the other one-step solvers without pretending to expose
//! OrdinaryDiffEq's stage/step limiter or threading options.

// The transformed tableaus retain the full f64 results of combining the
// upstream decimal Shu--Osher coefficients.
#![allow(clippy::excessive_precision)]

use crate::SolverStats;
use crate::explicit_rk::{ButcherTableau, ExplicitRungeKutta};
use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::solution::TrajectoryRecorder;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

const EMPTY: &[f64] = &[];

macro_rules! fixed_ssprk {
    ($algorithm:ident, $tableau:ident, $order:expr, $nodes:ident, $rows:ident, $weights:ident) => {
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $algorithm;

        struct $tableau;

        impl ButcherTableau for $tableau {
            const NODES: &'static [f64] = $nodes;
            const COEFFICIENTS: &'static [&'static [f64]] = $rows;
            const WEIGHTS: &'static [f64] = $weights;
            const ERROR_WEIGHTS: Option<&'static [f64]> = None;
            const ORDER: usize = $order;
            const FSAL: bool = false;
        }

        impl OdeAlgorithm for $algorithm {
            fn solve<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                ExplicitRungeKutta::<$tableau>::new().solve(problem, options)
            }
        }
    };
}

/// Adaptive SSPRK432 uses the same four-stage, third-order main method as
/// [`SspRk43`], but retains the full third/second-order embedded residual
/// from OrdinaryDiffEqSSPRK's dedicated constructor.  The shared explicit
/// kernel applies this tableau for both fixed and adaptive stepping.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SspRk432;

struct SspRk432Tableau;

impl ButcherTableau for SspRk432Tableau {
    const NODES: &'static [f64] = &[0.0, 0.5, 1.0, 0.5];
    const COEFFICIENTS: &'static [&'static [f64]] = &[
        EMPTY,
        &[0.5],
        &[0.5, 0.5],
        &[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
    ];
    const WEIGHTS: &'static [f64] = &[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5];
    // utilde = uprev + dt * (f₁ + f₂ + f₃) / 3, while the accepted state is
    // uprev + dt * (f₁ + f₂ + f₃) / 6 + dt * f₄ / 2.  The sign is immaterial
    // to the norm, but this is the conventional high-minus-low difference.
    const ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0, 0.5]);
    const ORDER: usize = 3;
    const FSAL: bool = false;
}

impl OdeAlgorithm for SspRk432 {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        ExplicitRungeKutta::<SspRk432Tableau>::new().solve(problem, options)
    }
}

// SSPRK53 (Ruuth 2006).
const SSPRK53_A2: &[f64] = &[0.377_268_915_331_368_03];
const SSPRK53_A3: &[f64] = &[0.377_268_915_331_368_03, 0.377_268_915_331_368_03];
const SSPRK53_A4: &[f64] = &[
    0.242_995_220_537_395_86,
    0.242_995_220_537_395_86,
    0.242_995_220_537_396,
];
const SSPRK53_A5: &[f64] = &[
    0.153_589_067_695_126_5,
    0.153_589_067_695_126_5,
    0.153_589_067_695_126_6,
    0.238_458_932_846_29,
];
const SSPRK53_A: &[&[f64]] = &[EMPTY, SSPRK53_A2, SSPRK53_A3, SSPRK53_A4, SSPRK53_A5];
const SSPRK53_B: &[f64] = &[
    0.206_734_020_864_804_47,
    0.206_734_020_864_804_47,
    0.117_097_251_841_844_12,
    0.181_802_560_120_139_43,
    0.287_632_146_308_408,
];
const SSPRK53_C: &[f64] = &[
    0.0,
    0.377_268_915_331_368,
    0.754_537_830_662_736,
    0.728_985_661_612_188,
    0.699_226_135_931_67,
];
fixed_ssprk!(SspRk53, SspRk53Tableau, 3, SSPRK53_C, SSPRK53_A, SSPRK53_B);

// Low-storage SSPRK53_2N1 (Higueras and Roldan 2018).
const SSPRK53_2N1_A2: &[f64] = &[0.443_568_244_942_995_02];
const SSPRK53_2N1_A3: &[f64] = &[0.443_568_244_942_995_02, 0.291_111_420_073_766];
const SSPRK53_2N1_A4: &[f64] = &[
    0.443_568_244_942_995_02,
    0.291_111_420_073_766,
    0.270_612_601_278_217_01,
];
const SSPRK53_2N1_A5: &[f64] = &[
    0.190_111_792_195_290_81,
    0.124_769_332_407_580_91,
    0.115_983_610_653_289_95,
    0.110_577_759_392_786,
];
const SSPRK53_2N1_A: &[&[f64]] = &[
    EMPTY,
    SSPRK53_2N1_A2,
    SSPRK53_2N1_A3,
    SSPRK53_2N1_A4,
    SSPRK53_2N1_A5,
];
const SSPRK53_2N1_B: &[f64] = &[
    0.190_111_792_195_290_81,
    0.124_769_332_407_580_91,
    0.115_983_610_653_289_95,
    0.110_577_759_392_786,
    0.458_557_505_351_052,
];
const SSPRK53_2N1_C: &[f64] = &[
    0.0,
    0.443_568_244_942_995,
    0.734_679_665_016_762,
    1.005_292_266_294_979,
    0.541_442_494_648_948,
];
fixed_ssprk!(
    SspRk53TwoN1,
    SspRk53TwoN1Tableau,
    3,
    SSPRK53_2N1_C,
    SSPRK53_2N1_A,
    SSPRK53_2N1_B
);

/// Parametric relaxation SSPRK22. The default `kappa = 0` is the standard
/// fixed-step two-stage SSPRK22 method; nonzero values apply the pinned
/// OrdinaryDiffEqSSPRK coefficient rescaling before each step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prrk22 {
    pub kappa: f64,
}

impl Default for Prrk22 {
    fn default() -> Self {
        Self { kappa: 0.0 }
    }
}

impl Prrk22 {
    pub const fn new(kappa: f64) -> Self {
        Self { kappa }
    }
}

#[allow(non_camel_case_types)]
pub type pRRK22 = Prrk22;

/// Parametric relaxation SSPRK33. The default `kappa = 0` is the standard
/// fixed-step three-stage SSPRK33 method; nonzero values apply the pinned
/// OrdinaryDiffEqSSPRK coefficient rescaling before each step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prrk33 {
    pub kappa: f64,
}

impl Default for Prrk33 {
    fn default() -> Self {
        Self { kappa: 0.0 }
    }
}

impl Prrk33 {
    pub const fn new(kappa: f64) -> Self {
        Self { kappa }
    }
}

#[allow(non_camel_case_types)]
pub type pRRK33 = Prrk33;

struct Prrk22Kernel {
    kappa: f64,
    first_derivative: Vec<f64>,
    second_derivative: Vec<f64>,
    stage_state: Vec<f64>,
}

impl Prrk22Kernel {
    fn new(kappa: f64, dimension: usize) -> Self {
        Self {
            kappa,
            first_derivative: vec![0.0; dimension],
            second_derivative: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
        }
    }

    fn evaluate<F, P>(
        problem: &OdeProblem<F, P>,
        derivative: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        (problem.rhs)(derivative, state, problem.parameters(), time);
        stats.rhs_evaluations += 1;
    }

    fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
        values
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }
}

impl<F, P> StepKernel<F, P> for Prrk22Kernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 2)
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats);
        Self::ensure_finite(&self.first_derivative)
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Ok(maximum_step)
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats);
        Self::ensure_finite(&self.first_derivative)?;

        let z = self.kappa * step;
        let psi1 = 1.0 + z;
        let psi2 = 0.5 + psi1 * (0.5 + 0.5 * z);
        let alpha_hat10 = 1.0;
        let beta_hat10 = 1.0 / psi1;
        let alpha_hat20 = 0.5 / psi2;
        let alpha_hat21 = psi1 * (0.5 + 0.5 * z) / psi2;
        let beta_hat21 = psi1 * 0.5 / psi2;
        let c_hat1 = beta_hat10;
        let c_hat2 = alpha_hat21 * c_hat1 + beta_hat21;
        let step_hat = c_hat2 * step;

        for ((output, value), derivative) in self
            .stage_state
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
        {
            *output = alpha_hat10 * value + beta_hat10 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.second_derivative,
            &self.stage_state,
            time + c_hat1 * step_hat,
            stats,
        );
        Self::ensure_finite(&self.second_derivative)?;
        for (((output, value), stage), derivative) in candidate
            .iter_mut()
            .zip(state)
            .zip(&self.stage_state)
            .zip(&self.second_derivative)
        {
            *output =
                alpha_hat20 * value + alpha_hat21 * stage + beta_hat21 * step_hat * derivative;
        }
        Self::ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn record_dense_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut TrajectoryRecorder<'_>,
        _: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        Ok(false)
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn reject_step(&mut self) {}
}

impl OdeAlgorithm for Prrk22 {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        drive_integration(
            problem,
            options,
            Prrk22Kernel::new(self.kappa, problem.initial_state().len()),
        )
    }
}

struct Prrk33Kernel {
    kappa: f64,
    first_derivative: Vec<f64>,
    second_derivative: Vec<f64>,
    third_derivative: Vec<f64>,
    stage_one: Vec<f64>,
    stage_two: Vec<f64>,
}

impl Prrk33Kernel {
    fn new(kappa: f64, dimension: usize) -> Self {
        Self {
            kappa,
            first_derivative: vec![0.0; dimension],
            second_derivative: vec![0.0; dimension],
            third_derivative: vec![0.0; dimension],
            stage_one: vec![0.0; dimension],
            stage_two: vec![0.0; dimension],
        }
    }

    fn evaluate<F, P>(
        problem: &OdeProblem<F, P>,
        derivative: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        (problem.rhs)(derivative, state, problem.parameters(), time);
        stats.rhs_evaluations += 1;
    }

    fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
        values
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }
}

impl<F, P> StepKernel<F, P> for Prrk33Kernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 3)
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats);
        Self::ensure_finite(&self.first_derivative)
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Ok(maximum_step)
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        // OrdinaryDiffEqSSPRK's pRRK33 coefficients (algorithms.jl and
        // ssprk_perform_step.jl) use the SSPRK(3,3) Shu--Osher form:
        // (α10,β10)=(1,1), (α20,α21,β21)=(3/4,1/4,1/4),
        // (α30,α32,β32)=(1/3,2/3,2/3).
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats);
        Self::ensure_finite(&self.first_derivative)?;

        let z = self.kappa * step;
        let psi1 = 1.0 + z;
        let psi2 = 0.75 + psi1 * (0.25 + 0.25 * z);
        let psi3 = 1.0 / 3.0 + psi2 * (2.0 / 3.0 + (2.0 / 3.0) * z);
        let alpha_hat10 = (1.0 + z) / psi1;
        let beta_hat10 = 1.0 / psi1;
        let alpha_hat20 = 0.75 / psi2;
        let alpha_hat21 = psi1 * (0.25 + 0.25 * z) / psi2;
        let beta_hat21 = psi1 * 0.25 / psi2;
        let alpha_hat30 = (1.0 / 3.0) / psi3;
        let alpha_hat32 = psi2 * (2.0 / 3.0 + (2.0 / 3.0) * z) / psi3;
        let beta_hat32 = psi2 * (2.0 / 3.0) / psi3;
        let c_hat1 = beta_hat10;
        let c_hat2 = alpha_hat21 * c_hat1 + beta_hat21;
        let c_hat3 = alpha_hat32 * c_hat2 + beta_hat32;
        let step_hat = c_hat3 * step;

        for ((output, value), derivative) in self
            .stage_one
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
        {
            *output = alpha_hat10 * value + beta_hat10 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.second_derivative,
            &self.stage_one,
            time + c_hat1 * step_hat,
            stats,
        );
        Self::ensure_finite(&self.second_derivative)?;

        for (((output, value), stage), derivative) in self
            .stage_two
            .iter_mut()
            .zip(state)
            .zip(&self.stage_one)
            .zip(&self.second_derivative)
        {
            *output =
                alpha_hat20 * value + alpha_hat21 * stage + beta_hat21 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.third_derivative,
            &self.stage_two,
            time + c_hat2 * step_hat,
            stats,
        );
        Self::ensure_finite(&self.third_derivative)?;

        for (((output, value), stage), derivative) in candidate
            .iter_mut()
            .zip(state)
            .zip(&self.stage_two)
            .zip(&self.third_derivative)
        {
            *output =
                alpha_hat30 * value + alpha_hat32 * stage + beta_hat32 * step_hat * derivative;
        }
        Self::ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn record_dense_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut TrajectoryRecorder<'_>,
        _: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        Ok(false)
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn reject_step(&mut self) {}
}

impl OdeAlgorithm for Prrk33 {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        drive_integration(
            problem,
            options,
            Prrk33Kernel::new(self.kappa, problem.initial_state().len()),
        )
    }
}

// Low-storage SSPRK53_2N2 (Higueras and Roldan 2018).
const SSPRK53_2N2_A2: &[f64] = &[0.465_388_589_249_323_03];
const SSPRK53_2N2_A3: &[f64] = &[0.465_388_589_249_323_03, 0.465_388_589_249_323_03];
const SSPRK53_2N2_A4: &[f64] = &[
    0.147_834_007_766_855_49,
    0.147_834_007_766_855_49,
    0.124_745_797_313_998,
];
const SSPRK53_2N2_A5: &[f64] = &[
    0.147_834_007_766_855_49,
    0.147_834_007_766_855_49,
    0.124_745_797_313_998,
    0.465_388_589_249_323_03,
];
const SSPRK53_2N2_A: &[&[f64]] = &[
    EMPTY,
    SSPRK53_2N2_A2,
    SSPRK53_2N2_A3,
    SSPRK53_2N2_A4,
    SSPRK53_2N2_A5,
];
const SSPRK53_2N2_B: &[f64] = &[
    0.141_147_331_533_921_92,
    0.141_147_331_533_921_92,
    0.119_103_423_338_901_92,
    0.444_338_609_844_586_78,
    0.154_263_303_748_666_01,
];
const SSPRK53_2N2_C: &[f64] = &[
    0.0,
    0.465_388_589_249_323,
    0.930_777_178_498_646,
    0.420_413_812_847_71,
    0.885_802_402_097_033,
];
fixed_ssprk!(
    SspRk53TwoN2,
    SspRk53TwoN2Tableau,
    3,
    SSPRK53_2N2_C,
    SSPRK53_2N2_A,
    SSPRK53_2N2_B
);

// Low-storage SSPRK53_H (Higueras and Roldan 2018).
const SSPRK53_H_A2: &[f64] = &[0.377_268_915_331_368_03];
const SSPRK53_H_A3: &[f64] = &[0.377_268_915_331_368_03, 0.377_268_915_331_368_03];
const SSPRK53_H_A4: &[f64] = &[
    0.260_811_979_144_497_66,
    0.260_811_979_144_497_66,
    0.260_811_979_144_498,
];
const SSPRK53_H_A5: &[f64] = &[
    0.219_153_436_331_986_97,
    0.117_097_251_841_843_61,
    0.117_097_251_841_843_76,
    0.169_383_144_652_957_01,
];
const SSPRK53_H_A: &[&[f64]] = &[
    EMPTY,
    SSPRK53_H_A2,
    SSPRK53_H_A3,
    SSPRK53_H_A4,
    SSPRK53_H_A5,
];
const SSPRK53_H_B: &[f64] = &[
    0.219_153_436_331_986_97,
    0.117_097_251_841_843_61,
    0.117_097_251_841_843_76,
    0.169_383_144_652_957_01,
    0.377_268_915_331_368_03,
];
const SSPRK53_H_C: &[f64] = &[
    0.0,
    0.377_268_915_331_368,
    0.754_537_830_662_737,
    0.782_435_937_433_493,
    0.622_731_084_668_631,
];
fixed_ssprk!(
    SspRk53H,
    SspRk53HTableau,
    3,
    SSPRK53_H_C,
    SSPRK53_H_A,
    SSPRK53_H_B
);

// SSPRK63 (Ruuth 2006).
const SSPRK63_A2: &[f64] = &[0.284_220_721_334_261_02];
const SSPRK63_A3: &[f64] = &[0.284_220_721_334_261_02, 0.284_220_721_334_261_02];
const SSPRK63_A4: &[f64] = &[
    0.284_220_721_334_261_02,
    0.284_220_721_334_261_02,
    0.284_220_721_334_261_02,
];
const SSPRK63_A5: &[f64] = &[
    0.148_712_861_660_383_11,
    0.120_713_785_765_929_67,
    0.120_713_785_765_929_67,
    0.120_713_785_765_93,
];
const SSPRK63_A6: &[f64] = &[
    0.148_712_861_660_383_11,
    0.120_713_785_765_929_67,
    0.120_713_785_765_929_67,
    0.120_713_785_765_93,
    0.284_220_721_334_261_02,
];
const SSPRK63_A: &[&[f64]] = &[
    EMPTY, SSPRK63_A2, SSPRK63_A3, SSPRK63_A4, SSPRK63_A5, SSPRK63_A6,
];
const SSPRK63_B: &[f64] = &[
    0.169_746_622_349_236_32,
    0.146_093_610_685_229_16,
    0.101_976_386_416_867_99,
    0.101_976_386_416_868_26,
    0.240_103_497_065_899_84,
    0.240_103_497_065_9,
];
const SSPRK63_C: &[f64] = &[
    0.0,
    0.284_220_721_334_261,
    0.568_441_442_668_522,
    0.852_662_164_002_783,
    0.510_854_218_958_172,
    0.795_074_940_292_433,
];
fixed_ssprk!(SspRk63, SspRk63Tableau, 3, SSPRK63_C, SSPRK63_A, SSPRK63_B);

// SSPRK73 (Ruuth 2006).
const SSPRK73_A2: &[f64] = &[0.233_213_863_663_009];
const SSPRK73_A3: &[f64] = &[0.233_213_863_663_009, 0.233_213_863_663_009];
const SSPRK73_A4: &[f64] = &[
    0.233_213_863_663_009,
    0.233_213_863_663_009,
    0.233_213_863_663_009,
];
const SSPRK73_A5: &[f64] = &[
    0.190_078_023_865_844_71,
    0.190_078_023_865_844_71,
    0.190_078_023_865_844_71,
    0.190_078_023_865_845,
];
const SSPRK73_A6: &[f64] = &[
    0.169_307_879_812_473_97,
    0.095_884_917_878_143_7,
    0.095_884_917_878_143_7,
    0.095_884_917_878_143_84,
    0.117_644_805_593_911_99,
];
const SSPRK73_A7: &[f64] = &[
    0.169_307_879_812_473_97,
    0.095_884_917_878_143_7,
    0.095_884_917_878_143_7,
    0.095_884_917_878_143_84,
    0.117_644_805_593_911_99,
    0.233_213_863_663_009,
];
const SSPRK73_A: &[&[f64]] = &[
    EMPTY, SSPRK73_A2, SSPRK73_A3, SSPRK73_A4, SSPRK73_A5, SSPRK73_A6, SSPRK73_A7,
];
const SSPRK73_B: &[f64] = &[
    0.176_989_315_165_324_43,
    0.112_391_719_832_538_73,
    0.112_391_719_832_538_73,
    0.084_359_646_634_108_83,
    0.103_504_017_606_329_38,
    0.205_181_790_464_579,
    0.205_181_790_464_579,
];
const SSPRK73_C: &[f64] = &[
    0.0,
    0.233_213_863_663_009,
    0.466_427_727_326_018,
    0.699_641_590_989_027,
    0.760_312_095_463_379,
    0.574_607_439_040_817,
    0.807_821_302_703_826,
];
fixed_ssprk!(SspRk73, SspRk73Tableau, 3, SSPRK73_C, SSPRK73_A, SSPRK73_B);

// SSPRK83 (Ruuth 2006).
const SSPRK83_A2: &[f64] = &[0.195_804_015_330_143];
const SSPRK83_A3: &[f64] = &[0.195_804_015_330_143, 0.195_804_015_330_143];
const SSPRK83_A4: &[f64] = &[
    0.195_804_015_330_143,
    0.195_804_015_330_143,
    0.195_804_015_330_143,
];
const SSPRK83_A5: &[f64] = &[
    0.195_804_015_330_143,
    0.195_804_015_330_143,
    0.195_804_015_330_143,
    0.195_804_015_330_143,
];
const SSPRK83_A6: &[f64] = &[
    0.113_298_671_247_345_7,
    0.112_133_754_621_672_92,
    0.112_133_754_621_672_92,
    0.112_133_754_621_672_92,
    0.112_133_754_621_673_01,
];
const SSPRK83_A7: &[f64] = &[
    0.113_649_649_861_106_04,
    0.111_656_736_433_452_77,
    0.111_656_736_433_452_77,
    0.111_656_736_433_452_77,
    0.111_656_736_433_452_85,
    0.194_971_062_960_412,
];
const SSPRK83_A8: &[f64] = &[
    0.142_210_235_791_870_03,
    0.140_910_149_518_052_14,
    0.120_472_098_379_644_22,
    0.072_839_787_419_852_71,
    0.072_839_787_419_852_77,
    0.127_190_272_908_641_66,
    0.127_733_653_231_943_99,
];
const SSPRK83_A: &[&[f64]] = &[
    EMPTY, SSPRK83_A2, SSPRK83_A3, SSPRK83_A4, SSPRK83_A5, SSPRK83_A6, SSPRK83_A7, SSPRK83_A8,
];
const SSPRK83_B: &[f64] = &[
    0.142_210_235_791_870_03,
    0.140_910_149_518_052_14,
    0.120_472_098_379_644_22,
    0.072_839_787_419_852_71,
    0.072_839_787_419_852_77,
    0.127_190_272_908_641_66,
    0.127_733_653_231_943_99,
    0.195_804_015_330_143,
];
const SSPRK83_C: &[f64] = &[
    0.0,
    0.195_804_015_330_143,
    0.391_608_030_660_286,
    0.587_412_045_990_429,
    0.783_216_061_320_572,
    0.561_833_689_734_037,
    0.755_247_658_555_329,
    0.804_195_984_669_857,
];
fixed_ssprk!(SspRk83, SspRk83Tableau, 3, SSPRK83_C, SSPRK83_A, SSPRK83_B);

// SSPRK54 (Ruuth 2006).
const SSPRK54_A2: &[f64] = &[0.391_752_226_571_890_02];
const SSPRK54_A3: &[f64] = &[0.217_669_096_261_168_76, 0.368_410_593_050_371];
const SSPRK54_A4: &[f64] = &[
    0.082_692_086_657_810_58,
    0.139_958_502_191_895_35,
    0.251_891_774_271_694_01,
];
const SSPRK54_A5: &[f64] = &[
    0.067_966_283_637_114_75,
    0.115_034_698_504_631_56,
    0.207_034_898_597_385_66,
    0.544_974_750_228_521,
];
const SSPRK54_A: &[&[f64]] = &[EMPTY, SSPRK54_A2, SSPRK54_A3, SSPRK54_A4, SSPRK54_A5];
const SSPRK54_B: &[f64] = &[
    0.146_811_876_084_786_57,
    0.248_482_909_444_976_17,
    0.104_258_830_331_980_98,
    0.274_438_900_901_350_7,
    0.226_007_483_236_906,
];
const SSPRK54_C: &[f64] = &[
    0.0,
    0.391_752_226_571_89,
    0.586_079_689_311_54,
    0.474_542_363_121_4,
    0.935_010_630_967_653,
];
fixed_ssprk!(SspRk54, SspRk54Tableau, 4, SSPRK54_C, SSPRK54_A, SSPRK54_B);

// SSPRK104 (Ketcheson 2008); exact rational coefficients.
const SSPRK104_A2: &[f64] = &[1.0 / 6.0];
const SSPRK104_A3: &[f64] = &[1.0 / 6.0, 1.0 / 6.0];
const SSPRK104_A4: &[f64] = &[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];
const SSPRK104_A5: &[f64] = &[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];
const SSPRK104_A6: &[f64] = &[1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0];
const SSPRK104_A7: &[f64] = &[
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 6.0,
];
const SSPRK104_A8: &[f64] = &[
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 6.0,
    1.0 / 6.0,
];
const SSPRK104_A9: &[f64] = &[
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 6.0,
    1.0 / 6.0,
    1.0 / 6.0,
];
const SSPRK104_A10: &[f64] = &[
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 6.0,
    1.0 / 6.0,
    1.0 / 6.0,
    1.0 / 6.0,
];
const SSPRK104_A: &[&[f64]] = &[
    EMPTY,
    SSPRK104_A2,
    SSPRK104_A3,
    SSPRK104_A4,
    SSPRK104_A5,
    SSPRK104_A6,
    SSPRK104_A7,
    SSPRK104_A8,
    SSPRK104_A9,
    SSPRK104_A10,
];
const SSPRK104_B: &[f64] = &[0.1; 10];
const SSPRK104_C: &[f64] = &[
    0.0,
    1.0 / 6.0,
    1.0 / 3.0,
    1.0 / 2.0,
    2.0 / 3.0,
    1.0 / 3.0,
    1.0 / 2.0,
    2.0 / 3.0,
    5.0 / 6.0,
    1.0,
];
fixed_ssprk!(
    SspRk104,
    SspRk104Tableau,
    4,
    SSPRK104_C,
    SSPRK104_A,
    SSPRK104_B
);

#[cfg(test)]
mod tests {
    use super::{
        SspRk53, SspRk53H, SspRk53TwoN1, SspRk53TwoN2, SspRk54, SspRk63, SspRk73, SspRk83,
        SspRk104, SspRk432,
    };
    use crate::{
        CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveError, SolveOptions, solve,
    };

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn exponential() -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
    }

    fn fixed(step: f64) -> SolveOptions {
        SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    fn endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
        solve(&exponential(), algorithm, &fixed(step))
            .unwrap()
            .last_state()[0]
    }

    fn observed_order<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
        let coarse = (endpoint(algorithm, 0.1) - std::f64::consts::E).abs();
        let fine = (endpoint(algorithm, 0.05) - std::f64::consts::E).abs();
        (coarse / fine).log2()
    }

    #[test]
    fn ssprk432_supports_fixed_and_adaptive_modes_at_third_order() {
        let fixed_order = observed_order(SspRk432);
        assert!(fixed_order > 2.9, "fixed observed order was {fixed_order}");

        let adaptive = SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let endpoint = solve(&exponential(), SspRk432, &adaptive)
            .unwrap()
            .last_state()[0];
        assert!((endpoint - std::f64::consts::E).abs() < 2.0e-8);
    }

    #[test]
    fn ruuth_third_order_methods_converge_at_order_three() {
        for order in [
            observed_order(SspRk53),
            observed_order(SspRk63),
            observed_order(SspRk73),
            observed_order(SspRk83),
        ] {
            assert!(order > 2.9, "observed order was {order}");
        }
    }

    #[test]
    fn low_storage_variants_converge_at_order_three() {
        for order in [
            observed_order(SspRk53TwoN1),
            observed_order(SspRk53TwoN2),
            observed_order(SspRk53H),
        ] {
            assert!(order > 2.9, "observed order was {order}");
        }
    }

    #[test]
    fn fourth_order_methods_converge_at_order_four() {
        for order in [observed_order(SspRk54), observed_order(SspRk104)] {
            assert!(order > 3.85, "observed order was {order}");
        }
    }

    #[test]
    fn fixed_step_methods_reject_adaptive_stepping() {
        assert_eq!(
            solve(&exponential(), SspRk104, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
    }

    #[test]
    fn positive_linear_decay_remains_nonnegative_at_ssp_steps() {
        fn decay(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = -u[0];
        }
        let problem = OdeProblem::new(decay as TestRhs, vec![1.0], (0.0, 6.0), ());
        for value in [
            solve(&problem, SspRk53, &fixed(0.5)).unwrap().last_state()[0],
            solve(&problem, SspRk54, &fixed(0.5)).unwrap().last_state()[0],
            solve(&problem, SspRk104, &fixed(0.5)).unwrap().last_state()[0],
        ] {
            assert!(value >= 0.0);
        }
    }

    #[test]
    fn shared_output_and_callback_features_work_for_extended_methods() {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        let backward = OdeProblem::new(rhs as TestRhs, vec![std::f64::consts::E], (1.0, 0.0), ());
        let backward_options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save_at: vec![1.0, 0.5, 0.0],
            ..SolveOptions::default()
        };
        let solution = solve(&backward, SspRk104, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-10);

        let terminating = exponential()
            .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
        let solution = solve(&terminating, SspRk53, &fixed(0.1)).unwrap();
        assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);

        let adaptive_backward =
            OdeProblem::new(rhs as TestRhs, vec![std::f64::consts::E], (1.0, 0.0), ());
        let adaptive_options = SolveOptions {
            absolute_tolerance: 1.0e-8,
            relative_tolerance: 1.0e-8,
            save_at: vec![0.8, 0.5, 0.2],
            ..SolveOptions::default()
        };
        let adaptive_solution = solve(&adaptive_backward, SspRk432, &adaptive_options).unwrap();
        assert_eq!(adaptive_solution.times(), &[0.8, 0.5, 0.2]);
        assert!((adaptive_solution.last_state()[0] - 0.2f64.exp()).abs() < 2.0e-7);

        let terminating = exponential()
            .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
        let callback_options = SolveOptions {
            save: SaveMode::Endpoints,
            save_at: Vec::new(),
            ..adaptive_options.clone()
        };
        let adaptive_solution = solve(&terminating, SspRk432, &callback_options).unwrap();
        assert!((adaptive_solution.times().last().unwrap() - 0.5).abs() < 1.0e-12);
        assert_eq!(adaptive_solution.stats().callback_invocations, 1);
    }
}
