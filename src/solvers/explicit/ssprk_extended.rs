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
use crate::callback::CallbackOutcome;
use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::solution::{BorrowedHermiteSegment, DenseSegment, HermiteSegment, TrajectoryRecorder};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

#[allow(clippy::too_many_arguments)]
fn apply_hermite_callbacks<F, P>(
    problem: &OdeProblem<F, P>,
    previous_state: &[f64],
    previous_time: f64,
    state: &mut [f64],
    time: &mut f64,
    state_before_effect: &mut [f64],
    event_tolerance: f64,
    start_derivative: &[f64],
    endpoint_state: &mut [f64],
    endpoint_derivative: &mut [f64],
    endpoint_prepared: &mut bool,
    stats: &mut SolverStats,
) -> Result<CallbackOutcome, SolveError>
where
    F: crate::OdeFunction<P>,
{
    if !problem.has_continuous_callbacks() {
        *endpoint_prepared = false;
        return problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            None,
        );
    }
    endpoint_state.copy_from_slice(state);
    problem.rhs.evaluate(
        endpoint_derivative,
        endpoint_state,
        problem.parameters(),
        *time,
    )?;
    stats.rhs_evaluations += 1;
    if !endpoint_derivative.iter().all(|value| value.is_finite()) {
        return Err(SolveError::NonFiniteDerivative);
    }
    *endpoint_prepared = true;
    let segment = BorrowedHermiteSegment::new(
        previous_time,
        *time,
        previous_state,
        endpoint_state,
        start_derivative,
        endpoint_derivative,
    )
    .map_err(|_| SolveError::NonFiniteDerivative)?;
    let mut interpolate = |sample_time: f64, output: &mut [f64]| {
        segment
            .interpolate(sample_time, output)
            .map_err(|_| SolveError::NonFiniteDerivative)
    };
    problem.apply_step_callbacks(
        previous_state,
        previous_time,
        state,
        time,
        state_before_effect,
        event_tolerance,
        Some(&mut interpolate),
    )
}

#[allow(clippy::too_many_arguments)]
fn record_hermite_step<F, P>(
    problem: &OdeProblem<F, P>,
    previous_state: &[f64],
    state: &[f64],
    start_derivative: &[f64],
    endpoint_state: &mut [f64],
    endpoint_derivative: &mut [f64],
    endpoint_prepared: &mut bool,
    previous_time: f64,
    attempted_time: f64,
    time: f64,
    final_time: bool,
    recorder: &mut TrajectoryRecorder<'_>,
    stats: &mut SolverStats,
) -> Result<bool, SolveError>
where
    F: crate::OdeFunction<P>,
{
    if !recorder.needs_dense_sampling() && !recorder.retains_dense_output() {
        *endpoint_prepared = false;
        return Ok(false);
    }
    if !*endpoint_prepared {
        endpoint_state.copy_from_slice(state);
        problem.rhs.evaluate(
            endpoint_derivative,
            endpoint_state,
            problem.parameters(),
            attempted_time,
        )?;
        stats.rhs_evaluations += 1;
        if !endpoint_derivative.iter().all(|value| value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
    }
    let segment = BorrowedHermiteSegment::new(
        previous_time,
        attempted_time,
        previous_state,
        endpoint_state,
        start_derivative,
        endpoint_derivative,
    )
    .map_err(|_| SolveError::NonFiniteDerivative)?;
    recorder
        .record_step_dense(
            previous_state,
            previous_time,
            state,
            time,
            final_time,
            &segment,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
    if recorder.retains_dense_output() {
        let segment = HermiteSegment::new_bounded(
            previous_time,
            attempted_time,
            time,
            previous_state.to_vec(),
            endpoint_state.to_vec(),
            start_derivative.to_vec(),
            endpoint_derivative.to_vec(),
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        recorder.retain_hermite_segment(segment);
    }
    *endpoint_prepared = false;
    Ok(true)
}

macro_rules! fixed_ssprk {
    ($algorithm:ident, $path:literal) => {
        crate::define_explicit_rk_from_file!(pub $algorithm, $path, crate = crate);
    };
}

crate::define_explicit_rk_from_file!(
    pub SspRk432,
    "src/tableau/resources/explicit/ssp_rk432.json",
    crate = crate
);
crate::define_explicit_rk_from_file!(
    pub SspRk932,
    "src/tableau/resources/explicit/ssp_rk932.json",
    crate = crate
);

fixed_ssprk!(SspRk53, "src/tableau/resources/explicit/ssp_rk53.json");
fixed_ssprk!(
    SspRk53TwoN1,
    "src/tableau/resources/explicit/ssp_rk53_two_n1.json"
);

/// Parametric relaxation SSPRK22. The default `kappa = 0` is the standard
/// fixed-step two-stage SSPRK22 method; nonzero values apply the pinned
/// OrdinaryDiffEqSSPRK coefficient rescaling before each step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prrk22 {
    /// Relaxation parameter applied to the SSPRK22 coefficients.
    pub kappa: f64,
}

impl Default for Prrk22 {
    fn default() -> Self {
        Self { kappa: 0.0 }
    }
}

impl Prrk22 {
    /// Creates the method with relaxation parameter `kappa`.
    pub const fn new(kappa: f64) -> Self {
        Self { kappa }
    }
}

#[allow(non_camel_case_types)]
/// SciML-compatible spelling of [`Prrk22`].
pub type pRRK22 = Prrk22;

/// Parametric relaxation SSPRK33. The default `kappa = 0` is the standard
/// fixed-step three-stage SSPRK33 method; nonzero values apply the pinned
/// OrdinaryDiffEqSSPRK coefficient rescaling before each step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prrk33 {
    /// Relaxation parameter applied to the SSPRK33 coefficients.
    pub kappa: f64,
}

impl Default for Prrk33 {
    fn default() -> Self {
        Self { kappa: 0.0 }
    }
}

impl Prrk33 {
    /// Creates the method with relaxation parameter `kappa`.
    pub const fn new(kappa: f64) -> Self {
        Self { kappa }
    }
}

#[allow(non_camel_case_types)]
/// SciML-compatible spelling of [`Prrk33`].
pub type pRRK33 = Prrk33;

struct Prrk22Kernel {
    kappa: f64,
    first_derivative: Vec<f64>,
    second_derivative: Vec<f64>,
    stage_state: Vec<f64>,
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_derivative: Vec<f64>,
    dense_endpoint_prepared: bool,
}

impl Prrk22Kernel {
    fn new(kappa: f64, dimension: usize) -> Self {
        Self {
            kappa,
            first_derivative: vec![0.0; dimension],
            second_derivative: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_derivative: vec![0.0; dimension],
            dense_endpoint_prepared: false,
        }
    }

    fn evaluate<F, P>(
        problem: &OdeProblem<F, P>,
        derivative: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        problem
            .rhs
            .evaluate(derivative, state, problem.parameters(), time)?;
        stats.rhs_evaluations += 1;
        Ok(())
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
    F: crate::OdeFunction<P>,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

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
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats)?;
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
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats)?;
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
        )?;
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

    fn apply_step_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        stats: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        apply_hermite_callbacks(
            problem,
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            &self.first_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_derivative,
            &mut self.dense_endpoint_prepared,
            stats,
        )
    }

    fn record_dense_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        stats: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        record_hermite_step(
            problem,
            previous_state,
            state,
            &self.first_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_derivative,
            &mut self.dense_endpoint_prepared,
            previous_time,
            attempted_time,
            time,
            final_time,
            recorder,
            stats,
        )
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
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
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
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_derivative: Vec<f64>,
    dense_endpoint_prepared: bool,
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
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_derivative: vec![0.0; dimension],
            dense_endpoint_prepared: false,
        }
    }

    fn evaluate<F, P>(
        problem: &OdeProblem<F, P>,
        derivative: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        problem
            .rhs
            .evaluate(derivative, state, problem.parameters(), time)?;
        stats.rhs_evaluations += 1;
        Ok(())
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
    F: crate::OdeFunction<P>,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

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
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats)?;
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
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats)?;
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
        )?;
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
        )?;
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

    fn apply_step_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        stats: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        apply_hermite_callbacks(
            problem,
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            &self.first_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_derivative,
            &mut self.dense_endpoint_prepared,
            stats,
        )
    }

    fn record_dense_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        stats: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        record_hermite_step(
            problem,
            previous_state,
            state,
            &self.first_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_derivative,
            &mut self.dense_endpoint_prepared,
            previous_time,
            attempted_time,
            time,
            final_time,
            recorder,
            stats,
        )
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
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        drive_integration(
            problem,
            options,
            Prrk33Kernel::new(self.kappa, problem.initial_state().len()),
        )
    }
}

/// Parametric-relaxation SSPRK(5,4) of Spiteri and Ruuth.
///
/// This is the fixed-step `pRRK54` method from OrdinaryDiffEqSSPRK.  The
/// relaxation parameter is applied to the Shu--Osher coefficients at every
/// attempted step; `kappa = 0` is the ordinary SSPRK(5,4) method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prrk54 {
    /// Relaxation parameter applied to the SSPRK(5,4) coefficients.
    pub kappa: f64,
}

impl Default for Prrk54 {
    fn default() -> Self {
        Self { kappa: 0.0 }
    }
}

impl Prrk54 {
    /// Creates the method with relaxation parameter `kappa`.
    pub const fn new(kappa: f64) -> Self {
        Self { kappa }
    }
}

#[allow(non_camel_case_types)]
/// SciML-compatible spelling of [`Prrk54`].
pub type pRRK54 = Prrk54;

struct Prrk54Kernel {
    kappa: f64,
    start_derivative: Vec<f64>,
    first_derivative: Vec<f64>,
    second_derivative: Vec<f64>,
    third_derivative: Vec<f64>,
    fourth_derivative: Vec<f64>,
    stage_one: Vec<f64>,
    stage_two: Vec<f64>,
    stage_three: Vec<f64>,
    stage_four: Vec<f64>,
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_derivative: Vec<f64>,
    dense_endpoint_prepared: bool,
}

impl Prrk54Kernel {
    fn new(kappa: f64, dimension: usize) -> Self {
        Self {
            kappa,
            start_derivative: vec![0.0; dimension],
            first_derivative: vec![0.0; dimension],
            second_derivative: vec![0.0; dimension],
            third_derivative: vec![0.0; dimension],
            fourth_derivative: vec![0.0; dimension],
            stage_one: vec![0.0; dimension],
            stage_two: vec![0.0; dimension],
            stage_three: vec![0.0; dimension],
            stage_four: vec![0.0; dimension],
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_derivative: vec![0.0; dimension],
            dense_endpoint_prepared: false,
        }
    }

    fn evaluate<F, P>(
        problem: &OdeProblem<F, P>,
        derivative: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        problem
            .rhs
            .evaluate(derivative, state, problem.parameters(), time)?;
        stats.rhs_evaluations += 1;
        Ok(())
    }

    fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
        values
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }
}

impl<F, P> StepKernel<F, P> for Prrk54Kernel
where
    F: crate::OdeFunction<P>,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 4)
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Self::evaluate(problem, &mut self.start_derivative, state, time, stats)?;
        Self::ensure_finite(&self.start_derivative)
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

    #[allow(clippy::too_many_lines)]
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
        // Base Shu--Osher coefficients from the pinned
        // OrdinaryDiffEqSSPRK pRRK54ConstantCache.  Keep these decimal
        // literals rather than reconstructing them from a tableau: the
        // relaxation transform is defined in this representation upstream.
        let beta10 = 0.391_752_226_571_89;
        let alpha20 = 0.444_370_493_651_235;
        let alpha21 = 0.555_629_506_348_765;
        let beta21 = 0.368_410_593_050_371;
        let alpha30 = 0.620_101_851_488_403;
        let alpha32 = 0.379_898_148_511_597;
        let beta32 = 0.251_891_774_271_694;
        let alpha40 = 0.178_079_954_393_132;
        let alpha43 = 0.821_920_045_606_868;
        let beta43 = 0.544_974_750_228_521;
        let alpha52 = 0.517_231_671_970_585;
        let alpha53 = 0.096_059_710_526_147;
        let beta53 = 0.063_692_468_666_29;
        let alpha54 = 0.386_708_617_503_269;
        let beta54 = 0.226_007_483_236_906;

        Self::evaluate(problem, &mut self.start_derivative, state, time, stats)?;
        Self::ensure_finite(&self.start_derivative)?;

        let z = self.kappa * step;
        let psi1 = 1.0 + z * beta10;
        let psi2 = alpha20 + psi1 * (alpha21 + z * beta21);
        let psi3 = alpha30 + psi2 * (alpha32 + z * beta32);
        let psi4 = alpha40 + psi3 * (alpha43 + z * beta43);
        let psi5 = psi2 * alpha52 + psi3 * (alpha53 + z * beta53) + psi4 * (alpha54 + z * beta54);

        let alpha_hat10 = (1.0 + z * beta10) / psi1;
        let beta_hat10 = beta10 / psi1;
        let alpha_hat20 = alpha20 / psi2;
        let alpha_hat21 = psi1 * (alpha21 + z * beta21) / psi2;
        let beta_hat21 = psi1 * beta21 / psi2;
        let alpha_hat30 = alpha30 / psi3;
        let alpha_hat32 = psi2 * (alpha32 + z * beta32) / psi3;
        let beta_hat32 = psi2 * beta32 / psi3;
        let alpha_hat40 = alpha40 / psi4;
        let alpha_hat43 = psi3 * (alpha43 + z * beta43) / psi4;
        let beta_hat43 = psi3 * beta43 / psi4;
        let alpha_hat52 = psi2 * alpha52 / psi5;
        let alpha_hat53 = psi3 * (alpha53 + z * beta53) / psi5;
        let beta_hat53 = psi3 * beta53 / psi5;
        let alpha_hat54 = psi4 * (alpha54 + z * beta54) / psi5;
        let beta_hat54 = psi4 * beta54 / psi5;

        let c_hat1 = beta_hat10;
        let c_hat2 = alpha_hat21 * c_hat1 + beta_hat21;
        let c_hat3 = alpha_hat32 * c_hat2 + beta_hat32;
        let c_hat4 = alpha_hat43 * c_hat3 + beta_hat43;
        let c_hat5 = alpha_hat52 * c_hat2
            + alpha_hat53 * c_hat3
            + beta_hat53
            + alpha_hat54 * c_hat4
            + beta_hat54;
        let step_hat = c_hat5 * step;

        for ((output, value), derivative) in self
            .stage_one
            .iter_mut()
            .zip(state)
            .zip(&self.start_derivative)
        {
            *output = alpha_hat10 * value + beta_hat10 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.second_derivative,
            &self.stage_one,
            time + c_hat1 * step_hat,
            stats,
        )?;
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
        )?;
        Self::ensure_finite(&self.third_derivative)?;

        for (((output, value), stage), derivative) in self
            .stage_three
            .iter_mut()
            .zip(state)
            .zip(&self.stage_two)
            .zip(&self.third_derivative)
        {
            *output =
                alpha_hat30 * value + alpha_hat32 * stage + beta_hat32 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.fourth_derivative,
            &self.stage_three,
            time + c_hat3 * step_hat,
            stats,
        )?;
        Self::ensure_finite(&self.fourth_derivative)?;

        for (((output, value), stage), derivative) in self
            .stage_four
            .iter_mut()
            .zip(state)
            .zip(&self.stage_three)
            .zip(&self.fourth_derivative)
        {
            *output =
                alpha_hat40 * value + alpha_hat43 * stage + beta_hat43 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.first_derivative,
            &self.stage_four,
            time + c_hat4 * step_hat,
            stats,
        )?;
        Self::ensure_finite(&self.first_derivative)?;

        for (
            ((((output, stage_two), stage_three), derivative_three), stage_four),
            derivative_four,
        ) in candidate
            .iter_mut()
            .zip(&self.stage_two)
            .zip(&self.stage_three)
            .zip(&self.fourth_derivative)
            .zip(&self.stage_four)
            .zip(&self.first_derivative)
        {
            *output = alpha_hat52 * stage_two
                + alpha_hat53 * stage_three
                + beta_hat53 * step_hat * derivative_three
                + alpha_hat54 * stage_four
                + beta_hat54 * step_hat * derivative_four;
        }
        Self::ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn apply_step_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        stats: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        apply_hermite_callbacks(
            problem,
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            &self.start_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_derivative,
            &mut self.dense_endpoint_prepared,
            stats,
        )
    }

    fn record_dense_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        stats: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        record_hermite_step(
            problem,
            previous_state,
            state,
            &self.start_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_derivative,
            &mut self.dense_endpoint_prepared,
            previous_time,
            attempted_time,
            time,
            final_time,
            recorder,
            stats,
        )
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

impl OdeAlgorithm for Prrk54 {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        drive_integration(
            problem,
            options,
            Prrk54Kernel::new(self.kappa, problem.initial_state().len()),
        )
    }
}

fixed_ssprk!(
    SspRk53TwoN2,
    "src/tableau/resources/explicit/ssp_rk53_two_n2.json"
);
fixed_ssprk!(SspRk53H, "src/tableau/resources/explicit/ssp_rk53_h.json");
fixed_ssprk!(SspRk63, "src/tableau/resources/explicit/ssp_rk63.json");
fixed_ssprk!(SspRk73, "src/tableau/resources/explicit/ssp_rk73.json");
fixed_ssprk!(SspRk83, "src/tableau/resources/explicit/ssp_rk83.json");
fixed_ssprk!(SspRk54, "src/tableau/resources/explicit/ssp_rk54.json");
fixed_ssprk!(SspRk104, "src/tableau/resources/explicit/ssp_rk104.json");

#[cfg(test)]
mod tests {
    use super::{
        SspRk53, SspRk53H, SspRk53TwoN1, SspRk53TwoN2, SspRk54, SspRk63, SspRk73, SspRk83,
        SspRk104, SspRk432, SspRk932,
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
    fn ssprk932_supports_fixed_adaptive_and_shared_output_features() {
        let fixed_order = observed_order(SspRk932);
        assert!(fixed_order > 2.9, "fixed observed order was {fixed_order}");

        let adaptive = SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let endpoint = solve(&exponential(), SspRk932, &adaptive)
            .unwrap()
            .last_state()[0];
        assert!((endpoint - std::f64::consts::E).abs() < 2.0e-8);

        let backward = OdeProblem::new(
            (|du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0]) as TestRhs,
            vec![std::f64::consts::E],
            (1.0, 0.0),
            (),
        );
        let save_at_options = SolveOptions {
            save_at: vec![0.8, 0.5, 0.2],
            ..adaptive.clone()
        };
        let saved = solve(&backward, SspRk932, &save_at_options).unwrap();
        assert_eq!(saved.times(), &[0.8, 0.5, 0.2]);
        assert!((saved.last_state()[0] - 0.2f64.exp()).abs() < 2.0e-7);

        let terminating = exponential()
            .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
        let terminated = solve(&terminating, SspRk932, &adaptive).unwrap();
        assert!((terminated.times().last().unwrap() - 0.5).abs() < 1.0e-12);
        assert_eq!(terminated.stats().callback_invocations, 1);
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
