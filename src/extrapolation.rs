//! Explicit and linearly implicit extrapolation methods.
//!
//! The base discretizations and extrapolation nodes follow the pinned
//! `OrdinaryDiffEqExtrapolation` implementation: explicit Euler with Romberg
//! nodes, modified midpoint with inverse-square nodes, linearly implicit Euler
//! with inverse-step nodes, and the semi-implicit midpoint variants with
//! inverse-square nodes.  Polynomial extrapolation is evaluated with the
//! Neville recurrence; the Deuflhard and Hairer--Wanner types retain distinct
//! order-window policies.

use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{factorize, solve_factorized};
use crate::solution::{BorrowedHermiteSegment, DenseSegment, TrajectoryRecorder};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

/// Subdividing sequences supported by OrdinaryDiffEq's extrapolation methods.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExtrapolationSequence {
    #[default]
    Harmonic,
    Romberg,
    Bulirsch,
}

impl ExtrapolationSequence {
    fn term(self, index: usize) -> usize {
        match self {
            Self::Harmonic => index + 1,
            Self::Romberg => 1usize.checked_shl(index as u32).unwrap_or(usize::MAX),
            Self::Bulirsch => {
                if index == 0 {
                    1
                } else if index % 2 == 1 {
                    1usize << index.div_ceil(2)
                } else {
                    3usize << (index / 2 - 1)
                }
            }
        }
    }
}

macro_rules! extrapolation_algorithm {
    ($name:ident, $min:expr, $init:expr, $max:expr, $kind:expr, $policy:expr, $factor:expr) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub struct $name {
            min_order: usize,
            init_order: usize,
            max_order: usize,
            sequence: ExtrapolationSequence,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    min_order: $min,
                    init_order: $init,
                    max_order: $max,
                    sequence: ExtrapolationSequence::Harmonic,
                }
            }
        }

        impl $name {
            pub fn new(
                min_order: usize,
                init_order: usize,
                max_order: usize,
                sequence: ExtrapolationSequence,
            ) -> Self {
                let strict_window = matches!(
                    $policy,
                    OrderPolicy::HairerWanner | OrderPolicy::Barycentric
                ) || matches!($kind, BaseMethod::LinearlyImplicitEuler);
                let min_order = min_order.max($min).min(if strict_window { 13 } else { 15 });
                let init_floor = min_order + usize::from(strict_window);
                let init_order = init_order.max(init_floor).min(14);
                let max_floor = init_order + usize::from(strict_window);
                let max_order = max_order.max(max_floor).min(15);
                Self {
                    min_order,
                    init_order,
                    max_order,
                    sequence,
                }
            }

            pub fn min_order(self) -> usize {
                self.min_order
            }
            pub fn init_order(self) -> usize {
                self.init_order
            }
            pub fn max_order(self) -> usize {
                self.max_order
            }
            pub fn sequence(self) -> ExtrapolationSequence {
                self.sequence
            }
        }

        impl OdeAlgorithm for $name {
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
                    ExtrapolationKernel::new(
                        problem.initial_state().len(),
                        $kind,
                        $policy,
                        self.min_order,
                        self.init_order,
                        self.max_order,
                        self.sequence,
                        $factor,
                    ),
                )
            }
        }
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BaseMethod {
    ExplicitEuler,
    ExplicitMidpoint,
    LinearlyImplicitEuler,
    LinearlyImplicitMidpoint,
    SmoothedLinearlyImplicitEuler,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OrderPolicy {
    AitkenNeville,
    Deuflhard,
    HairerWanner,
    Barycentric,
}

extrapolation_algorithm!(
    AitkenNeville,
    1,
    5,
    10,
    BaseMethod::ExplicitEuler,
    OrderPolicy::AitkenNeville,
    1
);
extrapolation_algorithm!(
    ExtrapolationMidpointDeuflhard,
    1,
    5,
    10,
    BaseMethod::ExplicitMidpoint,
    OrderPolicy::Deuflhard,
    2
);
extrapolation_algorithm!(
    ExtrapolationMidpointHairerWanner,
    2,
    5,
    10,
    BaseMethod::ExplicitMidpoint,
    OrderPolicy::HairerWanner,
    2
);
extrapolation_algorithm!(
    ImplicitEulerExtrapolation,
    3,
    5,
    12,
    BaseMethod::LinearlyImplicitEuler,
    OrderPolicy::Deuflhard,
    1
);
extrapolation_algorithm!(
    ImplicitDeuflhardExtrapolation,
    1,
    5,
    10,
    BaseMethod::LinearlyImplicitMidpoint,
    OrderPolicy::Deuflhard,
    4
);
extrapolation_algorithm!(
    ImplicitHairerWannerExtrapolation,
    2,
    5,
    10,
    BaseMethod::LinearlyImplicitMidpoint,
    OrderPolicy::HairerWanner,
    4
);
extrapolation_algorithm!(
    ImplicitEulerBarycentricExtrapolation,
    3,
    5,
    12,
    BaseMethod::SmoothedLinearlyImplicitEuler,
    OrderPolicy::Barycentric,
    2
);

struct ExtrapolationKernel {
    dimension: usize,
    base: BaseMethod,
    policy: OrderPolicy,
    minimum_order: usize,
    current_order: usize,
    maximum_order: usize,
    sequence: ExtrapolationSequence,
    sequence_factor: usize,
    raw: Vec<Vec<f64>>,
    diagonal: Vec<Vec<f64>>,
    derivative: Vec<f64>,
    start_derivative: Vec<f64>,
    end_derivative: Vec<f64>,
    temporary: Vec<f64>,
    previous: Vec<f64>,
    next: Vec<f64>,
    jacobian: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    correction: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    last_error: f64,
}

impl ExtrapolationKernel {
    #[allow(clippy::too_many_arguments)]
    fn new(
        dimension: usize,
        base: BaseMethod,
        policy: OrderPolicy,
        minimum_order: usize,
        current_order: usize,
        maximum_order: usize,
        sequence: ExtrapolationSequence,
        sequence_factor: usize,
    ) -> Self {
        Self {
            dimension,
            base,
            policy,
            minimum_order,
            current_order,
            maximum_order,
            sequence,
            sequence_factor,
            raw: Vec::new(),
            diagonal: Vec::new(),
            derivative: vec![0.0; dimension],
            start_derivative: vec![0.0; dimension],
            end_derivative: vec![0.0; dimension],
            temporary: vec![0.0; dimension],
            previous: vec![0.0; dimension],
            next: vec![0.0; dimension],
            jacobian: vec![0.0; dimension * dimension],
            matrix: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            correction: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            last_error: 0.0,
        }
    }

    fn evaluate<F, P>(
        problem: &OdeProblem<F, P>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        (problem.rhs)(output, state, problem.parameters(), time);
        stats.rhs_evaluations += 1;
        output
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }

    fn compute_jacobian<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        if problem.evaluate_jacobian(&mut self.jacobian, state, time) {
            if self.jacobian.iter().any(|value| !value.is_finite()) {
                return Err(SolveError::NonFiniteDerivative);
            }
        } else {
            let mut base = vec![0.0; self.dimension];
            Self::evaluate(problem, &mut base, state, time, stats)?;
            for column in 0..self.dimension {
                self.perturbed_state.copy_from_slice(state);
                let perturbation = f64::EPSILON.sqrt() * state[column].abs().max(1.0);
                self.perturbed_state[column] += perturbation;
                Self::evaluate(
                    problem,
                    &mut self.perturbed_derivative,
                    &self.perturbed_state,
                    time,
                    stats,
                )?;
                for (row, &base_value) in base.iter().enumerate() {
                    self.jacobian[row * self.dimension + column] =
                        (self.perturbed_derivative[row] - base_value) / perturbation;
                }
            }
        }
        stats.jacobian_evaluations += 1;
        Ok(())
    }

    fn factor_linearly_implicit(
        &mut self,
        substep: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                self.matrix[row * self.dimension + column] = f64::from(row == column)
                    - substep * self.jacobian[row * self.dimension + column];
            }
        }
        factorize(&mut self.matrix, &mut self.pivots, self.dimension)?;
        stats.linear_factorizations += 1;
        Ok(())
    }

    fn linear_increment(&mut self, substep: f64, stats: &mut SolverStats) {
        for component in 0..self.dimension {
            self.correction[component] = substep * self.derivative[component];
        }
        solve_factorized(
            &self.matrix,
            &self.pivots,
            &mut self.correction,
            self.dimension,
        );
        stats.linear_solves += 1;
    }

    fn explicit_euler<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        subdivisions: usize,
        stats: &mut SolverStats,
    ) -> Result<Vec<f64>, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        self.temporary.copy_from_slice(state);
        let h = step / subdivisions as f64;
        for index in 0..subdivisions {
            Self::evaluate(
                problem,
                &mut self.derivative,
                &self.temporary,
                time + index as f64 * h,
                stats,
            )?;
            for component in 0..self.dimension {
                self.temporary[component] += h * self.derivative[component];
            }
        }
        Ok(self.temporary.clone())
    }

    fn explicit_midpoint<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        subdivisions: usize,
        stats: &mut SolverStats,
    ) -> Result<Vec<f64>, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let subdivisions = subdivisions.max(2);
        let h = step / subdivisions as f64;
        self.previous.copy_from_slice(state);
        Self::evaluate(problem, &mut self.derivative, state, time, stats)?;
        for (component, temporary) in self.temporary.iter_mut().enumerate() {
            *temporary = state[component] + h * self.derivative[component];
        }
        for index in 1..subdivisions {
            Self::evaluate(
                problem,
                &mut self.derivative,
                &self.temporary,
                time + index as f64 * h,
                stats,
            )?;
            for component in 0..self.dimension {
                self.next[component] =
                    self.previous[component] + 2.0 * h * self.derivative[component];
            }
            self.previous.copy_from_slice(&self.temporary);
            self.temporary.copy_from_slice(&self.next);
        }
        Ok(self.temporary.clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn implicit_euler<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        subdivisions: usize,
        smooth: bool,
        stats: &mut SolverStats,
    ) -> Result<Vec<f64>, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let h = step / subdivisions as f64;
        self.factor_linearly_implicit(h, stats)?;
        self.temporary.copy_from_slice(state);
        self.previous.copy_from_slice(state);
        let count = subdivisions + usize::from(smooth);
        for index in 0..count {
            Self::evaluate(
                problem,
                &mut self.derivative,
                &self.temporary,
                time + index as f64 * h,
                stats,
            )?;
            self.linear_increment(h, stats);
            self.next.copy_from_slice(&self.temporary);
            for component in 0..self.dimension {
                self.next[component] += self.correction[component];
            }
            if smooth && index == subdivisions {
                for component in 0..self.dimension {
                    self.next[component] = 0.25
                        * (self.next[component]
                            + 2.0 * self.temporary[component]
                            + self.previous[component]);
                }
            }
            self.previous.copy_from_slice(&self.temporary);
            self.temporary.copy_from_slice(&self.next);
        }
        Ok(self.temporary.clone())
    }

    fn implicit_midpoint<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        subdivisions: usize,
        stats: &mut SolverStats,
    ) -> Result<Vec<f64>, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let subdivisions = subdivisions.max(4);
        let h = step / subdivisions as f64;
        self.factor_linearly_implicit(h, stats)?;
        self.previous.copy_from_slice(state);
        Self::evaluate(problem, &mut self.derivative, state, time, stats)?;
        self.linear_increment(h, stats);
        for (component, temporary) in self.temporary.iter_mut().enumerate() {
            *temporary = state[component] + self.correction[component];
        }
        for index in 1..subdivisions {
            Self::evaluate(
                problem,
                &mut self.derivative,
                &self.temporary,
                time + index as f64 * h,
                stats,
            )?;
            for component in 0..self.dimension {
                self.derivative[component] -=
                    (self.temporary[component] - self.previous[component]) / h;
            }
            self.linear_increment(h, stats);
            for component in 0..self.dimension {
                self.next[component] = 2.0 * self.temporary[component] - self.previous[component]
                    + 2.0 * self.correction[component];
            }
            self.previous.copy_from_slice(&self.temporary);
            self.temporary.copy_from_slice(&self.next);
        }
        Ok(self.temporary.clone())
    }

    fn raw_approximation<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        level: usize,
        stats: &mut SolverStats,
    ) -> Result<Vec<f64>, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let sequence = self.sequence.term(level);
        let subdivisions = self.sequence_factor.saturating_mul(sequence);
        match self.base {
            BaseMethod::ExplicitEuler => {
                // Aitken--Neville is defined with a Romberg sequence regardless
                // of the configurable sequences used by the other families.
                self.explicit_euler(problem, state, time, step, 1usize << level, stats)
            }
            BaseMethod::ExplicitMidpoint => {
                self.explicit_midpoint(problem, state, time, step, subdivisions, stats)
            }
            BaseMethod::LinearlyImplicitEuler => {
                self.implicit_euler(problem, state, time, step, subdivisions, false, stats)
            }
            BaseMethod::LinearlyImplicitMidpoint => {
                self.implicit_midpoint(problem, state, time, step, subdivisions, stats)
            }
            BaseMethod::SmoothedLinearlyImplicitEuler => {
                self.implicit_euler(problem, state, time, step, subdivisions, true, stats)
            }
        }
    }

    fn node(&self, level: usize) -> f64 {
        let subdivisions = match self.base {
            BaseMethod::ExplicitEuler => 1usize << level,
            _ => self
                .sequence_factor
                .saturating_mul(self.sequence.term(level)),
        } as f64;
        match self.base {
            BaseMethod::ExplicitEuler
            | BaseMethod::LinearlyImplicitEuler
            | BaseMethod::SmoothedLinearlyImplicitEuler => 1.0 / subdivisions,
            BaseMethod::ExplicitMidpoint | BaseMethod::LinearlyImplicitMidpoint => {
                1.0 / (subdivisions * subdivisions)
            }
        }
    }

    fn extrapolate(&mut self, levels: usize) {
        self.diagonal.clear();
        let mut tableau: Vec<Vec<Vec<f64>>> = Vec::with_capacity(levels);
        for level in 0..levels {
            let mut row = Vec::with_capacity(level + 1);
            row.push(self.raw[level].clone());
            for column in 1..=level {
                let ratio = self.node(level - column) / self.node(level) - 1.0;
                let mut value = vec![0.0; self.dimension];
                for component in 0..self.dimension {
                    value[component] = row[column - 1][component]
                        + (row[column - 1][component] - tableau[level - 1][column - 1][component])
                            / ratio;
                }
                row.push(value);
            }
            self.diagonal.push(row[level].clone());
            tableau.push(row);
        }
    }

    fn hermite_interpolate(
        &self,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        time: f64,
        query: f64,
        output: &mut [f64],
    ) -> Result<(), SolveError> {
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            time,
            previous_state,
            state,
            &self.start_derivative,
            &self.end_derivative,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        segment
            .interpolate(query, output)
            .map_err(|_| SolveError::NonFiniteDerivative)
    }
}

impl<F, P> StepKernel<F, P> for ExtrapolationKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        let controller_order = match self.base {
            BaseMethod::ExplicitEuler | BaseMethod::LinearlyImplicitEuler => self.current_order + 1,
            _ => 2 * (self.current_order + 1),
        };
        let (safety, maximum) = match self.policy {
            OrderPolicy::HairerWanner => (0.9, 4.0),
            OrderPolicy::Deuflhard => (0.8, 5.0),
            OrderPolicy::AitkenNeville | OrderPolicy::Barycentric => (0.9, 5.0),
        };
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(controller_order, safety, 0.2, maximum, 0.25),
        )
        .recover_nonlinear_and_singular_failures()
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Self::evaluate(problem, &mut self.start_derivative, state, time, stats)
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        options: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        let mut state_norm = 0.0_f64;
        let mut derivative_norm = 0.0_f64;
        for (&value, &derivative) in state.iter().zip(&self.start_derivative) {
            let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
            state_norm = state_norm.max(value.abs() / scale);
            derivative_norm = derivative_norm.max(derivative.abs() / scale);
        }
        let estimate = if derivative_norm <= 1.0e-14 {
            1.0e-3
        } else {
            0.01 * state_norm.max(1.0) / derivative_norm
        };
        Ok(estimate.clamp(1.0e-8, maximum_step))
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        if matches!(
            self.base,
            BaseMethod::LinearlyImplicitEuler
                | BaseMethod::LinearlyImplicitMidpoint
                | BaseMethod::SmoothedLinearlyImplicitEuler
        ) {
            self.compute_jacobian(problem, state, time, stats)?;
        }

        let levels = if options.adaptive {
            self.current_order + 1
        } else {
            self.current_order.max(1) + 1
        };
        self.raw.clear();
        for level in 0..levels {
            let approximation = self.raw_approximation(problem, state, time, step, level, stats)?;
            self.raw.push(approximation);
        }
        self.extrapolate(levels);
        candidate.copy_from_slice(self.diagonal.last().expect("at least two levels"));
        Self::evaluate(
            problem,
            &mut self.end_derivative,
            candidate,
            time + step,
            stats,
        )?;
        if !options.adaptive {
            self.last_error = 0.0;
            return Ok(StepEstimate::new(0.0));
        }
        let previous = &self.diagonal[levels - 2];
        let mut error_norm = 0.0_f64;
        for component in 0..self.dimension {
            let scale = options.absolute_tolerance
                + options.relative_tolerance
                    * state[component].abs().max(candidate[component].abs());
            error_norm =
                error_norm.max(((candidate[component] - previous[component]) / scale).abs());
        }
        self.last_error = error_norm;
        Ok(StepEstimate::new(error_norm))
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
        _: &mut SolverStats,
    ) -> Result<crate::callback::CallbackOutcome, SolveError> {
        let endpoint_time = *time;
        let endpoint_state = state.to_vec();
        let mut interpolator = |query: f64, output: &mut [f64]| {
            self.hermite_interpolate(
                previous_state,
                &endpoint_state,
                previous_time,
                endpoint_time,
                query,
                output,
            )
        };
        problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            Some(&mut interpolator),
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
        if time != attempted_time {
            Self::evaluate(problem, &mut self.end_derivative, state, time, stats)?;
        }
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            time,
            previous_state,
            state,
            &self.start_derivative,
            &self.end_derivative,
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
        Ok(true)
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if callback_applied {
            Self::evaluate(problem, &mut self.start_derivative, state, time, stats)?;
        } else {
            self.start_derivative.copy_from_slice(&self.end_derivative);
        }
        let (raise_threshold, lower_threshold) = match self.policy {
            OrderPolicy::HairerWanner => (0.08, 0.9),
            OrderPolicy::Deuflhard => (0.04, 0.75),
            OrderPolicy::AitkenNeville | OrderPolicy::Barycentric => (0.05, 0.8),
        };
        if !callback_applied
            && self.last_error < raise_threshold
            && self.current_order < self.maximum_order
        {
            self.current_order += 1;
        } else if self.last_error > lower_threshold && self.current_order > self.minimum_order {
            self.current_order -= 1;
        }
        Ok(())
    }

    fn reject_step(&mut self) {
        if self.current_order < self.maximum_order {
            self.current_order += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AitkenNeville, ExtrapolationMidpointDeuflhard, ExtrapolationMidpointHairerWanner,
        ImplicitDeuflhardExtrapolation, ImplicitEulerBarycentricExtrapolation,
        ImplicitEulerExtrapolation, ImplicitHairerWannerExtrapolation,
    };
    use crate::{OdeProblem, SaveMode, SolveOptions, solve};

    #[test]
    fn explicit_extrapolation_families_are_accurate() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.2),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        for value in [
            solve(&problem, AitkenNeville::default(), &options)
                .unwrap()
                .last_state()[0],
            solve(
                &problem,
                ExtrapolationMidpointDeuflhard::default(),
                &options,
            )
            .unwrap()
            .last_state()[0],
            solve(
                &problem,
                ExtrapolationMidpointHairerWanner::default(),
                &options,
            )
            .unwrap()
            .last_state()[0],
        ] {
            assert!((value - std::f64::consts::E).abs() < 2.0e-7, "{value}");
        }
    }

    #[test]
    fn implicit_extrapolation_families_handle_stiff_decay() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -40.0 * u[0],
            vec![1.0],
            (0.0, 0.2),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        for value in [
            solve(&problem, ImplicitEulerExtrapolation::default(), &options)
                .unwrap()
                .last_state()[0],
            solve(
                &problem,
                ImplicitDeuflhardExtrapolation::default(),
                &options,
            )
            .unwrap()
            .last_state()[0],
            solve(
                &problem,
                ImplicitHairerWannerExtrapolation::default(),
                &options,
            )
            .unwrap()
            .last_state()[0],
            solve(
                &problem,
                ImplicitEulerBarycentricExtrapolation::default(),
                &options,
            )
            .unwrap()
            .last_state()[0],
        ] {
            assert!((value - (-8.0_f64).exp()).abs() < 2.0e-5, "{value}");
        }
    }
}
