//! Explicit Runge--Kutta methods whose pinned Julia implementation evaluates
//! independent stage groups through SIMD lanes.
//!
//! Rust keeps the same tableaus and stage independence while evaluating the
//! lanes over ordinary contiguous `f64` state slices. This preserves the
//! numerical methods without imposing a packed-vector state type on users.

use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

struct SimdTableau {
    stages: usize,
    order: usize,
    c: &'static [f64],
    a: &'static [f64],
    b: &'static [f64],
    error: &'static [f64],
}

mod coefficient_data {
    use super::SimdTableau;
    use differential_equations_tableau_macros::define_coefficients_from_file;

    define_coefficients_from_file!(pub(super), "coefficients/explicit/simd.toml", crate = crate);

    pub(super) const MER5V2: SimdTableau = SimdTableau {
        stages: 14,
        order: 5,
        c: MER5V2_C,
        a: MER5V2_A,
        b: MER5V2_B,
        error: MER5V2_ERROR,
    };

    pub(super) const MER6V2: SimdTableau = SimdTableau {
        stages: 15,
        order: 6,
        c: MER6V2_C,
        a: MER6V2_A,
        b: MER6V2_B,
        error: MER6V2_ERROR,
    };

    pub(super) const RK6V4: SimdTableau = SimdTableau {
        stages: 22,
        order: 6,
        c: RK6V4_C,
        a: RK6V4_A,
        b: RK6V4_B,
        error: RK6V4_ERROR,
    };
}

use coefficient_data::{MER5V2, MER6V2, RK6V4};

macro_rules! simd_algorithm {
    ($name:ident, $tableau:ident) => {
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl $name {
            pub const fn order(self) -> usize {
                $tableau.order
            }

            pub const fn stage_count(self) -> usize {
                $tableau.stages
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
                    SimdRkKernel::new(problem.initial_state().len(), &$tableau),
                )
            }
        }
    };
}

simd_algorithm!(MER5v2, MER5V2);
simd_algorithm!(MER6v2, MER6V2);
simd_algorithm!(RK6v4, RK6V4);

struct SimdRkKernel {
    dimension: usize,
    tableau: &'static SimdTableau,
    stages: Vec<f64>,
    stage_state: Vec<f64>,
    endpoint_derivative: Vec<f64>,
    error: Vec<f64>,
    endpoint_valid: bool,
}

impl SimdRkKernel {
    fn new(dimension: usize, tableau: &'static SimdTableau) -> Self {
        debug_assert_eq!(tableau.c.len(), tableau.stages);
        debug_assert_eq!(tableau.a.len(), tableau.stages * tableau.stages);
        debug_assert_eq!(tableau.b.len(), tableau.stages);
        debug_assert_eq!(tableau.error.len(), tableau.stages);
        Self {
            dimension,
            tableau,
            stages: vec![0.0; dimension * tableau.stages],
            stage_state: vec![0.0; dimension],
            endpoint_derivative: vec![0.0; dimension],
            error: vec![0.0; dimension],
            endpoint_valid: false,
        }
    }

    fn stage(&self, index: usize) -> &[f64] {
        let start = index * self.dimension;
        &self.stages[start..start + self.dimension]
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
}

impl<F, P> StepKernel<F, P> for SimdRkKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(self.tableau.order, 0.9, 0.2, 6.0, 0.2),
        )
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        let dimension = self.dimension;
        Self::evaluate(problem, &mut self.stages[..dimension], state, time, stats)
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
        let scale = state
            .iter()
            .zip(self.stage(0))
            .map(|(state, derivative)| {
                derivative.abs()
                    / (options.absolute_tolerance + options.relative_tolerance * state.abs())
            })
            .fold(0.0_f64, f64::max);
        Ok((if scale == 0.0 { 1.0e-3 } else { 0.01 / scale }).clamp(f64::EPSILON, maximum_step))
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
        for stage in 1..self.tableau.stages {
            self.stage_state.copy_from_slice(state);
            for prior in 0..stage {
                let coefficient = self.tableau.a[stage * self.tableau.stages + prior];
                if coefficient == 0.0 {
                    continue;
                }
                let start = prior * self.dimension;
                for component in 0..self.dimension {
                    self.stage_state[component] +=
                        step * coefficient * self.stages[start + component];
                }
            }
            let stage_time = time + self.tableau.c[stage] * step;
            let dimension = self.dimension;
            let start = stage * dimension;
            Self::evaluate(
                problem,
                &mut self.stages[start..start + dimension],
                &self.stage_state,
                stage_time,
                stats,
            )?;
        }

        candidate.copy_from_slice(state);
        self.error.fill(0.0);
        for stage in 0..self.tableau.stages {
            let start = stage * self.dimension;
            let stage_values = &self.stages[start..start + self.dimension];
            for ((candidate, error), stage_value) in
                candidate.iter_mut().zip(&mut self.error).zip(stage_values)
            {
                *candidate += step * self.tableau.b[stage] * stage_value;
                *error += step * self.tableau.error[stage] * stage_value;
            }
        }
        let last = self.tableau.stages - 1;
        let last_start = last * self.dimension;
        self.endpoint_derivative
            .copy_from_slice(&self.stages[last_start..last_start + self.dimension]);
        self.endpoint_valid = true;
        let error_norm = if options.adaptive {
            scaled_error(&self.error, state, candidate, options)
        } else {
            0.0
        };
        Ok(StepEstimate::new(error_norm))
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
        let dimension = self.dimension;
        if self.endpoint_valid && !callback_applied {
            self.stages[..dimension].copy_from_slice(&self.endpoint_derivative);
            Ok(())
        } else {
            Self::evaluate(problem, &mut self.stages[..dimension], state, time, stats)
        }
    }

    fn reject_step(&mut self) {
        self.endpoint_valid = false;
    }
}

fn scaled_error(error: &[f64], old: &[f64], new: &[f64], options: &SolveOptions) -> f64 {
    (error
        .iter()
        .zip(old)
        .zip(new)
        .map(|((error, old), new)| {
            let scale =
                options.absolute_tolerance + options.relative_tolerance * old.abs().max(new.abs());
            (error / scale).powi(2)
        })
        .sum::<f64>()
        / error.len() as f64)
        .sqrt()
}
