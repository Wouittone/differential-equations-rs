use std::cell::RefCell;

use super::general::{mat_vec, matrix_exp};
use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::solver::validate_state_time_options;
use crate::tableau::{RungeKuttaTableau, TableauError, load_tableau};
use crate::{
    ConfigurationError, OdeProblem, SemilinearOdeProblem, Solution, SolveError, SolveOptions,
    SolverStats,
};

use differential_equations_tableau_macros::define_explicit_rk_tableau_from_file;

define_explicit_rk_tableau_from_file!(
    RKIP_TABLEAU,
    "RKIP",
    "src/tableau/resources/explicit/rkip.json",
    crate = crate
);

/// Observable interaction-picture exponential-cache counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RkipCacheStats {
    /// Matrix exponentials constructed since the algorithm was created.
    pub exponentials_built: usize,
    /// Cached matrix exponentials reused by subsequent stages or solves.
    pub cache_hits: usize,
    /// Distinct step magnitudes currently represented in the cache.
    pub cached_step_sizes: usize,
}

struct ExponentialCache {
    operator: Vec<f64>,
    positive: Vec<Vec<Option<Vec<f64>>>>,
    negative: Vec<Vec<Option<Vec<f64>>>>,
    stats: RkipCacheStats,
}

impl ExponentialCache {
    fn empty() -> Self {
        Self {
            operator: Vec::new(),
            positive: Vec::new(),
            negative: Vec::new(),
            stats: RkipCacheStats::default(),
        }
    }

    fn prepare(&mut self, operator: &[f64], steps: usize, stages: usize) {
        if self.operator != operator {
            self.operator = operator.to_vec();
            self.positive = vec![vec![None; stages]; steps];
            self.negative = vec![vec![None; stages]; steps];
            self.stats = RkipCacheStats::default();
        } else if self.positive.len() != steps {
            self.positive.resize_with(steps, || vec![None; stages]);
            self.negative.resize_with(steps, || vec![None; stages]);
        }
    }
}

/// Sixth-order adaptive Runge--Kutta in the interaction picture.
///
/// The exponential grid and its matrix exponentials are retained inside the
/// algorithm and recycled across solves that use the same dense operator.
pub struct RKIP {
    cached_steps: Vec<f64>,
    clamp_lower_dt: bool,
    clamp_higher_dt: bool,
    cache: RefCell<ExponentialCache>,
}

/// Algorithms specialized for semilinear interaction-picture problems.
pub trait InteractionPictureAlgorithm {
    /// Classical solution order.
    fn order(&self) -> usize;

    /// Embedded estimator order used by adaptive control.
    fn adaptive_order(&self) -> usize;

    /// Solves a semilinear problem after validating its common public inputs.
    fn solve_interaction_picture<G, P>(
        &self,
        problem: &SemilinearOdeProblem<G, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        G: Fn(&mut [f64], &[f64], &P, f64),
    {
        validate_state_time_options(problem.initial_state(), problem.time_span(), options)?;
        self.solve_validated(problem, options)
    }

    /// Executes the numerical method after common inputs have been checked.
    ///
    /// Implementors may rely on [`InteractionPictureAlgorithm::solve_interaction_picture`]
    /// having validated the state, time span, tolerances, step bounds, callback
    /// tolerance, and requested output times. User code should normally call
    /// that checked method or [`solve_rkip`]; direct callers of this lower-level
    /// hook are responsible for preserving those invariants.
    fn solve_validated<G, P>(
        &self,
        problem: &SemilinearOdeProblem<G, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        G: Fn(&mut [f64], &[f64], &P, f64);
}

fn cached_step_grid(dt_min: f64, dt_max: f64, count: usize) -> Vec<f64> {
    debug_assert!(count >= 2);
    let ratio = (dt_max / dt_min).powf(1.0 / (count - 1) as f64);
    (0..count)
        .map(|index| dt_min * ratio.powi(index as i32))
        .collect()
}

impl Default for RKIP {
    fn default() -> Self {
        Self {
            cached_steps: cached_step_grid(1.0e-3, 1.0, 100),
            clamp_lower_dt: false,
            clamp_higher_dt: true,
            cache: RefCell::new(ExponentialCache::empty()),
        }
    }
}

impl RKIP {
    /// Constructs an RKIP solver with a logarithmically spaced exponential cache.
    pub fn new(
        dt_min: f64,
        dt_max: f64,
        cached_step_count: usize,
    ) -> Result<Self, ConfigurationError> {
        if !dt_min.is_finite()
            || !dt_max.is_finite()
            || dt_min <= 0.0
            || dt_max < dt_min
            || cached_step_count < 2
        {
            return Err(ConfigurationError::InvalidBounds {
                context: "RKIP cache",
                reason: "step bounds must be finite, positive, ordered, and contain at least two steps",
            });
        }
        let cached_steps = cached_step_grid(dt_min, dt_max, cached_step_count);
        Ok(Self {
            cached_steps,
            clamp_lower_dt: false,
            clamp_higher_dt: true,
            cache: RefCell::new(ExponentialCache::empty()),
        })
    }

    /// Configures whether steps outside the cached range are clamped to it.
    ///
    /// After a rejected attempt, the error controller's reduction takes
    /// precedence over cache snapping, including the lower cache bound.
    #[must_use]
    pub fn with_clamping(mut self, lower: bool, higher: bool) -> Self {
        self.clamp_lower_dt = lower;
        self.clamp_higher_dt = higher;
        self
    }

    /// Returns the classical solution order.
    pub fn order(&self) -> usize {
        6
    }
    /// Returns the embedded estimator order.
    pub fn adaptive_order(&self) -> usize {
        5
    }
    /// Returns the compile-validated Verner pair, parsed lazily on first use.
    pub fn tableau(&self) -> Result<&'static RungeKuttaTableau, TableauError> {
        load_tableau(&RKIP_TABLEAU)
    }
    /// Returns observable exponential-cache usage statistics.
    pub fn cache_stats(&self) -> RkipCacheStats {
        self.cache.borrow().stats
    }

    fn snap(&self, step: f64) -> f64 {
        let sign = step.signum();
        let magnitude = step.abs();
        let first = self.cached_steps[0];
        let last = self.cached_steps.last().copied().unwrap_or(magnitude);
        if magnitude < first && !self.clamp_lower_dt {
            return step;
        }
        if magnitude > last && !self.clamp_higher_dt {
            return step;
        }
        let selected = self
            .cached_steps
            .iter()
            .copied()
            .find(|value| *value >= magnitude)
            .unwrap_or(last);
        sign * selected
    }
}

impl InteractionPictureAlgorithm for RKIP {
    fn order(&self) -> usize {
        self.order()
    }

    fn adaptive_order(&self) -> usize {
        self.adaptive_order()
    }

    fn solve_validated<G, P>(
        &self,
        problem: &SemilinearOdeProblem<G, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        G: Fn(&mut [f64], &[f64], &P, f64),
    {
        solve_rkip_kernel(problem, self, options)
    }
}

/// Solves a typed semilinear split with the pinned RKIP Verner 6(5) tableau.
pub fn solve_rkip<G, P, A>(
    problem: &SemilinearOdeProblem<G, P>,
    algorithm: &A,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    G: Fn(&mut [f64], &[f64], &P, f64),
    A: InteractionPictureAlgorithm,
{
    algorithm.solve_interaction_picture(problem, options)
}

fn solve_rkip_kernel<G, P>(
    problem: &SemilinearOdeProblem<G, P>,
    algorithm: &RKIP,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    G: Fn(&mut [f64], &[f64], &P, f64),
{
    let dummy = OdeProblem::new(
        noop as fn(&mut [f64], &[f64], &(), f64),
        problem.initial_state().to_vec(),
        problem.time_span(),
        (),
    );
    let kernel = RkipKernel::new(problem, algorithm)?;
    algorithm.cache.borrow_mut().prepare(
        problem.linear_operator(),
        algorithm.cached_steps.len(),
        kernel.tableau.b().len(),
    );
    drive_integration(&dummy, options, kernel)
}

fn noop(_: &mut [f64], _: &[f64], _: &(), _: f64) {}

struct RkipKernel<'a, G, P> {
    problem: &'a SemilinearOdeProblem<G, P>,
    algorithm: &'a RKIP,
    tableau: &'static RungeKuttaTableau,
    error_weights: &'static [f64],
    retry_step: bool,
    stages: Vec<Vec<f64>>,
    derivative: Vec<f64>,
}

impl<'a, G, P> RkipKernel<'a, G, P> {
    fn new(
        problem: &'a SemilinearOdeProblem<G, P>,
        algorithm: &'a RKIP,
    ) -> Result<Self, SolveError> {
        let tableau = algorithm
            .tableau()
            .map_err(|_| SolveError::InvalidTableau)?;
        let error_weights = tableau.error().ok_or(SolveError::InvalidTableau)?;
        Ok(Self {
            problem,
            algorithm,
            tableau,
            error_weights,
            retry_step: false,
            stages: vec![vec![0.0; problem.dimension()]; tableau.b().len()],
            derivative: vec![0.0; problem.dimension()],
        })
    }

    fn action(&self, vector: &[f64], step: f64, c: f64) -> Vec<f64> {
        if c == 0.0 || step == 0.0 {
            return vector.to_vec();
        }
        let magnitude = step.abs();
        // Equal nodes share the first matching slot. An unlisted node is
        // evaluated without caching rather than aliasing another exponential.
        let stage_index = self.tableau.c().iter().position(|node| *node == c);
        let grid_index = self
            .algorithm
            .cached_steps
            .iter()
            .position(|value| approx(*value, magnitude));
        let positive = step.is_sign_positive();
        let mut cache = self.algorithm.cache.borrow_mut();
        let matrix = if let (Some(grid_index), Some(stage_index)) = (grid_index, stage_index) {
            let existing = if positive {
                &cache.positive[grid_index][stage_index]
            } else {
                &cache.negative[grid_index][stage_index]
            }
            .clone();
            if let Some(matrix) = existing {
                cache.stats.cache_hits += 1;
                matrix
            } else {
                let matrix = matrix_exp(&scale(cache.operator.as_slice(), step * c), vector.len());
                if positive {
                    cache.positive[grid_index][stage_index] = Some(matrix.clone());
                } else {
                    cache.negative[grid_index][stage_index] = Some(matrix.clone());
                }
                cache.stats.exponentials_built += 1;
                cache.stats.cached_step_sizes = cache
                    .positive
                    .iter()
                    .filter(|row| row.iter().any(Option::is_some))
                    .count();
                matrix
            }
        } else {
            cache.stats.exponentials_built += 1;
            matrix_exp(&scale(cache.operator.as_slice(), step * c), vector.len())
        };
        mat_vec(&matrix, vector)
    }
}

impl<G, P> StepKernel<fn(&mut [f64], &[f64], &(), f64), ()> for RkipKernel<'_, G, P>
where
    G: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(
                self.tableau
                    .embedded_order()
                    .unwrap_or(self.tableau.order()),
                0.9,
                0.2,
                6.0,
                0.2,
            ),
        )
    }
    fn modify_step(&mut self, proposed_step: f64) -> f64 {
        let snapped = self.algorithm.snap(proposed_step);
        if std::mem::take(&mut self.retry_step) && snapped.abs() > proposed_step.abs() {
            proposed_step
        } else {
            snapped
        }
    }
    fn evaluate_dense_derivative(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.problem.evaluate(output, state, time);
        stats.rhs_evaluations += 1;
        checked(output)
    }
    fn initialize(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.problem.evaluate(&mut self.derivative, state, time);
        stats.rhs_evaluations += 1;
        checked(&self.derivative)
    }
    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        options: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        let mut scale_norm = 0.0_f64;
        for (state, derivative) in state.iter().zip(&self.derivative) {
            let tolerance = options.absolute_tolerance + options.relative_tolerance * state.abs();
            scale_norm = scale_norm.max((derivative / tolerance).abs());
        }
        Ok((if scale_norm == 0.0 {
            1.0e-3
        } else {
            0.01 / scale_norm
        })
        .clamp(f64::EPSILON, maximum_step))
    }
    fn attempt_step(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let n = state.len();
        for i in 0..self.stages.len() {
            let mut interaction = state.to_vec();
            for (coefficient, stage) in self.tableau.stage_row(i).iter().zip(&self.stages) {
                for (value, stage) in interaction.iter_mut().zip(stage) {
                    *value += step * coefficient * stage;
                }
            }
            let c = self.tableau.c()[i];
            let true_state = self.action(&interaction, step, c);
            self.problem
                .evaluate_nonlinear(&mut self.stages[i], &true_state, time + c * step);
            stats.rhs_evaluations += 1;
            checked(&self.stages[i])?;
            self.stages[i] = self.action(&self.stages[i], -step, c);
        }
        let mut interaction = state.to_vec();
        let mut embedded_error = vec![0.0; n];
        for i in 0..self.stages.len() {
            for k in 0..n {
                interaction[k] += step * self.tableau.b()[i] * self.stages[i][k];
                embedded_error[k] += step * self.error_weights[i] * self.stages[i][k];
            }
        }
        candidate.copy_from_slice(&self.action(&interaction, step, 1.0));
        embedded_error = self.action(&embedded_error, step, 1.0);
        checked(candidate)?;
        Ok(StepEstimate::new(if options.adaptive {
            scaled_error(&embedded_error, state, candidate, options)
        } else {
            0.0
        }))
    }
    fn accept_step(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        _: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.problem.evaluate(&mut self.derivative, state, time);
        stats.rhs_evaluations += 1;
        checked(&self.derivative)
    }
    fn reject_step(&mut self) {
        self.retry_step = true;
    }
}

fn approx(left: f64, right: f64) -> bool {
    (left - right).abs() <= 16.0 * f64::EPSILON * left.abs().max(right.abs()).max(1.0)
}
fn scale(matrix: &[f64], factor: f64) -> Vec<f64> {
    matrix.iter().map(|value| factor * value).collect()
}
fn checked(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}
fn scaled_error(error: &[f64], old: &[f64], new: &[f64], options: &SolveOptions) -> f64 {
    let sum = error
        .iter()
        .zip(old)
        .zip(new)
        .map(|((error, old), new)| {
            let tolerance =
                options.absolute_tolerance + options.relative_tolerance * old.abs().max(new.abs());
            (error / tolerance).powi(2)
        })
        .sum::<f64>();
    (sum / error.len() as f64).sqrt()
}

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    fn coefficients_preserve_legacy_bit_patterns() {
        let tableau = RKIP::default().tableau().unwrap();
        let mut hash = 0xcbf29ce484222325_u64;
        for value in tableau
            .a()
            .iter()
            .flatten()
            .chain(tableau.b())
            .chain(tableau.c())
            .chain(tableau.error().unwrap())
        {
            for byte in value.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        // FNV-1a over A, b, c, and b-b_hat from the pre-migration Rust constants.
        assert_eq!(hash, 0x554bfb7076b25bbb);
    }
}
