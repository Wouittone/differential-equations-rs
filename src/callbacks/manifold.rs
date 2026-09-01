use std::cell::RefCell;
use std::fmt;

use crate::callback::{CallbackAction, CallbackSave, CallbackSet};
use crate::linear::{factorize, solve_factorized};
use crate::{ConfigurationError, SolveError};

type Residual<P> = dyn Fn(&mut [f64], &[f64], &P, f64);
type Jacobian<P> = dyn Fn(&mut [f64], &[f64], &P, f64);

const DEFAULT_ABSOLUTE_TOLERANCE: f64 = 1.0e-10;
const DEFAULT_MAX_ITERATIONS: usize = 12;
const MAX_LINE_SEARCH_ITERATIONS: usize = 12;

/// Projects accepted states onto an implicitly defined manifold.
///
/// The manifold is the zero set of `residual(state, parameters, time)`. Its
/// residual dimension may be smaller than the state dimension, which supports
/// conservation laws such as one energy constraint on a multi-component
/// system. Projection uses Newton corrections in the row space of the
/// residual Jacobian and backtracks corrections that do not reduce the
/// residual norm.
///
/// By default, the Jacobian is approximated with finite differences. Supply an
/// analytic row-major `residual_dimension × state_dimension` Jacobian through
/// [`Self::with_jacobian`] when evaluations are expensive or the finite
/// difference is poorly conditioned.
///
/// Projection changes accepted endpoints after their dense segment was
/// constructed. Interior `save_at` samples therefore remain values of the
/// unprojected numerical interpolant. Add those same times to
/// [`crate::SolveOptions::time_stops`] when each requested sample must itself
/// be projected.
///
/// # Example
///
/// ```
/// use differential_equations::callbacks::ManifoldProjection;
/// use differential_equations::solvers::explicit::Euler;
/// use differential_equations::{CallbackSave, OdeProblem, SolveOptions, solve};
///
/// let projection = ManifoldProjection::new(
///     1,
///     |residual: &mut [f64], state: &[f64], _: &(), _| {
///         residual[0] = state[0] * state[0] + state[1] * state[1] - 1.0;
///     },
/// )
/// .with_save(CallbackSave::None)
/// .into_callback_set()?;
/// let problem = OdeProblem::new(
///     |derivative: &mut [f64], state: &[f64], _: &(), _| {
///         derivative[0] = state[1];
///         derivative[1] = -state[0];
///     },
///     [1.0, 0.0],
///     (0.0, 1.0),
///     (),
/// )
/// .with_callback_set(projection);
/// let options = SolveOptions::new()
///     .with_adaptive(false)
///     .with_initial_step(0.1);
/// let solution = solve(&problem, Euler, &options)?;
/// let state = solution.last_state();
/// assert!((state[0] * state[0] + state[1] * state[1] - 1.0).abs() < 1.0e-9);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub struct ManifoldProjection<P> {
    residual_dimension: usize,
    residual: Box<Residual<P>>,
    jacobian: Option<Box<Jacobian<P>>>,
    absolute_tolerance: f64,
    finite_difference_step: f64,
    max_iterations: usize,
    save: CallbackSave,
}

impl<P> ManifoldProjection<P> {
    /// Defines a manifold with `residual_dimension` independent constraints.
    ///
    /// The residual function must overwrite every output entry.
    pub fn new<R>(residual_dimension: usize, residual: R) -> Self
    where
        R: Fn(&mut [f64], &[f64], &P, f64) + 'static,
    {
        Self {
            residual_dimension,
            residual: Box::new(residual),
            jacobian: None,
            absolute_tolerance: DEFAULT_ABSOLUTE_TOLERANCE,
            finite_difference_step: f64::EPSILON.sqrt(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            save: CallbackSave::After,
        }
    }

    /// Supplies an analytic row-major residual Jacobian.
    pub fn with_jacobian<J>(mut self, jacobian: J) -> Self
    where
        J: Fn(&mut [f64], &[f64], &P, f64) + 'static,
    {
        self.jacobian = Some(Box::new(jacobian));
        self
    }

    /// Sets the maximum accepted infinity norm of the manifold residual.
    pub const fn with_absolute_tolerance(mut self, absolute_tolerance: f64) -> Self {
        self.absolute_tolerance = absolute_tolerance;
        self
    }

    /// Sets the relative perturbation used by the finite-difference Jacobian.
    pub const fn with_finite_difference_step(mut self, finite_difference_step: f64) -> Self {
        self.finite_difference_step = finite_difference_step;
        self
    }

    /// Sets the maximum number of Newton corrections per callback invocation.
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Selects which states are saved around a successful projection.
    pub const fn with_save(mut self, save: CallbackSave) -> Self {
        self.save = save;
        self
    }

    /// Builds an ordered callback policy for an ordinary or split ODE problem.
    pub fn into_callback_set(self) -> Result<CallbackSet<P>, ConfigurationError>
    where
        P: 'static,
    {
        self.validate()?;
        let save = self.save;
        let engine = RefCell::new(ProjectionEngine::new(self));
        Ok(CallbackSet::new().with_fallible_discrete_callback_saving(
            save,
            |_, _, _| true,
            move |state, parameters, time| engine.borrow_mut().project(state, parameters, time),
        ))
    }

    fn validate(&self) -> Result<(), ConfigurationError> {
        if self.residual_dimension == 0 {
            return Err(ConfigurationError::EmptyData {
                context: "manifold residual",
            });
        }
        validate_positive_finite(
            self.absolute_tolerance,
            "manifold projection absolute tolerance",
        )?;
        validate_positive_finite(
            self.finite_difference_step,
            "manifold projection finite-difference step",
        )?;
        if self.max_iterations == 0 {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "manifold projection maximum iterations",
                reason: "must be positive",
            });
        }
        Ok(())
    }
}

impl<P> fmt::Debug for ManifoldProjection<P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManifoldProjection")
            .field("residual_dimension", &self.residual_dimension)
            .field("has_analytic_jacobian", &self.jacobian.is_some())
            .field("absolute_tolerance", &self.absolute_tolerance)
            .field("finite_difference_step", &self.finite_difference_step)
            .field("max_iterations", &self.max_iterations)
            .field("save", &self.save)
            .finish_non_exhaustive()
    }
}

struct ProjectionEngine<P> {
    residual_dimension: usize,
    residual_function: Box<Residual<P>>,
    jacobian_function: Option<Box<Jacobian<P>>>,
    absolute_tolerance: f64,
    finite_difference_step: f64,
    max_iterations: usize,
    workspace: ProjectionWorkspace,
}

impl<P: 'static> ProjectionEngine<P> {
    fn new(configuration: ManifoldProjection<P>) -> Self {
        Self {
            residual_dimension: configuration.residual_dimension,
            residual_function: configuration.residual,
            jacobian_function: configuration.jacobian,
            absolute_tolerance: configuration.absolute_tolerance,
            finite_difference_step: configuration.finite_difference_step,
            max_iterations: configuration.max_iterations,
            workspace: ProjectionWorkspace::default(),
        }
    }

    fn project(
        &mut self,
        state: &mut [f64],
        parameters: &P,
        time: f64,
    ) -> Result<CallbackAction, SolveError> {
        let state_dimension = state.len();
        if self.residual_dimension > state_dimension {
            return Err(SolveError::InvalidManifoldDimension);
        }
        self.workspace
            .resize(self.residual_dimension, state_dimension)?;
        evaluate_residual(
            &self.residual_function,
            &mut self.workspace.residual,
            state,
            parameters,
            time,
        )?;
        let mut residual_norm = infinity_norm(&self.workspace.residual);
        if residual_norm <= self.absolute_tolerance {
            return Ok(CallbackAction::ContinueUnmodified);
        }

        for _ in 0..self.max_iterations {
            self.evaluate_jacobian(state, parameters, time)?;
            form_normal_matrix(
                &self.workspace.jacobian,
                self.residual_dimension,
                state_dimension,
                &mut self.workspace.normal_matrix,
            );
            self.workspace
                .multipliers
                .iter_mut()
                .zip(&self.workspace.residual)
                .for_each(|(multiplier, residual)| *multiplier = -*residual);
            factorize(
                &mut self.workspace.normal_matrix,
                &mut self.workspace.pivots,
                self.residual_dimension,
            )
            .map_err(|_| SolveError::ManifoldProjectionFailed)?;
            solve_factorized(
                &self.workspace.normal_matrix,
                &self.workspace.pivots,
                &mut self.workspace.multipliers,
                self.residual_dimension,
            );
            form_state_correction(
                &self.workspace.jacobian,
                &self.workspace.multipliers,
                self.residual_dimension,
                state_dimension,
                &mut self.workspace.correction,
            );

            let mut scale = 1.0;
            let mut accepted = false;
            for _ in 0..=MAX_LINE_SEARCH_ITERATIONS {
                for ((trial, state), correction) in self
                    .workspace
                    .trial_state
                    .iter_mut()
                    .zip(state.iter())
                    .zip(&self.workspace.correction)
                {
                    *trial = state + scale * correction;
                }
                if evaluate_residual_if_finite(
                    &self.residual_function,
                    &mut self.workspace.trial_residual,
                    &self.workspace.trial_state,
                    parameters,
                    time,
                ) {
                    let trial_norm = infinity_norm(&self.workspace.trial_residual);
                    if trial_norm < residual_norm {
                        state.copy_from_slice(&self.workspace.trial_state);
                        self.workspace
                            .residual
                            .copy_from_slice(&self.workspace.trial_residual);
                        residual_norm = trial_norm;
                        accepted = true;
                        break;
                    }
                }
                scale *= 0.5;
            }
            if !accepted {
                return Err(SolveError::ManifoldProjectionFailed);
            }
            if residual_norm <= self.absolute_tolerance {
                return Ok(CallbackAction::Continue);
            }
        }
        Err(SolveError::ManifoldProjectionFailed)
    }

    fn evaluate_jacobian(
        &mut self,
        state: &[f64],
        parameters: &P,
        time: f64,
    ) -> Result<(), SolveError> {
        if let Some(jacobian) = &self.jacobian_function {
            self.workspace.jacobian.fill(f64::NAN);
            jacobian(&mut self.workspace.jacobian, state, parameters, time);
            return self
                .workspace
                .jacobian
                .iter()
                .all(|value| value.is_finite())
                .then_some(())
                .ok_or(SolveError::NonFiniteManifoldProjection);
        }

        let state_dimension = state.len();
        self.workspace.trial_state.copy_from_slice(state);
        for (column, &value) in state.iter().enumerate() {
            let increment = self.finite_difference_step * value.abs().max(1.0);
            self.workspace.trial_state[column] = value + increment;
            evaluate_residual(
                &self.residual_function,
                &mut self.workspace.trial_residual,
                &self.workspace.trial_state,
                parameters,
                time,
            )?;
            for row in 0..self.residual_dimension {
                self.workspace.jacobian[row * state_dimension + column] =
                    (self.workspace.trial_residual[row] - self.workspace.residual[row]) / increment;
            }
            self.workspace.trial_state[column] = value;
        }
        self.workspace
            .jacobian
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteManifoldProjection)
    }
}

#[derive(Default)]
struct ProjectionWorkspace {
    residual: Vec<f64>,
    trial_residual: Vec<f64>,
    jacobian: Vec<f64>,
    normal_matrix: Vec<f64>,
    multipliers: Vec<f64>,
    pivots: Vec<usize>,
    correction: Vec<f64>,
    trial_state: Vec<f64>,
}

impl ProjectionWorkspace {
    fn resize(
        &mut self,
        residual_dimension: usize,
        state_dimension: usize,
    ) -> Result<(), SolveError> {
        let jacobian_length = residual_dimension
            .checked_mul(state_dimension)
            .ok_or(SolveError::ManifoldProjectionFailed)?;
        let normal_length = residual_dimension
            .checked_mul(residual_dimension)
            .ok_or(SolveError::ManifoldProjectionFailed)?;
        self.residual.resize(residual_dimension, 0.0);
        self.trial_residual.resize(residual_dimension, 0.0);
        self.jacobian.resize(jacobian_length, 0.0);
        self.normal_matrix.resize(normal_length, 0.0);
        self.multipliers.resize(residual_dimension, 0.0);
        self.pivots.resize(residual_dimension, 0);
        self.correction.resize(state_dimension, 0.0);
        self.trial_state.resize(state_dimension, 0.0);
        Ok(())
    }
}

fn evaluate_residual<P>(
    function: &Residual<P>,
    output: &mut [f64],
    state: &[f64],
    parameters: &P,
    time: f64,
) -> Result<(), SolveError> {
    output.fill(f64::NAN);
    function(output, state, parameters, time);
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteManifoldProjection)
}

fn evaluate_residual_if_finite<P>(
    function: &Residual<P>,
    output: &mut [f64],
    state: &[f64],
    parameters: &P,
    time: f64,
) -> bool {
    output.fill(f64::NAN);
    function(output, state, parameters, time);
    output.iter().all(|value| value.is_finite())
}

fn form_normal_matrix(
    jacobian: &[f64],
    residual_dimension: usize,
    state_dimension: usize,
    normal: &mut [f64],
) {
    for row in 0..residual_dimension {
        for column in 0..residual_dimension {
            normal[row * residual_dimension + column] = (0..state_dimension)
                .map(|state_index| {
                    jacobian[row * state_dimension + state_index]
                        * jacobian[column * state_dimension + state_index]
                })
                .sum();
        }
    }
}

fn form_state_correction(
    jacobian: &[f64],
    multipliers: &[f64],
    residual_dimension: usize,
    state_dimension: usize,
    correction: &mut [f64],
) {
    for state_index in 0..state_dimension {
        correction[state_index] = (0..residual_dimension)
            .map(|residual_index| {
                jacobian[residual_index * state_dimension + state_index]
                    * multipliers[residual_index]
            })
            .sum();
    }
}

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn validate_positive_finite(value: f64, parameter: &'static str) -> Result<(), ConfigurationError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ConfigurationError::InvalidParameter {
            parameter,
            reason: "must be finite and positive",
        });
    }
    Ok(())
}
