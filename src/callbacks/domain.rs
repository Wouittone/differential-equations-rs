use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use super::{ManifoldProjection, validate_reduction_factor_named};
use crate::{CallbackSave, CallbackSet, ConfigurationError, SolveError};

type DomainResidual<P> = dyn Fn(&mut [f64], &[f64], &P, f64);

/// Restricts predicted domain violations and projects accepted states.
///
/// The residual defines the domain by its zero set. For a region rather than
/// an equality constraint, use a residual that is zero throughout the region
/// and positive outside it. Before each attempt, the policy evaluates the
/// residual at the forward-Euler extrapolation `state + step * derivative`
/// and the corresponding future time. Each residual must be strictly below
/// the domain tolerance. This is a signed comparison, not an absolute norm.
/// An unacceptable prediction repeatedly reduces the step, with a further
/// `0.9` safety factor once an acceptable prediction is found.
///
/// A [`ManifoldProjection`] runs after initialization and every accepted step
/// in callback insertion order. Its tolerance is separate from the predictor's
/// tolerance: the predictor defaults to the solve's absolute tolerance, while
/// projection defaults to `10 * f64::EPSILON`. Nonlinear projection can only
/// enforce proximity to the zero set, not exact membership in an inequality
/// domain. Prefer [`super::PositiveDomain`] for componentwise non-negativity.
///
/// This policy supports ordinary and split first-order problems, including
/// ndarray-shaped states, fixed or adaptive stepping, and backward solves.
/// Residuals and Jacobians receive the state's contiguous flat representation.
/// The model and residual must remain defined outside the desired domain.
/// As with [`ManifoldProjection`], interior dense-output samples are not
/// projected; use matching `save_at` and `time_stops` when needed.
///
/// # Example
///
/// ```
/// use differential_equations::callbacks::GeneralDomain;
/// use differential_equations::solvers::explicit::Euler;
/// use differential_equations::{OdeProblem, SolveOptions, solve};
///
/// let domain = GeneralDomain::new(
///     1,
///     |residual: &mut [f64], state: &[f64], _: &(), _| {
///         residual[0] = state[0] * state[0] + state[1] * state[1] - 1.0;
///     },
/// )
/// .with_absolute_tolerance(1.0e-3)
/// .into_callback_set()?;
/// let problem = OdeProblem::new(
///     |du: &mut [f64], u: &[f64], _: &(), _| {
///         du[0] = u[1];
///         du[1] = -u[0];
///     },
///     [1.0, 0.0],
///     (0.0, 1.0),
///     (),
/// ).with_callback_set(domain);
/// let options = SolveOptions::new()
///     .with_adaptive(false)
///     .with_initial_step(0.1);
/// let solution = solve(&problem, Euler, &options)?;
/// let state = solution.last_state();
/// assert!((state[0] * state[0] + state[1] * state[1] - 1.0).abs() < 1.0e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub struct GeneralDomain<P> {
    residual_dimension: usize,
    residual: Rc<DomainResidual<P>>,
    projection: ManifoldProjection<P>,
    absolute_tolerance: Option<f64>,
    reduction_factor: f64,
}

impl<P: 'static> GeneralDomain<P> {
    /// Defines a domain with an in-place residual of the given dimension.
    ///
    /// The residual must overwrite every output entry. The dimension must be
    /// positive and cannot exceed the number of state components.
    pub fn new<R>(residual_dimension: usize, residual: R) -> Self
    where
        R: Fn(&mut [f64], &[f64], &P, f64) + 'static,
    {
        let residual: Rc<DomainResidual<P>> = Rc::new(residual);
        let projection_residual = Rc::clone(&residual);
        let projection = ManifoldProjection::new(
            residual_dimension,
            move |output, state, parameters, time| {
                projection_residual(output, state, parameters, time);
            },
        )
        .with_absolute_tolerance(10.0 * f64::EPSILON);
        Self {
            residual_dimension,
            residual,
            projection,
            absolute_tolerance: None,
            reduction_factor: 0.5,
        }
    }

    /// Overrides the predictor's residual tolerance, not the projection tolerance.
    ///
    /// The value must be finite and non-negative. Without an override,
    /// [`crate::SolveOptions::absolute_tolerance`] is used.
    pub const fn with_absolute_tolerance(mut self, absolute_tolerance: f64) -> Self {
        self.absolute_tolerance = Some(absolute_tolerance);
        self
    }

    /// Sets the factor used to reduce unacceptable predicted steps.
    ///
    /// The factor must lie strictly between zero and one; the default is `0.5`.
    pub const fn with_reduction_factor(mut self, reduction_factor: f64) -> Self {
        self.reduction_factor = reduction_factor;
        self
    }

    /// Supplies the projection's analytic row-major residual Jacobian.
    ///
    /// Its dimensions are `residual_dimension × state_dimension`, and the
    /// function must overwrite every entry.
    pub fn with_jacobian<J>(mut self, jacobian: J) -> Self
    where
        J: Fn(&mut [f64], &[f64], &P, f64) + 'static,
    {
        self.projection = self.projection.with_jacobian(jacobian);
        self
    }

    /// Sets the projection's accepted residual infinity norm.
    ///
    /// This must be finite and positive; its default is `10 * f64::EPSILON`.
    pub fn with_projection_absolute_tolerance(mut self, tolerance: f64) -> Self {
        self.projection = self.projection.with_absolute_tolerance(tolerance);
        self
    }

    /// Sets the projection's relative finite-difference perturbation.
    pub fn with_finite_difference_step(mut self, step: f64) -> Self {
        self.projection = self.projection.with_finite_difference_step(step);
        self
    }

    /// Sets the maximum number of Newton corrections per projection.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.projection = self.projection.with_max_iterations(max_iterations);
        self
    }

    /// Selects which states are saved around each projection.
    pub fn with_save(mut self, save: CallbackSave) -> Self {
        self.projection = self.projection.with_save(save);
        self
    }

    /// Builds a predictor and ordered projection for an ordinary or split ODE.
    ///
    /// The predictor reuses its residual buffer across attempts and solves.
    /// Invalid settings return a configuration error. Non-finite predictor
    /// residuals return [`SolveError::NonFiniteDomainResidual`] during solving;
    /// projection failures retain [`ManifoldProjection`]'s typed errors.
    pub fn into_callback_set(self) -> Result<CallbackSet<P>, ConfigurationError> {
        if matches!(
            self.absolute_tolerance,
            Some(tolerance) if !tolerance.is_finite() || tolerance < 0.0
        ) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "general-domain absolute tolerance",
                reason: "must be finite and non-negative",
            });
        }
        validate_reduction_factor_named(self.reduction_factor, "general-domain reduction factor")?;
        let callbacks = self.projection.into_callback_set()?;
        let residual = self.residual;
        let tolerance = self.absolute_tolerance;
        let scratch = RefCell::new(vec![f64::NAN; self.residual_dimension]);
        Ok(callbacks.with_predictive_domain(
            self.reduction_factor,
            move |prediction, parameters, time, default_tolerance| {
                let mut output = scratch.borrow_mut();
                output.fill(f64::NAN);
                residual(&mut output, prediction, parameters, time);
                if output.iter().any(|value| !value.is_finite()) {
                    return Err(SolveError::NonFiniteDomainResidual);
                }
                let tolerance = tolerance.unwrap_or(default_tolerance);
                Ok(output.iter().all(|value| *value < tolerance))
            },
        ))
    }
}

impl<P> fmt::Debug for GeneralDomain<P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeneralDomain")
            .field("projection", &self.projection)
            .field("absolute_tolerance", &self.absolute_tolerance)
            .field("reduction_factor", &self.reduction_factor)
            .finish_non_exhaustive()
    }
}
