//! Fixed-step eighth-order Gauss–Jackson integration of `q'' = a(t, q, v)`.
//!
//! The steady-state method is the summed Adams / double-sum Gauss–Jackson
//! predictor-corrector of Berry and Healy (2004), with nine acceleration samples.
//! Startup and a final partial interval use tenth-order extrapolated midpoint.
//! This avoids evaluating a force outside the requested integration domain.
//! Startup is explicitly counted, and partial intervals invalidate history.
//! Restart after any force/state discontinuity; never reuse history across one.

use std::{error::Error, fmt};

crate::tableau::define_gauss_jackson_tableau_from_file!(
    pub(super) TABLEAU,
    "GaussJackson8",
    "src/tableau/resources/gauss_jackson/gauss_jackson8.json",
    crate = crate
);

fn tableau() -> &'static crate::tableau::GaussJacksonTableau {
    crate::tableau::load_tableau(&TABLEAU).expect("validated Gauss-Jackson tableau resource")
}

/// Iteration settings for the fixed-step method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussJacksonConfig {
    /// Absolute component threshold for startup and corrector iteration.
    pub absolute_tolerance: f64,
    /// Relative component threshold for startup and corrector iteration.
    pub relative_tolerance: f64,
    /// Maximum corrected force evaluations per steady-state step.
    pub max_corrector_iterations: usize,
    /// Maximum doublings of midpoint substeps during startup/tail integration.
    pub max_startup_refinements: usize,
}
impl Default for GaussJacksonConfig {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-14,
            relative_tolerance: 1e-13,
            max_corrector_iterations: 12,
            max_startup_refinements: 5,
        }
    }
}
/// Separate work counters, including startup and failed force evaluations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GaussJacksonStatistics {
    /// Acceleration calls, including those returning an application error.
    pub acceleration_evaluations: usize,
    /// Successfully committed steps, including startup and partial steps.
    pub accepted_steps: usize,
    /// Committed startup/partial steps using extrapolated midpoint.
    pub startup_steps: usize,
    /// Completed corrector iterations.
    pub corrector_iterations: usize,
    /// Additional midpoint-grid doublings.
    pub startup_refinements: usize,
}
/// Solver failure with the application's original force error preserved.
#[derive(Debug)]
#[non_exhaustive]
pub enum GaussJacksonError<E> {
    /// Invalid finite time, state, step, configuration or dimensions.
    InvalidInput(&'static str),
    /// Acceleration callback failure.
    Acceleration(E),
    /// Force or state contained a nonfinite value.
    NonFinite,
    /// Corrector failed to converge; accepted state/history are unchanged.
    CorrectorConvergence {
        /// Iterations attempted before failure.
        iterations: usize,
    },
    /// Startup extrapolation failed its requested threshold.
    StartupConvergence {
        /// Number of midpoint-grid doublings attempted.
        refinements: usize,
    },
}
impl<E: fmt::Display> fmt::Display for GaussJacksonError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid Gauss-Jackson input: {s}"),
            Self::Acceleration(e) => write!(f, "Gauss-Jackson acceleration failed: {e}"),
            Self::NonFinite => write!(f, "Gauss-Jackson encountered nonfinite force or state"),
            Self::CorrectorConvergence { iterations } => write!(
                f,
                "Gauss-Jackson corrector did not converge in {iterations} iterations"
            ),
            Self::StartupConvergence { refinements } => write!(
                f,
                "Gauss-Jackson startup did not converge after {refinements} refinements"
            ),
        }
    }
}
impl<E: Error + 'static> Error for GaussJacksonError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Acceleration(e) => Some(e),
            _ => None,
        }
    }
}
/// Borrowed result of a committed fixed step.
#[derive(Clone, Copy, Debug)]
pub struct GaussJacksonStep<'a> {
    /// Exact accepted time.
    pub time: f64,
    /// Accepted positions.
    pub position: &'a [f64],
    /// Accepted velocities.
    pub velocity: &'a [f64],
    /// Whether this step used startup/partial-step extrapolation.
    pub startup: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct StateComponent {
    value: Box<[f64]>,
    center: Box<[f64]>,
    fixed: Box<[f64]>,
    candidate: Box<[f64]>,
    /// Snapshot taken at the start of the current dense interval.
    dense_start: Box<[f64]>,
}

#[derive(Clone, Debug)]
struct AccelerationState {
    initial: Box<[f64]>,
    candidate: Box<[f64]>,
    next: Box<[f64]>,
    dense_start: Box<[f64]>,
    dense_end: Box<[f64]>,
}

fn zeroed(dimension: usize) -> Box<[f64]> {
    vec![0.; dimension].into_boxed_slice()
}

impl AccelerationState {
    fn new(dimension: usize) -> Self {
        Self {
            initial: zeroed(dimension),
            candidate: zeroed(dimension),
            next: zeroed(dimension),
            dense_start: zeroed(dimension),
            dense_end: zeroed(dimension),
        }
    }
}

impl StateComponent {
    fn new(dimension: usize) -> Self {
        Self {
            value: zeroed(dimension),
            center: zeroed(dimension),
            fixed: zeroed(dimension),
            candidate: zeroed(dimension),
            dense_start: zeroed(dimension),
        }
    }
}

/// Persistent Gauss–Jackson state and reusable workspace.
///
/// Construct once, then call [`Self::try_step`] with a mutable fallible force.
/// No allocations occur in steps after construction. Cloning intentionally
/// copies all state/history and is a continuation snapshot; continuing a clone
/// requires the same force and parameters. Force mutation or discontinuities
/// require [`Self::restart`]. State is committed only after convergence.
#[derive(Clone, Debug)]
pub struct GaussJackson8 {
    time: f64,
    step: f64,
    config: GaussJacksonConfig,
    position: StateComponent,
    velocity: StateComponent,
    history: Box<[f64]>,
    history_len: usize,
    sum: Box<[f64]>,
    double_sum: Box<[f64]>,
    acceleration: AccelerationState,
    previous: Box<[f64]>,
    current: Box<[f64]>,
    next: Box<[f64]>,
    extrapolation: Box<[f64]>,
    dense_interval_start: f64,
    dense_valid: bool,
    statistics: GaussJacksonStatistics,
}
impl GaussJackson8 {
    /// Allocates workspace and validates a signed nonzero fixed step.
    /// It is quantized to the representable time increment at the initial epoch.
    /// If the representable grid spacing changes later, history safely restarts.
    pub fn new(
        time: f64,
        position: &[f64],
        velocity: &[f64],
        step: f64,
        config: GaussJacksonConfig,
    ) -> Result<Self, GaussJacksonError<std::convert::Infallible>> {
        validate_input(time, position, velocity, step, config)?;
        let n = position.len();
        let step = (time + step) - time;
        let mut solver = Self {
            time,
            step,
            config,
            position: StateComponent::new(n),
            velocity: StateComponent::new(n),
            history: zeroed(9 * n),
            history_len: 0,
            sum: zeroed(n),
            double_sum: zeroed(n),
            acceleration: AccelerationState::new(n),
            previous: zeroed(2 * n),
            current: zeroed(2 * n),
            next: zeroed(2 * n),
            extrapolation: zeroed(10 * n),
            dense_interval_start: time,
            dense_valid: false,
            statistics: GaussJacksonStatistics::default(),
        };
        solver.position.value.copy_from_slice(position);
        solver.velocity.value.copy_from_slice(velocity);
        Ok(solver)
    }
    /// Current accepted time.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Borrowed accepted position, without conversion.
    pub fn position(&self) -> &[f64] {
        &self.position.value
    }
    /// Borrowed accepted velocity, without conversion.
    pub fn velocity(&self) -> &[f64] {
        &self.velocity.value
    }
    /// Signed representable fixed step used by the history.
    /// The requested step is rounded to `(time + requested_step) - time`.
    pub fn step_size(&self) -> f64 {
        self.step
    }
    /// Current work counters.
    pub fn statistics(&self) -> GaussJacksonStatistics {
        self.statistics
    }
    /// Number of acceleration samples currently retained (at most nine).
    pub fn history_len(&self) -> usize {
        self.history_len
    }
    /// Clears history after a changed state, parameters, direction or force.
    /// Reuses all storage; dimensions must remain unchanged. Statistics persist.
    pub fn restart(
        &mut self,
        time: f64,
        position: &[f64],
        velocity: &[f64],
        step: f64,
    ) -> Result<(), GaussJacksonError<std::convert::Infallible>> {
        validate_input(time, position, velocity, step, self.config)?;
        if position.len() != self.position.value.len() {
            return Err(GaussJacksonError::InvalidInput("restart dimension differs"));
        }
        self.time = time;
        self.step = (time + step) - time;
        self.position.value.copy_from_slice(position);
        self.velocity.value.copy_from_slice(velocity);
        self.history_len = 0;
        self.dense_valid = false;
        Ok(())
    }
    /// Advances one full fixed interval with a mutable fallible acceleration.
    pub fn try_step<E, F>(
        &mut self,
        acceleration: &mut F,
    ) -> Result<GaussJacksonStep<'_>, GaussJacksonError<E>>
    where
        F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>,
    {
        let end = self.time + self.step;
        if end == self.time {
            return Err(GaussJacksonError::InvalidInput("step cannot advance time"));
        }
        self.try_step_to(end, acceleration)
    }
    /// Advances at most one interval toward `end`, landing exactly on short ends.
    /// A short interval uses high-order startup and clears the fixed-grid history.
    /// A zero interval is a no-op with no force call. Opposite direction fails.
    pub fn try_step_to<E, F>(
        &mut self,
        end: f64,
        acceleration: &mut F,
    ) -> Result<GaussJacksonStep<'_>, GaussJacksonError<E>>
    where
        F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>,
    {
        if !end.is_finite() {
            return Err(GaussJacksonError::InvalidInput("endpoint must be finite"));
        }
        let remaining = end - self.time;
        if remaining == 0. {
            return Ok(GaussJacksonStep {
                time: self.time,
                position: &self.position.value,
                velocity: &self.velocity.value,
                startup: false,
            });
        }
        if remaining.signum() != self.step.signum() {
            return Err(GaussJacksonError::InvalidInput(
                "endpoint opposes step direction",
            ));
        }
        let grid_end = self.time + self.step;
        let representable_step = grid_end - self.time;
        // Ordinary addition jitter below 1024 ulps of h is roundoff, not a
        // user step-size change. Every numerical update still uses actual dt.
        let grid_changed =
            (representable_step - self.step).abs() > 1024. * f64::EPSILON * self.step.abs();
        let on_grid = end == grid_end;
        let partial = remaining.abs() < representable_step.abs() && !on_grid;
        let h = if partial {
            remaining
        } else {
            representable_step
        };
        let new_time = if partial || on_grid {
            end
        } else {
            self.time + h
        };
        if new_time == self.time || !new_time.is_finite() {
            return Err(GaussJacksonError::InvalidInput(
                "step cannot advance finite time",
            ));
        }
        let n = self.position.value.len();
        if self.history_len == 0 {
            eval(
                acceleration,
                self.time,
                &self.position.value,
                &self.velocity.value,
                &mut self.acceleration.initial,
                &mut self.statistics,
            )?;
        } else {
            self.acceleration
                .initial
                .copy_from_slice(&self.history[(self.history_len - 1) * n..self.history_len * n]);
        }
        let startup = partial || grid_changed || self.history_len < 9;
        if startup {
            self.bootstrap(h, new_time, acceleration)?;
        } else {
            self.correct(h, new_time, acceleration)?;
        }
        // The endpoint force always corresponds to the final accepted state.
        // correct/bootstrap evaluated it even on the converged iteration.
        self.dense_q0.copy_from_slice(&self.position.value);
        self.dense_v0.copy_from_slice(&self.velocity.value);
        self.acceleration
            .dense_start
            .copy_from_slice(&self.acceleration.initial);
        self.acceleration
            .dense_end
            .copy_from_slice(&self.acceleration.candidate);
        self.dense_start = self.time;
        self.dense_valid = true;
        self.position
            .value
            .copy_from_slice(&self.position.candidate);
        self.velocity
            .value
            .copy_from_slice(&self.velocity.candidate);
        self.time = new_time;
        if partial {
            self.history_len = 0;
        } else if startup {
            if grid_changed {
                self.history_len = 0;
                self.step = h;
            }
            if self.history_len == 0 {
                self.history[..n].copy_from_slice(&self.acceleration.initial);
                self.history_len = 1;
            }
            self.history[self.history_len * n..(self.history_len + 1) * n]
                .copy_from_slice(&self.acceleration.candidate);
            self.history_len += 1;
            if self.history_len == 5 {
                self.position.center.copy_from_slice(&self.position.value);
                self.velocity.center.copy_from_slice(&self.velocity.value);
            }
            if self.history_len == 9 {
                self.initialize_sums();
            }
        } else {
            self.history.copy_within(n..9 * n, 0);
            self.history[8 * n..9 * n].copy_from_slice(&self.acceleration.candidate);
            for i in 0..n {
                self.double_sum[i] += h * self.sum[i];
                self.sum[i] += h * self.acceleration.candidate[i];
            }
        }
        self.statistics.accepted_steps += 1;
        if startup {
            self.statistics.startup_steps += 1;
        }
        Ok(GaussJacksonStep {
            time: self.time,
            position: &self.position.value,
            velocity: &self.velocity.value,
            startup,
        })
    }
    fn initialize_sums(&mut self) {
        let n = self.position.value.len();
        let h = self.step;
        for i in 0..n {
            let mut b = 0.;
            let mut a = 0.;
            for k in 0..9 {
                b += tableau().velocity().center()[k] * self.history[k * n + i];
                a += tableau().position().center()[k] * self.history[k * n + i];
            }
            self.sum[i] = self.velocity.center[i] - h * b;
            self.double_sum[i] = self.position.center[i] - h * (h * a);
            for k in 5..9 {
                self.double_sum[i] += h * self.sum[i];
                self.sum[i] += h * self.history[k * n + i];
            }
        }
    }
    fn correct<E, F>(
        &mut self,
        h: f64,
        t: f64,
        acceleration: &mut F,
    ) -> Result<(), GaussJacksonError<E>>
    where
        F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>,
    {
        let n = self.position.value.len();
        for i in 0..n {
            let mut a = 0.;
            let mut b = 0.;
            let mut ac = 0.;
            let mut bc = 0.;
            for k in 0..9 {
                a += tableau().position().predictor()[k] * self.history[k * n + i];
                b += tableau().velocity().predictor()[k] * self.history[k * n + i];
            }
            for k in 0..8 {
                ac += tableau().position().corrector()[k] * self.history[(k + 1) * n + i];
                bc += tableau().velocity().corrector()[k] * self.history[(k + 1) * n + i];
            }
            self.position.candidate[i] = self.double_sum[i] + h * self.sum[i] + h * (h * a);
            self.velocity.candidate[i] = self.sum[i] + h * b;
            self.position.fixed[i] = self.double_sum[i] + h * self.sum[i] + h * (h * ac);
            self.velocity.fixed[i] = self.sum[i] + h * bc;
        }
        eval(
            acceleration,
            t,
            &self.position.candidate,
            &self.velocity.candidate,
            &mut self.acceleration.candidate,
            &mut self.statistics,
        )?;
        for _ in 0..self.config.max_corrector_iterations {
            let mut converged = true;
            for i in 0..n {
                let q = self.position.fixed[i]
                    + h * (h
                        * (tableau().position().corrector()[8] * self.acceleration.candidate[i]));
                let v = self.velocity.fixed[i]
                    + h * (1. + tableau().velocity().corrector()[8])
                        * self.acceleration.candidate[i];
                converged &= close(q, self.position.candidate[i], self.config)
                    && close(v, self.velocity.candidate[i], self.config);
                self.position.candidate[i] = q;
                self.velocity.candidate[i] = v;
            }
            eval(
                acceleration,
                t,
                &self.position.candidate,
                &self.velocity.candidate,
                &mut self.acceleration.next,
                &mut self.statistics,
            )?;
            self.statistics.corrector_iterations += 1;
            self.acceleration
                .candidate
                .copy_from_slice(&self.acceleration.next);
            if converged {
                return Ok(());
            }
        }
        Err(GaussJacksonError::CorrectorConvergence {
            iterations: self.config.max_corrector_iterations,
        })
    }
    fn bootstrap<E, F>(
        &mut self,
        h: f64,
        t: f64,
        acceleration: &mut F,
    ) -> Result<(), GaussJacksonError<E>>
    where
        F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>,
    {
        let n = self.position.value.len();
        let width = 2 * n;
        for refinement in 0..=self.config.max_startup_refinements {
            let factor = 1usize << refinement;
            for level in 0..5 {
                let subdivisions = 2 * (level + 1) * factor;
                let dt = h / subdivisions as f64;
                self.previous[..n].copy_from_slice(&self.position.value);
                self.previous[n..].copy_from_slice(&self.velocity.value);
                for i in 0..n {
                    self.current[i] = self.position.value[i] + dt * self.velocity.value[i];
                    self.current[n + i] =
                        self.velocity.value[i] + dt * self.acceleration.initial[i];
                }
                for sub in 1..subdivisions {
                    eval(
                        acceleration,
                        self.time + dt * sub as f64,
                        &self.current[..n],
                        &self.current[n..],
                        &mut self.acceleration.next,
                        &mut self.statistics,
                    )?;
                    for i in 0..n {
                        self.next[i] = self.previous[i] + 2. * dt * self.current[n + i];
                        self.next[n + i] =
                            self.previous[n + i] + 2. * dt * self.acceleration.next[i];
                    }
                    std::mem::swap(&mut self.previous, &mut self.current);
                    std::mem::swap(&mut self.current, &mut self.next);
                }
                eval(
                    acceleration,
                    t,
                    &self.current[..n],
                    &self.current[n..],
                    &mut self.acceleration.next,
                    &mut self.statistics,
                )?;
                for i in 0..n {
                    self.extrapolation[level * width + i] =
                        0.5 * (self.current[i] + self.previous[i] + dt * self.current[n + i]);
                    self.extrapolation[level * width + n + i] = 0.5
                        * (self.current[n + i]
                            + self.previous[n + i]
                            + dt * self.acceleration.next[i]);
                }
                for prior in (0..level).rev() {
                    let ratio = ((level + 1) as f64 / (prior + 1) as f64).powi(2) - 1.;
                    for i in 0..width {
                        let finer = self.extrapolation[(prior + 1) * width + i];
                        let coarse = self.extrapolation[prior * width + i];
                        self.extrapolation[prior * width + i] = finer + (finer - coarse) / ratio;
                    }
                }
            }
            let converged = (0..width).all(|i| {
                close(
                    self.extrapolation[i],
                    self.extrapolation[width + i],
                    self.config,
                )
            });
            if converged {
                self.position
                    .candidate
                    .copy_from_slice(&self.extrapolation[..n]);
                self.velocity
                    .candidate
                    .copy_from_slice(&self.extrapolation[n..width]);
                eval(
                    acceleration,
                    t,
                    &self.position.candidate,
                    &self.velocity.candidate,
                    &mut self.acceleration.candidate,
                    &mut self.statistics,
                )?;
                return Ok(());
            }
            if refinement < self.config.max_startup_refinements {
                self.statistics.startup_refinements += 1;
            }
        }
        Err(GaussJacksonError::StartupConvergence {
            refinements: self.config.max_startup_refinements,
        })
    }
    /// Interpolates the last accepted interval using quintic Hermite position.
    /// Velocity is its analytic derivative. This is fifth-degree dense output,
    /// not eighth-order Gauss–Jackson endpoint accuracy. Caller buffers are reused.
    pub fn interpolate_into(
        &self,
        time: f64,
        position: &mut [f64],
        velocity: &mut [f64],
    ) -> Result<(), GaussJacksonError<std::convert::Infallible>> {
        let n = self.position.value.len();
        if !self.dense_valid {
            return Err(GaussJacksonError::InvalidInput(
                "no accepted dense interval",
            ));
        }
        if position.len() != n || velocity.len() != n {
            return Err(GaussJacksonError::InvalidInput(
                "dense output dimension differs",
            ));
        }
        let h = self.time - self.dense_start;
        let x = (time - self.dense_start) / h;
        if !time.is_finite() || !(0.0..=1.0).contains(&x) {
            return Err(GaussJacksonError::InvalidInput(
                "dense query outside last interval",
            ));
        }
        if x == 0. {
            position.copy_from_slice(&self.dense_q0);
            velocity.copy_from_slice(&self.dense_v0);
            return Ok(());
        }
        if x == 1. {
            position.copy_from_slice(&self.position.value);
            velocity.copy_from_slice(&self.velocity.value);
            return Ok(());
        }
        for i in 0..n {
            let c = self.dense_coefficients(i, h);
            position[i] = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * c[5]))));
            velocity[i] =
                (c[1] + x * (2. * c[2] + x * (3. * c[3] + x * (4. * c[4] + x * 5. * c[5])))) / h;
            if !position[i].is_finite() || !velocity[i].is_finite() {
                return Err(GaussJacksonError::NonFinite);
            }
        }
        Ok(())
    }
    /// Writes degree-major position polynomial coefficients (6 × dimension).
    /// Normalized variable is `(t - last_start) / (time - last_start)`.
    pub fn dense_coefficients_into(
        &self,
        output: &mut [f64],
    ) -> Result<(f64, f64), GaussJacksonError<std::convert::Infallible>> {
        let n = self.position.value.len();
        if !self.dense_valid || output.len() != 6 * n {
            return Err(GaussJacksonError::InvalidInput(
                "dense coefficient buffer or interval unavailable",
            ));
        }
        let h = self.time - self.dense_start;
        for i in 0..n {
            let c = self.dense_coefficients(i, h);
            if c.iter().any(|value| !value.is_finite()) {
                return Err(GaussJacksonError::NonFinite);
            }
            for degree in 0..6 {
                output[degree * n + i] = c[degree];
            }
        }
        Ok((self.dense_start, self.time))
    }
    /// Polynomial degree of the retained Hermite position interpolation.
    ///
    /// Position is quintic and velocity its quartic derivative. This reports
    /// representation degree, not the eighth-order endpoint method's accuracy.
    pub const fn dense_polynomial_degree(&self) -> usize {
        5
    }

    /// Exports the last accepted interval as a portable owned `[position, velocity]`
    /// segment. Export allocates; ordinary stepping/interpolation does not.
    ///
    /// Quality is `MethodSpecific`: this is the documented quintic Hermite
    /// extension, not an eighth-order continuous extension. Coefficients are
    /// degree-major, with all positions followed by all velocities per degree.
    pub fn export_dense_segment(
        &self,
    ) -> Result<crate::PortableDenseSegment, crate::InterpolationError> {
        if !self.dense_valid {
            return Err(crate::InterpolationError::InvalidSegmentData {
                context: "Gauss-Jackson has no accepted dense interval",
            });
        }
        let n = self.position.value.len();
        let dimension = 2 * n;
        let h = self.time - self.dense_start;
        let mut coefficients = vec![0.0; 6 * dimension];
        for i in 0..n {
            let c = self.dense_coefficients(i, h);
            for degree in 0..6 {
                coefficients[degree * dimension + i] = c[degree];
                if degree < 5 {
                    coefficients[degree * dimension + n + i] =
                        (degree + 1) as f64 * c[degree + 1] / h;
                }
            }
            coefficients[n + i] = self.dense_v0[i];
        }
        let mut end_state = Vec::with_capacity(dimension);
        end_state.extend_from_slice(&self.position.value);
        end_state.extend_from_slice(&self.velocity.value);
        crate::PortableDenseSegment::from_data(crate::DenseSegmentData {
            version: 1,
            start_time: self.dense_start,
            end_time: self.time,
            bound_time: self.time,
            dimension,
            coefficients,
            end_state,
            bound_state: None,
            quality: crate::InterpolationQuality::MethodSpecific,
        })
    }

    fn dense_coefficients(&self, i: usize, h: f64) -> [f64; 6] {
        let q0 = self.dense_q0[i];
        let v0 = h * self.dense_v0[i];
        let a0 = h * (h * self.acceleration.dense_start[i]);
        let dq = self.position.value[i] - q0;
        let v1 = h * self.velocity.value[i];
        let a1 = h * (h * self.acceleration.dense_end[i]);
        [
            q0,
            v0,
            a0 / 2.,
            10. * dq - 6. * v0 - 4. * v1 - 1.5 * a0 + 0.5 * a1,
            -15. * dq + 8. * v0 + 7. * v1 + 1.5 * a0 - a1,
            6. * dq - 3. * v0 - 3. * v1 - 0.5 * a0 + 0.5 * a1,
        ]
    }
}
fn close(a: f64, b: f64, c: GaussJacksonConfig) -> bool {
    a.is_finite()
        && b.is_finite()
        && (a - b).abs() <= c.absolute_tolerance + c.relative_tolerance * a.abs().max(b.abs())
}
fn validate_input<E>(
    t: f64,
    q: &[f64],
    v: &[f64],
    h: f64,
    c: GaussJacksonConfig,
) -> Result<(), GaussJacksonError<E>> {
    if q.is_empty() || q.len() != v.len() || q.len() > usize::MAX / 18 {
        return Err(GaussJacksonError::InvalidInput(
            "nonzero equal state dimensions required",
        ));
    }
    if !t.is_finite()
        || !h.is_finite()
        || h == 0.
        || !(t + h).is_finite()
        || t + h == t
        || !q.iter().chain(v).all(|x| x.is_finite())
    {
        return Err(GaussJacksonError::InvalidInput(
            "finite state/time and nonzero finite step required",
        ));
    }
    if !c.absolute_tolerance.is_finite()
        || !c.relative_tolerance.is_finite()
        || c.absolute_tolerance < 0.
        || c.relative_tolerance < 0.
        || c.absolute_tolerance + c.relative_tolerance == 0.
        || c.max_corrector_iterations == 0
        || c.max_startup_refinements > 20
    {
        return Err(GaussJacksonError::InvalidInput(
            "invalid iteration settings",
        ));
    }
    Ok(())
}
fn eval<E, F>(
    f: &mut F,
    t: f64,
    q: &[f64],
    v: &[f64],
    a: &mut [f64],
    stats: &mut GaussJacksonStatistics,
) -> Result<(), GaussJacksonError<E>>
where
    F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>,
{
    if !q.iter().chain(v).all(|x| x.is_finite()) {
        return Err(GaussJacksonError::NonFinite);
    }
    a.fill(f64::NAN);
    stats.acceleration_evaluations += 1;
    f(t, q, v, a).map_err(GaussJacksonError::Acceleration)?;
    if !a.iter().all(|x| x.is_finite()) {
        return Err(GaussJacksonError::NonFinite);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    fn harmonic(_: f64, q: &[f64], _: &[f64], a: &mut [f64]) -> Result<(), Infallible> {
        for (a, q) in a.iter_mut().zip(q) {
            *a = -q;
        }
        Ok(())
    }

    fn propagate(h: f64, end: f64) -> GaussJackson8 {
        let mut solver =
            GaussJackson8::new(0., &[1.], &[0.], h, GaussJacksonConfig::default()).unwrap();
        while (end - solver.time()) * h > 0. {
            solver.try_step_to(end, &mut harmonic).unwrap();
        }
        solver
    }

    #[test]
    fn harmonic_order_and_long_arc() {
        let mut errors = Vec::new();
        for h in [0.4, 0.2, 0.1] {
            let solver = propagate(h, 20.);
            let err = (solver.position()[0] - 20f64.cos())
                .abs()
                .max((solver.velocity()[0] + 20f64.sin()).abs());
            eprintln!("h={h}: error={err:.16e}, stats={:?}", solver.statistics());
            errors.push(err);
            assert!(solver.statistics().accepted_steps > solver.statistics().startup_steps);
        }
        assert!(errors[0] / errors[1] > 150., "errors={errors:?}");
        assert!(errors[1] < 1e-7);
        let solver = propagate(0.05, 200.);
        assert!((solver.position()[0] - 200f64.cos()).abs() < 1e-9);
    }

    #[test]
    fn velocity_dependent_forces_backward_and_partial_end() {
        for h in [0.1, -0.1] {
            let end = if h > 0. { 3.037 } else { -3.037 };
            let mut solver =
                GaussJackson8::new(0., &[1.], &[-1.], h, GaussJacksonConfig::default()).unwrap();
            let mut f = |_: f64, q: &[f64], v: &[f64], a: &mut [f64]| {
                a[0] = -q[0] - 2. * v[0];
                Ok::<_, Infallible>(())
            };
            while (end - solver.time()) * h > 0. {
                solver.try_step_to(end, &mut f).unwrap();
            }
            assert_eq!(solver.time(), end);
            assert!((solver.position()[0] - (-end).exp()).abs() < 2e-9);
            assert!((solver.velocity()[0] + (-end).exp()).abs() < 2e-9);
            assert_eq!(solver.history_len(), 0);
        }
    }

    #[test]
    fn startup_short_intervals_and_domain_are_accurate() {
        let mut solver = GaussJackson8::new(
            2.,
            &[2f64.cos()],
            &[-2f64.sin()],
            0.2,
            GaussJacksonConfig::default(),
        )
        .unwrap();
        let mut f = |t: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
            assert!((2.0..=2.03).contains(&t));
            a[0] = -q[0];
            Ok::<_, Infallible>(())
        };
        solver.try_step_to(2.03, &mut f).unwrap();
        assert!((solver.position()[0] - 2.03f64.cos()).abs() < 1e-14);
        assert!((solver.velocity()[0] + 2.03f64.sin()).abs() < 1e-14);
        assert_eq!(solver.statistics().startup_steps, 1);
    }

    #[test]
    fn continuation_clone_and_restart() {
        let mut original = propagate(0.1, 2.);
        let mut clone = original.clone();
        for _ in 0..10 {
            original.try_step(&mut harmonic).unwrap();
            clone.try_step(&mut harmonic).unwrap();
        }
        assert_eq!(original.position(), clone.position());
        assert_eq!(original.statistics(), clone.statistics());
        clone.restart(0., &[2.], &[0.], 0.1).unwrap();
        assert_eq!(clone.history_len(), 0);
        clone.try_step(&mut harmonic).unwrap();
        assert!((clone.position()[0] - 2. * 0.1f64.cos()).abs() < 1e-13);
    }

    #[test]
    fn errors_preserve_accepted_state_and_payload() {
        let mut solver = propagate(0.1, 2.);
        let before = solver.clone();
        let mut bad = |_: f64, _: &[f64], _: &[f64], _: &mut [f64]| Err::<(), _>(1234);
        assert!(matches!(
            solver.try_step(&mut bad),
            Err(GaussJacksonError::Acceleration(1234))
        ));
        assert_eq!(solver.time(), before.time());
        assert_eq!(solver.position(), before.position());
        assert_eq!(solver.history, before.history);
        solver.try_step(&mut harmonic).unwrap();
        let mut expected = before;
        expected.try_step(&mut harmonic).unwrap();
        assert_eq!(solver.position(), expected.position());
    }

    #[test]
    fn dense_quintic_has_exact_endpoints_and_consistent_velocity() {
        let solver = propagate(0.1, 2.);
        let mut q = [0.];
        let mut v = [0.];
        solver
            .interpolate_into(solver.time(), &mut q, &mut v)
            .unwrap();
        assert_eq!(q.as_slice(), solver.position());
        assert_eq!(v.as_slice(), solver.velocity());
        let mid = (solver.dense_start + solver.time()) / 2.;
        solver.interpolate_into(mid, &mut q, &mut v).unwrap();
        assert!((q[0] - mid.cos()).abs() < 1e-9);
        assert!((v[0] + mid.sin()).abs() < 1e-8);
    }

    #[test]
    fn validates_zero_steps_invalid_config_and_noop() {
        assert!(GaussJackson8::new(0., &[1.], &[0.], 0., GaussJacksonConfig::default()).is_err());
        let mut solver =
            GaussJackson8::new(0., &[1.], &[0.], 0.1, GaussJacksonConfig::default()).unwrap();
        solver.try_step_to(0., &mut harmonic).unwrap();
        assert_eq!(solver.statistics().acceleration_evaluations, 0);
        assert!(solver.try_step_to(-1., &mut harmonic).is_err());
    }

    #[test]
    fn corrector_and_startup_failures_are_explicit_and_transactional() {
        let mut solver =
            GaussJackson8::new(0., &[1.], &[0.], 0.2, GaussJacksonConfig::default()).unwrap();
        for _ in 0..8 {
            solver.try_step(&mut harmonic).unwrap();
        }
        let before = solver.clone();
        solver.config.max_corrector_iterations = 1;
        assert!(matches!(
            solver.try_step(&mut harmonic),
            Err(GaussJacksonError::CorrectorConvergence { iterations: 1 })
        ));
        assert_eq!(solver.position.value, before.position.value);
        assert_eq!(solver.velocity.value, before.velocity.value);
        assert_eq!(solver.history, before.history);
        assert_eq!(solver.sum, before.sum);
        assert_eq!(solver.double_sum, before.double_sum);
        let mut q = [0.];
        let mut v = [0.];
        let mut expected_q = [0.];
        let mut expected_v = [0.];
        solver.interpolate_into(1.55, &mut q, &mut v).unwrap();
        before
            .interpolate_into(1.55, &mut expected_q, &mut expected_v)
            .unwrap();
        assert_eq!(q, expected_q);
        assert_eq!(v, expected_v);
        let config = GaussJacksonConfig {
            absolute_tolerance: 1e-30,
            relative_tolerance: 0.,
            max_startup_refinements: 0,
            ..Default::default()
        };
        let mut startup = GaussJackson8::new(0., &[1.], &[0.], 1., config).unwrap();
        assert!(matches!(
            startup.try_step(&mut harmonic),
            Err(GaussJacksonError::StartupConvergence { refinements: 0 })
        ));
        assert_eq!(startup.time(), 0.);
        assert_eq!(startup.history_len(), 0);
        assert_eq!(startup.position(), &[1.]);
    }

    #[test]
    fn polynomial_acceleration_and_scaled_units() {
        for h in [0.05, -0.05] {
            let end = if h > 0. { 2. } else { -2. };
            let mut solver =
                GaussJackson8::new(0., &[2.], &[3.], h, GaussJacksonConfig::default()).unwrap();
            let mut f = |t: f64, _: &[f64], _: &[f64], a: &mut [f64]| {
                a[0] = 56. * t.powi(6);
                Ok::<_, Infallible>(())
            };
            while (end - solver.time()) * h > 0. {
                solver.try_step_to(end, &mut f).unwrap();
            }
            assert!((solver.position()[0] - (end.powi(8) + 3. * end + 2.)).abs() < 1e-9);
            assert!((solver.velocity()[0] - (8. * end.powi(7) + 3.)).abs() < 1e-9);
        }
    }

    #[test]
    fn two_body_circular_orbit_and_energy_long_arc() {
        let mut solver = GaussJackson8::new(
            0.,
            &[1., 0.],
            &[0., 1.],
            0.04,
            GaussJacksonConfig::default(),
        )
        .unwrap();
        let mut gravity = |_: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
            let r = q[0].hypot(q[1]);
            a[0] = -q[0] / r.powi(3);
            a[1] = -q[1] / r.powi(3);
            Ok::<_, Infallible>(())
        };
        let end = 100.;
        while solver.time() < end {
            solver.try_step_to(end, &mut gravity).unwrap();
        }
        assert!((solver.position()[0] - end.cos()).abs() < 1e-8);
        assert!((solver.position()[1] - end.sin()).abs() < 1e-8);
        let q = solver.position();
        let v = solver.velocity();
        let energy = 0.5 * (v[0] * v[0] + v[1] * v[1]) - 1. / q[0].hypot(q[1]);
        assert!((energy + 0.5).abs() < 1e-10);
    }

    #[test]
    fn dense_coefficients_round_trip_and_failed_force_preserves_dense() {
        let mut solver = propagate(0.1, 2.);
        let mut coefficients = [0.; 6];
        let (start, end) = solver.dense_coefficients_into(&mut coefficients).unwrap();
        let time = (start + end) / 2.;
        let mut q = [0.];
        let mut v = [0.];
        solver.interpolate_into(time, &mut q, &mut v).unwrap();
        let poly = coefficients.iter().rev().fold(0., |a, c| a * 0.5 + c);
        assert!((poly - q[0]).abs() < 1e-15);
        let mut calls = 0;
        let mut broken = |_: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
            calls += 1;
            if calls == 2 {
                Err(42)
            } else {
                a[0] = -q[0];
                Ok(())
            }
        };
        assert!(solver.try_step(&mut broken).is_err());
        let mut after = [0.; 6];
        solver.dense_coefficients_into(&mut after).unwrap();
        assert_eq!(coefficients, after);
    }

    #[test]
    fn agrees_with_independent_pinned_gj8_reference() {
        let solver = propagate(0.1, 20.);
        assert!((solver.position()[0] - 0.4080820618135249).abs() < 1e-12);
        assert!((solver.velocity()[0] - (-0.9129452507272791)).abs() < 1e-12);
        let mut damped =
            GaussJackson8::new(0., &[1.], &[-1.], 0.1, GaussJacksonConfig::default()).unwrap();
        let mut f = |_: f64, q: &[f64], v: &[f64], a: &mut [f64]| {
            a[0] = -q[0] - 2. * v[0];
            Ok::<_, Infallible>(())
        };
        while damped.time() < 3. {
            damped.try_step_to(3., &mut f).unwrap();
        }
        assert!((damped.position()[0] - 0.04978706836819483).abs() < 1e-12);
        assert!((damped.velocity()[0] - (-0.0497870683681613)).abs() < 1e-12);
    }

    #[test]
    fn startup_extrapolation_has_high_order_before_refinement() {
        let config = GaussJacksonConfig {
            absolute_tolerance: 1.,
            relative_tolerance: 0.,
            max_startup_refinements: 0,
            ..Default::default()
        };
        let mut errors = Vec::new();
        for h in [1.5, 0.75] {
            let mut solver = GaussJackson8::new(0., &[1.], &[0.], h, config).unwrap();
            solver.try_step(&mut harmonic).unwrap();
            errors.push(
                (solver.position()[0] - h.cos())
                    .abs()
                    .max((solver.velocity()[0] + h.sin()).abs()),
            );
            assert_eq!(solver.statistics().startup_refinements, 0);
        }
        assert!(errors[0] / errors[1] > 900., "startup errors={errors:?}");
    }

    #[test]
    fn eccentric_two_body_matches_kepler_reference() {
        let mut errors = Vec::new();
        for h in [0.02, 0.01] {
            let e = 0.6_f64;
            let mut solver = GaussJackson8::new(
                0.,
                &[1. - e, 0.],
                &[0., ((1. + e) / (1. - e)).sqrt()],
                h,
                GaussJacksonConfig::default(),
            )
            .unwrap();
            let mut gravity = |_: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
                let r = q[0].hypot(q[1]);
                for i in 0..2 {
                    a[i] = -q[i] / r.powi(3);
                }
                Ok::<_, Infallible>(())
            };
            let end = 30.;
            while solver.time() < end {
                solver.try_step_to(end, &mut gravity).unwrap();
            }
            let mean = end.rem_euclid(std::f64::consts::TAU);
            let mut eccentric = mean;
            for _ in 0..12 {
                eccentric -= (eccentric - e * eccentric.sin() - mean) / (1. - e * eccentric.cos());
            }
            let q = [eccentric.cos() - e, (1. - e * e).sqrt() * eccentric.sin()];
            let denominator = 1. - e * eccentric.cos();
            let v = [
                -eccentric.sin() / denominator,
                (1. - e * e).sqrt() * eccentric.cos() / denominator,
            ];
            let error = (0..2)
                .map(|i| {
                    (solver.position()[i] - q[i])
                        .abs()
                        .max((solver.velocity()[i] - v[i]).abs())
                })
                .fold(0., f64::max);
            assert!(error < 1e-7, "h={h}, Kepler error={error}");
            errors.push(error);
        }
        assert!(errors[0] / errors[1] > 150., "eccentric errors={errors:?}");
    }

    #[test]
    fn large_epoch_steps_match_accepted_time_and_restart_across_binades() {
        for h in [0.1, -0.1] {
            for start in [1e12, 2f64.powi(40) - h * 12.] {
                let mut solver =
                    GaussJackson8::new(start, &[0.], &[1.], h, GaussJacksonConfig::default())
                        .unwrap();
                assert_eq!(solver.step_size(), (start + h) - start);
                let mut force = |_: f64, _: &[f64], _: &[f64], a: &mut [f64]| {
                    a[0] = 0.;
                    Ok::<_, Infallible>(())
                };
                for _ in 0..40 {
                    solver.try_step(&mut force).unwrap();
                    assert!(
                        (solver.position()[0] - (solver.time() - start)).abs() < 2e-13,
                        "time={}, position={}",
                        solver.time(),
                        solver.position()[0]
                    );
                    assert!((solver.velocity()[0] - 1.).abs() < 2e-14);
                }
                let end = solver.time() + 0.037 * h.signum();
                solver.try_step_to(end, &mut force).unwrap();
                assert_eq!(solver.time(), end);
                assert!((solver.position()[0] - (end - start)).abs() < 2e-13);
            }
        }
        assert!(
            GaussJackson8::new(1e12, &[0.], &[1.], 1e-10, GaussJacksonConfig::default()).is_err()
        );
    }

    #[test]
    fn extreme_step_dense_output_remains_finite() {
        let mut solver =
            GaussJackson8::new(0., &[1.], &[0.], 1e200, GaussJacksonConfig::default()).unwrap();
        let mut force = |_: f64, _: &[f64], _: &[f64], a: &mut [f64]| {
            a[0] = 0.;
            Ok::<_, Infallible>(())
        };
        for _ in 0..12 {
            let start = solver.time();
            solver.try_step(&mut force).unwrap();
            let mut q = [0.];
            let mut v = [0.];
            solver
                .interpolate_into(start + (solver.time() - start) / 2., &mut q, &mut v)
                .unwrap();
            assert_eq!(q, [1.]);
            assert_eq!(v, [0.]);
            let mut coefficients = [0.; 6];
            solver.dense_coefficients_into(&mut coefficients).unwrap();
            assert!(coefficients.iter().all(|value| value.is_finite()));
            solver.export_dense_segment().unwrap();
        }
        solver.position.value[0] = 1e308;
        assert!(matches!(
            solver.dense_coefficients_into(&mut [0.; 6]),
            Err(GaussJacksonError::NonFinite)
        ));
        assert!(matches!(
            solver.interpolate_into(
                solver.dense_start + (solver.time - solver.dense_start) / 2.,
                &mut [0.],
                &mut [0.]
            ),
            Err(GaussJacksonError::NonFinite)
        ));
    }

    #[test]
    fn full_step_rejects_later_stationary_time() {
        let mut solver = GaussJackson8::new(
            2f64.powi(53) - 1.,
            &[1.],
            &[0.],
            1.,
            GaussJacksonConfig::default(),
        )
        .unwrap();
        let mut force = |_: f64, _: &[f64], _: &[f64], a: &mut [f64]| {
            a[0] = 0.;
            Ok::<_, Infallible>(())
        };
        solver.try_step(&mut force).unwrap();
        let calls = solver.statistics().acceleration_evaluations;
        assert!(solver.try_step(&mut force).is_err());
        assert_eq!(solver.statistics().acceleration_evaluations, calls);
        solver.try_step_to(solver.time(), &mut force).unwrap();
    }
}
rejects_later_stationary_time() {
        let mut solver = GaussJackson8::new(
            2f64.powi(53) - 1.,
            &[1.],
            &[0.],
            1.,
            GaussJacksonConfig::default(),
        )
        .unwrap();
        let mut force = |_: f64, _: &[f64], _: &[f64], a: &mut [f64]| {
            a[0] = 0.;
            Ok::<_, Infallible>(())
        };
        solver.try_step(&mut force).unwrap();
        let calls = solver.statistics().acceleration_evaluations;
        assert!(solver.try_step(&mut force).is_err());
        assert_eq!(solver.statistics().acceleration_evaluations, calls);
        solver.try_step_to(solver.time(), &mut force).unwrap();
    }
}
