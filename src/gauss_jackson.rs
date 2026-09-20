//! Fixed-step eighth-order Gauss–Jackson integration of `q'' = a(t, q, v)`.
//!
//! The steady-state method is the summed Adams / double-sum Gauss–Jackson
//! predictor-corrector of Berry and Healy (2004), with nine acceleration samples.
//! Startup and a final partial interval use tenth-order extrapolated midpoint.
//! This avoids evaluating a force outside the requested integration domain.
//! Startup is explicitly counted, and partial intervals invalidate history.
//! Restart after any force/state discontinuity; never reuse history across one.

#[path = "gauss_jackson/coefficients.rs"]
mod coefficients;
use coefficients::*;
use std::{error::Error, fmt};

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
    q: Vec<f64>,
    v: Vec<f64>,
    history: Vec<f64>,
    history_len: usize,
    sum: Vec<f64>,
    double_sum: Vec<f64>,
    cq: Vec<f64>,
    cv: Vec<f64>,
    ca: Vec<f64>,
    next_a: Vec<f64>,
    initial_a: Vec<f64>,
    fixed_q: Vec<f64>,
    fixed_v: Vec<f64>,
    previous: Vec<f64>,
    current: Vec<f64>,
    next: Vec<f64>,
    extrapolation: Vec<f64>,
    dense_q0: Vec<f64>,
    dense_v0: Vec<f64>,
    dense_a0: Vec<f64>,
    dense_a1: Vec<f64>,
    dense_start: f64,
    dense_valid: bool,
    statistics: GaussJacksonStatistics,
}
impl GaussJackson8 {
    /// Allocates workspace and validates a signed nonzero fixed step.
    pub fn new(
        time: f64,
        position: &[f64],
        velocity: &[f64],
        step: f64,
        config: GaussJacksonConfig,
    ) -> Result<Self, GaussJacksonError<std::convert::Infallible>> {
        validate_input(time, position, velocity, step, config)?;
        let n = position.len();
        Ok(Self {
            time,
            step,
            config,
            q: position.to_vec(),
            v: velocity.to_vec(),
            history: vec![0.; 9 * n],
            history_len: 0,
            sum: vec![0.; n],
            double_sum: vec![0.; n],
            cq: vec![0.; n],
            cv: vec![0.; n],
            ca: vec![0.; n],
            next_a: vec![0.; n],
            initial_a: vec![0.; n],
            fixed_q: vec![0.; n],
            fixed_v: vec![0.; n],
            previous: vec![0.; 2 * n],
            current: vec![0.; 2 * n],
            next: vec![0.; 2 * n],
            extrapolation: vec![0.; 10 * n],
            dense_q0: vec![0.; n],
            dense_v0: vec![0.; n],
            dense_a0: vec![0.; n],
            dense_a1: vec![0.; n],
            dense_start: time,
            dense_valid: false,
            statistics: GaussJacksonStatistics::default(),
        })
    }
    /// Current accepted time.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Borrowed accepted position, without conversion.
    pub fn position(&self) -> &[f64] {
        &self.q
    }
    /// Borrowed accepted velocity, without conversion.
    pub fn velocity(&self) -> &[f64] {
        &self.v
    }
    /// Signed fixed step used by the history.
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
        if position.len() != self.q.len() {
            return Err(GaussJacksonError::InvalidInput("restart dimension differs"));
        }
        self.time = time;
        self.step = step;
        self.q.copy_from_slice(position);
        self.v.copy_from_slice(velocity);
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
        self.try_step_to(self.time + self.step, acceleration)
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
                position: &self.q,
                velocity: &self.v,
                startup: false,
            });
        }
        if remaining.signum() != self.step.signum() {
            return Err(GaussJacksonError::InvalidInput(
                "endpoint opposes step direction",
            ));
        }
        let on_grid = end == self.time + self.step;
        let partial = remaining.abs() < self.step.abs() && !on_grid;
        let h = if partial { remaining } else { self.step };
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
        let n = self.q.len();
        if self.history_len == 0 {
            eval(
                acceleration,
                self.time,
                &self.q,
                &self.v,
                &mut self.initial_a,
                &mut self.statistics,
            )?;
        } else {
            self.initial_a
                .copy_from_slice(&self.history[(self.history_len - 1) * n..self.history_len * n]);
        }
        let startup = partial || self.history_len < 9;
        if startup {
            self.bootstrap(h, new_time, acceleration)?;
        } else {
            self.correct(h, new_time, acceleration)?;
        }
        // The endpoint force always corresponds to the final accepted state.
        // correct/bootstrap evaluated it even on the converged iteration.
        self.dense_q0.copy_from_slice(&self.q);
        self.dense_v0.copy_from_slice(&self.v);
        self.dense_a0.copy_from_slice(&self.initial_a);
        self.dense_a1.copy_from_slice(&self.ca);
        self.dense_start = self.time;
        self.dense_valid = true;
        self.q.copy_from_slice(&self.cq);
        self.v.copy_from_slice(&self.cv);
        self.time = new_time;
        if partial {
            self.history_len = 0;
        } else if startup {
            if self.history_len == 0 {
                self.history[..n].copy_from_slice(&self.initial_a);
                self.history_len = 1;
            }
            self.history[self.history_len * n..(self.history_len + 1) * n]
                .copy_from_slice(&self.ca);
            self.history_len += 1;
            if self.history_len == 9 {
                self.initialize_sums();
            }
        } else {
            self.history.copy_within(n..9 * n, 0);
            self.history[8 * n..9 * n].copy_from_slice(&self.ca);
            for i in 0..n {
                self.double_sum[i] += h * self.sum[i];
                self.sum[i] += h * self.ca[i];
            }
        }
        self.statistics.accepted_steps += 1;
        if startup {
            self.statistics.startup_steps += 1;
        }
        Ok(GaussJacksonStep {
            time: self.time,
            position: &self.q,
            velocity: &self.v,
            startup,
        })
    }
    fn initialize_sums(&mut self) {
        let n = self.q.len();
        let h = self.step;
        for i in 0..n {
            let mut b = 0.;
            let mut a = 0.;
            for k in 0..9 {
                b += VELOCITY_CORRECTOR[k] * self.history[k * n + i];
                a += POSITION_CORRECTOR[k] * self.history[k * n + i];
            }
            self.sum[i] = self.v[i] - h * b;
            self.double_sum[i] = self.q[i] - h * h * a;
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
        let n = self.q.len();
        for i in 0..n {
            let mut a = 0.;
            let mut b = 0.;
            let mut ac = 0.;
            let mut bc = 0.;
            for k in 0..9 {
                a += POSITION_PREDICTOR[k] * self.history[k * n + i];
                b += VELOCITY_PREDICTOR[k] * self.history[k * n + i];
            }
            for k in 0..8 {
                ac += POSITION_CORRECTOR[k] * self.history[(k + 1) * n + i];
                bc += VELOCITY_CORRECTOR[k] * self.history[(k + 1) * n + i];
            }
            self.cq[i] = self.double_sum[i] + h * self.sum[i] + h * h * a;
            self.cv[i] = self.sum[i] + h * b;
            self.fixed_q[i] = self.double_sum[i] + h * self.sum[i] + h * h * ac;
            self.fixed_v[i] = self.sum[i] + h * bc;
        }
        eval(
            acceleration,
            t,
            &self.cq,
            &self.cv,
            &mut self.ca,
            &mut self.statistics,
        )?;
        for _ in 0..self.config.max_corrector_iterations {
            let mut converged = true;
            for i in 0..n {
                let q = self.fixed_q[i] + h * h * POSITION_CORRECTOR[8] * self.ca[i];
                let v = self.fixed_v[i] + h * (1. + VELOCITY_CORRECTOR[8]) * self.ca[i];
                converged &= close(q, self.cq[i], self.config) && close(v, self.cv[i], self.config);
                self.cq[i] = q;
                self.cv[i] = v;
            }
            eval(
                acceleration,
                t,
                &self.cq,
                &self.cv,
                &mut self.next_a,
                &mut self.statistics,
            )?;
            self.statistics.corrector_iterations += 1;
            self.ca.copy_from_slice(&self.next_a);
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
        let n = self.q.len();
        let width = 2 * n;
        for refinement in 0..=self.config.max_startup_refinements {
            let factor = 1usize << refinement;
            for level in 0..5 {
                let subdivisions = 2 * (level + 1) * factor;
                let dt = h / subdivisions as f64;
                self.previous[..n].copy_from_slice(&self.q);
                self.previous[n..].copy_from_slice(&self.v);
                for i in 0..n {
                    self.current[i] = self.q[i] + dt * self.v[i];
                    self.current[n + i] = self.v[i] + dt * self.initial_a[i];
                }
                for sub in 1..subdivisions {
                    eval(
                        acceleration,
                        self.time + dt * sub as f64,
                        &self.current[..n],
                        &self.current[n..],
                        &mut self.next_a,
                        &mut self.statistics,
                    )?;
                    for i in 0..n {
                        self.next[i] = self.previous[i] + 2. * dt * self.current[n + i];
                        self.next[n + i] = self.previous[n + i] + 2. * dt * self.next_a[i];
                    }
                    std::mem::swap(&mut self.previous, &mut self.current);
                    std::mem::swap(&mut self.current, &mut self.next);
                }
                eval(
                    acceleration,
                    t,
                    &self.current[..n],
                    &self.current[n..],
                    &mut self.next_a,
                    &mut self.statistics,
                )?;
                for i in 0..n {
                    self.extrapolation[level * width + i] =
                        0.5 * (self.current[i] + self.previous[i] + dt * self.current[n + i]);
                    self.extrapolation[level * width + n + i] =
                        0.5 * (self.current[n + i] + self.previous[n + i] + dt * self.next_a[i]);
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
                self.cq.copy_from_slice(&self.extrapolation[..n]);
                self.cv.copy_from_slice(&self.extrapolation[n..width]);
                eval(
                    acceleration,
                    t,
                    &self.cq,
                    &self.cv,
                    &mut self.ca,
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
        let n = self.q.len();
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
            position.copy_from_slice(&self.q);
            velocity.copy_from_slice(&self.v);
            return Ok(());
        }
        for i in 0..n {
            let c = self.dense_coefficients(i, h);
            position[i] = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * c[5]))));
            velocity[i] =
                (c[1] + x * (2. * c[2] + x * (3. * c[3] + x * (4. * c[4] + x * 5. * c[5])))) / h;
        }
        Ok(())
    }
    /// Writes degree-major position polynomial coefficients (6 × dimension).
    /// Normalized variable is `(t - last_start) / (time - last_start)`.
    pub fn dense_coefficients_into(
        &self,
        output: &mut [f64],
    ) -> Result<(f64, f64), GaussJacksonError<std::convert::Infallible>> {
        let n = self.q.len();
        if !self.dense_valid || output.len() != 6 * n {
            return Err(GaussJacksonError::InvalidInput(
                "dense coefficient buffer or interval unavailable",
            ));
        }
        let h = self.time - self.dense_start;
        for i in 0..n {
            let c = self.dense_coefficients(i, h);
            for degree in 0..6 {
                output[degree * n + i] = c[degree];
            }
        }
        Ok((self.dense_start, self.time))
    }
    fn dense_coefficients(&self, i: usize, h: f64) -> [f64; 6] {
        let q0 = self.dense_q0[i];
        let v0 = h * self.dense_v0[i];
        let a0 = h * h * self.dense_a0[i];
        let dq = self.q[i] - q0;
        let v1 = h * self.v[i];
        let a1 = h * h * self.dense_a1[i];
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
    if !t.is_finite() || !h.is_finite() || h == 0. || !q.iter().chain(v).all(|x| x.is_finite()) {
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
#[path = "gauss_jackson/tests.rs"]
mod tests;
