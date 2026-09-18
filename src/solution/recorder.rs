use super::{
    CollocationSegment, DenseSegment, HermiteSegment, InterpolationError, OwnedDenseSegment,
    RungeKuttaSegment, Solution, SolverStats, StiffSegment, TaylorSegment,
};
use crate::callback::CallbackOutcome;
use crate::event::{times_are_numerically_equal, times_are_representably_equal};
use crate::{SaveMode, SolveOptions};

pub(crate) struct TrajectoryRecorder<'a> {
    times: Vec<f64>,
    values: Vec<f64>,
    dimension: usize,
    save_at: &'a [f64],
    next_save: usize,
    save_mode: SaveMode,
    interpolation: Vec<f64>,
    dense_segments: Vec<OwnedDenseSegment>,
    retain_dense_output: bool,
}

impl<'a> TrajectoryRecorder<'a> {
    pub(crate) fn new(state: &[f64], time: f64, options: &'a SolveOptions) -> Self {
        let save_initial = options.save_at.is_empty() || options.save_at.first() == Some(&time);
        let capacity = if options.save_at.is_empty() {
            2
        } else {
            options.save_at.len()
        };
        let mut times = Vec::with_capacity(capacity);
        let mut values = Vec::with_capacity(capacity * state.len());
        if save_initial {
            times.push(time);
            values.extend_from_slice(state);
        }
        Self {
            times,
            values,
            dimension: state.len(),
            save_at: &options.save_at,
            next_save: usize::from(!options.save_at.is_empty() && save_initial),
            save_mode: options.save,
            interpolation: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; state.len()]
            },
            dense_segments: Vec::new(),
            retain_dense_output: options.retain_dense_output,
        }
    }

    pub(crate) fn record_step(
        &mut self,
        previous_state: &[f64],
        previous_time: f64,
        state: &[f64],
        time: f64,
        final_time: bool,
    ) {
        if self.save_at.is_empty() {
            if self.save_mode == SaveMode::EveryStep || final_time {
                self.push_unique(time, state);
            }
            return;
        }

        let direction = (time - previous_time).signum();
        while let Some(&target) = self.save_at.get(self.next_save) {
            if direction * (target - previous_time) <= 0.0 {
                self.next_save += 1;
                continue;
            }
            if direction * (time - target) < 0.0 {
                break;
            }
            let fraction = (target - previous_time) / (time - previous_time);
            for ((output, previous), current) in
                self.interpolation.iter_mut().zip(previous_state).zip(state)
            {
                *output = previous + fraction * (current - previous);
            }
            self.push_interpolation_target(target);
            self.next_save += 1;
        }
    }

    /// Records an accepted step using its method-provided dense interpolant.
    ///
    /// This is deliberately separate from [`record_step`]: existing kernels
    /// still use the endpoint fallback until they expose their accepted-step
    /// derivative/stage data. The helper shares the recorder's preallocated
    /// scratch buffer and never evaluates the interpolant more than once per
    /// requested save point.
    #[allow(dead_code)]
    pub(crate) fn record_step_dense(
        &mut self,
        previous_state: &[f64],
        previous_time: f64,
        state: &[f64],
        time: f64,
        final_time: bool,
        segment: &dyn DenseSegment,
    ) -> Result<(), InterpolationError> {
        let _ = previous_state;
        if self.save_at.is_empty() {
            if self.save_mode == SaveMode::EveryStep || final_time {
                self.push_unique(time, state);
            }
            return Ok(());
        }

        let direction = (time - previous_time).signum();
        while let Some(&target) = self.save_at.get(self.next_save) {
            if direction * (target - previous_time) <= 0.0 {
                self.next_save += 1;
                continue;
            }
            if direction * (time - target) < 0.0 {
                break;
            }
            segment.interpolate(target, &mut self.interpolation)?;
            self.push_interpolation_target(target);
            self.next_save += 1;
        }
        Ok(())
    }

    pub(crate) fn finish(self, stats: SolverStats) -> Solution {
        if self.dense_segments.is_empty() {
            Solution::new(self.times, self.values, self.dimension, stats)
        } else {
            Solution::new_with_dense(
                self.times,
                self.values,
                self.dimension,
                stats,
                self.dense_segments,
            )
        }
    }

    pub(crate) fn retains_dense_output(&self) -> bool {
        self.retain_dense_output
    }

    pub(crate) fn needs_dense_sampling(&self) -> bool {
        !self.save_at.is_empty()
    }

    pub(crate) fn retain_runge_kutta_segment(&mut self, segment: RungeKuttaSegment) {
        debug_assert!(self.retain_dense_output);
        self.dense_segments
            .push(OwnedDenseSegment::RungeKutta(segment));
    }

    pub(crate) fn retain_hermite_segment(&mut self, segment: HermiteSegment) {
        debug_assert!(self.retain_dense_output);
        self.dense_segments
            .push(OwnedDenseSegment::Hermite(segment));
    }

    pub(crate) fn retain_stiff_segment(&mut self, segment: StiffSegment) {
        debug_assert!(self.retain_dense_output);
        self.dense_segments.push(OwnedDenseSegment::Stiff(segment));
    }

    pub(crate) fn retain_collocation_segment(&mut self, segment: CollocationSegment) {
        debug_assert!(self.retain_dense_output);
        self.dense_segments
            .push(OwnedDenseSegment::Collocation(segment));
    }

    pub(crate) fn retain_taylor_segment(&mut self, segment: TaylorSegment) {
        debug_assert!(self.retain_dense_output);
        self.dense_segments.push(OwnedDenseSegment::Taylor(segment));
    }

    pub(crate) fn record_callback(
        &mut self,
        time: f64,
        before: &[f64],
        after: &[f64],
        outcome: CallbackOutcome,
        boundary: bool,
    ) {
        let canonical_time = self
            .save_at
            .iter()
            .copied()
            .find(|target| times_are_numerically_equal(*target, time))
            .unwrap_or(time);
        let requested_at = self
            .save_at
            .iter()
            .any(|target| times_are_numerically_equal(*target, time));
        let globally_saved_after = (self.save_at.is_empty()
            && (self.save_mode == SaveMode::EveryStep || boundary))
            || outcome.terminate;
        let save_before = outcome.save_before || requested_at;
        let save_after = outcome.save_after || globally_saved_after;

        if save_before {
            self.push_unique(canonical_time, before);
        }
        if save_after {
            if save_before {
                self.push(canonical_time, after);
            } else {
                self.push_unique(canonical_time, after);
            }
        }
    }

    pub(crate) fn synchronize_endpoint(&mut self, time: f64, state: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_representably_equal(*saved, time))
        {
            let start = self.values.len() - self.dimension;
            self.values[start..].copy_from_slice(state);
        }
    }

    fn push_interpolation_target(&mut self, time: f64) {
        if let Some(saved) = self
            .times
            .last_mut()
            .filter(|saved| times_are_representably_equal(**saved, time))
        {
            *saved = time;
            let start = self.values.len() - self.dimension;
            self.values[start..].copy_from_slice(&self.interpolation);
        } else {
            self.times.push(time);
            self.values.extend_from_slice(&self.interpolation);
        }
    }

    fn push_unique(&mut self, time: f64, state: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_representably_equal(*saved, time))
        {
            let start = self.values.len() - self.dimension;
            self.values[start..].copy_from_slice(state);
        } else {
            self.push(time, state);
        }
    }

    fn push(&mut self, time: f64, state: &[f64]) {
        debug_assert_eq!(state.len(), self.dimension);
        self.times.push(time);
        self.values.extend_from_slice(state);
    }
}
