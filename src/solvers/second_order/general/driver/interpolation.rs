use super::*;

pub(crate) fn interpolate(current: &[f64], previous: &[f64], fraction: f64, output: &mut [f64]) {
    for ((output, previous), current) in output.iter_mut().zip(previous).zip(current) {
        *output = interpolate_value(*previous, *current, fraction);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn interpolate_partitioned(
    start_velocity: &[f64],
    start_position: &[f64],
    end_velocity: &[f64],
    end_position: &[f64],
    step: f64,
    theta: f64,
    velocity: &mut [f64],
    position: &mut [f64],
) {
    let theta2 = theta * theta;
    let theta3 = theta2 * theta;
    let h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
    let h10 = theta3 - 2.0 * theta2 + theta;
    let h01 = -2.0 * theta3 + 3.0 * theta2;
    let h11 = theta3 - theta2;
    for index in 0..velocity.len() {
        velocity[index] = interpolate_value(start_velocity[index], end_velocity[index], theta);
        position[index] = h00 * start_position[index]
            + h10 * step * start_velocity[index]
            + h01 * end_position[index]
            + h11 * step * end_velocity[index];
    }
}

pub(super) fn ensure_finite_state(velocity: &[f64], position: &[f64]) -> Result<(), SolveError> {
    velocity
        .iter()
        .chain(position)
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteCallbackState)
}

pub(super) struct PartitionedRecorder<'a> {
    times: Vec<f64>,
    velocities: Vec<f64>,
    positions: Vec<f64>,
    dimension: usize,
    save_at: &'a [f64],
    next_save: usize,
    save_mode: SaveMode,
    interpolation_velocity: Vec<f64>,
    interpolation_position: Vec<f64>,
    dense_segments: Vec<PartitionedDenseSegment>,
    retain_dense_output: bool,
}

impl<'a> PartitionedRecorder<'a> {
    pub(super) fn new(
        velocity: &[f64],
        position: &[f64],
        time: f64,
        options: &'a SolveOptions,
    ) -> Self {
        let save_initial = options.save_at.is_empty() || options.save_at.first() == Some(&time);
        let capacity = if options.save_at.is_empty() {
            2
        } else {
            options.save_at.len()
        };
        let mut recorder = Self {
            times: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity * velocity.len()),
            positions: Vec::with_capacity(capacity * position.len()),
            dimension: position.len(),
            save_at: &options.save_at,
            next_save: usize::from(!options.save_at.is_empty() && save_initial),
            save_mode: options.save,
            interpolation_velocity: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; velocity.len()]
            },
            interpolation_position: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; position.len()]
            },
            dense_segments: Vec::new(),
            retain_dense_output: options.retain_dense_output,
        };
        if save_initial {
            recorder.push_unique(time, velocity, position);
        }
        recorder
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_step(
        &mut self,
        previous_velocity: &[f64],
        previous_position: &[f64],
        previous_time: f64,
        velocity: &[f64],
        position: &[f64],
        time: f64,
        final_time: bool,
        mut interpolator: Option<&mut PartitionedInterpolator<'_>>,
    ) -> Result<(), SolveError> {
        let generic_segment = PartitionedDenseSegment::new(
            previous_time,
            time,
            previous_velocity,
            velocity,
            previous_position,
            position,
        );
        if self.retain_dense_output {
            self.dense_segments.push(generic_segment.clone());
        }
        if self.save_at.is_empty() {
            if self.save_mode == SaveMode::EveryStep || final_time {
                self.push_unique(time, velocity, position);
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
            if let Some(interpolator) = interpolator.as_deref_mut() {
                interpolator(
                    target,
                    &mut self.interpolation_velocity,
                    &mut self.interpolation_position,
                )
                .map_err(|_| SolveError::DenseOutputFailed)?;
            } else {
                generic_segment
                    .interpolate(
                        target,
                        &mut self.interpolation_velocity,
                        &mut self.interpolation_position,
                    )
                    .ok_or(SolveError::DenseOutputFailed)?;
            }
            self.times.push(target);
            self.velocities
                .extend_from_slice(&self.interpolation_velocity);
            self.positions
                .extend_from_slice(&self.interpolation_position);
            self.next_save += 1;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_callback(
        &mut self,
        time: f64,
        before_velocity: &[f64],
        before_position: &[f64],
        after_velocity: &[f64],
        after_position: &[f64],
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
            self.push_unique(canonical_time, before_velocity, before_position);
        }
        if save_after {
            if save_before {
                self.push(canonical_time, after_velocity, after_position);
            } else {
                self.push_unique(canonical_time, after_velocity, after_position);
            }
        }
    }

    pub(super) fn synchronize_endpoint(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_representably_equal(*saved, time))
        {
            let start = self.velocities.len() - self.dimension;
            self.velocities[start..].copy_from_slice(velocity);
            self.positions[start..].copy_from_slice(position);
        }
    }

    fn push_unique(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_representably_equal(*saved, time))
        {
            let start = self.velocities.len() - self.dimension;
            self.velocities[start..].copy_from_slice(velocity);
            self.positions[start..].copy_from_slice(position);
        } else {
            self.push(time, velocity, position);
        }
    }

    fn push(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        debug_assert_eq!(velocity.len(), self.dimension);
        debug_assert_eq!(position.len(), self.dimension);
        self.times.push(time);
        self.velocities.extend_from_slice(velocity);
        self.positions.extend_from_slice(position);
    }

    pub(super) fn finish(self, stats: SolverStats, state_shape: IxDyn) -> SecondOrderSolution {
        SecondOrderSolution {
            times: self.times,
            velocities: self.velocities,
            positions: self.positions,
            dimension: self.dimension,
            state_shape,
            stats,
            dense_segments: self.dense_segments,
            portable_segments: Vec::new(),
        }
    }
}
