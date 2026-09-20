use super::{DenseSegment, validate_dense_query};
use crate::solution::InterpolationError;

/// Borrowed collocation extension for one accepted step without creating a
/// temporary owning segment during dense recording.
pub(crate) struct BorrowedCollocationSegment<'a> {
    start_time: f64,
    attempted_time: f64,
    start_state: &'a [f64],
    midpoint_state: &'a [f64],
    endpoint_state: &'a [f64],
    stages: &'a [f64],
    first_half_stages: &'a [f64],
    second_half_stages: &'a [f64],
    lagrange: &'a [f64],
    dimension: usize,
    stage_count: usize,
    adaptive: bool,
}

/// Owning dynamic collocation extension used by variable-stage FIRK methods.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CollocationSegment {
    start_time: f64,
    attempted_time: f64,
    bound_time: f64,
    start_state: Vec<f64>,
    midpoint_state: Vec<f64>,
    endpoint_state: Vec<f64>,
    stages: Vec<f64>,
    first_half_stages: Vec<f64>,
    second_half_stages: Vec<f64>,
    lagrange: Vec<f64>,
    dimension: usize,
    stage_count: usize,
    adaptive: bool,
}

impl<'a> BorrowedCollocationSegment<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        start_time: f64,
        attempted_time: f64,
        start_state: &'a [f64],
        midpoint_state: &'a [f64],
        endpoint_state: &'a [f64],
        stages: &'a [f64],
        first_half_stages: &'a [f64],
        second_half_stages: &'a [f64],
        lagrange: &'a [f64],
        stage_count: usize,
        adaptive: bool,
    ) -> Result<Self, InterpolationError> {
        let dimension = start_state.len();
        if !start_time.is_finite()
            || !attempted_time.is_finite()
            || attempted_time == start_time
            || dimension == 0
            || endpoint_state.len() != dimension
            || midpoint_state.len() != dimension
            || stage_count == 0
            || lagrange.len() != stage_count * stage_count
            || stages.len() != stage_count * dimension
            || first_half_stages.len() != stage_count * dimension
            || second_half_stages.len() != stage_count * dimension
        {
            return Err(InterpolationError::InvalidSegmentData {
                context: "borrowed collocation segment",
            });
        }
        Ok(Self {
            start_time,
            attempted_time,
            start_state,
            midpoint_state,
            endpoint_state,
            stages,
            first_half_stages,
            second_half_stages,
            lagrange,
            dimension,
            stage_count,
            adaptive,
        })
    }

    pub(super) fn contains(&self, time: f64) -> bool {
        time.is_finite()
            && if self.start_time < self.attempted_time {
                (self.start_time..=self.attempted_time).contains(&time)
            } else {
                (self.attempted_time..=self.start_time).contains(&time)
            }
    }
}

impl BorrowedCollocationSegment<'_> {
    fn interpolate_piece(
        &self,
        start_time: f64,
        step: f64,
        start_state: &[f64],
        stages: &[f64],
        time: f64,
        output: &mut [f64],
    ) {
        let theta = ((time - start_time) / step).clamp(0.0, 1.0);
        output.copy_from_slice(start_state);
        for stage in 0..self.stage_count {
            let mut power = theta;
            let mut weight = 0.0;
            for degree in 0..self.stage_count {
                weight +=
                    self.lagrange[stage * self.stage_count + degree] * power / (degree + 1) as f64;
                power *= theta;
            }
            for component in 0..self.dimension {
                output[component] += step * weight * stages[stage * self.dimension + component];
            }
        }
    }
}

impl DenseSegment for BorrowedCollocationSegment<'_> {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        validate_dense_query(self.contains(time), output.len(), self.dimension)?;
        if time == self.attempted_time {
            output.copy_from_slice(self.endpoint_state);
            return Ok(());
        }
        let step = self.attempted_time - self.start_time;
        if self.adaptive {
            let half = 0.5 * step;
            if step.signum() * (time - (self.start_time + half)) <= 0.0 {
                self.interpolate_piece(
                    self.start_time,
                    half,
                    self.start_state,
                    self.first_half_stages,
                    time,
                    output,
                );
            } else {
                self.interpolate_piece(
                    self.start_time + half,
                    half,
                    self.midpoint_state,
                    self.second_half_stages,
                    time,
                    output,
                );
            }
        } else {
            self.interpolate_piece(
                self.start_time,
                step,
                self.start_state,
                self.stages,
                time,
                output,
            );
        }
        output
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(InterpolationError::NonFiniteResult {
                context: "borrowed collocation",
            })
    }
}

impl CollocationSegment {
    pub(super) fn time_bounds(&self) -> (f64, f64) {
        (self.start_time, self.bound_time)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        start_time: f64,
        attempted_time: f64,
        bound_time: f64,
        start_state: &[f64],
        midpoint_state: &[f64],
        endpoint_state: &[f64],
        stages: &[f64],
        first_half_stages: &[f64],
        second_half_stages: &[f64],
        lagrange: &[f64],
        stage_count: usize,
        adaptive: bool,
    ) -> Result<Self, InterpolationError> {
        let dimension = start_state.len();
        let within_step = if start_time < attempted_time {
            (start_time..=attempted_time).contains(&bound_time)
        } else {
            (attempted_time..=start_time).contains(&bound_time)
        };
        if !start_time.is_finite()
            || !attempted_time.is_finite()
            || !bound_time.is_finite()
            || attempted_time == start_time
            || !within_step
            || dimension == 0
            || endpoint_state.len() != dimension
            || midpoint_state.len() != dimension
            || stage_count == 0
            || lagrange.len() != stage_count * stage_count
            || stages.len() != stage_count * dimension
            || first_half_stages.len() != stage_count * dimension
            || second_half_stages.len() != stage_count * dimension
        {
            return Err(InterpolationError::InvalidSegmentData {
                context: "collocation segment",
            });
        }
        Ok(Self {
            start_time,
            attempted_time,
            bound_time,
            start_state: start_state.to_vec(),
            midpoint_state: midpoint_state.to_vec(),
            endpoint_state: endpoint_state.to_vec(),
            stages: stages.to_vec(),
            first_half_stages: first_half_stages.to_vec(),
            second_half_stages: second_half_stages.to_vec(),
            lagrange: lagrange.to_vec(),
            dimension,
            stage_count,
            adaptive,
        })
    }

    pub(super) fn contains(&self, time: f64) -> bool {
        time.is_finite()
            && if self.start_time < self.bound_time {
                (self.start_time..=self.bound_time).contains(&time)
            } else {
                (self.bound_time..=self.start_time).contains(&time)
            }
    }

    fn interpolate_piece(
        &self,
        start_time: f64,
        step: f64,
        start_state: &[f64],
        stages: &[f64],
        time: f64,
        output: &mut [f64],
    ) {
        let theta = ((time - start_time) / step).clamp(0.0, 1.0);
        output.copy_from_slice(start_state);
        for stage in 0..self.stage_count {
            let mut power = theta;
            let mut weight = 0.0;
            for degree in 0..self.stage_count {
                weight +=
                    self.lagrange[stage * self.stage_count + degree] * power / (degree + 1) as f64;
                power *= theta;
            }
            for component in 0..self.dimension {
                output[component] += step * weight * stages[stage * self.dimension + component];
            }
        }
    }
}

impl DenseSegment for CollocationSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        validate_dense_query(self.contains(time), output.len(), self.dimension)?;
        if time == self.bound_time {
            output.copy_from_slice(&self.endpoint_state);
            return Ok(());
        }
        let step = self.attempted_time - self.start_time;
        if self.adaptive {
            let half = 0.5 * step;
            if step.signum() * (time - (self.start_time + half)) <= 0.0 {
                self.interpolate_piece(
                    self.start_time,
                    half,
                    &self.start_state,
                    &self.first_half_stages,
                    time,
                    output,
                );
            } else {
                self.interpolate_piece(
                    self.start_time + half,
                    half,
                    &self.midpoint_state,
                    &self.second_half_stages,
                    time,
                    output,
                );
            }
        } else {
            self.interpolate_piece(
                self.start_time,
                step,
                &self.start_state,
                &self.stages,
                time,
                output,
            );
        }
        output
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(InterpolationError::NonFiniteResult {
                context: "collocation",
            })
    }
}

impl CollocationSegment {
    pub(crate) fn portable(&self) -> Result<Vec<crate::PortableDenseSegment>, InterpolationError> {
        let h = self.attempted_time - self.start_time;
        let midpoint = self.start_time + 0.5 * h;
        let mut output = Vec::new();
        if self.adaptive {
            let bound = if h.signum() * (self.bound_time - midpoint) <= 0.0 {
                self.bound_time
            } else {
                midpoint
            };
            output.push(self.portable_piece(
                self.start_time,
                midpoint,
                bound,
                &self.start_state,
                &self.first_half_stages,
            )?);
            if h.signum() * (self.bound_time - midpoint) > 0.0 {
                output.push(self.portable_piece(
                    midpoint,
                    self.attempted_time,
                    self.bound_time,
                    &self.midpoint_state,
                    &self.second_half_stages,
                )?);
            }
        } else {
            output.push(self.portable_piece(
                self.start_time,
                self.attempted_time,
                self.bound_time,
                &self.start_state,
                &self.stages,
            )?);
        }
        Ok(output)
    }
    fn portable_piece(
        &self,
        start: f64,
        end: f64,
        bound: f64,
        state: &[f64],
        stages: &[f64],
    ) -> Result<crate::PortableDenseSegment, InterpolationError> {
        let n = self.dimension;
        let mut c = vec![0.0; (self.stage_count + 1) * n];
        c[..n].copy_from_slice(state);
        for stage in 0..self.stage_count {
            for degree in 0..self.stage_count {
                let weight = (end - start) * self.lagrange[stage * self.stage_count + degree]
                    / (degree + 1) as f64;
                for i in 0..n {
                    c[(degree + 1) * n + i] += weight * stages[stage * n + i];
                }
            }
        }
        let mut end_state = vec![0.0; n];
        for row in c.chunks_exact(n) {
            for (out, &v) in end_state.iter_mut().zip(row) {
                *out += v;
            }
        }
        let bound_state = if bound == self.bound_time {
            Some(self.endpoint_state.clone())
        } else {
            None
        };
        crate::PortableDenseSegment::from_data(crate::DenseSegmentData {
            version: 1,
            start_time: start,
            end_time: end,
            bound_time: bound,
            dimension: n,
            coefficients: c,
            end_state,
            bound_state,
            quality: crate::InterpolationQuality::MethodSpecific,
        })
    }
}
