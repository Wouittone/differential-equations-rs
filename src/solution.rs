/// Work performed by an ODE solver.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SolverStats {
    /// Number of right-hand-side evaluations.
    pub rhs_evaluations: usize,
    /// Number of accepted time steps.
    pub accepted_steps: usize,
    /// Number of rejected time steps.
    pub rejected_steps: usize,
    /// Number of nonlinear iterations performed by implicit methods.
    pub nonlinear_iterations: usize,
    /// Number of Jacobian evaluations.
    pub jacobian_evaluations: usize,
    /// Number of linear systems solved.
    pub linear_solves: usize,
    /// Number of discrete or continuous callback effects applied.
    pub callback_invocations: usize,
}

use crate::{SaveMode, SolveOptions};

/// Method-specific dense interpolation seam. Segments own their endpoint
/// data and can be evaluated without mutating the solver kernel.
#[allow(dead_code)]
pub(crate) trait DenseSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), &'static str>;
}

#[allow(dead_code)]
pub(crate) struct HermiteSegment {
    start_time: f64,
    end_time: f64,
    start_state: Vec<f64>,
    end_state: Vec<f64>,
    start_derivative: Vec<f64>,
    end_derivative: Vec<f64>,
}

#[allow(dead_code)]
impl HermiteSegment {
    pub(crate) fn new(
        start_time: f64,
        end_time: f64,
        start_state: Vec<f64>,
        end_state: Vec<f64>,
        start_derivative: Vec<f64>,
        end_derivative: Vec<f64>,
    ) -> Result<Self, &'static str> {
        if !start_time.is_finite()
            || !end_time.is_finite()
            || end_time == start_time
            || start_state.is_empty()
            || start_state.len() != end_state.len()
            || start_state.len() != start_derivative.len()
            || start_state.len() != end_derivative.len()
        {
            return Err("invalid dense segment dimensions or times");
        }
        Ok(Self {
            start_time,
            end_time,
            start_state,
            end_state,
            start_derivative,
            end_derivative,
        })
    }
}

#[allow(dead_code)]
impl DenseSegment for HermiteSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), &'static str> {
        if output.len() != self.start_state.len() || !time.is_finite() {
            return Err("dense output dimension or time mismatch");
        }
        let h = self.end_time - self.start_time;
        let theta = (time - self.start_time) / h;
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
        let h10 = theta3 - 2.0 * theta2 + theta;
        let h01 = -2.0 * theta3 + 3.0 * theta2;
        let h11 = theta3 - theta2;
        for (((output, start), end), (start_derivative, end_derivative)) in output
            .iter_mut()
            .zip(&self.start_state)
            .zip(&self.end_state)
            .zip(self.start_derivative.iter().zip(&self.end_derivative))
        {
            *output =
                h00 * start + h10 * h * start_derivative + h01 * end + h11 * h * end_derivative;
        }
        Ok(())
    }
}

pub(crate) struct TrajectoryRecorder<'a> {
    times: Vec<f64>,
    values: Vec<f64>,
    dimension: usize,
    save_at: &'a [f64],
    next_save: usize,
    save_mode: SaveMode,
    interpolation: Vec<f64>,
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
            self.times.push(target);
            self.values.extend_from_slice(&self.interpolation);
            self.next_save += 1;
        }
    }

    pub(crate) fn finish(self, stats: SolverStats) -> Solution {
        Solution::new(self.times, self.values, self.dimension, stats)
    }

    pub(crate) fn force_state(&mut self, time: f64, state: &[f64]) {
        self.push_unique(time, state);
    }

    fn push_unique(&mut self, time: f64, state: &[f64]) {
        if self.times.last() == Some(&time) {
            let start = self.values.len() - self.dimension;
            self.values[start..].copy_from_slice(state);
        } else {
            self.times.push(time);
            self.values.extend_from_slice(state);
        }
    }
}

/// A saved ODE trajectory.
///
/// States are kept in one row-major allocation. The state at saved time `i`
/// occupies `values[i * dimension..(i + 1) * dimension]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    times: Vec<f64>,
    values: Vec<f64>,
    dimension: usize,
    stats: SolverStats,
}

impl Solution {
    pub(crate) fn new(
        times: Vec<f64>,
        values: Vec<f64>,
        dimension: usize,
        stats: SolverStats,
    ) -> Self {
        debug_assert_eq!(values.len(), times.len() * dimension);
        Self {
            times,
            values,
            dimension,
            stats,
        }
    }

    /// Saved times in integration order.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// All saved states in contiguous row-major storage.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Number of scalar components in each state.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns a saved state by its time index.
    pub fn state(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.dimension)?;
        self.values.get(start..start + self.dimension)
    }

    /// Returns the last saved state.
    pub fn last_state(&self) -> &[f64] {
        let start = self.values.len() - self.dimension;
        &self.values[start..]
    }

    /// Solver work counters.
    pub fn stats(&self) -> SolverStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseSegment, HermiteSegment, Solution, SolverStats};

    #[test]
    fn exposes_flat_states_as_slices() {
        let solution = Solution::new(
            vec![0.0, 0.5, 1.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            2,
            SolverStats::default(),
        );

        assert_eq!(solution.state(0), Some([1.0, 2.0].as_slice()));
        assert_eq!(solution.state(2), Some([5.0, 6.0].as_slice()));
        assert_eq!(solution.state(3), None);
        assert_eq!(solution.last_state(), &[5.0, 6.0]);
    }

    #[test]
    fn hermite_segment_matches_endpoints_and_midpoint() {
        let segment =
            HermiteSegment::new(0.0, 1.0, vec![0.0], vec![1.0], vec![0.0], vec![2.0]).unwrap();
        let mut output = [0.0];
        segment.interpolate(0.0, &mut output).unwrap();
        assert_eq!(output, [0.0]);
        segment.interpolate(1.0, &mut output).unwrap();
        assert_eq!(output, [1.0]);
        segment.interpolate(0.5, &mut output).unwrap();
        assert!((output[0] - 0.25).abs() < 1.0e-14);
    }
}
