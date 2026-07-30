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
    use super::{Solution, SolverStats};

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
}
