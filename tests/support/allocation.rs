//! Helpers for process-wide allocation measurements in integration tests.

/// Returns the least noisy of three process-wide allocation measurements.
///
/// `stats_alloc` observes allocations from every thread in the test process,
/// including occasional test-harness work. Repetition removes that bounded
/// noise while preserving allocation growth intrinsic to the measured solve.
pub fn minimum_measurement(mut measure: impl FnMut() -> usize) -> usize {
    (0..3)
        .map(|_| measure())
        .min()
        .expect("the allocation sample count is non-zero")
}
