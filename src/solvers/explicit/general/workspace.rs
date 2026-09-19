pub(super) struct Workspace {
    // Flat stage-major storage: every stage is one contiguous component array.
    // The other work vectors remain separate arrays rather than per-component
    // structs, keeping the hot saxpy-style loops friendly to SIMD.
    pub(super) stages: Vec<f64>,
    pub(super) dimension: usize,
    pub(super) temporary: Vec<f64>,
    pub(super) stiffness_reference_state: Vec<f64>,
}

impl Workspace {
    pub(super) fn new(stage_count: usize, dimension: usize, stiffness_detection: bool) -> Self {
        Self {
            stages: vec![0.0; stage_count * dimension],
            dimension,
            temporary: vec![0.0; dimension],
            stiffness_reference_state: if stiffness_detection {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
        }
    }

    pub(super) fn stage(&self, index: usize) -> &[f64] {
        let start = index * self.dimension;
        &self.stages[start..start + self.dimension]
    }

    pub(super) fn swap_stages(&mut self, left: usize, right: usize) {
        let left_start = left * self.dimension;
        let right_start = right * self.dimension;
        for offset in 0..self.dimension {
            self.stages.swap(left_start + offset, right_start + offset);
        }
    }
}
