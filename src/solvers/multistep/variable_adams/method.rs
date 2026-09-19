#[derive(Clone, Copy)]
pub(super) struct VariableAdamsMethod {
    pub(super) order: usize,
    pub(super) corrector: bool,
}

pub(super) const SAFETY: f64 = 0.9;
pub(super) const MIN_FACTOR: f64 = 0.2;
pub(super) const MAX_FACTOR: f64 = 5.0;

pub(super) const VCAB3_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 3,
    corrector: false,
};
pub(super) const VCAB4_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 4,
    corrector: false,
};
pub(super) const VCAB5_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 5,
    corrector: false,
};
pub(super) const VCABM3_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 3,
    corrector: true,
};
pub(super) const VCABM4_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 4,
    corrector: true,
};
pub(super) const VCABM5_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 5,
    corrector: true,
};
