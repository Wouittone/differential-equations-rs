use super::OdeProblem;

#[allow(dead_code)]
pub(crate) struct JacobianProvider<'a, F, P> {
    problem: &'a OdeProblem<F, P>,
}

#[allow(dead_code)]
impl<'a, F, P> JacobianProvider<'a, F, P> {
    pub(crate) fn new(problem: &'a OdeProblem<F, P>) -> Self {
        Self { problem }
    }

    pub(crate) fn evaluate(&self, jacobian: &mut [f64], state: &[f64], time: f64) -> bool {
        self.problem.evaluate_jacobian(jacobian, state, time)
    }

    pub(crate) fn is_analytic(&self) -> bool {
        self.problem.has_jacobian()
    }
}
