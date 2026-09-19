use crate::solution::RungeKuttaCoefficients;
use crate::tableau::RungeKuttaTableau;

pub(crate) trait TableauAccess: Copy {
    fn order(self) -> usize;
    fn fsal(self) -> bool;
    fn nodes(self) -> &'static [f64];
    fn weights(self) -> &'static [f64];
    fn stage_row(self, stage: usize) -> &'static [f64];
    fn error_weights(self) -> Option<&'static [f64]>;
    fn second_error_weights(self) -> Option<&'static [f64]>;
    fn dense_coefficients(self) -> Option<RungeKuttaCoefficients>;
    fn lazy_stage_count(self) -> usize;
    fn lazy_stage(self, stage: usize) -> (f64, &'static [(usize, f64)]);
}

#[derive(Clone, Copy)]
pub(crate) struct ResourceTableau(pub(crate) &'static RungeKuttaTableau);

impl TableauAccess for ResourceTableau {
    fn order(self) -> usize {
        self.0.order()
    }

    fn fsal(self) -> bool {
        self.0.fsal()
    }

    fn nodes(self) -> &'static [f64] {
        self.0.c()
    }

    fn weights(self) -> &'static [f64] {
        self.0.b()
    }

    fn stage_row(self, stage: usize) -> &'static [f64] {
        &self.0.a()[stage][..stage]
    }

    fn error_weights(self) -> Option<&'static [f64]> {
        self.0.error()
    }

    fn second_error_weights(self) -> Option<&'static [f64]> {
        self.0.second_error()
    }

    fn dense_coefficients(self) -> Option<RungeKuttaCoefficients> {
        self.0.dense().map(RungeKuttaCoefficients::from)
    }

    fn lazy_stage_count(self) -> usize {
        self.0.lazy_dense_stages().len()
    }

    fn lazy_stage(self, stage: usize) -> (f64, &'static [(usize, f64)]) {
        let stage = &self.0.lazy_dense_stages()[stage];
        (stage.node(), stage.coefficients())
    }
}
