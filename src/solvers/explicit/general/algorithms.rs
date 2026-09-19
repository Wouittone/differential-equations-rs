use super::kernel::integrate_resource;
use crate::tableau::{LazyTableau, RungeKuttaTableau, load_tableau};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

/// A generic explicit Runge--Kutta solver backed by a lazy text resource.
///
/// Use [`LazyTableau`] with `include_str!` to define downstream methods
/// without a procedural macro or source-level coefficient constants.
#[derive(Clone, Copy)]
pub struct ResourceExplicitRungeKutta {
    resource: &'static LazyTableau,
}

impl ResourceExplicitRungeKutta {
    /// Creates a solver referring to a lazily parsed tableau resource.
    pub const fn new(resource: &'static LazyTableau) -> Self {
        Self { resource }
    }

    /// Loads and returns the method tableau.
    pub fn tableau(&self) -> Result<&'static RungeKuttaTableau, crate::tableau::TableauError> {
        load_tableau(self.resource)
    }
}

impl std::fmt::Debug for ResourceExplicitRungeKutta {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ResourceExplicitRungeKutta { .. }")
    }
}

impl OdeAlgorithm for ResourceExplicitRungeKutta {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let tableau = load_tableau(self.resource).map_err(SolveError::from)?;
        integrate_resource(problem, options, tableau)
    }
}

crate::tableau::define_explicit_rk_from_file!(pub Rkm, "src/tableau/resources/explicit/rkm.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Rko65, "src/tableau/resources/explicit/rko65.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Msrk5, "src/tableau/resources/explicit/msrk5.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Msrk6, "src/tableau/resources/explicit/msrk6.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Stepanov5, "src/tableau/resources/explicit/stepanov5.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Sir54, "src/tableau/resources/explicit/sir54.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Ralston4, "src/tableau/resources/explicit/ralston4.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Alshina3, "src/tableau/resources/explicit/alshina3.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Alshina6, "src/tableau/resources/explicit/alshina6.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Bs3, "src/tableau/resources/explicit/bs3.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Dp5, "src/tableau/resources/explicit/dp5.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub OwrenZen3, "src/tableau/resources/explicit/owren_zen3.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub OwrenZen4, "src/tableau/resources/explicit/owren_zen4.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub OwrenZen5, "src/tableau/resources/explicit/owren_zen5.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Bs5, "src/tableau/resources/explicit/bs5.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub SspRk22, "src/tableau/resources/explicit/ssp_rk22.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub SspRk33, "src/tableau/resources/explicit/ssp_rk33.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub SspRk43, "src/tableau/resources/explicit/ssp_rk43.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Psrk3p5q4, "src/tableau/resources/explicit/psrk3p5q4.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Psrk3p6q5, "src/tableau/resources/explicit/psrk3p6q5.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Psrk4p7q6, "src/tableau/resources/explicit/psrk4p7q6.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Euler, "src/tableau/resources/explicit/euler.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Midpoint, "src/tableau/resources/explicit/midpoint.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Heun, "src/tableau/resources/explicit/heun.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Ralston, "src/tableau/resources/explicit/ralston.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Rk4, "src/tableau/resources/explicit/rk4.json", crate = crate);
crate::tableau::define_explicit_rk_from_file!(pub Alshina2, "src/tableau/resources/explicit/alshina2.json", crate = crate);
