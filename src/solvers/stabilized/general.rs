//! Explicit stabilized Runge--Kutta methods for regular ODE problems.
//!
//! The compact polynomial recurrences in this module are recovered from
//! OrdinaryDiffEqStabilizedRK at commit
//! `211142263781255a9aa2f910f6760b9f18ec29c8`. Each implemented method
//! estimates the Jacobian spectral radius by a matrix-free power iteration,
//! chooses a stage count from its own stability bound, and advances with its
//! method-specific Chebyshev, Legendre, or Gegenbauer recurrence.
//!
//! The two-step TSRKC recurrences keep their accepted-step history inside the
//! kernel. ROCK2, ROCK4, SERK2, ESERK4, and ESERK5 select their full
//! degree-indexed resources from the stabilized tableau resource tree. No
//! degree subset or substitute recurrence is used.

use super::resources::{
    eserk4_available_degrees, eserk4_tableau_for_degree, eserk5_available_degrees,
    eserk5_tableau_for_degree, rock2_available_degrees, rock2_tableau_for_degree,
    rock4_available_degrees, rock4_tableau_for_degree, serk2_available_degrees,
    serk2_tableau_for_degree,
};
use crate::integrator::integrate as drive_integration;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

mod chebyshev;
mod driver;
mod family;
mod orthogonal;
mod tabulated;
mod workspace;

use family::StabilizedFamily;
use workspace::StabilizedKernel;

macro_rules! implemented_method {
    ($name:ident, $family:expr, $documentation:literal) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl OdeAlgorithm for $name {
            fn solve_validated<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: crate::OdeFunction<P>,
            {
                drive_integration(
                    problem,
                    options,
                    StabilizedKernel::new($family, problem.initial_state().len()),
                )
            }
        }
    };
}

implemented_method!(
    RKC,
    StabilizedFamily::Rkc,
    "Second-order Runge--Kutta--Chebyshev method with variable stage count."
);
implemented_method!(
    ROCK2,
    StabilizedFamily::Rock2,
    "Second-order orthogonal-polynomial ROCK method with a tabulated finishing procedure."
);

impl ROCK2 {
    /// Returns the first available tableau whose degree is at least
    /// `requested_degree`, clamping to the largest supported degree.
    ///
    /// The returned value is parsed only on its first inspection or use. Read
    /// [`Rock2Tableau::degree`](crate::tableau::Rock2Tableau::degree) to inspect
    /// the degree selected from the discrete built-in catalogue.
    pub fn tableau(
        self,
        requested_degree: usize,
    ) -> Result<&'static crate::tableau::Rock2Tableau, crate::tableau::TableauError> {
        rock2_tableau_for_degree(requested_degree)
    }

    /// Iterates over the supported polynomial degrees in ascending order.
    pub fn available_degrees(self) -> impl ExactSizeIterator<Item = usize> {
        rock2_available_degrees()
    }
}
implemented_method!(
    ROCK4,
    StabilizedFamily::Rock4,
    "Fourth-order orthogonal-polynomial ROCK method with a tabulated finishing procedure."
);

impl ROCK4 {
    /// Returns the first available tableau whose degree is at least the
    /// requested degree, clamping to the largest supported degree.
    ///
    /// The returned value is parsed only on its first inspection or use. Its
    /// degree identifies the entry selected from the discrete built-in
    /// catalogue.
    pub fn tableau(
        self,
        requested_degree: usize,
    ) -> Result<&'static crate::tableau::Rock4Tableau, crate::tableau::TableauError> {
        rock4_tableau_for_degree(requested_degree)
    }

    /// Iterates over the supported polynomial degrees in ascending order.
    pub fn available_degrees(self) -> impl ExactSizeIterator<Item = usize> {
        rock4_available_degrees()
    }
}
implemented_method!(
    SERK2,
    StabilizedFamily::Serk2,
    "Second-order stabilized explicit Runge--Kutta method with tabulated finishing weights."
);

impl SERK2 {
    /// Returns the first available tableau whose degree is at least the
    /// requested degree, clamping to the largest supported degree.
    ///
    /// Only the selected degree is parsed on first inspection or use.
    pub fn tableau(
        self,
        requested_degree: usize,
    ) -> Result<&'static crate::tableau::Serk2Tableau, crate::tableau::TableauError> {
        serk2_tableau_for_degree(requested_degree)
    }

    /// Iterates over the supported polynomial degrees in ascending order.
    pub fn available_degrees(self) -> impl ExactSizeIterator<Item = usize> {
        serk2_available_degrees()
    }
}
implemented_method!(
    ESERK4,
    StabilizedFamily::Eserk4,
    "Fourth-order extrapolated stabilized explicit Runge--Kutta method."
);

impl ESERK4 {
    /// Returns the first available tableau whose degree is at least the
    /// requested degree, clamping to the largest supported degree.
    ///
    /// Only the selected degree is parsed on first inspection or use.
    pub fn tableau(
        self,
        requested_degree: usize,
    ) -> Result<&'static crate::tableau::EserkTableau, crate::tableau::TableauError> {
        eserk4_tableau_for_degree(requested_degree)
    }

    /// Iterates over the supported polynomial degrees in ascending order.
    pub fn available_degrees(self) -> impl ExactSizeIterator<Item = usize> {
        eserk4_available_degrees()
    }
}
implemented_method!(
    ESERK5,
    StabilizedFamily::Eserk5,
    "Fifth-order extrapolated stabilized explicit Runge--Kutta method."
);

impl ESERK5 {
    /// Returns the first available tableau whose degree is at least the
    /// requested degree, clamping to the largest supported degree.
    ///
    /// Only the selected degree is parsed on first inspection or use.
    pub fn tableau(
        self,
        requested_degree: usize,
    ) -> Result<&'static crate::tableau::EserkTableau, crate::tableau::TableauError> {
        eserk5_tableau_for_degree(requested_degree)
    }

    /// Iterates over the supported polynomial degrees in ascending order.
    pub fn available_degrees(self) -> impl ExactSizeIterator<Item = usize> {
        eserk5_available_degrees()
    }
}
implemented_method!(
    TSRKC2,
    StabilizedFamily::Tsrkc2,
    "Two-step second-order Runge--Kutta--Chebyshev method."
);
implemented_method!(
    TSRKC3,
    StabilizedFamily::Tsrkc3,
    "Two-step third-order Runge--Kutta--Chebyshev method."
);
implemented_method!(
    RKL1,
    StabilizedFamily::Rkl1,
    "First-order Runge--Kutta--Legendre super-time-stepping method."
);
implemented_method!(
    RKL2,
    StabilizedFamily::Rkl2,
    "Second-order Runge--Kutta--Legendre super-time-stepping method."
);
implemented_method!(
    RKG1,
    StabilizedFamily::Rkg1,
    "First-order Runge--Kutta--Gegenbauer super-time-stepping method."
);
implemented_method!(
    RKG2,
    StabilizedFamily::Rkg2,
    "Second-order Runge--Kutta--Gegenbauer super-time-stepping method."
);
implemented_method!(
    RKMC2,
    StabilizedFamily::Rkmc2,
    "Second-order monotone Runge--Kutta--Chebyshev method."
);

#[cfg(test)]
mod tests {
    use super::{
        ESERK4, ESERK5, ROCK2, ROCK4, SERK2, StabilizedFamily, StabilizedKernel, TSRKC2, TSRKC3,
    };
    use crate::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, SolverStats, solve};

    type ScalarRhs = fn(&mut [f64], &[f64], &(), f64);

    fn fixed_options(step: f64) -> SolveOptions {
        SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    fn exponential() -> OdeProblem<ScalarRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 1.0), ())
    }

    fn convergence_ratio<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
        let exact = std::f64::consts::E;
        let endpoint = |step| {
            solve(&exponential(), algorithm, &fixed_options(step))
                .expect("two-step convergence solve failed")
                .last_state()[0]
        };
        (endpoint(0.1) - exact).abs() / (endpoint(0.05) - exact).abs()
    }

    fn problem_convergence_ratio<A: OdeAlgorithm + Copy>(
        problem: &OdeProblem<ScalarRhs, ()>,
        algorithm: A,
        exact: f64,
        coarse: f64,
        fine: f64,
    ) -> f64 {
        let endpoint = |step| {
            solve(problem, algorithm, &fixed_options(step))
                .expect("convergence solve failed")
                .last_state()[0]
        };
        (endpoint(coarse) - exact).abs() / (endpoint(fine) - exact).abs()
    }

    #[test]
    fn two_step_methods_recover_their_formal_orders() {
        assert!(convergence_ratio(TSRKC2).log2() > 1.5);
        assert!(convergence_ratio(TSRKC3).log2() > 2.5);
    }

    #[test]
    fn tabulated_methods_recover_their_formal_orders() {
        assert!(convergence_ratio(ROCK2).log2() > 1.5);
        assert!(convergence_ratio(SERK2).log2() > 1.5);
        assert!(convergence_ratio(ROCK4).log2() > 3.5);
        assert!(convergence_ratio(ESERK4).log2() > 3.5);
        assert!(convergence_ratio(ESERK5).log2() > 4.5);
    }

    #[test]
    fn serk2_recovers_order_two_on_a_nonautonomous_problem() {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
            du[0] = u[0] + time;
        }
        let problem = OdeProblem::new(rhs as ScalarRhs, vec![0.0], (0.0, 1.0), ());
        let exact = std::f64::consts::E - 2.0;
        let endpoint = |step| {
            solve(&problem, SERK2, &fixed_options(step))
                .expect("nonautonomous SERK2 convergence solve failed")
                .last_state()[0]
        };
        let ratio = (endpoint(0.1) - exact).abs() / (endpoint(0.05) - exact).abs();
        assert!(
            ratio.log2() > 1.5,
            "SERK2 observed order was {}",
            ratio.log2()
        );
    }

    #[test]
    fn serk2_uses_input_state_nodes_in_a_multi_stage_recurrence() {
        fn rhs(du: &mut [f64], _: &[f64], _: &(), time: f64) {
            du[0] = time;
        }
        let problem = OdeProblem::new(rhs as ScalarRhs, vec![0.0], (0.0, 0.1), ());
        let mut kernel = StabilizedKernel::new(StabilizedFamily::Serk2, 1);
        let mut candidate = [0.0];
        let mut stats = SolverStats::default();

        // This radius requests degree 11 and therefore selects degree 20,
        // whose internal degree is two. It exercises both SERK2 stage nodes.
        kernel
            .run_serk2(&problem, &[0.0], 0.0, 0.1, 80.0, &mut candidate, &mut stats)
            .expect("multi-stage SERK2 step failed");

        assert!((candidate[0] - 0.005).abs() < 2.0e-15);
    }

    #[test]
    fn eserk_methods_use_input_state_nodes_across_recurrence_restarts() {
        fn rhs(du: &mut [f64], _: &[f64], _: &(), time: f64) {
            du[0] = time;
        }
        let problem = OdeProblem::new(rhs as ScalarRhs, vec![0.0], (0.0, 0.1), ());

        for family in [StabilizedFamily::Eserk4, StabilizedFamily::Eserk5] {
            let mut kernel = StabilizedKernel::new(family, 1);
            let mut candidate = [0.0];
            let mut stats = SolverStats::default();

            // A radius of nine selects degree four in both catalogues. With
            // internal degree two, the step crosses a recurrence restart.
            kernel
                .run_eserk(&problem, &[0.0], 0.0, 0.1, 9.0, &mut candidate, &mut stats)
                .expect("multi-stage ESERK step failed");

            assert!(
                (candidate[0] - 0.005).abs() < 2.0e-14,
                "{family:?} produced {}",
                candidate[0]
            );
        }
    }

    #[test]
    fn eserk_methods_recover_their_orders_on_a_nonautonomous_problem() {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
            du[0] = u[0] + time;
        }
        let problem = OdeProblem::new(rhs as ScalarRhs, vec![0.0], (0.0, 1.0), ());
        let exact = std::f64::consts::E - 2.0;

        assert!(problem_convergence_ratio(&problem, ESERK4, exact, 0.1, 0.05).log2() > 3.5);
        assert!(problem_convergence_ratio(&problem, ESERK5, exact, 0.2, 0.1).log2() > 4.5);
    }

    #[test]
    fn two_step_methods_stabilize_a_real_negative_mode() {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
            du[0] = -40.0 * (u[0] - time.cos()) - time.sin();
        }
        let problem = OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 1.0), ());
        for endpoint in [
            solve(&problem, TSRKC2, &fixed_options(0.05))
                .expect("TSRKC2 stiff solve failed")
                .last_state()[0],
            solve(&problem, TSRKC3, &fixed_options(0.05))
                .expect("TSRKC3 stiff solve failed")
                .last_state()[0],
            solve(&problem, ROCK2, &fixed_options(0.05))
                .expect("ROCK2 stiff solve failed")
                .last_state()[0],
            solve(&problem, ROCK4, &fixed_options(0.05))
                .expect("ROCK4 stiff solve failed")
                .last_state()[0],
            solve(&problem, SERK2, &fixed_options(0.05))
                .expect("SERK2 stiff solve failed")
                .last_state()[0],
            solve(&problem, ESERK4, &fixed_options(0.05))
                .expect("ESERK4 stiff solve failed")
                .last_state()[0],
            solve(&problem, ESERK5, &fixed_options(0.05))
                .expect("ESERK5 stiff solve failed")
                .last_state()[0],
        ] {
            assert!((endpoint - 1.0_f64.cos()).abs() < 1.5e-3);
        }
    }
}
