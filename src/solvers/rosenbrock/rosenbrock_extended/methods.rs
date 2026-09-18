use super::super::tableaux::*;
use super::kernel::ExtendedRosenbrockKernel;
use super::steps::{AdaptiveErrorEstimator, perform_rodas, perform_rosenbrock32, perform_tsit5da};
use super::workspace::Workspace;
use super::*;
use crate::integrator::integrate as drive_integration;
use crate::tableau::{RosenbrockKind, RosenbrockTableau, TableauError, load_tableau};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

#[allow(clippy::too_many_arguments)]
pub(crate) trait ExtendedRosenbrockMethod {
    const ERROR_ORDER: usize;
    const ADAPTIVE: bool;
    const HAS_STIFFNESS_ESTIMATE: bool = true;
    const RESOURCE_KIND: RosenbrockKind = RosenbrockKind::Rosenbrock;
    const SPECIAL_DENSE: bool = false;

    fn load_resource() -> Result<Option<&'static RosenbrockTableau>, SolveError> {
        Ok(None)
    }

    fn perform_step<F, P>(
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        workspace: &mut Workspace,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: crate::OdeFunction<P>;
}

macro_rules! algorithm {
    ($name:ident) => {
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
                    ExtendedRosenbrockKernel::<Self>::new(problem.initial_state().len())?,
                )
            }
        }
    };
}

algorithm!(Rosenbrock32);
algorithm!(Ros2);
algorithm!(Rodas3);
algorithm!(Rodas3d);
algorithm!(Ros3);
algorithm!(Ros3Pr);
algorithm!(Ros3Prl);
algorithm!(Ros3Prl2);
algorithm!(Ros3p);
algorithm!(Ros34Prw);
algorithm!(Ros34Pw3);
algorithm!(Grk4a);
algorithm!(Grk4t);
algorithm!(Rok4a);
algorithm!(Ros34Pw1b);
algorithm!(Ros34Pw2);
algorithm!(Rodas4);
algorithm!(Rodas42);
algorithm!(Rodas4P);
algorithm!(Rodas4P2);
algorithm!(Rodas4PW);
algorithm!(Rodas5);
algorithm!(Rodas5P);
algorithm!(Rodas5Pe);
algorithm!(Rodas5Pr);
algorithm!(Rodas6P);
algorithm!(RosenbrockW6S4OS);
algorithm!(Rodas23W);
algorithm!(HybridExplicitImplicitRK);
algorithm!(Rodas3P);
algorithm!(Ros2Pr);
algorithm!(Ros2S);
algorithm!(Ros34Pw1a);
algorithm!(Ros4LStab);
algorithm!(RosShamp4);
algorithm!(Scholz4_7);
algorithm!(Veldd4);
algorithm!(Velds4);

impl ExtendedRosenbrockMethod for Rosenbrock32 {
    const ERROR_ORDER: usize = 3;
    const ADAPTIVE: bool = true;
    const SPECIAL_DENSE: bool = true;

    fn perform_step<F, P>(
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        workspace: &mut Workspace,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        perform_rosenbrock32(
            problem, candidate, state, time, step, options, workspace, stats,
        )
    }
}

macro_rules! resource_access {
    ($name:ident, $tableau:ident) => {
        impl $name {
            /// Returns this method's shared, lazily parsed Rosenbrock tableau.
            ///
            pub fn tableau(&self) -> Result<&'static RosenbrockTableau, TableauError> {
                load_tableau(&$tableau)
            }
        }
    };
}

macro_rules! rodas_method {
    ($name:ident, $order:literal, $tableau:ident) => {
        rodas_method!(
            $name,
            $order,
            $tableau,
            false,
            AdaptiveErrorEstimator::Embedded
        );
    };
    ($name:ident, $order:literal, $tableau:ident, $residual_control:expr, $estimator:expr) => {
        resource_access!($name, $tableau);
        impl ExtendedRosenbrockMethod for $name {
            const ERROR_ORDER: usize = $order;
            const ADAPTIVE: bool = true;

            fn load_resource() -> Result<Option<&'static RosenbrockTableau>, SolveError> {
                load_tableau(&$tableau)
                    .map(Some)
                    .map_err(|_| SolveError::InvalidTableau)
            }

            fn perform_step<F, P>(
                problem: &OdeProblem<F, P>,
                state: &[f64],
                time: f64,
                step: f64,
                candidate: &mut [f64],
                options: &SolveOptions,
                workspace: &mut Workspace,
                stats: &mut SolverStats,
            ) -> Result<f64, SolveError>
            where
                F: crate::OdeFunction<P>,
            {
                perform_rodas(
                    problem,
                    candidate,
                    state,
                    time,
                    step,
                    options,
                    workspace.tableau.ok_or(SolveError::InvalidTableau)?,
                    $residual_control,
                    $estimator,
                    workspace,
                    stats,
                )
            }
        }
    };
}

rodas_method!(Ros2, 2, ROS2_TABLEAU);
rodas_method!(Rodas3, 3, RODAS3_TABLEAU);
rodas_method!(Rodas3d, 3, RODAS3D_TABLEAU);
rodas_method!(Ros3, 3, ROS3_TABLEAU);
rodas_method!(Ros3Pr, 3, ROS3PR_TABLEAU);
rodas_method!(Ros3Prl, 3, ROS3PRL_TABLEAU);
rodas_method!(Ros3Prl2, 3, ROS3PRL2_TABLEAU);
rodas_method!(Ros3p, 3, ROS3P_TABLEAU);
rodas_method!(Ros34Prw, 3, ROS34PRW_TABLEAU);
rodas_method!(Ros34Pw3, 4, ROS34PW3_TABLEAU);
rodas_method!(Grk4a, 4, GRK4A_TABLEAU);
rodas_method!(Grk4t, 4, GRK4T_TABLEAU);
rodas_method!(Rok4a, 4, ROK4A_TABLEAU);
rodas_method!(Ros34Pw1b, 3, ROS34PW1B_TABLEAU);
rodas_method!(Ros34Pw2, 3, ROS34PW2_TABLEAU);
rodas_method!(Rodas4, 4, RODAS4_TABLEAU);
rodas_method!(Rodas42, 4, RODAS42_TABLEAU);
rodas_method!(Rodas4P, 4, RODAS4P_TABLEAU);
rodas_method!(Rodas4P2, 4, RODAS4P2_TABLEAU);
rodas_method!(Rodas4PW, 4, RODAS4PW_TABLEAU);
rodas_method!(Rodas5, 5, RODAS5_TABLEAU);
rodas_method!(Rodas5P, 5, RODAS5P_TABLEAU);
rodas_method!(Rodas5Pe, 5, RODAS5PE_TABLEAU);
rodas_method!(Rodas6P, 6, RODAS6P_TABLEAU);
rodas_method!(Rodas23W, 3, RODAS23W_TABLEAU);
rodas_method!(Rodas3P, 3, RODAS3P_TABLEAU);
rodas_method!(Ros2Pr, 2, ROS2PR_TABLEAU);
rodas_method!(Ros2S, 2, ROS2S_TABLEAU);
rodas_method!(
    Ros34Pw1a,
    3,
    ROS34PW1A_TABLEAU,
    false,
    AdaptiveErrorEstimator::RichardsonStepDoubling { method_order: 3 }
);
rodas_method!(Ros4LStab, 4, ROS4LSTAB_TABLEAU);
rodas_method!(RosShamp4, 4, ROSSHAMP4_TABLEAU);
rodas_method!(Scholz4_7, 4, SCHOLZ4_7_TABLEAU);
rodas_method!(Veldd4, 4, VELDD4_TABLEAU);
rodas_method!(Velds4, 4, VELDS4_TABLEAU);

resource_access!(HybridExplicitImplicitRK, TSIT5DA_TABLEAU);

impl ExtendedRosenbrockMethod for HybridExplicitImplicitRK {
    const ERROR_ORDER: usize = 5;
    const ADAPTIVE: bool = true;
    const HAS_STIFFNESS_ESTIMATE: bool = false;
    const RESOURCE_KIND: RosenbrockKind = RosenbrockKind::HybridExplicitImplicit;

    fn load_resource() -> Result<Option<&'static RosenbrockTableau>, SolveError> {
        load_tableau(&TSIT5DA_TABLEAU)
            .map(Some)
            .map_err(|_| SolveError::InvalidTableau)
    }

    fn perform_step<F, P>(
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        workspace: &mut Workspace,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        perform_tsit5da(
            problem, candidate, state, time, step, options, workspace, stats,
        )
    }
}

rodas_method!(
    Rodas5Pr,
    5,
    RODAS5P_TABLEAU,
    true,
    AdaptiveErrorEstimator::Embedded
);

resource_access!(RosenbrockW6S4OS, ROSENBROCK_W6S4OS_TABLEAU);

impl ExtendedRosenbrockMethod for RosenbrockW6S4OS {
    const ERROR_ORDER: usize = 4;
    const ADAPTIVE: bool = false;

    fn load_resource() -> Result<Option<&'static RosenbrockTableau>, SolveError> {
        load_tableau(&ROSENBROCK_W6S4OS_TABLEAU)
            .map(Some)
            .map_err(|_| SolveError::InvalidTableau)
    }

    fn perform_step<F, P>(
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        workspace: &mut Workspace,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        perform_rodas(
            problem,
            candidate,
            state,
            time,
            step,
            options,
            workspace.tableau.ok_or(SolveError::InvalidTableau)?,
            false,
            AdaptiveErrorEstimator::Embedded,
            workspace,
            stats,
        )
    }
}
