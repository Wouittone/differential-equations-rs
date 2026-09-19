use differential_equations::solvers::{automatic::*, explicit::*, rosenbrock::*};
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);
type TrackingRhs = fn(&mut [f64], &[f64], &StiffnessProfile, f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[0] + time;
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

#[derive(Clone, Copy)]
enum StiffnessProfile {
    NonstiffThenStiff,
    StiffThenNonstiff,
}

fn tracking_rate(profile: &StiffnessProfile, time: f64) -> f64 {
    match profile {
        StiffnessProfile::NonstiffThenStiff if time < 0.5 => 1.0,
        StiffnessProfile::NonstiffThenStiff => 500.0,
        StiffnessProfile::StiffThenNonstiff if time < 0.5 => 500.0,
        StiffnessProfile::StiffThenNonstiff => 1.0,
    }
}

fn tracking_rhs(derivative: &mut [f64], state: &[f64], profile: &StiffnessProfile, time: f64) {
    let rate = tracking_rate(profile, time);
    derivative[0] = -rate * (state[0] - time.cos()) - time.sin();
}

fn tracking_jacobian(jacobian: &mut [f64], _: &[f64], profile: &StiffnessProfile, time: f64) {
    jacobian[0] = -tracking_rate(profile, time);
}

fn switching_problem(profile: StiffnessProfile) -> OdeProblem<TrackingRhs, StiffnessProfile> {
    OdeProblem::new(tracking_rhs as TrackingRhs, [1.0], (0.0, 1.0), profile)
        .with_jacobian(tracking_jacobian)
}

fn switching_options() -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(1.0 / 256.0)
        .with_max_step(1.0 / 256.0)
        .with_time_stops([0.5])
        .with_save(SaveMode::Endpoints)
}

fn switching_config(initial_branch: AutomaticBranch) -> AutoSwitchConfig {
    AutoSwitchConfig::new()
        .with_stiffness_thresholds(0.1, 0.2)
        .expect("comparison thresholds must be valid")
        .with_stiff_confirmations(1)
        .expect("comparison confirmation count must be valid")
        .with_nonstiff_confirmations(1)
        .expect("comparison confirmation count must be valid")
        .with_minimum_residence_steps(1)
        .expect("comparison residence count must be valid")
        .with_switch_step_factor(1.0)
        .expect("comparison step factor must be valid")
        .with_initial_branch(initial_branch)
}

fn print_switching_result(name: &str, profile: StiffnessProfile, initial_branch: AutomaticBranch) {
    let algorithm = AutoTsit5::new(Rodas5P)
        .with_switch_config(switching_config(initial_branch))
        .expect("comparison switching configuration must be valid");
    let solution = solve(&switching_problem(profile), algorithm, &switching_options())
        .expect("comparison switching problem must solve");
    let stats = solution.stats();
    let final_branch = match stats.final_automatic_branch {
        Some(AutomaticBranch::NonStiff) => 0,
        Some(AutomaticBranch::Stiff) => 1,
        None => panic!("automatic solve must report its final branch"),
    };
    println!(
        "{name},{:.17e},{},{},{},{},{}",
        solution.last_state()[0],
        stats.rhs_evaluations,
        stats.accepted_steps,
        stats.rejected_steps,
        stats.algorithm_switches,
        final_branch,
    );
}

fn main() {
    let options = SolveOptions::default()
        .with_absolute_tolerance(1.0e-10)
        .with_relative_tolerance(1.0e-10)
        .with_save(SaveMode::Endpoints);
    for (name, endpoint) in [
        (
            "auto_tsit5",
            solve(&problem(), AutoTsit5::new(Rodas5P), &options),
        ),
        (
            "auto_vern6",
            solve(&problem(), AutoVern6::new(Rodas5P), &options),
        ),
        (
            "auto_vern7",
            solve(&problem(), AutoVern7::new(Rodas5P), &options),
        ),
        (
            "auto_vern8",
            solve(&problem(), AutoVern8::new(Rodas5P), &options),
        ),
        (
            "auto_vern9",
            solve(&problem(), AutoVern9::new(Rodas5P), &options),
        ),
        (
            "default_ode_algorithm",
            solve(&problem(), DefaultODEAlgorithm::default(), &options),
        ),
        (
            "default_implicit_ode_algorithm",
            solve(&problem(), DefaultImplicitODEAlgorithm::default(), &options),
        ),
        ("explicit_rk", solve(&problem(), Dp5, &options)),
    ] {
        println!("{name},{:.17e}", endpoint.unwrap().last_state()[0]);
    }

    print_switching_result(
        "switch_nonstiff_to_stiff",
        StiffnessProfile::NonstiffThenStiff,
        AutomaticBranch::NonStiff,
    );
    print_switching_result(
        "switch_stiff_to_nonstiff",
        StiffnessProfile::StiffThenNonstiff,
        AutomaticBranch::Stiff,
    );
}
