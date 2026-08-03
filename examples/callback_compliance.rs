use differential_equations::{CallbackAction, OdeProblem, Rk4, SaveMode, SolveOptions, solve};

fn main() {
    let event_problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        vec![0.0],
        (0.0, 2.0),
        (),
    )
    .with_continuous_callback(
        |state, _: &(), _| state[0] - 0.75,
        |state, _: &(), _| {
            state[0] = 42.0;
            CallbackAction::Terminate
        },
    );
    let event_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.5),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let event = solve(&event_problem, Rk4, &event_options).expect("event solve failed");
    println!(
        "event,{:.17e},{:.17e}",
        event.times().last().unwrap(),
        event.last_state()[0]
    );

    let discrete_problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(
        |state, _: &(), time| time >= 0.5 && state[0] < 5.0,
        |state, _: &(), _| {
            state[0] += 10.0;
            CallbackAction::Continue
        },
    );
    let discrete_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let discrete =
        solve(&discrete_problem, Rk4, &discrete_options).expect("discrete callback solve failed");
    println!(
        "discrete,{:.17e},{:.17e}",
        discrete.times().last().unwrap(),
        discrete.last_state()[0]
    );

    let save_problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    );
    let save_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.3),
        save_at: vec![0.2, 0.5, 0.8],
        ..SolveOptions::default()
    };
    let saved = solve(&save_problem, Rk4, &save_options).expect("save-at solve failed");
    print!("save_at");
    for (&time, state) in saved.times().iter().zip(saved.values()) {
        print!(",{time:.17e},{state:.17e}");
    }
    println!();
}
