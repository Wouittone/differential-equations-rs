use std::any::TypeId;
use std::collections::HashSet;
use std::fmt::Debug;

use differential_equations::algorithms::*;
use differential_equations::*;

fn exercise_public_algorithm<A>(name: &'static str, algorithm: A, identities: &mut HashSet<TypeId>)
where
    A: OdeAlgorithm + Copy + Default + Debug + 'static,
{
    assert!(
        identities.insert(TypeId::of::<A>()),
        "{name} is an alias of another advertised low-storage method"
    );

    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), time: f64| {
            derivative[0] = state[0] + time;
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, algorithm, &options)
        .unwrap_or_else(|error| panic!("{name} failed through the public API: {error}"));
    let endpoint = solution.last_state()[0];
    assert!(
        endpoint.is_finite(),
        "{name} produced a non-finite endpoint"
    );
    assert!(
        (endpoint - 1.0).abs() > f64::EPSILON,
        "{name} did not advance the state"
    );
}

#[test]
fn remaining_low_storage_methods_are_distinct_public_algorithms() {
    let mut identities = HashSet::new();

    macro_rules! exercise {
        ($($algorithm:ident),+ $(,)?) => {
            $(exercise_public_algorithm(stringify!($algorithm), $algorithm, &mut identities);)+
        };
    }

    exercise!(
        RK46NL,
        CFRLDDRK64,
        TSLDDRK74,
        SHLDDRK52,
        SHLDDRK_2N,
        RDPK3Sp35,
        RDPK3Sp49,
        RDPK3Sp510,
        RDPK3SpFSAL35,
        RDPK3SpFSAL49,
        RDPK3SpFSAL510,
        CKLLSRK43_2,
        CKLLSRK54_3C,
        CKLLSRK95_4S,
        CKLLSRK95_4C,
        CKLLSRK95_4M,
        CKLLSRK54_3C_3R,
        CKLLSRK54_3M_3R,
        CKLLSRK54_3N_3R,
        CKLLSRK85_4C_3R,
        CKLLSRK85_4M_3R,
        CKLLSRK85_4P_3R,
        CKLLSRK54_3N_4R,
        CKLLSRK54_3M_4R,
        CKLLSRK65_4M_4R,
        CKLLSRK85_4FM_4R,
        CKLLSRK75_4M_5R,
    );

    assert_eq!(identities.len(), 27);
}
