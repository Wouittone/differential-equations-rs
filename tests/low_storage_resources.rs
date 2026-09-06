use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
use differential_equations::solvers::explicit::low_storage_rk::*;
use differential_equations::tableau::{
    LowStorageEndpointEvaluation, LowStorageNodePolicy, LowStorageRungeKuttaLayout,
    LowStorageRungeKuttaTableau,
};
use differential_equations::{OdeAlgorithm, OdeProblem, SaveMode, SolveError, SolveOptions, solve};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    hash_raw_bytes(hash, &(bytes.len() as u64).to_le_bytes());
    hash_raw_bytes(hash, bytes);
}

fn hash_raw_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn hash_usize(hash: &mut u64, value: usize) {
    hash_raw_bytes(hash, &(value as u64).to_le_bytes());
}

fn hash_f64(hash: &mut u64, value: f64) {
    hash_raw_bytes(hash, &value.to_bits().to_le_bytes());
}

fn hash_vector(hash: &mut u64, values: &[f64]) {
    hash_usize(hash, values.len());
    for value in values {
        hash_f64(hash, *value);
    }
}

fn fingerprint(hash: &mut u64, tableau: &LowStorageRungeKuttaTableau) {
    hash_bytes(hash, tableau.name().as_bytes());
    hash_bytes(hash, tableau.description().as_bytes());
    hash_usize(hash, tableau.order());
    hash_f64(hash, tableau.consistency_tolerance());
    hash_raw_bytes(
        hash,
        &[match tableau.node_policy() {
            LowStorageNodePolicy::Derived => 0,
            LowStorageNodePolicy::Independent => 1,
        }],
    );
    match tableau.layout() {
        LowStorageRungeKuttaLayout::TwoN(tableau) => {
            hash_raw_bytes(hash, &[0]);
            hash_vector(hash, tableau.a());
            hash_vector(hash, tableau.b());
            hash_vector(hash, tableau.c());
        }
        LowStorageRungeKuttaLayout::TwoC(tableau) => {
            hash_raw_bytes(hash, &[1]);
            hash_vector(hash, tableau.a());
            hash_vector(hash, tableau.b());
            hash_vector(hash, tableau.c());
        }
        LowStorageRungeKuttaLayout::ThreeS(tableau) => {
            hash_raw_bytes(hash, &[2]);
            hash_vector(hash, tableau.gamma1());
            hash_vector(hash, tableau.gamma2());
            hash_vector(hash, tableau.gamma3());
            hash_vector(hash, tableau.delta());
            hash_f64(hash, tableau.beta1());
            hash_vector(hash, tableau.beta2());
            hash_vector(hash, tableau.c());
            hash_raw_bytes(
                hash,
                &[match tableau.endpoint_evaluation() {
                    LowStorageEndpointEvaluation::Omit => 0,
                    LowStorageEndpointEvaluation::Evaluate => 1,
                }],
            );
        }
        LowStorageRungeKuttaLayout::AlternatingTwoN(tableau) => {
            hash_raw_bytes(hash, &[3]);
            for half in [tableau.first(), tableau.second()] {
                hash_vector(hash, half.a());
                hash_vector(hash, half.b());
                hash_vector(hash, half.c());
            }
        }
        LowStorageRungeKuttaLayout::RegisterPipeline(tableau) => {
            hash_raw_bytes(hash, &[4]);
            hash_usize(hash, tableau.history_states());
            hash_usize(hash, tableau.a().len());
            for row in tableau.a() {
                hash_vector(hash, row);
            }
            hash_vector(hash, tableau.b());
            hash_f64(hash, tableau.b_final());
            hash_vector(hash, tableau.c());
        }
    }
}

fn layout_name(tableau: &LowStorageRungeKuttaTableau) -> &'static str {
    match tableau.layout() {
        LowStorageRungeKuttaLayout::TwoN(_) => "two-n",
        LowStorageRungeKuttaLayout::TwoC(_) => "two-c",
        LowStorageRungeKuttaLayout::ThreeS(_) => "three-s",
        LowStorageRungeKuttaLayout::AlternatingTwoN(_) => "alternating-two-n",
        LowStorageRungeKuttaLayout::RegisterPipeline(_) => "register-pipeline",
    }
}

#[test]
fn all_builtin_resources_preserve_metadata_and_coefficient_bits() {
    let mut hash = FNV_OFFSET;
    macro_rules! check {
        ($method:expr, $name:literal, $order:literal, $layout:literal, $stages:expr) => {{
            let tableau = $method.tableau().unwrap();
            assert_eq!(tableau.name(), $name);
            assert_eq!(tableau.order(), $order);
            assert_eq!(layout_name(tableau), $layout);
            assert_eq!(
                (
                    tableau.layout().stages(),
                    tableau.layout().alternate_stages()
                ),
                $stages
            );
            fingerprint(&mut hash, tableau);
        }};
    }

    check!(Ork256, "Ork256", 2, "two-n", (5, None));
    check!(
        CarpenterKennedy2N54,
        "CarpenterKennedy2N54",
        4,
        "two-n",
        (5, None)
    );
    check!(Shlddrk64, "Shlddrk64", 4, "two-n", (6, None));
    check!(Dglddrk73C, "Dglddrk73C", 3, "two-n", (7, None));
    check!(Dglddrk84C, "Dglddrk84C", 4, "two-n", (8, None));
    check!(Dglddrk84F, "Dglddrk84F", 4, "two-n", (8, None));
    check!(Ndblsrk124, "Ndblsrk124", 4, "two-n", (12, None));
    check!(Ndblsrk134, "Ndblsrk134", 4, "two-n", (13, None));
    check!(Ndblsrk144, "Ndblsrk144", 4, "two-n", (14, None));
    check!(RK46NL, "RK46NL", 4, "two-n", (6, None));
    check!(SHLDDRK52, "SHLDDRK52", 2, "two-n", (5, None));
    check!(CFRLDDRK64, "CFRLDDRK64", 4, "two-c", (6, None));
    check!(TSLDDRK74, "TSLDDRK74", 4, "two-c", (7, None));
    check!(
        SHLDDRK_2N,
        "SHLDDRK_2N",
        4,
        "alternating-two-n",
        (5, Some(6))
    );
    check!(
        ParsaniKetchesonDeconinck3S32,
        "ParsaniKetchesonDeconinck3S32",
        2,
        "three-s",
        (3, None)
    );
    check!(
        ParsaniKetchesonDeconinck3S53,
        "ParsaniKetchesonDeconinck3S53",
        3,
        "three-s",
        (5, None)
    );
    check!(
        ParsaniKetchesonDeconinck3S82,
        "ParsaniKetchesonDeconinck3S82",
        2,
        "three-s",
        (8, None)
    );
    check!(
        ParsaniKetchesonDeconinck3S173,
        "ParsaniKetchesonDeconinck3S173",
        3,
        "three-s",
        (17, None)
    );
    check!(
        ParsaniKetchesonDeconinck3S184,
        "ParsaniKetchesonDeconinck3S184",
        4,
        "three-s",
        (18, None)
    );
    check!(
        ParsaniKetchesonDeconinck3S94,
        "ParsaniKetchesonDeconinck3S94",
        4,
        "three-s",
        (9, None)
    );
    check!(
        ParsaniKetchesonDeconinck3S105,
        "ParsaniKetchesonDeconinck3S105",
        5,
        "three-s",
        (10, None)
    );
    check!(
        ParsaniKetchesonDeconinck3S205,
        "ParsaniKetchesonDeconinck3S205",
        5,
        "three-s",
        (20, None)
    );
    check!(RDPK3Sp35, "RDPK3Sp35", 3, "three-s", (5, None));
    check!(RDPK3Sp49, "RDPK3Sp49", 4, "three-s", (9, None));
    check!(RDPK3Sp510, "RDPK3Sp510", 5, "three-s", (10, None));
    check!(RDPK3SpFSAL35, "RDPK3SpFSAL35", 3, "three-s", (5, None));
    check!(RDPK3SpFSAL49, "RDPK3SpFSAL49", 4, "three-s", (9, None));
    check!(RDPK3SpFSAL510, "RDPK3SpFSAL510", 5, "three-s", (10, None));
    check!(
        CKLLSRK43_2,
        "CKLLSRK43_2",
        3,
        "register-pipeline",
        (4, None)
    );
    check!(
        CKLLSRK54_3C,
        "CKLLSRK54_3C",
        4,
        "register-pipeline",
        (5, None)
    );
    check!(
        CKLLSRK95_4S,
        "CKLLSRK95_4S",
        5,
        "register-pipeline",
        (9, None)
    );
    check!(
        CKLLSRK95_4C,
        "CKLLSRK95_4C",
        5,
        "register-pipeline",
        (9, None)
    );
    check!(
        CKLLSRK95_4M,
        "CKLLSRK95_4M",
        5,
        "register-pipeline",
        (9, None)
    );
    check!(
        CKLLSRK54_3C_3R,
        "CKLLSRK54_3C_3R",
        4,
        "register-pipeline",
        (5, None)
    );
    check!(
        CKLLSRK54_3M_3R,
        "CKLLSRK54_3M_3R",
        4,
        "register-pipeline",
        (5, None)
    );
    check!(
        CKLLSRK54_3N_3R,
        "CKLLSRK54_3N_3R",
        4,
        "register-pipeline",
        (5, None)
    );
    check!(
        CKLLSRK85_4C_3R,
        "CKLLSRK85_4C_3R",
        5,
        "register-pipeline",
        (8, None)
    );
    check!(
        CKLLSRK85_4M_3R,
        "CKLLSRK85_4M_3R",
        5,
        "register-pipeline",
        (8, None)
    );
    check!(
        CKLLSRK85_4P_3R,
        "CKLLSRK85_4P_3R",
        5,
        "register-pipeline",
        (8, None)
    );
    check!(
        CKLLSRK54_3N_4R,
        "CKLLSRK54_3N_4R",
        4,
        "register-pipeline",
        (5, None)
    );
    check!(
        CKLLSRK54_3M_4R,
        "CKLLSRK54_3M_4R",
        4,
        "register-pipeline",
        (5, None)
    );
    check!(
        CKLLSRK65_4M_4R,
        "CKLLSRK65_4M_4R",
        5,
        "register-pipeline",
        (6, None)
    );
    check!(
        CKLLSRK85_4FM_4R,
        "CKLLSRK85_4FM_4R",
        5,
        "register-pipeline",
        (8, None)
    );
    check!(
        CKLLSRK75_4M_5R,
        "CKLLSRK75_4M_5R",
        5,
        "register-pipeline",
        (7, None)
    );

    assert_eq!(hash, 0x9742_b4f4_f32b_ca01);
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints)
}

fn assert_rhs_count<A: OdeAlgorithm>(algorithm: A, expected: usize) {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time| du[0] = u[0] + time,
        [1.0],
        (0.0, 0.1),
        (),
    );
    let solution = solve(&problem, algorithm, &fixed_options(0.1)).unwrap();
    assert_eq!(solution.stats().accepted_steps, 1);
    assert_eq!(solution.stats().rhs_evaluations, expected);
}

#[test]
fn recurrence_layouts_preserve_endpoint_evaluation_semantics() {
    assert_rhs_count(Ork256, 5);
    assert_rhs_count(CFRLDDRK64, 6);
    assert_rhs_count(ParsaniKetchesonDeconinck3S32, 4);
    assert_rhs_count(RDPK3Sp35, 5);
    assert_rhs_count(RDPK3SpFSAL35, 6);
    assert_rhs_count(CKLLSRK43_2, 5);

    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| du[0] = -u[0],
        [1.0],
        (0.0, 0.2),
        (),
    );
    let solution = solve(&problem, SHLDDRK_2N, &fixed_options(0.1)).unwrap();
    assert_eq!(solution.stats().accepted_steps, 2);
    assert_eq!(solution.stats().rhs_evaluations, 13);
}

fn solve_shapes<A: OdeAlgorithm + Copy>(algorithm: A) {
    for initial in [
        arr0(1.0).into_dyn(),
        array![1.0, 2.0].into_dyn(),
        array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
    ] {
        let shape = initial.shape().to_vec();
        let problem = OdeProblem::from_array(
            |mut du: ArrayViewMutD<'_, f64>, u: ArrayViewD<'_, f64>, _: &(), _| {
                du.zip_mut_with(&u, |du, u| *du = -*u);
            },
            initial.clone(),
            (0.0, 0.1),
            (),
        );
        let solution = solve(&problem, algorithm, &fixed_options(0.001)).unwrap();
        assert_eq!(solution.state_shape(), shape);
        for (actual, initial) in solution.last_state().iter().zip(initial) {
            assert!((actual - initial * (-0.1_f64).exp()).abs() < 1.0e-7);
        }
    }
}

#[test]
fn every_recurrence_layout_supports_scalar_vector_and_matrix_states() {
    solve_shapes(Ork256);
    solve_shapes(CFRLDDRK64);
    solve_shapes(ParsaniKetchesonDeconinck3S32);
    solve_shapes(SHLDDRK_2N);
    solve_shapes(CKLLSRK43_2);
}

#[test]
fn fixed_only_resources_reject_adaptive_configuration() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| du[0] = -u[0],
        [1.0],
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        solve(&problem, Ork256, &SolveOptions::default()).unwrap_err(),
        SolveError::AdaptiveStepUnsupported
    );
}
