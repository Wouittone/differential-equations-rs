use differential_equations::ndarray::{
    ArrayView0, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut0, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMutD, arr0, array,
};
use differential_equations::solvers::explicit::{
    CFRLDDRK64, CKLLSRK43_2, CKLLSRK54_3C, CKLLSRK54_3C_3R, CKLLSRK54_3M_3R, CKLLSRK54_3M_4R,
    CKLLSRK54_3N_3R, CKLLSRK54_3N_4R, CKLLSRK65_4M_4R, CKLLSRK75_4M_5R, CKLLSRK85_4C_3R,
    CKLLSRK85_4FM_4R, CKLLSRK85_4M_3R, CKLLSRK85_4P_3R, CKLLSRK95_4C, CKLLSRK95_4M, CKLLSRK95_4S,
    CarpenterKennedy2N54, Dglddrk73C, Dglddrk84C, Dglddrk84F, Ndblsrk124, Ndblsrk134, Ndblsrk144,
    Ork256, ParsaniKetchesonDeconinck3S32, ParsaniKetchesonDeconinck3S53,
    ParsaniKetchesonDeconinck3S82, ParsaniKetchesonDeconinck3S94, ParsaniKetchesonDeconinck3S105,
    ParsaniKetchesonDeconinck3S173, ParsaniKetchesonDeconinck3S184, ParsaniKetchesonDeconinck3S205,
    RDPK3Sp35, RDPK3Sp49, RDPK3Sp510, RDPK3SpFSAL35, RDPK3SpFSAL49, RDPK3SpFSAL510, RK46NL,
    SHLDDRK_2N, SHLDDRK52, Shlddrk64, TSLDDRK74,
};
use differential_equations::tableau::{
    LowStorageAdaptiveController, LowStorageEndpointEvaluation, LowStorageNodePolicy,
    LowStorageRungeKuttaLayout, LowStorageRungeKuttaTableau,
};
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveError, SolveOptions, solve,
};

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
    match tableau.embedded() {
        None => hash_raw_bytes(hash, &[0]),
        Some(embedded) => {
            hash_raw_bytes(hash, &[1]);
            hash_usize(hash, embedded.order());
            hash_vector(hash, embedded.error());
            match embedded.controller() {
                LowStorageAdaptiveController::StandardPi => hash_raw_bytes(hash, &[0]),
                LowStorageAdaptiveController::Pid(controller) => {
                    hash_raw_bytes(hash, &[1]);
                    for beta in controller.beta() {
                        hash_f64(hash, beta);
                    }
                    hash_f64(hash, controller.acceptance_safety());
                }
            }
        }
    }
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

    assert_eq!(hash, 0x978d_aaef_4d63_05eb);
}

#[test]
fn embedded_family_metadata_has_expected_semantics() {
    macro_rules! check_rdpk {
        ($method:expr, $fsal:literal) => {{
            let tableau = $method.tableau().unwrap();
            let embedded = tableau.embedded().unwrap();
            assert_eq!(embedded.order() + 1, tableau.order());
            assert_eq!(tableau.fsal(), $fsal);
            assert_eq!(
                embedded.error().len(),
                tableau.layout().stages() + usize::from($fsal)
            );
            assert!(matches!(
                embedded.controller(),
                LowStorageAdaptiveController::Pid(_)
            ));
        }};
    }
    check_rdpk!(RDPK3Sp35, false);
    check_rdpk!(RDPK3Sp49, false);
    check_rdpk!(RDPK3Sp510, false);
    check_rdpk!(RDPK3SpFSAL35, true);
    check_rdpk!(RDPK3SpFSAL49, true);
    check_rdpk!(RDPK3SpFSAL510, true);

    macro_rules! check_ckll {
        ($method:expr) => {{
            let tableau = $method.tableau().unwrap();
            let embedded = tableau.embedded().unwrap();
            assert_eq!(embedded.order() + 1, tableau.order());
            assert!(tableau.fsal());
            assert_eq!(embedded.error().len(), tableau.layout().stages());
            assert_eq!(
                embedded.controller(),
                LowStorageAdaptiveController::StandardPi
            );
        }};
    }
    check_ckll!(CKLLSRK43_2);
    check_ckll!(CKLLSRK54_3C);
    check_ckll!(CKLLSRK95_4S);
    check_ckll!(CKLLSRK95_4C);
    check_ckll!(CKLLSRK95_4M);
    check_ckll!(CKLLSRK54_3C_3R);
    check_ckll!(CKLLSRK54_3M_3R);
    check_ckll!(CKLLSRK54_3N_3R);
    check_ckll!(CKLLSRK85_4C_3R);
    check_ckll!(CKLLSRK85_4M_3R);
    check_ckll!(CKLLSRK85_4P_3R);
    check_ckll!(CKLLSRK54_3N_4R);
    check_ckll!(CKLLSRK54_3M_4R);
    check_ckll!(CKLLSRK65_4M_4R);
    check_ckll!(CKLLSRK85_4FM_4R);
    check_ckll!(CKLLSRK75_4M_5R);
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
    assert_eq!(solution.stats().rhs_evaluations, 12);
}

fn assert_adaptive_solution<A: OdeAlgorithm>(algorithm: A) {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| du[0] = -u[0],
        [1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_tolerances(1.0e-9, 1.0e-9)
        .with_save(SaveMode::Endpoints);
    let solution = solve(&problem, algorithm, &options).unwrap();
    assert!(solution.stats().accepted_steps > 0);
    let error = (solution.last_state()[0] - (-1.0_f64).exp()).abs();
    assert!(
        error < 1.0e-5,
        "{} produced error {error:e} after {} accepted and {} rejected steps",
        std::any::type_name::<A>(),
        solution.stats().accepted_steps,
        solution.stats().rejected_steps,
    );
}

#[test]
fn every_embedded_low_storage_method_supports_adaptive_stepping() {
    assert_adaptive_solution(RDPK3Sp35);
    assert_adaptive_solution(RDPK3Sp49);
    assert_adaptive_solution(RDPK3Sp510);
    assert_adaptive_solution(RDPK3SpFSAL35);
    assert_adaptive_solution(RDPK3SpFSAL49);
    assert_adaptive_solution(RDPK3SpFSAL510);
    assert_adaptive_solution(CKLLSRK43_2);
    assert_adaptive_solution(CKLLSRK54_3C);
    assert_adaptive_solution(CKLLSRK95_4S);
    assert_adaptive_solution(CKLLSRK95_4C);
    assert_adaptive_solution(CKLLSRK95_4M);
    assert_adaptive_solution(CKLLSRK54_3C_3R);
    assert_adaptive_solution(CKLLSRK54_3M_3R);
    assert_adaptive_solution(CKLLSRK54_3N_3R);
    assert_adaptive_solution(CKLLSRK85_4C_3R);
    assert_adaptive_solution(CKLLSRK85_4M_3R);
    assert_adaptive_solution(CKLLSRK85_4P_3R);
    assert_adaptive_solution(CKLLSRK54_3N_4R);
    assert_adaptive_solution(CKLLSRK54_3M_4R);
    assert_adaptive_solution(CKLLSRK65_4M_4R);
    assert_adaptive_solution(CKLLSRK85_4FM_4R);
    assert_adaptive_solution(CKLLSRK75_4M_5R);
}

#[test]
fn fsal_reuses_endpoint_derivatives_and_callbacks_invalidate_them() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time| du[0] = u[0] + time,
        [1.0],
        (0.0, 0.2),
        (),
    );
    let options = fixed_options(0.1);
    assert_eq!(
        solve(&problem, RDPK3Sp35, &options)
            .unwrap()
            .stats()
            .rhs_evaluations,
        10
    );
    assert_eq!(
        solve(&problem, RDPK3SpFSAL35, &options)
            .unwrap()
            .stats()
            .rhs_evaluations,
        11
    );
    assert_eq!(
        solve(&problem, CKLLSRK43_2, &options)
            .unwrap()
            .stats()
            .rhs_evaluations,
        9
    );

    let observation_problem =
        problem.with_preset_time_callback([0.1], |_, _, _| CallbackAction::ContinueUnmodified);
    assert_eq!(
        solve(&observation_problem, CKLLSRK43_2, &options)
            .unwrap()
            .stats()
            .rhs_evaluations,
        9
    );

    let callback_problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time| du[0] = u[0] + time,
        [1.0],
        (0.0, 0.2),
        (),
    )
    .with_preset_time_callback([0.1], |state, _, _| {
        state[0] += 1.0;
        CallbackAction::Continue
    });
    assert_eq!(
        solve(&callback_problem, CKLLSRK43_2, &options)
            .unwrap()
            .stats()
            .rhs_evaluations,
        10
    );
}

#[test]
fn continuous_root_truncation_invalidates_the_attempted_endpoint_derivative() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time| du[0] = u[0] + time,
        [1.0],
        (0.0, 0.15),
        (),
    )
    .with_continuous_callback(
        |_, _, time| time - 0.05,
        |_, _, _| CallbackAction::ContinueUnmodified,
    );
    let solution = solve(&problem, CKLLSRK43_2, &fixed_options(0.1)).unwrap();
    assert_eq!(solution.stats().accepted_steps, 2);
    assert_eq!(solution.stats().callback_invocations, 1);
    assert_eq!(solution.stats().rhs_evaluations, 10);
}

#[test]
fn dense_retention_and_save_at_reuse_cached_endpoint_derivatives() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time| du[0] = u[0] + time,
        [1.0],
        (0.0, 0.2),
        (),
    );
    let baseline = solve(&problem, CKLLSRK43_2, &fixed_options(0.1)).unwrap();
    let dense = solve(
        &problem,
        CKLLSRK43_2,
        &fixed_options(0.1).with_dense_output(true),
    )
    .unwrap();
    let sampled = solve(
        &problem,
        CKLLSRK43_2,
        &fixed_options(0.1).with_save_at([0.05, 0.15]),
    )
    .unwrap();
    assert_eq!(baseline.stats().rhs_evaluations, 9);
    assert_eq!(
        dense.stats().rhs_evaluations,
        baseline.stats().rhs_evaluations
    );
    assert_eq!(
        sampled.stats().rhs_evaluations,
        baseline.stats().rhs_evaluations
    );
    for &time in &[0.05, 0.15] {
        let index = sampled
            .times()
            .iter()
            .position(|saved| *saved == time)
            .expect("requested sample must be saved");
        let exact = 2.0 * time.exp() - time - 1.0;
        assert!((sampled.state(index).unwrap()[0] - exact).abs() < 2.0e-4);
    }

    let plain_non_fsal = solve(&problem, RDPK3Sp35, &fixed_options(0.1)).unwrap();
    let dense_non_fsal = solve(
        &problem,
        RDPK3Sp35,
        &fixed_options(0.1).with_dense_output(true),
    )
    .unwrap();
    assert_eq!(plain_non_fsal.stats().rhs_evaluations, 10);
    assert_eq!(dense_non_fsal.stats().rhs_evaluations, 11);
}

#[test]
fn rejected_register_pipeline_attempts_reuse_the_start_derivative() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| du[0] = u[0],
        [1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_initial_step(1.0)
        .with_tolerances(1.0e-12, 1.0e-12)
        .with_save(SaveMode::Endpoints);
    let solution = solve(&problem, CKLLSRK43_2, &options).unwrap();
    let stats = solution.stats();
    assert!(stats.rejected_steps > 0);
    assert_eq!(
        stats.rhs_evaluations,
        1 + 4 * (stats.accepted_steps + stats.rejected_steps)
    );
}

#[test]
fn rejected_three_s_pid_attempts_reuse_the_start_derivative() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| du[0] = u[0],
        [1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_initial_step(1.0)
        .with_tolerances(1.0e-12, 1.0e-12)
        .with_save(SaveMode::Endpoints);
    let solution = solve(&problem, RDPK3SpFSAL35, &options).unwrap();
    let stats = solution.stats();
    assert!(stats.rejected_steps > 0);
    assert_eq!(
        stats.rhs_evaluations,
        1 + 5 * (stats.accepted_steps + stats.rejected_steps)
    );
}

#[test]
fn tighter_tolerances_increase_adaptive_work_and_reduce_error() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| du[0] = u[0],
        [1.0],
        (0.0, 2.0),
        (),
    );
    let loose = solve(
        &problem,
        RDPK3SpFSAL510,
        &SolveOptions::new()
            .with_tolerances(1.0e-4, 1.0e-4)
            .with_save(SaveMode::Endpoints),
    )
    .unwrap();
    let tight = solve(
        &problem,
        RDPK3SpFSAL510,
        &SolveOptions::new()
            .with_tolerances(1.0e-10, 1.0e-10)
            .with_save(SaveMode::Endpoints),
    )
    .unwrap();
    let exact = 2.0_f64.exp();
    let loose_error = (loose.last_state()[0] - exact).abs();
    let tight_error = (tight.last_state()[0] - exact).abs();
    assert!(tight.stats().accepted_steps > loose.stats().accepted_steps);
    assert!(
        tight_error < loose_error,
        "tight={tight_error:e}, loose={loose_error:e}"
    );
}

#[test]
fn tighter_tolerances_increase_ckll_work_and_reduce_error() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| du[0] = u[0],
        [1.0],
        (0.0, 2.0),
        (),
    );
    let loose = solve(
        &problem,
        CKLLSRK95_4M,
        &SolveOptions::new()
            .with_tolerances(1.0e-4, 1.0e-4)
            .with_save(SaveMode::Endpoints),
    )
    .unwrap();
    let tight = solve(
        &problem,
        CKLLSRK95_4M,
        &SolveOptions::new()
            .with_tolerances(1.0e-10, 1.0e-10)
            .with_save(SaveMode::Endpoints),
    )
    .unwrap();
    let exact = 2.0_f64.exp();
    let loose_error = (loose.last_state()[0] - exact).abs();
    let tight_error = (tight.last_state()[0] - exact).abs();
    assert!(tight.stats().accepted_steps > loose.stats().accepted_steps);
    assert!(
        tight_error < loose_error,
        "tight={tight_error:e}, loose={loose_error:e}"
    );
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

fn solve_adaptive_shapes<A: OdeAlgorithm + Copy>(algorithm: A) {
    let options = SolveOptions::new()
        .with_tolerances(1.0e-9, 1.0e-9)
        .with_save(SaveMode::Endpoints);
    let scalar = OdeProblem::from_array(
        |mut du: ArrayViewMut0<'_, f64>, u: ArrayView0<'_, f64>, _: &(), _| {
            du[[]] = -u[[]];
        },
        arr0(1.0),
        (0.0, 0.1),
        (),
    );
    let vector = OdeProblem::from_array(
        |mut du: ArrayViewMut1<'_, f64>, u: ArrayView1<'_, f64>, _: &(), _| {
            du.zip_mut_with(&u, |du, u| *du = -*u);
        },
        array![1.0, 1.0],
        (0.0, 0.1),
        (),
    );
    let matrix = OdeProblem::from_array(
        |mut du: ArrayViewMut2<'_, f64>, u: ArrayView2<'_, f64>, _: &(), _| {
            du.zip_mut_with(&u, |du, u| *du = -*u);
        },
        array![[1.0, 1.0], [1.0, 1.0]],
        (0.0, 0.1),
        (),
    );

    let scalar_solution = solve(&scalar, algorithm, &options).unwrap();
    let vector_solution = solve(&vector, algorithm, &options).unwrap();
    let matrix_solution = solve(&matrix, algorithm, &options).unwrap();
    assert!(scalar_solution.state_shape().is_empty());
    assert_eq!(vector_solution.state_shape(), &[2]);
    assert_eq!(matrix_solution.state_shape(), &[2, 2]);
    let expected = (-0.1_f64).exp();
    for actual in scalar_solution
        .last_state()
        .iter()
        .chain(vector_solution.last_state())
        .chain(matrix_solution.last_state())
    {
        assert!((*actual - expected).abs() < 1.0e-7);
    }
}

#[test]
fn adaptive_low_storage_layouts_support_scalar_vector_and_matrix_states() {
    solve_adaptive_shapes(RDPK3SpFSAL510);
    solve_adaptive_shapes(CKLLSRK54_3M_3R);
}

#[test]
fn adaptive_low_storage_methods_integrate_backward() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| du[0] = -u[0],
        [(-1.0_f64).exp()],
        (1.0, 0.0),
        (),
    );
    let options = SolveOptions::new()
        .with_tolerances(1.0e-9, 1.0e-9)
        .with_save(SaveMode::Endpoints);
    for endpoint in [
        solve(&problem, RDPK3SpFSAL510, &options)
            .unwrap()
            .last_state()[0],
        solve(&problem, CKLLSRK95_4M, &options)
            .unwrap()
            .last_state()[0],
    ] {
        assert!((endpoint - 1.0).abs() < 1.0e-7);
    }
}

#[test]
fn adaptive_low_storage_handles_nonautonomous_forward_and_backward() {
    let exact = |time: f64| 2.0 * time.exp() - time - 1.0;
    let forward = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time| du[0] = u[0] + time,
        [exact(0.0)],
        (0.0, 1.0),
        (),
    );
    let backward = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time| du[0] = u[0] + time,
        [exact(1.0)],
        (1.0, 0.0),
        (),
    );
    let options = SolveOptions::new()
        .with_tolerances(1.0e-9, 1.0e-9)
        .with_save(SaveMode::Endpoints);
    macro_rules! check {
        ($method:expr) => {{
            let forward_solution = solve(&forward, $method, &options).unwrap();
            let backward_solution = solve(&backward, $method, &options).unwrap();
            assert!((forward_solution.last_state()[0] - exact(1.0)).abs() < 1.0e-6);
            assert!((backward_solution.last_state()[0] - exact(0.0)).abs() < 1.0e-6);
        }};
    }
    check!(RDPK3Sp35);
    check!(RDPK3SpFSAL510);
    check!(CKLLSRK43_2);
    check!(CKLLSRK95_4M);
}

#[test]
fn repeated_adaptive_solves_do_not_leak_controller_or_cache_state() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time| du[0] = u[0] + time,
        [1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_initial_step(0.2)
        .with_tolerances(1.0e-9, 1.0e-9)
        .with_save(SaveMode::EveryStep);
    macro_rules! check {
        ($method:expr) => {{
            let first = solve(&problem, $method, &options).unwrap();
            let second = solve(&problem, $method, &options).unwrap();
            assert_eq!(first.times(), second.times());
            assert_eq!(first.values(), second.values());
            assert_eq!(first.stats(), second.stats());
        }};
    }
    check!(RDPK3SpFSAL510);
    check!(CKLLSRK95_4M);
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
