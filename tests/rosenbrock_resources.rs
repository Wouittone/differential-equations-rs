use differential_equations::solvers::rosenbrock::*;
use differential_equations::tableau::RosenbrockTableau;
use differential_equations::{OdeAlgorithm, OdeProblem, SolveOptions, solve};

use differential_equations as renamed;
use differential_equations::tableau::{
    define_rosenbrock_pair_tableau_from_file, define_rosenbrock_tableau_from_file, load_tableau,
};

define_rosenbrock_tableau_from_file!(pub ORIGINAL_RESOURCE, "Ros2",
    "src/tableau/resources/rosenbrock/ros2.json");
define_rosenbrock_tableau_from_file!(pub RENAMED_RESOURCE, "Ros2",
    "src/tableau/resources/rosenbrock/ros2.json", crate = renamed);
define_rosenbrock_pair_tableau_from_file!(pub PAIR_RESOURCE, "Rosenbrock23/32",
    "src/tableau/resources/rosenbrock/rosenbrock23_32.json");
define_rosenbrock_pair_tableau_from_file!(pub RENAMED_PAIR_RESOURCE, "Rosenbrock23/32",
    "src/tableau/resources/rosenbrock/rosenbrock23_32.json", crate = renamed);

#[test]
fn downstream_definitions_support_original_and_renamed_dependencies() {
    assert_eq!(
        load_tableau(&ORIGINAL_RESOURCE).unwrap(),
        Ros2.tableau().unwrap()
    );
    assert_eq!(
        load_tableau(&RENAMED_RESOURCE).unwrap(),
        Ros2.tableau().unwrap()
    );
    assert_eq!(
        load_tableau(&PAIR_RESOURCE).unwrap(),
        Rosenbrock23.tableau().unwrap()
    );
    assert_eq!(
        load_tableau(&RENAMED_PAIR_RESOURCE).unwrap(),
        Rosenbrock32.tableau().unwrap()
    );
}

#[test]
fn low_order_pair_preserves_the_complete_stage_scheme() {
    let tableau = Rosenbrock23.tableau().unwrap();
    assert!(std::ptr::eq(tableau, Rosenbrock32.tableau().unwrap()));
    assert_eq!(tableau.name(), "Rosenbrock23/32");
    assert_eq!(tableau.orders(), [2, 3]);
    assert_eq!(tableau.nodes(), &[0.0, 0.5, 1.0]);
    assert_eq!(tableau.state()[1], [0.5, 0.0, 0.0]);
    assert_eq!(tableau.derivative()[2][1], 6.0 + 2.0_f64.sqrt());
    assert_eq!(tableau.stage()[2][1], -(6.0 + 2.0_f64.sqrt()));
    assert_eq!(tableau.post_solve()[1], [1.0, 0.0, 0.0]);
    assert_eq!(tableau.second_order(), &[0.0, 1.0, 0.0]);
    assert_eq!(tableau.third_order(), &[1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]);
    assert_eq!(tableau.error(), &[1.0 / 6.0, -2.0 / 6.0, 1.0 / 6.0]);
    assert_eq!(tableau.dense().len(), 2);
}

fn check_resource<A: OdeAlgorithm + Copy>(
    method: A,
    tableau: &'static RosenbrockTableau,
    hash_expected: u64,
    expected: [f64; 16],
    adaptive_supported: bool,
) {
    let mut hash = 0xcbf29ce484222325_u64;
    let gamma = tableau.gamma();
    let zero_error = vec![0.0; tableau.stages()];
    for value in std::iter::once(&gamma)
        .chain(tableau.a().iter().flatten())
        .chain(tableau.coupling().iter().flatten())
        .chain(tableau.c())
        .chain(tableau.d())
        .chain(tableau.b())
        .chain(tableau.btilde().unwrap_or(&zero_error))
        .chain(tableau.h().iter().flatten())
    {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    assert_eq!(
        hash,
        hash_expected,
        "{} coefficients changed",
        tableau.name()
    );
    let mut values = Vec::new();
    for adaptive in [false, true] {
        for span in [(0.0, 0.2), (0.2, 0.0)] {
            let problem = OdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), t: f64| {
                    du[0] = t.sin() - u[0] * u[0] + 0.1 * u[1];
                    du[1] = -2.0 * u[1] + u[0];
                },
                [0.4, 0.2],
                span,
                (),
            );
            let options = SolveOptions::new()
                .with_adaptive(adaptive && adaptive_supported)
                .with_initial_step(0.025)
                .with_tolerances(1e-7, 1e-7)
                .with_dense_output(true);
            let solution = solve(&problem, method, &options).unwrap();
            values.extend_from_slice(solution.last_state());
            values.extend_from_slice(&solution.interpolate(0.1).unwrap());
        }
    }
    for (actual, expected) in values.into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-11,
            "{}: {actual} != {expected}",
            tableau.name()
        );
    }
}

fn check_shapes<A: OdeAlgorithm + Copy>(method: A, adaptive_supported: bool) {
    use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
    for adaptive in [false, true] {
        for span in [(0.0, 0.2), (0.2, 0.0)] {
            for initial in [
                arr0(1.0).into_dyn(),
                array![1.0, 2.0, 3.0].into_dyn(),
                array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
            ] {
                let shaped = OdeProblem::from_array(
                    |mut du: ArrayViewMutD<'_, f64>, u: ArrayViewD<'_, f64>, _: &(), _| {
                        du.zip_mut_with(&u, |du, u| *du = -*u);
                    },
                    initial.clone(),
                    span,
                    (),
                );
                let flat = OdeProblem::new(
                    |du: &mut [f64], u: &[f64], _: &(), _| {
                        for (du, u) in du.iter_mut().zip(u) {
                            *du = -*u;
                        }
                    },
                    initial.iter().copied().collect::<Vec<_>>(),
                    span,
                    (),
                );
                let options = SolveOptions::new()
                    .with_adaptive(adaptive && adaptive_supported)
                    .with_initial_step(0.01)
                    .with_max_step(0.01)
                    .with_tolerances(1e-8, 1e-8)
                    .with_dense_output(true);
                let actual = solve(&shaped, method, &options).unwrap();
                let reference = solve(&flat, method, &options).unwrap();
                assert_eq!(actual.last_state_array().shape(), initial.shape());
                assert_eq!(actual.last_state(), reference.last_state());
                assert_eq!(actual.stats(), reference.stats());
                for (time, sample) in [
                    (span.1, actual.last_state_array().to_owned()),
                    (0.1, actual.interpolate_array(0.1).unwrap()),
                ] {
                    assert_eq!(sample.shape(), initial.shape());
                    for (actual, initial) in sample.iter().zip(initial.iter()) {
                        let expected = initial * (-(time - span.0)).exp();
                        assert!(
                            (actual - expected).abs() < 2e-4 * initial.abs(),
                            "{actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}

macro_rules! regression {
    ($test:ident, $method:ident, $hash:expr, $values:expr, $adaptive:expr) => {
        #[test]
        fn $test() {
            let tableau = $method.tableau().unwrap();
            assert!(std::ptr::eq(tableau, $method.tableau().unwrap()));
            check_resource($method, tableau, $hash, $values, $adaptive);
            check_shapes($method, $adaptive);
        }
    };
}

// Coefficient bit patterns and nonlinear endpoint/dense samples captured from
// the pre-migration kernels. Both directions, fixed and adaptive stepping.
regression!(
    ros2_resource,
    Ros2,
    0xa2dd9820ea50aa58,
    [
        0.39313259562669123,
        0.19871769065200795,
        0.39147060189749827,
        0.19944303192847868,
        0.4080068741308327,
        0.2002034515902629,
        0.3988803479814889,
        0.20026263678747858,
        0.3930153821962115,
        0.19882748170247555,
        0.39140557809265036,
        0.19951710917381466,
        0.40817298813725617,
        0.19990000118922752,
        0.39895485195872166,
        0.20014625016874996
    ],
    true
);
regression!(
    rodas3_resource,
    Rodas3,
    0x6a74a81430c14b07,
    [
        0.39301528342907555,
        0.19882765579286002,
        0.3914055313895811,
        0.1995172202242031,
        0.40817292712206316,
        0.19990020654416055,
        0.39895481796774035,
        0.20014633100473705,
        0.39301534443041924,
        0.19882753330758318,
        0.39140555953589234,
        0.19951714089798916,
        0.4081730052178656,
        0.19989998968258024,
        0.39895485820540866,
        0.20014624775675585
    ],
    true
);
regression!(
    rodas3d_resource,
    Rodas3d,
    0xcc3e99f810ce5485,
    [
        0.3930153013048414,
        0.19882752659132083,
        0.39140554194834554,
        0.19951713593000117,
        0.40817295585898344,
        0.19989995940667274,
        0.39895483033508594,
        0.20014623606415158,
        0.393015352610857,
        0.19882751070176038,
        0.3914055630900361,
        0.1995171275277322,
        0.4081730159121218,
        0.19989995148385398,
        0.39895486431736177,
        0.20014623061176345
    ],
    true
);
regression!(
    ros3_resource,
    Ros3,
    0x5f525e5a41e13cc9,
    [
        0.3930117338787903,
        0.19882817460609262,
        0.39139915460089125,
        0.19952341990273498,
        0.40816349488910986,
        0.1999130797229419,
        0.39895506964730953,
        0.2001455569053985,
        0.39301395237754094,
        0.19882773343423243,
        0.391403425181383,
        0.19951919543750804,
        0.40817165558078383,
        0.19990129767100062,
        0.39895571175304884,
        0.20014476442725695
    ],
    true
);
regression!(
    ros3pr_resource,
    Ros3Pr,
    0x6eccd2e4541ed504,
    [
        0.39301543494513846,
        0.19882817929064878,
        0.3914056235727318,
        0.19951753338883288,
        0.4081730910496032,
        0.19990108100322068,
        0.39895488215196523,
        0.20014669331063834,
        0.3930153785118911,
        0.19882766213826553,
        0.39140558389867347,
        0.19951725720991828,
        0.4081730351278939,
        0.19990013832558723,
        0.39895486898169913,
        0.20014628222469938
    ],
    true
);
regression!(
    ros3prl_resource,
    Ros3Prl,
    0xdce2461aef58c8e5,
    [
        0.393015315486566,
        0.19882770044686135,
        0.3914055452251239,
        0.19951724489328668,
        0.40817296090076693,
        0.1999002778860792,
        0.3989548367611929,
        0.20014636272136246,
        0.39301534305881214,
        0.19882757083225902,
        0.39140555894337287,
        0.19951716054238408,
        0.40817300005531865,
        0.19990005692071952,
        0.39895485519695456,
        0.20014628196224865
    ],
    true
);
regression!(
    ros3prl2_resource,
    Ros3Prl2,
    0xeba89c1aabb3e0b2,
    [
        0.3930153757300852,
        0.19882770691959376,
        0.3914055765279129,
        0.19951724695703843,
        0.408173034105484,
        0.19990026805486955,
        0.3989548715391881,
        0.20014636012250012,
        0.39301536194627834,
        0.1988275692949565,
        0.39140556762204104,
        0.19951715958127023,
        0.40817302402220934,
        0.19990004478631512,
        0.3989548685731154,
        0.2001462745440564
    ],
    true
);
regression!(
    ros3p_resource,
    Ros3p,
    0x615cd34a23ea5aa0,
    [
        0.3930151433845027,
        0.1988281464200292,
        0.3914054721558898,
        0.19951752254742086,
        0.4081727328081083,
        0.1999011321994352,
        0.3989547118224596,
        0.20014670741320637,
        0.39301521569377534,
        0.19882797934545193,
        0.39140547027076583,
        0.19951753321137722,
        0.4081728995303053,
        0.1999004714733521,
        0.39895481531496324,
        0.20014639035705054
    ],
    true
);
regression!(
    ros34prw_resource,
    Ros34Prw,
    0x3d5303a88cff2074,
    [
        0.39301533168792524,
        0.1988277014876281,
        0.39140554512103437,
        0.19951724504109328,
        0.40817298092835635,
        0.19990027345508377,
        0.39895485499320765,
        0.20014636127852478,
        0.39301535001732074,
        0.198827559450546,
        0.3914055595784412,
        0.1995171574354343,
        0.4081730109619507,
        0.19990002714355506,
        0.3989548639631942,
        0.2001462630671129
    ],
    true
);
regression!(
    ros34pw3_resource,
    Ros34Pw3,
    0xb50d7880aaa0e633,
    [
        0.3930153502482897,
        0.19882756877392285,
        0.39140556188831577,
        0.1995171624656103,
        0.40817303156100426,
        0.19989982736919545,
        0.3989548730028537,
        0.20014617888249697,
        0.3930153562966298,
        0.19882751031686483,
        0.39140556437401214,
        0.19951712739865446,
        0.40817302016461604,
        0.19989994942962416,
        0.39895486702158756,
        0.2001462294891495
    ],
    true
);
regression!(
    grk4a_resource,
    Grk4a,
    0x8d69fbf2570cbde,
    [
        0.39301535649886254,
        0.19882750993917508,
        0.39140556441658586,
        0.19951712723508705,
        0.40817301991993993,
        0.19989994995561425,
        0.39895486684038245,
        0.20014622980859903,
        0.39301535723677933,
        0.19882751000889262,
        0.3914055645569034,
        0.1995171277326583,
        0.40817301901187136,
        0.19989994925203158,
        0.3989548660553693,
        0.2001462383292893
    ],
    true
);
regression!(
    grk4t_resource,
    Grk4t,
    0x416883a5d3bdebcb,
    [
        0.39301535640016344,
        0.19882750929048051,
        0.39140556440546226,
        0.1995171267814832,
        0.40817302000653427,
        0.199899951051448,
        0.39895486692358517,
        0.2001462301945447,
        0.3930153569125791,
        0.19882750398551186,
        0.3914055640516497,
        0.1995171627122663,
        0.40817301928881417,
        0.1998999585040752,
        0.3989548659925796,
        0.2001462609232824
    ],
    true
);
regression!(
    rok4a_resource,
    Rok4a,
    0xccec508452164bf7,
    [
        0.3930153542865222,
        0.1988275193730295,
        0.39140556305172475,
        0.19951713341162508,
        0.40817302308470876,
        0.19989993224389507,
        0.3989548681008757,
        0.20014622302557467,
        0.39301535295682327,
        0.19882752574243087,
        0.3914055623966699,
        0.19951714025743275,
        0.40817302440318226,
        0.19989992395458572,
        0.3989548687208823,
        0.20014622331073684
    ],
    true
);
regression!(
    ros34pw1b_resource,
    Ros34Pw1b,
    0x8c6f84ac863ce330,
    [
        0.39301562662652473,
        0.1988277279088008,
        0.39140569441188366,
        0.1995172522180436,
        0.4081733276009387,
        0.199900233513746,
        0.3989550253128247,
        0.2001463519513187,
        0.39301537664216935,
        0.19882752778050178,
        0.3914055785011833,
        0.19951714124842881,
        0.40817304081348066,
        0.19989997064412074,
        0.3989548739495721,
        0.20014623521492397
    ],
    true
);
regression!(
    ros34pw2_resource,
    Ros34Pw2,
    0xf7722bde9ee87a6a,
    [
        0.39301529988075345,
        0.19882769836031297,
        0.3914055285128512,
        0.19951724411279426,
        0.40817294305171625,
        0.19990027807464955,
        0.398954836913807,
        0.20014636241479014,
        0.3930153478687017,
        0.1988275390560493,
        0.3914055593743734,
        0.19951714399548579,
        0.4081730091559825,
        0.19989999692712965,
        0.3989548622669059,
        0.2001462506479471
    ],
    true
);
regression!(
    rodas4_resource,
    Rodas4,
    0xf818ac4c7aea44da,
    [
        0.39301535638102486,
        0.19882750957864215,
        0.39140556441129737,
        0.19951712695367427,
        0.40817302003226513,
        0.1998999505831642,
        0.39895486695323823,
        0.2001462299982836,
        0.3930153567451036,
        0.19882750631142831,
        0.39140560141200265,
        0.1995171240949548,
        0.408173019447487,
        0.19989995463439442,
        0.3989549068983209,
        0.2001462174297601
    ],
    true
);
regression!(
    rodas42_resource,
    Rodas42,
    0xbc14c44b019d5494,
    [
        0.39301535621157124,
        0.1988275100206989,
        0.39140556433466617,
        0.1995171272007623,
        0.40817302025217356,
        0.19989994982851947,
        0.3989548670715727,
        0.2001462296646939,
        0.39301535521212266,
        0.1988275109345218,
        0.39140558251858537,
        0.19951712594377535,
        0.40817302114708814,
        0.19989994820855783,
        0.3989548758524244,
        0.20014622933448561
    ],
    true
);
regression!(
    rodas4p_resource,
    Rodas4P,
    0xccbbb5723ad2d8f1,
    [
        0.39301535628030065,
        0.1988275095715726,
        0.3914055643518558,
        0.1995171269536849,
        0.4081730201525736,
        0.19989995057667886,
        0.3989548670020524,
        0.2001462299996733,
        0.3930153552267988,
        0.19882750418885123,
        0.39140557269842435,
        0.19951718825608275,
        0.40817302104460496,
        0.19989995970131325,
        0.39895487283562325,
        0.20014627937845106
    ],
    true
);
regression!(
    rodas4p2_resource,
    Rodas4P2,
    0x827432ba94c1e43f,
    [
        0.3930153562722534,
        0.19882750957082895,
        0.39140556434724805,
        0.19951712695349874,
        0.40817302016240514,
        0.19989995057569432,
        0.39895486700629246,
        0.2001462299994461,
        0.39301535508800955,
        0.19882750417919734,
        0.39140556295452,
        0.19951714954485056,
        0.40817302124157906,
        0.1998999596839595,
        0.3989548648153483,
        0.20014624413908527
    ],
    true
);
regression!(
    rodas4pw_resource,
    Rodas4PW,
    0xaee242b51e61c5dc,
    [
        0.3930153565414149,
        0.19882750960136722,
        0.39140556449898384,
        0.19951712697066312,
        0.4081730198399739,
        0.1998999505950348,
        0.39895486686398285,
        0.20014623000699805,
        0.3930153597202459,
        0.19882750478165287,
        0.3914055692952244,
        0.1995171624452295,
        0.4081730152699328,
        0.1998999592170224,
        0.39895487136725855,
        0.20014626703858013
    ],
    true
);
regression!(
    rodas5_resource,
    Rodas5,
    0x99dd376021ca8a19,
    [
        0.39301535634285173,
        0.19882750989248496,
        0.3914055643918614,
        0.19951712714791422,
        0.4081730201020056,
        0.19989995008293646,
        0.3989548669844892,
        0.20014622979673752,
        0.3930153590851338,
        0.19882750789706666,
        0.39140555149771056,
        0.1995171390373224,
        0.40817302301112185,
        0.1998999457219972,
        0.3989548784189465,
        0.20014620658908455
    ],
    true
);
regression!(
    rodas5p_resource,
    Rodas5P,
    0x6e5e7c1757831147,
    [
        0.39301535633847967,
        0.1988275098931553,
        0.3914055643892317,
        0.19951712714799763,
        0.40817302009682543,
        0.1998999500851762,
        0.3989548669823854,
        0.20014622979804667,
        0.39301535450811653,
        0.1988275088905124,
        0.3914055608447164,
        0.19951712414614953,
        0.408173015573389,
        0.1998999487441568,
        0.3989548761732434,
        0.2001462307404559
    ],
    true
);
regression!(
    rodas5pe_resource,
    Rodas5Pe,
    0xde3324b4593b868,
    [
        0.39301535633847967,
        0.1988275098931553,
        0.3914055643892317,
        0.19951712714799763,
        0.40817302009682543,
        0.1998999500851762,
        0.3989548669823854,
        0.20014622979804667,
        0.39301535041297075,
        0.19882750682972755,
        0.3914055418102149,
        0.19951712911581188,
        0.40817300636599757,
        0.19989994426582067,
        0.3989549130219389,
        0.2001462142815313
    ],
    true
);
regression!(
    rodas6p_resource,
    Rodas6P,
    0xf05847f1e0f5003,
    [
        0.3930153563390706,
        0.19882750989284934,
        0.3914055643892248,
        0.1995171271476461,
        0.40817302009748097,
        0.19989995008452116,
        0.3989548669830201,
        0.20014622979794872,
        0.3930153443897342,
        0.198827501672137,
        0.3914055571749967,
        0.19951713261947665,
        0.40817300790074423,
        0.19989994252998294,
        0.3989548591740968,
        0.200146241100918
    ],
    true
);
regression!(
    rosenbrockw6s4os_resource,
    RosenbrockW6S4OS,
    0xc2c4c9c73b83090b,
    [
        0.3930153551110604,
        0.19882750911050795,
        0.391405563692533,
        0.199517126727864,
        0.408173021500675,
        0.19989995098700708,
        0.3989548675939492,
        0.2001462302159168,
        0.3930153551110604,
        0.19882750911050795,
        0.391405563692533,
        0.199517126727864,
        0.408173021500675,
        0.19989995098700708,
        0.3989548675939492,
        0.2001462302159168
    ],
    false
);
regression!(
    rodas23w_resource,
    Rodas23W,
    0x6a0f7600cf03cdc5,
    [
        0.39301146734450115,
        0.19883118878715159,
        0.39140352791075766,
        0.1995195968186802,
        0.40817776250559995,
        0.19989327869009074,
        0.39895711530458644,
        0.2001437026421703,
        0.3930142267962289,
        0.1988285752638186,
        0.391398105916129,
        0.19951791116819317,
        0.4081743827870476,
        0.19989803523994448,
        0.39895424705394206,
        0.20014555607301887
    ],
    true
);
regression!(
    rodas3p_resource,
    Rodas3P,
    0xc0c53429bbf5212a,
    [
        0.3930151185264156,
        0.1988276168850949,
        0.3914054521414886,
        0.1995171998977224,
        0.40817273239490914,
        0.19990019416044924,
        0.3989547176474119,
        0.20014632397163878,
        0.3930153183256935,
        0.198827527118225,
        0.39139863927451324,
        0.1995172563886659,
        0.4081729755494679,
        0.1998999873049257,
        0.3989535775591919,
        0.20014629589016886
    ],
    true
);
regression!(
    ros2pr_resource,
    Ros2Pr,
    0xad5cd25d7f15c3ef,
    [
        0.3930123394839658,
        0.19883063588172611,
        0.3914039453119458,
        0.19951921632749722,
        0.40817669236590537,
        0.19989436461535398,
        0.3989565294414148,
        0.20014410943835279,
        0.39301534854793313,
        0.1988275179604374,
        0.3914055603479571,
        0.19951713231123447,
        0.40817302972350794,
        0.19989993540061152,
        0.39895487160073795,
        0.20014622394681453
    ],
    true
);
regression!(
    ros2s_resource,
    Ros2S,
    0x2e488afc44a51d85,
    [
        0.3930111081967073,
        0.1988315154572816,
        0.39140333179902753,
        0.19951981689588125,
        0.40817819723020443,
        0.19989264750150995,
        0.39895732946168305,
        0.20014346298140914,
        0.39301531949231094,
        0.19882754449228335,
        0.39140554586825194,
        0.19951714951589272,
        0.40817306558279287,
        0.19989988602365263,
        0.3989548896281735,
        0.20014620423786644
    ],
    true
);
regression!(
    ros34pw1a_resource,
    Ros34Pw1a,
    0xed89855d1a508f32,
    [
        0.3930157094431728,
        0.19882773505628606,
        0.39140573572255755,
        0.1995172540815398,
        0.4081734244838691,
        0.19990022259282622,
        0.3989550735867339,
        0.2001463494566296,
        0.393015656956027,
        0.19882769841918316,
        0.39140569651903423,
        0.1995172664768597,
        0.4081733272758725,
        0.19990016197682448,
        0.39895500519752886,
        0.20014635331833755
    ],
    true
);
regression!(
    ros4lstab_resource,
    Ros4LStab,
    0xda241ca85537a384,
    [
        0.39301535508068786,
        0.19882751943599353,
        0.39140556357376044,
        0.19951713336809426,
        0.40817302215834156,
        0.19989993228405906,
        0.3989548677736964,
        0.2001462229569597,
        0.393015354265632,
        0.19882752576339408,
        0.3914055631464152,
        0.19951714015565938,
        0.40817302307060893,
        0.19989992421288671,
        0.39895486816699804,
        0.2001462232180744
    ],
    true
);
regression!(
    rosshamp4_resource,
    RosShamp4,
    0x88cbd883be2167e1,
    [
        0.39301535576328417,
        0.19882751416942698,
        0.3914055639851245,
        0.19951712998080337,
        0.40817302102218456,
        0.1998999422502523,
        0.3989548673048236,
        0.20014622683725591,
        0.39301535514970876,
        0.1988275187950194,
        0.39140556360879347,
        0.19951713985052758,
        0.40817302193927546,
        0.19989993410881693,
        0.39895486749333847,
        0.20014622903409687
    ],
    true
);
regression!(
    scholz4_7_resource,
    Scholz4_7,
    0x2ea50b60297ecf8b,
    [
        0.3930154348551589,
        0.19882817928617486,
        0.3914056235643803,
        0.19951753338875047,
        0.4081730911192834,
        0.19990108098911571,
        0.39895488222007974,
        0.20014669330564722,
        0.3930153563454755,
        0.19882750993625384,
        0.3914055643939957,
        0.19951712717302947,
        0.40817302010350137,
        0.19989995015049497,
        0.3989548669845513,
        0.200146229826648
    ],
    true
);
regression!(
    veldd4_resource,
    Veldd4,
    0x172c20fe92e7252d,
    [
        0.3930153563955807,
        0.19882750934141055,
        0.39140556440365526,
        0.19951712681417014,
        0.40817302001366557,
        0.19989995097078292,
        0.39895486692740734,
        0.20014623016328942,
        0.3930153568586433,
        0.198827504586145,
        0.3914055640150384,
        0.19951716178377996,
        0.40817301935559824,
        0.19989995777458344,
        0.3989548660302713,
        0.2001462604640672
    ],
    true
);
regression!(
    velds4_resource,
    Velds4,
    0xcefa6b841075fc7e,
    [
        0.39301535525845127,
        0.19882751453925265,
        0.3914055637929772,
        0.199517130098247,
        0.40817302164771285,
        0.1998999415202971,
        0.3989548676789553,
        0.20014622641626795,
        0.3930153541085197,
        0.19882751938491622,
        0.391405563224113,
        0.19951713997619153,
        0.40817302315058657,
        0.19989993317173244,
        0.3989548682354678,
        0.2001462287980141
    ],
    true
);
regression!(
    rodas5pr_resource,
    Rodas5Pr,
    0x6e5e7c1757831147,
    [
        0.39301535633847967,
        0.1988275098931553,
        0.3914055643892317,
        0.19951712714799763,
        0.40817302009682543,
        0.1998999500851762,
        0.3989548669823854,
        0.20014622979804667,
        0.3930153559981312,
        0.19882750975821567,
        0.39140556501231366,
        0.19951712671840335,
        0.40817301961322106,
        0.1998999498291334,
        0.3989548654237486,
        0.20014623078296281
    ],
    true
);

#[test]
fn residual_controller_shares_the_primary_tableau() {
    assert!(std::ptr::eq(
        Rodas5P.tableau().unwrap(),
        Rodas5Pr.tableau().unwrap()
    ));
    assert_eq!(Scholz4_7.tableau().unwrap().order(), 3);
    assert!(RosenbrockW6S4OS.tableau().unwrap().btilde().is_none());
}

#[test]
fn tsit5da_preserves_the_pinned_tableau_and_ordinary_ode_behavior() {
    use differential_equations::tableau::RosenbrockKind;
    let tableau = Tsit5DA.tableau().unwrap();
    assert!(std::ptr::eq(
        tableau,
        HybridExplicitImplicitRK.tableau().unwrap()
    ));
    assert_eq!(tableau.name(), "Tsit5DA");
    assert_eq!(tableau.kind(), RosenbrockKind::HybridExplicitImplicit);
    assert_eq!(tableau.order(), 5);
    assert_eq!(tableau.stages(), 12);
    assert_eq!(tableau.h().len(), 3);

    fn fingerprint<'a>(values: impl IntoIterator<Item = &'a f64>) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        for value in values {
            for byte in value.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        hash
    }
    // Captured from the Rust ODE kernel before removing its coefficient bank.
    assert_eq!(
        fingerprint(
            tableau
                .a()
                .iter()
                .flatten()
                .chain(tableau.c())
                .chain(tableau.b())
                .chain(tableau.btilde().unwrap())
                .chain(tableau.h().iter().flatten())
        ),
        0xa32e6529762d155a
    );
    // Pinned upstream Gamma (including its diagonal) and time-derivative weights.
    assert_eq!(
        fingerprint(
            std::iter::once(&tableau.gamma())
                .chain(tableau.coupling().iter().flatten())
                .chain(tableau.d())
        ),
        0xda5c78adbb383869
    );

    let expected = [
        [
            0.44962893785374275,
            0.20349528972537959,
            0.3917058678628182,
            0.19896435035942375,
        ],
        [
            0.32693116415116613,
            0.24106550167776616,
            0.32774178781860835,
            0.21784480221072403,
        ],
        [
            0.4496288914634726,
            0.20349528210506365,
            0.39170585720584794,
            0.19896434408935804,
        ],
        [
            0.3269310891469202,
            0.24106551717111876,
            0.32774175572246067,
            0.21784481472099607,
        ],
    ];
    let mut case = 0;
    for adaptive in [false, true] {
        for span in [(0.0, 0.5), (0.5, 0.0)] {
            let problem = OdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), t: f64| {
                    du[0] = t.sin() - u[0] * u[0] + 0.1 * u[1];
                    du[1] = -2.0 * u[1] + u[0];
                },
                [0.4, 0.2],
                span,
                (),
            )
            .with_jacobian(|_, _, _, _| panic!("ordinary Tsit5DA must not request a Jacobian"));
            let options = SolveOptions::new()
                .with_adaptive(adaptive)
                .with_initial_step(0.025)
                .with_tolerances(1e-8, 1e-8)
                .with_dense_output(true);
            let solution = solve(&problem, Tsit5DA, &options).unwrap();
            let sample = solution.interpolate(0.175).unwrap();
            for (actual, expected) in solution
                .last_state()
                .iter()
                .chain(&sample)
                .zip(expected[case])
            {
                assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
            }
            assert_eq!(solution.stats().accepted_steps, [20, 20, 8, 7][case]);
            assert_eq!(solution.stats().rhs_evaluations, [241, 241, 97, 85][case]);
            assert_eq!(solution.stats().rejected_steps, 0);
            assert_eq!(solution.stats().linear_solves, 0);
            assert_eq!(solution.stats().linear_factorizations, 0);
            assert_eq!(solution.stats().jacobian_evaluations, 0);
            case += 1;
        }
    }
    check_shapes(Tsit5DA, true);
}
