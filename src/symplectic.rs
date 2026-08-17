//! Explicit symplectic composition methods for partitioned second-order problems.
//!
//! The coefficient vectors are pinned copies of the `SymplecticTableau` data in
//! OrdinaryDiffEqSymplecticRK.  A stage is a drift of the position by `aᵢ`
//! followed by a kick of the velocity by `bᵢ`.

use differential_equations::{SaveMode, SecondOrderOdeProblem, SolveError, SolveOptions};

/// A pinned alternating drift/kick composition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymplecticTableau {
    /// Position (drift) coefficients.
    pub a: &'static [f64],
    /// Velocity (kick) coefficients.
    pub b: &'static [f64],
}

impl SymplecticTableau {
    /// Creates a validated view of a pinned composition.
    pub const fn new(a: &'static [f64], b: &'static [f64]) -> Self {
        Self { a, b }
    }

    /// Number of alternating stages.
    pub const fn stages(self) -> usize {
        self.a.len()
    }
}

/// A named explicit symplectic composition.
pub trait SymplecticAlgorithm: Copy {
    /// Returns the pinned composition coefficients.
    fn tableau() -> SymplecticTableau;
}

macro_rules! symplectic_algorithm {
    ($name:ident, $a:expr, $b:expr) => {
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl $name {
            /// Returns this method's pinned composition coefficients.
            pub const fn tableau() -> SymplecticTableau {
                SymplecticTableau::new($a, $b)
            }
        }

        impl SymplecticAlgorithm for $name {
            fn tableau() -> SymplecticTableau {
                Self::tableau()
            }
        }
    };
}

const MCATE2_A: &[f64] = &[0.7071067811865476, 0.2928932188134524];
const MCATE2_B: &[f64] = &[0.29289321881345254, 0.7071067811865475];

const MCATE3_A: &[f64] = &[0.9196615230173999, -0.18799161879915982, 0.2683300957817599];
const MCATE3_B: &[f64] = &[0.2683300957817599, -0.18799161879915982, 0.9196615230173999];

const CANDY_ROZ_A: &[f64] = &[
    0.6756035959798289,
    -0.17560359597982886,
    -0.17560359597982886,
    0.6756035959798289,
];
const CANDY_ROZ_B: &[f64] = &[
    0.0,
    1.3512071919596578,
    -1.7024143839193155,
    1.3512071919596578,
];

const MCATE4_A: &[f64] = &[
    0.515352837431122936,
    -0.085782019412973646,
    0.441583023616466524,
    0.128846158365384185,
];
const MCATE4_B: &[f64] = &[
    0.134496199277431089,
    -0.224819803079420806,
    0.756320000515668291,
    0.334003603286321425,
];

const CALVO_SANZ4_A: &[f64] = &[
    0.20517766154229,
    0.40302128160421,
    -0.12092087633891,
    0.51272193319241,
    0.0,
];
const CALVO_SANZ4_B: &[f64] = &[
    0.061758858135626,
    0.33897802655364,
    0.61479130717558,
    -0.14054801465937,
    0.12501982279453,
];

const MCATE42_A: &[f64] = &[
    0.40518861839525227722,
    -0.287144040816524089,
    0.76391084484254362356,
    -0.287144040816524089,
    0.40518861839525227722,
];
const MCATE42_B: &[f64] = &[
    -3.0 / 73.0,
    17.0 / 59.0,
    1.0 - 2.0 * (-3.0 / 73.0) - 2.0 * (17.0 / 59.0),
    17.0 / 59.0,
    -3.0 / 73.0,
];

const MCATE5_A: &[f64] = &[
    0.33983962583911,
    -0.088601336903027329,
    0.5858564768259621188,
    -0.603039356536491888,
    0.3235807965546976394,
    0.4423637942197494587,
];
const MCATE5_B: &[f64] = &[
    0.1193900292875672758,
    0.6989273703824752308,
    -0.1713123582716007754,
    0.401269502251353448,
    0.010705081848235984,
    -0.0589796254980311632,
];

const YOSHIDA6_A: &[f64] = &[
    0.78451361047756,
    0.23557321335936,
    -1.1776799841789,
    1.3151863206839,
    -1.1776799841789,
    0.23557321335936,
    0.78451361047756,
    0.0,
];
const YOSHIDA6_B: &[f64] = &[
    0.39225680523878,
    0.51004341191846,
    -0.47105338540977,
    0.0687531682525,
    0.0687531682525,
    -0.47105338540977,
    0.51004341191846,
    0.39225680523878,
];

const KAHAN_LI6_A: &[f64] = &[
    0.39216144400731413927925056,
    0.33259913678935943859974864,
    -0.70624617255763935980996482,
    0.08221359629355080023149045,
    0.79854399093482996339895035,
    0.08221359629355080023149045,
    -0.70624617255763935980996482,
    0.33259913678935943859974864,
    0.39216144400731413927925056,
    0.0,
];
const KAHAN_LI6_B: &[f64] = &[
    0.19608072200365706963962528,
    0.3623802903983367889394994,
    -0.18682351788413996060510809,
    -0.31201628813204427978923719,
    0.4403782936141903818111176,
    0.4403782936141903818111176,
    -0.31201628813204427978923719,
    -0.18682351788413996060510809,
    0.3623802903983367889394994,
    0.19608072200365706963962528,
];

const MCATE8_A: &[f64] = &[
    0.7416703643506129534482278,
    -0.4091008258000315939973001,
    0.19075471029623837995387626,
    -0.57386247111608226665638773,
    0.29906418130365592384446354,
    0.33462491824529818378495798,
    0.31529309239676659663205666,
    -0.79688793935291635401978884,
    0.31529309239676659663205666,
    0.33462491824529818378495798,
    0.29906418130365592384446354,
    -0.57386247111608226665638773,
    0.19075471029623837995387626,
    -0.4091008258000315939973001,
    0.7416703643506129534482278,
    0.0,
];
const MCATE8_B: &[f64] = &[
    0.3708351821753065,
    0.16628476927529068,
    -0.1091730577518966,
    -0.19155388040992194,
    -0.13739914490621316,
    0.31684454977447707,
    0.3249590053210324,
    -0.24079742347807487,
    -0.24079742347807487,
    0.3249590053210324,
    0.31684454977447707,
    -0.13739914490621316,
    -0.19155388040992194,
    -0.1091730577518966,
    0.16628476927529068,
    0.3708351821753065,
];

const KAHAN_LI8_A: &[f64] = &[
    0.13020248308889008087881763,
    0.56116298177510838456196441,
    -0.3894749626448472864080786,
    0.15884190655515560089621075,
    -0.39590389413323757733623154,
    0.18453964097831570709183254,
    0.25837438768632204729397911,
    0.29501172360931029887096624,
    -0.60550853383003451169892108,
    0.29501172360931029887096624,
    0.25837438768632204729397911,
    0.18453964097831570709183254,
    -0.39590389413323757733623154,
    0.15884190655515560089621075,
    -0.3894749626448472864080786,
    0.56116298177510838456196441,
    0.13020248308889008087881763,
    0.0,
];
const KAHAN_LI8_B: &[f64] = &[
    0.06510124154444503,
    0.3456827324319992,
    0.08584400956513055,
    -0.11531652804484584,
    -0.11853099378904099,
    -0.10568212657746093,
    0.22145701433231887,
    0.27669305564781616,
    -0.15524840511036214,
    -0.15524840511036214,
    0.27669305564781616,
    0.22145701433231887,
    -0.10568212657746093,
    -0.11853099378904099,
    -0.11531652804484584,
    0.08584400956513055,
    0.3456827324319992,
    0.06510124154444503,
];

const SOFSPA10_A: &[f64] = &[
    0.07879572252168641926390768,
    0.31309610341510852776481247,
    0.02791838323507806610952027,
    -0.2295928415939070941512134,
    0.13096206107716486317465686,
    -0.26973340565451071434460973,
    0.07497334315589143566613711,
    0.11199342399981020488957508,
    0.36613344954622675119314812,
    -0.39910563013603589787862981,
    0.10308739852747107731580277,
    0.41143087395589023782070412,
    -0.00486636058313526176219566,
    -0.39203335370863990644808194,
    0.0519425029624496470371829,
    0.05066509075992449633587434,
    0.0496743706397298790545688,
    0.04931773575959453791768001,
    0.0496743706397298790545688,
    0.05066509075992449633587434,
    0.0519425029624496470371829,
    -0.39203335370863990644808194,
    -0.00486636058313526176219566,
    0.41143087395589023782070412,
    0.10308739852747107731580277,
    -0.39910563013603589787862981,
    0.36613344954622675119314812,
    0.11199342399981020488957508,
    0.07497334315589143566613711,
    -0.26973340565451071434460973,
    0.13096206107716486317465686,
    -0.2295928415939070941512134,
    0.02791838323507806610952027,
    0.31309610341510852776481247,
    0.07879572252168641926390768,
    0.0,
];
const SOFSPA10_B: &[f64] = &[
    0.03939786126084321,
    0.19594591296839747,
    0.1705072433250933,
    -0.10083722917941451,
    -0.0493153902583711,
    -0.06938567228867291,
    -0.09738003124930963,
    0.09348338357785083,
    0.23906343677301847,
    -0.016486090294904582,
    -0.14800911580428242,
    0.2572591362416807,
    0.20328225668637748,
    -0.1984498571458876,
    -0.17004542537309514,
    0.05130379686118707,
    0.050169730699827185,
    0.0494960531996622,
    0.0494960531996622,
    0.050169730699827185,
    0.05130379686118707,
    -0.17004542537309514,
    -0.1984498571458876,
    0.20328225668637748,
    0.2572591362416807,
    -0.14800911580428242,
    -0.016486090294904582,
    0.23906343677301847,
    0.09348338357785083,
    -0.09738003124930963,
    -0.06938567228867291,
    -0.0493153902583711,
    -0.10083722917941451,
    0.1705072433250933,
    0.19594591296839747,
    0.03939786126084321,
];

symplectic_algorithm!(PseudoVerletLeapfrog, &[1.0, 0.0], &[0.5, 0.5]);
symplectic_algorithm!(McAte2, MCATE2_A, MCATE2_B);
symplectic_algorithm!(
    Ruth3,
    &[2.0 / 3.0, -2.0 / 3.0, 1.0],
    &[7.0 / 24.0, 3.0 / 4.0, -1.0 / 24.0]
);
symplectic_algorithm!(McAte3, MCATE3_A, MCATE3_B);
symplectic_algorithm!(CandyRoz4, CANDY_ROZ_A, CANDY_ROZ_B);
symplectic_algorithm!(McAte4, MCATE4_A, MCATE4_B);
symplectic_algorithm!(CalvoSanz4, CALVO_SANZ4_A, CALVO_SANZ4_B);
symplectic_algorithm!(McAte42, MCATE42_A, MCATE42_B);
symplectic_algorithm!(McAte5, MCATE5_A, MCATE5_B);
symplectic_algorithm!(Yoshida6, YOSHIDA6_A, YOSHIDA6_B);
symplectic_algorithm!(KahanLi6, KAHAN_LI6_A, KAHAN_LI6_B);
symplectic_algorithm!(McAte8, MCATE8_A, MCATE8_B);
symplectic_algorithm!(KahanLi8, KAHAN_LI8_A, KAHAN_LI8_B);
symplectic_algorithm!(SofSpa10, SOFSPA10_A, SOFSPA10_B);

/// A trajectory returned by [`solve_symplectic`].
#[derive(Clone, Debug, PartialEq)]
pub struct SymplecticSolution {
    times: Vec<f64>,
    positions: Vec<f64>,
    velocities: Vec<f64>,
    dimension: usize,
    rhs_evaluations: usize,
}

impl SymplecticSolution {
    /// Saved times in integration order.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Last position partition.
    pub fn last_position(&self) -> &[f64] {
        let start = self.positions.len() - self.dimension;
        &self.positions[start..]
    }

    /// Last velocity partition.
    pub fn last_velocity(&self) -> &[f64] {
        let start = self.velocities.len() - self.dimension;
        &self.velocities[start..]
    }

    /// Position partition at a saved index.
    pub fn position(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.dimension)?;
        self.positions.get(start..start + self.dimension)
    }

    /// Velocity partition at a saved index.
    pub fn velocity(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.dimension)?;
        self.velocities.get(start..start + self.dimension)
    }

    /// Number of acceleration evaluations.
    pub fn rhs_evaluations(&self) -> usize {
        self.rhs_evaluations
    }
}

/// Failure specific to a fixed-step symplectic composition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymplecticSolveError {
    /// Position and velocity partitions differ in size.
    StateDimensionMismatch,
    /// A common solver validation or execution error.
    Solve(SolveError),
}

impl From<SolveError> for SymplecticSolveError {
    fn from(error: SolveError) -> Self {
        Self::Solve(error)
    }
}

/// Solves a second-order problem with a pinned alternating drift/kick method.
///
/// The acceleration is passed separately because the shared second-order
/// problem deliberately keeps its callback field private.
pub fn solve_symplectic<F, P, A>(
    problem: &SecondOrderOdeProblem<F, P>,
    acceleration: &F,
    _algorithm: A,
    options: &SolveOptions,
) -> Result<SymplecticSolution, SymplecticSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
    A: SymplecticAlgorithm,
{
    let position = problem.initial_position();
    let velocity = problem.initial_velocity();
    if position.is_empty() {
        return Err(SolveError::EmptyState.into());
    }
    if position.len() != velocity.len() {
        return Err(SymplecticSolveError::StateDimensionMismatch);
    }
    if !position
        .iter()
        .chain(velocity)
        .all(|value| value.is_finite())
    {
        return Err(SolveError::NonFiniteInitialState.into());
    }
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported.into());
    }
    let initial_step = options
        .initial_step
        .ok_or(SolveError::InitialStepRequired)?;
    if !initial_step.is_finite() || initial_step == 0.0 {
        return Err(SolveError::InvalidInitialStep.into());
    }
    if options.max_step.is_nan() || options.max_step <= 0.0 {
        return Err(SolveError::InvalidMaxStep.into());
    }
    let (start, end) = problem.time_span();
    if !start.is_finite() || !end.is_finite() || start == end {
        return Err(SolveError::InvalidTimeSpan.into());
    }
    let tableau = A::tableau();
    if tableau.a.is_empty() || tableau.a.len() != tableau.b.len() {
        return Err(SolveError::InvalidTableau.into());
    }

    let direction = (end - start).signum();
    let step_size = initial_step.abs().min(options.max_step);
    let dimension = position.len();
    let mut position = position.to_vec();
    let mut velocity = velocity.to_vec();
    let mut acceleration = vec![0.0; dimension];
    let mut times = vec![start];
    let mut positions = position.clone();
    let mut velocities = velocity.clone();
    let mut time = start;
    let mut steps = 0usize;
    let mut rhs_evaluations = 0usize;

    while direction * (end - time) > 0.0 {
        if steps == options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        let step = direction * step_size.min((end - time).abs());
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }
        let stage_start = time;
        let mut stage_time = time;
        for (&a, &b) in tableau.a.iter().zip(tableau.b) {
            for (q, &v) in position.iter_mut().zip(&velocity) {
                *q += a * step * v;
            }
            stage_time += a * step;
            acceleration(
                &mut acceleration,
                &velocity,
                &position,
                problem.parameters(),
                stage_time,
            );
            rhs_evaluations += 1;
            if !acceleration.iter().all(|value| value.is_finite()) {
                return Err(SolveError::NonFiniteDerivative.into());
            }
            for (v, &a_value) in velocity.iter_mut().zip(&acceleration) {
                *v += b * step * a_value;
            }
        }
        time = stage_start + step;
        steps += 1;
        if options.save == SaveMode::EveryStep || time == end {
            times.push(time);
            positions.extend_from_slice(&position);
            velocities.extend_from_slice(&velocity);
        }
    }
    if times.last().copied() != Some(end) {
        times.push(end);
        positions.extend_from_slice(&position);
        velocities.extend_from_slice(&velocity);
    }

    Ok(SymplecticSolution {
        times,
        positions,
        velocities,
        dimension,
        rhs_evaluations,
    })
}
