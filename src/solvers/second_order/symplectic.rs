//! Explicit symplectic composition methods for partitioned second-order problems.
//!
//! The coefficient vectors are pinned copies of the `SymplecticTableau` data in
//! OrdinaryDiffEqSymplecticRK.  A stage is a drift of the position by `bᵢ`
//! followed by a kick of the velocity by `aᵢ`.

#![allow(clippy::excessive_precision)]

use crate::event::times_are_numerically_equal;
use crate::{InterpolationError, SaveMode, SecondOrderOdeProblem, SolveError, SolveOptions};
use thiserror::Error;

/// A pinned alternating drift/kick composition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymplecticTableau {
    /// Velocity (kick) coefficients.
    pub a: &'static [f64],
    /// Position (drift) coefficients.
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

#[allow(clippy::approx_constant)]
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
    0.4403787936141903818111176,
    0.4403787936141903818111176,
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
    dense_segments: Vec<SymplecticDenseSegment>,
}

impl SymplecticSolution {
    /// Saved times in integration order.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Number of scalar components in each partition.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// All saved positions in contiguous row-major storage.
    pub fn position_values(&self) -> &[f64] {
        &self.positions
    }

    /// All saved velocities in contiguous row-major storage.
    pub fn velocity_values(&self) -> &[f64] {
        &self.velocities
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
        partition(&self.positions, self.dimension, index)
    }

    /// Velocity partition at a saved index.
    pub fn velocity(&self, index: usize) -> Option<&[f64]> {
        partition(&self.velocities, self.dimension, index)
    }

    /// Number of acceleration evaluations.
    pub fn rhs_evaluations(&self) -> usize {
        self.rhs_evaluations
    }

    /// Interpolates `(position, velocity)` at a covered time.
    ///
    /// Retained segments use cubic-Hermite position interpolation consistent
    /// with `q' = v` and linear velocity interpolation. Saved-only solutions
    /// retain the stable linear compatibility fallback.
    pub fn interpolate(&self, time: f64) -> Option<(Vec<f64>, Vec<f64>)> {
        self.try_interpolate(time).ok()
    }

    /// Interpolates `(position, velocity)` and reports why the query fails.
    pub fn try_interpolate(&self, time: f64) -> Result<(Vec<f64>, Vec<f64>), InterpolationError> {
        if !time.is_finite() {
            return Err(InterpolationError::NonFiniteTime);
        }
        if self.times.is_empty() {
            return Err(InterpolationError::EmptySolution);
        }
        for (index, &saved_time) in self.times.iter().enumerate() {
            if time == saved_time {
                return Ok((
                    self.position(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic position",
                        })?
                        .to_vec(),
                    self.velocity(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic velocity",
                        })?
                        .to_vec(),
                ));
            }
        }
        for segment in &self.dense_segments {
            if segment.contains(time) {
                let mut position = vec![0.0; self.dimension];
                let mut velocity = vec![0.0; self.dimension];
                segment
                    .interpolate(time, &mut position, &mut velocity)
                    .ok_or(InterpolationError::InvalidSegmentData {
                        context: "symplectic dense segment",
                    })?;
                return Ok((position, velocity));
            }
        }
        for index in 1..self.times.len() {
            let left = self.times[index - 1];
            let right = self.times[index];
            if between(time, left, right) && left != right {
                let fraction = (time - left) / (right - left);
                let mut position = vec![0.0; self.dimension];
                let mut velocity = vec![0.0; self.dimension];
                interpolate(
                    self.position(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic position",
                        })?,
                    self.position(index - 1)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic position",
                        })?,
                    fraction,
                    &mut position,
                );
                interpolate(
                    self.velocity(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic velocity",
                        })?,
                    self.velocity(index - 1)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic velocity",
                        })?,
                    fraction,
                    &mut velocity,
                );
                return Ok((position, velocity));
            }
        }
        Err(InterpolationError::OutsideTimeSpan)
    }
}

fn between(time: f64, left: f64, right: f64) -> bool {
    (left <= time && time <= right) || (right <= time && time <= left)
}

#[derive(Clone, Debug, PartialEq)]
struct SymplecticDenseSegment {
    start_time: f64,
    end_time: f64,
    start_position: Vec<f64>,
    end_position: Vec<f64>,
    start_velocity: Vec<f64>,
    end_velocity: Vec<f64>,
}

impl SymplecticDenseSegment {
    fn new(
        start_time: f64,
        end_time: f64,
        start_position: &[f64],
        end_position: &[f64],
        start_velocity: &[f64],
        end_velocity: &[f64],
    ) -> Self {
        Self {
            start_time,
            end_time,
            start_position: start_position.to_vec(),
            end_position: end_position.to_vec(),
            start_velocity: start_velocity.to_vec(),
            end_velocity: end_velocity.to_vec(),
        }
    }

    fn contains(&self, time: f64) -> bool {
        between(time, self.start_time, self.end_time)
    }

    fn interpolate(&self, time: f64, position: &mut [f64], velocity: &mut [f64]) -> Option<()> {
        if !self.contains(time)
            || position.len() != self.start_position.len()
            || velocity.len() != self.start_velocity.len()
        {
            return None;
        }
        if time == self.start_time {
            position.copy_from_slice(&self.start_position);
            velocity.copy_from_slice(&self.start_velocity);
            return Some(());
        }
        if time == self.end_time {
            position.copy_from_slice(&self.end_position);
            velocity.copy_from_slice(&self.end_velocity);
            return Some(());
        }
        let step = self.end_time - self.start_time;
        let theta = (time - self.start_time) / step;
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
        let h10 = theta3 - 2.0 * theta2 + theta;
        let h01 = -2.0 * theta3 + 3.0 * theta2;
        let h11 = theta3 - theta2;
        for index in 0..position.len() {
            position[index] = h00 * self.start_position[index]
                + h10 * step * self.start_velocity[index]
                + h01 * self.end_position[index]
                + h11 * step * self.end_velocity[index];
            velocity[index] = self.start_velocity[index]
                + theta * (self.end_velocity[index] - self.start_velocity[index]);
        }
        Some(())
    }
}

fn partition(values: &[f64], dimension: usize, index: usize) -> Option<&[f64]> {
    let start = index.checked_mul(dimension)?;
    let end = start.checked_add(dimension)?;
    values.get(start..end)
}

/// Failure specific to a fixed-step symplectic composition.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum SymplecticSolveError {
    /// Position and velocity partitions differ in size.
    #[error("position and velocity dimensions must match")]
    StateDimensionMismatch,
    /// A common solver validation or execution error.
    #[error("{0}")]
    Solve(
        #[from]
        #[source]
        SolveError,
    ),
}

/// Solves a second-order problem with a pinned alternating drift/kick method.
///
pub fn solve_symplectic<F, P, A>(
    problem: &SecondOrderOdeProblem<F, P>,
    _algorithm: A,
    options: &SolveOptions,
) -> Result<SymplecticSolution, SymplecticSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
    A: SymplecticAlgorithm,
{
    validate(problem, options)?;
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported.into());
    }
    let fixed_step = options
        .initial_step
        .ok_or(SolveError::InitialStepRequired)?;
    let (start, end) = problem.time_span();
    let tableau = A::tableau();
    if tableau.a.is_empty()
        || tableau.a.len() != tableau.b.len()
        || !tableau
            .a
            .iter()
            .chain(tableau.b)
            .all(|coefficient| coefficient.is_finite())
    {
        return Err(SolveError::InvalidTableau.into());
    }

    let direction = (end - start).signum();
    let step_size = fixed_step.min(options.max_step);
    let dimension = problem.initial_position().len();
    let mut position = problem.initial_position().to_vec();
    let mut velocity = problem.initial_velocity().to_vec();
    let mut candidate_position = position.clone();
    let mut candidate_velocity = velocity.clone();
    let mut acceleration = vec![0.0; dimension];
    let mut recorder = SymplecticRecorder::new(&position, &velocity, start, options);
    let mut time = start;
    let mut steps = 0usize;
    let mut rhs_evaluations = 0usize;

    while direction * (end - time) > 0.0 {
        if steps >= options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        let step = direction * step_size.min((end - time).abs());
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }
        candidate_position.copy_from_slice(&position);
        candidate_velocity.copy_from_slice(&velocity);
        let previous_time = time;
        rhs_evaluations += perform_step(
            problem,
            tableau,
            &mut candidate_position,
            &mut candidate_velocity,
            &mut acceleration,
            time,
            step,
        )?;
        time += step;
        if direction * (end - time) <= 0.0 {
            time = end;
        }
        steps += 1;
        recorder.record_step(
            &position,
            &velocity,
            previous_time,
            &candidate_position,
            &candidate_velocity,
            time,
            time == end,
        )?;
        std::mem::swap(&mut position, &mut candidate_position);
        std::mem::swap(&mut velocity, &mut candidate_velocity);
    }

    Ok(recorder.finish(rhs_evaluations))
}

fn validate<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<(), SymplecticSolveError> {
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
    let (start, end) = problem.time_span();
    if !start.is_finite() || !end.is_finite() || start == end {
        return Err(SolveError::InvalidTimeSpan.into());
    }
    if !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance <= 0.0
        || !options.relative_tolerance.is_finite()
        || options.relative_tolerance <= 0.0
    {
        return Err(SolveError::InvalidTolerance.into());
    }
    if options
        .initial_step
        .is_some_and(|step| !step.is_finite() || step <= 0.0)
    {
        return Err(SolveError::InvalidInitialStep.into());
    }
    if options.max_step.is_nan() || options.max_step <= 0.0 {
        return Err(SolveError::InvalidMaxStep.into());
    }
    if options.max_steps == 0 {
        return Err(SolveError::InvalidMaxSteps.into());
    }
    if !options.event_tolerance.is_finite() || options.event_tolerance <= 0.0 {
        return Err(SolveError::InvalidEventTolerance.into());
    }
    let direction = (end - start).signum();
    if !options.save_at.iter().all(|time| {
        time.is_finite() && direction * (*time - start) >= 0.0 && direction * (end - *time) >= 0.0
    }) || options
        .save_at
        .windows(2)
        .any(|pair| direction * (pair[1] - pair[0]) <= 0.0)
    {
        return Err(SolveError::InvalidSaveAt.into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn perform_step<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    tableau: SymplecticTableau,
    position: &mut [f64],
    velocity: &mut [f64],
    acceleration: &mut [f64],
    time: f64,
    step: f64,
) -> Result<usize, SolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    let mut stage_time = time;
    for (stage, (&kick, &drift)) in tableau.a.iter().zip(tableau.b).enumerate() {
        for (position, &velocity) in position.iter_mut().zip(&*velocity) {
            *position += drift * step * velocity;
        }
        problem.evaluate_acceleration(acceleration, velocity, position, stage_time);
        if !acceleration.iter().all(|value| value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
        for (velocity, &acceleration) in velocity.iter_mut().zip(&*acceleration) {
            *velocity += kick * step * acceleration;
        }
        if stage + 1 < tableau.stages() {
            stage_time += kick * step;
        }
    }
    Ok(tableau.stages())
}

struct SymplecticRecorder<'a> {
    times: Vec<f64>,
    positions: Vec<f64>,
    velocities: Vec<f64>,
    dimension: usize,
    save_at: &'a [f64],
    next_save: usize,
    save_mode: SaveMode,
    interpolation_position: Vec<f64>,
    interpolation_velocity: Vec<f64>,
    dense_segments: Vec<SymplecticDenseSegment>,
    retain_dense_output: bool,
}

impl<'a> SymplecticRecorder<'a> {
    fn new(position: &[f64], velocity: &[f64], time: f64, options: &'a SolveOptions) -> Self {
        let save_initial = options.save_at.is_empty() || options.save_at.first() == Some(&time);
        let capacity = options.save_at.len().max(2);
        let mut recorder = Self {
            times: Vec::with_capacity(capacity),
            positions: Vec::with_capacity(capacity * position.len()),
            velocities: Vec::with_capacity(capacity * velocity.len()),
            dimension: position.len(),
            save_at: &options.save_at,
            next_save: usize::from(!options.save_at.is_empty() && save_initial),
            save_mode: options.save,
            interpolation_position: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; position.len()]
            },
            interpolation_velocity: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; velocity.len()]
            },
            dense_segments: Vec::new(),
            retain_dense_output: options.retain_dense_output,
        };
        if save_initial {
            recorder.push_unique(time, position, velocity);
        }
        recorder
    }

    #[allow(clippy::too_many_arguments)]
    fn record_step(
        &mut self,
        previous_position: &[f64],
        previous_velocity: &[f64],
        previous_time: f64,
        position: &[f64],
        velocity: &[f64],
        time: f64,
        final_time: bool,
    ) -> Result<(), SolveError> {
        let segment = SymplecticDenseSegment::new(
            previous_time,
            time,
            previous_position,
            position,
            previous_velocity,
            velocity,
        );
        if self.retain_dense_output {
            self.dense_segments.push(segment.clone());
        }
        if self.save_at.is_empty() {
            if self.save_mode == SaveMode::EveryStep || final_time {
                self.push_unique(time, position, velocity);
            }
            return Ok(());
        }

        let direction = (time - previous_time).signum();
        while let Some(&target) = self.save_at.get(self.next_save) {
            if direction * (target - previous_time) <= 0.0 {
                self.next_save += 1;
                continue;
            }
            if direction * (time - target) < 0.0 {
                break;
            }
            segment
                .interpolate(
                    target,
                    &mut self.interpolation_position,
                    &mut self.interpolation_velocity,
                )
                .ok_or(SolveError::DenseOutputFailed)?;
            self.times.push(target);
            self.positions
                .extend_from_slice(&self.interpolation_position);
            self.velocities
                .extend_from_slice(&self.interpolation_velocity);
            self.next_save += 1;
        }
        Ok(())
    }

    fn push_unique(&mut self, time: f64, position: &[f64], velocity: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_numerically_equal(*saved, time))
        {
            let start = self.positions.len() - self.dimension;
            self.positions[start..].copy_from_slice(position);
            self.velocities[start..].copy_from_slice(velocity);
        } else {
            self.times.push(time);
            self.positions.extend_from_slice(position);
            self.velocities.extend_from_slice(velocity);
        }
    }

    fn finish(self, rhs_evaluations: usize) -> SymplecticSolution {
        SymplecticSolution {
            times: self.times,
            positions: self.positions,
            velocities: self.velocities,
            dimension: self.dimension,
            rhs_evaluations,
            dense_segments: self.dense_segments,
        }
    }
}

fn interpolate(current: &[f64], previous: &[f64], fraction: f64, output: &mut [f64]) {
    for ((output, previous), current) in output.iter_mut().zip(previous).zip(current) {
        *output = previous + fraction * (current - previous);
    }
}
pub use super::general::{LeapfrogDriftKickDrift, SymplecticEuler, VelocityVerlet, VerletLeapfrog};
