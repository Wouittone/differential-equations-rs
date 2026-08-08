//! Compile-time coefficient metadata and structural validation.
//!
//! These records describe solver data only; no declarative file is parsed on
//! a solve path. Runtime caches and mutable history remain owned by kernels.

#![allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Scalar {
    Rational { numerator: i64, denominator: i64 },
    Decimal(&'static str),
    Symbol(&'static str),
}

impl Scalar {
    fn value(self) -> Result<f64, &'static str> {
        match self {
            Self::Rational {
                numerator,
                denominator,
            } => {
                if denominator == 0 {
                    return Err("zero scalar denominator");
                }
                let value = numerator as f64 / denominator as f64;
                value
                    .is_finite()
                    .then_some(value)
                    .ok_or("non-finite rational")
            }
            Self::Decimal(text) => text
                .parse::<f64>()
                .map_err(|_| "invalid decimal scalar")
                .and_then(|value| {
                    value
                        .is_finite()
                        .then_some(value)
                        .ok_or("non-finite decimal")
                }),
            Self::Symbol("sqrt2") => Ok(2.0_f64.sqrt()),
            Self::Symbol("sqrt3") => Ok(3.0_f64.sqrt()),
            Self::Symbol(_) => Err("unknown symbolic scalar"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DenseRecord {
    GenericHermite {
        order: u16,
    },
    FreeStagePolynomial {
        rows: Vec<Vec<Scalar>>,
        order: u16,
    },
    LazyExtraStagePolynomial {
        base_stages: usize,
        extra_stages: usize,
        rows: Vec<Vec<Scalar>>,
        order: u16,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FamilyCoefficients {
    ExplicitButcher {
        a_lower: Vec<Vec<Scalar>>,
        b: Vec<Scalar>,
        embedded: Option<Vec<Scalar>>,
    },
    Rosenbrock {
        a: Vec<Vec<Scalar>>,
        c_matrix: Vec<Vec<Scalar>>,
        gamma: Scalar,
        b: Vec<Scalar>,
        embedded: Option<Vec<Scalar>>,
        h: Vec<Vec<Scalar>>,
    },
    ShuOsher {
        alpha: Vec<Vec<Scalar>>,
        beta: Vec<Vec<Scalar>>,
    },
    LowStorage {
        variant: &'static str,
        a: Vec<Scalar>,
        b: Vec<Scalar>,
        register_count: usize,
    },
    Multistep {
        order: u16,
        history: Vec<Scalar>,
        corrector: Option<Vec<Scalar>>,
        variable_step: bool,
    },
    Symplectic {
        a: Vec<Scalar>,
        b: Vec<Scalar>,
    },
    Partitioned {
        aq: Vec<Vec<Scalar>>,
        ap: Vec<Vec<Scalar>>,
        bq: Vec<Scalar>,
        bp: Vec<Scalar>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Provenance {
    pub(crate) package: &'static str,
    pub(crate) path: &'static str,
    pub(crate) commit: &'static str,
    pub(crate) line: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MethodRecord {
    pub(crate) name: &'static str,
    pub(crate) family: &'static str,
    pub(crate) order: u16,
    pub(crate) embedded_order: Option<u16>,
    pub(crate) fsal: bool,
    pub(crate) stage_times: Vec<Scalar>,
    pub(crate) coefficients: FamilyCoefficients,
    pub(crate) dense: Option<DenseRecord>,
    pub(crate) provenance: Provenance,
}

pub(crate) fn validate_record(record: &MethodRecord) -> Result<(), String> {
    if record.name.is_empty() || record.family.is_empty() || record.provenance.path.is_empty() {
        return Err("method metadata must not be empty".into());
    }
    if record.order == 0 || record.provenance.commit.is_empty() {
        return Err("method order and commit are required".into());
    }
    for scalar in &record.stage_times {
        scalar.value().map_err(str::to_owned)?;
    }
    let stages = record.stage_times.len();
    match &record.coefficients {
        FamilyCoefficients::ExplicitButcher {
            a_lower,
            b,
            embedded,
        } => {
            if a_lower.len() != stages || b.len() != stages {
                return Err("explicit tableau dimensions do not match stages".into());
            }
            if embedded
                .as_ref()
                .is_some_and(|weights| weights.len() != stages)
            {
                return Err("embedded weights do not match stages".into());
            }
            for (row, entries) in a_lower.iter().enumerate() {
                if entries.len() > row {
                    return Err("explicit tableau is not strictly lower triangular".into());
                }
                for scalar in entries {
                    scalar.value().map_err(str::to_owned)?;
                }
            }
            for scalar in b.iter().chain(embedded.iter().flatten()) {
                scalar.value().map_err(str::to_owned)?;
            }
        }
        FamilyCoefficients::Rosenbrock {
            a,
            c_matrix,
            gamma,
            b,
            embedded,
            h,
        } => {
            if a.len() != stages || c_matrix.len() != stages || b.len() != stages {
                return Err("Rosenbrock dimensions do not match stages".into());
            }
            if embedded
                .as_ref()
                .is_some_and(|weights| weights.len() != stages)
            {
                return Err("Rosenbrock embedded weights do not match stages".into());
            }
            if !h.is_empty() && h.iter().any(|row| row.len() != stages) {
                return Err("Rosenbrock dense rows do not match stages".into());
            }
            gamma.value().map_err(str::to_owned)?;
            for matrix in [a, c_matrix] {
                if matrix.iter().any(|row| row.len() != stages) {
                    return Err("Rosenbrock matrices must be square".into());
                }
                for scalar in matrix.iter().flatten() {
                    scalar.value().map_err(str::to_owned)?;
                }
            }
        }
        FamilyCoefficients::ShuOsher { alpha, beta } => {
            if alpha.len() != stages || beta.len() != stages {
                return Err("Shu-Osher dimensions do not match stages".into());
            }
            for scalar in alpha.iter().chain(beta).flatten() {
                scalar.value().map_err(str::to_owned)?;
            }
        }
        FamilyCoefficients::LowStorage {
            variant,
            a,
            b,
            register_count,
        } => {
            if variant.is_empty() || *register_count == 0 || a.len() != b.len() {
                return Err("invalid low-storage recurrence metadata".into());
            }
            for scalar in a.iter().chain(b) {
                scalar.value().map_err(str::to_owned)?;
            }
        }
        FamilyCoefficients::Multistep {
            order,
            history,
            corrector,
            ..
        } => {
            if *order == 0 || history.len() < *order as usize {
                return Err("multistep history is shorter than the order".into());
            }
            if corrector
                .as_ref()
                .is_some_and(|weights| weights.len() != history.len())
            {
                return Err("multistep corrector length mismatch".into());
            }
            for scalar in history.iter().chain(corrector.iter().flatten()) {
                scalar.value().map_err(str::to_owned)?;
            }
        }
        FamilyCoefficients::Symplectic { a, b } => {
            if a.is_empty() || a.len() != b.len() {
                return Err("symplectic composition dimensions mismatch".into());
            }
            for scalar in a.iter().chain(b) {
                scalar.value().map_err(str::to_owned)?;
            }
        }
        FamilyCoefficients::Partitioned { aq, ap, bq, bp } => {
            if aq.is_empty() || aq.len() != ap.len() || aq.len() != bq.len() || bq.len() != bp.len()
            {
                return Err("partitioned tableau dimensions mismatch".into());
            }
        }
    }
    if let Some(dense) = &record.dense {
        match dense {
            DenseRecord::GenericHermite { order } if *order < 3 => {
                return Err("Hermite interpolation order must be at least three".into());
            }
            DenseRecord::FreeStagePolynomial { rows, order }
            | DenseRecord::LazyExtraStagePolynomial { rows, order, .. } => {
                if *order == 0 || rows.is_empty() || rows.iter().any(|row| row.is_empty()) {
                    return Err("dense polynomial rows must be non-empty".into());
                }
                for scalar in rows.iter().flatten() {
                    scalar.value().map_err(str::to_owned)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> Provenance {
        Provenance {
            package: "OrdinaryDiffEqExplicitTableaus",
            path: "lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_classic.jl",
            commit: "211142263781255a9aa2f910f6760b9f18ec29c8",
            line: 1,
        }
    }

    #[test]
    fn validates_rk4_shape_and_rejects_upper_entries() {
        let half = Scalar::Rational {
            numerator: 1,
            denominator: 2,
        };
        let record = MethodRecord {
            name: "RK4",
            family: "explicit",
            order: 4,
            embedded_order: None,
            fsal: false,
            stage_times: vec![
                Scalar::Rational {
                    numerator: 0,
                    denominator: 1,
                },
                half,
                half,
                Scalar::Rational {
                    numerator: 1,
                    denominator: 1,
                },
            ],
            coefficients: FamilyCoefficients::ExplicitButcher {
                a_lower: vec![vec![], vec![half], vec![half, half], vec![half, half, half]],
                b: vec![
                    Scalar::Rational {
                        numerator: 1,
                        denominator: 6
                    };
                    4
                ],
                embedded: None,
            },
            dense: None,
            provenance: provenance(),
        };
        assert!(validate_record(&record).is_ok());
        let mut invalid = record;
        if let FamilyCoefficients::ExplicitButcher { a_lower, .. } = &mut invalid.coefficients {
            a_lower[0].push(half);
        }
        assert!(validate_record(&invalid).is_err());
    }

    #[test]
    fn validates_multistep_and_dense_metadata() {
        let record = MethodRecord {
            name: "AB3",
            family: "multistep",
            order: 3,
            embedded_order: None,
            fsal: false,
            stage_times: vec![Scalar::Rational {
                numerator: 0,
                denominator: 1,
            }],
            coefficients: FamilyCoefficients::Multistep {
                order: 3,
                history: vec![
                    Scalar::Rational {
                        numerator: 23,
                        denominator: 12
                    };
                    3
                ],
                corrector: None,
                variable_step: false,
            },
            dense: Some(DenseRecord::GenericHermite { order: 3 }),
            provenance: provenance(),
        };
        assert!(validate_record(&record).is_ok());
    }
}
