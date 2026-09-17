//! Shared constant-step formulas, parsed individually on first use.

use crate::SolveError;
use crate::tableau::{
    LinearMultistepTableau, TableauAccessError, define_multistep_tableau_from_file,
    define_variable_multistep_tableau_from_file, load_tableau,
};

const MINIMUM_ORDER: usize = 1;
const MAXIMUM_ORDER: usize = 5;

define_variable_multistep_tableau_from_file!(pub(super) ABDF2_TABLEAU, "Abdf2",
    "src/tableau/resources/multistep/abdf2.json", crate = crate);

define_multistep_tableau_from_file!(pub(super) AB1, "Ab1", "src/tableau/resources/multistep/ab1.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AB2, "Ab2", "src/tableau/resources/multistep/ab2.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AB3, "Ab3", "src/tableau/resources/multistep/ab3.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AB4, "Ab4", "src/tableau/resources/multistep/ab4.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AB5, "Ab5", "src/tableau/resources/multistep/ab5.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AM3, "Am3", "src/tableau/resources/multistep/am3.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AM4, "Am4", "src/tableau/resources/multistep/am4.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AM5, "Am5", "src/tableau/resources/multistep/am5.json", crate = crate);

pub(crate) fn adams_bashforth(
    order: usize,
) -> Result<&'static LinearMultistepTableau, TableauAccessError> {
    let resource = match order {
        1 => &AB1,
        2 => &AB2,
        3 => &AB3,
        4 => &AB4,
        5 => &AB5,
        _ => {
            return Err(TableauAccessError::UnsupportedOrder {
                requested: order,
                minimum: MINIMUM_ORDER,
                maximum: MAXIMUM_ORDER,
            });
        }
    };
    let tableau = load_tableau(resource).map_err(TableauAccessError::Resource)?;
    if !is_adams(tableau, false) {
        return Err(TableauAccessError::IncompatibleFormula {
            context: "Adams--Bashforth",
        });
    }
    Ok(tableau)
}

pub(super) fn is_adams(tableau: &LinearMultistepTableau, corrector: bool) -> bool {
    tableau.alpha()[0] == 1.0
        && tableau.alpha()[1] == -1.0
        && tableau.alpha()[2..].iter().all(|value| *value == 0.0)
        && tableau.is_explicit() != corrector
        && tableau.beta().len() == tableau.order() + usize::from(!corrector)
}

define_multistep_tableau_from_file!(
    BDF1,
    "Bdf1",
    "src/tableau/resources/multistep/bdf1.json",
    crate = crate
);
define_multistep_tableau_from_file!(
    BDF2,
    "Bdf2",
    "src/tableau/resources/multistep/bdf2.json",
    crate = crate
);
define_multistep_tableau_from_file!(
    BDF3,
    "Bdf3",
    "src/tableau/resources/multistep/bdf3.json",
    crate = crate
);
define_multistep_tableau_from_file!(
    BDF4,
    "Bdf4",
    "src/tableau/resources/multistep/bdf4.json",
    crate = crate
);
define_multistep_tableau_from_file!(
    BDF5,
    "Bdf5",
    "src/tableau/resources/multistep/bdf5.json",
    crate = crate
);

pub(super) fn backward_differentiation(
    order: usize,
) -> Result<&'static LinearMultistepTableau, TableauAccessError> {
    let resource = match order {
        1 => &BDF1,
        2 => &BDF2,
        3 => &BDF3,
        4 => &BDF4,
        5 => &BDF5,
        _ => {
            return Err(TableauAccessError::UnsupportedOrder {
                requested: order,
                minimum: MINIMUM_ORDER,
                maximum: MAXIMUM_ORDER,
            });
        }
    };
    let tableau = load_tableau(resource).map_err(TableauAccessError::Resource)?;
    if tableau.order() != order || tableau.ndf_kappa().is_none() {
        return Err(TableauAccessError::IncompatibleFormula {
            context: "backward-differentiation",
        });
    }
    Ok(tableau)
}

pub(super) fn ndf_kappa(
    tableau: &LinearMultistepTableau,
    ndf: bool,
) -> Result<f64, TableauAccessError> {
    let kappa = tableau
        .ndf_kappa()
        .ok_or(TableauAccessError::IncompatibleFormula {
            context: "numerical-differentiation formula",
        })?;
    Ok(if ndf { kappa } else { 0.0 })
}

pub(super) fn error_constant(
    tableau: &LinearMultistepTableau,
    ndf: bool,
) -> Result<f64, TableauAccessError> {
    Ok(ndf_kappa(tableau, ndf)? * tableau.alpha()[0] + 1.0 / (tableau.order() + 1) as f64)
}

pub(crate) fn map_tableau_access_error(error: TableauAccessError) -> SolveError {
    match error {
        TableauAccessError::UnsupportedOrder { .. } => SolveError::InvalidMultistepOrder,
        TableauAccessError::Resource(_) | TableauAccessError::IncompatibleFormula { .. } => {
            SolveError::InvalidTableau
        }
    }
}

#[cfg(test)]
mod tests {
    use super::map_tableau_access_error;
    use crate::SolveError;
    use crate::tableau::{TableauAccessError, parse_tableau};

    #[test]
    fn access_errors_keep_solve_level_compatibility() {
        assert_eq!(
            map_tableau_access_error(TableauAccessError::UnsupportedOrder {
                requested: 6,
                minimum: 1,
                maximum: 5,
            }),
            SolveError::InvalidMultistepOrder
        );
        assert_eq!(
            map_tableau_access_error(TableauAccessError::IncompatibleFormula {
                context: "test formula",
            }),
            SolveError::InvalidTableau
        );
        assert_eq!(
            map_tableau_access_error(TableauAccessError::Resource(
                parse_tableau("{", "Broken").unwrap_err(),
            )),
            SolveError::InvalidTableau
        );
    }
}
