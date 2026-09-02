//! Shared constant-step formulas, parsed individually on first use.

use crate::SolveError;
use crate::tableau::{LinearMultistepTableau, define_multistep_tableau_from_file, load_tableau};

define_multistep_tableau_from_file!(pub(super) AB1, "Ab1", "src/tableau/resources/multistep/ab1.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AB2, "Ab2", "src/tableau/resources/multistep/ab2.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AB3, "Ab3", "src/tableau/resources/multistep/ab3.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AB4, "Ab4", "src/tableau/resources/multistep/ab4.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AB5, "Ab5", "src/tableau/resources/multistep/ab5.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AM3, "Am3", "src/tableau/resources/multistep/am3.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AM4, "Am4", "src/tableau/resources/multistep/am4.json", crate = crate);
define_multistep_tableau_from_file!(pub(super) AM5, "Am5", "src/tableau/resources/multistep/am5.json", crate = crate);

pub(crate) fn adams_bashforth(order: usize) -> Result<&'static LinearMultistepTableau, SolveError> {
    let resource = match order {
        1 => &AB1,
        2 => &AB2,
        3 => &AB3,
        4 => &AB4,
        5 => &AB5,
        _ => return Err(SolveError::InvalidMultistepOrder),
    };
    let tableau = load_tableau(resource).map_err(|_| SolveError::InvalidTableau)?;
    if !is_adams(tableau, false) {
        return Err(SolveError::InvalidTableau);
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
