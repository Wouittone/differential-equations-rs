//! Independently lazy MRI-GARK resources; no coefficient arrays are generated.

use crate::tableau::define_mri_tableau_from_file;

crate::tableau::define_mis_tableau_from_file!(pub(super) MIS_TABLEAU, "MIS",
    "src/tableau/resources/mri/mis.json", crate = crate);

define_mri_tableau_from_file!(pub(super) ERK22A_TABLEAU, "MRIGARKERK22a",
    "src/tableau/resources/mri/erk22a.json", crate = crate);
define_mri_tableau_from_file!(pub(super) ERK22B_TABLEAU, "MRIGARKERK22b",
    "src/tableau/resources/mri/erk22b.json", crate = crate);
define_mri_tableau_from_file!(pub(super) ERK33A_TABLEAU, "MRIGARKERK33a",
    "src/tableau/resources/mri/erk33a.json", crate = crate);
define_mri_tableau_from_file!(pub(super) ERK45A_TABLEAU, "MRIGARKERK45a",
    "src/tableau/resources/mri/erk45a.json", crate = crate);
define_mri_tableau_from_file!(pub(super) ESDIRK34A_TABLEAU, "MRIGARKESDIRK34a",
    "src/tableau/resources/mri/esdirk34a.json", crate = crate);
define_mri_tableau_from_file!(pub(super) IRK21A_TABLEAU, "MRIGARKIRK21a",
    "src/tableau/resources/mri/irk21a.json", crate = crate);
