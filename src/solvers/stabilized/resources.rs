mod eserk4;
mod eserk5;
mod rock2;
mod rock4;
mod serk2;

#[cfg(test)]
mod tests;

pub(super) use eserk4::{eserk4_available_degrees, eserk4_tableau_for_degree};
pub(super) use eserk5::{eserk5_available_degrees, eserk5_tableau_for_degree};
pub(super) use rock2::{rock2_available_degrees, rock2_tableau_for_degree};
pub(super) use rock4::{rock4_available_degrees, rock4_tableau_for_degree};
pub(super) use serk2::{serk2_available_degrees, serk2_tableau_for_degree};
