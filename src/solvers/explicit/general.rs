//! Resource-backed explicit Runge--Kutta methods and their shared kernel.

mod algorithms;
mod helpers;
mod kernel;
mod tableau_access;
mod workspace;

#[cfg(test)]
mod tests;

pub use algorithms::*;
pub(crate) use kernel::ExplicitKernel;
pub(crate) use tableau_access::ResourceTableau;
// Retain the pre-decomposition crate-visible path for internal extensions.
#[allow(unused_imports)]
pub(crate) use tableau_access::TableauAccess;
