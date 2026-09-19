//! Variable-step and variable-order Adams methods.

mod algorithms;
mod common;
mod fixed;
mod history;
mod method;
mod variable_order;

#[cfg(test)]
mod tests;

pub use algorithms::*;
