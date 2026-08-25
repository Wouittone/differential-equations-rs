//! Compile-time expansion of declarative Runge--Kutta tableau resources.

use proc_macro::TokenStream;
use proc_macro2::{Literal, TokenStream as TokenStream2};
use quote::quote;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitStr, Path as SynPath, Token, Visibility, parse_macro_input, parse_quote};

struct MacroInput {
    visibility: Visibility,
    name: Ident,
    path: LitStr,
    crate_path: SynPath,
}

impl Parse for MacroInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let visibility = input.parse()?;
        let name = input.parse()?;
        input.parse::<Token![,]>()?;
        let path = input.parse()?;
        let crate_path = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            input.parse::<Token![crate]>()?;
            input.parse::<Token![=]>()?;
            input.parse()?
        } else {
            parse_quote!(::differential_equations)
        };
        if !input.is_empty() {
            return Err(input.error(
                "expected `visibility Name, \"path/to/tableau.toml\"` with optional `, crate = path`",
            ));
        }
        Ok(Self {
            visibility,
            name,
            path,
            crate_path,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExplicitTableau {
    schema_version: u32,
    name: String,
    description: String,
    order: usize,
    embedded_order: Option<usize>,
    fsal: bool,
    nodes: Vec<Scalar>,
    coefficients: Vec<Vec<Scalar>>,
    weights: Vec<Scalar>,
    error_weights: Option<Vec<Scalar>>,
    second_error_weights: Option<Vec<Scalar>>,
    dense_coefficients: Option<Vec<Vec<Scalar>>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum Scalar {
    Integer(i64),
    Float(f64),
    Text(String),
}

impl Scalar {
    fn materialize(&self) -> Result<f64, String> {
        let value = match self {
            Self::Integer(value) => *value as f64,
            Self::Float(value) => *value,
            Self::Text(text) => parse_scalar(text)?,
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err("coefficient must be finite".into())
        }
    }
}

fn parse_scalar(text: &str) -> Result<f64, String> {
    let text = text.trim();
    if let Some((numerator, denominator)) = text.split_once('/') {
        if denominator.contains('/') {
            return Err(format!("invalid rational coefficient `{text}`"));
        }
        let numerator = numerator
            .trim()
            .parse::<i64>()
            .map_err(|_| format!("invalid rational numerator in `{text}`"))?;
        let denominator = denominator
            .trim()
            .parse::<i64>()
            .map_err(|_| format!("invalid rational denominator in `{text}`"))?;
        if denominator == 0 {
            return Err(format!("zero rational denominator in `{text}`"));
        }
        return Ok(numerator as f64 / denominator as f64);
    }
    text.parse::<f64>()
        .map_err(|_| format!("invalid decimal coefficient `{text}`"))
}

fn values(values: &[Scalar], label: &str) -> Result<Vec<f64>, String> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .materialize()
                .map_err(|error| format!("{label}[{index}]: {error}"))
        })
        .collect()
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= 64.0 * f64::EPSILON * scale
}

impl ExplicitTableau {
    fn validate(&self, requested_name: &str) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported schema_version {}; expected 1",
                self.schema_version
            ));
        }
        if self.name != requested_name {
            return Err(format!(
                "resource method name `{}` does not match generated Rust type `{requested_name}`",
                self.name
            ));
        }
        if self.description.trim().is_empty() {
            return Err("description must not be empty".into());
        }
        if self.order == 0 {
            return Err("order must be positive".into());
        }
        if self
            .embedded_order
            .is_some_and(|embedded| embedded == 0 || embedded >= self.order)
        {
            return Err("embedded_order must be positive and lower than order".into());
        }

        let nodes = values(&self.nodes, "nodes")?;
        let weights = values(&self.weights, "weights")?;
        let stages = nodes.len();
        if stages == 0 {
            return Err("a tableau must contain at least one stage".into());
        }
        if self.coefficients.len() != stages || weights.len() != stages {
            return Err("nodes, coefficient rows, and weights must have equal stage counts".into());
        }
        if !approximately_equal(nodes[0], 0.0) {
            return Err("the first explicit stage node must be zero".into());
        }
        for (row_index, row) in self.coefficients.iter().enumerate() {
            if row.len() != row_index {
                return Err(format!(
                    "coefficients[{row_index}] must contain exactly {row_index} strictly lower-triangular entries"
                ));
            }
            let materialized = values(row, &format!("coefficients[{row_index}]"))?;
            let row_sum = materialized.iter().sum::<f64>();
            if !approximately_equal(row_sum, nodes[row_index]) {
                return Err(format!(
                    "coefficients[{row_index}] sums to {row_sum}, but nodes[{row_index}] is {}",
                    nodes[row_index]
                ));
            }
        }
        let weight_sum = weights.iter().sum::<f64>();
        if !approximately_equal(weight_sum, 1.0) {
            return Err(format!("weights must sum to one; found {weight_sum}"));
        }

        self.validate_estimator("error_weights", self.error_weights.as_deref(), stages)?;
        self.validate_estimator(
            "second_error_weights",
            self.second_error_weights.as_deref(),
            stages,
        )?;
        if self.error_weights.is_some() != self.embedded_order.is_some() {
            return Err(
                "embedded_order and error_weights must either both be present or both be absent"
                    .into(),
            );
        }
        if self.second_error_weights.is_some() && self.error_weights.is_none() {
            return Err("second_error_weights requires error_weights".into());
        }

        if let Some(rows) = &self.dense_coefficients {
            if rows.len() != stages || rows.iter().any(Vec::is_empty) {
                return Err("dense_coefficients must contain one non-empty row per stage".into());
            }
            for (row_index, row) in rows.iter().enumerate() {
                values(row, &format!("dense_coefficients[{row_index}]"))?;
            }
        }

        if self.fsal {
            if stages < 2 || !approximately_equal(nodes[stages - 1], 1.0) {
                return Err("FSAL requires a final stage at node one".into());
            }
            let final_row = values(&self.coefficients[stages - 1], "coefficients[final]")?;
            if !approximately_equal(weights[stages - 1], 0.0)
                || final_row
                    .iter()
                    .zip(&weights[..stages - 1])
                    .any(|(left, right)| !approximately_equal(*left, *right))
            {
                return Err(
                    "FSAL requires the final stage row to equal the primary weights".into(),
                );
            }
        }
        Ok(())
    }

    fn validate_estimator(
        &self,
        label: &str,
        estimator: Option<&[Scalar]>,
        stages: usize,
    ) -> Result<(), String> {
        let Some(estimator) = estimator else {
            return Ok(());
        };
        if estimator.len() != stages {
            return Err(format!("{label} must contain one entry per stage"));
        }
        let estimator = values(estimator, label)?;
        let sum = estimator.iter().sum::<f64>();
        if !approximately_equal(sum, 0.0) {
            return Err(format!("{label} must sum to zero; found {sum}"));
        }
        Ok(())
    }
}

fn load_tableau(path: &Path, requested_name: &str) -> Result<ExplicitTableau, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))?;
    let tableau: ExplicitTableau = toml::from_str(&source)
        .map_err(|error| format!("invalid tableau TOML in `{}`: {error}", path.display()))?;
    tableau.validate(requested_name)?;
    Ok(tableau)
}

fn literal(value: f64) -> Literal {
    Literal::f64_suffixed(value)
}

fn emit_values(values: &[Scalar]) -> Result<Vec<Literal>, String> {
    values
        .iter()
        .map(Scalar::materialize)
        .map(|value| value.map(literal))
        .collect()
}

fn emit_rows(rows: &[Vec<Scalar>]) -> Result<Vec<TokenStream2>, String> {
    rows.iter()
        .map(|row| {
            let values = emit_values(row)?;
            Ok(quote! { &[#(#values),*] })
        })
        .collect()
}

/// Defines a zero-sized explicit Runge--Kutta algorithm from a TOML resource.
///
/// The path is relative to the invoking package's `CARGO_MANIFEST_DIR`. The
/// file is parsed and validated while compiling; the expansion contains only
/// static coefficient arrays, so solving performs no file I/O or parsing.
#[proc_macro]
pub fn define_explicit_rk_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as MacroInput);
    match expand(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand(input: MacroInput) -> Result<TokenStream2, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let relative_path = input.path.value();
    let path = manifest_dir.join(&relative_path);
    let tableau = load_tableau(&path, &input.name.to_string())?;

    let visibility = input.visibility;
    let name = input.name;
    let source_path = input.path;
    let crate_path = input.crate_path;
    let description = tableau.description;
    let order = tableau.order;
    let fsal = tableau.fsal;
    let nodes = emit_values(&tableau.nodes)?;
    let rows = emit_rows(&tableau.coefficients)?;
    let weights = emit_values(&tableau.weights)?;
    let error_weights = match tableau.error_weights {
        Some(values) => {
            let values = emit_values(&values)?;
            quote! { Some(&[#(#values),*]) }
        }
        None => quote! { None },
    };
    let second_error_weights = match tableau.second_error_weights {
        Some(values) => {
            let values = emit_values(&values)?;
            quote! { Some(&[#(#values),*]) }
        }
        None => quote! { None },
    };
    let dense_coefficients = match tableau.dense_coefficients {
        Some(rows) => {
            let rows = emit_rows(&rows)?;
            quote! { Some(&[#(#rows),*]) }
        }
        None => quote! { None },
    };

    Ok(quote! {
        const _: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path));

        #[doc = #description]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #visibility struct #name;

        impl #crate_path::algorithms::explicit::general::ButcherTableau for #name {
            const NODES: &'static [f64] = &[#(#nodes),*];
            const COEFFICIENTS: &'static [&'static [f64]] = &[#(#rows),*];
            const WEIGHTS: &'static [f64] = &[#(#weights),*];
            const ERROR_WEIGHTS: Option<&'static [f64]> = #error_weights;
            const SECOND_ERROR_WEIGHTS: Option<&'static [f64]> = #second_error_weights;
            const DENSE_COEFFICIENTS: Option<&'static [&'static [f64]]> = #dense_coefficients;
            const ORDER: usize = #order;
            const FSAL: bool = #fsal;
        }

        impl #crate_path::OdeAlgorithm for #name {
            fn solve<F, P>(
                &self,
                problem: &#crate_path::OdeProblem<F, P>,
                options: &#crate_path::SolveOptions,
            ) -> Result<
                #crate_path::Solution,
                #crate_path::SolveError,
            >
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                #crate_path::OdeAlgorithm::solve(
                    &#crate_path::algorithms::explicit::general::ExplicitRungeKutta::<Self>::new(),
                    problem,
                    options,
                )
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> ExplicitTableau {
        toml::from_str(source).unwrap()
    }

    const HEUN: &str = r#"
schema_version = 1
name = "FileHeun"
description = "Heun from a resource"
order = 2
embedded_order = 1
fsal = false
nodes = ["0", "1"]
coefficients = [[], ["1"]]
weights = ["1/2", "1/2"]
error_weights = ["-1/2", "1/2"]
"#;

    #[test]
    fn accepts_valid_explicit_tableau() {
        parse(HEUN).validate("FileHeun").unwrap();
    }

    #[test]
    fn rejects_non_triangular_rows() {
        let invalid = HEUN.replace("[[], [\"1\"]]", "[[], [\"1\", \"0\"]]");
        assert!(parse(&invalid).validate("FileHeun").is_err());
    }

    #[test]
    fn rejects_inconsistent_nodes() {
        let invalid = HEUN.replace("nodes = [\"0\", \"1\"]", "nodes = [\"0\", \"1/2\"]");
        assert!(parse(&invalid).validate("FileHeun").is_err());
    }
}
