//! Compile-time expansion of declarative Runge--Kutta tableau resources.

use proc_macro::TokenStream;
use proc_macro2::{Literal, TokenStream as TokenStream2};
use quote::quote;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use syn::parse::{Parse, ParseStream};
use syn::{
    BinOp, Expr, ExprBinary, ExprLit, ExprUnary, Ident, Lit, LitStr, Path as SynPath, Token, UnOp,
    Visibility, parse_macro_input, parse_quote,
};

struct MacroInput {
    visibility: Visibility,
    name: Ident,
    path: LitStr,
    crate_path: SynPath,
}

struct CoefficientMacroInput {
    visibility: Visibility,
    path: LitStr,
    crate_path: SynPath,
}

impl Parse for CoefficientMacroInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let visibility = input.parse()?;
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
                "expected `visibility, \"path/to/coefficients.toml\"` with optional `, crate = path`",
            ));
        }
        Ok(Self {
            visibility,
            path,
            crate_path,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoefficientResource {
    schema_version: u32,
    description: String,
    constants: Vec<CoefficientConstant>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum CoefficientType {
    F64,
    Usize,
    I32,
    Bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum CoefficientKind {
    Scalar,
    Slice,
    Array,
    Rows,
    Matrix,
    LazyStageSlice,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum ResourceValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    Text(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoefficientConstant {
    name: String,
    #[serde(rename = "type")]
    value_type: CoefficientType,
    kind: CoefficientKind,
    value: Option<ResourceValue>,
    values: Option<Vec<ResourceValue>>,
    rows: Option<Vec<Vec<ResourceValue>>>,
    stages: Option<Vec<LazyStage>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LazyStage {
    node: ResourceValue,
    weights: Vec<LazyStageWeight>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LazyStageWeight {
    index: usize,
    value: ResourceValue,
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
    #[serde(default)]
    error_estimator: ErrorEstimator,
    error_weights: Option<Vec<Scalar>>,
    second_error_weights: Option<Vec<Scalar>>,
    dense_coefficients: Option<Vec<Vec<Scalar>>>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum ErrorEstimator {
    /// Difference between primary and embedded weights; coefficients sum to zero.
    #[default]
    EmbeddedDifference,
    /// Direct weighted stage combination used by specialized upstream methods.
    StageCombination,
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

        self.validate_estimator(
            "error_weights",
            self.error_weights.as_deref(),
            stages,
            self.error_estimator == ErrorEstimator::EmbeddedDifference,
        )?;
        self.validate_estimator(
            "second_error_weights",
            self.second_error_weights.as_deref(),
            stages,
            true,
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
        if self.error_weights.is_none() && self.error_estimator != ErrorEstimator::default() {
            return Err("error_estimator requires error_weights".into());
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
        require_zero_sum: bool,
    ) -> Result<(), String> {
        let Some(estimator) = estimator else {
            return Ok(());
        };
        if estimator.len() != stages {
            return Err(format!("{label} must contain one entry per stage"));
        }
        let estimator = values(estimator, label)?;
        let sum = estimator.iter().sum::<f64>();
        if require_zero_sum && !approximately_equal(sum, 0.0) {
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

fn evaluate_numeric_expression(expression: &str) -> Result<f64, String> {
    fn evaluate(expression: &Expr) -> Result<f64, String> {
        match expression {
            Expr::Lit(ExprLit {
                lit: Lit::Float(value),
                ..
            }) => value
                .base10_parse::<f64>()
                .map_err(|error| error.to_string()),
            Expr::Lit(ExprLit {
                lit: Lit::Int(value),
                ..
            }) => value
                .base10_parse::<i64>()
                .map(|value| value as f64)
                .map_err(|error| error.to_string()),
            Expr::Unary(ExprUnary {
                op: UnOp::Neg(_),
                expr,
                ..
            }) => Ok(-evaluate(expr)?),
            Expr::Paren(expression) => evaluate(&expression.expr),
            Expr::Group(expression) => evaluate(&expression.expr),
            Expr::Binary(ExprBinary {
                left, op, right, ..
            }) => {
                let left = evaluate(left)?;
                let right = evaluate(right)?;
                match op {
                    BinOp::Add(_) => Ok(left + right),
                    BinOp::Sub(_) => Ok(left - right),
                    BinOp::Mul(_) => Ok(left * right),
                    BinOp::Div(_) if right != 0.0 => Ok(left / right),
                    BinOp::Div(_) => Err("division by zero".into()),
                    _ => Err("only +, -, *, and / are supported".into()),
                }
            }
            _ => Err("expected a numeric literal or arithmetic expression".into()),
        }
    }

    let expression = syn::parse_str::<Expr>(expression)
        .map_err(|error| format!("invalid numeric expression `{expression}`: {error}"))?;
    let value = evaluate(&expression)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("coefficient must be finite".into())
    }
}

impl ResourceValue {
    fn as_f64(&self) -> Result<f64, String> {
        match self {
            Self::Integer(value) => Ok(*value as f64),
            Self::Float(value) if value.is_finite() => Ok(*value),
            Self::Float(_) => Err("coefficient must be finite".into()),
            Self::Text(value) => evaluate_numeric_expression(value),
            Self::Bool(_) => Err("expected a numeric coefficient".into()),
        }
    }

    fn as_integer(&self) -> Result<i64, String> {
        let value = match self {
            Self::Integer(value) => *value,
            Self::Text(value) => {
                let value = value.replace('_', "");
                value
                    .parse::<i64>()
                    .map_err(|_| format!("invalid integer coefficient `{value}`"))?
            }
            _ => return Err("expected an integer coefficient".into()),
        };
        Ok(value)
    }

    fn as_bool(&self) -> Result<bool, String> {
        match self {
            Self::Bool(value) => Ok(*value),
            Self::Text(value) if value == "true" => Ok(true),
            Self::Text(value) if value == "false" => Ok(false),
            _ => Err("expected a Boolean coefficient".into()),
        }
    }
}

fn emit_resource_value(
    value: &ResourceValue,
    value_type: CoefficientType,
) -> Result<TokenStream2, String> {
    match value_type {
        CoefficientType::F64 => {
            let value = literal(value.as_f64()?);
            Ok(quote! { #value })
        }
        CoefficientType::Usize => {
            let value = value.as_integer()?;
            let value = usize::try_from(value)
                .map_err(|_| "usize coefficient must be non-negative".to_string())?;
            let value = Literal::usize_suffixed(value);
            Ok(quote! { #value })
        }
        CoefficientType::I32 => {
            let value = i32::try_from(value.as_integer()?)
                .map_err(|_| "i32 coefficient is out of range".to_string())?;
            let value = Literal::i32_suffixed(value);
            Ok(quote! { #value })
        }
        CoefficientType::Bool => {
            let value = value.as_bool()?;
            Ok(quote! { #value })
        }
    }
}

fn coefficient_type(value_type: CoefficientType) -> TokenStream2 {
    match value_type {
        CoefficientType::F64 => quote! { f64 },
        CoefficientType::Usize => quote! { usize },
        CoefficientType::I32 => quote! { i32 },
        CoefficientType::Bool => quote! { bool },
    }
}

fn required<'a, T>(value: &'a Option<T>, label: &str) -> Result<&'a T, String> {
    value
        .as_ref()
        .ok_or_else(|| format!("coefficient `{label}` is required for this constant kind"))
}

fn ensure_absent<T>(value: &Option<T>, label: &str) -> Result<(), String> {
    if value.is_some() {
        Err(format!(
            "coefficient `{label}` is not valid for this constant kind"
        ))
    } else {
        Ok(())
    }
}

fn expand_coefficient_constant(
    constant: CoefficientConstant,
    visibility: &Visibility,
    crate_path: &SynPath,
) -> Result<TokenStream2, String> {
    let name = syn::parse_str::<Ident>(&constant.name)
        .map_err(|_| format!("invalid Rust constant name `{}`", constant.name))?;
    let value_type = coefficient_type(constant.value_type);
    let label = constant.name;

    match constant.kind {
        CoefficientKind::Scalar => {
            ensure_absent(&constant.values, "values")?;
            ensure_absent(&constant.rows, "rows")?;
            ensure_absent(&constant.stages, "stages")?;
            let value =
                emit_resource_value(required(&constant.value, "value")?, constant.value_type)?;
            Ok(quote! { #visibility const #name: #value_type = #value; })
        }
        CoefficientKind::Slice | CoefficientKind::Array => {
            ensure_absent(&constant.value, "value")?;
            ensure_absent(&constant.rows, "rows")?;
            ensure_absent(&constant.stages, "stages")?;
            let values = required(&constant.values, "values")?
                .iter()
                .map(|value| emit_resource_value(value, constant.value_type))
                .collect::<Result<Vec<_>, _>>()?;
            if matches!(constant.kind, CoefficientKind::Slice) {
                Ok(quote! { #visibility const #name: &[#value_type] = &[#(#values),*]; })
            } else {
                let len = values.len();
                Ok(quote! { #visibility const #name: [#value_type; #len] = [#(#values),*]; })
            }
        }
        CoefficientKind::Rows | CoefficientKind::Matrix => {
            if !matches!(constant.value_type, CoefficientType::F64) {
                return Err(format!(
                    "{label}: rows and matrices currently require type f64"
                ));
            }
            ensure_absent(&constant.value, "value")?;
            ensure_absent(&constant.values, "values")?;
            ensure_absent(&constant.stages, "stages")?;
            let rows = required(&constant.rows, "rows")?;
            let emitted_rows = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|value| emit_resource_value(value, constant.value_type))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            if matches!(constant.kind, CoefficientKind::Rows) {
                let rows = emitted_rows.iter().map(|row| quote! { &[#(#row),*] });
                Ok(quote! { #visibility const #name: &[&[f64]] = &[#(#rows),*]; })
            } else {
                let row_count = emitted_rows.len();
                let column_count = emitted_rows.first().map_or(0, Vec::len);
                if emitted_rows.iter().any(|row| row.len() != column_count) {
                    return Err(format!("{label}: matrix rows must have equal lengths"));
                }
                let rows = emitted_rows.iter().map(|row| quote! { [#(#row),*] });
                Ok(quote! {
                    #visibility const #name: [[f64; #column_count]; #row_count] = [#(#rows),*];
                })
            }
        }
        CoefficientKind::LazyStageSlice => {
            if !matches!(constant.value_type, CoefficientType::F64) {
                return Err(format!("{label}: lazy stages require type f64"));
            }
            ensure_absent(&constant.value, "value")?;
            ensure_absent(&constant.values, "values")?;
            ensure_absent(&constant.rows, "rows")?;
            let stages = required(&constant.stages, "stages")?
                .iter()
                .map(|stage| {
                    let node = emit_resource_value(&stage.node, CoefficientType::F64)?;
                    let weights = stage
                        .weights
                        .iter()
                        .map(|weight| {
                            let index = weight.index;
                            let value = emit_resource_value(&weight.value, CoefficientType::F64)?;
                            Ok(quote! { (#index, #value) })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    Ok(quote! {
                        #crate_path::algorithms::explicit::general::LazyDenseStage::new(
                            #node,
                            &[#(#weights),*],
                        )
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(quote! {
                #visibility const #name: &[#crate_path::algorithms::explicit::general::LazyDenseStage] =
                    &[#(#stages),*];
            })
        }
    }
}

/// Defines typed coefficient constants from a declarative TOML resource.
///
/// The resource is parsed and validated during macro expansion. The source
/// file is tracked with `include_str!`, while the compiled crate contains only
/// the resulting constants and performs no runtime parsing or file I/O.
#[proc_macro]
pub fn define_coefficients_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as CoefficientMacroInput);
    match expand_coefficients(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_coefficients(input: CoefficientMacroInput) -> Result<TokenStream2, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let relative_path = input.path.value();
    let path = manifest_dir.join(&relative_path);
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))?;
    let resource: CoefficientResource = toml::from_str(&source)
        .map_err(|error| format!("invalid coefficient TOML in `{}`: {error}", path.display()))?;
    if resource.schema_version != 1 {
        return Err(format!(
            "unsupported schema_version {}; expected 1",
            resource.schema_version
        ));
    }
    if resource.description.trim().is_empty() {
        return Err("coefficient resource description must not be empty".into());
    }
    if resource.constants.is_empty() {
        return Err("coefficient resource must define at least one constant".into());
    }

    let mut names = std::collections::HashSet::new();
    let mut constants = Vec::with_capacity(resource.constants.len());
    for constant in resource.constants {
        if !names.insert(constant.name.clone()) {
            return Err(format!(
                "duplicate coefficient constant `{}`",
                constant.name
            ));
        }
        constants.push(expand_coefficient_constant(
            constant,
            &input.visibility,
            &input.crate_path,
        )?);
    }

    let source_path = input.path;
    let description = resource.description;
    Ok(quote! {
        const _: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path));
        #[doc = #description]
        const _: () = ();
        #(#constants)*
    })
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

    #[test]
    fn direct_stage_estimators_require_an_explicit_schema_marker() {
        let direct = HEUN.replace(
            "error_weights = [\"-1/2\", \"1/2\"]",
            "error_estimator = \"stage-combination\"\nerror_weights = [\"1\", \"0\"]",
        );
        parse(&direct).validate("FileHeun").unwrap();

        let unmarked = HEUN.replace(
            "error_weights = [\"-1/2\", \"1/2\"]",
            "error_weights = [\"1\", \"0\"]",
        );
        assert!(parse(&unmarked).validate("FileHeun").is_err());
    }
}
