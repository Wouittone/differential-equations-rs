//! Compile-time expansion of declarative Runge--Kutta tableau resources.

use differential_equations_tableau_core::{
    RungeKuttaKind, parse_numeric_expression, parse_tableau,
};
use proc_macro::TokenStream;
use proc_macro2::{Literal, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use serde::Deserialize;
use std::path::PathBuf;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitStr, Path as SynPath, Token, Visibility, parse_macro_input, parse_quote};

struct MacroInput {
    visibility: Visibility,
    name: Ident,
    path: LitStr,
    crate_path: SynPath,
}

struct TableauDataInput {
    visibility: Visibility,
    path: LitStr,
    crate_path: SynPath,
}

struct StaticTableauInput {
    visibility: Visibility,
    static_name: Ident,
    method_name: LitStr,
    path: LitStr,
    crate_path: SynPath,
}

impl Parse for StaticTableauInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let visibility = input.parse()?;
        let static_name = input.parse()?;
        input.parse::<Token![,]>()?;
        let method_name = input.parse()?;
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
                "expected `visibility STATIC_NAME, \"MethodName\", \"path/to/tableau.json\"` with optional `, crate = path`",
            ));
        }
        Ok(Self {
            visibility,
            static_name,
            method_name,
            path,
            crate_path,
        })
    }
}

impl Parse for TableauDataInput {
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
                "expected `visibility, \"path/to/tableau-data.json\"` with optional `, crate = path`",
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
struct TableauDataResource {
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
                "expected `visibility Name, \"path/to/tableau.json\"` with optional `, crate = path`",
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

fn literal(value: f64) -> Literal {
    Literal::f64_suffixed(value)
}

impl ResourceValue {
    fn as_f64(&self) -> Result<f64, String> {
        match self {
            Self::Integer(value) => Ok(*value as f64),
            Self::Float(value) if value.is_finite() => Ok(*value),
            Self::Float(_) => Err("coefficient must be finite".into()),
            Self::Text(value) => parse_numeric_expression(value).map_err(|error| error.to_string()),
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
                        #crate_path::tableau::LazyDenseStage::new(
                            #node,
                            &[#(#weights),*],
                        )
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(quote! {
                #visibility const #name: &[#crate_path::tableau::LazyDenseStage] =
                    &[#(#stages),*];
            })
        }
    }
}

/// Defines typed method data from a declarative JSON tableau resource.
///
/// The resource is parsed and validated during macro expansion. The source
/// file is tracked with `include_str!`, while the compiled crate contains only
/// the resulting constants and performs no runtime parsing or file I/O.
#[proc_macro]
pub fn define_tableau_data_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as TableauDataInput);
    match expand_tableau_data(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_tableau_data(input: TableauDataInput) -> Result<TokenStream2, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let relative_path = input.path.value();
    let path = manifest_dir.join(&relative_path);
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))?;
    let resource: TableauDataResource = serde_json::from_str(&source)
        .map_err(|error| format!("invalid tableau data JSON in `{}`: {error}", path.display()))?;
    if resource.description.trim().is_empty() {
        return Err("tableau data description must not be empty".into());
    }
    if resource.constants.is_empty() {
        return Err("tableau data must define at least one constant".into());
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

/// Defines a zero-sized explicit Runge--Kutta algorithm from a JSON resource.
///
/// The path is relative to the invoking package's `CARGO_MANIFEST_DIR`. The
/// file is parsed and validated while compiling. The expansion embeds the
/// source text and parses it lazily when the algorithm is first used; it does
/// not generate Rust coefficient arrays.
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

/// Defines a lazy implicit Runge--Kutta tableau static from a JSON resource.
///
/// The resource is parsed and validated during macro expansion. The emitted
/// static embeds only its source text and materializes coefficients on first
/// use, allowing specialized implicit kernels to share the canonical serde
/// representation without generated Rust constants.
#[proc_macro]
pub fn define_implicit_rk_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    match expand_static(input, RungeKuttaKind::Implicit) {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines a lazy explicit Runge--Kutta tableau static from a JSON resource.
///
/// This form is intended for public compatibility aliases and specialized
/// kernels that cannot use the zero-sized algorithm generated by
/// [`define_explicit_rk_from_file!`].
#[proc_macro]
pub fn define_explicit_rk_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    match expand_static(input, RungeKuttaKind::Explicit) {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_static(
    input: StaticTableauInput,
    expected_kind: RungeKuttaKind,
) -> Result<TokenStream2, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let path = manifest_dir.join(input.path.value());
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))?;
    let tableau = parse_tableau(&source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", path.display()))?;
    if tableau.kind() != expected_kind {
        let expected = match expected_kind {
            RungeKuttaKind::Explicit => "explicit",
            RungeKuttaKind::Implicit => "implicit",
        };
        return Err(format!(
            "tableau `{}` is not an {expected} Runge--Kutta method",
            path.display(),
        ));
    }

    let visibility = input.visibility;
    let static_name = input.static_name;
    let method_name = input.method_name;
    let source_path = input.path;
    let crate_path = input.crate_path;
    Ok(quote! {
        #visibility static #static_name: #crate_path::tableau::LazyTableau =
            ::std::sync::LazyLock::new(|| {
                #crate_path::tableau::parse_tableau(
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                    #method_name,
                )
            });
    })
}

fn expand(input: MacroInput) -> Result<TokenStream2, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let relative_path = input.path.value();
    let path = manifest_dir.join(&relative_path);
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))?;
    let tableau = parse_tableau(&source, &input.name.to_string())
        .map_err(|error| format!("invalid tableau `{}`: {error}", path.display()))?;
    if tableau.kind() != RungeKuttaKind::Explicit {
        return Err(format!(
            "tableau `{}` is not an explicit Runge--Kutta method",
            path.display()
        ));
    }

    let visibility = input.visibility;
    let name = input.name;
    let static_name = format_ident!("__{}_TABLEAU", name.to_string().to_uppercase());
    let source_path = input.path;
    let crate_path = input.crate_path;
    let description = tableau.description();

    Ok(quote! {
        static #static_name: #crate_path::tableau::LazyTableau = ::std::sync::LazyLock::new(|| {
            #crate_path::tableau::parse_tableau(
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                stringify!(#name),
            )
        });

        #[doc = #description]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #visibility struct #name;

        impl #name {
            #[doc = "Returns the lazily parsed, compile-time-validated tableau."]
            #visibility fn tableau(
                self,
            ) -> ::std::result::Result<
                &'static #crate_path::tableau::RungeKuttaTableau,
                #crate_path::tableau::TableauError,
            > {
                #crate_path::tableau::load_tableau(&#static_name)
            }
        }

        impl #crate_path::OdeAlgorithm for #name {
            fn solve_validated<F, P>(
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
                #crate_path::OdeAlgorithm::solve_validated(
                    &#crate_path::tableau::ResourceExplicitRungeKutta::new(&#static_name),
                    problem,
                    options,
                )
            }
        }
    })
}
