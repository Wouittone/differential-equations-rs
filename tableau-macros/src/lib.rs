//! Compile-time validation and expansion of declarative solver tableau resources.

use differential_equations_tableau_core::{
    RungeKuttaKind, parse_eserk_tableau, parse_irkn_tableau, parse_low_storage_tableau,
    parse_multistep_tableau, parse_rkn_tableau, parse_rock2_tableau, parse_rock4_tableau,
    parse_serk2_tableau, parse_symplectic_tableau, parse_tableau,
};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::path::PathBuf;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitStr, Path as SynPath, Token, Visibility, parse_macro_input, parse_quote};

struct MacroInput {
    visibility: Visibility,
    name: Ident,
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

struct DegreeStaticTableauInput {
    visibility: Visibility,
    static_name: Ident,
    method_name: LitStr,
    degree: syn::LitInt,
    path: LitStr,
    crate_path: SynPath,
}

struct OrderDegreeStaticTableauInput {
    visibility: Visibility,
    static_name: Ident,
    method_name: LitStr,
    order: syn::LitInt,
    degree: syn::LitInt,
    path: LitStr,
    crate_path: SynPath,
}

impl Parse for OrderDegreeStaticTableauInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let visibility = input.parse()?;
        let static_name = input.parse()?;
        input.parse::<Token![,]>()?;
        let method_name = input.parse()?;
        input.parse::<Token![,]>()?;
        let order = input.parse()?;
        input.parse::<Token![,]>()?;
        let degree = input.parse()?;
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
                "expected `visibility STATIC_NAME, \"MethodName\", order, degree, \"path/to/tableau.json\"` with optional `, crate = path`",
            ));
        }
        Ok(Self {
            visibility,
            static_name,
            method_name,
            order,
            degree,
            path,
            crate_path,
        })
    }
}

impl Parse for DegreeStaticTableauInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let visibility = input.parse()?;
        let static_name = input.parse()?;
        input.parse::<Token![,]>()?;
        let method_name = input.parse()?;
        input.parse::<Token![,]>()?;
        let degree = input.parse()?;
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
                "expected `visibility STATIC_NAME, \"MethodName\", degree, \"path/to/tableau.json\"` with optional `, crate = path`",
            ));
        }
        Ok(Self {
            visibility,
            static_name,
            method_name,
            degree,
            path,
            crate_path,
        })
    }
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

/// Defines a lazy canonical linear multistep tableau from a JSON resource.
///
/// Accepts `visibility STATIC_NAME, "MethodName", "resource.json"` and an
/// optional `crate = local_name`. Validates coefficients and polynomial order
/// conditions at compile time, then embeds only source text for lazy parsing.
#[proc_macro]
pub fn define_multistep_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    let result =
        read_static_resource(&input).and_then(|source| expand_multistep_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines a lazy variable-step two-step tableau from a JSON resource.
///
/// Coefficient functions, the startup formula, and the local-defect estimator
/// are validated during macro expansion. Only source text is embedded and the
/// tableau is materialized on first inspection or use.
#[proc_macro]
pub fn define_variable_multistep_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    let result = read_static_resource(&input).and_then(|source| {
        differential_equations_tableau_core::parse_variable_multistep_tableau(
            &source,
            &input.method_name.value(),
        )
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
        Ok(emit_lazy_static(
            input,
            quote!(LazyVariableMultistepTableau),
            quote!(parse_variable_multistep_tableau),
        ))
    });
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines a lazy MRI-GARK tableau validated from a JSON resource.
#[proc_macro]
pub fn define_mri_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    let result = read_static_resource(&input).and_then(|source| {
        differential_equations_tableau_core::parse_mri_tableau(&source, &input.method_name.value())
            .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
        Ok(emit_lazy_static(
            input,
            quote!(LazyMriTableau),
            quote!(parse_mri_tableau),
        ))
    });
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines a lazy MIS coupling tableau validated from a JSON resource.
#[proc_macro]
pub fn define_mis_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    let result = read_static_resource(&input).and_then(|source| {
        differential_equations_tableau_core::parse_mis_tableau(&source, &input.method_name.value())
            .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
        Ok(emit_lazy_static(
            input,
            quote!(LazyMisTableau),
            quote!(parse_mis_tableau),
        ))
    });
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_multistep_source(
    input: StaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    parse_multistep_tableau(source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_lazy_static(
        input,
        quote!(LazyMultistepTableau),
        quote!(parse_multistep_tableau),
    ))
}

/// Defines a lazy Rosenbrock or hybrid tableau, validated at compile time.
///
/// Accepts `visibility STATIC, "MethodName", "resource.json", crate = local_name`.
/// Only source text is embedded; coefficient arrays are parsed on first use.
#[proc_macro]
pub fn define_rosenbrock_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    match read_static_resource(&input).and_then(|source| expand_rosenbrock_source(input, &source)) {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines the lazy low-storage Rosenbrock 2/3 pair from a JSON resource.
///
/// The resource is validated during macro expansion and embedded as source
/// text, then materialized only when one of the pair's solvers is first used.
#[proc_macro]
pub fn define_rosenbrock_pair_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    let result = read_static_resource(&input).and_then(|source| {
        differential_equations_tableau_core::parse_rosenbrock_pair_tableau(
            &source,
            &input.method_name.value(),
        )
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
        Ok(emit_lazy_static(
            input,
            quote!(LazyRosenbrockPairTableau),
            quote!(parse_rosenbrock_pair_tableau),
        ))
    });
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines one lazily materialized RKN tableau from a validated JSON resource.
///
/// The resource is parsed during macro expansion for compile-time diagnostics.
/// The expansion embeds only its source text and parses it once on first use.
#[proc_macro]
pub fn define_rkn_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    let result = read_static_resource(&input).and_then(|source| expand_rkn_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines one lazily materialized improved RKN tableau from validated JSON.
#[proc_macro]
pub fn define_irkn_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    let result = read_static_resource(&input).and_then(|source| expand_irkn_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines one lazily materialized low-storage Runge--Kutta tableau.
///
/// The resource is parsed and validated during macro expansion. The emitted
/// static embeds only its source text and materializes coefficients on first
/// inspection or use.
#[proc_macro]
pub fn define_low_storage_rk_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StaticTableauInput);
    let result = read_static_resource(&input)
        .and_then(|source| expand_low_storage_rk_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines one independently lazy, degree-specific ROCK2 tableau.
///
/// The JSON resource is fully parsed and validated during macro expansion.
/// The expansion embeds the original source with `include_str!` and parses it
/// once, only when this particular degree is inspected or selected.
#[proc_macro]
pub fn define_rock2_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DegreeStaticTableauInput);
    let result =
        read_degree_static_resource(&input).and_then(|source| expand_rock2_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_rock2_source(
    input: DegreeStaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    let degree = input
        .degree
        .base10_parse::<usize>()
        .map_err(|error| format!("invalid ROCK2 degree: {error}"))?;
    parse_rock2_tableau(source, &input.method_name.value(), degree)
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;

    Ok(emit_degree_lazy_static(
        input,
        degree,
        quote!(LazyRock2Tableau),
        quote!(parse_rock2_tableau),
    ))
}

/// Defines one independently lazy, degree-specific ROCK4 tableau.
///
/// The shared parser validates the recurrence, finishing tableau, fourth-order
/// primary method, and third-order embedded companion during macro expansion.
#[proc_macro]
pub fn define_rock4_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DegreeStaticTableauInput);
    let result =
        read_degree_static_resource(&input).and_then(|source| expand_rock4_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_rock4_source(
    input: DegreeStaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    let degree = input
        .degree
        .base10_parse::<usize>()
        .map_err(|error| format!("invalid ROCK4 degree: {error}"))?;
    parse_rock4_tableau(source, &input.method_name.value(), degree)
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_degree_lazy_static(
        input,
        degree,
        quote!(LazyRock4Tableau),
        quote!(parse_rock4_tableau),
    ))
}

/// Defines one independently lazy, degree-specific SERK2 tableau.
///
/// The shared parser reconstructs the recurrence and validates its second-order
/// conditions during macro expansion. The emitted static embeds only the JSON
/// source and parses it on first use.
#[proc_macro]
pub fn define_serk2_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DegreeStaticTableauInput);
    let result =
        read_degree_static_resource(&input).and_then(|source| expand_serk2_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_serk2_source(
    input: DegreeStaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    let degree = input
        .degree
        .base10_parse::<usize>()
        .map_err(|error| format!("invalid SERK2 degree: {error}"))?;
    parse_serk2_tableau(source, &input.method_name.value(), degree)
        .map_err(|error| format!("invalid tableau '{}': {error}", input.path.value()))?;
    Ok(emit_degree_lazy_static(
        input,
        degree,
        quote!(LazySerk2Tableau),
        quote!(parse_serk2_tableau),
    ))
}

/// Defines one independently lazy, degree-specific ESERK tableau.
///
/// The shared parser validates the stabilized recurrence and extrapolation
/// moments during macro expansion. The expansion embeds only the source JSON
/// and parses it when this particular degree is first selected or inspected.
#[proc_macro]
pub fn define_eserk_tableau_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as OrderDegreeStaticTableauInput);
    let result = read_order_degree_static_resource(&input)
        .and_then(|source| expand_eserk_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_eserk_source(
    input: OrderDegreeStaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    let order = input
        .order
        .base10_parse::<usize>()
        .map_err(|error| format!("invalid ESERK order: {error}"))?;
    let degree = input
        .degree
        .base10_parse::<usize>()
        .map_err(|error| format!("invalid ESERK degree: {error}"))?;
    parse_eserk_tableau(source, &input.method_name.value(), order, degree)
        .map_err(|error| format!("invalid tableau '{}': {error}", input.path.value()))?;

    let visibility = input.visibility;
    let static_name = input.static_name;
    let method_name = input.method_name;
    let source_path = input.path;
    let crate_path = input.crate_path;
    Ok(quote! {
        #visibility static #static_name: #crate_path::tableau::LazyEserkTableau =
            ::std::sync::LazyLock::new(|| {
                #crate_path::tableau::parse_eserk_tableau(
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                    #method_name,
                    #order,
                    #degree,
                )
            });
    })
}

fn emit_degree_lazy_static(
    input: DegreeStaticTableauInput,
    degree: usize,
    lazy_type: TokenStream2,
    parser: TokenStream2,
) -> TokenStream2 {
    let visibility = input.visibility;
    let static_name = input.static_name;
    let method_name = input.method_name;
    let source_path = input.path;
    let crate_path = input.crate_path;
    quote! {
        #visibility static #static_name: #crate_path::tableau::#lazy_type =
            ::std::sync::LazyLock::new(|| {
                #crate_path::tableau::#parser(
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                    #method_name,
                    #degree,
                )
            });
    }
}

fn read_degree_static_resource(input: &DegreeStaticTableauInput) -> Result<String, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let path = manifest_dir.join(input.path.value());
    std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))
}

fn read_order_degree_static_resource(
    input: &OrderDegreeStaticTableauInput,
) -> Result<String, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let path = manifest_dir.join(input.path.value());
    std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))
}

#[cfg(test)]
mod rock2_tests {
    use super::*;

    fn source() -> &'static str {
        r#"{
            "name":"ROCK2",
            "description":"Macro test",
            "kind":"rock2",
            "order":2,
            "degree":2,
            "recurrence":{
                "first":"0.09326607661089206",
                "stages":[["0.1268473641290642","0.02103378190528467"]]
            },
            "finishing":{
                "first":"0.3889624104727243",
                "second":"0.4219428123056774"
            }
        }"#
    }

    fn input() -> DegreeStaticTableauInput {
        syn::parse_str("pub TABLEAU, \"ROCK2\", 2, \"degree.json\", crate = renamed").unwrap()
    }

    #[test]
    fn expansion_embeds_only_the_validated_source() {
        let tokens = expand_rock2_source(input(), source()).unwrap().to_string();
        assert!(tokens.contains("LazyRock2Tableau"));
        assert!(tokens.contains("LazyLock"));
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("parse_rock2_tableau"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("0.09326607661089206"));
        assert!(!tokens.contains("0.3889624104727243"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn expansion_reports_resource_identity_and_shape_errors() {
        for invalid in [
            source().replace(r#""degree":2"#, r#""degree":3"#),
            source().replace(r#""name":"ROCK2""#, r#""name":"ROCK4""#),
            source().replace(
                r#"["0.1268473641290642","0.02103378190528467"]"#,
                r#"["0.1268473641290642"]"#,
            ),
        ] {
            let error = expand_rock2_source(input(), &invalid).unwrap_err();
            assert!(error.contains("degree.json"), "{error}");
        }
    }
}

#[cfg(test)]
mod rock4_tests {
    use super::*;

    fn source() -> &'static str {
        r#"{
            "name":"ROCK4",
            "description":"Macro test",
            "kind":"rock4",
            "order":4,
            "embedded_order":3,
            "degree":1,
            "recurrence":{"first":"0.1762962957651941","stages":[]},
            "finishing":{
                "A":[[],["-0.149352078672699"],["0.629768962985252","-0.35520106157365"],["0.0146745996307541","-0.0558517281602565","0.590312931352706"]],
                "b":["0.934502625489809","-0.426556402801135","-0.428612609028723","0.744370090574855"],
                "b_hat":["1.1350997211054","-0.58433336098972","-0.319172911177732","0.482853558185876","0.109256697110981"]
            }
        }"#
    }

    fn input() -> DegreeStaticTableauInput {
        syn::parse_str("pub TABLEAU, \"ROCK4\", 1, \"rock4.json\", crate = renamed").unwrap()
    }

    #[test]
    fn expansion_uses_the_shared_lazy_parser_without_emitting_coefficients() {
        let tokens = expand_rock4_source(input(), source()).unwrap().to_string();
        for required in [
            "LazyRock4Tableau",
            "LazyLock",
            "include_str",
            "parse_rock4_tableau",
            "renamed",
        ] {
            assert!(tokens.contains(required), "missing {required}: {tokens}");
        }
        assert!(!tokens.contains("0.1762962957651941"));
        assert!(!tokens.contains("0.934502625489809"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn expansion_rejects_an_invalid_order_formula_with_its_path() {
        let invalid = source().replace("0.934502625489809", "0.9");
        let error = expand_rock4_source(input(), &invalid).unwrap_err();
        assert!(error.contains("rock4.json"), "{error}");
        assert!(error.contains("order condition"), "{error}");
    }
}

#[cfg(test)]
mod serk2_tests {
    use super::*;

    fn source() -> &'static str {
        r#"{
            "name":"SERK2",
            "description":"Macro test",
            "kind":"serk2",
            "order":2,
            "degree":2,
            "alpha":"2.5 / 4",
            "subdivisions":1,
            "weights":["1.32","-0.96","0.64"]
        }"#
    }

    fn input() -> DegreeStaticTableauInput {
        syn::parse_str("pub TABLEAU, \"SERK2\", 2, \"serk2.json\", crate = renamed").unwrap()
    }

    #[test]
    fn expansion_uses_the_shared_lazy_parser_without_emitting_coefficients() {
        let tokens = expand_serk2_source(input(), source()).unwrap().to_string();
        for required in [
            "LazySerk2Tableau",
            "LazyLock",
            "include_str",
            "parse_serk2_tableau",
            "renamed",
        ] {
            assert!(tokens.contains(required), "missing {required}: {tokens}");
        }
        assert!(!tokens.contains("1.32"));
        assert!(!tokens.contains("- 0.96"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn expansion_rejects_invalid_metadata_and_order_conditions() {
        for invalid in [
            source().replace(r#""degree":2"#, r#""degree":3"#),
            source().replace(r#""name":"SERK2""#, r#""name":"OTHER""#),
            source().replace("0.64", "0.63"),
        ] {
            let error = expand_serk2_source(input(), &invalid).unwrap_err();
            assert!(error.contains("serk2.json"), "{error}");
        }
    }
}

#[cfg(test)]
mod eserk_tests {
    use super::*;

    fn source() -> &'static str {
        r#"{
            "name":"ESERK4",
            "description":"Synthetic degree-one extrapolated stabilized method",
            "kind":"eserk",
            "order":4,
            "embedded_order":3,
            "degree":1,
            "internal_degree":1,
            "alpha":"1",
            "subdivisions":4,
            "solution_combination":[-1,24,-81,64],
            "error_combination":[-1,12,-27,16],
            "combination_denominator":"6",
            "weights":["0","1"]
        }"#
    }

    fn input() -> OrderDegreeStaticTableauInput {
        syn::parse_str("pub TABLEAU, \"ESERK4\", 4, 1, \"eserk.json\", crate = renamed").unwrap()
    }

    #[test]
    fn expansion_embeds_only_the_validated_source() {
        let tokens = expand_eserk_source(input(), source()).unwrap().to_string();
        for required in [
            "LazyEserkTableau",
            "LazyLock",
            "include_str",
            "parse_eserk_tableau",
            "renamed",
        ] {
            assert!(tokens.contains(required), "missing {required}: {tokens}");
        }
        for forbidden in ["0.129", "Vec", "from (["] {
            assert!(!tokens.contains(forbidden), "emitted {forbidden}: {tokens}");
        }
    }

    #[test]
    fn expansion_rejects_invalid_metadata_and_moments() {
        for invalid in [
            source().replace(r#""order":4"#, r#""order":5"#),
            source().replace(r#""degree":1"#, r#""degree":2"#),
            source().replace("[-1,24,-81,64]", "[-1,24,-81,63]"),
            source().replace("[-1,12,-27,16]", "[0,0,0,0]"),
        ] {
            let error = expand_eserk_source(input(), &invalid).unwrap_err();
            assert!(error.contains("eserk.json"), "{error}");
        }
    }
}

/// Defines a named fixed or adaptive RKN solver from one JSON resource.
///
/// The generated zero-sized type implements `SecondOrderOdeAlgorithm` and
/// exposes its independently lazy tableau through `tableau()`.
#[proc_macro]
pub fn define_rkn_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as MacroInput);
    let result = read_algorithm_resource(&input)
        .and_then(|source| expand_rkn_algorithm_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines a named low-storage Runge--Kutta solver from one JSON resource.
///
/// The generated zero-sized type implements [`OdeAlgorithm`](https://docs.rs/differential-equations/latest/differential_equations/trait.OdeAlgorithm.html)
/// and exposes its independently lazy tableau through `tableau()`.
#[proc_macro]
pub fn define_low_storage_rk_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as MacroInput);
    let result = read_algorithm_resource(&input)
        .and_then(|source| expand_low_storage_rk_algorithm_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_low_storage_rk_algorithm_source(
    input: MacroInput,
    source: &str,
) -> Result<TokenStream2, String> {
    let tableau = parse_low_storage_tableau(source, &input.name.to_string())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    let visibility = input.visibility;
    let name = input.name;
    let static_name = format_ident!("__{}_TABLEAU", name.to_string().to_uppercase());
    let source_path = input.path;
    let crate_path = input.crate_path;
    let description = tableau.description();
    Ok(quote! {
        static #static_name: #crate_path::tableau::LazyLowStorageRungeKuttaTableau =
            ::std::sync::LazyLock::new(|| {
                #crate_path::tableau::parse_low_storage_tableau(
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                    stringify!(#name),
                )
            });

        #[doc = #description]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #[allow(
            non_camel_case_types,
            reason = "preserve the published low-storage algorithm name"
        )]
        #visibility struct #name;

        impl #name {
            /// Returns the lazily initialized, compile-time-validated tableau.
            #visibility fn tableau(
                self,
            ) -> ::std::result::Result<
                &'static #crate_path::tableau::LowStorageRungeKuttaTableau,
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
            ) -> ::std::result::Result<#crate_path::Solution, #crate_path::SolveError>
            where
                F: #crate_path::OdeFunction<P>,
            {
                #crate_path::OdeAlgorithm::solve_validated(
                    &#crate_path::solvers::explicit::ResourceLowStorageRungeKutta::new(
                        &#static_name,
                    ),
                    problem,
                    options,
                )
            }
        }
    })
}

fn expand_rkn_algorithm_source(input: MacroInput, source: &str) -> Result<TokenStream2, String> {
    let tableau = parse_rkn_tableau(source, &input.name.to_string())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    let visibility = input.visibility;
    let name = input.name;
    let static_name = format_ident!("__{}_TABLEAU", name.to_string().to_uppercase());
    let source_path = input.path;
    let crate_path = input.crate_path;
    let description = tableau.description();
    Ok(quote! {
        static #static_name: #crate_path::tableau::LazyRungeKuttaNystromTableau =
            ::std::sync::LazyLock::new(|| {
                #crate_path::tableau::parse_rkn_tableau(
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                    stringify!(#name),
                )
            });

        #[doc = #description]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #visibility struct #name;

        impl #name {
            /// Returns the lazily initialized, compile-time-validated tableau.
            #visibility fn tableau(
                self,
            ) -> ::std::result::Result<
                &'static #crate_path::tableau::RungeKuttaNystromTableau,
                #crate_path::tableau::TableauError,
            > {
                #crate_path::tableau::load_tableau(&#static_name)
            }
        }

        impl #crate_path::solvers::second_order::SecondOrderOdeAlgorithm for #name {
            fn solve_validated<F, P>(
                &self,
                problem: &#crate_path::solvers::second_order::SecondOrderOdeProblem<F, P>,
                options: &#crate_path::SolveOptions,
            ) -> ::std::result::Result<
                #crate_path::solvers::second_order::SecondOrderSolution,
                #crate_path::solvers::second_order::SecondOrderSolveError,
            >
            where
                F: #crate_path::solvers::second_order::SecondOrderFunction<P>,
            {
                #crate_path::solvers::second_order::SecondOrderOdeAlgorithm::solve_validated(
                    &#crate_path::solvers::second_order::ResourceRungeKuttaNystrom::new(
                        &#static_name,
                    ),
                    problem,
                    options,
                )
            }
        }
    })
}

fn expand_rkn_source(input: StaticTableauInput, source: &str) -> Result<TokenStream2, String> {
    parse_rkn_tableau(source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_lazy_static(
        input,
        quote!(LazyRungeKuttaNystromTableau),
        quote!(parse_rkn_tableau),
    ))
}

fn expand_irkn_source(input: StaticTableauInput, source: &str) -> Result<TokenStream2, String> {
    parse_irkn_tableau(source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_lazy_static(
        input,
        quote!(LazyIrknTableau),
        quote!(parse_irkn_tableau),
    ))
}

fn expand_low_storage_rk_source(
    input: StaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    parse_low_storage_tableau(source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_lazy_static(
        input,
        quote!(LazyLowStorageRungeKuttaTableau),
        quote!(parse_low_storage_tableau),
    ))
}

#[cfg(test)]
mod low_storage_tests {
    use super::*;

    fn source() -> &'static str {
        r#"{"name":"Test2N","description":"Macro test","kind":"low-storage-runge-kutta","layout":"two-n","order":2,"A":["-1/2"],"b":["1/2",1],"c":["1/2"]}"#
    }

    #[test]
    fn static_expansion_embeds_only_validated_source() {
        let input: StaticTableauInput =
            syn::parse_str("pub TABLEAU, \"Test2N\", \"method.json\", crate = renamed").unwrap();
        let tokens = expand_low_storage_rk_source(input, source())
            .unwrap()
            .to_string();
        assert!(tokens.contains("LazyLowStorageRungeKuttaTableau"));
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("LazyLock"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("0.5"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn algorithm_expansion_is_one_resource_backed_solver_value() {
        let input = MacroInput {
            visibility: parse_quote!(pub),
            name: parse_quote!(Test2N),
            path: LitStr::new("method.json", proc_macro2::Span::call_site()),
            crate_path: parse_quote!(renamed),
        };
        let tokens = expand_low_storage_rk_algorithm_source(input, source())
            .unwrap()
            .to_string();
        assert!(tokens.contains("ResourceLowStorageRungeKutta"));
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("LazyLock"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("0.5"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn expansion_rejects_invalid_recurrence_with_resource_path() {
        let input: StaticTableauInput =
            syn::parse_str("TABLEAU, \"Test2N\", \"method.json\"").unwrap();
        let invalid = source().replace("\"c\":[\"1/2\"]", "\"c\":[0]");
        let error = expand_low_storage_rk_source(input, &invalid).unwrap_err();
        assert!(error.contains("method.json"), "{error}");
        assert!(error.contains("stage-row sum"), "{error}");
    }
}

fn expand_rosenbrock_source(
    input: StaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    differential_equations_tableau_core::parse_rosenbrock_tableau(
        source,
        &input.method_name.value(),
    )
    .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_lazy_static(
        input,
        quote!(LazyRosenbrockTableau),
        quote!(parse_rosenbrock_tableau),
    ))
}

fn emit_lazy_static(
    input: StaticTableauInput,
    lazy_type: TokenStream2,
    parser: TokenStream2,
) -> TokenStream2 {
    let visibility = input.visibility;
    let static_name = input.static_name;
    let method_name = input.method_name;
    let source_path = input.path;
    let crate_path = input.crate_path;
    quote! {
        #[doc = concat!("Lazily parsed tableau `", #method_name, "`.")]
        #visibility static #static_name: #crate_path::tableau::#lazy_type =
            ::std::sync::LazyLock::new(|| {
                #crate_path::tableau::#parser(
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                    #method_name,
                )
            });
    }
}

fn read_static_resource(input: &StaticTableauInput) -> Result<String, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let path = manifest_dir.join(input.path.value());
    std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))
}

fn read_algorithm_resource(input: &MacroInput) -> Result<String, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let path = manifest_dir.join(input.path.value());
    std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))
}

fn expand_static(
    input: StaticTableauInput,
    expected_kind: RungeKuttaKind,
) -> Result<TokenStream2, String> {
    let source = read_static_resource(&input)?;
    expand_static_source(input, &source, expected_kind)
}

fn expand_static_source(
    input: StaticTableauInput,
    source: &str,
    expected_kind: RungeKuttaKind,
) -> Result<TokenStream2, String> {
    let tableau = parse_tableau(source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    if tableau.kind() != expected_kind {
        let expected = match expected_kind {
            RungeKuttaKind::Explicit => "explicit",
            RungeKuttaKind::Implicit => "implicit",
        };
        return Err(format!(
            "tableau `{}` is not an {expected} Runge--Kutta method",
            input.path.value(),
        ));
    }

    Ok(emit_lazy_static(
        input,
        quote!(LazyTableau),
        quote!(parse_tableau),
    ))
}

#[cfg(test)]
mod implicit_tests {
    use super::*;

    #[test]
    fn implicit_expansion_validates_predictors_and_embeds_only_source() {
        let source = r#"{"name":"Pair","description":"Implicit pair","kind":"implicit-runge-kutta","order":1,"embedded_order":2,"A":[[0,0],["1/2","1/2"]],"b":[0,1],"c":[0,1],"error":["-1/2","1/2"],"stage_predictors":[[],[1]]}"#;
        let input = || {
            syn::parse_str::<StaticTableauInput>(
                "pub TABLEAU, \"Pair\", \"pair.json\", crate = renamed",
            )
            .unwrap()
        };
        let tokens = expand_static_source(input(), source, RungeKuttaKind::Implicit)
            .unwrap()
            .to_string();
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("LazyLock"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("const "));
        for invalid in [
            source.replace("[[],[1]]", "[[],[1,0]]"),
            source.replace("[[],[1]]", "[[],[2]]"),
        ] {
            let error =
                expand_static_source(input(), &invalid, RungeKuttaKind::Implicit).unwrap_err();
            assert!(error.contains("pair.json"), "{error}");
            assert!(error.contains("stage_predictors"), "{error}");
        }
    }
}

/// Defines a symplectic composition from a validated JSON resource.
///
/// Like [`define_explicit_rk_from_file!`], accepts a visibility, method name,
/// resource path, and optional `crate = path`. The expansion contains an
/// `include_str!` resource and a per-method lazy tableau, not coefficient arrays.
#[proc_macro]
pub fn define_symplectic_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as MacroInput);
    match expand_symplectic(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_symplectic(input: MacroInput) -> Result<TokenStream2, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let path = manifest_dir.join(input.path.value());
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))?;
    expand_symplectic_source(input, &source)
}

fn expand_symplectic_source(input: MacroInput, source: &str) -> Result<TokenStream2, String> {
    parse_symplectic_tableau(source, &input.name.to_string())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    let visibility = input.visibility;
    let name = input.name;
    let static_name = format_ident!("__{}_TABLEAU", name.to_string().to_uppercase());
    let source_path = input.path;
    let crate_path = input.crate_path;
    Ok(quote! {
        static #static_name: #crate_path::tableau::LazySymplecticTableau = ::std::sync::LazyLock::new(|| {
            #crate_path::tableau::parse_symplectic_tableau(
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                stringify!(#name),
            )
        });
        #[doc = concat!("Resource-backed symplectic composition `", stringify!(#name), "`.")]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #visibility struct #name;
        impl #name {
            /// Returns this method's lazily initialized tableau.
            pub fn tableau() -> ::std::result::Result<&'static #crate_path::tableau::SymplecticTableau, #crate_path::tableau::TableauError> {
                #crate_path::tableau::load_tableau(&#static_name)
            }
        }
        impl #crate_path::solvers::second_order::SymplecticAlgorithm for #name {
            fn tableau() -> ::std::result::Result<&'static #crate_path::tableau::SymplecticTableau, #crate_path::tableau::TableauError> {
                Self::tableau()
            }
        }
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
                F: #crate_path::OdeFunction<P>,
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

#[cfg(test)]
mod symplectic_tests {
    use super::*;

    fn input() -> MacroInput {
        MacroInput {
            visibility: parse_quote!(pub),
            name: parse_quote!(TestComposition),
            path: LitStr::new("composition.json", proc_macro2::Span::call_site()),
            crate_path: parse_quote!(crate),
        }
    }

    const SOURCE: &str = r#"{"name":"TestComposition","description":"Macro test","kind":"symplectic-composition","order":2,"a":[1,0],"b":["1/2","1/2"]}"#;

    #[test]
    fn expansion_rejects_malformed_compositions_with_the_resource_path() {
        let source = SOURCE.replace("[1,0]", "[1]");
        let error = expand_symplectic_source(input(), &source).unwrap_err();
        assert!(error.contains("composition.json"), "{error}");
        assert!(error.contains("same non-zero stage count"), "{error}");
    }

    #[test]
    fn expansion_embeds_only_source_and_lazy_loading_not_coefficient_arrays() {
        let tokens = expand_symplectic_source(input(), SOURCE)
            .unwrap()
            .to_string();
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("LazyLock"));
        assert!(tokens.contains("parse_symplectic_tableau"));
        assert!(!tokens.contains("0.5"));
        assert!(!tokens.contains("const "));
    }
}

#[cfg(test)]
mod multistep_tests {

    use super::*;

    const SOURCE: &str = r#"{"name":"AB2","description":"Macro test","kind":"linear-multistep","order":2,"alpha":[1,-1,0],"beta":[0,"3/2","-1/2"]}"#;

    fn input() -> StaticTableauInput {
        syn::parse_str("pub FORMULA, \"AB2\", \"formula.json\", crate = renamed").unwrap()
    }

    #[test]
    fn multistep_expansion_embeds_source_not_coefficients() {
        let tokens = expand_multistep_source(input(), SOURCE)
            .unwrap()
            .to_string();
        assert!(tokens.contains("LazyMultistepTableau"));
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("parse_multistep_tableau"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("1.5"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn multistep_expansion_rejects_incorrect_order_conditions_with_the_path() {
        let invalid = SOURCE.replace("\"order\":2", "\"order\":3");
        let error = expand_multistep_source(input(), &invalid).unwrap_err();
        assert!(error.contains("formula.json"), "{error}");
        assert!(error.contains("order condition 3 failed"), "{error}");
    }

    #[test]
    fn multistep_expansion_validates_bdf_modifiers_before_runtime_use() {
        let source = r#"{"name":"AB2","description":"BDF macro test","kind":"backward-differentiation","order":2,"alpha":["3/2",-2,"1/2"],"beta":[1,0,0],"ndf_kappa":"-1/9"}"#;
        let tokens = expand_multistep_source(input(), source)
            .unwrap()
            .to_string();
        assert!(tokens.contains("include_str"));
        assert!(!tokens.contains("const "));
        let invalid = source.replace("\"-1/9\"", "1");
        let error = expand_multistep_source(input(), &invalid).unwrap_err();
        assert!(error.contains("formula.json"), "{error}");
        assert!(error.contains("NDF modifier"), "{error}");
    }
}

#[cfg(test)]
mod rosenbrock_tests {
    use super::*;

    const SOURCE: &str = r#"{"name":"Test","description":"Macro test","kind":"rosenbrock","order":2,"gamma":"1/2","A":[[0,0],[2,0]],"C":[[0,0],[-4,0]],"c":[0,1],"d":[0.5,-0.5],"b":[3,1],"btilde":[1,1],"H":[[1,0],[0,1]]}"#;

    fn input() -> StaticTableauInput {
        syn::parse_str("pub FORMULA, \"Test\", \"formula.json\", crate = renamed").unwrap()
    }

    #[test]
    fn rosenbrock_expansion_embeds_only_source_and_the_shared_parser() {
        let tokens = expand_rosenbrock_source(input(), SOURCE)
            .unwrap()
            .to_string();
        assert!(tokens.contains("LazyRosenbrockTableau"));
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("parse_rosenbrock_tableau"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("0.5"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn rosenbrock_validation_fails_during_macro_expansion() {
        for invalid in [
            SOURCE.replace("\"1/2\"", "0"),
            SOURCE.replace("\"1/2\"", "\"1/0\""),
            SOURCE.replace("[2,0]", "[2,1]"),
            SOURCE.replace("[0,1]]", "[0]]"),
            SOURCE.replace("\"btilde\":[1,1]", "\"btilde\":[1]"),
        ] {
            let error = expand_rosenbrock_source(input(), &invalid).unwrap_err();
            assert!(error.contains("formula.json"), "{error}");
        }
        let tokens =
            expand_rosenbrock_source(input(), &SOURCE.replace(",\"btilde\":[1,1]", "")).unwrap();
        assert!(tokens.to_string().contains("include_str"));
    }

    #[test]
    fn hybrid_validation_rejects_invalid_diagonals_at_compile_time() {
        let source = SOURCE
            .replace("\"rosenbrock\"", "\"hybrid-explicit-implicit\"")
            .replace("\"C\":[[0,0],[-4,0]]", "\"C\":[[0.5,0],[0,0.5]]")
            .replace("\"b\":[3,1]", "\"b\":[0.5,0.5]")
            .replace("\"btilde\":[1,1]", "\"btilde\":[1,-1]");
        let tokens = expand_rosenbrock_source(input(), &source)
            .unwrap()
            .to_string();
        assert!(tokens.contains("include_str"));
        assert!(!tokens.contains("0.5"));
        assert!(!tokens.contains("const "));
        let invalid = source.replace("\"C\":[[0.5,0],[0,0.5]]", "\"C\":[[0.5,0],[0,0]]");
        let error = expand_rosenbrock_source(input(), &invalid).unwrap_err();
        assert!(error.contains("formula.json"), "{error}");
        assert!(error.contains("must equal gamma"), "{error}");
    }
}

#[cfg(test)]
mod second_order_tests {
    use super::*;

    const RKN: &str = r#"{"name":"TestRkn","description":"RKN macro test","kind":"fixed-runge-kutta-nystrom","order":2,"A":[[0,0],["1/8",0]],"b":["1/2",0],"b_velocity":[0,1],"c":[0,"1/2"]}"#;
    const IRKN: &str = r#"{"name":"TestIrkn","description":"IRKN macro test","kind":"improved-runge-kutta-nystrom","order":3,"bootstrap_order":4,"bootstrap_seed":"previous-endpoint","velocity_history":["3/2","-1/2"],"c":["1/2"],"A":["1/8"],"velocity_weights":["2/3","5/6"],"history_weights":["1/3","5/12"]}"#;

    fn input(name: &str) -> StaticTableauInput {
        syn::parse_str(&format!(
            "pub TABLEAU, \"{name}\", \"second-order.json\", crate = renamed"
        ))
        .unwrap()
    }

    #[test]
    fn expansions_embed_only_source_and_use_the_shared_lazy_parsers() {
        let rkn = expand_rkn_source(input("TestRkn"), RKN)
            .unwrap()
            .to_string();
        assert!(rkn.contains("LazyRungeKuttaNystromTableau"));
        assert!(rkn.contains("parse_rkn_tableau"));
        assert!(rkn.contains("include_str"));
        assert!(rkn.contains("renamed"));
        assert!(!rkn.contains("0.125"));
        assert!(!rkn.contains("const "));

        let irkn = expand_irkn_source(input("TestIrkn"), IRKN)
            .unwrap()
            .to_string();
        assert!(irkn.contains("LazyIrknTableau"));
        assert!(irkn.contains("parse_irkn_tableau"));
        assert!(irkn.contains("include_str"));
        assert!(!irkn.contains("0.125"));
        assert!(!irkn.contains("const "));

        let algorithm_input: MacroInput =
            syn::parse_str("pub TestRkn, \"second-order.json\", crate = renamed").unwrap();
        let algorithm = expand_rkn_algorithm_source(algorithm_input, RKN)
            .unwrap()
            .to_string();
        assert!(algorithm.contains("ResourceRungeKuttaNystrom"));
        assert!(algorithm.contains("SecondOrderOdeAlgorithm"));
        assert!(algorithm.contains("include_str"));
        assert!(!algorithm.contains("0.125"));
        assert!(!algorithm.contains("const "));
    }

    #[test]
    fn malformed_second_order_resources_fail_during_expansion() {
        let rkn = RKN.replace("[\"1/8\",0]", "[\"1/8\",1]");
        let error = expand_rkn_source(input("TestRkn"), &rkn).unwrap_err();
        assert!(error.contains("second-order.json"), "{error}");
        assert!(error.contains("strictly lower triangular"), "{error}");

        let irkn = IRKN.replace("\"c\":[\"1/2\"]", "\"c\":[]");
        let error = expand_irkn_source(input("TestIrkn"), &irkn).unwrap_err();
        assert!(error.contains("second-order.json"), "{error}");
        assert!(error.contains("IRKN"), "{error}");
    }
}
