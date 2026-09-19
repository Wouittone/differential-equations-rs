use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitStr, Path as SynPath, Token, Visibility, parse_quote};

pub(crate) struct MacroInput {
    pub(crate) visibility: Visibility,
    pub(crate) name: Ident,
    pub(crate) path: LitStr,
    pub(crate) crate_path: SynPath,
}

pub(crate) struct StaticTableauInput {
    pub(crate) visibility: Visibility,
    pub(crate) static_name: Ident,
    pub(crate) method_name: LitStr,
    pub(crate) path: LitStr,
    pub(crate) crate_path: SynPath,
}

pub(crate) struct DegreeStaticTableauInput {
    pub(crate) visibility: Visibility,
    pub(crate) static_name: Ident,
    pub(crate) method_name: LitStr,
    pub(crate) degree: syn::LitInt,
    pub(crate) path: LitStr,
    pub(crate) crate_path: SynPath,
}

pub(crate) struct OrderDegreeStaticTableauInput {
    pub(crate) visibility: Visibility,
    pub(crate) static_name: Ident,
    pub(crate) method_name: LitStr,
    pub(crate) order: syn::LitInt,
    pub(crate) degree: syn::LitInt,
    pub(crate) path: LitStr,
    pub(crate) crate_path: SynPath,
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
