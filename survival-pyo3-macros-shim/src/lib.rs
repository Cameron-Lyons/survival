use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

#[proc_macro_attribute]
pub fn pyclass(_attr: TokenStream, item: TokenStream) -> TokenStream {
    strip_pyo3_helpers(item)
}

#[proc_macro_attribute]
pub fn pyfunction(_attr: TokenStream, item: TokenStream) -> TokenStream {
    strip_pyo3_helpers(item)
}

#[proc_macro_attribute]
pub fn pymethods(_attr: TokenStream, item: TokenStream) -> TokenStream {
    strip_pyo3_helpers(item)
}

#[proc_macro_attribute]
pub fn pymodule(_attr: TokenStream, item: TokenStream) -> TokenStream {
    strip_pyo3_helpers(item)
}

#[proc_macro_attribute]
pub fn pyo3(_attr: TokenStream, item: TokenStream) -> TokenStream {
    strip_pyo3_helpers(item)
}

#[proc_macro_attribute]
pub fn new(_attr: TokenStream, item: TokenStream) -> TokenStream {
    strip_pyo3_helpers(item)
}

#[proc_macro_attribute]
pub fn staticmethod(_attr: TokenStream, item: TokenStream) -> TokenStream {
    strip_pyo3_helpers(item)
}

#[proc_macro_attribute]
pub fn classmethod(_attr: TokenStream, item: TokenStream) -> TokenStream {
    strip_pyo3_helpers(item)
}

#[proc_macro_attribute]
pub fn getter(_attr: TokenStream, item: TokenStream) -> TokenStream {
    strip_pyo3_helpers(item)
}

#[proc_macro_attribute]
pub fn setter(_attr: TokenStream, item: TokenStream) -> TokenStream {
    strip_pyo3_helpers(item)
}

fn strip_pyo3_helpers(stream: TokenStream) -> TokenStream {
    let mut stripped = Vec::new();
    let mut iter = stream.into_iter().peekable();

    while let Some(token) = iter.next() {
        if is_pound(&token)
            && let Some(TokenTree::Group(group)) = iter.peek()
            && group.delimiter() == Delimiter::Bracket
            && is_pyo3_helper_attr(group)
        {
            iter.next();
            continue;
        }

        stripped.push(match token {
            TokenTree::Group(group) => {
                let mut new_group =
                    Group::new(group.delimiter(), strip_pyo3_helpers(group.stream()));
                new_group.set_span(group.span());
                TokenTree::Group(new_group)
            }
            other => other,
        });
    }

    stripped.into_iter().collect()
}

fn is_pound(token: &TokenTree) -> bool {
    matches!(token, TokenTree::Punct(punct) if punct.as_char() == '#')
}

fn is_pyo3_helper_attr(group: &Group) -> bool {
    let Some(TokenTree::Ident(ident)) = group.stream().into_iter().next() else {
        return false;
    };

    matches!(
        ident.to_string().as_str(),
        "pyo3" | "new" | "staticmethod" | "classmethod" | "getter" | "setter"
    )
}
