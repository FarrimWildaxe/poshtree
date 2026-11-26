//! P/Invoke (`[DllImport]`) extraction from the parsed C# tree.
//!
//! This reads `Add-Type` imports from the real C# parse rather than scanning
//! the source for a pattern. With the C# front-end compiled in, it is what
//! populates [`CSharpMemberDef::imports`](crate::v2::CSharpMemberDef::imports)
//! and `apis`. The `csharp` feature is enabled by default; when disabled, a
//! separate lightweight scanner in the PowerShell parser populates those
//! fields with more limited signature support.
//!
//! Extraction recognizes parsed methods carrying a `DllImport` or
//! `DllImportAttribute` attribute; it does not verify `extern` or semantic
//! validity. The library is taken from the first double-quoted substring in
//! the attribute source, without decoding escapes. Method and parameter names
//! come from the parsed tree. Their spans anchor a backward token scan for the
//! supported full type syntax immediately before each name:
//! qualified names, balanced generic arguments, array ranks, `?`, and `*`.
//! Types retain source spelling, including internal whitespace and comments,
//! but exclude attributes, declaration modifiers, and parameter defaults.
//!
//! This is bounded syntactic extraction, not type resolution or a full C#
//! grammar validator. Only methods and parameters recognized by the existing
//! parser are reported. An anchored type outside the supported token shapes
//! (for example, an array rank containing a size expression) is reported as an
//! empty string; a declaration without a method anchor yields no import.

use super::ast::{CsNode, CsNodeKind, CsUnit};
use super::lexer::cs_lex;
use super::tokens::{CsToken, CsTokenKind as K};
use crate::v2::ast::{CSharpImport, CSharpParam};
use crate::v2::span::Span;

/// The `[DllImport]` imports declared in `unit`. `src` is the text the unit's
/// spans index (the C# region, or the whole file when spans are file-absolute).
pub fn csharp_imports(unit: &CsUnit, src: &str) -> Vec<CSharpImport> {
    let mut out = Vec::new();
    unit.root.walk(&mut |n| {
        if let CsNodeKind::Method(name) = &n.kind {
            if let Some(imp) = method_import(n, name, src) {
                out.push(imp);
            }
        }
    });
    out
}

/// Imports plus the flat list of their function names (the `apis` field).
pub fn csharp_imports_and_apis(unit: &CsUnit, src: &str) -> (Vec<CSharpImport>, Vec<String>) {
    let imports = csharp_imports(unit, src);
    let apis = imports.iter().map(|i| i.function.clone()).collect();
    (imports, apis)
}

fn method_import(method: &CsNode, name: &super::ast::CsName, src: &str) -> Option<CSharpImport> {
    // Require a DllImport attribute and take its library argument.
    let mut dll = None;
    for c in &method.children {
        if let CsNodeKind::Attribute(a) = &c.kind {
            if a.text == "DllImport" || a.text == "DllImportAttribute" {
                dll = Some(first_string_literal(c.span.slice(src)));
            }
        }
    }
    let dll = dll?;

    // Re-lex only the imported method, keeping file-absolute AST name anchors.
    // Identifier reference spans omit punctuation and can also include defaults.
    let tokens = cs_lex(method.span.slice(src), method.span.start);
    let returns = signature_type_span(&tokens, name.span)
        .map(|span| span.slice(src))
        .unwrap_or_default()
        .to_string();

    let mut params = Vec::new();
    for c in &method.children {
        if c.span.start < name.span.end {
            continue; // attributes and the return type precede the name
        }
        if let CsNodeKind::Param(pname) = &c.kind {
            params.push(CSharpParam {
                type_name: signature_type_span(&tokens, pname.span)
                    .map(|span| span.slice(src))
                    .unwrap_or_default()
                    .to_string(),
                name: pname.text.clone(),
            });
        }
    }

    Some(CSharpImport {
        dll,
        function: name.text.clone(),
        returns,
        params,
    })
}

/// The contents of the first double-quoted string in `s` (the DllImport library
/// argument), without quotes.
fn first_string_literal(s: &str) -> String {
    if let Some(a) = s.find('"') {
        if let Some(rel) = s[a + 1..].find('"') {
            return s[a + 1..a + 1 + rel].to_string();
        }
    }
    String::new()
}

/// Finds the supported type token run immediately before an AST declaration
/// name. Adjacent identifiers end the run, excluding modifiers; generic commas
/// are only traversed inside balanced `<...>`, never between parameters.
/// Returns `None` if the anchor is absent or the backward run is unsupported.
fn signature_type_span(tokens: &[CsToken], name_span: Span) -> Option<Span> {
    let mut i = tokens
        .binary_search_by_key(&name_span.start, |t| t.span.start)
        .ok()?;
    if tokens[i].span != name_span || tokens[i].kind != K::Ident {
        return None;
    }
    let end = tokens.get(i.checked_sub(1)?)?.span.end;
    let mut generic_depth = 0usize;
    loop {
        i = i.checked_sub(1)?;
        let token = &tokens[i];
        match token.kind {
            K::RBracket => {
                // Only rank commas belong inside a type's brackets, not
                // attribute arguments or the parser's loose size expressions.
                while i > 0 && tokens[i - 1].kind == K::Comma {
                    i -= 1;
                }
                i = i.checked_sub(1)?;
                if tokens[i].kind != K::LBracket {
                    return None;
                }
            }
            K::Op if token.text == "?" || token.text == "*" => {}
            K::Gt => generic_depth += 1,
            K::Lt if generic_depth > 0 => generic_depth -= 1,
            K::Ident if generic_depth == 0 => {
                if i > 0 && matches!(tokens[i - 1].kind, K::Dot | K::ColonColon) {
                    i -= 1;
                } else {
                    return Some(Span::new(token.span.start, end));
                }
            }
            K::Ident | K::Dot | K::ColonColon | K::Comma if generic_depth > 0 => {}
            _ => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::csharp::parser::cs_parse;
    use crate::v2::{parse, NodeKind};

    fn imports(code: &str) -> Vec<CSharpImport> {
        csharp_imports(&cs_parse(code, 0), code)
    }

    fn signature_code(returns: &str, params: &str) -> String {
        format!(
            "public unsafe class W {{ [DllImport(\"native.dll\")] public static extern {returns} Go({params}); }}"
        )
    }

    fn expected_import(returns: &str, params: &[(&str, &str)]) -> Vec<CSharpImport> {
        vec![CSharpImport {
            dll: "native.dll".into(),
            function: "Go".into(),
            returns: returns.into(),
            params: params
                .iter()
                .map(|(type_name, name)| CSharpParam {
                    type_name: (*type_name).into(),
                    name: (*name).into(),
                })
                .collect(),
        }]
    }

    #[test]
    fn imports_with_array_types_preserve_return_and_parameter_suffixes() {
        let code = signature_code("byte[]", "byte[] value");

        let actual = imports(&code);

        assert_eq!(actual, expected_import("byte[]", &[("byte[]", "value")]));
    }

    #[test]
    fn imports_with_nullable_types_preserve_return_and_parameter_suffixes() {
        let code = signature_code("int?", "int? value");

        let actual = imports(&code);

        assert_eq!(actual, expected_import("int?", &[("int?", "value")]));
    }

    #[test]
    fn imports_with_pointer_types_preserve_return_and_parameter_suffixes() {
        let code = signature_code("byte*", "byte* value");

        let actual = imports(&code);

        assert_eq!(actual, expected_import("byte*", &[("byte*", "value")]));
    }

    #[test]
    fn imports_with_generic_types_preserve_closing_delimiters() {
        let code = signature_code("List<int>", "List<int> value");

        let actual = imports(&code);

        assert_eq!(
            actual,
            expected_import("List<int>", &[("List<int>", "value")])
        );
    }

    #[test]
    fn imports_with_multidimensional_arrays_preserve_ranks() {
        let code = signature_code("int[,]", "int[,] value");

        let actual = imports(&code);

        assert_eq!(actual, expected_import("int[,]", &[("int[,]", "value")]));
    }

    #[test]
    fn imports_with_nested_qualified_generic_arrays_preserve_full_types() {
        let code = signature_code(
            "global::Native.Outer<int[]>.Inner<System.Collections.Generic.List<byte?[,]>>[]",
            "global::Native.Outer<int[]>.Inner<System.Collections.Generic.List<byte?[,]>>[] value",
        );

        let actual = imports(&code);

        assert_eq!(
            actual,
            expected_import(
                "global::Native.Outer<int[]>.Inner<System.Collections.Generic.List<byte?[,]>>[]",
                &[(
                    "global::Native.Outer<int[]>.Inner<System.Collections.Generic.List<byte?[,]>>[]",
                    "value"
                )]
            )
        );
    }

    #[test]
    fn imports_after_identifier_defaults_exclude_defaults_and_parameter_attributes() {
        let code = signature_code(
            "void",
            "int first = DefaultValue, [Optional] in int second = OtherDefault",
        );

        let actual = imports(&code);

        assert_eq!(
            actual,
            expected_import("void", &[("int", "first"), ("int", "second")])
        );
    }

    #[test]
    fn imports_with_parameter_modifiers_report_only_types() {
        let code = signature_code(
            "void",
            "ref byte[] data, out int? result, in Native.Handle handle, params int[] rest",
        );

        let actual = imports(&code);

        assert_eq!(
            actual,
            expected_import(
                "void",
                &[
                    ("byte[]", "data"),
                    ("int?", "result"),
                    ("Native.Handle", "handle"),
                    ("int[]", "rest")
                ]
            )
        );
    }

    #[test]
    fn imports_with_type_trivia_preserve_internal_spelling_only() {
        let code = signature_code(
            "/* before */ Native /* qualified */ . Box < int /* inner */ [] > [ , ] /* after */",
            "[In] ref /* before */ byte /* element */ [ ] /* after */ data",
        );

        let actual = imports(&code);

        assert_eq!(
            actual,
            expected_import(
                "Native /* qualified */ . Box < int /* inner */ [] > [ , ]",
                &[("byte /* element */ [ ]", "data")]
            )
        );
    }

    #[test]
    fn imports_with_unicode_and_absolute_spans_preserve_full_types() {
        let prefix = "# 前置き 🦀\n";
        let code = signature_code("名前.箱<型[]>[]", "名前.箱<型[]>[] 値");
        let source = format!("{prefix}{code}\n# 後置き");
        let unit = cs_parse(&code, prefix.len());

        let actual = csharp_imports(&unit, &source);

        assert_eq!(
            actual,
            expected_import("名前.箱<型[]>[]", &[("名前.箱<型[]>[]", "値")])
        );
    }

    #[test]
    fn imports_with_empty_source_return_no_imports() {
        let code = "";

        let actual = imports(code);

        assert_eq!(actual, Vec::<CSharpImport>::new());
    }

    #[test]
    fn imports_with_empty_parameter_lists_keep_the_return_type() {
        let code = signature_code("byte[]", "");

        let actual = imports(&code);

        assert_eq!(actual, expected_import("byte[]", &[]));
    }

    #[test]
    fn imports_with_non_rank_array_contents_report_empty_types() {
        // The shallow parser still provides method and parameter name anchors.
        let code = signature_code("byte[1]", "int[Size] value");

        let actual = imports(&code);

        assert_eq!(actual, expected_import("", &[("", "value")]));
    }

    #[test]
    fn imports_with_unclosed_generics_and_no_method_anchor_return_no_imports() {
        let code = signature_code("List<int", "");

        let actual = imports(&code);

        assert_eq!(actual, Vec::<CSharpImport>::new());
    }

    #[test]
    fn public_parse_with_embedded_csharp_stores_full_import_types() {
        let source = r#"# 前置き 🦀
Add-Type -TypeDefinition @'
public unsafe class Native {
    [DllImport("native.dll")]
    public static extern global::Native.Box<System.Collections.Generic.List<byte?[,]>>[] Go(
        [In] ref byte /* element */ [] data,
        out int? result,
        byte* pointer,
        [Optional] in Native.Handle handle = DefaultHandle,
        params int[][] rest);
}
'@
"#;

        let parsed = parse(source);

        let mut actual = Vec::new();
        parsed.script.walk(&mut |node| {
            if let NodeKind::CSharpMemberDef(def) = &node.kind {
                actual.push(def.imports.clone());
            }
        });
        assert_eq!(
            actual,
            vec![vec![CSharpImport {
                dll: "native.dll".into(),
                function: "Go".into(),
                returns: "global::Native.Box<System.Collections.Generic.List<byte?[,]>>[]".into(),
                params: vec![
                    CSharpParam {
                        type_name: "byte /* element */ []".into(),
                        name: "data".into(),
                    },
                    CSharpParam {
                        type_name: "int?".into(),
                        name: "result".into(),
                    },
                    CSharpParam {
                        type_name: "byte*".into(),
                        name: "pointer".into(),
                    },
                    CSharpParam {
                        type_name: "Native.Handle".into(),
                        name: "handle".into(),
                    },
                    CSharpParam {
                        type_name: "int[][]".into(),
                        name: "rest".into(),
                    },
                ],
            }]]
        );
    }

    #[test]
    fn extracts_a_pinvoke_signature() {
        let code = "public class W { [DllImport(\"user32.dll\")] public static extern int MessageBox(IntPtr hWnd, string text, string caption, uint type); }";
        let imps = imports(code);
        assert_eq!(imps.len(), 1);
        let i = &imps[0];
        assert_eq!(i.dll, "user32.dll");
        assert_eq!(i.function, "MessageBox");
        assert_eq!(i.returns, "int");
        assert_eq!(i.params.len(), 4);
        assert_eq!(i.params[0].type_name, "IntPtr");
        assert_eq!(i.params[0].name, "hWnd");
        assert_eq!(i.params[3].type_name, "uint");
        assert_eq!(i.params[3].name, "type");
    }

    #[test]
    fn ignores_non_pinvoke_methods() {
        let code = "public class W { public int Add(int a, int b) { return a + b; } [DllImport(\"k.dll\")] static extern void Go(); }";
        let imps = imports(code);
        assert_eq!(imps.len(), 1);
        assert_eq!(imps[0].function, "Go");
        assert_eq!(imps[0].returns, "void");
        assert!(imps[0].params.is_empty());
    }

    #[test]
    fn multiple_imports_collected() {
        let code = "public class N { [DllImport(\"a.dll\")] public static extern void A(); [DllImport(\"b.dll\")] public static extern int B(long x); }";
        let imps = imports(code);
        let names: Vec<&str> = imps.iter().map(|i| i.function.as_str()).collect();
        assert_eq!(names, ["A", "B"]);
        assert_eq!(imps[1].params[0].type_name, "long");
    }
}
