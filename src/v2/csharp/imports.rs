//! P/Invoke (`[DllImport]`) extraction from the parsed C# tree.
//!
//! This reads `Add-Type` imports from the real C# parse rather than scanning
//! the source for a pattern. With the C# front-end compiled in, it is what
//! populates [`CSharpMemberDef::imports`](crate::v2::CSharpMemberDef::imports)
//! and `apis`; without it, a lightweight scanner does the same job so the
//! default build keeps its imports without pulling in the parser.
//!
//! A P/Invoke declaration is an `extern` method carrying a `[DllImport("…")]`
//! attribute. The library comes from the attribute's first string argument; the
//! return type, name, and parameters come straight from the method node's
//! children.

use super::ast::{CsNode, CsNodeKind, CsUnit};
use crate::v2::ast::{CSharpImport, CSharpParam};

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

    // The method name divides the signature: type references that end before it
    // form the return type; references after it (inside the parameter list)
    // type the parameters. The body Block, if any, is past the name too but is
    // not a reference, so it is ignored.
    let name_start = name.span.start;
    let name_end = name.span.end;

    let return_refs: Vec<&CsNode> = method
        .children
        .iter()
        .filter(|c| matches!(c.kind, CsNodeKind::NameRef { .. }) && c.span.end <= name_start)
        .collect();
    let returns = join_span_text(&return_refs, src);

    let mut params = Vec::new();
    let mut pending: Vec<&CsNode> = Vec::new();
    for c in &method.children {
        if c.span.start < name_end {
            continue; // attributes and the return type precede the name
        }
        match &c.kind {
            CsNodeKind::Param(pname) => {
                params.push(CSharpParam {
                    type_name: join_span_text(&pending, src),
                    name: pname.text.clone(),
                });
                pending.clear();
            }
            CsNodeKind::NameRef { .. } => pending.push(c),
            _ => {}
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

/// The trimmed source text spanning a run of reference nodes (a type, possibly
/// dotted or generic). Empty when there are none.
fn join_span_text(refs: &[&CsNode], src: &str) -> String {
    match (refs.first(), refs.last()) {
        (Some(first), Some(last)) => src[first.span.start..last.span.end].trim().to_string(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::csharp::parser::cs_parse;

    fn imports(code: &str) -> Vec<CSharpImport> {
        csharp_imports(&cs_parse(code, 0), code)
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
