//! C#-side refactoring over Add-Type blocks.
//!
//! Built on the resolver, these helpers turn a symbol into the edits that
//! rename it correctly: the declaration plus every reference the resolver bound
//! to it, with shadowing respected. C# is case-sensitive, so matching here is
//! exact. Everything returns [`Vec<TextEdit>`] (a comment returns one), so the
//! results compose through [`apply_edits`](crate::v2::apply_edits) and any
//! overlap with another refactor is reported rather than silently applied.

use super::ast::{CsNode, CsUnit};
use super::csharp_code_span;
use super::parser::cs_parse;
use super::resolve::{resolve, Decl, DeclKind};
use crate::v2::ast::Node;
use crate::v2::edit::TextEdit;
use crate::v2::span::Span;

/// Parses the C# carried by an `Add-Type` block into a [`CsUnit`].
///
/// `addtype` is the `Add-Type` command node or its `CSharpMemberDef`; `src` is
/// the whole PowerShell file. Returns `None` when the node has no extracted C#.
/// The unit's spans index `src` directly.
pub fn csharp_unit(addtype: &Node, src: &str) -> Option<CsUnit> {
    let span = csharp_code_span(addtype)?;
    Some(cs_parse(span.slice(src), span.start))
}

/// The C# symbols declared in `unit` (types, members, parameters, locals).
pub fn csharp_symbols(unit: &CsUnit) -> Vec<Decl> {
    resolve(unit).symbols().to_vec()
}

/// Every span that references the declaration whose name is at `name_span`
/// (including the declaration itself). Empty if no declaration sits there.
pub fn csharp_references(unit: &CsUnit, name_span: Span) -> Vec<Span> {
    let r = resolve(unit);
    match r.decl_at(name_span) {
        Some(id) => r.references_of(id),
        None => Vec::new(),
    }
}

/// Renames the C# declaration whose name is at `name_span`, plus every
/// reference bound to it, to `to`. Works for any symbol kind. Returns no edits
/// if no declaration is at `name_span`.
///
/// For an overloaded method, bare call sites bind to the first overload, so
/// renaming a single overload this way is incomplete; use
/// [`rename_csharp_method`], which renames the whole overload group.
pub fn rename_csharp_symbol(unit: &CsUnit, name_span: Span, to: &str) -> Vec<TextEdit> {
    let r = resolve(unit);
    let Some(id) = r.decl_at(name_span) else {
        return Vec::new();
    };
    edits_from_spans(r.references_of(id), to)
}

/// Renames a local variable by the span of its declared name. A no-op (empty)
/// if `name_span` is not a local declaration, so callers cannot accidentally
/// rename a field or method through this entry point.
pub fn rename_csharp_local(unit: &CsUnit, name_span: Span, to: &str) -> Vec<TextEdit> {
    rename_kind(unit, name_span, to, DeclKind::Local)
}

/// Renames a parameter by the span of its declared name. A no-op if
/// `name_span` is not a parameter declaration.
pub fn rename_csharp_parameter(unit: &CsUnit, name_span: Span, to: &str) -> Vec<TextEdit> {
    rename_kind(unit, name_span, to, DeclKind::Param)
}

/// Renames a field `from` declared in type `type_name`, plus its references
/// (bare uses bound to it, and `this.from`/`obj.from` member accesses). C#
/// case-sensitive match.
pub fn rename_csharp_field(unit: &CsUnit, type_name: &str, from: &str, to: &str) -> Vec<TextEdit> {
    rename_member_named(unit, type_name, from, to, DeclKind::Field)
}

/// Renames a method `from` declared in type `type_name`, plus its references
/// (bare and member-access call sites). Renames the whole method group when
/// `from` is overloaded.
pub fn rename_csharp_method(unit: &CsUnit, type_name: &str, from: &str, to: &str) -> Vec<TextEdit> {
    rename_member_named(unit, type_name, from, to, DeclKind::Method)
}

/// Inserts `text` as a `//` comment line immediately above the C# node
/// `before`, indented to match it. `text` is the comment body without `//`;
/// embedded newlines become multiple comment lines.
pub fn add_csharp_comment(src: &str, before: &CsNode, text: &str) -> TextEdit {
    let at = before.span.start.min(src.len());
    let line_start = src[..at].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let indent: String = src[line_start..at]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    let mut body = String::new();
    for line in text.split('\n') {
        body.push_str(&indent);
        body.push_str("// ");
        body.push_str(line);
        body.push('\n');
    }
    TextEdit::insert(line_start, body)
}

// internals

fn rename_kind(unit: &CsUnit, name_span: Span, to: &str, want: DeclKind) -> Vec<TextEdit> {
    let r = resolve(unit);
    let Some(id) = r.decl_at(name_span) else {
        return Vec::new();
    };
    if r.decl(id).map(|d| d.kind) != Some(want) {
        return Vec::new();
    }
    edits_from_spans(r.references_of(id), to)
}

fn rename_member_named(
    unit: &CsUnit,
    type_name: &str,
    from: &str,
    to: &str,
    kind: DeclKind,
) -> Vec<TextEdit> {
    let r = resolve(unit);
    let mut spans = Vec::new();
    for id in r.find(from, Some(kind)) {
        if r.enclosing_type(id) == Some(type_name) {
            spans.extend(r.references_of(id));
        }
    }
    edits_from_spans(spans, to)
}

/// Turns a set of spans into replacement edits: sorted by start, deduplicated,
/// each rewritten to `to`. Shared by the C#-side and cross-language renames.
pub(super) fn edits_from_spans(mut spans: Vec<Span>, to: &str) -> Vec<TextEdit> {
    spans.sort_by_key(|s| (s.start, s.end));
    spans.dedup();
    // `dedup` removes identical spans; also drop a span fully covered by one
    // already kept. Sorted by (start, end), any covering span starts no later,
    // so tracking the furthest end seen so far is enough: a span ending within
    // it is contained and skipped. This keeps a benign double collection (the
    // same site reached by two paths) from reaching the applier as an overlap
    // error, while genuinely disjoint spans are all kept.
    let mut kept: Vec<Span> = Vec::with_capacity(spans.len());
    let mut covered_to = 0usize;
    for s in spans {
        if !kept.is_empty() && s.end <= covered_to {
            continue; // contained in some earlier span
        }
        covered_to = covered_to.max(s.end);
        kept.push(s);
    }
    kept.into_iter().map(|s| TextEdit::replace(s, to)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::{apply_edits, parse, NodeKind};

    #[test]
    fn edits_from_spans_drops_identical_and_contained_spans() {
        // Identical spans collapse, and a span fully inside another is dropped,
        // so a benign double collection does not reach the applier as overlap.
        let spans = vec![
            Span::new(0, 10),
            Span::new(0, 10), // identical
            Span::new(2, 5),  // contained in [0,10]
            Span::new(10, 12),
            Span::new(20, 25),
            Span::new(22, 24), // contained in [20,25]
        ];
        let edits = edits_from_spans(spans, "X");
        let ranges: Vec<(usize, usize)> =
            edits.iter().map(|e| (e.span.start, e.span.end)).collect();
        assert_eq!(ranges, vec![(0, 10), (10, 12), (20, 25)]);
    }

    /// Parse PowerShell, find its first Add-Type node, return (src, node).
    fn add_type(src: &str) -> Node {
        let out = parse(src);
        let mut found = None;
        out.script.walk(&mut |n| {
            if matches!(n.kind, NodeKind::CSharpMemberDef(_)) && found.is_none() {
                found = Some(n.clone());
            }
        });
        found.expect("CSharpMemberDef")
    }

    const PS: &str = "Add-Type -TypeDefinition @'\npublic class Win32 {\n  public int count;\n  public int Bump() { count = count + 1; return this.count; }\n}\n'@\n";

    #[test]
    fn rename_field_updates_declaration_and_uses() {
        let node = add_type(PS);
        let unit = csharp_unit(&node, PS).unwrap();
        let edits = rename_csharp_field(&unit, "Win32", "count", "counter");
        // declaration + `count = count + 1` (2) + `this.count` (1) = 4 edits.
        assert_eq!(edits.len(), 4);
        let out = apply_edits(PS, &edits).unwrap();
        assert!(out.contains("public int counter;"));
        assert!(out.contains("counter = counter + 1"));
        assert!(out.contains("return this.counter;"));
    }

    #[test]
    fn rename_method_updates_call_sites() {
        let src = "Add-Type -TypeDefinition @'\npublic class C {\n  public void Run() { }\n  public void Go() { Run(); this.Run(); }\n}\n'@\n";
        let node = add_type(src);
        let unit = csharp_unit(&node, src).unwrap();
        let edits = rename_csharp_method(&unit, "C", "Run", "Execute");
        // declaration + bare call + this.Run
        assert_eq!(edits.len(), 3);
        let out = apply_edits(src, &edits).unwrap();
        assert!(out.contains("public void Execute()"));
        assert!(out.contains("Execute();"));
        assert!(out.contains("this.Execute();"));
    }

    #[test]
    fn rename_symbol_by_span_renames_a_local_only_in_its_method() {
        let src = "Add-Type -TypeDefinition @'\npublic class C {\n  void A() { int x = 1; x = x + 1; }\n  void B() { int x = 2; }\n}\n'@\n";
        let node = add_type(src);
        let unit = csharp_unit(&node, src).unwrap();
        // Find the local x declared in A (the first one).
        let syms = csharp_symbols(&unit);
        let first_x = syms
            .iter()
            .find(|d| d.name == "x" && d.kind == DeclKind::Local)
            .unwrap();
        let edits = rename_csharp_local(&unit, first_x.span, "n");
        // declaration + two uses in A only.
        assert_eq!(edits.len(), 3);
        let out = apply_edits(src, &edits).unwrap();
        assert!(out.contains("int n = 1; n = n + 1;"));
        assert!(out.contains("int x = 2;")); // B's x untouched
    }

    #[test]
    fn rename_local_refuses_a_field_span() {
        let node = add_type(PS);
        let unit = csharp_unit(&node, PS).unwrap();
        let syms = csharp_symbols(&unit);
        let field = syms
            .iter()
            .find(|d| d.name == "count" && d.kind == DeclKind::Field)
            .unwrap();
        // Using the local entry point on a field span yields nothing.
        assert!(rename_csharp_local(&unit, field.span, "x").is_empty());
    }

    #[test]
    fn add_comment_matches_indentation() {
        let src = "Add-Type -TypeDefinition @'\npublic class C {\n  public void M() { }\n}\n'@\n";
        let node = add_type(src);
        let unit = csharp_unit(&node, src).unwrap();
        // Find method M's node.
        let mut method = None;
        unit.root.walk(&mut |n| {
            if let crate::v2::csharp::ast::CsNodeKind::Method(name) = &n.kind {
                if name.text == "M" {
                    method = Some(n.clone());
                }
            }
        });
        let edit = add_csharp_comment(src, &method.unwrap(), "does nothing yet");
        let out = apply_edits(src, &[edit]).unwrap();
        assert!(out.contains("  // does nothing yet\n  public void M()"));
    }
}
