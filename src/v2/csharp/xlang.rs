//! Cross-language renaming: a C# type or member defined in an `Add-Type` block
//! is used directly from PowerShell as ordinary syntax, and these helpers keep
//! both sides in sync in one edit set.
//!
//! The C# side comes from the resolver (declaration plus bound references). The
//! PowerShell side is read straight from the v2 tree, where the constructs that
//! name a C# symbol have precise spans:
//!
//! * a type literal `[Win32]` is a [`TypeExpression`](NodeKind::TypeExpression),
//! * a cast `[Win32]$x` is a [`Cast`](NodeKind::Cast),
//! * `New-Object Win32` is a [`Command`](NodeKind::Command) with a bareword
//!   argument,
//! * a static use `[Win32]::Member` is a
//!   [`MemberAccess`](NodeKind::MemberAccess) /
//!   [`InvokeMember`](NodeKind::InvokeMember) whose target is the type literal.
//!
//! C# is case-sensitive but PowerShell resolves type and member names without
//! regard to case, so the PowerShell side matches case-insensitively and is
//! rewritten to the new canonical name. Matching is by name within the file, so
//! the instance form `$obj.Member` (whose receiver type is unknown) is left
//! alone, and the rare clash of two declared symbols sharing a name is
//! documented rather than guessed.

use super::refactor::csharp_unit;
use super::resolve::{resolve, DeclKind};
use crate::v2::ast::{Node, NodeKind, StringKind};
use crate::v2::edit::TextEdit;
use crate::v2::span::Span;

/// Renames a C# type `from` to `to` everywhere it is defined or used: the C#
/// declaration, references within the C#, and PowerShell call sites
/// (`[from]`, `[from]$x`, `[from]::...`, `New-Object from`).
///
/// `scope` is the PowerShell subtree to operate within (the whole script, or a
/// narrower node); `src` is the file. Returns edits to apply with
/// [`apply_edits`](crate::v2::apply_edits).
pub fn rename_type(scope: &Node, src: &str, from: &str, to: &str) -> Vec<TextEdit> {
    let mut spans = Vec::new();
    // C# side: the declaration and its in-C# references, per Add-Type block.
    each_csharp_unit(scope, src, &mut |unit| {
        let r = resolve(unit);
        for id in r.find(from, Some(DeclKind::Type)) {
            spans.extend(r.references_of(id));
        }
    });
    // PowerShell side.
    ps_type_refs(scope, src, from, &mut spans);
    super::refactor::edits_from_spans(spans, to)
}

/// Renames a C# member `from` (field, method, property, or enum member)
/// declared in type `type_name`, to `to`: the C# declaration, references within
/// the C#, and PowerShell static call sites `[type_name]::from`.
pub fn rename_member(
    scope: &Node,
    src: &str,
    type_name: &str,
    from: &str,
    to: &str,
) -> Vec<TextEdit> {
    let mut spans = Vec::new();
    each_csharp_unit(scope, src, &mut |unit| {
        let r = resolve(unit);
        for id in r.find(from, None) {
            let Some(d) = r.decl(id) else { continue };
            let is_member = matches!(
                d.kind,
                DeclKind::Field | DeclKind::Method | DeclKind::Property | DeclKind::EnumMember
            );
            if is_member && r.enclosing_type(id) == Some(type_name) {
                spans.extend(r.references_of(id));
            }
        }
    });
    ps_static_member_refs(scope, src, type_name, from, &mut spans);
    super::refactor::edits_from_spans(spans, to)
}

// PowerShell-side extraction

fn ps_type_refs(scope: &Node, src: &str, from: &str, out: &mut Vec<Span>) {
    scope.walk(&mut |n| match &n.kind {
        // Inside the brackets: `[ ... ]`.
        NodeKind::TypeExpression(_) if n.span.end > n.span.start + 1 => {
            let inner_start = n.span.start + 1;
            let inner_end = n.span.end - 1;
            find_type_name_spans(&src[inner_start..inner_end], inner_start, from, out);
        }
        NodeKind::Cast { .. } => {
            // The type literal at the start: `[ ... ]$operand`. Scan to the
            // matching close bracket so a generic argument's inner `]` does not
            // cut the type short (`[List[Logger]]`).
            let rest = &src[n.span.start..n.span.end];
            if let Some(close) = matching_bracket(rest) {
                let inner_start = n.span.start + 1;
                let inner_end = n.span.start + close;
                if inner_end > inner_start {
                    find_type_name_spans(&src[inner_start..inner_end], inner_start, from, out);
                }
            }
        }
        NodeKind::Command { name, elements, .. } => {
            if matches!(&name.kind, NodeKind::BareWord(w) if w.eq_ignore_ascii_case("new-object")) {
                if let Some(node) = new_object_type_arg(elements) {
                    match &node.kind {
                        NodeKind::BareWord(w) => {
                            if w.eq_ignore_ascii_case(from) {
                                out.push(node.span);
                            }
                        }
                        // A quoted type name: `New-Object 'Logger'` or
                        // `New-Object -TypeName "My.Logger"`. Match inside the
                        // quotes so they survive the rename. Interpolated
                        // strings are skipped, since their value is not a
                        // static name.
                        NodeKind::StringLiteral { kind, parts, .. }
                            if parts.is_empty()
                                && matches!(kind, StringKind::Single | StringKind::Double)
                                && node.span.end >= node.span.start + 2 =>
                        {
                            let inner_start = node.span.start + 1;
                            let inner_end = node.span.end - 1;
                            find_type_name_spans(
                                &src[inner_start..inner_end],
                                inner_start,
                                from,
                                out,
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        _ => {}
    });
}

/// Byte offset of the `]` that matches the leading `[` in `s`, accounting for
/// nested brackets (generic arguments). `s` must start with `[`.
fn matching_bracket(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, b) in s.bytes().enumerate() {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn ps_static_member_refs(
    scope: &Node,
    src: &str,
    type_name: &str,
    member: &str,
    out: &mut Vec<Span>,
) {
    scope.walk(&mut |n| {
        let (target, m, is_static) = match &n.kind {
            NodeKind::MemberAccess {
                target,
                member,
                is_static,
            } => (target, member, *is_static),
            NodeKind::InvokeMember {
                target,
                member,
                is_static,
                ..
            } => (target, member, *is_static),
            _ => return,
        };
        if !is_static || !m.eq_ignore_ascii_case(member) {
            return;
        }
        let NodeKind::TypeExpression(_) = &target.kind else {
            return;
        };
        // The target type must match.
        if target.span.end <= target.span.start + 1 {
            return;
        }
        let t_inner = &src[target.span.start + 1..target.span.end - 1];
        if !type_text_matches(t_inner, type_name) {
            return;
        }
        if let Some(s) = member_span_after(src, target.span.end, member) {
            out.push(s);
        }
    });
}

/// The type argument of a `New-Object` call: the value of `-TypeName`, or the
/// first positional argument.
fn new_object_type_arg(elements: &[Node]) -> Option<&Node> {
    let mut i = 0;
    while i < elements.len() {
        match &elements[i].kind {
            NodeKind::CommandParameter { name, argument }
                if "typename".starts_with(&name.to_ascii_lowercase()) =>
            {
                if let Some(arg) = argument {
                    return Some(arg);
                }
                if let Some(next) = elements.get(i + 1) {
                    if !matches!(next.kind, NodeKind::CommandParameter { .. }) {
                        return Some(next);
                    }
                }
            }
            NodeKind::BareWord(_) | NodeKind::StringLiteral { .. } => return Some(&elements[i]),
            _ => {}
        }
        i += 1;
    }
    None
}

/// Whether a bracketed type text names `target` (whole, or by last segment),
/// case-insensitively.
fn type_text_matches(inner: &str, target: &str) -> bool {
    let t = inner.trim();
    t.eq_ignore_ascii_case(target)
        || t.rsplit('.')
            .next()
            .is_some_and(|seg| seg.eq_ignore_ascii_case(target))
}

/// Span of the part of a (possibly dotted) type text that names `from`: the
/// whole trimmed text if it matches, else its last segment. `start` is the byte
/// offset of `text` in the file.
/// Pushes a span for every whole-identifier occurrence of `from` inside a
/// type-expression interior (case-insensitive). Identifier boundaries are
/// respected, so `Logger` does not match inside `LoggerHelper`, and `.` is a
/// namespace separator, so only the simple name (after the last `.`) of each
/// dotted run is compared. This covers plain (`Logger`), array (`Logger[]`),
/// dotted (`My.Logger`), and generic (`List[Logger]`, `Dictionary[Logger,
/// Logger]`) type references; the latter contribute one span per occurrence.
fn find_type_name_spans(text: &str, base: usize, from: &str, out: &mut Vec<Span>) {
    let bytes = text.as_bytes();
    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut i = 0;
    while i < bytes.len() {
        if !is_word(bytes[i]) {
            i += 1;
            continue;
        }
        // A maximal dotted run: ident ('.' ident)*.
        let run_start = i;
        while i < bytes.len() && (is_word(bytes[i]) || bytes[i] == b'.') {
            i += 1;
        }
        let run = &text[run_start..i];
        // The simple name is the part after the last '.'.
        let dot = run.rfind('.').map_or(0, |d| d + 1);
        let name = &run[dot..];
        if name.eq_ignore_ascii_case(from) {
            let s = run_start + dot;
            out.push(Span::new(base + s, base + s + name.len()));
        }
    }
}

/// Scans from `offset` past separator/whitespace, then over an identifier; if
/// that identifier matches `member` case-insensitively, returns its span.
fn member_span_after(src: &str, offset: usize, member: &str) -> Option<Span> {
    let b = src.as_bytes();
    let n = b.len();
    let mut i = offset.min(n);
    while i < n && (b[i] == b':' || b[i] == b'.' || b[i].is_ascii_whitespace()) {
        i += 1;
    }
    let start = i;
    while i < n && (b[i] == b'_' || b[i].is_ascii_alphanumeric()) {
        i += 1;
    }
    if i > start && src[start..i].eq_ignore_ascii_case(member) {
        Some(Span::new(start, i))
    } else {
        None
    }
}

fn each_csharp_unit(scope: &Node, src: &str, f: &mut impl FnMut(&crate::v2::csharp::ast::CsUnit)) {
    scope.walk(&mut |n| {
        if matches!(n.kind, NodeKind::CSharpMemberDef(_)) {
            if let Some(unit) = csharp_unit(n, src) {
                f(&unit);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::{apply_edits, parse};

    const PS: &str = "Add-Type -TypeDefinition @'\npublic class Win32 {\n  [DllImport(\"user32.dll\")]\n  public static extern int MessageBox(IntPtr h, string t, string c, uint ty);\n}\n'@\n[Win32]::MessageBox(0, 'hi', 'title', 0)\n$inst = New-Object Win32\n[Win32]$casted = $inst\n";

    #[test]
    fn rename_type_updates_csharp_and_all_powershell_sites() {
        let out = parse(PS);
        let edits = rename_type(&out.script, PS, "Win32", "NativeApi");
        let result = apply_edits(PS, &edits).unwrap();
        // C# declaration.
        assert!(result.contains("public class NativeApi {"));
        // PowerShell static call, New-Object, and cast.
        assert!(result.contains("[NativeApi]::MessageBox"));
        assert!(result.contains("New-Object NativeApi"));
        assert!(result.contains("[NativeApi]$casted"));
        // No old type name remains.
        assert!(!result.contains("Win32"));
    }

    #[test]
    fn rename_type_handles_array_generic_and_multiple_occurrences() {
        // Array, generic-argument, and repeated type references on the
        // PowerShell side all rename, while a longer name that merely contains
        // the target as a prefix is left alone.
        let src = "[Logger[]]::new()\n\
                   [System.Collections.Generic.List[Logger]]$x = $null\n\
                   [System.Collections.Generic.Dictionary[Logger,Logger]]::new()\n\
                   [LoggerHelper]::Init()\n\
                   [My.Logger]::X()\n";
        let out = parse(src);
        let edits = rename_type(&out.script, src, "Logger", "Tracer");
        let result = apply_edits(src, &edits).unwrap();
        assert!(result.contains("[Tracer[]]::new()"));
        assert!(result.contains("List[Tracer]"));
        assert!(result.contains("Dictionary[Tracer,Tracer]"));
        assert!(result.contains("[My.Tracer]::X()"));
        // The prefix-only name keeps its full spelling.
        assert!(result.contains("[LoggerHelper]::Init()"));
        // No standalone old type name survives (LoggerHelper still present).
        assert!(!result.contains("[Logger]"));
        assert!(!result.contains("[Logger["));
    }

    #[test]
    fn rename_type_handles_quoted_new_object_names() {
        // Quoted spellings are common (`New-Object -TypeName 'X'`). The match
        // happens inside the quotes, dotted names rename their simple segment,
        // and interpolated or prefix-only strings are left alone.
        let src = "New-Object -TypeName 'Logger'\n\
                   New-Object \"Logger\"\n\
                   New-Object 'My.Logger'\n\
                   New-Object \"$prefix.Logger\"\n\
                   New-Object 'LoggerHelper'\n";
        let out = parse(src);
        let edits = rename_type(&out.script, src, "Logger", "Tracer");
        let result = apply_edits(src, &edits).unwrap();
        assert!(result.contains("-TypeName 'Tracer'"));
        assert!(result.contains("New-Object \"Tracer\""));
        assert!(result.contains("'My.Tracer'"));
        assert!(
            result.contains("\"$prefix.Logger\""),
            "interpolated untouched"
        );
        assert!(result.contains("'LoggerHelper'"), "prefix-only untouched");
    }

    #[test]
    fn rename_member_updates_csharp_decl_and_static_call_site() {
        let out = parse(PS);
        let edits = rename_member(&out.script, PS, "Win32", "MessageBox", "ShowMessage");
        let result = apply_edits(PS, &edits).unwrap();
        // C# extern declaration.
        assert!(result.contains("extern int ShowMessage("));
        // PowerShell static call site.
        assert!(result.contains("[Win32]::ShowMessage(0, 'hi', 'title', 0)"));
        // The type name is untouched.
        assert!(result.contains("public class Win32"));
        assert!(!result.contains("MessageBox"));
    }

    #[test]
    fn rename_member_leaves_unrelated_receiver_alone() {
        // The class has a field `Length`; the body also calls the BCL property
        // `s.Length`, whose receiver is not our type. Renaming the field must
        // touch `this.Length` but not `s.Length`.
        let src = "Add-Type -TypeDefinition @'\npublic class C {\n  public int Length;\n  public int Measure(string s) { return s.Length + this.Length; }\n}\n'@\n";
        let out = parse(src);
        let edits = rename_member(&out.script, src, "C", "Length", "Size");
        let result = apply_edits(src, &edits).unwrap();
        assert!(result.contains("public int Size;"));
        // this.Length renamed, s.Length (BCL) left intact.
        assert!(result.contains("return s.Length + this.Size;"));
    }
}
