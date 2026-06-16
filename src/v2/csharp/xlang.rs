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
    each_csharp_unit(scope, src, &mut |unit, _owner| {
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
    each_csharp_unit(scope, src, &mut |unit, owner| {
        let r = resolve(unit);
        for id in r.find(from, None) {
            let Some(d) = r.decl(id) else { continue };
            let is_member = matches!(
                d.kind,
                DeclKind::Field | DeclKind::Method | DeclKind::Property | DeclKind::EnumMember
            );
            // The member belongs to `type_name` when its own enclosing class
            // matches, or, for a `-MemberDefinition` block with no enclosing
            // class, when the synthetic owner from `-Name` matches.
            let owned = r.enclosing_type(id) == Some(type_name)
                || (r.enclosing_type(id).is_none() && owner == Some(type_name));
            if is_member && owned {
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
            find_type_name_spans(
                safe_inner(src, inner_start, inner_end),
                inner_start,
                from,
                out,
            );
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
                    find_type_name_spans(
                        safe_inner(src, inner_start, inner_end),
                        inner_start,
                        from,
                        out,
                    );
                }
            }
        }
        NodeKind::Command { name, elements, .. } => {
            let cmd = match &name.kind {
                NodeKind::BareWord(w) => w.to_ascii_lowercase(),
                _ => String::new(),
            };
            match cmd.as_str() {
                // `New-Object Logger` / `-TypeName 'Logger'`: the argument is
                // the type name.
                "new-object" => {
                    if let Some(node) = new_object_type_arg(elements) {
                        push_type_name_from_arg(src, node, from, out);
                    }
                }
                // `Add-Type ... -Name Win32`: for a member-definition the
                // generated type's declaration site is the `-Name` argument.
                "add-type" => {
                    if let Some(node) = named_parameter_value(elements, "name") {
                        push_type_name_from_arg(src, node, from, out);
                    }
                }
                _ => {}
            }
        }
        // A PowerShell-native type declaration: `class Logger : Base { ... }`
        // or `enum E { ... }`. The node stores the name string but not its
        // span, so scan the header (everything before the body `{`) for the
        // declared name and for any base type after `:`. Both are renameable;
        // member-signature types are not, since the parser drops parameter and
        // property type annotations (tracked separately).
        NodeKind::ClassDefinition { .. } | NodeKind::EnumDefinition { .. } => {
            let header_end = type_header_end(src, n.span);
            let header = &src[n.span.start..header_end];
            find_type_name_spans_in_header(header, n.span.start, from, out);
        }
        _ => {}
    });
}

/// Byte offset of the body-opening `{` for a `class`/`enum` declaration, or
/// the node end if none is found. Everything before it is the header (keyword,
/// name, and optional `: base, interface` list).
fn type_header_end(src: &str, node: Span) -> usize {
    let bytes = src.as_bytes();
    let mut i = node.start;
    while i < node.end {
        if bytes[i] == b'{' {
            return i;
        }
        i += 1;
    }
    node.end
}

/// Finds whole-word matches of `from` in a `class`/`enum` header, skipping
/// `<# #>` and `#` comment regions so a name inside a header comment is not
/// matched. The keyword itself (`class`/`enum`) never equals a type name, so
/// every match is either the declared name or a base type, both of which a
/// type rename should rewrite.
fn find_type_name_spans_in_header(text: &str, base: usize, from: &str, out: &mut Vec<Span>) {
    // Build a copy with comment bytes blanked to spaces, preserving offsets and
    // UTF-8 validity (ASCII space is one byte, replacing whole comment runs).
    let mut masked = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'<' && bytes[i + 1] == b'#' {
            let mut j = i + 2;
            while j + 1 < bytes.len() && !(bytes[j] == b'#' && bytes[j + 1] == b'>') {
                j += 1;
            }
            let stop = (j + 2).min(bytes.len());
            for _ in i..stop {
                masked.push(' ');
            }
            i = stop;
        } else if bytes[i] == b'#' {
            let mut j = i;
            while j < bytes.len() && bytes[j] != b'\n' {
                j += 1;
            }
            for _ in i..j {
                masked.push(' ');
            }
            i = j;
        } else {
            // Copy one full UTF-8 character so we never split a multibyte byte.
            let ch_len = utf8_len(bytes[i]);
            let end = (i + ch_len).min(bytes.len());
            masked.push_str(&text[i..end]);
            i = end;
        }
    }
    find_type_name_spans(&masked, base, from, out);
}

/// Slices `src[start..end]`, snapping both ends inward to the nearest UTF-8
/// character boundary. Offsets computed as `span +/- 1` (skipping a bracket or
/// quote) can land inside a multibyte character when one sits adjacent to the
/// delimiter; this keeps the slice valid instead of panicking.
fn safe_inner(src: &str, mut start: usize, mut end: usize) -> &str {
    if start > end || end > src.len() {
        return "";
    }
    while start < end && !src.is_char_boundary(start) {
        start += 1;
    }
    while end > start && !src.is_char_boundary(end) {
        end -= 1;
    }
    &src[start..end]
}

/// Length in bytes of a UTF-8 character from its leading byte.
fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
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
        let t_inner = safe_inner(src, target.span.start + 1, target.span.end - 1);
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
/// The value bound to a named command parameter: either the inline
/// `-Name value` argument, or the element following `-Name` when the name
/// stands alone. Matching is case-insensitive and exact (not prefix-based, so
/// `-Namespace` does not satisfy a request for `name`).
fn named_parameter_value<'a>(elements: &'a [Node], param: &str) -> Option<&'a Node> {
    let mut i = 0;
    while i < elements.len() {
        if let NodeKind::CommandParameter { name, argument } = &elements[i].kind {
            if name.eq_ignore_ascii_case(param) {
                if let Some(arg) = argument {
                    return Some(arg);
                }
                if let Some(next) = elements.get(i + 1) {
                    if !matches!(next.kind, NodeKind::CommandParameter { .. }) {
                        return Some(next);
                    }
                }
            }
        }
        i += 1;
    }
    None
}

/// Pushes the span of a type-name argument, handling a bareword and a
/// single/double quoted literal (matching inside the quotes, skipping
/// interpolation), the same way the `New-Object` paths do.
fn push_type_name_from_arg(src: &str, node: &Node, from: &str, out: &mut Vec<Span>) {
    match &node.kind {
        NodeKind::BareWord(w) => {
            if w.eq_ignore_ascii_case(from) {
                out.push(node.span);
            }
        }
        NodeKind::StringLiteral { kind, parts, .. }
            if parts.is_empty()
                && matches!(kind, StringKind::Single | StringKind::Double)
                && node.span.end >= node.span.start + 2 =>
        {
            let inner_start = node.span.start + 1;
            let inner_end = node.span.end - 1;
            find_type_name_spans(
                safe_inner(src, inner_start, inner_end),
                inner_start,
                from,
                out,
            );
        }
        _ => {}
    }
}

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
    // A word character: ASCII alphanumeric, `_`, or any non-ASCII byte (the
    // continuation and lead bytes of a multibyte UTF-8 identifier). Every
    // boundary character that separates type names (`.`, `[`, `,`, space) is
    // ASCII below 0x80, so this keeps multibyte identifiers whole without
    // matching across a real boundary.
    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80;
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

/// Visits each embedded C# unit with the type name that owns its members, if
/// any. For a `-TypeDefinition` block the members are owned by the class
/// declared in the C# itself, so the owner is `None` (the resolver already
/// knows it). For a `-MemberDefinition` block the members have no enclosing
/// class in the C#; PowerShell wraps them in a type named by `-Name`, so that
/// name is the synthetic owner.
fn each_csharp_unit(
    scope: &Node,
    src: &str,
    f: &mut impl FnMut(&crate::v2::csharp::ast::CsUnit, Option<&str>),
) {
    scope.walk(&mut |n| {
        if let NodeKind::Command {
            elements, csharp, ..
        } = &n.kind
        {
            // The C# definition is held in the command's dedicated `csharp`
            // field. For a member definition, `-Name` names the generated type.
            let Some(node) = csharp else { return };
            let NodeKind::CSharpMemberDef(def) = &node.kind else {
                return;
            };
            let owner = if def.parameter.eq_ignore_ascii_case("memberdefinition") {
                named_parameter_value(elements, "name").and_then(|arg| arg_name_text(arg, src))
            } else {
                None
            };
            if let Some(unit) = csharp_unit(node, src) {
                f(&unit, owner.as_deref());
            }
        }
    });
}

/// The plain text of a type-name argument (bareword or simple quoted string),
/// for use as a synthetic owner name. Interpolated strings yield `None`.
fn arg_name_text(node: &Node, src: &str) -> Option<String> {
    match &node.kind {
        NodeKind::BareWord(w) => Some(w.clone()),
        NodeKind::StringLiteral { kind, parts, .. }
            if parts.is_empty()
                && matches!(kind, StringKind::Single | StringKind::Double)
                && node.span.end >= node.span.start + 2 =>
        {
            Some(safe_inner(src, node.span.start + 1, node.span.end - 1).to_string())
        }
        _ => None,
    }
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
    fn rename_type_reaches_constructors_and_new_expressions() {
        // X1: a constructor name is the type name, and `new T()` names the
        // type; a type rename must reach both, leaving no old name behind.
        let src = "Add-Type -TypeDefinition @\"\n\
            public class Logger {\n\
              public Logger() { }\n\
              public Logger(int n) { }\n\
              void M() { var a = new Logger(); var b = new Logger(5); }\n\
            }\n\
            \"@\n[Logger]::M()\n";
        let out = parse(src);
        let result = apply_edits(src, &rename_type(&out.script, src, "Logger", "Tracer")).unwrap();
        assert!(result.contains("public Tracer()"));
        assert!(result.contains("public Tracer(int n)"));
        assert!(result.contains("new Tracer()"));
        assert!(result.contains("new Tracer(5)"));
        assert!(result.contains("class Tracer"));
        assert!(!result.contains("Logger"), "no old name should remain");
    }

    #[test]
    fn rename_member_reaches_member_definition_csharp_side() {
        // X2: a `-MemberDefinition` block has no enclosing class in the C#;
        // the `-Name` value owns its members, so a member rename must edit the
        // C# declaration and the PowerShell call site together.
        let src = "Add-Type -MemberDefinition @\"\n\
            [DllImport(\"user32.dll\")]\n\
            public static extern int MessageBox(int h, string m, string c, int t);\n\
            \"@ -Name Win32 -Namespace Native\n\
            [Native.Win32]::MessageBox(0, 'm', 'c', 0)\n";
        let out = parse(src);
        let result = apply_edits(
            src,
            &rename_member(&out.script, src, "Win32", "MessageBox", "Show"),
        )
        .unwrap();
        assert!(result.contains("extern int Show("), "C# declaration");
        assert!(
            result.contains("[Native.Win32]::Show("),
            "PowerShell call site"
        );
        // A member rename under the wrong owner name is a no-op.
        let none = rename_member(&out.script, src, "Other", "MessageBox", "Show");
        assert!(none.is_empty());
    }

    #[test]
    fn rename_type_rewrites_add_type_name_argument() {
        // X3: `-Name` is the declaration site for a member-definition type.
        let src = "Add-Type -MemberDefinition @\"\npublic static int F() { return 0; }\n\"@ -Name Win32 -Namespace Native\n[Native.Win32]::F()\n";
        let out = parse(src);
        let result = apply_edits(src, &rename_type(&out.script, src, "Win32", "WinApi")).unwrap();
        assert!(result.contains("-Name WinApi"));
        assert!(result.contains("[Native.WinApi]::"));
        assert!(result.contains("-Namespace Native"), "namespace untouched");
        // A different command's -Name is left alone.
        let other = "Get-Process -Name Win32\nAdd-Type -MemberDefinition $s -Name Win32\n";
        let o2 = parse(other);
        let r2 = apply_edits(other, &rename_type(&o2.script, other, "Win32", "WinApi")).unwrap();
        assert!(r2.contains("Get-Process -Name Win32"));
    }

    #[test]
    fn rename_type_rewrites_powershell_class_declarations() {
        // X4: a PowerShell-native class/enum declaration and any base type in
        // its header are renameable; a name inside a header comment is not.
        let src = "class Base { }\nclass Logger : Base { [int]$X }\n[Logger]::M()\nenum Color { Red }\n[Color]::Red\n";
        let out = parse(src);
        let renamed = apply_edits(src, &rename_type(&out.script, src, "Logger", "Tracer")).unwrap();
        assert!(renamed.contains("class Tracer : Base"), "declaration");
        assert!(renamed.contains("[Tracer]::M()"), "reference");
        let base = apply_edits(src, &rename_type(&out.script, src, "Base", "Root")).unwrap();
        assert!(base.contains("class Root"), "base declaration");
        assert!(base.contains(": Root"), "base reference in header");
        let en = apply_edits(src, &rename_type(&out.script, src, "Color", "Hue")).unwrap();
        assert!(en.contains("enum Hue") && en.contains("[Hue]::Red"));
        // Comment trap: the commented name is skipped, the real one renamed.
        let cs = "class <# Logger #> Logger { }\n";
        let oc = parse(cs);
        let rc = apply_edits(cs, &rename_type(&oc.script, cs, "Logger", "Tracer")).unwrap();
        assert!(rc.contains("<# Logger #> Tracer"));
    }

    #[test]
    fn rename_paths_do_not_panic_on_multibyte_adjacent_to_delimiters() {
        // A multibyte character next to a bracket or quote makes a `span +/- 1`
        // interior slice land mid-character; the rename collectors must not
        // panic on it. These inputs are malformed on purpose.
        for src in [
            "[文]::M()\n",
            "[a文]$x = 1\n",
            "New-Object '文'\n",
            "return 0; -d \"@\nq -e \"@$x]=w\n",
            "Add-Type -MemberDefinition @\"\nint F();\n\"@ -Name 文\n",
        ] {
            let out = parse(src);
            // None of these should panic; edits may be empty.
            let _ = rename_type(&out.script, src, "文", "X");
            let _ = rename_type(&out.script, src, "Logger", "X");
            let _ = rename_member(&out.script, src, "文", "F", "G");
        }
    }

    #[test]
    fn rename_type_handles_non_ascii_identifiers() {
        // Non-ASCII type names are whole identifiers; matching is by exact
        // bytes (PowerShell case-insensitivity is ASCII-only) and a longer
        // name containing the target stays untouched.
        let src = "class \u{141}ogger { }\n[\u{141}ogger]::M()\nclass \u{141}oggerHelper { }\n";
        let out = parse(src);
        let renamed = apply_edits(
            src,
            &rename_type(&out.script, src, "\u{141}ogger", "Tracer"),
        )
        .unwrap();
        assert!(renamed.contains("class Tracer"));
        assert!(renamed.contains("[Tracer]::M()"));
        assert!(
            renamed.contains("\u{141}oggerHelper"),
            "prefix-only name untouched"
        );
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
