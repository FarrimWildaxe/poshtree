//! Cross-language renaming: a C# type or member defined in an `Add-Type` block
//! can be used directly from PowerShell as ordinary syntax. These helpers
//! collect one edit set for resolver-selected C# references and recognized
//! PowerShell name sites, not all possible runtime uses.
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
//! C# declaration/owner comparisons are case-sensitive. PowerShell-side name,
//! command, and parameter comparisons here ignore ASCII case only, not general
//! Unicode case. Type matching uses simple names in bracketed/quoted type text;
//! bareword type arguments require a whole-text match. Instance `$obj.Member`
//! uses are left alone. Static members use stored parser spans and accept only
//! nonempty names whose characters are all Unicode alphanumeric or `_`.
//!
//! Each embedded C# unit is resolved independently. Simple-name ownership and
//! overload limits still apply; this is not full semantic binding and does not
//! guarantee refusal of ambiguous same-named symbols. PowerShell matches are
//! syntactic and do not require a corresponding C# declaration to be found.
//! Apply edits to the original source to preserve untouched bytes;
//! [`apply_edits`](crate::v2::apply_edits) validates spans and overlap, not
//! semantic rename validity, replacement names, or collisions.

use super::refactor::csharp_unit;
use super::resolve::{resolve, DeclKind};
use crate::v2::ast::{Node, NodeKind, StringKind};
use crate::v2::edit::TextEdit;
use crate::v2::span::Span;

/// Renames a C# type `from` to `to`: the C# declaration, resolver-bound
/// references within the C#, and PowerShell call sites
/// (`[from]`, `[from]$x`, `[from]::...`, `New-Object from`).
///
/// C# references include the stored final identifiers in base-class and
/// interface lists when they bind to a type declared in the same embedded C#
/// unit. Binding uses simple names, not namespace-qualified type identity;
/// generic arguments in those lists are not tracked. This is not full C#
/// semantic resolution. C# comments and string literal contents are left alone.
///
/// `scope` is the PowerShell subtree to operate within (the whole script, or a
/// narrower node); `src` is the file. Returns edits to apply with
/// [`apply_edits`](crate::v2::apply_edits).
/// PowerShell matching also includes `Add-Type -Name` arguments, class/enum
/// declaration names, and class base headers. It uses ASCII-insensitive
/// syntactic matching, not a shared PowerShell/C# type identity.
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
///
/// C# declarations are filtered by case-sensitive simple owner name (or the
/// synthetic `-Name` owner of a member-definition block) and use the resolver's
/// supported receivers only. All matching overloads are included. PowerShell
/// static sites use ASCII-insensitive type/member comparisons, without binding
/// receiver types or requiring a matching C# declaration; instance sites are
/// omitted. Only stored literal member spans with nonempty Unicode
/// alphanumeric/underscore names are edited, including keyword and digit-initial
/// generic names. Parser-recognized comments/continuations around `::` are left
/// untouched. Replacement validity and name collisions are not checked.
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
        NodeKind::TypeExpression(_) => {
            find_in_interior(src, n.span, from, out);
        }
        NodeKind::Cast { .. } => {
            // The type literal at the start: `[ ... ]$operand`. Scan to the
            // matching close bracket so a generic argument's inner `]` does not
            // cut the type short (`[List[Logger]]`).
            let rest = &src[n.span.start..n.span.end];
            if let Some(close) = matching_bracket(rest) {
                find_in_byte_range(src, n.span.start + 1, n.span.start + close, from, out);
            }
        }
        NodeKind::Command { name, elements, .. } => {
            let cmd = match &name.kind {
                NodeKind::BareWord(w) => w.as_str(),
                _ => "",
            };
            // Compared ignoring ASCII case without allocating, since every
            // command in the script reaches here but only these two matter.
            if cmd.eq_ignore_ascii_case("new-object") {
                // `New-Object Logger` / `-TypeName 'Logger'`: the argument is
                // the type name.
                if let Some(node) = new_object_type_arg(elements) {
                    push_type_name_from_arg(src, node, from, out);
                }
            } else if cmd.eq_ignore_ascii_case("add-type") {
                // `Add-Type ... -Name Win32`: for a member-definition the
                // generated type's declaration site is the `-Name` argument.
                if let Some(node) = named_parameter_value(elements, "name") {
                    push_type_name_from_arg(src, node, from, out);
                }
            }
        }
        // A PowerShell-native type declaration. The name is renamed from its
        // stored span; base types (after the name, before `{`) are matched by
        // a comment-aware scan, since `bases` are kept as plain strings.
        NodeKind::ClassDefinition {
            name, name_span, ..
        } => {
            if name.eq_ignore_ascii_case(from) {
                out.push(*name_span);
            }
            let base_start = name_span.end;
            let base_end = type_header_end(src, n.span).max(base_start);
            find_type_name_spans_in_header(
                safe_inner(src, base_start, base_end),
                base_start,
                from,
                out,
            );
        }
        NodeKind::EnumDefinition {
            name, name_span, ..
        } if name.eq_ignore_ascii_case(from) => {
            // An enum's only header type name is its own; the backing type
            // after `:` is a primitive, not a renamable user type.
            out.push(*name_span);
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
    // Common case: a header with no comment needs no masking, so scan directly
    // and skip building a copy.
    if !text.as_bytes().contains(&b'#') {
        find_type_name_spans(text, base, from, out);
        return;
    }
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

/// Runs `find_type_name_spans` over `src[start..end]` after snapping both ends
/// to char boundaries, with results offset back to the original source. The
/// single place that turns a byte range into type-name matches.
fn find_in_byte_range(src: &str, start: usize, end: usize, from: &str, out: &mut Vec<Span>) {
    if end <= start {
        return;
    }
    find_type_name_spans(safe_inner(src, start, end), start, from, out);
}

/// Matches `from` inside a node's delimited interior: one delimiter character
/// on each side (`[...]`, `'...'`, `"..."`). A node too short to have an
/// interior contributes nothing.
fn find_in_interior(src: &str, span: Span, from: &str, out: &mut Vec<Span>) {
    if span.end <= span.start + 1 {
        return;
    }
    find_in_byte_range(src, span.start + 1, span.end - 1, from, out);
}

/// Whether a node is a plain single/double-quoted string literal (no
/// interpolation) long enough to have an interior. The shape `New-Object` and
/// `Add-Type -Name` accept for a quoted type name.
fn is_plain_quoted(node: &Node) -> bool {
    matches!(
        &node.kind,
        NodeKind::StringLiteral { kind, parts, .. }
            if parts.is_empty()
                && matches!(kind, StringKind::Single | StringKind::Double)
                && node.span.end >= node.span.start + 2
    )
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
        let (target, m, member_span, is_static) = match &n.kind {
            NodeKind::MemberAccess {
                target,
                member,
                member_span,
                is_static,
            } => (target, member, *member_span, *is_static),
            NodeKind::InvokeMember {
                target,
                member,
                member_span,
                is_static,
                ..
            } => (target, member, *member_span, *is_static),
            _ => return,
        };
        let Some(member_span) = member_span else {
            return;
        };
        if !is_static || !is_plain_member_name(m) || !m.eq_ignore_ascii_case(member) {
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
        out.push(member_span);
    });
}

// Generic tokens admit punctuation and escapes too; a stored token span alone
// is not permission to rename those broader spellings.
fn is_plain_member_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// The type argument of a `New-Object` call: the value of `-TypeName`, or the
/// first positional argument.
/// The value bound to a named command parameter: either the inline
/// `-Name value` argument, or the element following `-Name` when the name
/// stands alone. Matching ignores ASCII case and is exact (not prefix-based, so
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
        // A quoted type name: match inside the quotes so they survive.
        _ if is_plain_quoted(node) => find_in_interior(src, node.span, from, out),
        _ => {}
    }
}

fn new_object_type_arg(elements: &[Node]) -> Option<&Node> {
    let mut i = 0;
    while i < elements.len() {
        match &elements[i].kind {
            NodeKind::CommandParameter { name, argument }
                if is_prefix_ignore_ascii_case(name, "typename") =>
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

/// Whether `prefix` is a non-empty ASCII-insensitive prefix of `full`, without
/// allocating. Lets an abbreviated parameter (`-Type`) match `-TypeName`.
fn is_prefix_ignore_ascii_case(prefix: &str, full: &str) -> bool {
    !prefix.is_empty()
        && prefix.len() <= full.len()
        && full.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

/// Whether a bracketed type text names `target` (whole, or by last segment),
/// ignoring ASCII case.
fn type_text_matches(inner: &str, target: &str) -> bool {
    let t = inner.trim();
    t.eq_ignore_ascii_case(target)
        || t.rsplit('.')
            .next()
            .is_some_and(|seg| seg.eq_ignore_ascii_case(target))
}

/// Pushes a span for every whole-identifier occurrence of `from` inside a
/// type-expression interior (ignoring ASCII case). Identifier boundaries are
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
        _ if is_plain_quoted(node) => {
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

    // Exact edit sets keep the stored-span expansion bounded: only recognized
    // plain static names change, never surrounding trivia or unrelated text.
    macro_rules! member_rename_cases {
        ($($name:ident: $src:expr, $owner:expr, $from:expr,
            [$(($start:expr, $end:expr)),*], $expected:expr;)*) => {
            $(
                #[test]
                fn $name() {
                    let src = $src;
                    let out = parse(src);
                    let expected_edits = vec![$(
                        TextEdit::replace(Span::new($start, $end), "Renamed")
                    ),*];

                    let edits = rename_member(&out.script, src, $owner, $from, "Renamed");

                    assert_eq!(edits.len(), expected_edits.len(), "{src:?}");
                    assert_eq!(edits, expected_edits, "{src:?}");
                    assert_eq!(apply_edits(src, &edits), Ok($expected.into()), "{src:?}");
                }
            )*
        };
    }

    member_rename_cases! {
        rename_member_ascii_access_and_call_without_declaration_use_exact_syntax_edits:
            "[T]::M\n[t]::m()\n$obj.M\n[Other]::M()\n# M\n'M'\n", "T", "M",
            [(5, 6), (12, 13)],
            "[T]::Renamed\n[t]::Renamed()\n$obj.M\n[Other]::M()\n# M\n'M'\n";
        rename_member_csharp_declaration_and_static_sites_use_exact_edits:
            "Add-Type -TypeDefinition 'class T { public static void M() {} }'\n[T]::M\n[T]::M()\n",
            "T", "M", [(55, 56), (70, 71), (77, 78)],
            "Add-Type -TypeDefinition 'class T { public static void Renamed() {} }'\n[T]::Renamed\n[T]::Renamed()\n";
        rename_member_keyword_retains_current_edit:
            "[T]::return", "T", "return", [(5, 11)], "[T]::Renamed";
        rename_member_digit_initial_generic_retains_current_edit:
            "[T]::9Lives", "T", "9Lives", [(5, 11)], "[T]::Renamed";
        rename_member_ascii_alnum_underscore_retains_current_edit:
            "[T]::_M2", "T", "_M2", [(5, 8)], "[T]::Renamed";
        rename_member_whitespace_retains_current_edit:
            "[T] \t:: \tM", "T", "M", [(9, 10)], "[T] \t:: \tRenamed";
        rename_member_unicode_initial_uses_stored_span:
            "[T]::猫", "T", "猫", [(5, 8)], "[T]::Renamed";
        rename_member_mixed_unicode_alnum_underscore_uses_stored_span:
            "[T]::M猫_2", "T", "M猫_2", [(5, 11)], "[T]::Renamed";
        rename_member_digit_initial_unicode_generic_uses_stored_span:
            "[T]::9猫_2", "T", "9猫_2", [(5, 11)], "[T]::Renamed";
        rename_member_accented_name_uses_stored_span:
            "[T]::Méthod", "T", "Méthod", [(5, 12)], "[T]::Renamed";
        rename_member_unicode_call_uses_stored_span:
            "[T]::猫()", "T", "猫", [(5, 8)], "[T]::Renamed()";
        rename_member_comment_before_separator_uses_stored_span:
            "[T]<# gap #>::M", "T", "M", [(14, 15)], "[T]<# gap #>::Renamed";
        rename_member_comment_after_separator_uses_stored_span:
            "[T]::<# gap #>M", "T", "M", [(14, 15)], "[T]::<# gap #>Renamed";
        rename_member_continuation_before_separator_uses_stored_span:
            "[T]`\n::M", "T", "M", [(7, 8)], "[T]`\n::Renamed";
        rename_member_continuation_after_separator_uses_stored_span:
            "[T]::`\nM", "T", "M", [(7, 8)], "[T]::`\nRenamed";
        rename_member_line_comment_after_separator_uses_stored_span:
            "[T]::# gap\nM", "T", "M", [(11, 12)], "[T]::# gap\nRenamed";
        rename_member_instance_access_is_excluded:
            "$obj.M", "T", "M", [], "$obj.M";
        rename_member_wrong_static_receiver_is_excluded:
            "[Other]::M", "T", "M", [], "[Other]::M";
        rename_member_dynamic_variable_text_is_not_an_editable_name:
            "[T]::$name", "T", "$name", [], "[T]::$name";
        rename_member_quoted_name_keeps_current_recovery_noop:
            "[T]::'M'", "T", "M", [], "[T]::'M'";
        rename_member_missing_name_keeps_current_noop:
            "[T]::", "T", "", [], "[T]::";
        rename_member_number_token_keeps_current_noop:
            "[T]::123", "T", "123", [], "[T]::123";
        rename_member_punctuation_generic_keeps_current_noop:
            "[T]::Foo-Bar", "T", "Foo-Bar", [], "[T]::Foo-Bar";
        rename_member_combining_mark_generic_keeps_current_noop:
            "[T]::e\u{301}", "T", "e\u{301}", [], "[T]::e\u{301}";
        rename_member_unclosed_call_retains_current_recovered_edit:
            "[T]::M(", "T", "M", [(5, 6)], "[T]::Renamed(";
    }

    member_rename_cases! {
        rename_member_comment_before_separator_call_uses_stored_span:
            "[T]<# M #>::M()", "T", "M", [(12, 13)], "[T]<# M #>::Renamed()";
        rename_member_comment_after_separator_call_uses_stored_span:
            "[T]::<# M #>M()", "T", "M", [(12, 13)], "[T]::<# M #>Renamed()";
        rename_member_continuation_before_separator_call_uses_stored_span:
            "[T]`\r\n::M()", "T", "M", [(8, 9)], "[T]`\r\n::Renamed()";
        rename_member_continuation_after_separator_call_uses_stored_span:
            "[T]::`\r\n猫()", "T", "猫", [(8, 11)], "[T]::`\r\nRenamed()";
        rename_member_line_comment_after_separator_call_uses_stored_span:
            "[T]::# M\r\nM()", "T", "M", [(10, 11)], "[T]::# M\r\nRenamed()";
        rename_member_bom_unicode_prefix_access_and_call_preserve_byte_offsets:
            "\u{feff}'猫';\r\n[T]::Méthod\n[T]::Méthod()", "T", "Méthod",
            [(16, 23), (29, 36)],
            "\u{feff}'猫';\r\n[T]::Renamed\n[T]::Renamed()";
        rename_member_nested_double_interpolation_preserves_absolute_unicode_span:
            "\u{feff}'猫';\r\n\"$(\"$([T]::猫())\")\"", "T", "猫", [(22, 25)],
            "\u{feff}'猫';\r\n\"$(\"$([T]::Renamed())\")\"";
        rename_member_nested_here_in_double_preserves_trivia_and_unicode_span:
            "\"$(@\"\r\n$([T]::<# 猫 #>猫)\r\n\"@)\"", "T", "猫", [(23, 26)],
            "\"$(@\"\r\n$([T]::<# 猫 #>Renamed)\r\n\"@)\"";
        rename_member_nested_double_in_here_preserves_unicode_call_span:
            "@\"\n$(\"$([T]::猫())\")\n\"@", "T", "猫", [(13, 16)],
            "@\"\n$(\"$([T]::Renamed())\")\n\"@";
        rename_member_repeated_unicode_sites_use_distinct_spans:
            "[T]::猫;\"$([T]::猫) $([T]::猫())\"", "T", "猫",
            [(5, 8), (17, 20), (29, 32)],
            "[T]::Renamed;\"$([T]::Renamed) $([T]::Renamed())\"";
        rename_member_unicode_name_still_ignores_only_ascii_case:
            "[T]::MéTHOD\n[t]::méthod()", "T", "Méthod", [(5, 12), (18, 25)],
            "[T]::Renamed\n[t]::Renamed()";
        rename_member_unicode_alphanumeric_characters_are_plain_names:
            "[T]::猫²_1()", "T", "猫²_1", [(5, 12)], "[T]::Renamed()";
        rename_member_keyword_call_retains_current_edit:
            "[T]::return()", "T", "return", [(5, 11)], "[T]::Renamed()";
        rename_member_digit_initial_generic_call_retains_current_edit:
            "[T]::9Lives()", "T", "9Lives", [(5, 11)], "[T]::Renamed()";
        rename_member_ascii_underscore_call_retains_current_edit:
            "[T]::_M2()", "T", "_M2", [(5, 8)], "[T]::Renamed()";
        rename_member_unicode_wrong_type_and_instance_sites_are_excluded:
            "[Other]::<# 猫 #>猫()\n$obj.猫\n$obj.猫()", "T", "猫", [],
            "[Other]::<# 猫 #>猫()\n$obj.猫\n$obj.猫()";
        rename_member_non_type_static_receiver_is_excluded:
            "$type::猫()", "T", "猫", [], "$type::猫()";
        rename_member_unicode_similar_names_are_excluded:
            "[T]::猫Helper\n[T]::猫_2()", "T", "猫", [],
            "[T]::猫Helper\n[T]::猫_2()";
        rename_member_non_ascii_case_differences_are_excluded:
            "[T]::MÉthod\n[T]::MÉTHOD()", "T", "Méthod", [],
            "[T]::MÉthod\n[T]::MÉTHOD()";
        rename_member_non_ascii_type_case_differences_are_excluded:
            "[É]::猫\n[É]::猫()", "é", "猫", [], "[É]::猫\n[É]::猫()";
        rename_member_normalization_differences_are_excluded:
            "[T]::e\u{301}\n[T]::e\u{301}()", "T", "é", [],
            "[T]::e\u{301}\n[T]::e\u{301}()";
        rename_member_dynamic_call_text_is_excluded:
            "[T]::$name()", "T", "$name", [], "[T]::$name()";
        rename_member_missing_call_name_is_excluded:
            "[T]::()", "T", "", [], "[T]::()";
        rename_member_quoted_and_computed_calls_are_excluded:
            "[T]::\"猫\"()\n[T]::('猫')", "T", "猫", [],
            "[T]::\"猫\"()\n[T]::('猫')";
        rename_member_number_call_token_is_excluded:
            "[T]::123()", "T", "123", [], "[T]::123()";
        rename_member_punctuation_call_generic_is_excluded:
            "[T]::Foo-Bar()", "T", "Foo-Bar", [], "[T]::Foo-Bar()";
        rename_member_escaped_access_and_call_names_are_excluded:
            "[T]::M`x\n[T]::M`x()", "T", "M`x", [], "[T]::M`x\n[T]::M`x()";
        rename_member_escaped_name_is_not_decoded:
            "[T]::M`x\n[T]::M`x()", "T", "Mx", [], "[T]::M`x\n[T]::M`x()";
        rename_member_combining_mark_call_is_excluded:
            "[T]::e\u{301}()", "T", "e\u{301}", [], "[T]::e\u{301}()";
        rename_member_non_alphanumeric_unicode_name_is_excluded:
            "[T]::猫🦀()", "T", "猫🦀", [], "[T]::猫🦀()";
        rename_member_mixed_csharp_and_powershell_preserves_comments_and_strings:
            "Add-Type -TypeDefinition @'\nclass T { public static int 猫; // 猫\n/* 猫 */ string Text = \"猫\"; }\n'@\n# 猫\n'猫'\n[T]::<# 猫 #>猫\n[T]::`\n猫()\n",
            "T", "猫", [(56, 59), (130, 133), (141, 144)],
            "Add-Type -TypeDefinition @'\nclass T { public static int Renamed; // 猫\n/* 猫 */ string Text = \"猫\"; }\n'@\n# 猫\n'猫'\n[T]::<# 猫 #>Renamed\n[T]::`\nRenamed()\n";
    }

    // Publicly constructed nodes may deliberately have no editable literal.
    // Source text that looks like a plain name must not become a fallback.
    fn member_without_span(src: &str) -> Node {
        let out = parse(src);
        let mut member_node = None;
        out.script.walk(&mut |node| {
            if matches!(
                node.kind,
                NodeKind::MemberAccess { .. } | NodeKind::InvokeMember { .. }
            ) {
                member_node = Some(node.clone());
            }
        });
        let mut node = member_node.expect("fixture contains a member node");
        match &mut node.kind {
            NodeKind::MemberAccess { member_span, .. }
            | NodeKind::InvokeMember { member_span, .. } => *member_span = None,
            _ => unreachable!("selected a member node above"),
        }
        node
    }

    #[test]
    fn rename_member_access_without_stored_span_does_not_rescan_source() {
        let src = "[T]::M";
        let node = member_without_span(src);

        let edits = rename_member(&node, src, "T", "M", "Renamed");

        assert_eq!(edits.len(), 0);
        assert_eq!(edits, vec![]);
        assert_eq!(apply_edits(src, &edits), Ok("[T]::M".into()));
    }

    #[test]
    fn rename_member_call_without_stored_span_does_not_rescan_source() {
        let src = "[T]::M()";
        let node = member_without_span(src);

        let edits = rename_member(&node, src, "T", "M", "Renamed");

        assert_eq!(edits.len(), 0);
        assert_eq!(edits, vec![]);
        assert_eq!(apply_edits(src, &edits), Ok("[T]::M()".into()));
    }

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
    fn rename_type_updates_embedded_base_declaration_header_and_powershell_use() {
        let src = "Add-Type -TypeDefinition @'\nclass Base{}class Child:Base{}\n'@\n[Base]\n";

        let out = parse(src);
        let edits = rename_type(&out.script, src, "Base", "Root");
        let result = apply_edits(src, &edits).unwrap();

        assert_eq!(out.errors, vec![]);
        assert_eq!(
            edits,
            vec![
                TextEdit::replace(Span::new(34, 38), "Root"),
                TextEdit::replace(Span::new(52, 56), "Root"),
                TextEdit::replace(Span::new(63, 67), "Root"),
            ]
        );
        assert_eq!(
            result,
            "Add-Type -TypeDefinition @'\nclass Root{}class Child:Root{}\n'@\n[Root]\n"
        );
    }

    #[test]
    fn rename_type_with_commented_base_header_leaves_csharp_comments_and_strings_untouched() {
        let src = "Add-Type -TypeDefinition @'\nclass Base {}\n// Base\n/* Base */\nclass Child : /* Base */ // Base\nBase { string Text = \"Base\"; }\n'@\n[Base]\n";

        let out = parse(src);
        let edits = rename_type(&out.script, src, "Base", "Root");
        let result = apply_edits(src, &edits).unwrap();

        assert_eq!(out.errors, vec![]);
        assert_eq!(
            edits,
            vec![
                TextEdit::replace(Span::new(34, 38), "Root"),
                TextEdit::replace(Span::new(94, 98), "Root"),
                TextEdit::replace(Span::new(129, 133), "Root"),
            ]
        );
        assert_eq!(
            result,
            "Add-Type -TypeDefinition @'\nclass Root {}\n// Base\n/* Base */\nclass Child : /* Base */ // Base\nRoot { string Text = \"Base\"; }\n'@\n[Root]\n"
        );
    }

    #[test]
    fn rename_type_with_base_header_leaves_similar_and_differently_cased_csharp_names_untouched() {
        let src = "Add-Type -TypeDefinition @'\nclass Base{}class Child:Base{}class BaseHelper{}class BASE{}class Other:BASE{}\n'@\n[Base]\n[BaseHelper]\n";

        let out = parse(src);
        let edits = rename_type(&out.script, src, "Base", "Root");
        let result = apply_edits(src, &edits).unwrap();

        assert_eq!(out.errors, vec![]);
        assert_eq!(
            edits,
            vec![
                TextEdit::replace(Span::new(34, 38), "Root"),
                TextEdit::replace(Span::new(52, 56), "Root"),
                TextEdit::replace(Span::new(111, 115), "Root"),
            ]
        );
        assert_eq!(
            result,
            "Add-Type -TypeDefinition @'\nclass Root{}class Child:Root{}class BaseHelper{}class BASE{}class Other:BASE{}\n'@\n[Root]\n[BaseHelper]\n"
        );
    }

    #[test]
    fn rename_type_with_unicode_before_embedded_base_preserves_original_file_byte_offsets() {
        let src =
            "# café 🦀\nAdd-Type -TypeDefinition @'\nclass Base{}class Child:Base{}\n'@\n[Base]\n";

        let out = parse(src);
        let edits = rename_type(&out.script, src, "Base", "Root");
        let result = apply_edits(src, &edits).unwrap();

        assert_eq!(out.errors, vec![]);
        assert_eq!(
            edits,
            vec![
                TextEdit::replace(Span::new(47, 51), "Root"),
                TextEdit::replace(Span::new(65, 69), "Root"),
                TextEdit::replace(Span::new(76, 80), "Root"),
            ]
        );
        assert_eq!(
            result,
            "# café 🦀\nAdd-Type -TypeDefinition @'\nclass Root{}class Child:Root{}\n'@\n[Root]\n"
        );
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
    fn class_and_enum_name_span_is_populated() {
        // The durable X4 path renames from the stored name_span; confirm the
        // span actually points at the name text.
        let out = parse("class Logger { }\nenum Color { Red }\n");
        let mut found = Vec::new();
        out.script.walk(&mut |n| match &n.kind {
            NodeKind::ClassDefinition {
                name, name_span, ..
            }
            | NodeKind::EnumDefinition {
                name, name_span, ..
            } => {
                found.push((
                    name.clone(),
                    name_span
                        .slice("class Logger { }\nenum Color { Red }\n")
                        .to_string(),
                ));
            }
            _ => {}
        });
        assert_eq!(
            found,
            vec![
                ("Logger".into(), "Logger".into()),
                ("Color".into(), "Color".into())
            ]
        );
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
